"""One recipe, explicit model choices, sequential generation and persistent comparison evidence."""
from contextlib import contextmanager
import gc
import json
import os
import time
from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from app import jobs
from recipetriage_ml.inference.contracts import Recipe, ProviderError
from recipetriage_ml.inference.service import triage
from recipetriage_ml.inference.fireworks_provider import FireworksProvider, DEFAULT_MODEL
from recipetriage_ml.alignment.preferences import reference_directory, ml_directory
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.training.runs import now

router = APIRouter(prefix='/api/v1/playground', tags=['Model comparison'])


def references():
    if os.getenv('TRAINING_RUNS_DIR') == '/training':
        return {'sft': Path('/training/references/sft-v1'), 'lora': Path('/training/references/lora-v1'), 'qlora': reference_directory()}
    root = ml_directory()
    return {'sft': root/'training/results/sft-v1/local',
            'lora': root/'training/results/experiments-v1/runs/771d20b1-77df-446b-b34d-b8259ddbc585',
            'qlora': reference_directory()}


def catalog(session):
    from recipetriage_ml.evaluation.base_provider import MODEL_ID, MODEL_REVISION
    from recipetriage_ml.inference.hf_provider import MODEL_ID as INSTRUCT_ID
    rows = [
        {'key': 'fireworks', 'title': 'Fireworks', 'model': os.getenv('FIREWORKS_RECIPE_MODEL') or DEFAULT_MODEL,
         'available': bool(os.getenv('FIREWORKS_API_KEY')), 'reason': 'Requires FIREWORKS_API_KEY', 'hosted': True},
        {'key': 'base', 'title': 'Local base', 'model': MODEL_ID, 'revision': MODEL_REVISION, 'available': True, 'hosted': False},
        {'key': 'instruct', 'title': 'Local instruct', 'model': INSTRUCT_ID, 'available': True, 'hosted': False},
    ]
    for key, directory in references().items():
        present = (directory/'adapter/adapter_model.safetensors').is_file() and (directory/'adapter-provenance.json').is_file()
        identity = json.loads((directory/'adapter-provenance.json').read_text()) if present else {}
        rows.append({'key': key, 'title': {'sft': 'SFT', 'lora': 'LoRA', 'qlora': 'QLoRA'}[key], 'model': identity.get('model'),
                     'revision': identity.get('revision'), 'available': present, 'hosted': False,
                     'reason': 'Install the delivered reference adapter; see USE_CASES.md'})
    registered = [row[0] for row in session.execute(text('SELECT payload FROM model_registry ORDER BY created_at DESC'))]
    dpo = [model for model in registered if model['identity'].get('training_stage') == 'dpo']
    if not dpo:
        rows.append({'key': 'dpo', 'title': 'DPO', 'available': False, 'hosted': False,
                     'reason': 'No completed DPO checkpoint; explicit human preferences are still required'})
    for model in dpo:
        rows.append({'key': 'registry:'+model['model_id'], 'title': 'DPO '+model['model_id'][:8],
                     'model': model['name'], 'revision': model['identity']['revision'],
                     'available': Path(model['artifact_directory']).is_dir(), 'hosted': False,
                     'reason': 'Registered artifact is unavailable'})
    production = session.execute(text("SELECT payload FROM model_registry WHERE stage='production'")).scalar_one_or_none()
    rows.append({'key': 'production', 'title': 'Registered production', 'available': production is not None,
                 'model': production['name'] if production else None, 'hosted': False, 'reason': 'No model has passed the production gate'})
    return rows


@router.get('/models')
def models(session: Session = Depends(get_session)):
    return {'models': catalog(session), 'default_models': ['fireworks', 'base'],
            'note': 'Local models run sequentially. Opening this page does not generate responses.'}


@contextmanager
def provider_for(key, session):
    available = {row['key']: row for row in catalog(session)}
    if key not in available: raise ValueError('Unknown model selection')
    if not available[key]['available']: raise ValueError(available[key].get('reason', 'Model unavailable'))
    provider = None
    try:
        if key == 'fireworks':
            provider = FireworksProvider(model=available[key]['model'], reasoning_effort='none')
        elif key == 'base':
            from recipetriage_ml.evaluation.base_provider import BaseProvider
            provider = BaseProvider()
        elif key == 'instruct':
            from recipetriage_ml.inference.hf_provider import HFProvider
            provider = HFProvider()
        elif key == 'production':
            from app.serving import provider as production_provider
            provider = production_provider(session)
        else:
            from recipetriage_ml.training.sft import load_adapter
            if key.startswith('registry:'):
                from app.registry import verify_artifacts
                record = session.execute(text('SELECT payload FROM model_registry WHERE model_id=:id'), {'id': key.split(':', 1)[1]}).scalar_one()
                verify_artifacts(record); directory = Path(record['artifact_directory'])
            else: directory = references()[key]
            provider = load_adapter(directory)
        yield provider
    finally:
        # Release comparison-only local weights between candidates on memory-limited machines.
        if key == 'base':
            from recipetriage_ml.evaluation.base_provider import resources
            resources.cache_clear()
        elif key == 'instruct':
            from recipetriage_ml.inference.hf_provider import clear_resources
            clear_resources()
        del provider
        gc.collect()


def generate_one(recipe, key, session, temperature=0, max_new_tokens=128):
    started = time.perf_counter()
    with provider_for(key, session) as selected:
        if callable(getattr(selected, 'prepare', None)): selected.prepare()
        load_ms = round((time.perf_counter() - started) * 1000)
        result = triage(recipe, key, temperature, max_new_tokens, provider_instance=selected)
        return {**result, 'model_load_ms': load_ms,
                'total_ms': round((time.perf_counter() - started) * 1000),
                'latency_scope': 'Generation only after model preparation; Fireworks includes the hosted API round trip'}


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    recipe: Recipe
    models: list[str] = Field(min_length=2, max_length=6)
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    max_new_tokens: int = Field(default=128, ge=1, le=256, strict=True)

    @model_validator(mode='after')
    def unique(self):
        if len(self.models) != len(set(self.models)): raise ValueError('Choose different models')
        return self


@router.post('/comparisons', status_code=202)
def compare(body: ComparisonRequest, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    known = {row['key']: row for row in catalog(session)}
    for key in body.models:
        if key not in known or not known[key]['available']: raise HTTPException(422, f'{key}: model unavailable; inspect the model catalog')
    job = jobs.create(session, 'playground-comparison', body.model_dump())
    engine = request.app.state.engine
    def operation():
        result = {'recipe': body.recipe.model_dump(), 'recipe_sha256': digest(body.recipe.model_dump()),
                  'temperature': body.temperature, 'max_new_tokens': body.max_new_tokens,
                  'rows': [], 'started_at': now(), 'status': 'running', 'training_exported': False}
        for key in body.models:
            row = {'key': key, 'title': known[key]['title'], 'status': 'running', 'result': None, 'error': None}
            try:
                with Session(engine) as db:
                    row['result'] = generate_one(body.recipe, key, db, body.temperature, body.max_new_tokens)
                row['status'] = 'completed'
            except (ValueError, ProviderError) as exc:
                row.update(status='failed', error=str(exc))
            except Exception:
                import logging; logging.getLogger(__name__).exception('Comparison model failed')
                row.update(status='failed', error='Model could not run; inspect backend logs')
            result['rows'].append(row)
            job['result'] = result; jobs.persist(engine, job)
        result['status'] = 'completed' if all(row['status'] == 'completed' for row in result['rows']) else 'partial'
        result['finished_at'] = now()
        return result
    background.add_task(jobs.execute, engine, job, operation)
    return job


@router.get('/comparisons')
def history(session: Session = Depends(get_session)):
    return [row[0] for row in session.execute(text("SELECT payload FROM learning_jobs WHERE kind='playground-comparison' ORDER BY created_at DESC LIMIT 20"))]


@router.get('/comparisons/{run_id}')
def comparison(run_id: UUID, session: Session = Depends(get_session)):
    payload = session.execute(text("SELECT payload FROM learning_jobs WHERE run_id=:id AND kind='playground-comparison'"), {'id': str(run_id)}).scalar_one_or_none()
    if payload is None: raise HTTPException(404, 'Comparison not found')
    return payload
