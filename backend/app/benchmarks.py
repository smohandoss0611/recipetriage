"""Baseline API and PostgreSQL persistence for the single API worker deployment."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from recipetriage_ml.evaluation.benchmark import RunConfig, descriptor, load_benchmark, manifest, new_run, run_benchmark
from recipetriage_ml.evaluation.metrics import inspect_response, summarize
from recipetriage_ml.data.pipeline import canonical, digest
from recipetriage_ml.inference.prompts import build_messages
from recipetriage_ml.evaluation.base_provider import render_prompt
from app.db import get_session

router = APIRouter(prefix='/api/v1/benchmarks', tags=['Baseline'])
logger = logging.getLogger(__name__)
TERMINAL = {'completed', 'blocked', 'failed', 'interrupted'}


def validate_result(payload):
    """Verify imported evidence against frozen inputs and recompute, never trust scores."""
    UUID(payload['run_id'])
    config = RunConfig.model_validate(payload['config'])
    reference = new_run(config)
    for key in ['benchmark', 'protocol', 'protocol_sha256', 'model_config']:
        # Hosted aliases may differ from this server's configured model; keep the recorded descriptor.
        if key == 'model_config' and config.provider == 'fireworks':
            continue
        if payload[key] != reference[key]:
            raise ValueError(f'Imported run {key} differs from the frozen contract')
    if payload['status'] not in TERMINAL:
        raise ValueError('Only finished or interrupted runs can be imported')
    cases = {case.recipe.id: case for case in load_benchmark().cases}
    for row in payload['rows']:
        case = cases[row['case_id']]
        messages = build_messages(case.recipe.inference_recipe())
        if row['messages'] != messages or row['messages_sha256'] != digest(messages) or row['expected_labels'] != case.labels or row['category'] != case.category:
            raise ValueError('Recorded input or answer key differs from benchmark')
        if config.provider == 'hf-base' and row['rendered_prompt'] != render_prompt(messages):
            raise ValueError('Recorded base prompt differs from fixed formatter')
        checked = inspect_response(row['raw_output'], row['finish_reason'])
        for key in ['prediction', 'json_valid', 'schema_valid', 'usable']:
            if row[key] != checked[key]:
                raise ValueError(f'Recorded {key} disagrees with raw output')
    calculated = summarize(load_benchmark().cases, payload['rows'])
    if calculated != payload['summary']:
        raise ValueError('Recorded metrics disagree with recomputed scores')
    return payload


def insert_run(session, payload):
    session.execute(text('INSERT INTO benchmark_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),
                    {'id': payload['run_id'], 'payload': json.dumps(payload, allow_nan=False)})
    session.commit()


def import_run(session, payload):
    validate_result(payload)
    existing = session.execute(text('SELECT payload FROM benchmark_runs WHERE run_id=:id'), {'id': payload['run_id']}).scalar_one_or_none()
    if existing is not None:
        if existing != payload:
            raise ValueError('Run ID already contains different evidence; never overwrite completed results')
        return existing
    insert_run(session, payload)
    return payload


def persist_progress(engine, payload):
    with Session(engine) as session:
        changed = session.execute(text("UPDATE benchmark_runs SET payload=CAST(:payload AS jsonb),updated_at=now() WHERE run_id=:id AND payload->>'status' IN ('queued','running')"),
                                  {'id': payload['run_id'], 'payload': json.dumps(payload, allow_nan=False)})
        if changed.rowcount != 1:
            raise RuntimeError('Run is no longer active; completed evidence cannot be overwritten')
        session.commit()


def recover_interrupted(engine):
    # This app is deployed with one Uvicorn worker. A process restart stops its jobs.
    with Session(engine) as session:
        session.execute(text("""UPDATE benchmark_runs SET payload=payload || CAST(:patch AS jsonb),updated_at=now()
            WHERE payload->>'status' IN ('queued','running')"""),
            {'patch': json.dumps({'status': 'interrupted', 'error': 'API process restarted; partial evidence retained. Start a new run.',
                                  'finished_at': datetime.now(timezone.utc).isoformat()})})
        session.commit()


def execute_run(engine, payload):
    try:
        run_benchmark(RunConfig.model_validate(payload['config']), run=payload,
                      on_update=lambda result: persist_progress(engine, result))
    except Exception as exc:
        logger.exception('Benchmark background job failed')
        payload.update(status='failed', error=f'{type(exc).__name__}: benchmark worker failed; inspect backend logs',
                       finished_at=datetime.now(timezone.utc).isoformat())
        try:
            persist_progress(engine, payload)
        except Exception:
            logger.exception('Could not persist terminal benchmark failure')


@router.get('/manifest')
def benchmark_manifest():
    return manifest()


@router.get('/providers')
def providers():
    import os
    return [{'id': name, **descriptor(name), 'configured': name == 'hf-base' or bool(os.getenv('FIREWORKS_API_KEY'))}
            for name in ['hf-base', 'fireworks']]


@router.post('/runs', status_code=202)
def start_run(config: RunConfig, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    # Serialize enqueue requests across threads; only one local benchmark job at a time.
    locked = session.execute(text('SELECT pg_try_advisory_xact_lock(74291002)')).scalar_one()
    if not locked:
        raise HTTPException(409, 'Another benchmark request is being started; retry shortly')
    if session.execute(text("SELECT run_id FROM learning_jobs WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first():
        raise HTTPException(409, 'A generation or alignment worker is active; inspect its progress first')
    active = session.execute(text("SELECT run_id FROM benchmark_runs WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first()
    if active:
        raise HTTPException(409, 'A benchmark is already running; inspect its progress before starting another')
    from recipetriage_ml.training.runs import ACTIVE
    if session.execute(text("SELECT experiment_id FROM experiment_registry WHERE payload->>'status' IN ('queued','running') UNION ALL SELECT run_id FROM diagnostic_runs WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first():
        raise HTTPException(409, 'An experiment or diagnostic worker is active; inspect its progress first')
    if session.execute(text("SELECT experiment_id FROM lora_experiments WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first():
        raise HTTPException(409, 'A LoRA rank experiment is active; inspect LoRA Training first')
    if session.execute(text("SELECT run_id FROM training_runs WHERE payload->>'status' = ANY(:states) LIMIT 1"), {'states': sorted(ACTIVE)}).first():
        raise HTTPException(409, 'A training worker is active; inspect Training Monitor before starting a benchmark')
    payload = new_run(config)
    insert_run(session, payload)
    background.add_task(execute_run, request.app.state.engine, payload)
    return {'run_id': payload['run_id'], 'status': payload['status']}


@router.get('/runs')
def list_runs(limit: int = Query(default=50, ge=1, le=100), session: Session = Depends(get_session)):
    keys = ['run_id', 'status', 'created_at', 'finished_at', 'config', 'model_config', 'protocol_sha256', 'setup_latency_ms', 'summary', 'error']
    return [{key: row[0][key] for key in keys} for row in session.execute(text('SELECT payload FROM benchmark_runs ORDER BY created_at DESC,run_id LIMIT :limit'), {'limit': limit})]


@router.get('/runs/{run_id}')
def get_run(run_id: UUID, session: Session = Depends(get_session)):
    payload = session.execute(text('SELECT payload FROM benchmark_runs WHERE run_id=:id'), {'id': str(run_id)}).scalar_one_or_none()
    if payload is None:
        raise HTTPException(404, 'Benchmark run not found')
    return payload


@router.get('/runs/{run_id}/download')
def download_run(run_id: UUID, session: Session = Depends(get_session)):
    payload = get_run(run_id, session)
    return Response(json.dumps(payload, indent=2, ensure_ascii=False), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename=benchmark-{run_id}.json'})


if __name__ == '__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser = argparse.ArgumentParser(description='Import local benchmark evidence into PostgreSQL')
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    engine = build_engine(Settings())
    try:
        with Session(engine) as session:
            for path in args.files:
                result = import_run(session, json.loads(path.read_text()))
                print(f"Imported {result['run_id']} ({result['config']['provider']}, {result['status']})")
    finally:
        engine.dispose()
