"""Saved product recipes and explicit, revision-bound human correction records."""
import copy
import json
from typing import Literal
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.db import get_session
from app import jobs
from app.intake import fetch_public_page, from_page, screenshot_text
from app.playground import generate_one, catalog
from recipetriage_ml.inference.normalization import normalize
from recipetriage_ml.inference.contracts import Recipe, Label, ProviderError
from recipetriage_ml.data.schemas import Recipe as DataRecipe, TrainingExample, LABELS, POLICY
from recipetriage_ml.data.pipeline import digest, build_dataset
from recipetriage_ml.training.data import seed_snapshot, verify_snapshot
from recipetriage_ml.training.runs import now

router = APIRouter(prefix='/api/v1/recipes', tags=['Recipe library'])


class IntakeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['text', 'message', 'url', 'screenshot']
    content: str = Field(min_length=1, max_length=5_400_000)


@router.post('/intake', status_code=201)
def intake(body: IntakeRequest, session: Session = Depends(get_session)):
    source = {'kind': body.kind, 'input_sha256': digest(body.content)}
    try:
        if body.kind == 'url':
            if len(body.content) > 2000: raise ValueError('URL exceeds 2000 characters')
            html, final_url = fetch_public_page(body.content.strip())
            recipe, provenance, source_text = from_page(html)
            source.update(url=final_url, original_url=body.content.strip())
        else:
            if body.kind == 'screenshot':
                source_text, ocr = screenshot_text(body.content); source['ocr'] = ocr
            else: source_text = body.content
            recipe, provenance = normalize(source_text)
    except (ValueError, ProviderError) as exc:
        raise HTTPException(422 if isinstance(exc, ValueError) else 503, str(exc)) from exc
    draft = {'draft_id': str(uuid4()), 'created_at': now(), 'recipe': recipe.model_dump(),
             'source': source, 'source_text': source_text, 'normalization': provenance, 'reviewed': False}
    session.execute(text('INSERT INTO recipe_intakes(draft_id,payload) VALUES (:id,CAST(:p AS jsonb))'), {'id': draft['draft_id'], 'p': json.dumps(draft)})
    session.commit()
    return draft


def get_record(session, recipe_id, lock=False):
    query = 'SELECT payload FROM library_recipes WHERE recipe_id=:id' + (' FOR UPDATE' if lock else '')
    record = session.execute(text(query), {'id': str(recipe_id)}).scalar_one_or_none()
    if record is None: raise HTTPException(404, 'Recipe not found')
    return record


def example(record):
    review = record.get('review')
    prediction = (record.get('prediction') or {}).get('result') or {}
    annotation = review['example'] if review else prediction.get('prediction') if prediction.get('valid_json') else record.get('seed_annotation') or {}
    return {'recipe': record['recipe'], 'labels': annotation.get('labels', []),
            'rationale': annotation.get('rationale') or annotation.get('explanation') or 'Run triage to propose labels, then review them.',
            'reviewed': bool(review), 'reviewed_by': review['reviewer'] if review else None,
            'library_id': record['recipe_id'], 'revision': record['revision'],
            'prediction': record.get('prediction'), 'review': review, 'triage_job': record.get('triage_job'),
            'source': record['source'], 'created_at': record['created_at']}


def persist(session, record):
    session.execute(text('UPDATE library_recipes SET payload=CAST(:p AS jsonb),updated_at=now() WHERE recipe_id=:id'),
                    {'id': record['recipe_id'], 'p': json.dumps(record)})


@router.get('')
def list_recipes(session: Session = Depends(get_session)):
    rows = [row[0] for row in session.execute(text('SELECT payload FROM library_recipes ORDER BY created_at DESC,recipe_id'))]
    for row in rows:
        pending = row.get('triage_job')
        if pending and pending['status'] in {'queued', 'running'}:
            job = session.execute(text('SELECT payload FROM learning_jobs WHERE run_id=:id'), {'id': pending['run_id']}).scalar_one_or_none()
            if job:
                row['triage_job'] = {**pending, 'status': job['status'], 'error': job.get('error')}
    return {'examples': [example(row) for row in rows], 'labels': LABELS, 'policy': POLICY}


def seed_library(session):
    for seed in seed_snapshot()['examples']:
        rid = str(uuid5(NAMESPACE_URL, 'recipetriage-library:'+seed['recipe']['id']))
        recipe = {**seed['recipe'], 'id': 'recipe-'+rid}
        body = DataRecipe.model_validate(recipe).inference_recipe().model_dump()
        record = {'recipe_id': rid, 'revision': 1, 'recipe': recipe, 'created_at': now(), 'review': None,
                  'prediction': None, 'triage_job': None, 'source': {'kind': 'original-seed', 'seed_id': seed['recipe']['id']},
                  'seed_annotation': {'labels': seed['labels'], 'rationale': seed['rationale']}, 'edit_history': []}
        session.execute(text('INSERT INTO library_recipes(recipe_id,content_sha256,payload) VALUES (:id,:hash,CAST(:p AS jsonb)) ON CONFLICT DO NOTHING'),
                        {'id': rid, 'hash': digest(body), 'p': json.dumps(record)})
    session.commit()


@router.post('/seed')
def load_seed(session: Session = Depends(get_session)):
    seed_library(session); return list_recipes(session)


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    draft_id: UUID
    recipe: Recipe
    auto_triage: bool = True
    model_key: str = Field(default='fireworks', max_length=100)


def queue_triage(record, model_key, session, engine, background):
    available = {row['key']: row for row in catalog(session)}
    if model_key not in available or not available[model_key]['available']:
        raise HTTPException(422, 'Selected triage model is unavailable; choose an available model')
    job = jobs.create(session, 'recipe-triage', {'recipe_id': record['recipe_id'], 'revision': record['revision'], 'model_key': model_key})
    current = get_record(session, record['recipe_id'], lock=True)
    if current['revision'] != record['revision']:
        job.update(status='failed', error='Recipe changed before triage could start')
        session.execute(text('UPDATE learning_jobs SET payload=CAST(:p AS jsonb) WHERE run_id=:id'), {'id': job['run_id'], 'p': json.dumps(job)})
        session.commit()
        raise HTTPException(409, 'Recipe changed; reload before triage')
    record['triage_job'] = {'run_id': job['run_id'], 'status': 'queued', 'error': None}
    current['triage_job'] = record['triage_job']
    persist(session, current); session.commit()
    frozen = copy.deepcopy(current)
    def operation():
        try:
            with Session(engine) as db:
                result = generate_one(DataRecipe.model_validate(frozen['recipe']).inference_recipe(), model_key, db)
                evidence = {'prediction_id': str(uuid4()), 'recipe_id': frozen['recipe_id'], 'revision': frozen['revision'],
                            'model_key': model_key, 'recipe_sha256': digest(frozen['recipe']), 'result': result, 'created_at': now()}
                db.execute(text('INSERT INTO library_predictions(prediction_id,recipe_id,payload) VALUES (:id,:recipe,CAST(:p AS jsonb))'),
                           {'id': evidence['prediction_id'], 'recipe': frozen['recipe_id'], 'p': json.dumps(evidence)})
                current = get_record(db, frozen['recipe_id'], lock=True)
                if current['revision'] == frozen['revision']:
                    current['prediction'] = evidence
                    current['triage_job'] = {'run_id': job['run_id'], 'status': 'completed', 'error': None}
                    persist(db, current)
                db.commit()
            return evidence
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ValueError, ProviderError)) else 'Triage failed; inspect backend logs'
            with Session(engine) as db:
                current = get_record(db, frozen['recipe_id'], lock=True)
                if current['revision'] == frozen['revision']:
                    current['triage_job'] = {'run_id': job['run_id'], 'status': 'failed', 'error': message}
                    persist(db, current); db.commit()
            raise
    background.add_task(jobs.execute, engine, job, operation)
    return job


@router.post('', status_code=201)
def save(body: SaveRequest, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    draft = session.execute(text('SELECT payload FROM recipe_intakes WHERE draft_id=:id'), {'id': str(body.draft_id)}).scalar_one_or_none()
    if draft is None: raise HTTPException(404, 'Intake draft not found; normalize the source first')
    content_hash = digest(body.recipe.model_dump())
    existing = session.execute(text('SELECT payload FROM library_recipes WHERE content_sha256=:hash'), {'hash': content_hash}).scalar_one_or_none()
    if existing: return {**example(existing), 'already_saved': True}
    rid = str(uuid4()); source = draft['source']
    recipe = DataRecipe(**body.recipe.model_dump(), id='recipe-'+rid,
        source_type='published' if source['kind']=='url' else 'personal',
        source_uri=source['url'] if source['kind']=='url' else 'personal:recipe-intake/'+draft['draft_id'],
        source_notes='Imported from '+source['kind']+'; normalized content was displayed before saving. Intake '+draft['draft_id'])
    record = {'recipe_id': rid, 'revision': 1, 'recipe': recipe.model_dump(), 'created_at': now(), 'review': None,
              'prediction': None, 'triage_job': None, 'source': {**source, 'draft_id': draft['draft_id'], 'normalization': draft['normalization']},
              'seed_annotation': None, 'edit_history': []}
    try:
        session.execute(text('INSERT INTO library_recipes(recipe_id,content_sha256,payload) VALUES (:id,:hash,CAST(:p AS jsonb))'),
                        {'id': rid, 'hash': content_hash, 'p': json.dumps(record)})
        session.commit()
    except IntegrityError:
        session.rollback(); raise HTTPException(409, 'This recipe was saved concurrently; refresh the library')
    warning = None
    if body.auto_triage:
        try: queue_triage(record, body.model_key, session, request.app.state.engine, background)
        except HTTPException as exc:
            session.rollback(); warning = 'Recipe saved. Triage was not started: '+str(exc.detail)
    return {**example(record), 'warning': warning, 'already_saved': False}


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=1, strict=True)


class TriageRequest(RevisionRequest):
    model_key: str = Field(default='fireworks', max_length=100)


@router.post('/{recipe_id}/triage', status_code=202)
def triage_saved(recipe_id: UUID, body: TriageRequest, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    record = get_record(session, recipe_id)
    if record['revision'] != body.revision: raise HTTPException(409, 'Recipe changed; reload before triage')
    return queue_triage(record, body.model_key, session, request.app.state.engine, background)


class EditRequest(RevisionRequest):
    recipe: Recipe
    actor: str = Field(min_length=1, max_length=100, pattern=r'.*\S.*')


@router.put('/{recipe_id}')
def edit(recipe_id: UUID, body: EditRequest, session: Session = Depends(get_session)):
    record = get_record(session, recipe_id, lock=True)
    if record['revision'] != body.revision: raise HTTPException(409, 'Recipe changed; reload before editing')
    previous = copy.deepcopy(record['recipe'])
    record['recipe'] = DataRecipe.model_validate({**previous, **body.recipe.model_dump()}).model_dump()
    record['edit_history'].append({'revision': record['revision'], 'recipe': previous, 'actor': body.actor, 'at': now()})
    record.update(revision=record['revision']+1, review=None, prediction=None, seed_annotation=None, triage_job=None)
    try:
        session.execute(text('UPDATE library_recipes SET content_sha256=:hash WHERE recipe_id=:id'),
                        {'id': str(recipe_id), 'hash': digest(body.recipe.model_dump())})
        persist(session, record); session.commit()
    except IntegrityError:
        session.rollback(); raise HTTPException(409, 'An identical recipe is already saved')
    return example(record)


class ReviewRequest(RevisionRequest):
    labels: list[Label] = Field(min_length=1, max_length=7)
    reviewer: str = Field(min_length=1, max_length=100, pattern=r'.*\S.*')
    rationale: str = Field(min_length=1, max_length=2000, pattern=r'.*\S.*')


@router.post('/{recipe_id}/review')
def review(recipe_id: UUID, body: ReviewRequest, session: Session = Depends(get_session)):
    record = get_record(session, recipe_id, lock=True)
    if record['revision'] != body.revision: raise HTTPException(409, 'Recipe changed; reload before reviewing')
    review_id = str(uuid4())
    try:
        verified = TrainingExample(recipe=DataRecipe.model_validate(record['recipe']), labels=body.labels, rationale=body.rationale,
            reviewed=True, reviewed_by=body.reviewer.strip(),
            annotation_notes=f'Explicit library review {review_id}; recipe revision {record["revision"]}; source hash {digest(record["source"])}')
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    evidence = {'review_id': review_id, 'recipe_id': str(recipe_id), 'revision': record['revision'], 'reviewer': body.reviewer.strip(),
                'at': now(), 'recipe_sha256': digest(record['recipe']), 'example': verified.model_dump()}
    session.execute(text('INSERT INTO library_reviews(review_id,recipe_id,payload) VALUES (:id,:recipe,CAST(:p AS jsonb))'),
                    {'id': review_id, 'recipe': str(recipe_id), 'p': json.dumps(evidence)})
    record['review'] = evidence; persist(session, record); session.commit()
    return example(record)


@router.post('/dataset-version')
def dataset_version(session: Session = Depends(get_session)):
    records = [row[0] for row in session.execute(text('SELECT payload FROM library_recipes ORDER BY recipe_id'))]
    approved = [row['review']['example'] for row in records if row.get('review') and row['review']['revision']==row['revision']]
    if len(approved) < 3: raise HTTPException(422, 'Explicitly review at least three independent recipes before making train/validation/test splits')
    try:
        payload = build_dataset(approved); splits = verify_snapshot(payload)
        if any(not rows for rows in splits.values()): raise ValueError('Need enough independent recipe groups for every split')
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    session.execute(text('INSERT INTO dataset_versions(version,payload) VALUES (:v,CAST(:p AS jsonb)) ON CONFLICT DO NOTHING'),
                    {'v': payload['metadata']['version'], 'p': json.dumps(payload)})
    session.commit()
    return payload


@router.get('/{recipe_id}/history')
def history(recipe_id: UUID, session: Session = Depends(get_session)):
    record = get_record(session, recipe_id)
    return {'edits': record['edit_history'],
            'predictions': [row[0] for row in session.execute(text('SELECT payload FROM library_predictions WHERE recipe_id=:id ORDER BY created_at'), {'id': str(recipe_id)})],
            'reviews': [row[0] for row in session.execute(text('SELECT payload FROM library_reviews WHERE recipe_id=:id ORDER BY created_at'), {'id': str(recipe_id)})]}


if __name__ == '__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', action='store_true', required=True); parser.parse_args()
    engine = build_engine(Settings())
    with Session(engine) as db: seed_library(db)
    engine.dispose(); print('Starter recipes available; no reviews or predictions were created')
