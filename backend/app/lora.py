"""LoRA architecture inspection and isolated rank experiments with PostgreSQL evidence."""
import json
import logging
from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from app.training import lock_queue, snapshot_for
from recipetriage_ml.training.lora import architecture_report, estimated_parameters, target_paths
from recipetriage_ml.training.lora_experiment import ExperimentConfig, new_experiment, run_experiment, compare_ranks
from recipetriage_ml.training.runs import ACTIVE, now

router = APIRouter(prefix='/api/v1/training/lora', tags=['LoRA Training'])
logger = logging.getLogger(__name__)


@router.get('/architecture')
def architecture(policy: str = 'query-value'):
    report = architecture_report()
    try:
        return {**report, 'selected_target_paths': target_paths(report, policy),
                'estimates': {str(rank): estimated_parameters(report, rank, policy) for rank in [4, 8, 16, 32]}}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def persist_experiment(engine, experiment):
    with engine.begin() as connection:
        result = connection.execute(text("""UPDATE lora_experiments SET payload=CAST(:payload AS jsonb),updated_at=now()
            WHERE experiment_id=:id AND payload->>'status' IN ('queued','running')"""),
            {'id': experiment['experiment_id'], 'payload': json.dumps(experiment, allow_nan=False)})
        if result.rowcount != 1:
            raise RuntimeError('Experiment is no longer active; completed evidence is immutable')


def persist_child(engine, run):
    with engine.begin() as connection:
        result = connection.execute(text("""INSERT INTO training_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))
            ON CONFLICT(run_id) DO UPDATE SET payload=EXCLUDED.payload,updated_at=now()
            WHERE training_runs.payload->>'status' = ANY(:active)"""),
            {'id': run['run_id'], 'payload': json.dumps(run, allow_nan=False), 'active': sorted(ACTIVE)})
        if result.rowcount != 1:
            existing = connection.execute(text('SELECT payload FROM training_runs WHERE run_id=:id'), {'id': run['run_id']}).scalar_one()
            if existing != run:
                raise RuntimeError('Cannot overwrite completed child run')


def execute(engine, config, snapshot, experiment):
    try:
        run_experiment(config, snapshot, experiment=experiment,
                       on_update=lambda value: persist_experiment(engine, value),
                       on_run_update=lambda value: persist_child(engine, value))
    except Exception:
        logger.exception('LoRA experiment failed: %s', experiment['experiment_id'])
        # Keep an interrupted child from blocking the single-worker queue forever.
        with engine.begin() as connection:
            connection.execute(text("""UPDATE lora_experiments SET payload=payload || CAST(:patch AS jsonb),updated_at=now()
                WHERE experiment_id=:id AND payload->>'status' IN ('queued','running')"""),
                {'id': experiment['experiment_id'], 'patch': json.dumps({'status': 'failed',
                 'error': 'Experiment failed before completion; inspect backend and worker logs.', 'finished_at': now()})})
            connection.execute(text("""UPDATE training_runs SET payload=payload || CAST(:patch AS jsonb),updated_at=now()
                WHERE payload->>'experiment_id'=:id AND payload->>'status' = ANY(:active)"""),
                {'id': experiment['experiment_id'], 'active': sorted(ACTIVE),
                 'patch': json.dumps({'status': 'interrupted', 'error': 'Experiment worker stopped; inspect its retained log.', 'finished_at': now()})})


@router.post('/experiments', status_code=202)
def start(config: ExperimentConfig, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    snapshot = snapshot_for(config.training.dataset_version, session)
    lock_queue(session)
    experiment = new_experiment(config)
    session.execute(text('INSERT INTO lora_experiments(experiment_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),
                    {'id': experiment['experiment_id'], 'payload': json.dumps(experiment)})
    session.commit()
    background.add_task(execute, request.app.state.engine, config, snapshot, experiment)
    return {'experiment_id': experiment['experiment_id'], 'status': 'queued'}


@router.get('/experiments')
def list_experiments(limit: int = Query(30, ge=1, le=100), session: Session = Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM lora_experiments ORDER BY created_at DESC,experiment_id LIMIT :limit'), {'limit': limit})]


@router.get('/experiments/{experiment_id}')
def get_experiment(experiment_id: UUID, session: Session = Depends(get_session)):
    value = session.execute(text('SELECT payload FROM lora_experiments WHERE experiment_id=:id'), {'id': str(experiment_id)}).scalar_one_or_none()
    if value is None:
        raise HTTPException(404, 'LoRA experiment not found')
    return value


@router.get('/experiments/{experiment_id}/download')
def download(experiment_id: UUID, session: Session = Depends(get_session)):
    return Response(json.dumps(get_experiment(experiment_id, session), indent=2), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename=lora-{experiment_id}.json'})


def recover_interrupted(engine):
    with engine.begin() as connection:
        connection.execute(text("""UPDATE lora_experiments SET payload=payload || CAST(:patch AS jsonb),updated_at=now()
            WHERE payload->>'status' IN ('queued','running')"""),
            {'patch': json.dumps({'status': 'interrupted', 'error': 'API process restarted; create a new experiment. Existing child evidence retained.', 'finished_at': now()})})


def import_experiment(session, path):
    from app.training import import_run
    path = Path(path)
    experiment = json.loads(path.read_text(encoding='utf-8'))
    UUID(experiment['experiment_id'])
    ExperimentConfig.model_validate(experiment['config'])
    if experiment['status'] != 'completed':
        raise ValueError('Only completed experiment bundles can be imported')
    runs = [json.loads((path.parent / 'runs' / row['run_id'] / 'run.json').read_text(encoding='utf-8')) for row in experiment['runs']]
    if compare_ranks(runs) != experiment['comparison']:
        raise ValueError('Experiment comparison differs from its child evidence')
    existing = session.execute(text('SELECT payload FROM lora_experiments WHERE experiment_id=:id'), {'id': experiment['experiment_id']}).scalar_one_or_none()
    if existing is not None and existing != experiment:
        raise ValueError('Experiment ID already contains different evidence')
    for run in runs:
        import_run(session, run)
    if existing is None:
        session.execute(text('INSERT INTO lora_experiments(experiment_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),
                        {'id': experiment['experiment_id'], 'payload': json.dumps(experiment, allow_nan=False)})
        session.commit()
    return experiment


if __name__ == '__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser = argparse.ArgumentParser(description='Import a completed local rank comparison and child run evidence')
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    engine = build_engine(Settings())
    try:
        with Session(engine) as session:
            result = import_experiment(session, args.experiment)
            print('Imported LoRA experiment', result['experiment_id'])
    finally:
        engine.dispose()
