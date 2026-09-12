"""Single-worker training API with PostgreSQL journals and persistent adapter files."""
import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.data import seed_snapshot, verify_snapshot
from recipetriage_ml.training.runs import ACTIVE, new_training_run, now, run_root
from recipetriage_ml.training.fireworks_sft import FireworksClient, preflight, run_managed
from recipetriage_ml.inference.contracts import ProviderError
from app.db import get_session

router = APIRouter(prefix='/api/v1/training', tags=['Training Monitor'])
logger = logging.getLogger(__name__)


def snapshot_for(version, session):
    payload = session.execute(text('SELECT payload FROM dataset_versions WHERE version=:version'), {'version': version}).scalar_one_or_none()
    if payload is None:
        payload = seed_snapshot()
        if payload['metadata']['version'] != version:
            raise HTTPException(404, 'Save this dataset version in Data Lab before starting training')
    verify_snapshot(payload)
    return payload


def persist(engine, run):
    with engine.begin() as connection:
        result = connection.execute(text("""UPDATE training_runs SET payload=CAST(:payload AS jsonb),updated_at=now()
            WHERE run_id=:id AND payload->>'status' NOT IN ('completed','failed','interrupted','blocked','ready','waiting','evaluation-blocked')"""),
            {'id': run['run_id'], 'payload': json.dumps(run, allow_nan=False)})
        if result.rowcount != 1:
            raise RuntimeError('Training run is not active; finished evidence cannot be overwritten')


def execute(engine, config, snapshot, run):
    try:
        if config.provider == 'local':
            from recipetriage_ml.training.sft import run_local
            runner = run_local
        else:
            runner = run_managed
        runner(config, snapshot, run=run, on_update=lambda value: persist(engine, value))
    except Exception:
        logger.exception('Training worker failed for %s; full error retained in server logs', run['run_id'])
        # run_local already saves ordinary failures; also cover missing dependencies or unexpected managed errors.
        if run['status'] in ACTIVE:
            run.update(status='failed', error='Training worker failed; inspect backend logs', finished_at=now())
            persist(engine, run)


def lock_queue(session):
    if not session.execute(text('SELECT pg_try_advisory_xact_lock(74291002)')).scalar_one():
        raise HTTPException(409, 'Another job is being queued; retry shortly')
    if session.execute(text("SELECT run_id FROM learning_jobs WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first():
        raise HTTPException(409, 'A generation, alignment or deployment job is active')
    busy = session.execute(text("""SELECT run_id FROM training_runs WHERE payload->>'status' = ANY(:states)
        UNION ALL SELECT run_id FROM benchmark_runs WHERE payload->>'status' IN ('queued','running') LIMIT 1"""), {'states': sorted(ACTIVE)}).first()
    if busy:
        raise HTTPException(409, 'A training or benchmark worker is active; inspect its progress first')
    if session.execute(text("SELECT experiment_id FROM experiment_registry WHERE payload->>'status' IN ('queued','running') UNION ALL SELECT run_id FROM diagnostic_runs WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first():
        raise HTTPException(409, 'An experiment or diagnostic worker is active; inspect its progress first')
    if session.execute(text("SELECT experiment_id FROM lora_experiments WHERE payload->>'status' IN ('queued','running') LIMIT 1")).first():
        raise HTTPException(409, 'A LoRA rank experiment is active; inspect LoRA Training first')


@router.get('/options')
def options():
    seed = seed_snapshot()['metadata']
    config = SFTConfig(dataset_version=seed['version'])
    return {'defaults': config.model_dump(), 'parameters': config.resolved(), 'schema': SFTConfig.model_json_schema(),
            'seed_dataset': seed, 'fireworks_configured': bool(os.getenv('FIREWORKS_API_KEY')),
            'fireworks_model': os.getenv('FIREWORKS_SFT_MODEL') or os.getenv('FIREWORKS_MODEL'),
            'fireworks_deployment_shape': os.getenv('FIREWORKS_SFT_DEPLOYMENT_SHAPE') or None}


@router.post('/preflight')
def check(config: SFTConfig, session: Session = Depends(get_session)):
    try:
        snapshot = snapshot_for(config.dataset_version, session)
        if config.provider == 'local':
            return local_configuration_check(config, snapshot)
        return {**preflight(config, snapshot, FireworksClient()), 'provider': 'fireworks'}
    except (ValueError, ProviderError) as exc:
        raise HTTPException(422, str(exc)) from exc


def local_configuration_check(config, snapshot):
    """Inspect configuration without importing models, writing runs or making requests.

    snapshot_for has already checked immutable contents and benchmark overlap.
    Package presence is not a claim that imports, tokenization or training work.
    """
    packages = ['torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate']
    if config.method == 'qlora':
        packages.append('bitsandbytes')
    runtime, issues = {}, []
    for package in packages:
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            issues.append(f'Missing local training dependency: {package}. Install the project training dependencies on the backend.')
    return {'provider': 'local', 'supported': not issues, 'issues': issues,
            'scope': 'Configuration, dataset integrity, benchmark overlap and installed package metadata.',
            'dataset_version': config.dataset_version, 'counts': snapshot['metadata']['counts'],
            'dataset_status': snapshot['metadata']['status'], 'warnings': snapshot['metadata']['warnings'],
            'parameters': config.resolved(), 'installed_packages': runtime,
            'deferred_checks': ['Library imports and runtime compatibility', 'Tokenized sequence lengths and answer masking',
                                'Model loading, LoRA target modules and available memory']
                                + (['CPU NF4 forward/backward capability'] if config.method == 'qlora' else []),
            'training_started': False, 'upload_performed': False, 'checked_at': now()}


@router.post('/runs', status_code=202)
def start(config: SFTConfig, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    snapshot = snapshot_for(config.dataset_version, session)
    lock_queue(session)
    run = new_training_run(config)
    session.execute(text('INSERT INTO training_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),
                    {'id': run['run_id'], 'payload': json.dumps(run)})
    session.commit()
    background.add_task(execute, request.app.state.engine, config, snapshot, run)
    return {'run_id': run['run_id'], 'status': 'queued'}


@router.get('/runs')
def list_runs(limit: int = Query(50, ge=1, le=100), session: Session = Depends(get_session)):
    keys = ['run_id', 'status', 'phase', 'created_at', 'updated_at', 'finished_at', 'config', 'comparison', 'error']
    return [{key: row[0][key] for key in keys} for row in session.execute(text('SELECT payload FROM training_runs ORDER BY created_at DESC,run_id LIMIT :limit'), {'limit': limit})]


@router.get('/runs/{run_id}')
def get_run(run_id: UUID, session: Session = Depends(get_session)):
    run = session.execute(text('SELECT payload FROM training_runs WHERE run_id=:id'), {'id': str(run_id)}).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, 'Training run not found')
    return run


@router.post('/runs/{run_id}/resume', status_code=202)
def resume(run_id: UUID, request: Request, background: BackgroundTasks, session: Session = Depends(get_session)):
    lock_queue(session)
    run = get_run(run_id, session)
    if run['config']['provider'] != 'fireworks' or run['status'] not in {'waiting', 'evaluation-blocked'}:
        raise HTTPException(409, 'Only a waiting managed run can resume; completed runs are immutable')
    config = SFTConfig.model_validate(run['config'])
    snapshot = snapshot_for(config.dataset_version, session)
    run.update(status='queued', error=None, finished_at=None)
    session.execute(text('UPDATE training_runs SET payload=CAST(:payload AS jsonb),updated_at=now() WHERE run_id=:id'),
                    {'id': str(run_id), 'payload': json.dumps(run)})
    session.commit()
    background.add_task(execute, request.app.state.engine, config, snapshot, run)
    return {'run_id': str(run_id), 'status': 'queued'}


@router.get('/runs/{run_id}/download')
def download(run_id: UUID, session: Session = Depends(get_session)):
    run = get_run(run_id, session)
    return Response(json.dumps(run, indent=2, ensure_ascii=False), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename=training-{run_id}.json'})


def recover_interrupted(engine):
    with engine.begin() as connection:
        rows = connection.execute(text("SELECT run_id,payload FROM training_runs WHERE payload->>'status' = ANY(:states)"), {'states': sorted(ACTIVE)}).all()
        for run_id, run in rows:
            remote = run['config']['provider'] == 'fireworks'
            run.update(status='waiting' if remote else 'interrupted',
                       error='API restarted. Resume managed monitoring using the saved resource IDs.' if remote else 'API restarted during local work. Checkpoints retained; start a new run.',
                       updated_at=now())
            connection.execute(text('UPDATE training_runs SET payload=CAST(:payload AS jsonb),updated_at=now() WHERE run_id=:id'),
                               {'id': str(run_id), 'payload': json.dumps(run)})


def import_run(session, run):
    """Import locally generated evidence; do not accept uploads through the public API."""
    UUID(run['run_id'])
    SFTConfig.model_validate(run['config'])
    if run['status'] in ACTIVE:
        raise ValueError('Cannot import an active run')
    # Verify embedded benchmark metrics and input contract, including the checkpoint descriptor.
    from app.benchmarks import validate_result
    from recipetriage_ml.evaluation.benchmark import descriptor
    from recipetriage_ml.training.runs import compare
    import copy
    for key in ['baseline', 'benchmark']:
        if run.get(key):
            evidence = copy.deepcopy(run[key])
            if evidence['config']['provider'] == 'hf-base':
                evidence['model_config'] = descriptor('hf-base')
            validate_result(evidence)
    if run.get('comparison') and run['comparison'] != compare(run['baseline'], run['benchmark']):
        raise ValueError('Comparison does not match saved benchmark metrics')
    existing = session.execute(text('SELECT payload FROM training_runs WHERE run_id=:id'), {'id': run['run_id']}).scalar_one_or_none()
    if existing is not None:
        if existing != run:
            raise ValueError('Run ID already contains different evidence')
        return
    session.execute(text('INSERT INTO training_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),
                    {'id': run['run_id'], 'payload': json.dumps(run, allow_nan=False)})
    session.commit()


if __name__ == '__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser = argparse.ArgumentParser(description='Import completed CLI training journals into PostgreSQL')
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    engine = build_engine(Settings())
    try:
        with Session(engine) as session:
            for path in args.files:
                run = json.loads(path.read_text())
                import_run(session, run)
                print('Imported', run['run_id'], run['status'])
    finally:
        engine.dispose()
