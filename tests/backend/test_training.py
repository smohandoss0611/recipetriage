import copy
import os
import json
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.main import app
from app.db import get_session, build_engine
from app.config import Settings
from app.migrate import migrate
from app.training import import_run, persist, recover_interrupted
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.data import seed_snapshot
from recipetriage_ml.training.runs import new_training_run, now


def test_training_options_explain_parameters_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'never-return-this-secret')
    result = TestClient(app).get('/api/v1/training/options')
    assert result.status_code == 200
    assert result.json()['parameters']['effective_batch_size'] == 2
    assert result.json()['fireworks_configured']
    assert 'never-return-this-secret' not in result.text


@pytest.fixture
def database():
    if os.getenv('RUN_DB_TESTS') != '1': pytest.skip('Requires PostgreSQL')
    engine = build_engine(Settings()); migrate(engine)
    ids = []
    def dependency():
        with Session(engine) as session: yield session
    app.dependency_overrides[get_session] = dependency
    try:
        yield engine, ids
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for run_id in ids:
                connection.execute(text('DELETE FROM training_runs WHERE run_id=:id'), {'id': run_id})
        engine.dispose()


@pytest.mark.integration
def test_training_import_download_immutable_and_restart_recovery(database):
    engine, ids = database
    config = SFTConfig(dataset_version=seed_snapshot()['metadata']['version'])
    run = new_training_run(config); run.update(status='completed', phase='finished', finished_at=now())
    ids.append(run['run_id'])
    with Session(engine) as session:
        import_run(session, run); import_run(session, run)
        changed = copy.deepcopy(run); changed['error'] = 'changed'
        with pytest.raises(ValueError, match='different evidence'): import_run(session, changed)
    client = TestClient(app)
    assert client.get('/api/v1/training/runs/' + run['run_id'] + '/download').json() == run
    assert client.post('/api/v1/training/runs/' + run['run_id'] + '/resume').status_code == 409
    with pytest.raises(RuntimeError, match='not active'): persist(engine, run)
    waiting = new_training_run(config.model_copy(update={'provider': 'fireworks'})); ids.append(waiting['run_id'])
    waiting['managed']['job_name'] = 'accounts/example/supervisedFineTuningJobs/saved-id'
    with engine.begin() as connection:
        connection.execute(text('INSERT INTO training_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))'), {'id': waiting['run_id'], 'payload': json.dumps(waiting)})
    recover_interrupted(engine)
    saved = client.get('/api/v1/training/runs/' + waiting['run_id']).json()
    assert saved['status'] == 'waiting' and saved['managed']['job_name'].endswith('saved-id')


@pytest.mark.integration
def test_enqueue_saves_progress_and_excludes_other_benchmark_workers(database, monkeypatch):
    import app.training as api
    engine, ids = database
    snapshot = seed_snapshot()
    def fake_execute(engine, config, snapshot, run):
        ids.append(run['run_id'])
        run.update(status='training', phase='optimizer-updates'); persist(engine, run)
        # The same queue gate blocks a separate benchmark while this training worker is active.
        assert TestClient(app).post('/api/v1/benchmarks/runs', json={'provider': 'hf-base'}).status_code == 409
        run.update(status='completed', phase='finished', finished_at=now()); persist(engine, run)
    monkeypatch.setattr(api, 'execute', fake_execute)
    with TestClient(app) as client:
        result = client.post('/api/v1/training/runs', json={'dataset_version': snapshot['metadata']['version']})
        assert result.status_code == 202 and result.json()['status'] == 'queued'
        saved = client.get('/api/v1/training/runs/' + result.json()['run_id']).json()
        assert saved['status'] == 'completed'
        assert client.post('/api/v1/training/runs', json={'dataset_version': 'v1-' + '0' * 64}).status_code == 404
        assert client.get('/api/v1/training/runs/' + str(uuid4())).status_code == 404
        assert client.get('/api/v1/training/runs/not-a-uuid').status_code == 422
