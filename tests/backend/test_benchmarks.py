import copy
import json
import os
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.main import app
from app.db import get_session, build_engine
from app.config import Settings
from app.migrate import migrate
from app.benchmarks import import_run, validate_result, persist_progress
from recipetriage_ml.evaluation.benchmark import RunConfig, new_run, run_benchmark
from recipetriage_ml.inference.contracts import Generation


class Dummy:
    def generate(self, *args):
        return Generation(raw_output='{"labels":["unclear"],"explanation":"Test fixture."}',model='fixture',finish_reason='stop')


def result():
    return run_benchmark(RunConfig(provider='hf-base'), provider_instance=Dummy())


def test_manifest_and_public_provider_configuration(monkeypatch):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'never-return-this-value')
    client = TestClient(app)
    response = client.get('/api/v1/benchmarks/manifest')
    assert response.status_code == 200 and len(response.json()['cases']) == 10
    response = client.get('/api/v1/benchmarks/providers')
    assert response.json()[1]['configured']
    assert 'never-return-this-value' not in response.text


def test_import_checks_scores_targets_and_raw_outputs():
    payload = result()
    assert validate_result(payload) == payload
    for change in ['summary', 'expected_labels', 'prediction', 'messages']:
        bad = copy.deepcopy(payload)
        if change == 'summary': bad['summary']['labels']['exact_match'] = 1
        elif change == 'expected_labels': bad['rows'][0]['expected_labels'] = ['unclear']
        elif change == 'prediction': bad['rows'][0]['prediction']['labels'] = ['dessert']
        else: bad['rows'][0]['messages'][1]['content'] = 'Leaked answer'
        with pytest.raises(ValueError): validate_result(bad)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_DB_TESTS') != '1', reason='Requires PostgreSQL')
def test_postgres_run_persistence_download_and_immutable_history():
    engine = build_engine(Settings()); migrate(engine)
    payload = result(); queued = new_run(RunConfig(provider='hf-base'))
    def session():
        with Session(engine) as db: yield db
    app.dependency_overrides[get_session] = session
    client = TestClient(app)
    try:
        with Session(engine) as db:
            import_run(db, payload)
            assert import_run(db, payload) == payload
            bad = copy.deepcopy(payload); bad['runtime']['python'] = 'changed'
            with pytest.raises(ValueError, match='different evidence'): import_run(db, bad)
        response = client.get('/api/v1/benchmarks/runs/' + payload['run_id'])
        assert response.status_code == 200 and response.json() == payload
        downloaded = client.get('/api/v1/benchmarks/runs/' + payload['run_id'] + '/download')
        assert downloaded.json() == payload
        assert any(row['run_id'] == payload['run_id'] for row in client.get('/api/v1/benchmarks/runs').json())
        assert client.get('/api/v1/benchmarks/runs/' + str(uuid4())).status_code == 404
        assert client.get('/api/v1/benchmarks/runs/not-a-uuid').status_code == 422
        with pytest.raises(RuntimeError, match='no longer active'): persist_progress(engine, payload)
        from app.benchmarks import insert_run
        with Session(engine) as db: insert_run(db, queued)
        assert client.post('/api/v1/benchmarks/runs', json={'provider':'hf-base'}).status_code == 409
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for run_id in [payload['run_id'], queued['run_id']]:
                connection.execute(text('DELETE FROM benchmark_runs WHERE run_id=:id'), {'id':run_id})
        engine.dispose()


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_DB_TESTS') != '1', reason='Requires PostgreSQL')
def test_enqueue_executes_background_job_and_persists_each_case(monkeypatch):
    import app.benchmarks as api
    engine = build_engine(Settings()); migrate(engine)
    saved_ids = []
    checkpoints = []
    original_persist = api.persist_progress

    def session():
        with Session(engine) as db: yield db

    def mocked_run(config, *, run, on_update):
        saved_ids.append(run['run_id'])
        return run_benchmark(config, run=run, on_update=on_update, provider_instance=Dummy())

    def record_progress(engine, payload):
        original_persist(engine, payload)
        with Session(engine) as db:
            stored = db.execute(text('SELECT payload FROM benchmark_runs WHERE run_id=:id'),
                                {'id': payload['run_id']}).scalar_one()
        checkpoints.append((stored['status'], len(stored['rows'])))

    monkeypatch.setattr(api, 'run_benchmark', mocked_run)
    monkeypatch.setattr(api, 'persist_progress', record_progress)
    app.dependency_overrides[get_session] = session
    try:
        with TestClient(app) as client:
            response = client.post('/api/v1/benchmarks/runs', json={'provider': 'hf-base', 'reasoning': 'disabled'})
            assert response.status_code == 202 and response.json()['status'] == 'queued'
            completed = client.get('/api/v1/benchmarks/runs/' + response.json()['run_id']).json()
            assert completed['status'] == 'completed'
            assert completed['summary']['attempted_cases'] == 10
            assert validate_result(completed) == completed
            assert ('running', 0) in checkpoints
            assert {size for _, size in checkpoints} >= set(range(1, 11))
            assert checkpoints[-1] == ('completed', 10)
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for run_id in saved_ids:
                connection.execute(text('DELETE FROM benchmark_runs WHERE run_id=:id'), {'id': run_id})
        engine.dispose()
