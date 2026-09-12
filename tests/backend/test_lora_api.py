import copy
import json
import os
from pathlib import Path
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.main import app
from app.db import build_engine, get_session
from app.config import Settings
from app.migrate import migrate
from app.lora import import_experiment, persist_experiment, persist_child, recover_interrupted
from recipetriage_ml.training.lora_experiment import ExperimentConfig, new_experiment
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.data import seed_snapshot
from recipetriage_ml.training.runs import now


def test_architecture_endpoint_is_lightweight_and_targets_are_real():
    client = TestClient(app)
    report = client.get('/api/v1/training/lora/architecture').json()
    assert report['estimates'] == {'4': 270336, '8': 540672, '16': 1081344, '32': 2162688}
    assert len(report['selected_target_paths']) == 48
    assert client.get('/api/v1/training/lora/architecture?policy=attention').json()['estimates']['4'] > 270336
    assert client.get('/api/v1/training/lora/architecture?policy=made-up').status_code == 422
    # Session construction is a dependency even when request validation fails.
    app.dependency_overrides[get_session] = lambda: None
    try:
        assert client.post('/api/v1/training/lora/experiments', json={'training': {'dataset_version': seed_snapshot()['metadata']['version']}, 'ranks': [4]}).status_code == 422
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def database():
    if os.getenv('RUN_DB_TESTS') != '1': pytest.skip('Requires PostgreSQL')
    engine = build_engine(Settings()); migrate(engine); ids = []; child_ids = []
    def dependency():
        with Session(engine) as session: yield session
    app.dependency_overrides[get_session] = dependency
    try:
        yield engine, ids, child_ids
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for name, values, key in [('lora_experiments', ids, 'experiment_id'), ('training_runs', child_ids, 'run_id')]:
                for identifier in values:
                    connection.execute(text(f'DELETE FROM {name} WHERE {key}=:id'), {'id': identifier})
        engine.dispose()


@pytest.mark.integration
def test_lora_import_immutable_download_and_child_runs(database, tmp_path):
    engine, ids, child_ids = database
    source = Path(__file__).parents[2]/'ml/training/results/lora-v1'
    experiment = json.loads((source/'experiment.json').read_text()); experiment['experiment_id'] = str(uuid4()); ids.append(experiment['experiment_id'])
    # Reassign IDs only to avoid touching the user's imported evidence in a shared test DB.
    for summary in experiment['runs']:
        child = json.loads((source/'runs'/summary['run_id']/'run.json').read_text())
        old = child['run_id']; child['run_id'] = str(uuid4()); child_ids.append(child['run_id']); child['experiment_id'] = experiment['experiment_id']
        summary['run_id'] = child['run_id']
        for row in experiment['comparison']['rows']:
            if row['run_id'] == old: row['run_id'] = child['run_id']
        target = tmp_path/'runs'/child['run_id']; target.mkdir(parents=True); (target/'run.json').write_text(json.dumps(child))
    path = tmp_path/'experiment.json'; path.write_text(json.dumps(experiment))
    with Session(engine) as session:
        import_experiment(session, path); import_experiment(session, path)
    client = TestClient(app)
    assert client.get(f"/api/v1/training/lora/experiments/{experiment['experiment_id']}/download").json() == experiment
    assert client.get(f'/api/v1/training/runs/{child_ids[0]}').json()['status'] == 'completed'
    with pytest.raises(RuntimeError, match='immutable'): persist_experiment(engine, experiment)
    changed = json.loads((tmp_path/'runs'/child_ids[0]/'run.json').read_text()); changed['error'] = 'overwrite'
    with pytest.raises(RuntimeError, match='overwrite'): persist_child(engine, changed)
    assert client.get(f'/api/v1/training/lora/experiments/{uuid4()}').status_code == 404


@pytest.mark.integration
def test_experiment_queue_blocks_sft_benchmark_and_recovers(database, monkeypatch):
    import app.lora as api
    engine, ids, _ = database
    config = ExperimentConfig(training=SFTConfig(dataset_version=seed_snapshot()['metadata']['version']))
    def fake_execute(engine, config, snapshot, experiment):
        ids.append(experiment['experiment_id'])
        experiment.update(status='running'); persist_experiment(engine, experiment)
        client = TestClient(app)
        assert client.post('/api/v1/training/runs', json=config.training.model_dump()).status_code == 409
        assert client.post('/api/v1/benchmarks/runs', json={'provider': 'hf-base'}).status_code == 409
        assert client.post('/api/v1/training/lora/experiments', json=config.model_dump()).status_code == 409
    monkeypatch.setattr(api, 'execute', fake_execute)
    with TestClient(app) as client:
        response = client.post('/api/v1/training/lora/experiments', json=config.model_dump())
        assert response.status_code == 202
        recover_interrupted(engine)
        saved = client.get(f"/api/v1/training/lora/experiments/{response.json()['experiment_id']}").json()
        assert saved['status'] == 'interrupted'


@pytest.mark.integration
def test_failure_before_worker_start_releases_queue(database, monkeypatch):
    import app.lora as api
    engine, ids, _ = database
    config = ExperimentConfig(training=SFTConfig(dataset_version=seed_snapshot()['metadata']['version']))
    experiment = new_experiment(config); ids.append(experiment['experiment_id'])
    with engine.begin() as connection:
        connection.execute(text('INSERT INTO lora_experiments(experiment_id,payload) VALUES (:id,CAST(:payload AS jsonb))'), {'id': experiment['experiment_id'], 'payload': json.dumps(experiment)})
    def fail(*args, **kwargs): raise ValueError('invalid snapshot')
    monkeypatch.setattr(api, 'run_experiment', fail)
    api.execute(engine, config, {}, experiment)
    assert TestClient(app).get(f"/api/v1/training/lora/experiments/{experiment['experiment_id']}").json()['status'] == 'failed'
