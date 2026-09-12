"""Read-only provider dispatch and real dataset validation, without model downloads."""
import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.training as training
from app.db import get_session
from app.main import app
from recipetriage_ml.data.pipeline import build_dataset
from recipetriage_ml.training.data import seed_snapshot


@pytest.fixture
def preflight_api(monkeypatch):
    seed = json.loads(seed_snapshot()['files']['raw.json'])
    rows = [row for row in seed if row['recipe']['id'] in {'seed-001', 'seed-002', 'seed-007'}]
    snapshot = build_dataset(rows)
    queries = []
    class ReadOnlySession:
        def execute(self, statement, params):
            # A preflight must never create a run, upload, or change a dataset.
            assert str(statement).startswith('SELECT payload FROM dataset_versions')
            queries.append(params['version'])
            payload = snapshot if params['version'] == snapshot['metadata']['version'] else None
            return SimpleNamespace(scalar_one_or_none=lambda: payload)
    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_session] = lambda: ReadOnlySession()
    monkeypatch.setattr(training, 'version', lambda name: 'test-installed-version')
    def no_fireworks():
        pytest.fail('Local configuration checking must not instantiate a Fireworks client')
    monkeypatch.setattr(training, 'FireworksClient', no_fireworks)
    try:
        yield TestClient(app), snapshot, queries
    finally:
        app.dependency_overrides = previous


def test_local_configuration_check_accepts_small_teaching_splits_without_fireworks(preflight_api):
    client, snapshot, queries = preflight_api
    version = snapshot['metadata']['version']
    response = client.post('/api/v1/training/preflight', json={'provider': 'local', 'dataset_version': version})
    assert response.status_code == 200
    check = response.json()
    assert check['provider'] == 'local' and check['supported'] and not check['issues']
    assert check['counts'] == {'train': 1, 'validation': 1, 'test': 1}
    assert check['dataset_status'] == 'teaching-draft' and check['warnings']
    assert not check['training_started'] and not check['upload_performed']
    assert check['deferred_checks']
    assert check['parameters']['effective_batch_size'] == 2
    assert queries == [version]


def test_local_configuration_reports_missing_dependencies(preflight_api, monkeypatch):
    client, snapshot, _ = preflight_api
    def missing_trl(name):
        if name == 'trl':
            raise PackageNotFoundError(name)
        return 'test-installed-version'
    monkeypatch.setattr(training, 'version', missing_trl)
    response = client.post('/api/v1/training/preflight', json={'dataset_version': snapshot['metadata']['version']})
    assert response.status_code == 200
    assert not response.json()['supported']
    assert any('trl' in issue for issue in response.json()['issues'])


def test_local_configuration_rejects_tampering_unknown_versions_and_invalid_parameters(preflight_api):
    client, snapshot, queries = preflight_api
    config = {'dataset_version': snapshot['metadata']['version']}
    response = client.post('/api/v1/training/preflight', json={**config, 'learning_rate': 0})
    assert response.status_code == 422 and not queries
    response = client.post('/api/v1/training/preflight', json={'dataset_version': 'v1-' + '0' * 64})
    assert response.status_code == 404
    snapshot['files']['train.jsonl'] += '\n'
    response = client.post('/api/v1/training/preflight', json=config)
    assert response.status_code == 422
    assert 'immutable version' in response.json()['detail']


def test_fireworks_keeps_its_provider_specific_preflight(preflight_api, monkeypatch):
    client, snapshot, _ = preflight_api
    sentinel = object()
    seen = []
    monkeypatch.setattr(training, 'FireworksClient', lambda: sentinel)
    def managed(config, data, remote):
        seen.append((config.provider, data['metadata']['version'], remote))
        return {'supported': False, 'issues': ['Provider-specific dataset constraint'], 'upload_performed': False}
    monkeypatch.setattr(training, 'preflight', managed)
    response = client.post('/api/v1/training/preflight', json={'provider': 'fireworks', 'dataset_version': snapshot['metadata']['version']})
    assert response.status_code == 200
    assert response.json()['provider'] == 'fireworks'
    assert not response.json()['supported']
    assert seen == [('fireworks', snapshot['metadata']['version'], sentinel)]


def test_qlora_configuration_checks_bitsandbytes_package_without_running_a_probe(preflight_api, monkeypatch):
    client, snapshot, _ = preflight_api
    checked = []
    def packages(name):
        checked.append(name)
        return 'test-installed-version'
    monkeypatch.setattr(training, 'version', packages)
    response = client.post('/api/v1/training/preflight', json={'dataset_version': snapshot['metadata']['version'], 'method': 'qlora'})
    assert response.status_code == 200
    assert 'bitsandbytes' in checked
    assert 'CPU NF4 forward/backward capability' in response.json()['deferred_checks']
