from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.auth import configured_token
from app.db import get_session
from app.main import app


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv('APP_ENV', 'cloud')
    monkeypatch.setenv('API_AUTH_TOKEN', 'test-secret-' + 'x' * 32)
    app.dependency_overrides[get_session] = lambda: Mock()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_cloud_api_and_documentation_require_token(http):
    for path in ['/api/v1/datasets/seed', '/docs', '/openapi.json']:
        assert http.get(path).status_code == 401
        assert http.get(path, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    response = http.get('/api/v1/datasets/seed', headers={'Authorization': 'Bearer test-secret-' + 'x' * 32})
    assert response.status_code == 200
    assert len(response.json()['examples']) == 7
    assert http.post('/api/v1/curation/generate', json={'count': 1}).status_code == 401


def test_health_remains_public_but_cloud_fails_closed(http, monkeypatch):
    assert http.get('/health').status_code == 200
    monkeypatch.delenv('API_AUTH_TOKEN')
    assert http.get('/api/v1/datasets/seed').status_code == 503
    with pytest.raises(RuntimeError, match='API_AUTH_TOKEN'):
        configured_token()


def test_local_mode_preserves_existing_compose(http, monkeypatch):
    monkeypatch.setenv('APP_ENV', 'local')
    monkeypatch.delenv('API_AUTH_TOKEN')
    assert http.get('/api/v1/datasets/seed').status_code == 200


def test_cloud_database_url_preserves_ssl_and_escapes(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv('DATABASE_URL', 'postgresql://recipe:p%40ss@db.example/recipe?sslmode=require')
    settings = Settings(_env_file=None)
    assert settings.database_url.drivername == 'postgresql+psycopg'
    assert settings.database_url.password == 'p@ss'
    assert settings.database_url.query['sslmode'] == 'require'
    assert 'p@ss' not in repr(settings)
