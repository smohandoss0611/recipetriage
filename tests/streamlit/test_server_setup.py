from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from infra.server.setup import create_configuration
from streamlit_ui.settings import configuration


def read_env(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines())


def test_setup_shares_only_required_service_secrets_without_a_workspace_password(tmp_path):
    destination = tmp_path/'private'
    create_configuration(destination, 'recipes.mydomain.org', 'test-hosted-key')
    backend, frontend, postgres = [read_env(destination/name) for name in ['backend.env', 'streamlit.env', 'postgres.env']]
    assert backend['API_AUTH_TOKEN'] == frontend['API_AUTH_TOKEN']
    assert backend['POSTGRES_PASSWORD'] == postgres['POSTGRES_PASSWORD']
    assert backend['API_AUTH_TOKEN'] != backend['POSTGRES_PASSWORD']
    assert 'POSTGRES_PASSWORD' not in frontend and 'FIREWORKS_API_KEY' not in frontend
    assert 'WORKSPACE_PASSWORD_HASH' not in frontend
    for path in destination.iterdir():
        assert path.stat().st_mode & 0o077 == 0
    original = (destination/'backend.env').read_bytes()
    with pytest.raises(ValueError, match='refusing to overwrite'):
        create_configuration(destination, 'recipes.mydomain.org')
    assert (destination/'backend.env').read_bytes() == original


@pytest.mark.parametrize('domain', ['https://recipes.mydomain.org', 'evil.example\n:80', 'recipes.mydomain.org/path', 'localhost', '127.0.0.1'])
def test_setup_rejects_urls_and_caddy_configuration_injection(tmp_path, domain):
    with pytest.raises(ValueError, match='hostname'):
        create_configuration(tmp_path/'config', domain)
    assert not (tmp_path/'config').exists()


def test_invalid_multiline_secret_does_not_leave_partial_credentials(tmp_path):
    with pytest.raises(ValueError, match='single line'):
        create_configuration(tmp_path/'config', 'recipes.mydomain.org', 'key\nINJECT=value')
    assert not (tmp_path/'config').exists()


def test_server_configuration_keeps_api_credentials_separate_from_local_secrets(monkeypatch):
    monkeypatch.setenv('RECIPETRIAGE_DEPLOYMENT', 'single-server')
    monkeypatch.delenv('API_AUTH_TOKEN', raising=False)
    monkeypatch.delenv('WORKSPACE_PASSWORD_HASH', raising=False)
    backend, _ = configuration()
    assert backend == {'url': 'http://backend:8000', 'token': '', 'transport': 'docker'}


def test_local_setup_binds_no_sign_in_app_only_to_loopback(tmp_path):
    create_configuration(tmp_path/'config', 'localhost', local_check=True)
    config = read_env(tmp_path/'config'/'compose.env')
    assert config['SERVER_BIND'] == '127.0.0.1'
    assert config['APP_DOMAIN'] == 'localhost'
    assert config['HTTPS_PORT'] == '8543'
