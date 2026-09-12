import io
import json
import os
import subprocess
import sys
from uuid import uuid4
from zipfile import ZipFile

import pytest
pytest.importorskip('streamlit')
from sqlalchemy import text, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.embedded import create_app, build_runtime, database_settings
from app.db import get_session
from streamlit_ui.client import APIError
from streamlit_ui.embedded import EmbeddedAPI


@pytest.fixture
def api():
    application = create_app(None)
    def no_database():
        yield None
    application.dependency_overrides[get_session] = no_database
    return EmbeddedAPI(application)


def test_validation_preview_and_chat_keep_existing_contracts(api):
    seeds = api.get('/api/v1/datasets/seed')['examples']
    snapshot = api.post('/api/v1/datasets/preview', {'raw': json.dumps(seeds)})
    assert snapshot['metadata']['counts'] == {'train': 5, 'validation': 1, 'test': 1}
    repeated = api.post('/api/v1/datasets/preview', {'raw': json.dumps(seeds)})
    assert snapshot['files'] == repeated['files']
    assert snapshot['metadata']['version'] == repeated['metadata']['version']
    chat = api.post('/api/v1/datasets/chat-preview', {'example': seeds[0]})
    assert [m['role'] for m in chat['messages']] == ['system', 'user', 'assistant']
    seeds[0]['labels'] = ['invalid-label']
    with pytest.raises(APIError, match='HTTP 422'):
        api.post('/api/v1/datasets/preview', {'raw': json.dumps(seeds)})
    seeds[0]['provenance'] = {'source': 'synthetic'}
    with pytest.raises(APIError, match='human review queue'):
        api.post('/api/v1/datasets/preview', {'raw': json.dumps(seeds)})


@pytest.mark.parametrize('path', [
    '/api/v1/triage', '/api/v1/playground/comparisons', '/api/v1/curation/generate',
    '/api/v1/training/runs', '/api/v1/training/preflight', '/api/v1/training/runs/any/resume',
    '/api/v1/training/lora/experiments', '/api/v1/experiments', '/api/v1/benchmarks/runs',
    '/api/v1/alignment/pairs/generate', '/api/v1/alignment/train/dpo', '/api/v1/alignment/train/grpo',
    '/api/v1/alignment/retrain-approved', '/api/v1/analysis/diagnostics',
    '/api/v1/models/any/stage', '/api/v1/models/any/rollback', '/api/v1/recipes/any/triage',
])
def test_workers_and_provider_calls_are_absent_even_with_keys(api, monkeypatch, path):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'must-never-be-used')
    from recipetriage_ml.inference.fireworks_provider import FireworksProvider
    def forbidden(*args, **kwargs):
        pytest.fail('A paid provider must not be called')
    monkeypatch.setattr(FireworksProvider, 'generate', forbidden)
    with pytest.raises(APIError, match='does not run'):
        api.post(path, {})


@pytest.mark.parametrize('kind,content', [
    ('text', 'Make ravioli'), ('message', '{"title":"incomplete"}'),
    ('url', 'https://example.com'), ('screenshot', 'image-data'),
])
def test_intake_never_falls_back_to_paid_extraction(api, monkeypatch, kind, content):
    from recipetriage_ml.inference.fireworks_provider import FireworksProvider
    monkeypatch.setattr(FireworksProvider, 'generate', lambda *a, **k: pytest.fail('Paid extraction invoked'))
    with pytest.raises(APIError, match='HTTP 422'):
        api.post('/api/v1/recipes/intake', {'kind': kind, 'content': content})


def test_auto_triage_is_rejected_before_recipe_is_written(api):
    recipe = api.get('/api/v1/datasets/seed')['examples'][0]['recipe']
    recipe = {k: v for k, v in recipe.items() if k in {'title', 'ingredients', 'instructions', 'equipment', 'pantry_items', 'time_minutes'}}
    with pytest.raises(APIError, match='auto_triage=false'):
        api.post('/api/v1/recipes', {'draft_id': '00000000-0000-0000-0000-000000000001', 'recipe': recipe})


def test_runtime_and_options_do_not_import_training_frameworks():
    code = '''
import sys
from app.embedded import create_app
from streamlit_ui.embedded import EmbeddedAPI
api = EmbeddedAPI(create_app(None))
for path in ['/api/v1/experiments/options', '/api/v1/training/options', '/api/v1/training/lora/architecture', '/api/v1/alignment/options']:
    api.get(path)
assert not {'torch', 'trl', 'peft', 'bitsandbytes'}.intersection(sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_database_url_requires_postgresql_and_remote_ssl(monkeypatch):
    monkeypatch.setenv('POSTGRES_PASSWORD', 'not-a-default-for-the-cloud')
    for value in ['sqlite:///file.db', 'postgresql://u:p@db.example/app', 'postgresql://u:p@db.example/app?sslmode=disable']:
        with pytest.raises(ValueError):
            database_settings(value)
    settings = database_settings('postgresql://u:p@db.example/app?sslmode=require&channel_binding=require')
    assert settings.database_url.query['channel_binding'] == 'require'
    assert settings.database_url.drivername == 'postgresql+psycopg'


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv('CLOUD_TEST_DATABASE_URL'), reason='Requires a dedicated test PostgreSQL database')
def test_real_postgres_survives_runtime_restart_and_retains_review_gates(test_database_url):
    url = test_database_url
    # A dedicated disposable DB is required; never empty an existing workspace.
    first = build_runtime(url)
    api = EmbeddedAPI(first)
    assert api.get('/health')['database'] == 'ok'
    seeds = api.get('/api/v1/datasets/seed')['examples']
    snapshot = api.post('/api/v1/datasets/versions', {'raw': json.dumps(seeds)})
    version = snapshot['metadata']['version']
    rows = api.get('/api/v1/recipes')['examples']
    assert len(rows) == 7 and not any(row['reviewed'] for row in rows)
    first.state.engine.dispose()
    second = build_runtime(url)
    api = EmbeddedAPI(second)
    try:
        assert api.get('/api/v1/datasets/versions/' + version) == snapshot
        with ZipFile(io.BytesIO(api.request('GET', '/api/v1/datasets/versions/' + version + '/download', binary=True))) as archive:
            assert json.loads(archive.read('metadata.json'))['version'] == version
        assert len(api.get('/api/v1/recipes')['examples']) == 7
        row = rows[0]
        reviewed = api.post('/api/v1/recipes/' + row['library_id'] + '/review', {
            'revision': row['revision'], 'labels': row['labels'], 'rationale': row['rationale'], 'reviewer': 'Test Reviewer',
        })
        assert reviewed['reviewed'] and reviewed['reviewed_by'] == 'Test Reviewer'
        with pytest.raises(APIError, match='at least three'):
            api.post('/api/v1/recipes/dataset-version')
        invalid = {**row['recipe'], 'title': ''}
        content = {k: v for k, v in invalid.items() if k in {'title','ingredients','instructions','equipment','pantry_items','time_minutes'}}
        with pytest.raises(APIError, match='HTTP 422'):
            api.post('/api/v1/recipes/intake', {'kind': 'text', 'content': json.dumps(content)})
        content['title'] = 'Manually entered lentil recipe'
        draft = api.post('/api/v1/recipes/intake', {'kind': 'text', 'content': json.dumps(content)})
        saved = api.post('/api/v1/recipes', {'draft_id': draft['draft_id'], 'recipe': draft['recipe'], 'auto_triage': False})
        assert not saved['reviewed'] and saved['prediction'] is None
        with Session(second.state.engine) as session:
            for table in ['training_runs','learning_jobs','benchmark_runs']:
                assert session.execute(text('SELECT count(*) FROM ' + table)).scalar_one() == 0
    finally:
        second.state.engine.dispose()


@pytest.fixture
def test_database_url():
    configured = os.getenv('CLOUD_TEST_DATABASE_URL')
    if not configured:
        pytest.skip('Requires a dedicated test PostgreSQL database')
    url = make_url(configured).set(drivername='postgresql+psycopg')
    if not url.database.endswith('_test'):
        pytest.fail('Use a disposable database whose name ends in _test')
    admin = create_engine(url, isolation_level='AUTOCOMMIT')
    name = 'cloud_check_' + uuid4().hex
    with admin.connect() as connection:
        connection.execute(text('CREATE DATABASE ' + name))
    try:
        yield url.set(database=name).render_as_string(hide_password=False)
    finally:
        with admin.connect() as connection:
            connection.execute(text('DROP DATABASE ' + name))
        admin.dispose()
