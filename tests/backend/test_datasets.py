import io
import json
import os
from uuid import uuid4
from zipfile import ZipFile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.main import app
from app.db import get_session, build_engine
from app.config import Settings
from app.migrate import migrate

client=TestClient(app)


def seed(): return client.get('/api/v1/datasets/seed').json()['examples']


def test_seed_and_preview_api():
    response=client.post('/api/v1/datasets/preview',json={'raw':json.dumps(seed())})
    assert response.status_code==200,response.text
    assert response.json()['metadata']['counts']=={'train':5,'validation':1,'test':1}


def test_import_errors_are_reported_not_silently_dropped():
    rows=seed(); rows[0]['labels']=['invalid']
    response=client.post('/api/v1/datasets/validate',json={'raw':json.dumps(rows)})
    assert response.status_code==200
    assert response.json()['issues'][0]['row']==1
    assert len(response.json()['drafts'])==7
    assert client.post('/api/v1/datasets/preview',json={'raw':json.dumps(rows)}).status_code==422


def test_chat_preview_has_no_model_dependency():
    response=client.post('/api/v1/datasets/chat-preview',json={'example':seed()[0]})
    assert response.status_code==200
    assert response.json()['rendered_text'] is None
    assert response.json()['messages'][-1]['role']=='assistant'


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_DB_TESTS')!='1',reason='Requires PostgreSQL')
def test_snapshot_persistence_idempotence_and_zip():
    engine=build_engine(Settings()); migrate(engine); migrate(engine)
    rows=seed()
    suffix=uuid4().hex[:8]
    for row in rows: row['recipe']['id']+='-'+suffix
    version=None
    def session():
        with Session(engine) as db: yield db
    app.dependency_overrides[get_session]=session
    try:
        body={'raw':json.dumps(rows)}
        first=client.post('/api/v1/datasets/versions',json=body)
        assert first.status_code==200,first.text
        version=first.json()['metadata']['version']
        repeated=client.post('/api/v1/datasets/versions',json=body)
        assert repeated.json()==first.json()
        stored=client.get('/api/v1/datasets/versions/'+version)
        assert stored.json()==first.json()
        downloaded=client.get('/api/v1/datasets/versions/'+version+'/download')
        assert downloaded.status_code==200
        with ZipFile(io.BytesIO(downloaded.content)) as archive:
            assert json.loads(archive.read('metadata.json'))['version']==version
            assert len(json.loads(archive.read('examples.json')))==7
        assert client.get('/api/v1/datasets/versions/missing').status_code==404
    finally:
        app.dependency_overrides.clear()
        if version:
            with engine.begin() as connection:
                connection.execute(text('DELETE FROM dataset_versions WHERE version=:version'),{'version':version})
        engine.dispose()
