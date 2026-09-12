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
from app.db import build_engine,get_session
from app.config import Settings
from app.migrate import migrate
from app.experiments import import_experiment,persist,recover_interrupted
from app.analysis import import_diagnostic,save_failure,validate_diagnostic
from recipetriage_ml.training.experiments import default_matrix,compare_matrix,summary
from recipetriage_ml.training.runs import compare


@pytest.fixture
def database():
    if os.getenv('RUN_DB_TESTS')!='1':pytest.skip('Requires PostgreSQL')
    engine=build_engine(Settings());migrate(engine);created={name:[] for name in ['experiment_registry','training_runs','failure_analyses','diagnostic_runs']}
    def dependency():
        with Session(engine) as session:yield session
    app.dependency_overrides[get_session]=dependency
    try:yield engine,created
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for table,ids in created.items():
                key='experiment_id' if table=='experiment_registry' else 'source_run_id' if table=='failure_analyses' else 'run_id'
                for identifier in ids:connection.execute(text(f'DELETE FROM {table} WHERE {key}=:id'),{'id':identifier})
        engine.dispose()


def isolated_bundle(tmp_path,created):
    root=Path(__file__).parents[2]/'ml/training/results/experiments-v1'
    payload=json.loads((root/'experiment.json').read_text());payload['experiment_id']=str(uuid4());created['experiment_registry'].append(payload['experiment_id'])
    baseline=payload['baseline'];baseline['run_id']=str(uuid4());created['failure_analyses'].append(baseline['run_id']);children=[]
    for row in payload['runs']:
        child=json.loads((root/'runs'/row['run_id']/'run.json').read_text());child['run_id']=str(uuid4());child['experiment_id']=payload['experiment_id'];child['baseline']=copy.deepcopy(baseline)
        child['benchmark']['run_id']=str(uuid4());child['benchmark']['training_run_id']=child['run_id'];child['comparison']=compare(child['baseline'],child['benchmark'])
        created['training_runs'].append(child['run_id']);created['failure_analyses'].append(child['benchmark']['run_id']);children.append(child)
        directory=tmp_path/'runs'/child['run_id'];directory.mkdir(parents=True);(directory/'run.json').write_text(json.dumps(child))
    payload['runs']=[summary(child) for child in children];payload['comparison']=compare_matrix(children)
    path=tmp_path/'experiment.json';path.write_text(json.dumps(payload));return path,payload


@pytest.mark.integration
def test_registry_import_failure_category_persistence_and_immutability(database,tmp_path):
    engine,created=database;path,payload=isolated_bundle(tmp_path,created)
    with Session(engine) as session:
        import_experiment(session,path);import_experiment(session,path)
        counts=session.execute(text('SELECT category,COUNT(*) FROM failure_categories WHERE source_run_id=:id GROUP BY category'),{'id':created['failure_analyses'][-1]}).all()
        assert counts
    client=TestClient(app)
    assert client.get('/api/v1/experiments/'+payload['experiment_id']+'/download').json()==payload
    assert any(row['source_run_id']==created['failure_analyses'][-1] for row in client.get('/api/v1/analysis/failures').json())
    with pytest.raises(RuntimeError,match='immutable'):persist(engine,payload)
    assert client.get('/api/v1/analysis/collection-priorities/'+created['failure_analyses'][-1]).status_code==200


@pytest.mark.integration
def test_matrix_queue_excludes_training_and_diagnostics_and_recovers(database,monkeypatch):
    import app.experiments as module
    engine,created=database;config=default_matrix()
    def fake(engine,config,snapshot,payload):
        created['experiment_registry'].append(payload['experiment_id']);payload.update(status='running');persist(engine,payload)
        client=TestClient(app)
        assert client.post('/api/v1/training/runs',json=config.runs[0].model_dump()).status_code==409
        assert client.post('/api/v1/analysis/diagnostics',json={}).status_code==409
        assert client.post('/api/v1/benchmarks/runs',json={'provider':'hf-base'}).status_code==409
    monkeypatch.setattr(module,'execute',fake)
    with TestClient(app) as client:
        result=client.post('/api/v1/experiments',json=config.model_dump());assert result.status_code==202
        recover_interrupted(engine)
        assert client.get('/api/v1/experiments/'+result.json()['experiment_id']).json()['status']=='interrupted'


@pytest.mark.integration
def test_diagnostic_import_and_counterfactual_tamper_rejected(database,tmp_path):
    engine,created=database;source=Path(__file__).parents[2]/'ml/evaluation/results/diagnostics-v1/base.json'
    payload=json.loads(source.read_text());payload['run_id']=str(uuid4());created['diagnostic_runs'].append(payload['run_id'])
    path=tmp_path/'diagnostic.json';path.write_text(json.dumps(payload))
    with Session(engine) as session:import_diagnostic(session,path);import_diagnostic(session,path)
    assert TestClient(app).get('/api/v1/analysis/diagnostics/'+payload['run_id']+'/download').json()==payload
    changed=copy.deepcopy(payload);changed['shortcut']['rows'][0]['recipe']['time_minutes']=2
    with pytest.raises(ValueError):validate_diagnostic(changed)
    changed=copy.deepcopy(payload);changed['red_team']['passed']=99
    with pytest.raises(ValueError):validate_diagnostic(changed)
