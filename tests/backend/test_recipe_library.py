"""Human choices below are isolated test fixtures, never real user approvals."""
import copy
import json
import os
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from app.main import app
from app.db import get_session, build_engine
from app.config import Settings
from app.migrate import migrate
from recipetriage_ml.training.data import seed_snapshot, verify_snapshot
from recipetriage_ml.inference.contracts import ProviderError

pytestmark=[pytest.mark.integration,pytest.mark.skipif(os.getenv('RUN_DB_TESTS')!='1',reason='Requires PostgreSQL')]


@pytest.fixture
def client():
    primary=build_engine(Settings());name='test_product_'+uuid4().hex
    with primary.begin() as db:db.execute(text('CREATE SCHEMA '+name))
    engine=create_engine(primary.url,hide_parameters=True,connect_args={'options':'-c search_path='+name});migrate(engine)
    previous=getattr(app.state,'engine',None);app.state.engine=engine
    def session():
        with Session(engine) as db:yield db
    app.dependency_overrides[get_session]=session
    try:yield TestClient(app)
    finally:
        app.dependency_overrides.clear();app.state.engine=previous;engine.dispose()
        with primary.begin() as db:db.execute(text('DROP SCHEMA '+name+' CASCADE'))
        primary.dispose()


def recipe(index=0):
    item=seed_snapshot()['examples'][index]['recipe']
    keys=['title','ingredients','instructions','equipment','time_minutes','pantry_items']
    return {key:item[key] for key in keys}


def save(client,index=0,auto=False):
    draft=client.post('/api/v1/recipes/intake',json={'kind':'message','content':json.dumps(recipe(index))})
    assert draft.status_code==201,draft.text
    response=client.post('/api/v1/recipes',json={'draft_id':draft.json()['draft_id'],'recipe':draft.json()['recipe'],'auto_triage':auto})
    assert response.status_code==201,response.text
    return response.json(),draft.json()


def test_intake_saved_library_duplicate_and_human_review_boundary(client):
    item,draft=save(client)
    assert not item['reviewed'] and item['labels']==[] and item['prediction'] is None
    assert client.post('/api/v1/recipes/dataset-version',json={}).status_code==422
    duplicate=client.post('/api/v1/recipes',json={'draft_id':draft['draft_id'],'recipe':draft['recipe'],'auto_triage':False})
    assert duplicate.json()['already_saved'] and duplicate.json()['library_id']==item['library_id']
    route='/api/v1/recipes/'+item['library_id']
    original=seed_snapshot()['examples'][0]
    bad=client.post(route+'/review',json={'revision':1,'labels':['unclear','meal-prep'],'reviewer':'Test fixture','rationale':'Invalid label combination.'})
    assert bad.status_code==422
    reviewed=client.post(route+'/review',json={'revision':1,'labels':original['labels'],'reviewer':'Test fixture','rationale':original['rationale']})
    assert reviewed.status_code==200,reviewed.text
    assert reviewed.json()['reviewed'] and reviewed.json()['review']['example']['reviewed_by']=='Test fixture'
    changed={**draft['recipe'],'title':'Edited oatmeal title'}
    edited=client.put(route,json={'revision':1,'recipe':changed,'actor':'Test fixture'})
    assert edited.status_code==200 and edited.json()['revision']==2 and not edited.json()['reviewed']
    assert client.post(route+'/review',json={'revision':1,'labels':['meal-prep'],'reviewer':'Test fixture','rationale':'Stale.'}).status_code==409
    history=client.get(route+'/history').json()
    assert len(history['reviews'])==1 and len(history['edits'])==1


def test_only_current_human_reviews_enter_versioned_library_dataset(client):
    for index in range(4):
        item,_=save(client,index)
        if index<3:
            original=seed_snapshot()['examples'][index]
            response=client.post('/api/v1/recipes/'+item['library_id']+'/review',json={
                'revision':1,'labels':original['labels'],'reviewer':'Test fixture','rationale':original['rationale']})
            assert response.status_code==200,response.text
    response=client.post('/api/v1/recipes/dataset-version',json={})
    assert response.status_code==200,response.text
    payload=response.json();splits=verify_snapshot(payload)
    assert len(payload['examples'])==3 and all(item['reviewed'] for item in payload['examples'])
    assert all(splits.values()) and client.get('/api/v1/datasets/versions/'+payload['metadata']['version']).status_code==200
    again=client.post('/api/v1/recipes/dataset-version',json={}).json()
    assert again['metadata']['version']==payload['metadata']['version']


def test_auto_triage_persists_raw_prediction_without_verifying_labels(client,monkeypatch):
    import app.library as library
    seen=[]
    monkeypatch.setattr(library,'catalog',lambda _: [{'key':'fireworks','available':True}])
    def predict(value,*args):
        seen.append(value.model_dump())
        return {'valid_json':True,'prediction':{'labels':['meal-prep'],'explanation':'Fixture only'},'raw_output':'{"labels":["meal-prep"],"explanation":"Fixture only"}','model':'fixture','revision':None}
    monkeypatch.setattr(library,'generate_one',predict)
    item,_=save(client,auto=True)
    stored=client.get('/api/v1/recipes').json()['examples'][0]
    assert stored['prediction']['result']['model']=='fixture' and not stored['reviewed']
    assert seen==[recipe()] and 'labels' not in seen[0]
    assert len(client.get('/api/v1/recipes/'+item['library_id']+'/history').json()['predictions'])==1


def test_comparison_uses_identical_recipe_and_settings_and_retains_provider_failure(client,monkeypatch):
    import app.playground as playground
    monkeypatch.setattr(playground,'catalog',lambda _: [{'key':'fireworks','title':'Fireworks','available':True},{'key':'base','title':'Base','available':True}])
    calls=[]
    def generate(value,key,db,temperature,max_new_tokens):
        calls.append((value.model_dump(),temperature,max_new_tokens))
        if key=='base':raise ProviderError('Fixture provider unavailable')
        return {'valid_json':False,'prediction':None,'raw_output':'broken','error':'Invalid JSON','model':'fixture'}
    monkeypatch.setattr(playground,'generate_one',generate)
    body={'recipe':recipe(),'models':['fireworks','base'],'temperature':0,'max_new_tokens':64}
    response=client.post('/api/v1/playground/comparisons',json=body)
    assert response.status_code==202,response.text
    job=client.get('/api/v1/playground/comparisons/'+response.json()['run_id']).json()
    assert job['status']=='completed' and job['result']['status']=='partial'
    assert calls==[(recipe(),0,64),(recipe(),0,64)] and job['result']['rows'][1]['error']=='Fixture provider unavailable'
    assert not job['result']['training_exported']
    assert client.post('/api/v1/playground/comparisons',json={**body,'models':['base','base']}).status_code==422
