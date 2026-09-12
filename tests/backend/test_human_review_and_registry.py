import copy,json,os
from pathlib import Path
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,text
from sqlalchemy.orm import Session
from app.main import app
from app.db import get_session,build_engine
from app.config import Settings
from app.migrate import migrate
from app.curation import insert,approved_example
from recipetriage_ml.training.data import seed_snapshot,verify_snapshot
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.training.data import file_hash

pytestmark=[pytest.mark.integration,pytest.mark.skipif(os.getenv('RUN_DB_TESTS')!='1',reason='Requires PostgreSQL')]


@pytest.fixture
def isolated_db():
    """A separate schema avoids touching the user's production pointer or review queue."""
    primary=build_engine(Settings());name='test_lifecycle_'+uuid4().hex
    with primary.begin() as c:c.execute(text('CREATE SCHEMA '+name))
    engine=create_engine(primary.url,hide_parameters=True,connect_args={'options':'-c search_path='+name})
    migrate(engine)
    def session():
        with Session(engine) as db:yield db
    app.dependency_overrides[get_session]=session
    try:yield engine,TestClient(app)
    finally:
        app.dependency_overrides.clear();engine.dispose()
        with primary.begin() as c:c.execute(text('DROP SCHEMA '+name+' CASCADE'))
        primary.dispose()


def proposal():
    # Offline fixture, never exported as an actual Fireworks/human training record.
    draft={'recipe':{'title':'Quick Millet Leeks','ingredients':['millet','leeks','sage'],
        'instructions':['Stuff leeks with millet for 35 minutes.','Bake for 25 minutes.'],
        'equipment':['oven'],'time_minutes':60,'pantry_items':None},'labels':['weekend-project'],
        'rationale':'A substantial 60-minute active project.','category':'misleading-title','challenge_notes':'Fixture'}
    return {'candidate_id':str(uuid4()),'kind':'synthetic','revision':1,'status':'pending','created_at':'test',
        'draft':draft,'original':copy.deepcopy(draft),'history':[],'review':None,'quality':{'passed':True,'errors':[],'warnings':[]},
        'category':'misleading-title','generation_id':str(uuid4()),'model':'test-fixture','prompt_version':'fixture',
        'prompt_sha256':'0'*64,'finish_reason':'stop'}


def test_explicit_review_edit_reset_export_boundary_and_frozen_holdouts(isolated_db):
    engine,client=isolated_db;item=proposal()
    with Session(engine) as db:insert(db,item);db.commit()
    assert client.post('/api/v1/curation/build-version',json={}).status_code==422
    assert client.post('/api/v1/curation/seed-review',json={}).status_code==200
    endpoint='/api/v1/curation/candidates/'+item['candidate_id']+'/review'
    edited=copy.deepcopy(item['draft']);edited['rationale']='Preparing the filled leeks is active work; total time is 60 minutes.'
    r=client.post(endpoint,json={'revision':1,'action':'edit','reviewer':'Test fixture','draft':edited});assert r.status_code==200,r.text
    assert r.json()['status']=='pending' and r.json()['review'] is None
    assert client.post(endpoint,json={'revision':1,'action':'approve','reviewer':'Test fixture'}).status_code==409
    r=client.post(endpoint,json={'revision':2,'action':'approve','reviewer':'Test fixture'});assert r.status_code==200,r.text
    approved=r.json()['approved_example']
    assert approved['provenance']['model']=='test-fixture'
    assert client.post('/api/v1/datasets/preview',json={'raw':json.dumps([approved])}).status_code==422
    for row in client.get('/api/v1/curation/candidates').json():
        if row['kind']=='seed':
            r=client.post('/api/v1/curation/candidates/'+row['candidate_id']+'/review',json={'revision':1,'action':'approve','reviewer':'Test fixture'})
            assert r.status_code==200,r.text
    r=client.post('/api/v1/curation/build-version',json={});assert r.status_code==200,r.text
    payload=r.json();verify_snapshot(payload)
    assert payload['metadata']['counts']=={'train':6,'validation':1,'test':1}
    for name in ['validation','test']:assert payload['metadata']['split_ids'][name]==seed_snapshot()['metadata']['split_ids'][name]
    tampered=copy.deepcopy(approved);tampered['labels']=['unclear']
    from recipetriage_ml.data.schemas import TrainingExample
    with pytest.raises(ValueError,match='changed after approval'):TrainingExample.model_validate(tampered)


def test_choices_require_explicit_content_bound_decision(isolated_db):
    engine,client=isolated_db
    from app.alignment import insert_pairs
    from recipetriage_ml.alignment.preferences import generate_pairs
    from recipetriage_ml.inference.contracts import Generation
    class Provider:
        model='fixture'
        def generate(self,*args):return Generation(raw_output='{}',model='fixture',finish_reason='stop')
    row=generate_pairs(local=Provider(),hosted=Provider(),count=1)[0]
    with Session(engine) as db:insert_pairs(db,[row])
    route='/api/v1/alignment/pairs/'+row['pair_id']+'/choice'
    assert client.post(route,json={'choice':'a','reviewer':'Test','pair_sha256':'wrong'}).status_code==409
    r=client.post(route,json={'choice':'neither','reviewer':'Test','pair_sha256':row['pair_sha256']});assert r.status_code==200,r.text
    assert r.json()['status']=='neither'
    assert client.post(route,json={'choice':'a','reviewer':'Test','pair_sha256':row['pair_sha256']}).status_code==409
    assert client.post('/api/v1/alignment/train/dpo',json={}).status_code==422


def test_promotions_checksums_single_production_and_rollback(isolated_db,tmp_path):
    import runpy
    engine,client=isolated_db
    helper=runpy.run_path(str(Path(__file__).parents[1]/'ml/test_alignment_lifecycle.py'))
    benchmark,shortcut=helper['perfect_evidence']()
    ids=[]
    for i in range(2):
        directory=tmp_path/str(i);directory.mkdir();(directory/'weights.bin').write_bytes(b'test weights')
        mid=str(uuid4());ids.append(mid)
        payload={'model_id':mid,'name':'Test fixture','identity':benchmark['model_config'],'format':'test',
            'artifact_directory':str(directory),'artifact_files':{'weights.bin':file_hash(directory/'weights.bin')},
            'dataset_lineage':{'fixture':True},'training_config':{},'benchmark':benchmark,'shortcut':shortcut,
            'reference_benchmark':benchmark,'evidence_sha256':'fixture'}
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO model_registry(model_id,stage,payload) VALUES (:id,'candidate',CAST(:p AS jsonb))"),{'id':mid,'p':json.dumps(payload)})
    action={'actor':'Test fixture','reason':'Isolated transaction test'}
    assert client.post('/api/v1/models/'+ids[0]+'/stage',json={**action,'stage':'production'}).status_code==409
    assert client.post('/api/v1/models/'+ids[0]+'/stage',json={**action,'stage':'staging'}).status_code==200
    file=tmp_path/'0/weights.bin';file.write_bytes(b'tampered')
    assert client.post('/api/v1/models/'+ids[0]+'/stage',json={**action,'stage':'production'}).status_code==409
    file.write_bytes(b'test weights')
    assert client.post('/api/v1/models/'+ids[0]+'/stage',json={**action,'stage':'production'}).status_code==200
    for stage in ['staging','production']:
        r=client.post('/api/v1/models/'+ids[1]+'/stage',json={**action,'stage':stage});assert r.status_code==200,r.text
    with Session(engine) as db:
        from app.registry import production
        assert production(db)['model_id']==ids[1]
    r=client.post('/api/v1/models/'+ids[0]+'/rollback',json={**action,'stage':'production'});assert r.status_code==200,r.text
    assert r.json()['action']=='rollback' and r.json()['previous_production_id']==ids[1]
    with Session(engine) as db:
        assert production(db)['model_id']==ids[0]
        assert db.execute(text("SELECT count(*) FROM model_registry WHERE stage='production'")).scalar_one()==1


def test_alignment_job_blocks_benchmark_queue(isolated_db):
    engine,client=isolated_db
    with engine.begin() as db:
        db.execute(text("INSERT INTO learning_jobs(run_id,kind,payload) VALUES (:id,'grpo',CAST(:p AS jsonb))"),
                   {'id':str(uuid4()),'p':json.dumps({'status':'running'})})
    response=client.post('/api/v1/benchmarks/runs',json={'provider':'hf-base','reasoning':'disabled'})
    assert response.status_code==409 and 'alignment worker' in response.text


def test_pending_generator_revision_repair_is_idempotent_and_preserves_content(isolated_db):
    engine,client=isolated_db;item=proposal();item['revision']=None
    with Session(engine) as db:insert(db,item);db.commit()
    migrate(engine);migrate(engine)
    fixed=client.get('/api/v1/curation/candidates').json()[0]
    assert fixed['revision']==1 and fixed['model_revision'] is None
    assert fixed['draft']==item['draft'] and fixed['history']==[] and fixed['review'] is None
    assert fixed['status']=='pending'
