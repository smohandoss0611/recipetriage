"""Immutable model versions, transactional promotion and audited rollback."""
import json,os
from pathlib import Path
from typing import Literal
from uuid import UUID,uuid4,uuid5,NAMESPACE_URL
from fastapi import APIRouter,Depends,HTTPException
from pydantic import BaseModel,ConfigDict,Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.training.data import file_hash
from recipetriage_ml.training.runs import now
from recipetriage_ml.deployment.gates import GateConfig,evaluate_gate

router=APIRouter(prefix='/api/v1/models',tags=['Model registry'])


def reference_benchmark():
    from recipetriage_ml.alignment.preferences import ml_directory
    return json.loads((ml_directory()/'training/results/experiments-v1/baseline.json').read_text())


def register_export(session,row,directory=None):
    directory=Path(directory or row['directory']).resolve()
    manifest=json.loads((directory/'deployment-manifest.json').read_text())
    if manifest['artifact_sha256']!=row['identity']['revision']:raise ValueError('Deployment evidence belongs to different weights')
    identity=row['identity'];model_id=str(uuid5(NAMESPACE_URL,'recipetriage-model:'+digest(identity)))
    if row['benchmark']['model_config']!=identity or row['shortcut']['model_config']!=identity:raise ValueError('Evaluation identity differs from exported weights')
    payload={'model_id':model_id,'name':row['name'],'identity':identity,'format':manifest['format'],
        'artifact_directory':str(directory),'artifact_files':{**manifest['files'],'deployment-manifest.json':file_hash(directory/'deployment-manifest.json')},
        'dataset_lineage':{'source_adapter':manifest['source_adapter'],'base_model':identity['base_model'],'base_revision':identity['base_revision']},
        'training_config':None,'benchmark':row['benchmark'],'shortcut':row['shortcut'],'reference_benchmark':reference_benchmark(),
        'operational_measurements':{k:v for k,v in row.items() if k not in {'benchmark','shortcut'}},'created_at':now()}
    verify_artifacts(payload)
    payload['evidence_sha256']=digest({k:v for k,v in payload.items() if k!='created_at'})
    existing=session.execute(text('SELECT payload FROM model_registry WHERE model_id=:id'),{'id':model_id}).scalar_one_or_none()
    if existing:return existing
    session.execute(text("INSERT INTO model_registry(model_id,stage,payload) VALUES (:id,'candidate',CAST(:p AS jsonb))"),{'id':model_id,'p':json.dumps(payload)})
    session.commit();return payload


def thresholds():return GateConfig.model_validate_json(os.getenv('MODEL_GATE_JSON') or '{}')


@router.get('/gate-config')
def gate_config():return thresholds().model_dump()


def register_adapter(session,directory,run=None):
    directory=Path(directory).resolve();identity=json.loads((directory/'adapter-provenance.json').read_text())
    benchmark=json.loads((directory/'benchmark.json').read_text())
    shortcut=json.loads((directory/'shortcut.json').read_text()) if (directory/'shortcut.json').exists() else None
    run=run or json.loads((directory/'run.json').read_text())
    if benchmark['model_config']!=identity or (shortcut and shortcut['model_config']!=identity):raise ValueError('Evaluation identity differs from adapter provenance')
    base=Path(__file__).resolve().parents[2]/'ml/training/results/experiments-v1/baseline.json'
    # Installed package is under site-packages: importlib resources locate bundled evidence in Docker.
    if not base.exists():
        import recipetriage_ml
        base=Path(recipetriage_ml.__file__).resolve().parent/'training/results/experiments-v1/baseline.json'
    if not base.exists():base=Path('/app/ml/training/results/experiments-v1/baseline.json')
    baseline=json.loads(base.read_text())
    if file_hash(directory/'adapter/adapter_model.safetensors')!=identity['revision']:raise ValueError('Checkpoint checksum mismatch')
    model_id=str(uuid5(NAMESPACE_URL,'recipetriage-model:'+digest(identity)))
    payload={'model_id':model_id,'name':identity['model'],'identity':identity,'format':'peft-nf4' if identity.get('quantization')=='nf4-double-float32' else 'peft-fp32',
        'artifact_directory':str(directory),'artifact_files':{str(p.relative_to(directory)):file_hash(p) for p in (directory/'adapter').iterdir() if p.is_file()},
        'dataset_lineage':run.get('dataset_lineage') or run.get('dataset') or {'version':identity.get('dataset_version')},
        'training_config':run.get('config'),'benchmark':benchmark,'shortcut':shortcut,'reference_benchmark':baseline,
        'created_at':now()}
    payload['evidence_sha256']=digest({k:v for k,v in payload.items() if k!='created_at'})
    existing=session.execute(text('SELECT payload FROM model_registry WHERE model_id=:id'),{'id':model_id}).scalar_one_or_none()
    if existing:
        if existing['identity']!=identity:raise ValueError('Model version already exists with different identity')
        return existing
    session.execute(text("INSERT INTO model_registry(model_id,stage,payload) VALUES (:id,'candidate',CAST(:p AS jsonb))"),{'id':model_id,'p':json.dumps(payload)})
    session.commit();return payload


def verify_artifacts(model):
    directory=Path(model['artifact_directory'])
    if model['format'].startswith('peft-'):
        provenance=directory/'adapter-provenance.json'
        if not provenance.is_file() or json.loads(provenance.read_text())!=model['identity']:
            raise ValueError('Adapter provenance changed after registration')
    for name,expected in model['artifact_files'].items():
        path=directory/name
        if directory not in path.resolve().parents or not path.is_file() or file_hash(path)!=expected:raise ValueError('Model files unavailable or changed; restore this immutable version before promotion/rollback')


@router.get('')
def models(session:Session=Depends(get_session)):
    reference=session.execute(text("SELECT payload FROM model_registry WHERE stage='production'")).scalar_one_or_none()
    return [{**row[1],'stage':row[0],'gate':evaluate_gate(row[1]['benchmark'],row[1]['shortcut'],reference['benchmark'] if reference else row[1]['reference_benchmark'],thresholds())}
            for row in session.execute(text('SELECT stage,payload FROM model_registry ORDER BY created_at DESC'))]


@router.get('/history')
def history(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM model_stage_events ORDER BY created_at DESC'))]


class StageRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    stage:Literal['staging','production','archived']
    actor:str=Field(min_length=1,max_length=100,pattern=r'.*\S.*')
    reason:str=Field(min_length=1,max_length=2000,pattern=r'.*\S.*')


def transition(session,model_id,stage,actor,reason,rollback=False):
    session.execute(text('SELECT pg_advisory_xact_lock(74291003)'))
    row=session.execute(text('SELECT stage,payload FROM model_registry WHERE model_id=:id FOR UPDATE'),{'id':str(model_id)}).first()
    if not row:raise HTTPException(404,'Model version not found')
    old,model=row
    if not rollback and (old,stage) not in {('candidate','staging'),('staging','production'),('candidate','archived'),('staging','archived')}:
        raise HTTPException(409,'Allowed transitions: candidate → staging → production, or archive an inactive version')
    if rollback:
        past=session.execute(text("SELECT event_id FROM model_stage_events WHERE model_id=:id AND payload->>'to'='production' LIMIT 1"),{'id':str(model_id)}).first()
        if not past or old!='archived':raise HTTPException(409,'Rollback requires an archived version that previously served in production')
    gate=None
    if stage in {'staging','production'}:
        try:verify_artifacts(model)
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc
        reference=session.execute(text("SELECT payload FROM model_registry WHERE stage='production'")).scalar_one_or_none()
        gate=evaluate_gate(model['benchmark'],model['shortcut'],reference['benchmark'] if reference else model['reference_benchmark'],thresholds())
        gate['reference_model_id']=reference['model_id'] if reference else None
        if not gate['passed']:raise HTTPException(422,{'message':'Evaluation quality gate failed','gate':gate})
    prior=None
    if stage=='production':
        prior=session.execute(text("SELECT model_id FROM model_registry WHERE stage='production' FOR UPDATE")).scalar_one_or_none()
        if prior:session.execute(text("UPDATE model_registry SET stage='archived' WHERE model_id=:id"),{'id':prior})
    session.execute(text('UPDATE model_registry SET stage=:stage WHERE model_id=:id'),{'stage':stage,'id':str(model_id)})
    event={'event_id':str(uuid4()),'model_id':str(model_id),'from':old,'to':stage,'actor':actor.strip(),'reason':reason,
           'action':'rollback' if rollback else 'promote' if stage!='archived' else 'archive','at':now(),
           'previous_production_id':str(prior) if prior else None,'gate':gate,'evidence_sha256':model['evidence_sha256']}
    session.execute(text('INSERT INTO model_stage_events(event_id,model_id,actor,action,payload) VALUES (:id,:model,:actor,:action,CAST(:p AS jsonb))'),
                    {'id':event['event_id'],'model':str(model_id),'actor':actor,'action':event['action'],'p':json.dumps(event)})
    session.commit();return event


@router.post('/{model_id}/stage')
def promote(model_id:UUID,body:StageRequest,session:Session=Depends(get_session)):
    return transition(session,model_id,body.stage,body.actor,body.reason)


@router.post('/{model_id}/rollback')
def rollback(model_id:UUID,body:StageRequest,session:Session=Depends(get_session)):
    if body.stage!='production':raise HTTPException(422,'Rollback restores production')
    return transition(session,model_id,'production',body.actor,body.reason,rollback=True)


def production(session):
    model=session.execute(text("SELECT payload FROM model_registry WHERE stage='production'")).scalar_one_or_none()
    if model is None:raise HTTPException(503,'No model has passed the production promotion gate')
    verify_artifacts(model)
    return model


if __name__=='__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path,nargs='?');parser.add_argument('--deployment',type=Path);parser.add_argument('--artifact-directory',type=Path)
    args=parser.parse_args();engine=build_engine(Settings())
    if bool(args.directory)==bool(args.deployment):parser.error('Choose an adapter directory or --deployment RESULT.json')
    with Session(engine) as db:
        value=register_export(db,json.loads(args.deployment.read_text()),args.artifact_directory) if args.deployment else register_adapter(db,args.directory)
        print(value['model_id'])
    engine.dispose()
