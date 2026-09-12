"""Preference review and isolated learning jobs. No inferred or automated human choices."""
import json
from pathlib import Path
from typing import Literal
from uuid import UUID,uuid4
from fastapi import APIRouter,BackgroundTasks,Depends,HTTPException,Request,Response
from pydantic import BaseModel,ConfigDict,Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from app import jobs
from recipetriage_ml.alignment.preferences import PreferencePair,HumanChoice,generate_pairs,preference_dataset,reference_directory
from recipetriage_ml.alignment.runner import LearningConfig
from recipetriage_ml.alignment.rewards import WEIGHTS,VERSION
from recipetriage_ml.training.runs import now,run_root

router=APIRouter(prefix='/api/v1/alignment',tags=['Human preferences and learning'])


@router.get('/options')
def options():
    return {'defaults':LearningConfig().model_dump(),'reward_weights':WEIGHTS,'reward_version':VERSION,
            'reference_available':(reference_directory()/'adapter/adapter_model.safetensors').exists(),
            'dpo_minimum_independent_choices':3,'grpo_production_changed':False}


@router.get('/pairs')
def pairs(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM preference_pairs ORDER BY created_at,pair_id'))]


def insert_pairs(session,rows):
    for raw in rows:
        pair=PreferencePair.model_validate(raw)
        if pair.status!='pending' or pair.decision is not None:raise ValueError('Imported generations must not contain human choices')
        session.execute(text('INSERT INTO preference_pairs(pair_id,status,payload) VALUES (:id,:status,CAST(:p AS jsonb)) ON CONFLICT(pair_id) DO NOTHING'),
                        {'id':pair.pair_id,'status':'pending','p':json.dumps(pair.model_dump())})
    session.commit()


@router.post('/pairs/generate',status_code=202)
def generate(request:Request,background:BackgroundTasks,session:Session=Depends(get_session)):
    if not options()['reference_available']:raise HTTPException(422,'Install the saved reference adapter in /training/references/qlora-v1; see README')
    job=jobs.create(session,'preference-generation',{})
    def operation():
        def save(rows):
            with Session(request.app.state.engine) as db:insert_pairs(db,rows)
        return generate_pairs(on_update=save)
    background.add_task(jobs.execute,request.app.state.engine,job,operation);return job


class ChoiceRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    choice:Literal['a','b','tie','neither']
    reviewer:str=Field(min_length=1,max_length=100,pattern=r'.*\S.*')
    notes:str=Field(default='',max_length=4000)
    pair_sha256:str


@router.post('/pairs/{pair_id}/choice')
def choose(pair_id:UUID,body:ChoiceRequest,session:Session=Depends(get_session)):
    value=session.execute(text('SELECT payload FROM preference_pairs WHERE pair_id=:id FOR UPDATE'),{'id':str(pair_id)}).scalar_one_or_none()
    if value is None:raise HTTPException(404,'Pair not found')
    pair=PreferencePair.model_validate(value)
    if pair.decision is not None:raise HTTPException(409,'This explicit choice is immutable; create a new pair for another review')
    if body.pair_sha256!=pair.pair_sha256:raise HTTPException(409,'Candidate evidence changed; reload before choosing')
    pair.decision=HumanChoice(**body.model_dump(),decided_at=now(),decision_id=str(uuid4()))
    pair.status='chosen' if body.choice in {'a','b'} else body.choice
    value=PreferencePair.model_validate(pair.model_dump()).model_dump()
    session.execute(text('UPDATE preference_pairs SET status=:status,payload=CAST(:p AS jsonb) WHERE pair_id=:id'),
                    {'status':pair.status,'p':json.dumps(value),'id':pair.pair_id})
    session.commit();return value


@router.get('/pairs/download')
def download_pairs(session:Session=Depends(get_session)):
    return Response(json.dumps(pairs(session),indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename=preference-pairs.json'})


@router.get('/jobs')
def list_jobs(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM learning_jobs ORDER BY created_at DESC LIMIT 50'))]


@router.post('/train/{method}',status_code=202)
def start(method:Literal['dpo','grpo'],config:LearningConfig,request:Request,background:BackgroundTasks,session:Session=Depends(get_session)):
    data=pairs(session) if method=='dpo' else None
    try:
        if method=='dpo':preference_dataset(data)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if not options()['reference_available']:raise HTTPException(422,'Reference adapter is unavailable; see README installation command')
    job=jobs.create(session,method,config.model_dump())
    def operation():
        from recipetriage_ml.alignment.runner import run
        result=run(method,config,pairs=data,root=run_root()/'alignment')
        from app.registry import register_adapter
        with Session(request.app.state.engine) as db:register_adapter(db,run_root()/'alignment'/result['run_id'],result)
        return result
    background.add_task(jobs.execute,request.app.state.engine,job,operation);return job


class RetrainRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    dataset_version:str


@router.post('/retrain-approved',status_code=202)
def retrain(body:RetrainRequest,request:Request,background:BackgroundTasks,session:Session=Depends(get_session)):
    from app.training import snapshot_for
    from app.curation import candidates
    snapshot=snapshot_for(body.dataset_version,session)
    if not snapshot['metadata'].get('holdouts_frozen') or snapshot['metadata']['status']!='reviewed':
        raise HTTPException(422,'Build the improved version from the explicit review queue first')
    approved={x['approved_example']['recipe']['id']:x['approved_example'] for x in candidates(session) if x['status']=='approved'}
    if any(approved.get(x['recipe']['id'])!=x for x in snapshot['examples']):raise HTTPException(422,'Dataset records no longer match persisted human approvals')
    job=jobs.create(session,'qlora-reviewed',body.model_dump())
    def operation():
        from recipetriage_ml.training.config import SFTConfig
        from recipetriage_ml.training.sft import run_local,load_adapter
        from recipetriage_ml.evaluation import shortcut_tests,behaviors
        source=reference_directory();source_run=json.loads((source/'run.json').read_text())
        config=SFTConfig.model_validate({**source_run['config'],'dataset_version':body.dataset_version})
        root=run_root()/'reviewed';baseline=json.loads((source/'baseline.json').read_text())
        result=run_local(config,snapshot,root=root,reference_baseline=baseline)
        directory=root/result['run_id'];provider=load_adapter(directory)
        shortcut=shortcut_tests.run_suite(provider,provider.identity);(directory/'shortcut.json').write_text(json.dumps(shortcut,indent=2)+'\n')
        from recipetriage_ml.alignment.preferences import ml_directory
        before=json.loads((ml_directory()/'evaluation/results/diagnostics-v1/qlora.json').read_text())['shortcut']
        result['behavior_comparison']=behaviors.compare(behaviors.report(source_run['benchmark'],before),behaviors.report(result['benchmark'],shortcut))
        (directory/'run.json').write_text(json.dumps(result,indent=2)+'\n')
        from app.registry import register_adapter
        with Session(request.app.state.engine) as db:register_adapter(db,directory,result)
        return result
    background.add_task(jobs.execute,request.app.state.engine,job,operation);return job


if __name__=='__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser=argparse.ArgumentParser();parser.add_argument('pairs',type=Path,nargs='?');parser.add_argument('--run',type=Path)
    args=parser.parse_args();engine=build_engine(Settings())
    if bool(args.pairs)==bool(args.run):parser.error('Choose a pairs file or --run DIRECTORY')
    with Session(engine) as db:
        if args.pairs:insert_pairs(db,json.loads(args.pairs.read_text()));print('Imported unchosen preference pairs')
        else:
            result=json.loads((args.run/'run.json').read_text())
            if result['status']!='completed' or result['kind'] not in {'grpo','dpo'}:raise ValueError('Only completed alignment evidence can be imported')
            from app.registry import register_adapter
            register_adapter(db,args.run,result)
            if (args.run/'reward-components.json').exists():result['reward_components']=json.loads((args.run/'reward-components.json').read_text())
            payload={'run_id':result['run_id'],'kind':result['kind'],'status':'completed','config':result['config'],
                     'created_at':result['created_at'],'updated_at':result['updated_at'],'result':result,'error':None}
            existing=db.execute(text('SELECT payload FROM learning_jobs WHERE run_id=:id'),{'id':payload['run_id']}).scalar_one_or_none()
            if existing is not None and existing!=payload:raise ValueError('Run ID already has different evidence')
            if existing is None:
                db.execute(text('INSERT INTO learning_jobs(run_id,kind,payload) VALUES (:id,:kind,CAST(:p AS jsonb))'),{'id':payload['run_id'],'kind':payload['kind'],'p':json.dumps(payload)})
                db.commit()
            print('Imported completed',payload['kind'],'experiment as a candidate; no production change')
    engine.dispose()
