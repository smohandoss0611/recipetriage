"""Human-only approval boundary. Generation and edits always produce pending records."""
import copy
import json
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from app import jobs
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.data.schemas import Recipe, TrainingExample
from recipetriage_ml.data.synthetic_fireworks import Draft, generate_batch, quality_checks
from recipetriage_ml.data.approved import build_approved
from recipetriage_ml.training.data import seed_snapshot, verify_snapshot
from recipetriage_ml.training.runs import now

router=APIRouter(prefix='/api/v1/curation', tags=['Human review'])


class GenerateRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    count:int=Field(default=6,ge=1,le=6,strict=True)


class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int=Field(ge=1,strict=True)
    action:Literal['approve','reject','edit']
    reviewer:str=Field(min_length=1,max_length=100,pattern=r'.*\S.*')
    notes:str=Field(default='',max_length=4000)
    draft:dict|None=None


def insert(session, item):
    session.execute(text('INSERT INTO review_candidates(candidate_id,kind,status,payload) VALUES (:id,:kind,:status,CAST(:payload AS jsonb)) ON CONFLICT(candidate_id) DO NOTHING'),
                    {'id':item['candidate_id'],'kind':item['kind'],'status':item['status'],'payload':json.dumps(item)})


@router.get('/candidates')
def candidates(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM review_candidates ORDER BY created_at,candidate_id'))]


@router.post('/seed-review')
def queue_seed(session:Session=Depends(get_session)):
    """Legacy proposals require review too before entering the improved version."""
    from uuid import uuid5,NAMESPACE_URL
    for example in seed_snapshot()['examples']:
        cid=str(uuid5(NAMESPACE_URL,'recipetriage-seed-review:'+example['recipe']['id']))
        insert(session,{'candidate_id':cid,'kind':'seed','revision':1,'status':'pending','created_at':now(),
            'original':example,'draft':example,'review':None,'history':[],'quality':{'passed':True,'errors':[],
            'warnings':['Legacy annotation is provisional. Check the source and every label before approving.']}})
    session.commit();return candidates(session)


@router.post('/generate',status_code=202)
def generate(body:GenerateRequest,request:Request,background:BackgroundTasks,session:Session=Depends(get_session)):
    import os
    if not os.getenv('FIREWORKS_API_KEY'):raise HTTPException(422,'Configure FIREWORKS_API_KEY on the server')
    job=jobs.create(session,'synthetic-generation',body.model_dump())
    def operation():
        def save(batch):
            with Session(request.app.state.engine) as db:
                for item in batch['items']:insert(db,item)
                db.commit()
        return generate_batch(body.count,on_update=save,existing=[x['recipe'] for x in seed_snapshot()['examples']])
    background.add_task(jobs.execute,request.app.state.engine,job,operation)
    return job


def approved_example(item, reviewer, review_id, timestamp):
    if item['kind']=='seed':
        return TrainingExample.model_validate({**item['draft'],'reviewed':True,'reviewed_by':reviewer}).model_dump()
    draft=Draft.model_validate(item['draft'])
    recipe=Recipe.model_validate({**draft.recipe.model_dump(),'id':'syn-'+item['candidate_id'],
        'source_type':'synthetic','source_uri':'synthetic://fireworks/'+item['candidate_id'],
        'source_notes':'Fireworks-generated fictional proposal; human reviewed; see provenance.'})
    content={'recipe':recipe.model_dump(),'labels':sorted(draft.labels),'rationale':' '.join(draft.rationale.split()),
             'annotation_notes':draft.challenge_notes}
    content['annotation_notes']=' '.join(content['annotation_notes'].split())
    provenance={'source':'synthetic-fireworks','generation_id':item['generation_id'],'candidate_id':item['candidate_id'],
        'candidate_revision':item['revision'],'model':item['model'],'prompt_version':item['prompt_version'],
        'prompt_sha256':item['prompt_sha256'],'human_review_id':review_id,'approved_at':timestamp,
        'approved_content_sha256':digest(content)}
    return TrainingExample.model_validate({**content,'reviewed':True,'reviewed_by':reviewer,'provenance':provenance}).model_dump()


@router.post('/candidates/{candidate_id}/review')
def review(candidate_id:UUID,body:ReviewRequest,session:Session=Depends(get_session)):
    item=session.execute(text('SELECT payload FROM review_candidates WHERE candidate_id=:id FOR UPDATE'),{'id':str(candidate_id)}).scalar_one_or_none()
    if item is None:raise HTTPException(404,'Candidate not found')
    if item['revision']!=body.revision:raise HTTPException(409,'Candidate changed; reload before deciding')
    if item['status']=='approved':raise HTTPException(409,'Approved content is immutable; create a new proposal for further edits')
    stamp=now();decision={'review_id':str(uuid4()),'action':body.action,'reviewer':body.reviewer.strip(),'at':stamp,
                        'revision':item['revision'],'content_sha256':digest(item['draft']),'notes':body.notes}
    try:
        if body.action=='edit':
            if body.draft is None:raise ValueError('Provide the edited draft')
            item['draft']=body.draft;item['revision']+=1;item['status']='pending';item['review']=None
        elif body.draft is not None:raise ValueError('Save edits separately, then explicitly approve the displayed revision')
        if item['kind']=='synthetic':
            others=[x['draft']['recipe'] for x in candidates(session) if x['candidate_id']!=item['candidate_id'] and x['status']=='approved']
            item['quality']=quality_checks(item['draft'],focus=item['category'],others=others,
                finish_reason='stop' if item['revision']>1 else item.get('finish_reason'))
        else:
            TrainingExample.model_validate({**item['draft'],'reviewed':False,'reviewed_by':None})
        if body.action=='approve':
            if not item['quality']['passed']:raise ValueError('Resolve quality errors before approving')
            item['approved_example']=approved_example(item,body.reviewer.strip(),decision['review_id'],stamp)
            item['status']='approved';item['review']=decision
        elif body.action=='reject':item['status']='rejected';item['review']=decision
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    item['history'].append(decision)
    session.execute(text('UPDATE review_candidates SET status=:status,payload=CAST(:payload AS jsonb) WHERE candidate_id=:id'),
                    {'status':item['status'],'payload':json.dumps(item),'id':str(candidate_id)})
    session.commit();return item


def verify_approvals(records,session):
    for row in records:
        if not isinstance(row,dict) or not isinstance(row.get('recipe'),dict):continue
        if row['recipe'].get('source_type')=='synthetic' or row.get('provenance'):
            cid=(row.get('provenance') or {}).get('candidate_id')
            if not cid:raise HTTPException(422,'Synthetic examples require a persisted human review')
            try:UUID(cid)
            except ValueError:raise HTTPException(422,'Invalid candidate ID')
            item=session.execute(text('SELECT payload FROM review_candidates WHERE candidate_id=:id'),{'id':cid}).scalar_one_or_none()
            if not item or item['status']!='approved' or item.get('approved_example')!=row:
                raise HTTPException(422,'Synthetic example differs from its persisted human approval')


@router.post('/build-version')
def build_version(session:Session=Depends(get_session)):
    approved=[x['approved_example'] for x in candidates(session) if x['status']=='approved']
    try:
        payload=build_approved(seed_snapshot(),approved);verify_snapshot(payload)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    verify_approvals(approved,session)
    session.execute(text('INSERT INTO dataset_versions(version,payload) VALUES (:v,CAST(:p AS jsonb)) ON CONFLICT(version) DO NOTHING'),
                    {'v':payload['metadata']['version'],'p':json.dumps(payload)})
    session.commit()
    return session.execute(text('SELECT payload FROM dataset_versions WHERE version=:v'),{'v':payload['metadata']['version']}).scalar_one()


def import_batch(session,payload):
    for item in payload['items']:
        if item['status']!='pending' or item.get('review') is not None:raise ValueError('Generator imports cannot approve examples')
        if digest(item['messages'])!=item['prompt_sha256']:raise ValueError('Generation prompt hash mismatch')
        insert(session,item)
    session.commit()


if __name__=='__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path);args=parser.parse_args()
    engine=build_engine(Settings())
    with Session(engine) as db:import_batch(db,json.loads(args.batch.read_text()))
    engine.dispose();print('Imported proposals as pending human review')
