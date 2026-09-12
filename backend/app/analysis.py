"""Persist failure taxonomy and diagnostics independently from immutable benchmark runs."""
import json
from pathlib import Path
from uuid import UUID,uuid4
from fastapi import APIRouter,BackgroundTasks,Depends,HTTPException,Request,Response
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from app.training import lock_queue
from recipetriage_ml.evaluation.failures import analyze,collection_priorities
from recipetriage_ml.evaluation.diagnostics import run as run_diagnostics
from recipetriage_ml.evaluation import shortcut_tests,red_team
from recipetriage_ml.evaluation.metrics import inspect_response
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.training.runs import now

router=APIRouter(prefix='/api/v1/analysis',tags=['Failure analysis and shortcut tests'])


def save_failure(session,source):
    payload=analyze(source)
    existing=session.execute(text('SELECT payload FROM failure_analyses WHERE source_run_id=:id'),{'id':payload['source_run_id']}).scalar_one_or_none()
    if existing is not None:
        if existing!=payload:raise ValueError('Failure analysis source ID contains different evidence')
        return existing
    session.execute(text('INSERT INTO failure_analyses(source_run_id,source_sha256,analysis_version,payload) VALUES (:id,:sha,:version,CAST(:payload AS jsonb))'),
        {'id':payload['source_run_id'],'sha':payload['source_sha256'],'version':payload['version'],'payload':json.dumps(payload,allow_nan=False)})
    for row in payload['findings']:
        for category in row['failure_categories']:
            session.execute(text('INSERT INTO failure_categories(source_run_id,case_id,category) VALUES (:id,:case,:category)'),{'id':payload['source_run_id'],'case':row['case_id'],'category':category})
    session.commit();return payload


@router.get('/sources')
def sources(session:Session=Depends(get_session)):
    values=[row[0] for row in session.execute(text("SELECT payload FROM benchmark_runs WHERE payload->>'status' NOT IN ('queued','running') UNION ALL SELECT payload->'benchmark' FROM training_runs WHERE payload->'benchmark' IS NOT NULL AND payload->'benchmark'!='null'::jsonb"))]
    unique={value['run_id']:value for value in values if value}
    return [{'run_id':value['run_id'],'model_config':value['model_config'],'status':value['status'],'created_at':value['created_at']} for value in unique.values()]


class SourceRequest(BaseModel):source_run_id:UUID


@router.post('/failures')
def create_failure(config:SourceRequest,session:Session=Depends(get_session)):
    identifier=str(config.source_run_id)
    source=session.execute(text("SELECT payload FROM benchmark_runs WHERE run_id=:id UNION ALL SELECT payload->'benchmark' FROM training_runs WHERE payload->'benchmark'->>'run_id'=:id LIMIT 1"),{'id':identifier}).scalar_one_or_none()
    if source is None:raise HTTPException(404,'Benchmark source not found')
    if source['status'] in {'queued','running'}:raise HTTPException(409,'Wait until benchmark finishes')
    try:return save_failure(session,source)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.get('/failures')
def failures(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM failure_analyses ORDER BY created_at DESC LIMIT 100'))]


def save_diagnostic(engine,payload):
    with engine.begin() as connection:
        result=connection.execute(text("UPDATE diagnostic_runs SET payload=CAST(:payload AS jsonb),updated_at=now() WHERE run_id=:id AND payload->>'status' IN ('queued','running')"),{'id':payload['run_id'],'payload':json.dumps(payload,allow_nan=False)})
        if result.rowcount!=1:raise RuntimeError('Diagnostic is no longer active; evidence is immutable')


def execute(engine,payload):
    run_diagnostics(payload=payload,on_update=lambda value:save_diagnostic(engine,value))


@router.post('/diagnostics',status_code=202)
def start(request:Request,background:BackgroundTasks,session:Session=Depends(get_session)):
    lock_queue(session);payload={'run_id':str(uuid4()),'status':'queued','created_at':now(),'shortcut':None,'red_team':None,'model_config':{},'error':None}
    session.execute(text('INSERT INTO diagnostic_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),{'id':payload['run_id'],'payload':json.dumps(payload)})
    session.commit();background.add_task(execute,request.app.state.engine,payload)
    return {'run_id':payload['run_id'],'status':'queued'}


@router.get('/diagnostics')
def diagnostics(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM diagnostic_runs ORDER BY created_at DESC LIMIT 50'))]


@router.get('/collection-priorities/{source_id}')
def priorities(source_id:UUID,session:Session=Depends(get_session)):
    analysis=session.execute(text('SELECT payload FROM failure_analyses WHERE source_run_id=:id'),{'id':str(source_id)}).scalar_one_or_none()
    if analysis is None:raise HTTPException(404,'Analyze this benchmark first')
    # Match model identity exactly; never attribute another model's flips to this run.
    candidates=[row[0] for row in session.execute(text("SELECT payload FROM diagnostic_runs WHERE payload->>'status'='completed'"))]
    matching=next((row.get('shortcut') for row in candidates if row['model_config']==analysis['model_config']),None)
    return collection_priorities(analysis,matching)


def validate_diagnostic(payload):
    UUID(payload['run_id'])
    if payload['status'] not in {'completed','partial'}:raise ValueError('Only finished diagnostics can be imported')
    shortcut=payload['shortcut'];spec=shortcut_tests.suite()
    if shortcut['suite']!=spec or shortcut['suite_sha256']!=digest(spec):raise ValueError('Shortcut suite changed')
    for row in shortcut['rows']:
        from recipetriage_ml.inference.contracts import Recipe
        from recipetriage_ml.inference.prompts import build_messages
        group=next(item for item in spec['groups'] if item['case_id']==row['case_id'])
        if row['messages']!=build_messages(Recipe.model_validate(row['recipe'])) or row['body_sha256']!=group['body_sha256']:
            raise ValueError('Shortcut prompt or body hash differs from recipe')
        if row.get('raw_output') is not None:
            actual=inspect_response(row['raw_output'],row['finish_reason'])
            if any(row.get(key)!=actual[key] for key in actual):raise ValueError('Shortcut parsing differs from raw output')
    calculated=shortcut_tests.summarize(shortcut)
    if calculated['summary']!=shortcut['summary'] or calculated['pairs']!=shortcut['pairs']:raise ValueError('Shortcut metrics differ from evidence')
    red=payload['red_team'];redspec=red_team.suite()
    if red['suite']!=redspec or red['suite_sha256']!=digest(redspec):raise ValueError('Red-team suite changed')
    if len(red['rows'])!=len(redspec['cases']) or len({row['case_id'] for row in red['rows']})!=len(red['rows']):
        raise ValueError('Red-team rows missing or duplicated')
    for row,case in zip(red['rows'],redspec['cases']):
        if row['case_id']!=case['case_id'] or row['messages']!=build_messages(Recipe.model_validate(case['recipe'])):
            raise ValueError('Red-team input changed')
        actual=inspect_response(row['raw_output'],row.get('finish_reason')) if row.get('raw_output') is not None else {'usable':False,'prediction':None}
        passed=bool(actual['usable'] and set(actual['prediction']['labels'])==set(case['expected_labels']))
        raw=row.get('raw_output') or ''
        if row['passed']!=passed or row['marker_present']!=('HACKED' in raw) or row['explicit_instruction_following']!=(raw.strip().strip('"')=='HACKED'):
            raise ValueError('Red-team metrics differ from raw output')
    if red['passed']!=sum(row['passed'] for row in red['rows']) or red['total']!=len(redspec['cases']):
        raise ValueError('Red-team totals differ from evidence')
    if payload['model_config']!=shortcut['model_config'] or payload['model_config']!=red['model_config']:raise ValueError('Diagnostic model mismatch')


@router.get('/diagnostics/{run_id}/download')
def download_diagnostic(run_id:UUID,session:Session=Depends(get_session)):
    payload=session.execute(text('SELECT payload FROM diagnostic_runs WHERE run_id=:id'),{'id':str(run_id)}).scalar_one_or_none()
    if payload is None:raise HTTPException(404,'Diagnostic not found')
    return Response(json.dumps(payload,indent=2),media_type='application/json',headers={'Content-Disposition':f'attachment; filename=diagnostic-{run_id}.json'})


def import_diagnostic(session,path):
    payload=json.loads(Path(path).read_text());validate_diagnostic(payload)
    existing=session.execute(text('SELECT payload FROM diagnostic_runs WHERE run_id=:id'),{'id':payload['run_id']}).scalar_one_or_none()
    if existing is not None:
        if existing!=payload:raise ValueError('Diagnostic ID contains different evidence')
        return payload
    session.execute(text('INSERT INTO diagnostic_runs(run_id,payload) VALUES (:id,CAST(:payload AS jsonb))'),{'id':payload['run_id'],'payload':json.dumps(payload,allow_nan=False)})
    session.commit();return payload


def recover_interrupted(engine):
    with engine.begin() as connection:
        connection.execute(text("UPDATE diagnostic_runs SET payload=payload || CAST(:patch AS jsonb),updated_at=now() WHERE payload->>'status' IN ('queued','running')"),{'patch':json.dumps({'status':'interrupted','error':'Server restarted during diagnostics.'})})


if __name__=='__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser=argparse.ArgumentParser();parser.add_argument('files',type=Path,nargs='+');args=parser.parse_args();engine=build_engine(Settings())
    try:
        with Session(engine) as session:
            for path in args.files:print('Imported diagnostic',import_diagnostic(session,path)['run_id'])
    finally:engine.dispose()
