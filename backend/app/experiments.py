"""PostgreSQL experiment registry, immutable evidence imports and config-driven jobs."""
import json
import logging
from pathlib import Path
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db import get_session
from app.training import lock_queue, snapshot_for, import_run
from app.lora import persist_child
from recipetriage_ml.training.experiments import MatrixConfig, default_matrix, new_experiment, run_matrix, compare_matrix, summary
from recipetriage_ml.training.qlora import capability
from recipetriage_ml.training.runs import now, ACTIVE

router=APIRouter(prefix='/api/v1/experiments',tags=['Experiment registry'])
logger=logging.getLogger(__name__)


@router.get('/options')
def options():
    return {'defaults':default_matrix().model_dump(),'qlora':capability(),
            'tracking':'MLflow; training evidence remains in PostgreSQL if tracking export fails',
            'matrix_limit':8,'fixed_benchmark':True}


def persist(engine,payload):
    with engine.begin() as connection:
        changed=connection.execute(text("UPDATE experiment_registry SET payload=CAST(:payload AS jsonb),updated_at=now() WHERE experiment_id=:id AND payload->>'status' IN ('queued','running')"),
            {'id':payload['experiment_id'],'payload':json.dumps(payload,allow_nan=False)})
        if changed.rowcount!=1:raise RuntimeError('Experiment record is not active; evidence is immutable')


def execute(engine,config,snapshot,payload):
    def child(value):
        persist_child(engine,value)
        if value['status']=='completed' and value.get('benchmark'):
            from app.analysis import save_failure
            with Session(engine) as session:save_failure(session,value['benchmark'])
    try:run_matrix(config,snapshot,experiment=payload,on_update=lambda value:persist(engine,value),on_run_update=child)
    except Exception:
        logger.exception('Experiment matrix failed')
        with engine.begin() as connection:
            connection.execute(text("UPDATE experiment_registry SET payload=payload || CAST(:patch AS jsonb),updated_at=now() WHERE experiment_id=:id AND payload->>'status' IN ('queued','running')"),
                {'id':payload['experiment_id'],'patch':json.dumps({'status':'failed','error':'Experiment worker failed; inspect logs.','finished_at':now()})})
            connection.execute(text("UPDATE training_runs SET payload=payload || CAST(:patch AS jsonb),updated_at=now() WHERE payload->>'experiment_id'=:id AND payload->>'status'=ANY(:active)"),
                {'id':payload['experiment_id'],'active':sorted(ACTIVE),'patch':json.dumps({'status':'interrupted','error':'Parent experiment stopped; inspect worker log.','finished_at':now()})})


def insert(session,payload):
    session.execute(text('INSERT INTO experiment_registry(experiment_id,name,dataset_version,benchmark_sha256,payload) VALUES (:id,:name,:dataset,:benchmark,CAST(:payload AS jsonb))'),
        {'id':payload['experiment_id'],'name':payload['name'],'dataset':payload['dataset_version'],'benchmark':payload['benchmark_sha256'],'payload':json.dumps(payload,allow_nan=False)})
    session.commit()


@router.post('',status_code=202)
def start(config:MatrixConfig,request:Request,background:BackgroundTasks,session:Session=Depends(get_session)):
    snapshot=snapshot_for(config.runs[0].dataset_version,session);lock_queue(session)
    if any(row.method=='qlora' for row in config.runs) and not capability()['supported']:
        raise HTTPException(422,capability()['error'])
    payload=new_experiment(config);insert(session,payload)
    background.add_task(execute,request.app.state.engine,config,snapshot,payload)
    return {'experiment_id':payload['experiment_id'],'status':'queued'}


@router.get('')
def list_experiments(session:Session=Depends(get_session)):
    return [row[0] for row in session.execute(text('SELECT payload FROM experiment_registry ORDER BY created_at DESC,experiment_id LIMIT 50'))]


@router.get('/{experiment_id}')
def get(experiment_id:UUID,session:Session=Depends(get_session)):
    value=session.execute(text('SELECT payload FROM experiment_registry WHERE experiment_id=:id'),{'id':str(experiment_id)}).scalar_one_or_none()
    if value is None:raise HTTPException(404,'Experiment not found')
    return value


@router.get('/{experiment_id}/download')
def download(experiment_id:UUID,session:Session=Depends(get_session)):
    return Response(json.dumps(get(experiment_id,session),indent=2),media_type='application/json',headers={'Content-Disposition':f'attachment; filename=experiment-{experiment_id}.json'})


def import_experiment(session,path):
    path=Path(path);payload=json.loads(path.read_text());UUID(payload['experiment_id']);config=MatrixConfig.model_validate(payload['config'])
    if payload['status'] not in {'completed','partial'}:raise ValueError('Import a finished experiment')
    if len(payload['runs'])!=len(config.runs):raise ValueError('Experiment has missing child runs')
    children=[]
    for index,row in enumerate(payload['runs']):
        child_id=str(UUID(row['run_id']));child=json.loads((path.parent/'runs'/child_id/'run.json').read_text())
        if child['run_id']!=child_id or child['experiment_id']!=payload['experiment_id'] or SFTConfig.model_validate(child['config'])!=config.runs[index]:
            raise ValueError('Child identity or configuration differs from registry matrix')
        if summary(child)!=row:raise ValueError('Child summary differs from full evidence')
        children.append(child)
    if compare_matrix(children)!=payload['comparison']:raise ValueError('Comparison differs from child evidence')
    existing=session.execute(text('SELECT payload FROM experiment_registry WHERE experiment_id=:id'),{'id':payload['experiment_id']}).scalar_one_or_none()
    if existing is not None and existing!=payload:raise ValueError('Experiment ID contains different evidence')
    from app.analysis import save_failure
    for child in children:
        import_run(session,child)
        if child.get('benchmark'):save_failure(session,child['benchmark'])
    if payload.get('baseline'):save_failure(session,payload['baseline'])
    if existing is None:insert(session,payload)
    return payload


def recover_interrupted(engine):
    with engine.begin() as connection:
        connection.execute(text("UPDATE experiment_registry SET payload=payload || CAST(:patch AS jsonb),updated_at=now() WHERE payload->>'status' IN ('queued','running')"),
            {'patch':json.dumps({'status':'interrupted','error':'Server restarted; create a new experiment.','finished_at':now()})})


from recipetriage_ml.training.config import SFTConfig
if __name__=='__main__':
    import argparse
    from app.config import Settings
    from app.db import build_engine
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();engine=build_engine(Settings())
    try:
        with Session(engine) as session:print('Imported',import_experiment(session,args.experiment)['experiment_id'])
    finally:engine.dispose()
