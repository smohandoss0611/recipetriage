"""Persistent journals for optional learning jobs; all share the existing single-worker gate."""
import json
import logging
from uuid import uuid4
from sqlalchemy import text
from recipetriage_ml.training.runs import now


def create(session, kind, config):
    from app.training import lock_queue
    lock_queue(session)
    job = {'run_id':str(uuid4()), 'kind':kind, 'status':'queued', 'config':config,
           'created_at':now(), 'updated_at':now(), 'result':None, 'error':None}
    session.execute(text('INSERT INTO learning_jobs(run_id,kind,payload) VALUES (:id,:kind,CAST(:payload AS jsonb))'),
                    {'id':job['run_id'], 'kind':kind, 'payload':json.dumps(job)})
    session.commit()
    return job


def persist(engine, job):
    job['updated_at'] = now()
    with engine.begin() as connection:
        changed = connection.execute(text("UPDATE learning_jobs SET payload=CAST(:payload AS jsonb) WHERE run_id=:id AND payload->>'status' IN ('queued','running')"),
                                     {'id':job['run_id'], 'payload':json.dumps(job,allow_nan=False)})
        if changed.rowcount != 1: raise ValueError('Finished job evidence is immutable')


def execute(engine, job, operation):
    try:
        job['status']='running'; persist(engine,job)
        result=operation()
        job.update(status='completed',result=result);persist(engine,job)
    except Exception as exc:
        logging.getLogger(__name__).exception('Learning job failed')
        job.update(status='failed',error=str(exc) if isinstance(exc,ValueError) else f'{type(exc).__name__}: inspect worker logs')
        persist(engine,job)


def recover(engine):
    with engine.begin() as connection:
        connection.execute(text("UPDATE learning_jobs SET payload=payload || CAST(:patch AS jsonb) WHERE payload->>'status' IN ('queued','running')"),
                           {'patch':json.dumps({'status':'interrupted','error':'Server restarted; evidence retained. Start a new job.'})})
