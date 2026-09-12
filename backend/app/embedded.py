"""Application services for the free Streamlit profile; no listening API server.

Reuse the existing route handlers, Pydantic contracts and session dependencies
through an in-process ASGI adapter. Only explicitly listed operations exist in
this profile. Training, external generation and artifact promotion are absent,
including when callers bypass the UI or secrets contain a provider key.
"""
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.routing import APIRoute
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import get_session

COMPUTE_NOTICE = (
    'This Streamlit profile does not run model inference, paid providers, training, '
    'benchmark workers or model promotion. Run those operations in the existing '
    'local deployment. Saved evidence and human reviews remain available here.'
)

# Exact route templates are deliberately enumerated, rather than allowing every
# GET or every future route in a module. Dependencies run for each request.
READ_ROUTES = {
    '/api/v1/datasets/seed', '/api/v1/datasets/versions',
    '/api/v1/datasets/versions/{version}', '/api/v1/datasets/versions/{version}/download',
    '/api/v1/recipes', '/api/v1/recipes/{recipe_id}/history',
    '/api/v1/curation/candidates', '/api/v1/alignment/options',
    '/api/v1/alignment/pairs', '/api/v1/alignment/pairs/download', '/api/v1/alignment/jobs',
    '/api/v1/benchmarks/manifest', '/api/v1/benchmarks/runs',
    '/api/v1/benchmarks/runs/{run_id}', '/api/v1/benchmarks/runs/{run_id}/download',
    '/api/v1/training/options', '/api/v1/training/runs',
    '/api/v1/training/runs/{run_id}', '/api/v1/training/runs/{run_id}/download',
    '/api/v1/training/lora/architecture', '/api/v1/training/lora/experiments',
    '/api/v1/training/lora/experiments/{experiment_id}',
    '/api/v1/training/lora/experiments/{experiment_id}/download',
    '/api/v1/experiments', '/api/v1/experiments/{experiment_id}',
    '/api/v1/experiments/{experiment_id}/download',
    '/api/v1/analysis/sources', '/api/v1/analysis/failures', '/api/v1/analysis/diagnostics',
    '/api/v1/analysis/collection-priorities/{source_id}',
    '/api/v1/analysis/diagnostics/{run_id}/download',
    '/api/v1/playground/comparisons', '/api/v1/deployment/results',
    '/api/v1/models', '/api/v1/models/history', '/api/v1/models/gate-config',
}
WRITE_ROUTES = {
    ('POST', '/api/v1/datasets/validate'), ('POST', '/api/v1/datasets/preview'),
    ('POST', '/api/v1/datasets/versions'), ('POST', '/api/v1/datasets/chat-preview'),
    ('POST', '/api/v1/recipes/seed'), ('PUT', '/api/v1/recipes/{recipe_id}'),
    ('POST', '/api/v1/recipes/{recipe_id}/review'), ('POST', '/api/v1/recipes/dataset-version'),
    ('POST', '/api/v1/curation/seed-review'), ('POST', '/api/v1/curation/candidates/{candidate_id}/review'),
    ('POST', '/api/v1/curation/build-version'), ('POST', '/api/v1/alignment/pairs/{pair_id}/choice'),
    ('POST', '/api/v1/analysis/failures'), ('POST', '/api/v1/deployment/select'),
    ('POST', '/api/v1/tokens'),
}


def database_settings(database_url):
    # Do not read the local Docker .env, provider keys, or localhost defaults.
    settings = Settings(_env_file=None, DATABASE_URL=database_url)
    url = settings.database_url
    if url.host not in {'localhost', '127.0.0.1', '::1'} and url.query.get('sslmode') not in {'require', 'verify-ca', 'verify-full'}:
        raise ValueError('A remote PostgreSQL database requires sslmode=require or certificate verification in database.url.')
    return settings


def build_runtime(database_url):
    from app.migrate import migrate
    from app.library import seed_library

    settings = database_settings(database_url)
    # Close each connection after use so an idle free database can suspend.
    engine = create_engine(settings.database_url, poolclass=NullPool, hide_parameters=True,
                           connect_args={'connect_timeout': 15, 'options': '-c statement_timeout=15000'})
    try:
        migrate(engine)
        with Session(engine) as session:
            seed_library(session)
        return create_app(engine)
    except Exception:
        engine.dispose()
        raise


def create_app(engine):
    from app import (alignment, analysis, benchmarks, curation, datasets, deployment,
                     experiments, inference, library, lora, playground, registry, training)
    from recipetriage_ml.inference.contracts import Recipe
    from recipetriage_ml.data.pipeline import digest
    from recipetriage_ml.training.experiments import default_matrix

    application = FastAPI(title='RecipeTriage embedded services', docs_url=None, redoc_url=None, openapi_url=None)
    application.state.engine = engine

    @application.get('/health')
    def health(session: Session = Depends(get_session)):
        session.execute(text('SELECT 1'))
        return {'status': 'ok', 'database': 'ok', 'mode': 'embedded'}

    @application.post('/api/v1/recipes/intake', status_code=201)
    def intake(body: library.IntakeRequest, session: Session = Depends(get_session)):
        # Never use normalize()'s Fireworks fallback in the $0 profile.
        if body.kind not in {'text', 'message'}:
            raise HTTPException(422, 'Use the structured recipe form or paste Recipe JSON in this profile.')
        try:
            recipe = Recipe.model_validate_json(body.content)
        except ValueError:
            raise HTTPException(422, 'Provide valid Recipe JSON with title, ingredients and instructions; automatic extraction is unavailable in this profile.') from None
        source_hash = digest(body.content)
        return library.store_intake(recipe, {'kind': body.kind, 'input_sha256': source_hash}, body.content,
                                    {'method': 'validated-json', 'input_sha256': source_hash}, session)

    @application.post('/api/v1/recipes', status_code=201)
    def save(body: library.SaveRequest, request: Request, background: BackgroundTasks,
             session: Session = Depends(get_session)):
        if body.auto_triage:
            raise HTTPException(422, 'Set auto_triage=false. Model inference is unavailable in this profile.')
        return library.save(body, request, background, session)

    @application.get('/api/v1/playground/models')
    def models():
        return {'models': [{'key': key, 'title': title, 'available': False, 'reason': COMPUTE_NOTICE}
                           for key, title in [('fireworks', 'Fireworks'), ('base', 'Local base'),
                                              ('instruct', 'Local instruct'), ('sft', 'SFT'),
                                              ('lora', 'LoRA'), ('qlora', 'QLoRA'), ('dpo', 'DPO')]]}

    @application.get('/api/v1/benchmarks/providers')
    def providers():
        return [{'id': key, 'configured': False, 'reason': COMPUTE_NOTICE} for key in ['hf-base', 'fireworks']]

    @application.get('/api/v1/experiments/options')
    def experiment_options():
        # The usual capability probe loads torch and attempts a real NF4 layer.
        return {'defaults': default_matrix().model_dump(), 'qlora': {'supported': False, 'reason': COMPUTE_NOTICE},
                'tracking': 'Saved evidence only; training runs on the local host.', 'matrix_limit': 8, 'fixed_benchmark': True}

    selected = set()
    for module in [alignment, analysis, benchmarks, curation, datasets, deployment, experiments,
                   inference, library, lora, playground, registry, training]:
        for route in module.router.routes:
            if not isinstance(route, APIRoute):
                continue
            if ('GET' in route.methods and route.path in READ_ROUTES) or any((method, route.path) in WRITE_ROUTES for method in route.methods):
                application.router.routes.append(route)
                selected.update((method, route.path) for method in route.methods)
    missing = ({('GET', path) for path in READ_ROUTES} | WRITE_ROUTES) - selected
    if missing:
        raise RuntimeError('Embedded route contracts changed: ' + str(sorted(missing)))

    @application.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
    def unavailable(path: str):
        raise HTTPException(403, COMPUTE_NOTICE)

    return application
