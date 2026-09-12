import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import build_engine, get_session

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env", override=False)
    from app.auth import configured_token
    configured_token()
    app.state.engine = build_engine(Settings())
    try:
        yield
    finally:
        app.state.engine.dispose()


app = FastAPI(title="RecipeTriage AI", version="0.1.0", lifespan=lifespan)
from app.auth import APITokenMiddleware
app.add_middleware(APITokenMiddleware)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(session: Session = Depends(get_session)) -> HealthResponse:
    """Readiness check: API is usable only when PostgreSQL responds."""
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.exception("Database readiness check failed")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return HealthResponse(status="ok", database="ok")


from app.inference import router as inference_router
app.include_router(inference_router)

from app.datasets import router as datasets_router
app.include_router(datasets_router)

from app.benchmarks import router as benchmarks_router
app.include_router(benchmarks_router)

from app.training import router as training_router
app.include_router(training_router)

from app.lora import router as lora_router
app.include_router(lora_router)

from app.experiments import router as experiments_router
app.include_router(experiments_router)
from app.analysis import router as analysis_router
app.include_router(analysis_router)
from app.curation import router as curation_router
app.include_router(curation_router)
from app.alignment import router as alignment_router
app.include_router(alignment_router)
from app.registry import router as registry_router
app.include_router(registry_router)
from app.deployment import router as deployment_router
app.include_router(deployment_router)
from app.playground import router as playground_router
app.include_router(playground_router)
from app.library import router as library_router
app.include_router(library_router)
