import logging
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from recipetriage_ml.inference.contracts import ProviderError, Recipe
from recipetriage_ml.inference.hf_provider import inspect_tokens
from recipetriage_ml.inference.prompts import SYSTEM_PROMPT
from recipetriage_ml.inference.service import triage

router = APIRouter(prefix="/api/v1", tags=["inference"])
logger = logging.getLogger(__name__)


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: Recipe
    provider: Literal["hf", "fireworks", "production"] = "hf"
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    max_new_tokens: int = Field(default=128, ge=1, le=256, strict=True)


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=10000)
    chat_template: bool = False


def call(operation):
    try:
        return operation()
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (ImportError, OSError, RuntimeError) as exc:
        logger.exception("Local inference/tokenizer unavailable")
        raise HTTPException(503, "Local model/tokenizer unavailable; inspect backend logs and install/download the HF dependencies") from exc


@router.post("/triage")
def predict(request: TriageRequest, http_request: Request):
    if request.provider=='production':
        from sqlalchemy.orm import Session
        from app.serving import provider
        def invoke():
            with Session(http_request.app.state.engine) as session:
                selected=provider(session)
            return triage(request.recipe,'production',request.temperature,request.max_new_tokens,provider_instance=selected)
        return call(invoke)
    return call(lambda: triage(request.recipe, request.provider, request.temperature, request.max_new_tokens))


@router.post("/tokens")
def tokens(request: TokenRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request.text}] if request.chat_template else None
    return call(lambda: inspect_tokens(request.text, messages))


@router.get("/learning-example")
def learning_example():
    import json
    from importlib.resources import files
    return json.loads(files("recipetriage_ml").joinpath("examples/ravioli.json").read_text())
