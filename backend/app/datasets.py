import io
import json
from importlib.resources import files
from typing import Literal
from zipfile import ZipFile, ZIP_DEFLATED
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from recipetriage_ml.data.pipeline import as_drafts, build_dataset, chat_messages, load_raw, render_chat, validate_clean
from recipetriage_ml.data.schemas import LABELS, POLICY, Recipe, TrainingExample
from app.db import get_session

router = APIRouter(prefix="/api/v1/datasets", tags=["Data Lab"])


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw: str = Field(max_length=2_000_000)
    format: Literal["json", "jsonl"] = "json"
    seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)
    ratios: list[float] = Field(default=[0.7,0.15,0.15], min_length=3, max_length=3)


def parse(request):
    try:
        return as_drafts(load_raw(request.raw,request.format))
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc


@router.get("/seed")
def seed():
    return {"examples":json.loads(files("recipetriage_ml").joinpath("data/seed.json").read_text()),
            "labels":LABELS,"policy":POLICY,"schema":TrainingExample.model_json_schema(),"recipe_schema":Recipe.model_json_schema()}


@router.post("/validate")
def validate(request: BuildRequest):
    drafts = parse(request)
    examples, issues, duplicates = validate_clean(drafts)
    return {"drafts":drafts,"examples":[x.model_dump() for x in examples],"issues":issues,"duplicates":duplicates}


@router.post("/preview")
def preview(request: BuildRequest):
    try:
        records=parse(request)
        if any(isinstance(x,dict) and (x.get('provenance') or isinstance(x.get('recipe'),dict) and x['recipe'].get('source_type')=='synthetic') for x in records):
            raise ValueError('Create synthetic dataset versions through the human review queue; generic imports cannot bypass approval')
        return build_dataset(records,request.seed,request.ratios)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc


@router.post("/versions")
def save_version(request: BuildRequest, session: Session = Depends(get_session)):
    payload = preview(request)
    version = payload['metadata']['version']
    session.execute(text("INSERT INTO dataset_versions(version,payload) VALUES (:version,CAST(:payload AS jsonb)) ON CONFLICT(version) DO NOTHING"),
                    {"version":version,"payload":json.dumps(payload)})
    session.commit()
    return read_version(version,session)


@router.get("/versions")
def versions(session: Session = Depends(get_session)):
    return [row[0] for row in session.execute(text("SELECT payload->'metadata' FROM dataset_versions ORDER BY created_at DESC, version"))]


@router.get("/versions/{version}")
def read_version(version: str, session: Session = Depends(get_session)):
    payload = session.execute(text("SELECT payload FROM dataset_versions WHERE version=:version"),{"version":version}).scalar_one_or_none()
    if payload is None:
        raise HTTPException(404,"Dataset version not found")
    return payload


@router.get("/versions/{version}/download")
def download(version: str, session: Session = Depends(get_session)):
    payload = read_version(version,session)
    output = io.BytesIO()
    with ZipFile(output,'w',ZIP_DEFLATED) as archive:
        for name,content in payload['files'].items():
            archive.writestr(name,content)
        archive.writestr('metadata.json',json.dumps(payload['metadata'],indent=2))
    return Response(output.getvalue(),media_type='application/zip',headers={"Content-Disposition":f"attachment; filename=recipetriage-{payload['metadata']['version']}.zip"})


class ChatRequest(BaseModel):
    example: TrainingExample
    render_model: bool = False


@router.post("/chat-preview")
def chat_preview(request: ChatRequest):
    result = {"messages":chat_messages(request.example),"rendered_text":None}
    if request.render_model:
        from app.inference import call
        from recipetriage_ml.inference.hf_provider import get_tokenizer, MODEL_ID, MODEL_REVISION
        def render():
            tokenizer = get_tokenizer()
            result.update(rendered_text=render_chat(request.example,tokenizer),model=MODEL_ID,revision=MODEL_REVISION)
            return result
        return call(render)
    return result
