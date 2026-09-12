"""Explicit human preference pairs; automatic scores never select a winner."""
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator
from recipetriage_ml.inference.contracts import Recipe
from recipetriage_ml.inference.prompts import build_messages
from recipetriage_ml.inference.fireworks_provider import FireworksProvider, DEFAULT_MODEL
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.training.data import seed_snapshot, verify_snapshot
from recipetriage_ml.training.runs import now
from recipetriage_ml.evaluation.metrics import inspect_response
from recipetriage_ml.evaluation.base_provider import render_prompt

REFERENCE_RUN='041818c9-7e00-4802-b53d-f7d567616e2b'


def ml_directory():
    # Docker keeps full experiment artifacts beside the installed package.
    # Community Cloud may deny traversal of /app altogether; its editable
    # package already points at the repository, so use that portable fallback.
    docker_root = Path('/app/ml')
    try:
        if docker_root.is_dir():
            return docker_root
    except PermissionError:
        pass
    return Path(__file__).resolve().parents[1]


def reference_directory():
    default=Path('/training/references/qlora-v1') if os.getenv('TRAINING_RUNS_DIR')=='/training' else ml_directory()/'training/results/experiments-v1/runs'/REFERENCE_RUN
    return Path(os.getenv('REFERENCE_ADAPTER_DIR',str(default))).resolve()


class Candidate(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    provider:str
    model:str
    revision:str|None=None
    raw_output:str
    finish_reason:str
    input_tokens:int|None=None
    output_tokens:int|None=None
    quality:dict


class HumanChoice(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    choice:Literal['a','b','tie','neither']
    reviewer:str=Field(min_length=1,max_length=100,pattern=r'.*\S.*')
    notes:str=Field(default='',max_length=4000)
    decided_at:str
    pair_sha256:str
    decision_id:str


class PreferencePair(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    pair_id:str
    recipe_id:str
    recipe:Recipe
    dataset_version:str
    candidates:dict[str,Candidate]
    messages:list[dict[str,str]]
    pair_sha256:str
    created_at:str
    status:Literal['pending','chosen','tie','neither']='pending'
    decision:HumanChoice|None=None

    @model_validator(mode='after')
    def valid(self):
        if set(self.candidates)!={'a','b'}:raise ValueError('A pair needs candidates a and b')
        if self.messages!=build_messages(self.recipe):raise ValueError('Prompt differs from recipe')
        expected=digest({'recipe':self.recipe.model_dump(),'candidates':{k:v.model_dump() for k,v in self.candidates.items()},'messages':self.messages})
        if expected!=self.pair_sha256:raise ValueError('Preference evidence changed')
        if self.decision:
            if self.decision.pair_sha256!=self.pair_sha256:raise ValueError('Choice belongs to different candidate content')
            if self.status!=('chosen' if self.decision.choice in {'a','b'} else self.decision.choice):raise ValueError('Choice/status mismatch')
        elif self.status!='pending':raise ValueError('Only an explicit human choice may set preference status')
        return self


def generate_pairs(on_update=None, local=None, hosted=None, count=5):
    from recipetriage_ml.training.sft import load_adapter
    snapshot=seed_snapshot();recipes=verify_snapshot(snapshot)['train'][:count]
    local=local or load_adapter(reference_directory())
    hosted=hosted or FireworksProvider(model=os.getenv('FIREWORKS_SYNTHETIC_MODEL') or DEFAULT_MODEL,reasoning_effort='none')
    result=[]
    for example in recipes:
        recipe=example.recipe.inference_recipe();messages=build_messages(recipe);pair_id=str(uuid4());outputs=[]
        for name,provider in [('fireworks',hosted),('current-qlora',local)]:
            generation=provider.generate(messages,0,256)
            outputs.append(Candidate(provider=name,**generation.model_dump(),quality=inspect_response(generation.raw_output,generation.finish_reason)).model_dump())
        # Randomized presentation independent of model quality. There is no suggested winner.
        if int(pair_id.replace('-',''),16)%2:outputs.reverse()
        candidates=dict(zip(['a','b'],outputs));key=digest({'recipe':recipe.model_dump(),'candidates':candidates,'messages':messages})
        pair=PreferencePair(pair_id=pair_id,recipe_id=example.recipe.id,recipe=recipe,dataset_version=snapshot['metadata']['version'],
                            candidates=candidates,messages=messages,pair_sha256=key,created_at=now()).model_dump()
        result.append(pair)
        if on_update:on_update(result)
    return result


def preference_dataset(pairs, tokenizer=None, max_length=1024):
    approved=[PreferencePair.model_validate(x) for x in pairs]
    approved=[x for x in approved if x.status=='chosen' and x.decision and x.decision.choice in {'a','b'}]
    # Only current train recipes are eligible. Benchmark and original holdouts never enter DPO.
    snapshot=seed_snapshot();allowed={x.recipe.id:x for x in verify_snapshot(snapshot)['train']}
    grouped={}
    for pair in approved:
        if pair.recipe_id not in allowed or pair.recipe!=allowed[pair.recipe_id].recipe.inference_recipe() or pair.dataset_version!=snapshot['metadata']['version']:
            raise ValueError('Preference recipe is not in the fixed eligible training split')
        chosen=pair.candidates[pair.decision.choice];rejected=pair.candidates['b' if pair.decision.choice=='a' else 'a']
        if chosen.raw_output==rejected.raw_output or not chosen.raw_output.strip() or not rejected.raw_output.strip():
            raise ValueError('Identical or empty responses cannot form an informative DPO pair')
        if chosen.finish_reason!='stop' or rejected.finish_reason!='stop':
            raise ValueError('Incomplete candidate generations must be regenerated before DPO')
        row={'prompt':render_prompt(pair.messages),'chosen':chosen.raw_output,'rejected':rejected.raw_output,'pair_id':pair.pair_id}
        if tokenizer:
            for key in ['chosen','rejected']:
                # TRL tokenizes explicit prompt and completion separately; never silently truncate.
                if len(tokenizer.encode(row['prompt'],add_special_tokens=False))+len(tokenizer.encode(row[key],add_special_tokens=False))+2>max_length:
                    raise ValueError('Preference pair exceeds sequence budget; no truncation is permitted')
        grouped.setdefault(digest(pair.recipe.model_dump(exclude={'title'})),[]).append(row)
    if len(grouped)<3:raise ValueError('Choose a or b explicitly on at least three independent, complete recipe pairs before DPO')
    keys=sorted(grouped,key=lambda k:digest([42,k]));validation={keys[-1]}
    return {'train':[r for k in keys if k not in validation for r in grouped[k]],
            'validation':[r for k in keys if k in validation for r in grouped[k]],
            'version':'preferences-'+digest([x.model_dump() for x in sorted(approved,key=lambda x:x.pair_id)]),
            'human_decisions':[x.decision.model_dump() for x in approved], 'group_count':len(grouped)}
