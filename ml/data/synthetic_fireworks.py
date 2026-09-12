"""Bounded Fireworks proposals. This module only writes review journals, never train.jsonl."""
import json
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field
from recipetriage_ml.inference.contracts import Recipe, Label, ProviderError
from recipetriage_ml.inference.fireworks_provider import FireworksProvider, DEFAULT_MODEL
from recipetriage_ml.data.pipeline import digest, _json
from recipetriage_ml.data.schemas import POLICY, TrainingExample
from recipetriage_ml.training.runs import now

PROMPT_VERSION = 'synthetic-fireworks-v1'
BRIEFS = [
    ('misleading-title', 'A savory stuffed vegetable with substantial active preparation.'),
    ('ambiguous', 'A grain-and-herb skillet with unknown timing and contradictory effort notes.'),
    ('misleading-title', 'A bean dish with a long passive rest; avoid claiming a weekend project.'),
    ('ambiguous', 'A lentil and vegetable dish with missing timing and no decisive supported label.'),
    ('misleading-title', 'A dessert with extended cooling and an explicitly required appliance.'),
    ('misleading-title', 'A portioned meal-prep bake with more than 30 minutes of elapsed time.'),
]


class Draft(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    recipe: Recipe
    labels: list[Label] = Field(min_length=1, max_length=7)
    rationale: str = Field(min_length=1, max_length=2000)
    category: Literal['misleading-title', 'ambiguous']
    challenge_notes: str = Field(min_length=1, max_length=2000)


def messages(focus, brief):
    template = files('recipetriage_ml').joinpath('data/prompts/synthetic-v1.txt').read_text()
    prompt = template.format(policy=json.dumps(POLICY), focus=focus, brief=brief,
                             schema=json.dumps(Draft.model_json_schema()))
    return [{'role': 'system', 'content': 'Generate fictional recipe proposals as JSON for human review. Never approve your own output.'},
            {'role': 'user', 'content': prompt}]


def forbidden_recipes():
    from recipetriage_ml.evaluation.benchmark import load_benchmark
    from recipetriage_ml.evaluation.shortcut_tests import suite
    return [x.recipe.inference_recipe().model_dump() for x in load_benchmark().cases] + [g['recipe'] for g in suite()['groups']]


def body_key(recipe):
    return digest({'ingredients': sorted(x.casefold().strip() for x in recipe['ingredients']),
                   'instructions': [x.casefold().strip() for x in recipe['instructions']]})


def quality_checks(raw, *, focus=None, others=(), finish_reason='stop'):
    errors, warnings = [], []
    if finish_reason != 'stop': errors.append('Generation did not finish normally; edit and recheck or reject.')
    try:
        draft = Draft.model_validate(raw)
        recipe = draft.recipe.model_dump()
        # Reuse all existing multi-label/time/pantry constraints without claiming review.
        TrainingExample.model_validate({'recipe': {**recipe, 'id': 'quality-probe', 'source_type': 'personal',
            'source_uri': 'personal:quality-probe', 'source_notes': 'Temporary validation only'},
            'labels': draft.labels, 'rationale': draft.rationale})
        if focus and draft.category != focus: errors.append('Requested and returned categories differ.')
        if draft.category == 'misleading-title' and (not re.match(r'^(Quick|Easy|Simple)\b', recipe['title']) or recipe['time_minutes'] is None or recipe['time_minutes'] <= 30):
            errors.append('Misleading-title proposal requires a declared adjective and known time >30 minutes.')
        if draft.category == 'ambiguous' and draft.labels != ['unclear']:
            errors.append('This ambiguous template requires unclear alone.')
        tokens = set(re.findall(r'\w+', ' '.join(recipe['ingredients'] + recipe['instructions']).casefold()))
        for other in list(others) + forbidden_recipes():
            if body_key(recipe) == body_key(other): errors.append('Duplicate recipe body or held-out overlap.'); break
            other_tokens = set(re.findall(r'\w+', ' '.join(other['ingredients'] + other['instructions']).casefold()))
            if len(tokens & other_tokens) / max(1, len(tokens | other_tokens)) > .72:
                errors.append('High lexical similarity to an existing or held-out recipe; use a distinct source.'); break
        if len(' '.join(recipe['ingredients'] + recipe['instructions']).split()) > 180:
            errors.append('Recipe is too long for the learning budget; shorten and recheck.')
        warnings += ['Human must check plausibility, label evidence and semantic overlap; automated checks cannot establish correctness.',
                     'Synthetic proposal; no published recipe source and no human approval yet.']
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return {'passed': not errors, 'errors': errors, 'warnings': warnings, 'version': 'synthetic-quality-v1'}


def generate_batch(count=6, provider=None, on_update=None, existing=()):
    if isinstance(count, bool) or not 1 <= count <= 6: raise ValueError('Generate 1–6 proposals per batch')
    provider = provider or FireworksProvider(model=os.getenv('FIREWORKS_SYNTHETIC_MODEL') or DEFAULT_MODEL, reasoning_effort='none')
    batch = {'generation_id': str(uuid4()), 'created_at': now(), 'status': 'running', 'items': [],
             'provider': 'fireworks', 'requested_model': provider.model, 'prompt_version': PROMPT_VERSION,
             'temperature': .7, 'max_new_tokens': 800, 'automatic_retries': 0, 'training_exported': False}
    seen = list(existing)
    for focus, brief in BRIEFS[:count]:
        prompt = messages(focus, brief)
        item = {'candidate_id': str(uuid4()), 'revision': 1, 'kind': 'synthetic', 'status': 'pending',
                'created_at': now(), 'generation_id': batch['generation_id'], 'category': focus,
                'prompt_version': PROMPT_VERSION, 'messages': prompt, 'prompt_sha256': digest(prompt),
                'model': provider.model, 'original': None, 'draft': None, 'raw_output': None,
                'finish_reason': None, 'quality': None, 'review': None, 'history': [], 'error': None}
        try:
            output = provider.generate(prompt, .7, 800)
            generation=output.model_dump()
            item['model_revision']=generation.pop('revision',None)
            item.update(generation)  # recipe review revision is distinct from provider model revision
            item['original'] = item['draft'] = _json(output.raw_output)
            item['quality'] = quality_checks(item['draft'], focus=focus, others=seen, finish_reason=output.finish_reason)
            if item['quality']['passed']: seen.append(item['draft']['recipe'])
        except (ValueError, ProviderError) as exc:
            item['error'] = str(exc)
            item['quality'] = {'passed': False, 'errors': [str(exc)], 'warnings': [], 'version': 'synthetic-quality-v1'}
        batch['items'].append(item)
        if on_update: on_update(batch)
        # Authentication/model-access failures are not repaired or retried behind the user's back.
        if item['raw_output'] is None: break
    batch['status'] = 'completed' if len(batch['items']) == count and all(x['raw_output'] is not None for x in batch['items']) else 'partial'
    if on_update: on_update(batch)
    return batch


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=6); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output path; evidence is immutable')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(value):
        temp = args.output.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(args.output)
    result = generate_batch(args.count, on_update=save)
    print(json.dumps({'status': result['status'], 'pending': len(result['items']),
                      'quality_passed': sum(x['quality']['passed'] for x in result['items']), 'approved': 0}))
    if result['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__': main()
