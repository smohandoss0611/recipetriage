"""Fixed benchmark contract, source-overlap audit, and checkpointed evidence files."""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version as package_version, PackageNotFoundError
from importlib.resources import files
import json
import logging
import os
from pathlib import Path
import platform
import time
from typing import Literal
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator
from recipetriage_ml.data.schemas import TrainingExample, LABELS, POLICY
from recipetriage_ml.data.pipeline import canonical, digest, fingerprint, source_key
from recipetriage_ml.inference.contracts import ProviderError
from recipetriage_ml.inference.fireworks_provider import FireworksProvider, DEFAULT_MODEL
from recipetriage_ml.inference.prompts import build_messages, SYSTEM_PROMPT, PROMPT_VERSION
from .base_provider import BaseProvider, MODEL_ID, MODEL_REVISION, FORMAT_VERSION, render_prompt
from .metrics import inspect_response, summarize, METRICS_VERSION

CATEGORIES = ['normal', 'misleading-title', 'special-equipment', 'multi-label', 'ambiguous', 'hard']
logger = logging.getLogger(__name__)


class BenchmarkCase(TrainingExample):
    category: Literal['normal', 'misleading-title', 'special-equipment', 'multi-label', 'ambiguous', 'hard']
    challenge: str = Field(min_length=1, max_length=2000)


class Benchmark(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Literal['RecipeTriage-Bench-v1']
    description: str
    annotation_status: Literal['provisional-unreviewed']
    cases: list[BenchmarkCase] = Field(min_length=6, max_length=100)

    @model_validator(mode='after')
    def unique_cases(self):
        if set(case.category for case in self.cases) != set(CATEGORIES):
            raise ValueError('Every benchmark category must be represented')
        for values in [[case.recipe.id for case in self.cases], [fingerprint(case) for case in self.cases]]:
            if len(values) != len(set(values)):
                raise ValueError('Benchmark IDs and normalized recipe bodies must be unique')
        if set(label for case in self.cases for label in case.labels) != set(LABELS):
            raise ValueError('The answer key must cover the complete label vocabulary')
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: Literal['hf-base', 'fireworks']
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    max_new_tokens: int = Field(default=128, ge=1, le=256, strict=True)
    reasoning: Literal['provider-default', 'disabled'] = 'provider-default'


def config_payload(config):
    # Preserve the original default-protocol artifacts byte for byte.
    result = config.model_dump(exclude={'reasoning'})
    if config.reasoning != 'provider-default':
        result['reasoning'] = config.reasoning
    return result


def load_benchmark():
    benchmark = Benchmark.model_validate_json(files('recipetriage_ml').joinpath('evaluation/bench_v1.json').read_text())
    expected = files('recipetriage_ml').joinpath('evaluation/bench_v1.sha256').read_text().strip()
    if digest(benchmark.model_dump()) != expected:
        raise ValueError('Frozen benchmark hash mismatch; create a new benchmark version for answer-key changes')
    return benchmark


def manifest(benchmark=None):
    benchmark = benchmark or load_benchmark()
    seed = [TrainingExample.model_validate(row) for row in json.loads(files('recipetriage_ml').joinpath('data/seed.json').read_text())]
    overlap = sorted(case.recipe.id for case in benchmark.cases if any(
        fingerprint(case) == fingerprint(row) or source_key(case.recipe) == source_key(row.recipe) for row in seed))
    payload = benchmark.model_dump()
    return {**payload, 'benchmark_sha256': digest(payload), 'prompt_version': PROMPT_VERSION,
            'prompt_sha256': digest(SYSTEM_PROMPT), 'metrics_version': METRICS_VERSION,
            'policy_sha256': digest(POLICY), 'category_counts': dict(Counter(case.category for case in benchmark.cases)),
            'label_support': {label: sum(label in case.labels for case in benchmark.cases) for label in LABELS},
            'contamination_audit': {'dataset_v1_source_or_body_overlap': overlap, 'pretraining_overlap': 'unknown',
                                   'paraphrase_detection': 'not performed', 'training_allowed': False}}


def descriptor(provider):
    if provider == 'hf-base':
        return {'model': MODEL_ID, 'revision': MODEL_REVISION, 'training_stage': 'pretrained-base', 'prompt_format': FORMAT_VERSION}
    return {'model': os.getenv('FIREWORKS_MODEL') or DEFAULT_MODEL, 'revision': None,
            'training_stage': 'hosted-model; see provider model card', 'prompt_format': 'provider-chat-template'}


def new_run(config, benchmark=None):
    spec = manifest(benchmark)
    runtime = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['torch', 'transformers', 'httpx']:
        try:
            runtime[name] = package_version(name)
        except PackageNotFoundError:
            runtime[name] = None
    protocol = {key: spec[key] for key in ['benchmark_sha256', 'prompt_sha256', 'metrics_version', 'policy_sha256']}
    protocol.update(temperature=config.temperature, max_new_tokens=config.max_new_tokens, ordering='file-order', repetitions=1)
    if config.reasoning != 'provider-default':
        protocol['reasoning'] = config.reasoning
    return {'run_id': str(uuid4()), 'status': 'queued', 'created_at': datetime.now(timezone.utc).isoformat(),
            'finished_at': None, 'benchmark': spec, 'config': config_payload(config), 'model_config': descriptor(config.provider),
            'protocol_sha256': digest(protocol), 'protocol': protocol, 'runtime': runtime, 'rows': [],
            'setup_latency_ms': None, 'setup': None, 'error': None, 'summary': summarize((benchmark or load_benchmark()).cases, []),
            'warnings': ['Provisional human-unreviewed answer key; teaching benchmark only.',
                         'Ten cases and one repetition do not establish population quality or stable tail latency.',
                         'Provider-native formatting, model sizes, and hardware differ; this is a system comparison.',
                         'JSON/schema-invalid or truncated answers receive an empty label set and fail exact match.',
                         'Latency includes provider call and output transfer; local model setup is measured separately.']}


def run_benchmark(config, on_update=None, run=None, provider_instance=None):
    benchmark = load_benchmark()
    run = run or new_run(config, benchmark)
    if run['benchmark']['benchmark_sha256'] != manifest(benchmark)['benchmark_sha256'] or run['config'] != config_payload(config) or run['rows']:
        raise ValueError('Run must be a fresh queued record matching the fixed benchmark and configuration')
    def update():
        run['summary'] = summarize(benchmark.cases, run['rows'])
        if on_update:
            on_update(run)
    run['status'] = 'running'
    update()
    started = time.perf_counter()
    try:
        if provider_instance is None and config.provider == 'fireworks' and not os.getenv('FIREWORKS_API_KEY'):
            raise ProviderError('FIREWORKS_API_KEY is not configured; no hosted calls made')
        provider = provider_instance or (BaseProvider() if config.provider == 'hf-base' else FireworksProvider(reasoning_effort='none' if config.reasoning == 'disabled' else None))
        run['setup'] = provider.prepare() if hasattr(provider, 'prepare') else {'warmup_request': False}
    except Exception as exc:
        logger.exception('Benchmark provider setup failed')
        run.update(status='blocked', error=str(exc) if isinstance(exc, ProviderError) else f'{type(exc).__name__}: provider setup failed; inspect runner logs')
    run['setup_latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
    update()
    fatal_error = run['error']
    for case in benchmark.cases:
        messages = build_messages(case.recipe.inference_recipe())
        row = {'case_id': case.recipe.id, 'category': case.category, 'expected_labels': case.labels,
               'messages': messages, 'messages_sha256': digest(messages), 'attempted': False,
               'rendered_prompt': render_prompt(messages) if config.provider == 'hf-base' else None,
               'raw_output': None, 'finish_reason': None, 'prediction': None, 'json_valid': False,
               'schema_valid': False, 'usable': False, 'error': fatal_error, 'latency_ms': None,
               'model': run['model_config']['model'], 'revision': run['model_config']['revision'],
               'input_tokens': None, 'output_tokens': None}
        if not fatal_error:
            started = time.perf_counter()
            row['attempted'] = True
            try:
                generation = provider.generate(messages, config.temperature, config.max_new_tokens)
                row.update(generation.model_dump())
                row.update(inspect_response(generation.raw_output, generation.finish_reason))
            except Exception as exc:
                logger.exception('Benchmark case failed: %s', case.recipe.id)
                row['error'] = str(exc) if isinstance(exc, (ProviderError, ValueError)) else f'{type(exc).__name__}: inference failed; inspect runner logs'
                if isinstance(exc, ProviderError) and any(f'HTTP {code}' in str(exc) for code in [401, 403, 404, 429]):
                    fatal_error = row['error']
            row['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
        run['rows'].append(row)
        update()
    run['status'] = 'blocked' if fatal_error else 'completed'
    run['error'] = fatal_error
    run['finished_at'] = datetime.now(timezone.utc).isoformat()
    update()
    return run


def save_file(run, directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / f"{run['run_id']}.json"
    temp = target / f".{run['run_id']}.tmp"
    temp.write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(destination)
    return destination
