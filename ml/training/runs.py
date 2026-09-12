"""Atomic run journals and measured comparisons, shared by API and command line."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4, UUID
from recipetriage_ml.evaluation.benchmark import RunConfig, new_run, run_benchmark

ACTIVE = {'queued', 'preparing', 'baseline', 'training', 'evaluating', 'submitting', 'monitoring'}


def now():
    return datetime.now(timezone.utc).isoformat()


def run_root():
    return Path(os.getenv('TRAINING_RUNS_DIR', 'ml/training/runs')).resolve()


def new_training_run(config):
    return {'run_id': str(uuid4()), 'status': 'queued', 'phase': 'queued', 'created_at': now(),
            'updated_at': now(), 'finished_at': None, 'config': config.model_dump(), 'parameters': config.resolved(),
            'dataset': None, 'history': [], 'checkpoints': [], 'artifacts': {}, 'baseline': None,
            'benchmark': None, 'comparison': None, 'managed': {}, 'error': None,
            'warnings': ['Learning run with a tiny provisional dataset; no production-quality claim.',
                         'Choose parameters using validation data. The fixed benchmark is held out.']}


class Journal:
    def __init__(self, run, root=None, on_update=None):
        self.run = run
        self.directory = (Path(root) if root else run_root()) / str(UUID(run['run_id']))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.on_update = on_update

    def write(self, **patch):
        self.run.update(patch, updated_at=now())
        text = json.dumps(self.run, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        temp = self.directory / '.run.tmp'
        temp.write_text(text)
        temp.replace(self.directory / 'run.json')
        if self.on_update:
            self.on_update(self.run)

    def artifact(self, name, payload):
        (self.directory / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        self.run['artifacts'][name] = name


def evaluate(journal, provider, descriptor, key):
    config = RunConfig(provider='hf-base' if journal.run['config']['provider'] == 'local' else 'fireworks',
                       temperature=0, max_new_tokens=128, reasoning='disabled')
    evidence = new_run(config)
    # Provider identifies the runtime transport; this descriptor identifies the actual checkpoint.
    evidence['model_config'] = descriptor
    evidence['training_run_id'] = journal.run['run_id']
    evidence['purpose'] = key
    journal.run[key] = evidence
    def update(result):
        journal.run[key] = result
        journal.write()
    result = run_benchmark(config, run=evidence, provider_instance=provider, on_update=update)
    journal.artifact(f'{key}.json', result)
    journal.write()
    return result


def compare(baseline, trained):
    if (not baseline or not trained or baseline['status'] != 'completed' or trained['status'] != 'completed'
        or baseline['protocol_sha256'] != trained['protocol_sha256']
        or baseline['model_config']['prompt_format'] != trained['model_config']['prompt_format']
        or trained['model_config'].get('base_model') != baseline['model_config']['model']
        or (trained['config']['provider'] == 'hf-base' and trained['model_config'].get('base_revision') != baseline['model_config']['revision'])):
        return {'comparable': False, 'claim': 'No improvement claim: complete benchmark runs with the same protocol and formatter are required.'}
    before, after = baseline['summary'], trained['summary']
    if (not before.get('scorable') or not after.get('scorable')
        or not before.get('returned_responses') or not after.get('returned_responses')):
        return {'comparable': False, 'claim': 'No improvement claim: aggregate metrics or provider responses unavailable.'}
    def values(summary):
        scores = summary['labels']
        return {'micro_f1': scores['micro']['f1'], 'macro_f1': scores['macro']['f1'],
                'exact_match': scores['exact_match'], 'json_validity': summary['json_validity']}
    left, right = values(before), values(after)
    delta = {name: right[name] - left[name] for name in left}
    improved = delta['micro_f1'] > 0
    return {'comparable': True, 'baseline_run_id': baseline['run_id'], 'trained_run_id': trained['run_id'],
            'before': left, 'after': right, 'delta': delta,
            'claim': ('Micro-F1 increased on these ten fixed cases.' if improved else 'No measured Micro-F1 improvement on these ten fixed cases.')
                     + ' This is descriptive evidence only, not proof of generalization; annotations are provisional.'}
