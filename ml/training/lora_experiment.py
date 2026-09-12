"""Predeclared rank comparison, with one fresh process per rank and no hosted calls."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from recipetriage_ml.data.pipeline import digest
from .config import SFTConfig
from .data import seed_snapshot, verify_snapshot, load_snapshot
from .lora import architecture_report, estimated_parameters
from .runs import Journal, new_training_run, now, run_root


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    training: SFTConfig
    ranks: list[StrictInt] = Field(default=[4, 16], min_length=2, max_length=4)

    @model_validator(mode='after')
    def controlled(self):
        if len(set(self.ranks)) != len(self.ranks) or any(type(rank) is not int or rank not in {4, 8, 16, 32} for rank in self.ranks):
            raise ValueError('Choose at least two distinct ranks from 4, 8, 16, 32')
        if self.training.provider != 'local':
            raise ValueError('Rank experiments use local CPU workers only')
        if self.training.method != 'lora':
            raise ValueError('Use the experiment matrix or QLoRA page for quantization comparisons')
        if self.training.lora_alpha is not None:
            raise ValueError('Rank experiment fixes alpha/r=2; leave lora_alpha null')
        return self


def new_experiment(config):
    return {'experiment_id': str(uuid4()), 'status': 'queued', 'created_at': now(), 'updated_at': now(),
            'finished_at': None, 'config': config.model_dump(), 'runs': [], 'comparison': None, 'error': None,
            'protocol': {'process_isolation': 'fresh Python process per rank, sequential', 'alpha_over_rank': 2,
                         'repetitions_per_rank': 1, 'rank_order': config.ranks,
                         'memory': 'process RSS sampled every 20 ms during trainer.train()',
                         'selection': 'checkpoint by validation loss; no automatic winning rank selected',
                         'dataset_version': config.training.dataset_version}}


def run_summary(run):
    summary = (run.get('benchmark') or {}).get('summary') or {}
    scores = summary.get('labels') or {}
    resources = run.get('training_resources') or {}
    return {'run_id': run['run_id'], 'rank': run['config']['lora_rank'], 'alpha': run['parameters']['lora_alpha'],
            'status': run['status'], 'phase': run['phase'], 'error': run['error'],
            'trainable_parameters': (run.get('model_parameters') or {}).get('trainable'),
            'frozen_parameters': (run.get('model_parameters') or {}).get('frozen'),
            'training_seconds': resources.get('train_wall_seconds'),
            'peak_rss_mib': resources.get('peak_rss_bytes', 0) / 2**20 if resources else None,
            'start_rss_mib': resources.get('start_rss_bytes', 0) / 2**20 if resources else None,
            'peak_rss_increase_mib': resources.get('peak_rss_increase_bytes', 0) / 2**20 if resources else None,
            'micro_f1': scores.get('micro', {}).get('f1'), 'macro_f1': scores.get('macro', {}).get('f1'),
            'exact_match': scores.get('exact_match'), 'json_validity': summary.get('json_validity'),
            'best_validation_loss': run.get('best_validation_loss'), 'optimizer_steps': run.get('optimizer_steps'),
            'history': run.get('history', []), 'baseline_comparison': run.get('comparison')}


def compare_ranks(runs):
    if len(runs) < 2 or any(run['status'] != 'completed' or not (run.get('comparison') or {}).get('comparable') for run in runs):
        return {'comparable': False, 'claim': 'No rank comparison: complete training and benchmark evidence is required for every rank.'}
    reference = runs[0]
    normalized = lambda run: {key: value for key, value in run['config'].items() if key not in {'lora_rank', 'lora_alpha'}}
    for run in runs:
        if (normalized(run) != normalized(reference)
            or run['benchmark']['protocol_sha256'] != reference['benchmark']['protocol_sha256']
            or run['model_parameters']['architecture_sha256'] != reference['model_parameters']['architecture_sha256']
            or run['model_parameters']['target_paths'] != reference['model_parameters']['target_paths']
            or run['model_parameters']['scaling'] != 2
            or run['dataset']['logical_messages_sha256'] != reference['dataset']['logical_messages_sha256']
            or not run.get('training_resources')):
            return {'comparable': False, 'claim': 'No rank comparison: dataset, architecture, targets, settings or measurements differ.'}
    return {'comparable': True, 'rows': [run_summary(run) for run in runs],
            'claim': 'Observed results for the predeclared ranks; no best rank selected. One run per rank and ten provisional benchmark cases cannot establish general superiority.',
            'memory_note': 'Peak RSS is total process memory, not adapter memory. Sampling can miss short peaks. Timing includes epoch evaluation and checkpoint IO.'}


def run_experiment(config, snapshot, experiment=None, root=None, on_update=None, on_run_update=None):
    verify_snapshot(snapshot)
    if snapshot['metadata']['version'] != config.training.dataset_version:
        raise ValueError('Experiment dataset version does not match snapshot')
    experiment = experiment or new_experiment(config)
    if experiment['status'] != 'queued':
        raise ValueError('Create a new experiment; previous evidence cannot be overwritten')
    directory = (Path(root) if root else run_root() / 'lora-experiments') / experiment['experiment_id']
    directory.mkdir(parents=True, exist_ok=True)
    def update(**values):
        experiment.update(values, updated_at=now())
        temp = directory / '.experiment.tmp'
        temp.write_text(json.dumps(experiment, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(directory / 'experiment.json')
        if on_update:
            on_update(experiment)
    report = architecture_report()
    experiment['parameter_estimates'] = {str(rank): estimated_parameters(report, rank, config.training.lora_target_policy) for rank in config.ranks}
    experiment['protocol_sha256'] = digest({'protocol': experiment['protocol'], 'config': config.model_dump()})
    update(status='running')
    completed = []
    process = None
    try:
        for rank in config.ranks:
            settings = config.training.model_copy(update={'lora_rank': rank})
            child = new_training_run(settings)
            child['experiment_id'] = experiment['experiment_id']
            journal = Journal(child, directory / 'runs')
            journal.write()
            # Each worker consumes the exact same checked snapshot; no resplitting.
            (journal.directory / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf-8')
            experiment['runs'].append(run_summary(child)); update()
            last_update = None
            with (journal.directory / 'worker.log').open('wb') as log:
                environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                               'HF_HOME': os.environ.get('HF_HOME', str(Path('.cache/huggingface').resolve()))}
                process = subprocess.Popen([sys.executable, '-m', 'recipetriage_ml.training.lora', '--worker', str(journal.directory / 'run.json')],
                                           stdout=log, stderr=subprocess.STDOUT, env=environment)
                def read_progress():
                    nonlocal child, last_update
                    child = json.loads((journal.directory / 'run.json').read_text(encoding='utf-8'))
                    if child['updated_at'] != last_update:
                        experiment['runs'][-1] = run_summary(child)
                        if on_run_update:
                            on_run_update(child)
                        update()
                        last_update = child['updated_at']
                while process.poll() is None:
                    read_progress()
                    time.sleep(1)
                read_progress()
                if process.returncode != 0 or child['status'] != 'completed':
                    if child['status'] in {'queued', 'preparing', 'baseline', 'training', 'evaluating'}:
                        child.update(status='failed', error=f'Worker exited with code {process.returncode}; inspect worker.log', finished_at=now())
                        Journal(child, directory / 'runs').write()
                        read_progress()
                    raise RuntimeError(f'Rank {rank} worker ended with exit {process.returncode}, status {child["status"]}. Inspect runs/{child["run_id"]}/worker.log; evidence retained.')
            completed.append(child)
        comparison = compare_ranks(completed)
        update(status='completed', comparison=comparison, finished_at=now())
        rows = comparison.get('rows', experiment['runs'])
        columns = ['rank', 'alpha', 'trainable_parameters', 'frozen_parameters', 'training_seconds', 'start_rss_mib',
                   'peak_rss_mib', 'peak_rss_increase_mib', 'micro_f1', 'macro_f1', 'exact_match', 'json_validity', 'run_id']
        with (directory / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    except BaseException as exc:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        update(status='failed', error=str(exc), finished_at=now())
        raise
    return experiment


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='ExperimentConfig JSON; defaults to ranks 4 and 16')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    snapshot = load_snapshot(args.dataset) if args.dataset else seed_snapshot()
    config = ExperimentConfig.model_validate_json(args.config.read_text()) if args.config else ExperimentConfig(training=SFTConfig(dataset_version=snapshot['metadata']['version']))
    result = run_experiment(config, snapshot, root=args.output)
    print(json.dumps({'experiment_id': result['experiment_id'], 'status': result['status'], 'rows': result['runs']}, indent=2))


if __name__ == '__main__':
    main()
