"""Config-driven experiment matrix, isolated workers and fixed-protocol comparisons."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.evaluation.benchmark import RunConfig, run_benchmark, manifest, descriptor
from .config import SFTConfig
from .data import seed_snapshot, verify_snapshot, load_snapshot
from .runs import Journal, new_training_run, now, run_root, ACTIVE
from .lora_experiment import run_summary


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(default='LoRA learning-rate and rank matrix', min_length=1, max_length=120)
    runs: list[SFTConfig] = Field(min_length=2, max_length=8)
    mlflow: bool = Field(default=True, strict=True)

    @model_validator(mode='after')
    def fixed(self):
        fixed_keys = ['dataset_version', 'epochs', 'sequence_length', 'batch_size', 'gradient_accumulation_steps',
                      'seed', 'lora_alpha', 'lora_dropout', 'lora_target_policy', 'gradient_checkpointing']
        if any(row.provider != 'local' for row in self.runs):
            raise ValueError('This controlled matrix uses local CPU workers only')
        if any(any(getattr(row, key) != getattr(self.runs[0], key) for key in fixed_keys) for row in self.runs):
            raise ValueError('Keep dataset, seed, training budget, target policy and checkpointing fixed across this matrix')
        if self.runs[0].lora_alpha is not None:
            raise ValueError('Leave alpha null: all matrix runs use alpha/r=2')
        signatures = [(row.method, row.learning_rate, row.lora_rank) for row in self.runs]
        if len(signatures) != len(set(signatures)):
            raise ValueError('Duplicate method/learning-rate/rank combination')
        return self


def default_matrix(snapshot=None):
    version = (snapshot or seed_snapshot())['metadata']['version']
    rows = [SFTConfig(dataset_version=version, learning_rate=lr, lora_rank=rank)
            for lr in [0.00005, 0.0002] for rank in [4, 16]]
    rows.append(SFTConfig(dataset_version=version, learning_rate=0.0002, lora_rank=4, method='qlora'))
    return MatrixConfig(name='Fixed v1 · learning rate × rank and NF4 comparison', runs=rows)


def summary(run):
    result = run_summary(run)
    result.update(method=run['config'].get('method', 'lora'), learning_rate=run['config']['learning_rate'],
                  gradient_checkpointing=run['config'].get('gradient_checkpointing', True), tracking=run.get('tracking'),
                  schema_validity=(run.get('benchmark') or {}).get('summary', {}).get('schema_validity'))
    return result


def compare_matrix(runs):
    rows = [summary(run) for run in runs]
    if len(runs) < 2 or any(run['status'] != 'completed' or not (run.get('comparison') or {}).get('comparable') for run in runs):
        return {'comparable': False, 'rows': rows, 'effects': [], 'claim': 'Incomplete evidence: no overall matrix ranking. Inspect failed or blocked runs.'}
    reference = runs[0]
    def control(run):
        return {key: value for key, value in SFTConfig.model_validate(run['config']).model_dump().items()
                if key not in {'method', 'learning_rate', 'lora_rank'}}
    if any(control(run) != control(reference) or run['benchmark']['protocol_sha256'] != reference['benchmark']['protocol_sha256']
           or run['dataset']['logical_messages_sha256'] != reference['dataset']['logical_messages_sha256']
           or run['model_parameters']['target_paths'] != reference['model_parameters']['target_paths']
           or run['model_parameters']['architecture_sha256'] != reference['model_parameters']['architecture_sha256']
           or run['baseline'] != reference['baseline'] for run in runs):
        return {'comparable': False, 'rows': rows, 'effects': [], 'claim': 'Dataset, architecture, baseline or protocol mismatch; comparison rejected.'}
    effects = []
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            changed = [key for key in ['method', 'learning_rate', 'rank'] if left[key] != right[key]]
            if len(changed) != 1: continue
            delta = right['micro_f1'] - left['micro_f1']
            effects.append({'factor': changed[0], 'from_run': left['run_id'], 'to_run': right['run_id'],
                           'from_value': left[changed[0]], 'to_value': right[changed[0]],
                           'micro_f1_delta': delta, 'effect': 'helped on these cases' if delta > 0 else 'hurt on these cases' if delta < 0 else 'no Micro-F1 change'})
    return {'comparable': True, 'rows': rows, 'effects': effects,
            'claim': 'Descriptive comparisons on ten fixed provisional cases, one run per setting. No automatic winner or deployment selection; repeated benchmark use is exploratory.'}


def new_experiment(config):
    return {'experiment_id': str(uuid4()), 'name': config.name, 'status': 'queued', 'config': config.model_dump(),
            'created_at': now(), 'updated_at': now(), 'finished_at': None, 'runs': [], 'comparison': None, 'error': None,
            'dataset_version': config.runs[0].dataset_version, 'benchmark_sha256': manifest()['benchmark_sha256'],
            'protocol': {'processes': 'fresh sequential CPU worker per run; separate base benchmark worker',
                         'rss_scope': 'trainer.train; base loading, initial evaluation and benchmarks excluded',
                         'alpha_over_rank': 2, 'fixed_benchmark': True, 'repetitions': 1}}


def run_matrix(config, snapshot, experiment=None, root=None, on_update=None, on_run_update=None):
    experiment = experiment or new_experiment(config)
    if experiment['status'] != 'queued': raise ValueError('Create a new experiment; do not overwrite previous evidence')
    directory = (Path(root) if root else run_root() / 'experiments') / experiment['experiment_id']
    directory.mkdir(parents=True, exist_ok=True)
    def update(**patch):
        experiment.update(patch, updated_at=now())
        temp=directory/'.experiment.tmp';temp.write_text(json.dumps(experiment, indent=2, allow_nan=False)+'\n');temp.replace(directory/'experiment.json')
        if on_update: on_update(experiment)
    process = None; completed = []
    env = {**os.environ, 'HF_HOME': os.getenv('HF_HOME', str(Path('.cache/huggingface').resolve())),
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
    try:
        verify_snapshot(snapshot)
        if snapshot['metadata']['version'] != config.runs[0].dataset_version: raise ValueError('Dataset version mismatch')
        (directory/'snapshot.json').write_text(json.dumps(snapshot))
        update(status='running', phase='fixed-base-benchmark')
        with (directory/'baseline.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, '-m', 'recipetriage_ml.training.experiments', '--baseline', str(directory/'baseline.json')], env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None: time.sleep(1)
        if process.returncode: raise RuntimeError('Base benchmark worker failed; inspect baseline.log')
        baseline = json.loads((directory/'baseline.json').read_text())
        experiment['baseline'] = baseline
        for settings in config.runs:
            child = new_training_run(settings); child['experiment_id'] = experiment['experiment_id']; child['tracking_enabled'] = config.mlflow
            journal=Journal(child,directory/'runs');journal.write()
            (journal.directory/'snapshot.json').write_text(json.dumps(snapshot))
            (journal.directory/'reference_baseline.json').write_text(json.dumps(baseline))
            experiment['runs'].append(summary(child));update(phase=f'{settings.method} lr={settings.learning_rate} rank={settings.lora_rank}')
            with (journal.directory/'worker.log').open('wb') as log:
                process=subprocess.Popen([sys.executable,'-m','recipetriage_ml.training.experiments','--worker',str(journal.directory/'run.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
                last=None
                def progress(final=False):
                    nonlocal child,last
                    child=json.loads((journal.directory/'run.json').read_text())
                    if last != child['updated_at'] or final:
                        experiment['runs'][-1]=summary(child);update()
                        if on_run_update and (final or child['status'] in ACTIVE): on_run_update(child)
                        last=child['updated_at']
                while process.poll() is None: progress();time.sleep(1)
                progress()
                if process.returncode and child['status'] in ACTIVE:
                    child.update(status='failed',error=f'Worker exited {process.returncode}; inspect worker.log',finished_at=now());Journal(child,directory/'runs').write()
                progress(final=True)
            completed.append(child)
        comparison=compare_matrix(completed)
        update(status='completed' if all(row['status']=='completed' for row in completed) else 'partial',phase='finished',comparison=comparison,finished_at=now())
        columns=['method','learning_rate','rank','trainable_parameters','training_seconds','peak_rss_mib','micro_f1','macro_f1','exact_match','json_validity','schema_validity','run_id']
        with (directory/'comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(comparison['rows'])
    except BaseException as exc:
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        update(status='failed',error=str(exc),finished_at=now());raise
    return experiment


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv('.env')
    os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--dataset',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--api',help='Submit this config to a running RecipeTriage API instead of a local worker');parser.add_argument('--write-config',type=Path);parser.add_argument('--worker',type=Path);parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    if args.baseline:
        result=run_benchmark(RunConfig(provider='hf-base',temperature=0,max_new_tokens=128,reasoning='disabled'))
        args.baseline.write_text(json.dumps(result,indent=2))
        if result['status']!='completed' or not result['summary']['scorable']:raise RuntimeError('Baseline incomplete')
        return
    if args.worker:
        from .sft import run_local
        from .tracking import export_run
        job=json.loads(args.worker.read_text());directory=args.worker.parent
        result=run_local(SFTConfig.model_validate(job['config']),json.loads((directory/'snapshot.json').read_text()),run=job,root=directory.parent,
                         reference_baseline=json.loads((directory/'reference_baseline.json').read_text()))
        if job.get('tracking_enabled'):Journal(result,directory.parent).write(tracking=export_run(result,directory))
        return
    snapshot=load_snapshot(args.dataset) if args.dataset else seed_snapshot()
    config=MatrixConfig.model_validate_json(args.config.read_text()) if args.config else default_matrix(snapshot)
    if args.write_config:args.write_config.parent.mkdir(parents=True,exist_ok=True);args.write_config.write_text(config.model_dump_json(indent=2)+'\n');return
    if args.api:
        import httpx
        response=httpx.post(args.api.rstrip('/')+'/api/v1/experiments',json=config.model_dump(),timeout=30)
        response.raise_for_status();print(json.dumps(response.json(),indent=2));return
    result=run_matrix(config,snapshot,root=args.output)
    print(json.dumps({'experiment_id':result['experiment_id'],'status':result['status'],'comparison':result['comparison']},indent=2))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
