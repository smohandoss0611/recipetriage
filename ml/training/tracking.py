"""Optional MLflow export. PostgreSQL/files stay authoritative when tracking fails."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from .runs import run_root


def tracking_uri():
    if os.getenv('MLFLOW_TRACKING_URI'):
        return os.environ['MLFLOW_TRACKING_URI']
    directory = run_root() / 'tracking'
    directory.mkdir(parents=True, exist_ok=True)
    return 'sqlite:///' + str(directory / 'mlflow.db')


def export_run(run, directory):
    """Log parameters, step history, final scores and evidence, never secrets/headers."""
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Metric, Param
        uri = tracking_uri()
        client = MlflowClient(tracking_uri=uri)
        name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'RecipeTriage-fixed-benchmark')
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            if urlparse(uri).scheme in {'http', 'https'}:
                # The tracking server chooses its artifact store and proxies uploads.
                # A client-local file URI would leave the dashboard unable to read them.
                experiment_id = client.create_experiment(name)
            else:
                artifacts = run_root() / 'mlflow-artifacts'; artifacts.mkdir(parents=True, exist_ok=True)
                experiment_id = client.create_experiment(name, artifact_location=artifacts.as_uri())
        else:
            experiment_id = experiment.experiment_id
        matches = client.search_runs([experiment_id], filter_string=f"tags.recipetriage_run_id = '{run['run_id']}'")
        tracked = matches[0] if matches else client.create_run(experiment_id, tags={
            'recipetriage_run_id': run['run_id'], 'dataset_version': run['config']['dataset_version'],
            'benchmark_sha256': run['benchmark']['benchmark']['benchmark_sha256'],
            'mlflow.runName': f"{run['config'].get('method', 'lora')}-lr{run['config']['learning_rate']}-rank{run['config']['lora_rank']}",
            'source_created_at': run.get('created_at', ''),
            'method': run['config'].get('method', 'lora'), 'annotation_status': 'provisional-unreviewed'})
        tracked_id = tracked.info.run_id
        params = [Param(key, str(value)) for key, value in run['config'].items() if value is not None]
        metrics = []
        import time
        timestamp = int(time.time() * 1000)
        for row in run.get('history', []):
            for key in ['loss', 'eval_loss', 'grad_norm', 'learning_rate']:
                if isinstance(row.get(key), (int, float)):
                    metrics.append(Metric(key, row[key], timestamp, int(row['step'])))
        summary = run['benchmark']['summary']; labels = summary.get('labels') or {}
        final = {'benchmark_micro_f1': labels.get('micro', {}).get('f1'),
                 'benchmark_exact_match': labels.get('exact_match'), 'json_validity': summary['json_validity'],
                 'training_seconds': run.get('training_resources', {}).get('train_wall_seconds'),
                 'peak_rss_bytes': run.get('training_resources', {}).get('peak_rss_bytes'),
                 'trainable_parameters': run.get('model_parameters', {}).get('trainable')}
        metrics += [Metric(key, value, timestamp, run.get('optimizer_steps', 0)) for key, value in final.items() if value is not None]
        client.log_batch(tracked_id, metrics=metrics, params=params)
        for filename in ['run.json', 'baseline.json', 'benchmark.json', 'parameter_report.json', 'quantization.json', 'loss_by_step.csv']:
            path = Path(directory) / filename
            if path.is_file(): client.log_artifact(tracked_id, str(path), artifact_path='evidence')
        client.set_terminated(tracked_id, status='FINISHED' if run['status'] == 'completed' else 'FAILED')
        return {'status': 'synced', 'provider': 'mlflow', 'run_id': tracked_id, 'experiment_id': experiment_id,
                'note': 'Full source evidence remains in the RecipeTriage registry; no credentials are exported.'}
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception('MLflow export failed')
        return {'status': 'failed', 'provider': 'mlflow', 'error': f'{type(exc).__name__}: MLflow export failed; inspect logs and retry export. Training evidence is retained.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export a saved training journal to MLflow without changing it')
    parser.add_argument('run', type=Path, nargs='+')
    args = parser.parse_args()
    failed = False
    for path in args.run:
        result = export_run(json.loads(path.read_text()), path.parent)
        print(json.dumps({'source': str(path), **result}, indent=2))
        failed = failed or result['status'] != 'synced'
    if failed:
        raise SystemExit(1)
