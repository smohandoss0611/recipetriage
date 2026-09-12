"""Managed SFT with explicit capability checks, resumable IDs and automatic evaluation.

Training V2 owns tokenization, LoRA targets, optimizer implementation and hardware.
It accepts samples per update, not our local microbatch/accumulation controls.
"""
import json
import os
from pathlib import Path
import re
import time
import httpx
from recipetriage_ml.inference.contracts import ProviderError
from recipetriage_ml.inference.fireworks_provider import FireworksProvider
from .config import SFTConfig
from .data import load_snapshot, prepare_snapshot, seed_snapshot, verify_snapshot
from .runs import Journal, compare, evaluate, new_training_run, now


class ManagedError(ProviderError):
    def __init__(self, operation, status=None):
        self.status = status
        super().__init__(f'Fireworks {operation}: HTTP {status}' if status else f'Fireworks {operation}: connection failed or timed out; inspect the saved resource ID before retrying')


class FireworksClient:
    def __init__(self, api_key=None, transport=None):
        self.api_key = api_key or os.getenv('FIREWORKS_API_KEY', '')
        self.transport = transport

    def request(self, method, path, **kwargs):
        if not self.api_key:
            raise ProviderError('FIREWORKS_API_KEY is not configured')
        if not path.startswith('/v1/accounts') or '..' in path or '?' in path or '#' in path:
            raise ValueError('Invalid Fireworks resource path')
        try:
            with httpx.Client(timeout=httpx.Timeout(120, connect=10), transport=self.transport) as client:
                response = client.request(method, 'https://api.fireworks.ai' + path,
                                          headers={'Authorization': 'Bearer ' + self.api_key}, **kwargs)
                if not response.is_success:
                    raise ManagedError(f'{method} {path}', response.status_code)
                return response.json() if response.content else {}
        except httpx.RequestError as exc:
            raise ManagedError(f'{method} {path}') from exc

    def account(self):
        account = os.getenv('FIREWORKS_ACCOUNT_ID')
        if account:
            if not re.fullmatch(r'[a-z0-9-]+', account):
                raise ValueError('FIREWORKS_ACCOUNT_ID must be the account ID without slashes')
            return 'accounts/' + account
        response = self.request('GET', '/v1/accounts')
        accounts = response.get('accounts', [])
        if len(accounts) != 1 or response.get('nextPageToken'):
            raise ValueError('Set FIREWORKS_ACCOUNT_ID: the API key does not resolve to exactly one account')
        return accounts[0]['name']

    def get_or_create(self, name, collection, body, params=None):
        """GET first; deterministic IDs make an uncertain POST resumable without a second job."""
        try:
            return self.request('GET', '/v1/' + name)
        except ManagedError as exc:
            if exc.status != 404:
                raise
        try:
            return self.request('POST', '/v1/' + collection, json=body, params=params)
        except ManagedError as exc:
            if exc.status == 409:
                return self.request('GET', '/v1/' + name)
            raise


def preflight(config, snapshot, client):
    verify_snapshot(snapshot)
    counts = snapshot['metadata']['counts']
    issues = []
    for split in ['train', 'validation']:
        if counts[split] < 3:
            issues.append(f'{split} has {counts[split]} examples; Fireworks documents a minimum of 3 per dataset. Add independent hand-reviewed recipes and create a new version; no duplication or automatic re-splitting.')
    account = client.account()
    model_name = config.fireworks_model or os.getenv('FIREWORKS_SFT_MODEL') or os.getenv('FIREWORKS_MODEL')
    metadata = {}
    if not model_name or not re.fullmatch(r'accounts/[a-z0-9-]+/models/[a-z0-9_.-]+', model_name):
        issues.append('Choose FIREWORKS_SFT_MODEL from currently tunable models.')
    else:
        raw = client.request('GET', '/v1/' + model_name)
        metadata = {key: raw.get(key) for key in ['name', 'state', 'tunable', 'supportsLora', 'supervisedLoraTunable',
                                                 'trainingContextLength', 'contextLength', 'updateTime', 'huggingFaceUrl']}
        supported = raw.get('supervisedLoraTunable') or (raw.get('tunable') and raw.get('supportsLora'))
        if not supported:
            issues.append(f'{model_name} does not report support for supervised LoRA tuning.')
        context = raw.get('trainingContextLength') or raw.get('contextLength')
        if context and config.sequence_length > int(context):
            issues.append(f'sequence_length exceeds the provider model limit of {context}.')
    shape = config.fireworks_deployment_shape or os.getenv('FIREWORKS_SFT_DEPLOYMENT_SHAPE')
    if not shape:
        issues.append('Set a compatible FIREWORKS_SFT_DEPLOYMENT_SHAPE for temporary preemptible benchmark deployments.')
    elif not re.fullmatch(r'accounts/[a-z0-9-]+/deploymentShapes/[a-z0-9_.-]+', shape):
        issues.append('Invalid deployment shape resource name.')
    return {'supported': not issues, 'issues': issues, 'account': account, 'model': model_name,
            'model_metadata': metadata, 'deployment_shape': shape, 'checked_at': now(),
            'counts': counts, 'minimum_examples_per_uploaded_dataset': 3,
            'renderer': 'provider registered chat renderer; token boundaries may differ from local',
            'upload_performed': False}


def job_payload(config, train_dataset, validation_dataset, model, output_id):
    return {'dataset': train_dataset, 'evaluationDataset': validation_dataset, 'evalAutoCarveout': False,
            'baseModel': model, 'outputModel': output_id, 'displayName': 'RecipeTriage SFT learning run',
            'epochs': config.epochs, 'learningRate': config.learning_rate, 'maxContextLength': config.sequence_length,
            'loraRank': config.lora_rank, 'batchSizeSamples': config.effective_batch_size,
            'learningRateWarmupSteps': 0, 'lrScheduler': {'constant': {}}, 'optimizerWeightDecay': 0.0}


def safe_job(job):
    # In particular, do not persist signed download URLs, user emails or credentials.
    keys = ['name', 'state', 'baseModel', 'outputModel', 'createTime', 'completedTime', 'updateTime',
            'jobProgress', 'estimatedCost']
    result = {key: job[key] for key in keys if key in job}
    result['status_code'] = (job.get('status') or {}).get('code')
    result['metrics_available_in_provider_console'] = bool(job.get('metricsFileSignedUrl'))
    return result


def cleanup_deployment(journal, client):
    deployment = journal.run['managed'].get('active_deployment')
    if not deployment:
        return
    try:
        client.request('DELETE', '/v1/' + deployment)
    except ManagedError as exc:
        if exc.status != 404:
            journal.run['managed']['cleanup_error'] = str(exc)
            journal.write()
            raise
    journal.run['managed'].pop('active_deployment', None)
    journal.run['managed'].pop('cleanup_error', None)
    journal.write()


def deployed_benchmark(journal, client, model, key, timeout=600):
    managed = journal.run['managed']
    account = managed['preflight']['account']
    name = f"{account}/deployments/rt-{journal.run['run_id'][:24]}-{key}"
    # Persist ownership before POST so a restart can clean up even after a lost response.
    managed['active_deployment'] = name
    journal.write(phase=f'deploy-{key}')
    try:
        client.get_or_create(name, account + '/deployments',
            {'baseModel': model, 'displayName': f'RecipeTriage {key}',
             'deploymentShape': managed['preflight']['deployment_shape'],
             'minReplicaCount': 1, 'maxReplicaCount': 1, 'preemptible': True},
            {'deploymentId': name.rsplit('/', 1)[1]})
        deadline = time.monotonic() + timeout
        while True:
            deployment = client.request('GET', '/v1/' + name)
            managed['deployment_state'] = deployment.get('state')
            journal.write()
            if deployment.get('state') == 'READY':
                break
            if deployment.get('state') in {'FAILED', 'DELETED', 'DELETING'} or time.monotonic() >= deadline:
                raise ProviderError('Preemptible evaluation capacity did not become ready; benchmark has not completed')
            time.sleep(10)
        identity = {'model': model, 'revision': None, 'base_model': managed['preflight']['model'],
                    'training_stage': 'hosted-base-before-sft' if key == 'baseline' else 'managed-supervised-fine-tuned-lora',
                    'prompt_format': 'provider-chat-template', 'deployment': name}
        provider = FireworksProvider(api_key=client.api_key, model=model + '#' + name,
                                     transport=client.transport, reasoning_effort='none')
        return evaluate(journal, provider, identity, key)
    finally:
        cleanup_deployment(journal, client)


def run_managed(config, snapshot, run=None, root=None, on_update=None, client=None, preflight_only=False, poll_seconds=1800):
    if config.provider != 'fireworks':
        raise ValueError('run_managed requires provider=fireworks')
    if run and run['status'] in {'completed', 'failed'}:
        raise ValueError('Finished training evidence is immutable; start a new run')
    journal = Journal(run or new_training_run(config), root, on_update)
    client = client or FireworksClient()
    managed = journal.run['managed']
    try:
        if snapshot['metadata']['version'] != config.dataset_version:
            raise ValueError('Requested dataset version does not match snapshot')
        _, data = prepare_snapshot(snapshot, journal.directory / 'data')
        journal.write(status='preparing', phase='provider-preflight', dataset=data, error=None)
        if not preflight_only:
            cleanup_deployment(journal, client)
        if 'preflight' not in managed or not managed.get('job_name'):
            managed['preflight'] = preflight(config, snapshot, client)
            journal.write()
        check = managed['preflight']
        journal.run['parameters']['base_model'] = check['model']
        if not check['supported'] or preflight_only:
            journal.write(status='blocked' if not check['supported'] else 'ready', phase='preflight-finished',
                          error='; '.join(check['issues']) or None, finished_at=now())
            return journal.run
        # A resumed worker cleans its own leftover evaluation deployment before continuing.
        cleanup_deployment(journal, client)
        account = check['account']
        if not journal.run['baseline'] or journal.run['baseline']['status'] != 'completed':
            journal.write(status='baseline', phase='benchmark-hosted-base')
            before = deployed_benchmark(journal, client, check['model'], 'baseline')
            if before['status'] != 'completed' or not before['summary']['returned_responses']:
                raise ProviderError('Hosted base benchmark blocked; no fine-tuning job submitted')
        journal.write(status='submitting', phase='upload-datasets')
        datasets = managed.setdefault('datasets', {})
        for split in ['train', 'validation']:
            name = f"{account}/datasets/rt-{journal.run['run_id'][:24]}-{split}"
            datasets[split] = name
            journal.write()
            dataset = client.get_or_create(name, account + '/datasets',
                {'datasetId': name.rsplit('/', 1)[1], 'dataset': {'userUploaded': {}, 'exampleCount': str(data['counts'][split]), 'format': 'CHAT'}})
            if dataset.get('state') != 'READY':
                with (journal.directory / 'data' / f'{split}.fireworks.jsonl').open('rb') as handle:
                    client.request('POST', '/v1/' + name + ':upload', files={'file': (split + '.jsonl', handle, 'application/jsonl')})
                deadline = time.monotonic() + 120
                while client.request('GET', '/v1/' + name).get('state') != 'READY':
                    if time.monotonic() >= deadline:
                        raise ProviderError('Dataset upload still processing; resume the saved run')
                    time.sleep(5)
            check['upload_performed'] = True
        job_id = 'rt-' + journal.run['run_id']
        name = f'{account}/supervisedFineTuningJobs/{job_id}'
        payload = job_payload(config, datasets['train'], datasets['validation'], check['model'], job_id)
        managed.update(job_name=name, submitted_payload=payload)
        journal.write(phase='create-or-resume-job')
        client.get_or_create(name, account + '/supervisedFineTuningJobs', payload, {'supervisedFineTuningJobId': job_id})
        deadline = time.monotonic() + poll_seconds
        journal.write(status='monitoring', phase='managed-optimizer-updates')
        while True:
            job = client.request('GET', '/v1/' + name)
            managed['job'] = safe_job(job)
            journal.write()
            state = job.get('state')
            if state == 'JOB_STATE_COMPLETED':
                break
            if state in {'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED', 'JOB_STATE_DELETED', 'JOB_STATE_ARCHIVED', 'JOB_STATE_EARLY_STOPPED'}:
                journal.write(status='failed', phase='managed-job-ended', finished_at=now(),
                              error=f'Managed training ended as {state}; inspect the provider console for details')
                return journal.run
            if time.monotonic() >= deadline or state == 'JOB_STATE_PAUSED':
                journal.write(status='waiting', phase='resume-monitoring', error='Remote job still active or paused. Resume monitoring; this will reuse the saved job ID.')
                return journal.run
            time.sleep(10)
        output = job.get('outputModel') or job_id
        if '/' not in output:
            output = f'{account}/models/{output}'
        managed['output_model'] = output
        journal.write(status='evaluating', phase='benchmark-managed-checkpoint')
        after = deployed_benchmark(journal, client, output, 'benchmark')
        journal.write(comparison=compare(journal.run['baseline'], after),
                      status='completed' if after['status'] == 'completed' and after['summary']['returned_responses'] else 'evaluation-blocked',
                      phase='finished', finished_at=now())
    except (ValueError, ProviderError) as exc:
        # A saved remote job can continue after a network outage; never resubmit with a new ID.
        journal.write(status='waiting' if managed.get('job_name') or managed.get('active_deployment') else 'blocked', error=str(exc), finished_at=None)
    return journal.run


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Existing run.json; reuse saved resource IDs')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    snapshot = load_snapshot(args.dataset) if args.dataset else seed_snapshot()
    run = json.loads(args.resume.read_text()) if args.resume else None
    config = SFTConfig.model_validate(run['config']) if run else SFTConfig.model_validate_json(args.config.read_text()) if args.config else SFTConfig(provider='fireworks', dataset_version=snapshot['metadata']['version'])
    result = run_managed(config, snapshot, run=run, root=args.resume.parent.parent if args.resume else args.output, preflight_only=args.preflight_only)
    print(json.dumps({'run_id': result['run_id'], 'status': result['status'], 'error': result['error']}, indent=2))


if __name__ == '__main__':
    main()
