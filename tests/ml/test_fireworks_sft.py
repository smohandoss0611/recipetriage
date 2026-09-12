import json
import httpx
import pytest
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.data import seed_snapshot
from recipetriage_ml.training.fireworks_sft import FireworksClient, ManagedError, job_payload, preflight, run_managed, safe_job, deployed_benchmark
from recipetriage_ml.training.runs import Journal, new_training_run


def config(**kwargs):
    return SFTConfig(provider='fireworks', dataset_version=seed_snapshot()['metadata']['version'],
                     fireworks_model='accounts/fireworks/models/example',
                     fireworks_deployment_shape='accounts/fireworks/deploymentShapes/example', **kwargs)


def test_managed_v2_parameter_mapping_does_not_send_deprecated_fields():
    value = config(batch_size=2, gradient_accumulation_steps=4)
    payload = job_payload(value, 'train', 'validation', value.fireworks_model, 'output')
    assert payload['batchSizeSamples'] == 8 and payload['evalAutoCarveout'] is False
    assert payload['evaluationDataset'] == 'validation' and payload['epochs'] == 3
    assert payload['lrScheduler'] == {'constant': {}}
    assert not {'batchSize', 'gradientAccumulationSteps', 'jinjaTemplate', 'seed', 'earlyStop'} & payload.keys()


def test_actual_seed_blocked_before_upload_or_paid_job(tmp_path, monkeypatch):
    monkeypatch.delenv('FIREWORKS_ACCOUNT_ID', raising=False)
    requests = []
    def handler(request):
        requests.append(request)
        assert request.method == 'GET'
        payload = {'accounts': [{'name': 'accounts/example'}]} if request.url.path == '/v1/accounts' else {'name': 'accounts/fireworks/models/example', 'tunable': False, 'supportsLora': False}
        return httpx.Response(200, json=payload)
    client = FireworksClient('test-secret', httpx.MockTransport(handler))
    run = run_managed(config(), seed_snapshot(), root=tmp_path, client=client)
    assert run['status'] == 'blocked' and 'validation has 1' in run['error']
    assert len(requests) == 2 and not run['managed']['preflight']['upload_performed']
    assert 'test-secret' not in json.dumps(run)


def test_remote_ids_are_reused_and_uncertain_post_not_retried():
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={'name': 'accounts/example/supervisedFineTuningJobs/one'})
    client = FireworksClient('secret', httpx.MockTransport(handler))
    name = 'accounts/example/supervisedFineTuningJobs/one'
    client.get_or_create(name, 'accounts/example/supervisedFineTuningJobs', {})
    assert calls == ['GET']
    calls.clear()
    def unavailable(request):
        calls.append(request.method)
        return httpx.Response(404 if request.method == 'GET' else 503, json={'secret': 'upstream private body'})
    client.transport = httpx.MockTransport(unavailable)
    with pytest.raises(ManagedError, match='HTTP 503') as exc:
        client.get_or_create(name, 'accounts/example/supervisedFineTuningJobs', {})
    assert calls == ['GET', 'POST'] and 'private body' not in str(exc.value)


def test_signed_urls_and_user_identity_never_persisted():
    result = safe_job({'name': 'job', 'state': 'JOB_STATE_RUNNING', 'createdBy': 'private@example.com',
                       'metricsFileSignedUrl': 'https://example.com?secret=token',
                       'trainerLogsSignedUrl': 'secret', 'status': {'code': 'OK', 'message': 'private'}})
    assert 'secret' not in json.dumps(result) and 'private' not in json.dumps(result)
    assert result['metrics_available_in_provider_console'] is True


def test_preemptible_deployment_is_deleted_even_when_benchmark_fails(tmp_path, monkeypatch):
    import recipetriage_ml.training.fireworks_sft as module
    run = new_training_run(config())
    run['managed']['preflight'] = {'account': 'accounts/example', 'deployment_shape': config().fireworks_deployment_shape, 'model': config().fireworks_model}
    journal = Journal(run, tmp_path)
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == 'GET' and len(calls) == 1: return httpx.Response(404)
        return httpx.Response(200, json={'state': 'READY'})
    client = FireworksClient('test-secret', httpx.MockTransport(handler))
    def fail(*args): raise ValueError('Benchmark worker failed')
    monkeypatch.setattr(module, 'evaluate', fail)
    with pytest.raises(ValueError, match='Benchmark worker failed'):
        deployed_benchmark(journal, client, config().fireworks_model, 'baseline', timeout=0)
    assert [call[0] for call in calls] == ['GET', 'POST', 'GET', 'DELETE']
    assert calls[1][2]['preemptible'] is True
    assert 'active_deployment' not in run['managed']


def test_managed_completion_automatically_evaluates_and_preserves_job_identity(tmp_path, monkeypatch):
    import recipetriage_ml.training.fireworks_sft as module
    from recipetriage_ml.evaluation.benchmark import RunConfig, new_run, run_benchmark
    from recipetriage_ml.inference.contracts import Generation
    class Dummy:
        def generate(self, *args): return Generation(raw_output='{"labels":["unclear"],"explanation":"Fixture."}', model='mock', finish_reason='stop')
    # Mock supported provider constraints for lifecycle testing; production preflight still blocks the seed.
    monkeypatch.setattr(module, 'preflight', lambda *args: {'supported': True, 'issues': [], 'account': 'accounts/example', 'model': config().fireworks_model, 'deployment_shape': config().fireworks_deployment_shape})
    evaluated = []
    def fake_benchmark(journal, client, model, key):
        evaluated.append((model, key))
        evidence = run_benchmark(RunConfig(provider='fireworks', reasoning='disabled'), provider_instance=Dummy())
        evidence['model_config'].update(model=model, base_model=config().fireworks_model)
        journal.run[key] = evidence
        return evidence
    monkeypatch.setattr(module, 'deployed_benchmark', fake_benchmark)
    created = []
    class Client:
        def get_or_create(self, name, collection, body, params=None):
            created.append((name, body)); return {'state': 'READY'}
        def request(self, method, path, **kwargs):
            return {'name': path[4:], 'state': 'JOB_STATE_COMPLETED', 'outputModel': 'trained'}
    run = run_managed(config(), seed_snapshot(), root=tmp_path, client=Client(), poll_seconds=0)
    assert run['status'] == 'completed' and run['comparison']['comparable']
    assert evaluated == [(config().fireworks_model, 'baseline'), ('accounts/example/models/trained', 'benchmark')]
    assert run['managed']['job_name'].endswith('rt-' + run['run_id'])
    assert len(created) == 3
    with pytest.raises(ValueError, match='immutable'):
        run_managed(config(), seed_snapshot(), run=run, root=tmp_path, client=Client())
