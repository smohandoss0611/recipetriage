import copy
import json
import math
import pytest
from recipetriage_ml.evaluation.metrics import inspect_response, label_metrics, latency_metrics, summarize
from recipetriage_ml.evaluation.benchmark import Benchmark, RunConfig, load_benchmark, manifest, new_run, run_benchmark, save_file
from recipetriage_ml.evaluation.base_provider import render_prompt
from recipetriage_ml.inference.contracts import Generation, ProviderError


def test_hand_calculated_multilabel_metrics():
    actual = label_metrics([['dessert', 'meal-prep'], ['dessert']], [['dessert'], ['dessert', 'weeknight-30min']])
    assert actual['micro']['precision'] == pytest.approx(2/3)
    assert actual['micro']['recall'] == pytest.approx(2/3)
    assert actual['micro']['f1'] == pytest.approx(2/3)
    assert actual['macro']['f1'] == pytest.approx(1/7)
    assert actual['exact_match'] == 0
    assert actual['per_label']['meal-prep']['fn'] == 1
    assert actual['per_label']['weeknight-30min']['fp'] == 1
    assert actual['per_label']['unclear']['support'] == 0


def test_set_order_and_unusable_prediction_policy():
    assert label_metrics([['dessert', 'meal-prep']], [['meal-prep', 'dessert']])['exact_match'] == 1
    result = label_metrics([['dessert']], [['dessert']], [False])
    assert result['exact_match'] == 0
    assert result['per_label']['dessert']['fn'] == 1


@pytest.mark.parametrize('raw,syntax,schema', [
    ('{"labels":["dessert"],"explanation":"A dessert."}', True, True),
    ('{"labels":["bogus"],"explanation":"A dessert."}', True, False),
    ('[]', True, False),
    ('{"labels":["dessert"],"labels":["meal-prep"],"explanation":"A dessert."}', True, False),
    ('{"labels":["dessert","dessert"],"explanation":"A dessert."}', True, False),
    ('{"labels":["dessert","unclear"],"explanation":"A dessert."}', True, False),
    ('```json\n{"labels":["dessert"]}\n```', False, False),
    ('{"labels":', False, False), ('NaN', False, False), ('', False, False),
])
def test_json_syntax_is_distinct_from_schema(raw, syntax, schema):
    result = inspect_response(raw, 'stop')
    assert result['json_valid'] is syntax
    assert result['schema_valid'] is schema
    assert result['usable'] is schema


def test_truncation_does_not_count_as_a_usable_answer():
    result = inspect_response('{"labels":["dessert"],"explanation":"Dessert."}', 'length')
    assert result['json_valid'] and result['schema_valid'] and not result['usable']


def test_latency_uses_nearest_rank_p95_and_null_for_no_measurement():
    result = latency_metrics([10, 20, 30, 40])
    assert result['p50_ms'] == 25 and result['p95_ms'] == 40 and result['mean_ms'] == 25
    assert latency_metrics([])['mean_ms'] is None
    with pytest.raises(ValueError): latency_metrics([float('nan')])


def test_benchmark_is_frozen_covered_and_separate_from_seed():
    a = manifest(); b = manifest()
    assert a == b
    assert a['benchmark_sha256'] == '173e79cc1968dd1207a7aafd051dad40d87ff551c7081a17b896c9556a698448'
    assert len(a['cases']) == 10 and len(a['category_counts']) == 6
    assert min(a['label_support'].values()) > 0
    assert not a['contamination_audit']['dataset_v1_source_or_body_overlap']
    data = load_benchmark().model_dump()
    data['cases'][1] = copy.deepcopy(data['cases'][0])
    with pytest.raises(ValueError): Benchmark.model_validate(data)


class PerfectProvider:
    def __init__(self): self.index = 0
    def generate(self, messages, temperature, max_new_tokens):
        case = load_benchmark().cases[self.index]; self.index += 1
        assert case.recipe.source_uri not in messages[1]['content']
        assert 'expected_labels' not in messages[1]['content']
        return Generation(raw_output=json.dumps({'labels': case.labels, 'explanation': 'Test fixture answer.'}),
                          model='mock-provider', finish_reason='stop')


def test_runner_success_preserves_exact_inputs_and_incremental_progress(tmp_path):
    updates = []
    def persist(run):
        updates.append(len(run['rows'])); save_file(run, tmp_path)
    config = RunConfig(provider='hf-base')
    run = run_benchmark(config, on_update=persist, provider_instance=PerfectProvider())
    assert run['status'] == 'completed'
    assert run['summary']['labels']['exact_match'] == 1
    assert run['summary']['labels']['macro']['f1'] == 1
    assert run['summary']['json_validity'] == 1
    assert set(range(11)) <= set(updates)
    assert json.loads((tmp_path / (run['run_id'] + '.json')).read_text()) == run
    assert run['rows'][0]['rendered_prompt'] == render_prompt(run['rows'][0]['messages'])


def test_model_errors_are_preserved_and_count_as_failed_label_sets():
    class Broken:
        def generate(self, *args): raise ProviderError('Transient connection failure')
    run = run_benchmark(RunConfig(provider='fireworks'), provider_instance=Broken())
    assert run['status'] == 'completed'
    assert run['summary']['scorable']
    assert run['summary']['labels']['exact_match'] == 0
    assert run['summary']['json_validity'] is None
    assert all(row['attempted'] and row['error'] for row in run['rows'])


def test_missing_credentials_are_blocked_not_a_zero_model_score(monkeypatch):
    monkeypatch.delenv('FIREWORKS_API_KEY', raising=False)
    run = run_benchmark(RunConfig(provider='fireworks'))
    assert run['status'] == 'blocked'
    assert run['summary']['attempted_cases'] == 0
    assert run['summary']['labels'] is None
    assert run['summary']['latency_all_attempts']['count'] == 0


def test_authorization_failure_stops_further_paid_requests():
    class Forbidden:
        count = 0
        def generate(self, *args):
            self.count += 1
            raise ProviderError('Fireworks returned HTTP 403; check access')
    provider = Forbidden()
    run = run_benchmark(RunConfig(provider='fireworks'), provider_instance=provider)
    assert provider.count == 1
    assert run['status'] == 'blocked' and run['summary']['labels'] is None


def test_partial_runs_cannot_appear_as_complete_scores():
    run = run_benchmark(RunConfig(provider='hf-base'), provider_instance=PerfectProvider())
    summary = summarize(load_benchmark().cases, run['rows'][:-1])
    assert summary['labels'] is None and not summary['scorable']
    with pytest.raises(ValueError): summarize(load_benchmark().cases, run['rows'] + run['rows'][:1])


def test_model_comparison_requires_matching_protocol():
    a = new_run(RunConfig(provider='hf-base'))
    b = new_run(RunConfig(provider='fireworks'))
    c = new_run(RunConfig(provider='fireworks', max_new_tokens=256))
    assert a['protocol_sha256'] == b['protocol_sha256']
    assert a['protocol_sha256'] != c['protocol_sha256']
    disabled = new_run(RunConfig(provider='hf-base', reasoning='disabled'))
    assert disabled['protocol_sha256'] != a['protocol_sha256']
    assert disabled['protocol_sha256'] == new_run(RunConfig(provider='fireworks', reasoning='disabled'))['protocol_sha256']


def test_fireworks_reasoning_mode_is_explicit_and_opt_in():
    import httpx
    from recipetriage_ml.inference.fireworks_provider import FireworksProvider
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'model':'test-model', 'choices':[{'finish_reason':'stop', 'message':{'content':'{"labels":["dessert"],"explanation":"Dessert."}'}}]})
    for mode in [None, 'none']:
        FireworksProvider(api_key='test-key', model='test-model', transport=httpx.MockTransport(respond), reasoning_effort=mode).generate([])
    assert 'reasoning_effort' not in requests[0]
    assert requests[1]['reasoning_effort'] == 'none'
