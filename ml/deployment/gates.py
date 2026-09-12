"""Fail-closed promotion gates; recompute metrics from complete immutable evidence."""
import copy
import math
from pydantic import BaseModel,ConfigDict,Field
from recipetriage_ml.data.schemas import LABELS
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.evaluation.benchmark import load_benchmark,new_run,RunConfig
from recipetriage_ml.evaluation.metrics import inspect_response,summarize
from recipetriage_ml.evaluation import shortcut_tests,behaviors
from recipetriage_ml.inference.prompts import build_messages
from recipetriage_ml.inference.contracts import Recipe


class GateConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    min_macro_f1:float=Field(default=.45,ge=0,le=1,allow_inf_nan=False)
    min_json_validity:float=Field(default=.95,ge=0,le=1,allow_inf_nan=False)
    min_schema_validity:float=Field(default=.95,ge=0,le=1,allow_inf_nan=False)
    min_shortcut_accuracy:float=Field(default=.67,ge=0,le=1,allow_inf_nan=False)
    min_shortcut_pair_coverage:float=Field(default=1.,gt=0,le=1,allow_inf_nan=False)
    max_prediction_flip_rate:float=Field(default=.0,ge=0,le=1,allow_inf_nan=False)
    max_per_label_f1_regression:float=Field(default=.05,ge=0,le=1,allow_inf_nan=False)
    important_labels:list[str]=Field(default_factory=lambda:list(LABELS),min_length=1)


def checked_report(benchmark,shortcut):
    expected=new_run(RunConfig(provider='hf-base',temperature=0,max_new_tokens=128,reasoning='disabled'))
    if benchmark['status']!='completed' or benchmark['protocol_sha256']!=expected['protocol_sha256']:
        raise ValueError('Complete fixed-protocol benchmark required')
    cases=load_benchmark().cases;by_id={x.recipe.id:x for x in cases}
    if len(benchmark['rows'])!=len(cases):raise ValueError('Missing benchmark responses')
    for row in benchmark['rows']:
        case=by_id[row['case_id']];messages=build_messages(case.recipe.inference_recipe())
        if row['messages']!=messages or row['expected_labels']!=case.labels or row['category']!=case.category:
            raise ValueError('Benchmark input/answer key changed')
        if row['raw_output'] is None:raise ValueError('All planned benchmark responses are required')
        checked=inspect_response(row['raw_output'],row['finish_reason'])
        if any(row[k]!=checked[k] for k in ['prediction','usable','json_valid','schema_valid']):raise ValueError('Benchmark score differs from raw output')
        if row.get('model')!=benchmark['model_config']['model'] or row.get('revision')!=benchmark['model_config']['revision']:
            raise ValueError('Benchmark mixes checkpoint identities')
    if summarize(cases,benchmark['rows'])!=benchmark['summary']:raise ValueError('Benchmark aggregates changed')
    spec=shortcut_tests.suite()
    if not shortcut or shortcut['status']!='completed' or shortcut['suite']!=spec or shortcut['suite_sha256']!=digest(spec):raise ValueError('Complete unchanged shortcut suite required')
    if shortcut['model_config']!=benchmark['model_config']:raise ValueError('Benchmark and shortcut checkpoint differ')
    if len(shortcut['rows'])!=spec['expected_rows']:raise ValueError('Missing title variants')
    for row in shortcut['rows']:
        checked=inspect_response(row['raw_output'],row['finish_reason'])
        if any(row[k]!=checked[k] for k in ['prediction','usable','json_valid','schema_valid']):raise ValueError('Shortcut score differs from raw output')
        if row['messages']!=build_messages(Recipe.model_validate(row['recipe'])):raise ValueError('Shortcut prompt changed')
        if row.get('model')!=benchmark['model_config']['model'] or row.get('revision')!=benchmark['model_config']['revision']:raise ValueError('Shortcut mixes checkpoints')
    if shortcut_tests.summarize(shortcut)['summary']!=shortcut['summary']:raise ValueError('Shortcut aggregates changed')
    return behaviors.report(benchmark,shortcut)


def evaluate_gate(benchmark,shortcut,baseline,config=None):
    config=config or GateConfig();failures=[]
    try:
        metrics=checked_report(benchmark,shortcut)
        if not baseline or baseline['protocol_sha256']!=benchmark['protocol_sha256'] or not baseline['summary'].get('labels'):
            raise ValueError('A comparable reference benchmark is required for per-label regression checks')
        # Baseline raw rows are recomputed too, not trusted client scores.
        checked=copy.deepcopy(baseline)
        for row in checked['rows']:row.update(inspect_response(row['raw_output'],row['finish_reason']))
        if summarize(load_benchmark().cases,checked['rows'])!=baseline['summary']:raise ValueError('Baseline metrics do not match raw responses')
        for key,minimum in [('macro_f1',config.min_macro_f1),('json_validity',config.min_json_validity),('schema_validity',config.min_schema_validity)]:
            if metrics[key]<minimum:failures.append(f'{key}: {metrics[key]:.4f} < {minimum:.4f}')
        s=metrics['shortcut']
        if s['misleading_title_accuracy']<config.min_shortcut_accuracy:failures.append('Misleading-title accuracy below threshold')
        if s['pair_coverage']<config.min_shortcut_pair_coverage:failures.append('Insufficient usable counterfactual-pair coverage')
        if s['prediction_flip_rate'] is None or s['prediction_flip_rate']>config.max_prediction_flip_rate:failures.append('Prediction-flip rate unavailable or above threshold')
        for label in config.important_labels:
            if label not in LABELS:raise ValueError('Unknown important label in gate config')
            before=baseline['summary']['labels']['per_label'][label]['f1'];after=metrics['per_label'][label]['f1']
            if after+config.max_per_label_f1_regression<before-1e-12:failures.append(f'{label} F1 regressed by {before-after:.4f}')
    except (ValueError,KeyError,TypeError) as exc:
        metrics=None;failures.append(str(exc))
    return {'passed':not failures,'failures':failures,'metrics':metrics,'thresholds':config.model_dump(),
            'policy':'Complete evidence, absolute quality, coverage and per-label nonregression; no size-based promotion.'}
