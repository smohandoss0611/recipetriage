"""Offline fixtures exercise mechanics; they are not human-reviewed training evidence."""
import copy,json
from pathlib import Path
import pytest
from recipetriage_ml.alignment.rewards import components
from recipetriage_ml.alignment.preferences import PreferencePair,preference_dataset,generate_pairs
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.inference.contracts import Generation
from recipetriage_ml.evaluation.benchmark import RunConfig,new_run,run_benchmark,descriptor,load_benchmark
from recipetriage_ml.evaluation.shortcut_tests import suite,run_suite
from recipetriage_ml.deployment.gates import evaluate_gate,GateConfig
from recipetriage_ml.deployment.benchmark import select_candidate,Constraints
from recipetriage_ml.training.lora import target_paths,estimated_parameters,architecture_report


def perfect_evidence():
    truth={digest(x.recipe.inference_recipe().model_dump(exclude={'title'})):x.labels for x in load_benchmark().cases}
    truth.update({digest({k:v for k,v in x['recipe'].items() if k!='title'}):x['expected_labels'] for x in suite()['groups']})
    identity=descriptor('hf-base')
    class Provider:
        def prepare(self):return {'fixture':True}
        def generate(self,messages,*args):
            recipe=json.loads(messages[-1]['content']);key=digest({k:v for k,v in recipe.items() if k!='title'})
            return Generation(raw_output=json.dumps({'labels':truth[key],'explanation':'Test fixture only.'}),
                              model=identity['model'],revision=identity['revision'],finish_reason='stop')
    config=RunConfig(provider='hf-base',reasoning='disabled');provider=Provider()
    return run_benchmark(config,provider_instance=provider),run_suite(provider,identity)


def test_reward_correctness_gates_consistency_and_uncertainty():
    good=json.dumps({'labels':['weekend-project'],'explanation':'Active preparation takes 60 minutes.'})
    wrong=json.dumps({'labels':['weeknight-30min'],'explanation':'Quick title.'})
    correct=components(good,['weekend-project'],counterfactual_raw=good)
    assert correct['total']==pytest.approx(1.)
    assert components(wrong,['weekend-project'],counterfactual_raw=wrong)['shortcut_consistency']==0
    assert components('{bad',['unclear'])['total']==0
    unclear=json.dumps({'labels':['unclear'],'explanation':'No evidence.'})
    assert components(unclear,['unclear'])['uncertainty']==1
    assert components(unclear,['dessert'])['uncertainty']==0


def test_gate_recomputes_metrics_and_requires_coverage_and_nonregression():
    benchmark,shortcuts=perfect_evidence()
    assert evaluate_gate(benchmark,shortcuts,benchmark)['passed']
    bad=copy.deepcopy(benchmark);bad['summary']['labels']['macro']['f1']=.99
    assert not evaluate_gate(bad,shortcuts,benchmark)['passed']
    bad=copy.deepcopy(shortcuts);bad['rows'].pop()
    assert not evaluate_gate(benchmark,bad,benchmark)['passed']
    assert not evaluate_gate(benchmark,None,benchmark)['passed']
    bad=copy.deepcopy(shortcuts);bad['rows'][1]['recipe']['time_minutes']+=1
    assert not evaluate_gate(benchmark,bad,benchmark)['passed']


def test_pending_or_tied_pairs_never_form_dpo_dataset():
    class Provider:
        model='fixture'
        def generate(self,*args):return Generation(raw_output='{}',model=self.model,finish_reason='stop')
    rows=generate_pairs(local=Provider(),hosted=Provider(),count=3)
    assert all(x['status']=='pending' and x['decision'] is None for x in rows)
    with pytest.raises(ValueError,match='three independent'):preference_dataset(rows)
    fake=copy.deepcopy(rows[0]);fake['status']='chosen'
    with pytest.raises(ValueError,match='explicit human'):PreferencePair.model_validate(fake)


def test_layer_subsets_come_from_real_architecture_and_have_equal_budgets():
    report=architecture_report()
    assert len(target_paths(report,layers=[0,1,2,3,4,5]))==12
    assert estimated_parameters(report,4,layers=list(range(6)))==67584
    assert estimated_parameters(report,4,layers=list(range(18,24)))==67584
    for bad in [[24],[-1],[0,0],[True]]:
        with pytest.raises(ValueError):target_paths(report,layers=bad)


def test_deployment_selection_is_quality_first_not_smallest():
    common={'schema_validity':1.,'peak_ram_mib':1000.,'p95_ms':1000.}
    rows=[{'name':'small','micro_f1':.5,'size_mib':100.,**common},{'name':'better','micro_f1':.8,'size_mib':1000.,**common}]
    assert select_candidate(rows)['recommended']=='better'
    assert select_candidate(rows,Constraints(max_size_mib=200))['recommended']=='small'
    assert select_candidate(rows,Constraints(min_micro_f1=.9))['recommended'] is None


def test_gate_rejects_important_label_regression_even_if_absolute_thresholds_pass():
    from recipetriage_ml.evaluation.metrics import inspect_response,summarize
    baseline,shortcut=perfect_evidence()
    candidate=copy.deepcopy(baseline)
    for row in candidate['rows']:
        if 'dessert' in row['expected_labels']:
            row['raw_output']=json.dumps({'labels':['unclear'],'explanation':'Deliberate test regression.'})
            row.update(inspect_response(row['raw_output'],'stop'))
    candidate['summary']=summarize(load_benchmark().cases,candidate['rows'])
    config=GateConfig(min_macro_f1=0,min_json_validity=0,min_schema_validity=0,
                      min_shortcut_accuracy=0,important_labels=['dessert'])
    gate=evaluate_gate(candidate,shortcut,baseline,config)
    assert not gate['passed'] and any('dessert F1 regressed' in x for x in gate['failures'])


def test_deployment_selection_rejects_nonfinite_or_missing_measurements():
    row={'name':'invalid','micro_f1':.8,'schema_validity':1.,'peak_ram_mib':1000.,'p95_ms':float('nan'),'size_mib':100.}
    assert select_candidate([row])['recommended'] is None
