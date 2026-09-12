import copy
import json
import pytest
from recipetriage_ml.evaluation import failures,shortcut_tests,red_team
from recipetriage_ml.evaluation.benchmark import RunConfig,new_run,run_benchmark,descriptor
from recipetriage_ml.inference.contracts import Generation


class Constant:
    def __init__(self,raw='{"labels":["weeknight-30min"],"explanation":"Fixture."}'):self.raw=raw
    def generate(self,messages,*args):return Generation(raw_output=self.raw,model='test',finish_reason='stop')


def test_failure_taxonomy_reparses_raw_and_separates_format_from_semantics():
    run=run_benchmark(RunConfig(provider='hf-base'),provider_instance=Constant('{"explanation":"Missing labels"}'))
    # Stored flags cannot mask invalid schema; analysis derives from raw output.
    run['rows'][0]['usable']=True
    result=failures.analyze(run)
    assert result['category_counts']=={'schema_violation':10}
    assert result['misleading_title_accuracy']==0 and result['misleading_title_total']==1
    assert result['failed_cases']==10
    assert result['collection_priorities'][0]['topic']=='Complete structured answers'
    assert all(row['predicted_labels'] is None for row in result['findings'])


def test_semantic_errors_include_evidence_and_missing_results_are_not_correct():
    run=run_benchmark(RunConfig(provider='hf-base'),provider_instance=Constant())
    report=failures.analyze(run)
    assert report['findings'][0]['exact_match']
    assert 'time_reasoning' in report['findings'][2]['failure_categories']
    assert 'equipment_reasoning' in report['findings'][3]['failure_categories']
    assert 'multi_label_omission' in report['findings'][3]['failure_categories']
    del run['rows'][-1]
    assert 'missing_result' in failures.analyze(run)['findings'][-1]['failure_categories']
    run['rows'].append(copy.deepcopy(run['rows'][0]))
    with pytest.raises(ValueError,match='Duplicate'):failures.analyze(run)


@pytest.mark.parametrize('field,value',[('time_minutes',10),('ingredients',['changed']),('equipment',[]),('instructions',['changed']),('pantry_items',['flour'])])
def test_counterfactual_cannot_change_body(field,value):
    group=shortcut_tests.suite()['groups'][0];changed=copy.deepcopy(group['variants'][1]);changed[field]=value
    with pytest.raises(ValueError,match='body'):shortcut_tests.assert_title_only(group['reference'],changed,group['stem'])


def test_adjective_only_changes_and_denominators():
    spec=shortcut_tests.suite()
    assert len(spec['groups'])==3 and spec['expected_rows']==12
    for group in spec['groups']:
        for variant in group['variants']:
            shortcut_tests.assert_title_only(group['reference'],variant,group['stem'])
        changed=copy.deepcopy(group['variants'][0]);changed['title']='Very Quick '+group['stem']
        with pytest.raises(ValueError,match='adjective'):shortcut_tests.assert_title_only(group['reference'],changed,group['stem'])
    class Flips:
        def generate(self,messages,*args):
            recipe=json.loads(messages[1]['content'])
            adjective=recipe['title'].split()[0]
            if adjective=='Simple':return Generation(raw_output='invalid',model='test',finish_reason='length')
            labels=['weekend-project'] if adjective=='Traditional' else ['weeknight-30min']
            return Generation(raw_output=json.dumps({'labels':labels,'explanation':'Controlled test response.'}),model='test',finish_reason='stop')
    result=shortcut_tests.run_suite(Flips(),descriptor('hf-base'))
    score=result['summary'];assert score['prediction_flip_rate']==1 and score['eligible_pairs']==6
    assert score['excluded_pairs']==3 and score['pair_coverage']==2/3
    assert score['misleading_correct']==0 and score['misleading_total']==3
    assert result['suite_sha256']==shortcut_tests.digest(spec)
    invalid=shortcut_tests.run_suite(Constant('bad JSON'),descriptor('hf-base'))
    assert invalid['summary']['prediction_flip_rate'] is None
    assert invalid['summary']['eligible_pairs']==0 and invalid['summary']['excluded_pairs']==9
    assert invalid['summary']['exact_set_accuracy']==0


def test_red_team_is_separate_and_does_not_call_every_format_failure_injection_success():
    result=red_team.run_suite(Constant('malformed'),descriptor('hf-base'))
    assert result['passed']==0 and result['total']==3
    assert all(not row['explicit_instruction_following'] for row in result['rows'])
    attacked=red_team.run_suite(Constant('HACKED'),descriptor('hf-base'))
    assert all(row['explicit_instruction_following'] for row in attacked['rows'])
    assert 'NOT title-adjective' in result['suite']['note']
