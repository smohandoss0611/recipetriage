"""Title-adjective counterfactuals with exact recipe-body invariance checks."""
import json
from itertools import combinations
from pathlib import Path
import time
from uuid import uuid4
from .benchmark import load_benchmark, manifest, descriptor
from .metrics import inspect_response
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.inference.contracts import Recipe
from recipetriage_ml.inference.prompts import build_messages, PROMPT_VERSION
from recipetriage_ml.inference.experiment import fixture

ADJECTIVES=('Traditional','Easy','Quick','Simple')
VERSION='shortcut-title-adjectives-v1'


def suite():
    cases={case.recipe.id:case for case in load_benchmark().cases}
    groups=[]
    for case_id,stem in [('bench-003','Brown Bread'),('bench-010','Turnip Pancakes')]:
        case=cases[case_id]
        groups.append({'case_id':case_id,'stem':stem,'recipe':case.recipe.inference_recipe().model_dump(),
                       'expected_labels':case.labels,'source_uri':case.recipe.source_uri})
    ravioli=fixture()
    groups.append({'case_id':'ravioli-title-fixture','stem':'Ravioli','recipe':ravioli['recipe'],
                   'expected_labels':ravioli['expected_labels'],'source_uri':ravioli['source_url']})
    for group in groups:
        reference=Recipe.model_validate({**group['recipe'],'title':'Traditional '+group['stem']})
        group['reference']=reference.model_dump()
        group['variants']=[Recipe.model_validate({**group['recipe'],'title':adj+' '+group['stem']}).model_dump() for adj in ADJECTIVES]
        for variant in group['variants']:assert_title_only(reference.model_dump(),variant,group['stem'])
        group['body_sha256']=digest(reference.model_dump(exclude={'title'}))
    return {'version':VERSION,'adjectives':list(ADJECTIVES),'reference_adjective':'Traditional','groups':groups,
            'expected_rows':len(groups)*len(ADJECTIVES),'annotation_status':'provisional-unreviewed',
            'benchmark_sha256':manifest()['benchmark_sha256'],'prompt_version':PROMPT_VERSION,
            'temperature':0,'max_new_tokens':128,'training_allowed':False}


def assert_title_only(reference, candidate, stem):
    left=Recipe.model_validate(reference).model_dump();right=Recipe.model_validate(candidate).model_dump()
    if left['title']!='Traditional '+stem or right['title'] not in {adj+' '+stem for adj in ADJECTIVES}:
        raise ValueError('Counterfactual may change only the declared title adjective')
    if ({k:v for k,v in reference.items() if k!='title'} != {k:v for k,v in candidate.items() if k!='title'}
        or {k:v for k,v in left.items() if k!='title'} != {k:v for k,v in right.items() if k!='title'}):
        raise ValueError('Counterfactual changed recipe body: time, ingredients, equipment, instructions and pantry must stay identical')


def summarize(result):
    spec=result['suite']; rows=result['rows'];by_key={}
    expected={(g['case_id'],adj) for g in spec['groups'] for adj in ADJECTIVES}
    for row in rows:
        key=(row['case_id'],row['adjective'])
        if key not in expected or key in by_key:raise ValueError('Unknown or duplicate shortcut row')
        by_key[key]=row
    pairs=[];correct=0;misleading_correct=0;misleading_total=0;usable_count=0
    for group in spec['groups']:
        for adj in ADJECTIVES:
            row=by_key.get((group['case_id'],adj))
            if row:
                assert_title_only(group['reference'],row['recipe'],group['stem'])
                if row['recipe']['title']!=adj+' '+group['stem']:raise ValueError('Row adjective/title mismatch')
            usable=bool(row and row['usable']);usable_count+=usable
            exact=usable and set(row['prediction']['labels'])==set(group['expected_labels'])
            correct+=exact
            # Easy/Simple suggest ease, not a precise duration. Quick on these
            # 70/75/80-minute cases is the predeclared misleading-time subset.
            if adj=='Quick':misleading_total+=1;misleading_correct+=exact
        baseline=by_key.get((group['case_id'],'Traditional'))
        for adj in ADJECTIVES[1:]:
            changed=by_key.get((group['case_id'],adj));eligible=bool(baseline and changed and baseline['usable'] and changed['usable'])
            pairs.append({'case_id':group['case_id'],'reference':'Traditional','adjective':adj,'eligible':eligible,
                          'flipped':set(baseline['prediction']['labels'])!=set(changed['prediction']['labels']) if eligible else None,
                          'body_sha256':group['body_sha256']})
    valid=[p for p in pairs if p['eligible']];flips=sum(p['flipped'] for p in valid)
    return {'pairs':pairs,'summary':{'planned_rows':len(expected),'recorded_rows':len(rows),'usable_rows':usable_count,
        'planned_pairs':len(pairs),'eligible_pairs':len(valid),'excluded_pairs':len(pairs)-len(valid),'flipped_pairs':flips,
        'prediction_flip_rate':flips/len(valid) if valid else None,'pair_coverage':len(valid)/len(pairs),
        'exact_set_accuracy':correct/len(expected),'correct_rows':correct,
        'misleading_title_accuracy':misleading_correct/misleading_total,'misleading_correct':misleading_correct,'misleading_total':misleading_total,
        'accuracy_policy':'Exact label-set accuracy over all planned cases; invalid/missing answers are incorrect.',
        'flip_policy':'Usable reference/variant pairs only; exclusions and coverage reported. Invalid-to-invalid is not a stable prediction.'}}


def run_suite(provider, identity, on_update=None):
    spec=suite();result={'run_id':str(uuid4()),'status':'running','suite':spec,'suite_sha256':digest(spec),
        'model_config':identity,'rows':[],'error':None,'limitations':'Three source recipes; one generation per title. No statistical generalization or causal shortcut claim. Only title adjectives vary.'}
    if hasattr(provider,'prepare'):provider.prepare()
    for group in spec['groups']:
        for adjective,variant in zip(ADJECTIVES,group['variants']):
            assert_title_only(group['reference'],variant,group['stem'])
            row={'case_id':group['case_id'],'adjective':adjective,'recipe':variant,'body_sha256':group['body_sha256'],
                 'expected_labels':group['expected_labels'],'raw_output':None,'finish_reason':None,'prediction':None,'usable':False,'error':None}
            messages=build_messages(Recipe.model_validate(variant));row['messages']=messages;t=time.perf_counter()
            try:
                output=provider.generate(messages,0,128);row.update(output.model_dump());row.update(inspect_response(output.raw_output,output.finish_reason))
            except Exception as exc:
                import logging;logging.getLogger(__name__).exception('Shortcut provider failed')
                row['error']=f'{type(exc).__name__}: provider failed; inspect log'
            row['latency_ms']=(time.perf_counter()-t)*1000;result['rows'].append(row);result.update(summarize(result))
            if on_update:on_update(result)
    result['status']='completed' if all(row['raw_output'] is not None for row in result['rows']) else 'partial'
    if on_update:on_update(result)
    return result


def main():
    import argparse,os
    from dotenv import load_dotenv
    from .base_provider import BaseProvider
    from recipetriage_ml.training.sft import load_adapter
    load_dotenv('.env');os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--adapter',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    provider=load_adapter(args.adapter) if args.adapter else BaseProvider()
    identity=provider.identity if args.adapter else descriptor('hf-base')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(result):
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(args.output)
    result=run_suite(provider,identity,save);print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
