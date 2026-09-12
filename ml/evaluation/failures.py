"""Explain benchmark failures without treating malformed JSON as a semantic judgment."""
from collections import Counter
from .benchmark import load_benchmark, manifest
from .metrics import inspect_response
from recipetriage_ml.data.pipeline import digest

VERSION = 'failure-analysis-v1'
CATEGORY_DESCRIPTIONS = {
    'missing_result': 'No recorded result for this benchmark case.',
    'provider_error': 'The provider did not return a response.',
    'invalid_json': 'The complete response is not syntactically valid JSON.',
    'schema_violation': 'JSON violates the required labels/explanation schema.',
    'incomplete_generation': 'Generation did not finish normally.',
    'false_positive': 'A usable response included an unsupported label.',
    'false_negative': 'A usable response omitted an expected label.',
    'time_reasoning': 'The predicted time-related label differs from the answer key.',
    'equipment_reasoning': 'The required-equipment label differs from the answer key.',
    'pantry_reasoning': 'The pantry-coverage label differs from the answer key.',
    'dessert_reasoning': 'The dessert label differs from the answer key.',
    'ambiguity_handling': 'The unclear decision differs from the answer key.',
    'multi_label_omission': 'A usable answer omitted part of a multi-label set.',
}


def analyze(run):
    benchmark = load_benchmark()
    if run.get('benchmark', {}).get('benchmark_sha256') != manifest()['benchmark_sha256']:
        raise ValueError('Failure analysis requires the unchanged RecipeTriage-Bench-v1')
    supplied = run.get('rows', [])
    known = {case.recipe.id for case in benchmark.cases}
    if len({row['case_id'] for row in supplied}) != len(supplied) or any(row['case_id'] not in known for row in supplied):
        raise ValueError('Duplicate or unknown benchmark result')
    by_id={row['case_id']:row for row in supplied}; findings=[]
    for case in benchmark.cases:
        row=by_id.get(case.recipe.id); categories=[]; inspected=None; guess=[]
        if row is None: categories.append('missing_result')
        elif row.get('raw_output') is None: categories.append('provider_error')
        else:
            inspected=inspect_response(row['raw_output'],row.get('finish_reason'))
            if not inspected['json_valid']:categories.append('invalid_json')
            elif not inspected['schema_valid']:categories.append('schema_violation')
            if row.get('finish_reason')!='stop':categories.append('incomplete_generation')
            if inspected['usable']:guess=inspected['prediction']['labels']
        usable=bool(inspected and inspected['usable']); truth=set(case.labels); predicted=set(guess)
        missing=sorted(truth-predicted);extra=sorted(predicted-truth)
        if usable:
            if extra:categories.append('false_positive')
            if missing:categories.append('false_negative')
            mismatches=truth^predicted
            if mismatches & {'weeknight-30min','weekend-project'}:categories.append('time_reasoning')
            for label,category in [('needs-special-equipment','equipment_reasoning'),('have-most-of-this','pantry_reasoning'),('dessert','dessert_reasoning'),('unclear','ambiguity_handling')]:
                if label in mismatches:categories.append(category)
            if len(truth)>1 and missing:categories.append('multi_label_omission')
        findings.append({'case_id':case.recipe.id,'title':case.recipe.title,'benchmark_category':case.category,
            'expected_labels':case.labels,'predicted_labels':guess if usable else None,'usable':usable,
            'exact_match':usable and truth==predicted,'missing_labels':missing,'extra_labels':extra,
            'failure_categories':categories,'error':inspected['error'] if inspected else (row or {}).get('error','Missing result'),
            'raw_output':row.get('raw_output') if row else None,'time_minutes':case.recipe.time_minutes,
            'equipment':case.recipe.equipment,'source_uri':case.recipe.source_uri})
    misleading=[row for row in findings if row['benchmark_category']=='misleading-title']
    result={'version':VERSION,'source_run_id':run['run_id'],'source_sha256':digest(run),
        'model_config':run['model_config'],'benchmark_sha256':manifest()['benchmark_sha256'],
        'category_counts':dict(Counter(category for row in findings for category in row['failure_categories'])),
        'failed_cases':sum(not row['exact_match'] for row in findings),'total_cases':len(findings),
        'misleading_title_accuracy':sum(row['exact_match'] for row in misleading)/len(misleading),
        'misleading_title_correct':sum(row['exact_match'] for row in misleading),'misleading_title_total':len(misleading),
        'findings':findings,'category_descriptions':CATEGORY_DESCRIPTIONS,
        'limitations':'Categories can overlap. Semantic diagnoses require a usable answer. The provisional answer key is not a verified ground truth; title correlation alone does not establish a shortcut.'}
    result['collection_priorities']=collection_priorities(result)
    return result


def collection_priorities(analysis, shortcut=None):
    counts=analysis['category_counts']; findings=analysis['findings']; plans=[]
    rules=[
      (['invalid_json','schema_violation','incomplete_generation'],'Complete structured answers',
       'Collect human-reviewed recipe/answer pairs with both labels and a concise explanation, including long and short outputs. Check the output-token budget before attributing truncation to missing data.'),
      (['time_reasoning'],'Elapsed time and active work',
       'Collect source-linked recipes around the 30-minute boundary, with separate prep/cook/rest/cooling evidence. Include quick-sounding titles on long recipes and slow-sounding titles on short recipes.'),
      (['equipment_reasoning'],'Required versus optional equipment',
       'Collect matched natural examples with required appliances, optional appliance alternatives and ordinary tools. Annotate the selected preparation method.'),
      (['multi_label_omission'],'Complete multi-label combinations',
       'Collect recipes with two or three independently supported labels. Ask reviewers to justify every positive label and plausible exclusion.'),
      (['pantry_reasoning'],'Explicit pantry coverage',
       'Collect consented pantry contexts covering missing, below-threshold and above-threshold inventories; do not infer the user pantry.'),
      (['dessert_reasoning'],'Serving-context contrasts',
       'Collect savory pancakes, breakfast drinks and actual desserts with explicit serving context, rather than using dish-name keywords as labels.'),
      (['ambiguity_handling'],'Missing and conflicting evidence',
       'Collect recipes with unknown total time, unspecified required steps and supported labels despite some missing fields. Adjudicate when unclear must be used alone.'),
    ]
    for categories,title,action in rules:
        affected=[row['case_id'] for row in findings if set(row['failure_categories']) & set(categories)]
        if affected:plans.append({'topic':title,'observed_cases':len(affected),'case_ids':affected,'action':action,'basis':'observed benchmark failures'})
    plans.sort(key=lambda row:(-row['observed_cases'],row['topic']))
    if shortcut and shortcut['summary']['flipped_pairs']:
        plans.insert(0,{'topic':'Title adjective invariance','observed_cases':shortcut['summary']['flipped_pairs'],
            'case_ids':sorted({p['case_id'] for p in shortcut['pairs'] if p['flipped'] is True}),
            'action':'Collect new human-reviewed title counterfactual groups with identical recipe bodies; keep every group in one split. Use new recipe sources so this evaluated suite stays held out.',
            'basis':'observed usable prediction flips; pair count, not independent recipes'})
    # Structural coverage gaps remain useful even when malformed outputs prevent diagnosis.
    plans.append({'topic':'Label coverage and annotation review','observed_cases':None,'case_ids':[],
        'action':'Independently review the provisional seed labels, then collect new training examples for dessert and have-most-of-this, which are absent from the current five-example training split. Expand validation beyond one recipe.',
        'basis':'known teaching-dataset coverage limitation, not a measured causal explanation'})
    return [{'priority':i+1,**row,'split_rule':'New sources only; exclude existing benchmark/shortcut/red-team cases and group related recipes before splitting.'} for i,row in enumerate(plans)]
