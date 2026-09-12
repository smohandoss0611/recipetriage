"""Measured label/format/uncertainty/shortcut behavior, with explicit denominators."""
from .benchmark import load_benchmark


def report(benchmark,shortcut):
    summary=benchmark['summary'];unclear=summary['labels']['per_label']['unclear']
    false_unclear=sum(bool(row['usable'] and row['prediction']['labels']==['unclear'] and row['expected_labels']!=['unclear']) for row in benchmark['rows'])
    shortcuts=shortcut.get('summary') if shortcut else None
    return {'macro_f1':summary['labels']['macro']['f1'],'micro_f1':summary['labels']['micro']['f1'],
            'exact_match':summary['labels']['exact_match'],'json_validity':summary['json_valid_count']/summary['expected_cases'],
            'schema_validity':summary['schema_valid_count']/summary['expected_cases'],
            'per_label':summary['labels']['per_label'],'unclear':unclear,'false_unclear_cases':false_unclear,
            'shortcut':shortcuts,'annotation_status':'provisional; accuracy is agreement with the current answer key'}


def compare(before,after):
    delta={k:after[k]-before[k] for k in ['macro_f1','micro_f1','exact_match','json_validity','schema_validity']}
    a,b=before.get('shortcut'),after.get('shortcut')
    if a and b:delta['misleading_title_accuracy']=b['misleading_title_accuracy']-a['misleading_title_accuracy']
    return {'before':before,'after':after,'delta':delta,
            'claim':'Descriptive fixed-benchmark comparison. Reward or loss improvement alone is not task improvement.'}
