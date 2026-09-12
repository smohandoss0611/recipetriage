"""Build the next version only from reviewed records; retain original holdouts."""
import hashlib
import json
from .pipeline import build_dataset, canonical, digest, export_jsonl, fingerprint, group_examples
from .schemas import TrainingExample

METHOD = 'frozen-holdouts-approved-v1'


def build_approved(parent, records):
    rows = [TrainingExample.model_validate(x) for x in records]
    if not rows or any(not x.reviewed or not x.reviewed_by for x in rows):
        raise ValueError('Every record in the next version needs explicit human review, including the legacy seed')
    if not any(x.recipe.source_type == 'synthetic' for x in rows):
        raise ValueError('Approve at least one new synthetic example before creating this augmented version')
    prior = {x['recipe']['id']: TrainingExample.model_validate(x) for x in parent['examples']}
    current = {x.recipe.id: x for x in rows}
    if len(current) != len(rows) or not prior.keys() <= current.keys():
        raise ValueError('Review every parent record; parent recipes cannot disappear or repeat')
    for key, old in prior.items():
        new = current[key]
        if new.recipe != old.recipe:
            raise ValueError('Parent recipe bodies and source metadata are frozen in this augmentation')
        if key in parent['metadata']['split_ids']['validation'] + parent['metadata']['split_ids']['test']:
            if new.labels != old.labels or new.rationale != old.rationale:
                raise ValueError('Holdout labels/rationales changed; create a separately reviewed benchmark protocol instead')
    base = build_dataset([x.model_dump() for x in rows])
    if base['metadata']['duplicates_removed']:
        raise ValueError('Duplicate approved recipes must be reconciled before version creation')
    ids = parent['metadata']['split_ids']
    holdout = set(ids['validation'] + ids['test'])
    for group in group_examples(rows):
        if any(x.recipe.id in holdout for x in group) and any(x.recipe.id not in holdout for x in group):
            raise ValueError('An approved example overlaps a frozen holdout group')
    splits = {name: [current[key] for key in ids[name]] for name in ['validation', 'test']}
    splits['train'] = sorted([x for x in rows if x.recipe.id not in holdout], key=lambda x: x.recipe.id)
    files = {}
    for name, examples in splits.items():
        files[name+'.jsonl'] = export_jsonl(examples)
        files[name+'.chat.jsonl'] = export_jsonl(examples, chat=True)
    serial = [x.model_dump() for x in sorted(rows, key=lambda x:x.recipe.id)]
    frozen_parent=json.loads(canonical(parent))
    frozen_parent['metadata'].pop('created_at',None)  # volatile creation time is not dataset identity
    files.update({'raw.json': canonical(serial)+'\n', 'examples.json': json.dumps(serial, indent=2)+'\n',
                  'parent_snapshot.json': canonical(frozen_parent)+'\n'})
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
    content = digest({'method': METHOD, 'parent_version': parent['metadata']['version'], 'files': hashes})
    meta = {**base['metadata'], 'version': 'v1-'+content, 'content_sha256': content,
            'split_method': METHOD, 'parent_version': parent['metadata']['version'],
            'file_sha256': hashes, 'counts': {name:len(x) for name,x in splits.items()},
            'split_ids': {name:[x.recipe.id for x in values] for name,values in splits.items()},
            'actual_ratios': {name:len(x)/len(rows) for name,x in splits.items()},
            'label_distribution': {name:{label:sum(label in x.labels for x in values) for label in base['metadata']['label_policy']} for name,values in splits.items()},
            'review_policy': 'all records explicitly human reviewed; synthetic content bound to approval hash',
            'holdouts_frozen': True}
    return {'metadata':meta, 'examples':serial, 'files':files}
