"""Revalidate versioned data, audit overlap, and mask everything except answers."""
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from recipetriage_ml.data.pipeline import build_dataset, chat_messages, digest, fingerprint, source_key
from recipetriage_ml.data.schemas import TrainingExample
from recipetriage_ml.evaluation.benchmark import load_benchmark
from recipetriage_ml.evaluation.base_provider import render_prompt


def seed_snapshot():
    return build_dataset(json.loads(files('recipetriage_ml').joinpath('data/seed.json').read_text()))


def load_snapshot(directory):
    path = Path(directory)
    metadata = json.loads((path / 'metadata.json').read_text())
    payload = {'metadata': metadata, 'files': {name: (path / name).read_text() for name in metadata['file_sha256']}}
    verify_snapshot(payload)
    return payload


def verify_snapshot(payload):
    """A version label alone is insufficient: rebuild from raw data and check bytes."""
    meta = payload['metadata']
    if meta.get('split_method') == 'frozen-holdouts-approved-v1':
        from recipetriage_ml.data.approved import build_approved
        parent = json.loads(payload['files']['parent_snapshot.json'])
        verify_snapshot(parent)
        rebuilt = build_approved(parent, json.loads(payload['files']['raw.json']))
    else:
        rebuilt = build_dataset(json.loads(payload['files']['raw.json']), meta['seed'], meta['requested_ratios'])
    if rebuilt['metadata']['version'] != meta['version'] or rebuilt['files'] != payload['files']:
        raise ValueError('Dataset content does not match its immutable version')
    if 'examples' in payload and payload['examples']!=rebuilt['examples']:
        raise ValueError('Dataset example view differs from immutable export files')
    for key, value in rebuilt['metadata'].items():
        if key != 'created_at' and meta.get(key) != value:
            raise ValueError(f'Dataset metadata mismatch: {key}')
    splits = {name: [TrainingExample.model_validate_json(line) for line in payload['files'][name + '.jsonl'].splitlines()]
              for name in ['train', 'validation', 'test']}
    benchmark = load_benchmark()
    for name, examples in splits.items():
        for example in examples:
            for case in benchmark.cases:
                if (example.recipe.id == case.recipe.id or fingerprint(example) == fingerprint(case)
                    or source_key(example.recipe) == source_key(case.recipe)
                    or (example.recipe.group_id and example.recipe.group_id == case.recipe.group_id)):
                    raise ValueError(f'Benchmark leakage: {name}/{example.recipe.id} overlaps {case.recipe.id}')
    return splits


def prepare_snapshot(payload, directory):
    splits = verify_snapshot(payload)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    logical_hashes = {}
    for name in ['train', 'validation']:
        messages = [chat_messages(example) for example in splits[name]]
        logical_hashes[name] = digest(messages)
        # Same logical messages for both providers. Test examples are never exported for training.
        (target / f'{name}.chat.jsonl').write_text(payload['files'][f'{name}.chat.jsonl'])
        managed = [{'messages': [{**m, 'weight': int(m['role'] == 'assistant')} for m in row]} for row in messages]
        (target / f'{name}.fireworks.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in managed))
    return splits, {'version': payload['metadata']['version'], 'counts': payload['metadata']['counts'],
                    'split_ids': payload['metadata']['split_ids'], 'logical_messages_sha256': logical_hashes,
                    'status': payload['metadata']['status'], 'warnings': payload['metadata']['warnings'],
                    'contamination_audit': {'source_body_group_overlap': [], 'paraphrases': 'not audited', 'pretraining': 'unknown'},
                    'test_used_for_training_or_selection': False}


def encode_example(example, tokenizer, sequence_length):
    messages = chat_messages(example)
    prompt = render_prompt(messages[:-1])
    answer = messages[-1]['content']
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    # Fail instead of silently masking the wrong boundary if a tokenizer merges across it.
    if ids[:len(prefix)] != prefix:
        raise ValueError('Tokenizer merged across prompt/answer boundary; formatter needs an explicit boundary')
    if tokenizer.eos_token_id is None:
        raise ValueError('A terminal EOS token is required')
    ids.append(tokenizer.eos_token_id)
    if len(ids) > sequence_length:
        raise ValueError(f'{example.recipe.id}: {len(ids)} tokens exceeds sequence_length={sequence_length}; no truncation performed')
    labels = [-100] * len(prefix) + ids[len(prefix):]
    if sum(label != -100 for label in labels) < 2:
        raise ValueError('Assistant answer must contain supervised tokens as well as EOS')
    return {'input_ids': ids, 'labels': labels}


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()
