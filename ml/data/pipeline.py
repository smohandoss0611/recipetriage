"""Pure, deterministic dataset engineering. No model/network/database needed."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from urllib.parse import urlsplit, urlunsplit
from pydantic import ValidationError
from recipetriage_ml.inference.prompts import build_messages, PROMPT_VERSION, SYSTEM_PROMPT
from .schemas import LABELS, POLICY, Recipe, TrainingExample, clean

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=invalid)


def load_raw(text, format="json"):
    if len(text.encode()) > 2_000_000:
        raise ValueError("Dataset v1 accepts at most 2 MB of raw text")
    if format == "json":
        records = _json(text)
        if not isinstance(records, list):
            raise ValueError("JSON input must be an array of records")
    elif format == "jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(_json(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc
    else:
        raise ValueError("format must be json or jsonl")
    if len(records) > 1000:
        raise ValueError("Dataset v1 accepts at most 1000 records")
    return records


def load_recipes(text, format="json"):
    """Unlabeled recipe loader; does not invent target labels."""
    return [Recipe.model_validate(row) for row in load_raw(text, format)]


def as_drafts(records):
    return [{"recipe": row, "labels": [], "rationale": "", "annotation_notes": "", "reviewed": False, "reviewed_by": None}
            if isinstance(row, dict) and "title" in row and "recipe" not in row else row for row in records]


def fingerprint(example):
    # Ignore title, time, labels, IDs and source. Renamed copies must not leak.
    recipe = example.recipe
    return digest({"ingredients": sorted(clean(x).casefold() for x in recipe.ingredients),
                   "instructions": [clean(x).casefold() for x in recipe.instructions]})


def source_key(recipe):
    uri = urlsplit(recipe.source_uri)
    # Group all query/fragment variants conservatively; may over-group URL-based catalogs.
    return urlunsplit((uri.scheme.lower(), uri.netloc.lower(), uri.path.rstrip('/'), '', ''))


def validate_clean(records):
    candidates, issues = [], []
    for row, raw in enumerate(records, 1):
        try:
            candidates.append((row, TrainingExample.model_validate(raw)))
        except ValidationError as exc:
            issues.append({"row": row, "kind": "invalid", "detail": str(exc)})
    candidates.sort(key=lambda pair: (pair[1].recipe.id, canonical(pair[1].model_dump())))
    accepted, ids, bodies, duplicates = [], {}, {}, []
    for row, item in candidates:
        key = fingerprint(item)
        if item.recipe.id in ids and ids[item.recipe.id] != key:
            issues.append({"row": row, "kind": "id_conflict", "detail": f"ID {item.recipe.id} identifies different recipes"})
            continue
        ids[item.recipe.id] = key
        if key in bodies:
            previous = bodies[key]
            signature = lambda x: (x.labels, x.rationale, x.reviewed, x.reviewed_by, x.recipe.time_minutes, x.recipe.equipment, x.recipe.pantry_items, x.recipe.group_id)
            if signature(item) != signature(previous):
                issues.append({"row": row, "kind": "duplicate_conflict", "detail": f"Conflicting duplicate of {previous.recipe.id}; reconcile annotation and recipe metadata first"})
            else:
                duplicates.append({"removed": item.recipe.id, "kept": previous.recipe.id,
                                   "source_uri": item.recipe.source_uri, "annotation_notes": item.annotation_notes})
            continue
        bodies[key] = item
        accepted.append(item)
    return accepted, issues, duplicates


def group_examples(examples, duplicate_aliases=()):
    parent = list(range(len(examples)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys = {}
    for i, example in enumerate(examples):
        tokens = ['body:'+fingerprint(example), 'source:'+source_key(example.recipe)]
        if example.recipe.group_id:
            tokens.append('group:'+example.recipe.group_id)
        # A removed copy can connect its survivor to another source family.
        for alias in duplicate_aliases:
            if alias['kept'] == example.recipe.id:
                recipe = example.recipe.model_copy(update={"source_uri": alias['source_uri']})
                tokens.append('source:'+source_key(recipe))
        for key in tokens:
            if key in keys:
                parent[find(i)] = find(keys[key])
            else:
                keys[key] = i
    groups = {}
    for i, example in enumerate(examples):
        groups.setdefault(find(i), []).append(example)
    return [sorted(values, key=lambda x: x.recipe.id) for values in groups.values()]


def split_examples(examples, seed=42, ratios=(0.7, 0.15, 0.15), duplicate_aliases=()):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError("seed must be an integer from 0 to 4294967295")
    if len(ratios) != 3 or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) or x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-9):
        raise ValueError("Three positive finite split ratios must sum to 1")
    if len({fingerprint(x) for x in examples}) != len(examples) or len({x.recipe.id for x in examples}) != len(examples):
        raise ValueError("De-duplicate recipe IDs and bodies before splitting")
    groups = group_examples(examples, duplicate_aliases)
    if len(groups) < 3:
        raise ValueError("At least three independent recipe groups are required for three nonempty splits")
    names = ["train", "validation", "test"]
    n = len(examples)
    targets = [int(n*r) for r in ratios]
    for i in sorted(range(3), key=lambda i: (-(n*ratios[i]-targets[i]), i))[:n-sum(targets)]:
        targets[i] += 1
    for i in range(3):
        if targets[i] == 0:
            donor = max(range(3), key=lambda j: targets[j])
            targets[donor] -= 1
            targets[i] += 1
    total = Counter(label for x in examples for label in x.labels)
    key = lambda group: digest([seed, sorted(fingerprint(x) for x in group)])
    ordered = sorted(groups, key=lambda g: (min(total[l] for x in g for l in x.labels), -len(g), key(g)))
    results = {name: [] for name in names}
    counts = [Counter() for _ in names]
    for position, group in enumerate(ordered):
        empty = [i for i,name in enumerate(names) if not results[name]]
        if len(ordered)-position == len(empty):
            candidates = empty
        else:
            candidates = [i for i,name in enumerate(names) if len(results[name])+len(group) <= targets[i]] or list(range(3))
        labels = Counter(label for x in group for label in x.labels)
        def score(i):
            # Encourage label coverage with limited capacity; groups stay atomic.
            deficit = sum((total[l]*ratios[i]-counts[i][l])*amount/total[l] for l,amount in labels.items())
            overflow = max(0, len(results[names[i]])+len(group)-targets[i])
            remaining = (targets[i]-len(results[names[i]]))/max(1,targets[i])
            return (deficit-overflow, remaining, digest([key(group), names[i]]))
        selected = max(candidates, key=score)
        results[names[selected]].extend(group)
        counts[selected].update(labels)
    for values in results.values():
        values.sort(key=lambda x: x.recipe.id)
    return results


def chat_messages(example):
    return build_messages(example.recipe.inference_recipe()) + [{"role": "assistant", "content": canonical({"labels":example.labels,"explanation":example.rationale})}]


def render_chat(example, tokenizer):
    # Complete supervised conversation: no extra empty assistant-generation marker.
    return tokenizer.apply_chat_template(chat_messages(example), tokenize=False, add_generation_prompt=False)


def export_jsonl(examples, chat=False):
    return ''.join(canonical({"messages":chat_messages(x)} if chat else x.model_dump())+'\n' for x in examples)


def build_dataset(records, seed=42, ratios=(0.7,0.15,0.15)):
    records = as_drafts(records)
    examples, issues, duplicates = validate_clean(records)
    if issues:
        raise ValueError(canonical({"issues":issues,"duplicates":duplicates}))
    splits = split_examples(examples, seed, ratios, duplicates)
    distributions = {name:{label:sum(label in x.labels for x in rows) for label in LABELS} for name,rows in splits.items()}
    warnings = ["Teaching seed only: too small for reliable model-quality estimates.",
                "Source grouping and exact-body de-duplication do not detect every paraphrase. Assign group_id to known related recipes."]
    for label in LABELS:
        total = sum(counts[label] for counts in distributions.values())
        if total < 3:
            warnings.append(f"{label}: {total} examples; cannot cover all three splits.")
        elif any(counts[label]==0 for counts in distributions.values()):
            warnings.append(f"{label}: missing from at least one split; stratification is approximate.")
    if any(not x.reviewed for x in examples):
        warnings.append("Unreviewed annotations: this export is a teaching draft, not approved training data.")
    files = {}
    for name, rows in splits.items():
        files[name+'.jsonl'] = export_jsonl(rows)
        files[name+'.chat.jsonl'] = export_jsonl(rows, chat=True)
    files['examples.json'] = json.dumps([x.model_dump() for x in examples],indent=2,ensure_ascii=False)+'\n'
    files['raw.json'] = json.dumps(sorted(records,key=canonical),indent=2,ensure_ascii=False)+'\n'
    input_hash = digest(sorted(records,key=canonical))
    identity = {"pipeline":PIPELINE_VERSION,"schema":SCHEMA_VERSION,"prompt":PROMPT_VERSION,
                "system_prompt":SYSTEM_PROMPT,"label_policy":POLICY,"seed":seed,"ratios":list(ratios),
                "input_sha256":input_hash,"file_hashes":{name:hashlib.sha256(content.encode()).hexdigest() for name,content in files.items()}}
    content_hash = digest(identity)
    metadata = {"version":"v1-"+content_hash, "content_sha256":content_hash,
                "created_at":datetime.now(timezone.utc).isoformat(), "schema_version":SCHEMA_VERSION,
                "pipeline_version":PIPELINE_VERSION,"prompt_version":PROMPT_VERSION,"system_prompt_sha256":digest(SYSTEM_PROMPT),
                "status":"reviewed" if all(x.reviewed for x in examples) else "teaching-draft",
                "seed":seed,"requested_ratios":list(ratios),"actual_ratios":{name:len(rows)/len(examples) for name,rows in splits.items()},
                "split_method":"grouped-rare-label-greedy-v1","label_policy":POLICY,
                "counts":{name:len(rows) for name,rows in splits.items()},"label_distribution":distributions,
                "split_ids":{name:[x.recipe.id for x in rows] for name,rows in splits.items()},
                "groups":[[x.recipe.id for x in group] for group in sorted(group_examples(examples,duplicates),key=lambda g:g[0].recipe.id)],
                "fingerprints":{x.recipe.id:fingerprint(x) for x in examples},
                "sources":sorted({x.recipe.source_uri for x in examples} | {x['source_uri'] for x in duplicates}),
                "input_sha256":input_hash,"input_rows":len(records),"duplicates_removed":duplicates,
                "warnings":warnings,"file_sha256":identity['file_hashes']}
    return {"metadata":metadata,"examples":[x.model_dump() for x in examples],"files":files}
