"""Multi-label metrics with explicit failure and denominator policies."""
import json
import math
from statistics import mean, median
from recipetriage_ml.data.schemas import LABELS
from recipetriage_ml.inference.parsing import parse_prediction

METRICS_VERSION = "1.0.0"


def inspect_response(raw, finish_reason):
    result = dict(json_valid=False, schema_valid=False, usable=False, prediction=None, error=None)
    try:
        # JSON syntax and application schema are deliberately separate measures.
        json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {x}")))
        result['json_valid'] = True
        prediction = parse_prediction(raw)
        result.update(schema_valid=True, prediction=prediction.model_dump())
        if finish_reason != 'stop':
            raise ValueError(f"Incomplete generation: {finish_reason}")
        result['usable'] = True
    except (ValueError, TypeError) as exc:
        result['error'] = str(exc)
    return result


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def prf(tp, fp, fn):
    return {'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'f1': ratio(2 * tp, 2 * tp + fp + fn)}


def label_metrics(expected, predicted, usable=None):
    if not expected or len(expected) != len(predicted):
        raise ValueError('Expected and predicted lists must have the same nonzero length')
    usable = [True] * len(expected) if usable is None else usable
    if len(usable) != len(expected):
        raise ValueError('Usability flags must align with examples')
    for labels in [*expected, *predicted]:
        if len(labels) != len(set(labels)) or not set(labels) <= set(LABELS):
            raise ValueError('Metrics require unique known labels')
    truth = [set(x) for x in expected]
    guesses = [set(x) if ok else set() for x, ok in zip(predicted, usable)]
    per_label = {}
    for label in LABELS:
        tp = sum(label in y and label in p for y, p in zip(truth, guesses))
        fp = sum(label not in y and label in p for y, p in zip(truth, guesses))
        fn = sum(label in y and label not in p for y, p in zip(truth, guesses))
        per_label[label] = dict(tp=tp, fp=fp, fn=fn, support=tp + fn, **prf(tp, fp, fn))
    totals = {key: sum(row[key] for row in per_label.values()) for key in ['tp', 'fp', 'fn']}
    return {'per_label': per_label, 'micro': prf(**totals),
            'macro': {key: mean(row[key] for row in per_label.values()) for key in ['precision', 'recall', 'f1']},
            'exact_match': mean(bool(ok and y == p) for y, p, ok in zip(truth, guesses, usable)),
            'zero_division': 0, 'macro_label_count': len(LABELS), 'samples': len(truth)}


def latency_metrics(values):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Latency must contain nonnegative finite numbers')
    if not values:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'min_ms': None, 'max_ms': None}
    ordered = sorted(values)
    return {'count': len(values), 'mean_ms': mean(values), 'p50_ms': median(values),
            'p95_ms': ordered[math.ceil(0.95 * len(values)) - 1], 'min_ms': min(values), 'max_ms': max(values)}


def summarize(cases, rows):
    by_id = {row['case_id']: row for row in rows}
    ids = {case.recipe.id for case in cases}
    if len(by_id) != len(rows) or not set(by_id) <= ids:
        raise ValueError('Unexpected or repeated case results')
    attempted = [row for row in rows if row['attempted']]
    responses = [row for row in rows if row['raw_output'] is not None]
    complete = len(rows) == len(cases) and len(attempted) == len(cases)
    result = {'expected_cases': len(cases), 'recorded_cases': len(rows), 'attempted_cases': len(attempted),
              'returned_responses': len(responses), 'usable_responses': sum(row['usable'] for row in rows),
              'json_valid_count': sum(row['json_valid'] for row in responses),
              'schema_valid_count': sum(row['schema_valid'] for row in responses),
              'json_validity': sum(row['json_valid'] for row in responses) / len(responses) if responses else None,
              'schema_validity': sum(row['schema_valid'] for row in responses) / len(responses) if responses else None,
              'usable_response_rate': ratio(sum(row['usable'] for row in rows), len(cases)),
              'scorable': complete, 'labels': None, 'categories': {},
              'latency_all_attempts': latency_metrics([row['latency_ms'] for row in attempted]),
              'latency_usable': latency_metrics([row['latency_ms'] for row in rows if row['usable']])}
    if complete:
        def score(subset):
            selected = [by_id[case.recipe.id] for case in subset]
            return label_metrics([case.labels for case in subset],
                                 [row['prediction']['labels'] if row['prediction'] else [] for row in selected],
                                 [row['usable'] for row in selected])
        result['labels'] = score(cases)
        result['categories'] = {category: score([case for case in cases if case.category == category])
                                for category in sorted({case.category for case in cases})}
    return result
