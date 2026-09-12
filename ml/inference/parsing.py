import json
from .contracts import Prediction


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_prediction(raw: str) -> Prediction:
    """Validate the complete answer. Do not rescue fenced/truncated/mixed output."""
    data = json.loads(raw, object_pairs_hook=_unique_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))
    return Prediction.model_validate(data)
