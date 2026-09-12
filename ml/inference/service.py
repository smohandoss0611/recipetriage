import time
from .contracts import ProviderError
from .fireworks_provider import FireworksProvider
from .hf_provider import HFProvider
from .parsing import parse_prediction
from .prompts import PROMPT_VERSION, build_messages


def get_provider(name):
    if name == "hf":
        return HFProvider()
    if name == "fireworks":
        # The small completion budget is for the JSON answer, as in model comparison.
        return FireworksProvider(reasoning_effort="none")
    raise ValueError("Unknown provider")


def triage(recipe, provider="hf", temperature=0.0, max_new_tokens=128, provider_instance=None):
    started = time.perf_counter()
    generation = (provider_instance or get_provider(provider)).generate(build_messages(recipe), temperature, max_new_tokens)
    result = {**generation.model_dump(), "provider": provider, "prompt_version": PROMPT_VERSION,
              "temperature": temperature, "max_new_tokens": max_new_tokens,
              "latency_ms": round((time.perf_counter()-started)*1000), "prediction": None,
              "valid_json": False, "error": None}
    try:
        prediction = parse_prediction(generation.raw_output)
        if generation.finish_reason != "stop":
            raise ValueError(f"Generation did not finish normally: {generation.finish_reason}")
        result.update(prediction=prediction.model_dump(), valid_json=True)
    except ValueError as exc:
        result["error"] = str(exc)
    # Never enforce recipe truth here: wrong but well-formed predictions are evidence.
    return result
