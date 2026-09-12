"""Small local instruct model; weights and tokenizer load lazily once per process."""
from functools import lru_cache
import os
from threading import Lock
from .contracts import Generation, ProviderError

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_load_lock = Lock()
_generation_lock = Lock()


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)


def get_tokenizer():
    with _load_lock:
        return _tokenizer()


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.getenv("HF_CPU_THREADS", "4")))
    # CPU float32 favors reproducibility across macOS and Linux for this lesson.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                dtype=torch.float32, trust_remote_code=False)
    model.eval()
    return model


def get_model():
    with _load_lock:
        return _model()


def clear_resources():
    """Release optional comparison caches between models on small machines."""
    with _load_lock:
        _model.cache_clear()
        _tokenizer.cache_clear()


def inspect_tokens(text, messages=None):
    tokenizer = get_tokenizer()
    if messages is None:
        rendered = text
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) > 4096:
        raise ValueError("Inspection is limited to 4096 tokens; shorten the input")
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "rendered_text": rendered,
            "tokens": tokenizer.convert_ids_to_tokens(ids), "token_ids": ids,
            "token_count": len(ids), "decoded_text": tokenizer.decode(ids, skip_special_tokens=False)}


class HFProvider:
    def generate(self, messages, temperature=0.0, max_new_tokens=128):
        import torch
        if not _generation_lock.acquire(blocking=False):
            raise ProviderError("Local model is busy; retry after the current inference finishes")
        try:
            tokenizer = get_tokenizer()
            model = get_model()
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            count = inputs.input_ids.shape[1]
            if count + max_new_tokens > 4096:
                raise ValueError("Input plus output budget exceeds the learning limit of 4096 tokens")
            options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                       "pad_token_id": tokenizer.eos_token_id, "max_time": 120}
            if temperature > 0:
                options["temperature"] = temperature
            with torch.inference_mode():
                output = model.generate(**inputs, **options)
            new_ids = output[0, count:]
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            reason = "stop" if len(new_ids) and new_ids[-1].item() in eos_ids else ("length" if len(new_ids) >= max_new_tokens else "time_limit")
            return Generation(raw_output=tokenizer.decode(new_ids, skip_special_tokens=True),
                              model=MODEL_ID, revision=MODEL_REVISION, finish_reason=reason,
                              input_tokens=count, output_tokens=len(new_ids))
        finally:
            _generation_lock.release()
