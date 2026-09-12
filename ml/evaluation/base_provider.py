"""The pretrained Qwen checkpoint, explicitly distinct from the instruct model."""
from functools import lru_cache
import os
from recipetriage_ml.inference.contracts import Generation, ProviderError
from recipetriage_ml.inference.hf_provider import _generation_lock

MODEL_ID = 'Qwen/Qwen2.5-0.5B'
MODEL_REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
FORMAT_VERSION = 'plain-role-completion-v1'


def render_prompt(messages):
    # No demonstrations, gold labels, or invented instruct chat template.
    return '\n\n'.join(f"{message['role'].capitalize()}:\n{message['content']}" for message in messages) + '\n\nAssistant:\n'


@lru_cache(maxsize=1)
def resources():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.getenv('HF_CPU_THREADS', '4')))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False)
    model.eval()
    return tokenizer, model


class BaseProvider:
    def prepare(self):
        cached = resources.cache_info().currsize > 0
        resources()
        return {'already_loaded': cached, 'device': 'cpu', 'dtype': 'float32', 'cpu_threads': int(os.getenv('HF_CPU_THREADS', '4'))}

    def generate(self, messages, temperature=0.0, max_new_tokens=128):
        import torch
        if not _generation_lock.acquire(blocking=False):
            raise ProviderError('Local model is busy; retry after the current inference finishes')
        try:
            tokenizer, model = resources()
            inputs = tokenizer(render_prompt(messages), return_tensors='pt', add_special_tokens=False)
            count = inputs.input_ids.shape[1]
            if count + max_new_tokens > 4096:
                raise ValueError('Input plus output budget exceeds 4096 tokens')
            options = {'max_new_tokens': max_new_tokens, 'do_sample': temperature > 0,
                       'pad_token_id': tokenizer.eos_token_id, 'max_time': 120}
            if temperature > 0:
                options['temperature'] = temperature
            with torch.inference_mode():
                output = model.generate(**inputs, **options)
            generated = output[0, count:]
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            stopped = len(generated) > 0 and generated[-1].item() in eos_ids
            reason = 'stop' if stopped else 'length' if len(generated) >= max_new_tokens else 'time_limit'
            return Generation(raw_output=tokenizer.decode(generated, skip_special_tokens=True), model=MODEL_ID,
                              revision=MODEL_REVISION, finish_reason=reason, input_tokens=count, output_tokens=len(generated))
        finally:
            _generation_lock.release()
