"""Real bitsandbytes NF4 storage with float32 CPU computation and PEFT adapters."""
from functools import lru_cache
from importlib.metadata import version
import platform
from .lora import inspect_model
from recipetriage_ml.evaluation.base_provider import MODEL_ID, MODEL_REVISION


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32,
                              bnb_4bit_quant_storage=torch.uint8)


@lru_cache(maxsize=1)
def capability():
    """Probe actual CPU NF4 quantization/backward; never silently substitute float weights."""
    result = {'device': 'cpu', 'platform': platform.platform(), 'supported': False,
              'compute_dtype': 'float32', 'quant_type': 'nf4', 'double_quantization': True}
    try:
        import torch
        import bitsandbytes as bnb
        result.update(torch=version('torch'), bitsandbytes=version('bitsandbytes'))
        # Keep the capability probe from changing the caller's initialization seed.
        with torch.random.fork_rng(devices=[]):
            layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.float32,
                                     compress_statistics=True, quant_type='nf4').to('cpu')
            x = torch.randn(2, 64, requires_grad=True)
            layer(x).square().mean().backward()
            if layer.weight.quant_state.quant_type != 'nf4' or not torch.isfinite(x.grad).all():
                raise ValueError('NF4 backward probe failed')
        result.update(supported=True, error=None)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: CPU NF4 probe failed. Install ml/requirements-experiments.lock and inspect worker logs.'
    return result


def load_nf4(gradient_checkpointing=True):
    import torch
    import bitsandbytes as bnb
    from accelerate import init_empty_weights
    from peft import prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM
    if not capability()['supported']:
        raise ValueError(capability()['error'])
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    with init_empty_weights():
        structure = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    structure.tie_weights()
    report = inspect_model(structure)
    del structure
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        quantization_config=quantization_config(), device_map={'': 'cpu'}, dtype=torch.float32,
        trust_remote_code=False)
    expected = {row['name']: row for row in report['projections'] if row['eligible']}
    quantized = {}
    for name, layer in model.named_modules():
        if isinstance(layer, bnb.nn.Linear4bit):
            state = layer.weight.quant_state
            if name not in expected or state is None or state.quant_type != 'nf4' or not state.nested:
                raise ValueError('NF4/double-quantization audit failed')
            quantized[name] = {'logical_shape': list(state.shape), 'packed_shape': list(layer.weight.shape),
                               'storage_dtype': str(layer.weight.dtype), 'packed_bytes': layer.weight.numel() * layer.weight.element_size()}
            if list(state.shape) != expected[name]['weight_shape']:
                raise ValueError('Quantized shape differs from original projection')
    if set(quantized) != set(expected):
        raise ValueError('Not every eligible Transformer projection loaded in NF4')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    audit = {'config': quantization_config().to_dict(), 'quantized_modules': quantized,
             'quantized_module_count': len(quantized), 'compute_dtype': 'float32',
             'base_logical_parameters': report['base_parameters'],
             'note': 'Linear weights use NF4; embeddings, output head and norms remain float32. Packed bytes exclude scale metadata.'}
    return model, report, audit


def main():
    import argparse, json, os
    from pathlib import Path
    from .config import SFTConfig
    from .data import seed_snapshot, load_snapshot
    from .sft import run_local
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(capability(), indent=2)); return
    os.environ.setdefault('HF_HOME', str(Path('.cache/huggingface').resolve()))
    snapshot = load_snapshot(args.dataset) if args.dataset else seed_snapshot()
    settings = SFTConfig.model_validate_json(args.config.read_text()) if args.config else SFTConfig(dataset_version=snapshot['metadata']['version'], method='qlora')
    if settings.method != 'qlora' or settings.provider != 'local':
        parser.error('QLoRA requires method=qlora and provider=local')
    result = run_local(settings, snapshot, root=args.output)
    print(json.dumps({'run_id': result['run_id'], 'status': result['status'], 'comparison': result['comparison']}, indent=2))


if __name__ == '__main__':
    main()
