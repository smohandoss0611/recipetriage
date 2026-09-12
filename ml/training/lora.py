"""Inspect actual projections, attach PEFT LoRA, and verify adapter checkpoints.

For column vectors: y = W x + b + (alpha / r) B A dropout(x).
W is frozen; A has shape (r, in_features), B has shape (out_features, r).
"""
from collections import Counter
from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path
from threading import Event, Thread
import time
from recipetriage_ml.data.pipeline import digest
from recipetriage_ml.evaluation.base_provider import MODEL_ID, MODEL_REVISION
from .data import file_hash

POLICIES = {
    'query-value': ['q_proj', 'v_proj'],
    'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'attention-and-mlp': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
}


def inspect_model(model):
    """Read the instantiated module tree, including each linear layer's real shape."""
    import torch
    if hasattr(model, 'peft_config'):
        raise ValueError('Inspect the unadapted base model before inserting LoRA')
    config = model.config
    if config.model_type != 'qwen2':
        raise ValueError('This lesson supports the inspected Qwen2 architecture; inspect and define a new policy for another architecture')
    projections = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            role = name.rsplit('.', 1)[-1]
            eligible = name.startswith('model.layers.') and ('.self_attn.' in name or '.mlp.' in name)
            projections.append({'name': name, 'role': role, 'type': type(layer).__name__,
                                'in_features': layer.in_features, 'out_features': layer.out_features,
                                'weight_shape': list(layer.weight.shape), 'has_bias': layer.bias is not None,
                                'eligible': eligible,
                                'reason': 'Transformer projection' if eligible else 'Output head excluded; tied to the token embedding in this model'})
    report = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'class': type(model).__name__,
              'model_type': config.model_type, 'layers': config.num_hidden_layers,
              'hidden_size': config.hidden_size, 'intermediate_size': config.intermediate_size,
              'attention_heads': config.num_attention_heads, 'kv_heads': config.num_key_value_heads,
              'tied_embeddings': config.tie_word_embeddings,
              'base_parameters': sum(p.numel() for p in model.parameters()),
              'projections': projections, 'policies': POLICIES}
    report['architecture_sha256'] = digest(report)
    return report


def target_paths(report, policy='query-value', layers=None):
    if policy not in POLICIES:
        raise ValueError('Unknown target policy')
    chosen = [row for row in report['projections'] if row['eligible'] and row['role'] in POLICIES[policy]]
    counts = Counter(row['role'] for row in chosen)
    if any(counts[role] != report['layers'] for role in POLICIES[policy]):
        raise ValueError('Projection inventory does not cover every Transformer block for this target policy')
    if layers is not None:
        if not layers or len(set(layers))!=len(layers) or any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<report['layers'] for x in layers):
            raise ValueError('Layer subset must contain unique valid block indices from the inspected model')
        chosen=[row for row in chosen if int(row['name'].split('.')[2]) in layers]
    return [row['name'] for row in chosen]


def estimated_parameters(report, rank, policy='query-value', layers=None):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError('Rank must be a positive integer')
    selected = set(target_paths(report, policy, layers))
    return sum(rank * (row['in_features'] + row['out_features']) for row in report['projections'] if row['name'] in selected)


def checkpoint_targets(report, targets):
    """PEFT may shorten a long full-path list to equivalent suffixes when saving.

    Resolve those suffixes against this exact base's inspected tree, rejecting
    unmatched names and any match outside its Transformer projections.
    """
    if not isinstance(targets, (list, set)) or not targets:
        raise ValueError('Checkpoint targets must be a nonempty list of module names')
    selected = set()
    for target in targets:
        matches = [row for row in report['projections'] if row['name'] == target or row['name'].endswith('.' + target)]
        if not matches or any(not row['eligible'] for row in matches):
            raise ValueError('Checkpoint contains targets outside the inspected Transformer projections')
        selected.update(row['name'] for row in matches)
    return sorted(selected)


def attach_adapter(model, rank=8, alpha=None, dropout=0.0, policy='query-value', inspection=None, layers=None):
    """Use full validated paths, then assert that only A/B matrices are trainable."""
    from peft import LoraConfig, get_peft_model
    report = inspection or inspect_model(model)
    paths = target_paths(report, policy, layers)
    # Quantized storage changes weight.shape; in/out feature dimensions retain
    # the original projection shape and must match the pre-quantization inspection.
    modules = dict(model.named_modules())
    for row in report['projections']:
        layer = modules.get(row['name'])
        if layer is None or (layer.in_features, layer.out_features) != (row['in_features'], row['out_features']):
            raise ValueError('Loaded projection dimensions differ from architecture inspection')
    if not 0 <= dropout < 1:
        raise ValueError('LoRA dropout must be in [0,1)')
    alpha = 2 * rank if alpha is None else alpha
    if alpha <= 0:
        raise ValueError('LoRA alpha must be positive')
    expected = estimated_parameters(report, rank, policy, layers)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=paths,
                        bias='none', task_type='CAUSAL_LM', revision=MODEL_REVISION,
                        init_lora_weights=True, use_rslora=False, modules_to_save=None)
    adapted = get_peft_model(model, config)
    actual = parameter_report(adapted)
    if actual['trainable'] != expected or actual['non_adapter_trainable_names']:
        raise ValueError('Trainable parameter audit failed; base weights must remain frozen')
    actual.update(expected_trainable=expected, base_parameters=report['base_parameters'],
                  rank=rank, alpha=alpha, scaling=alpha / rank, dropout=dropout,
                  policy=policy, target_paths=paths, architecture_sha256=report['architecture_sha256'])
    if inspection:
        actual['frozen_storage_elements'] = actual['frozen']
        actual['frozen'] = report['base_parameters']
        actual['total'] = actual['frozen'] + actual['trainable']
        actual['trainable_percent'] = 100 * actual['trainable'] / actual['total']
    return adapted, actual, report


def parameter_report(model):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for _, p in trainable)
    total = sum(p.numel() for p in model.parameters())
    return {'trainable': count, 'frozen': total - count, 'total': total,
            'trainable_percent': 100 * count / total if total else 0,
            'trainable_weight_bytes': sum(p.numel() * p.element_size() for _, p in trainable),
            'frozen_weight_bytes': sum(p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad),
            'non_adapter_trainable_names': [name for name, _ in trainable if '.lora_A.' not in name and '.lora_B.' not in name],
            'trainable_tensors': [{'name': name, 'shape': list(p.shape), 'parameters': p.numel()} for name, p in trainable]}


def save_adapter(model, tokenizer, directory):
    """Save adapter weights, config and tokenizer; full base weights are not copied."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Adapter export already exists; do not overwrite checkpoint evidence')
    model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(directory)
    manifest = {'base_model': MODEL_ID, 'base_revision': MODEL_REVISION,
                'format': 'peft-lora-v1', 'full_training_resume_state': False,
                'files': {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}}
    quantization = getattr(model.config, 'quantization_config', None)
    if quantization:
        manifest['quantization'] = quantization.to_dict() if hasattr(quantization, 'to_dict') else quantization
    (directory / 'adapter-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def load_adapter(directory):
    """Reject changed weights/config/tokenizer, then load onto the exact frozen base."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    directory = Path(directory)
    manifest = json.loads((directory / 'adapter-manifest.json').read_text(encoding='utf-8'))
    if manifest['base_model'] != MODEL_ID or manifest['base_revision'] != MODEL_REVISION:
        raise ValueError('Adapter belongs to a different base checkpoint')
    required = {'adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json', 'tokenizer.json'}
    actual_files = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'adapter-manifest.json'}
    if manifest.get('format') != 'peft-lora-v1' or not required <= manifest['files'].keys() or actual_files != manifest['files'].keys():
        raise ValueError('Adapter manifest must cover every checkpoint and tokenizer file')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_hash(directory / name) != expected:
            raise ValueError(f'Adapter checkpoint checksum mismatch: {name}')
    config = json.loads((directory / 'adapter_config.json').read_text(encoding='utf-8'))
    if config.get('peft_type') != 'LORA' or config.get('revision') != MODEL_REVISION or config.get('base_model_name_or_path') != MODEL_ID:
        raise ValueError('Adapter configuration disagrees with its base provenance')
    if manifest.get('quantization'):
        from .qlora import load_nf4, quantization_config
        expected = quantization_config().to_dict()
        if any(manifest['quantization'].get(key) != expected[key] for key in ['load_in_4bit', 'bnb_4bit_quant_type', 'bnb_4bit_compute_dtype', 'bnb_4bit_use_double_quant']):
            raise ValueError('Unsupported quantized adapter configuration')
        base, report, _ = load_nf4(gradient_checkpointing=False)
    else:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False)
        report = inspect_model(base)
    # Check target paths against the real loaded model before PEFT applies the checkpoint.
    checkpoint_targets(report, config.get('target_modules'))
    model = PeftModel.from_pretrained(base, directory, is_trainable=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
    return model, tokenizer


class TrainingResources:
    """Sample process RSS during trainer.train(); do not call it adapter-only memory."""
    def __init__(self, interval=0.02):
        import psutil
        self.process = psutil.Process()
        self.interval = interval
        self.stop = Event()

    def __enter__(self):
        self.started = time.perf_counter()
        self.start_rss = self.peak_rss = self.process.memory_info().rss
        self.samples = 1
        def sample():
            while not self.stop.wait(self.interval):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
                self.samples += 1
        self.thread = Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.started
        self.stop.set(); self.thread.join()
        end = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, end)
        self.result = {'train_wall_seconds': elapsed, 'start_rss_bytes': self.start_rss,
                       'peak_rss_bytes': self.peak_rss, 'end_rss_bytes': end,
                       'peak_rss_increase_bytes': self.peak_rss - self.start_rss,
                       'sample_interval_seconds': self.interval, 'samples': self.samples,
                       'scope': 'trainer.train: updates, epoch validation and checkpoint IO; excludes initial validation and benchmarks',
                       'memory_metric': 'sampled process resident set size (RSS); not GPU memory or adapter-only allocation',
                       'limitations': 'Sampling can miss short peaks; RSS includes shared libraries, base weights, activations, gradients, optimizer and allocator caches.'}


@lru_cache(maxsize=1)
def architecture_report():
    """Bundled output of the inspector for the pinned model; training re-inspects live."""
    return json.loads(files('recipetriage_ml').joinpath('training/architecture_qwen.json').read_text(encoding='utf-8'))


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', type=Path, help='Write architecture inspection without allocating full weights')
    parser.add_argument('--worker', type=Path, help='Internal isolated experiment worker: queued run JSON')
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(Path('.cache/huggingface').resolve()))
    if args.worker:
        from .config import SFTConfig
        from .sft import run_local
        job = json.loads(args.worker.read_text(encoding='utf-8'))
        snapshot = json.loads((args.worker.parent / 'snapshot.json').read_text(encoding='utf-8'))
        run_local(SFTConfig.model_validate(job['config']), snapshot, run=job, root=args.worker.parent.parent)
    elif args.inspect:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.tie_weights()
        report = inspect_model(model)
        args.inspect.parent.mkdir(parents=True, exist_ok=True)
        args.inspect.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'class': report['class'], 'base_parameters': report['base_parameters'],
                          'linear_modules': len(report['projections']), 'selected_targets': len(target_paths(report))}, indent=2))
    else:
        parser.error('Specify --inspect PATH or use the LoRA experiment CLI')


if __name__ == '__main__':
    main()
