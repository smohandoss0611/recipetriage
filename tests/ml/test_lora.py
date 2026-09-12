"""Offline tests use tiny random matrices, not generated recipe training data."""
import copy
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.data import seed_snapshot
from recipetriage_ml.training.lora_experiment import ExperimentConfig, compare_ranks


def tiny():
    pytest.importorskip('peft')
    from transformers import Qwen2Config, Qwen2ForCausalLM
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True))
    from recipetriage_ml.evaluation.base_provider import MODEL_ID
    model.name_or_path = MODEL_ID
    return model


def test_inspected_shapes_targets_and_frozen_optimizer_step():
    torch = pytest.importorskip('torch')
    from recipetriage_ml.training.lora import attach_adapter, inspect_model, estimated_parameters, target_paths
    torch.manual_seed(42)
    base = tiny().eval()
    report = inspect_model(base)
    assert len(report['projections']) == 15
    assert len(target_paths(report)) == 4
    assert [p['weight_shape'] for p in report['projections'] if p['role'] == 'v_proj'] == [[8, 16]] * 2
    assert estimated_parameters(report, 4) == 448
    assert estimated_parameters(report, 16) == 1792
    tokens = torch.tensor([[1, 2, 3, 4]])
    before = base(tokens).logits.detach().clone()
    model, audit, _ = attach_adapter(base, rank=4)
    model.eval()
    assert torch.equal(before, model(tokens).logits)
    assert audit['trainable'] == 448 and audit['non_adapter_trainable_names'] == []
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    adapters = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(tokens, labels=tokens).loss.backward()
    # B=0 initially, so A's first gradient is zero; B's gradient can be nonzero.
    assert all(torch.count_nonzero(p.grad) == 0 for name, p in model.named_parameters() if '.lora_A.' in name)
    assert any(torch.count_nonzero(p.grad) > 0 for name, p in model.named_parameters() if '.lora_B.' in name)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    assert any(not torch.equal(value, dict(model.named_parameters())[name]) for name, value in adapters.items())
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match='unadapted'): inspect_model(model)
    with pytest.raises(ValueError, match='Unknown'): target_paths(report, 'copied-guessed-name')
    report['projections'] = report['projections'][1:]
    with pytest.raises(ValueError, match='every Transformer block'): target_paths(report)


def test_dropout_train_eval_and_adapter_checkpoint_integrity(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from recipetriage_ml.training.lora import attach_adapter, save_adapter, load_adapter
    base = tiny(); original = copy.deepcopy(base)
    model, _, _ = attach_adapter(base, rank=4, dropout=0.5)
    # Populate B to observe the adapter branch rather than its zero initialization.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if '.lora_B.' in name: p.fill_(0.1)
    inputs = torch.ones(32, 16)
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    model.train(); assert not torch.equal(layer(inputs), layer(inputs))
    model.eval(); expected = layer(inputs).detach()
    assert torch.equal(expected, layer(inputs))
    class Tokenizer:
        def save_pretrained(self, directory):
            for name in ['tokenizer.json', 'tokenizer_config.json']:
                (Path(directory) / name).write_text('{}')
    path = tmp_path/'adapter'
    save_adapter(model, Tokenizer(), path)
    with pytest.raises(ValueError, match='already exists'): save_adapter(model, Tokenizer(), path)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: copy.deepcopy(original))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    loaded, _ = load_adapter(path)
    assert not any(p.requires_grad for p in loaded.parameters())
    assert torch.equal(expected, loaded.base_model.model.model.layers[0].self_attn.q_proj(inputs))
    manifest_path = path/'adapter-manifest.json'; manifest = json.loads(manifest_path.read_text())
    wrong = copy.deepcopy(manifest); del wrong['files']['tokenizer.json']; manifest_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match='cover every'): load_adapter(path)
    manifest_path.write_text(json.dumps(manifest))
    with (path/'adapter_model.safetensors').open('ab') as file: file.write(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'): load_adapter(path)


def test_peft_compressed_target_names_resolve_against_inspected_architecture():
    from recipetriage_ml.training.lora import architecture_report, checkpoint_targets, target_paths
    report = architecture_report()
    assert checkpoint_targets(report, ['q_proj', 'v_proj']) == sorted(target_paths(report))
    for invalid in [['lm_head'], ['q_proj', 'invented'], '.*q_proj']:
        with pytest.raises(ValueError): checkpoint_targets(report, invalid)


@pytest.mark.parametrize('ranks', [[4], [4, 4], [4, 7], [4, '16'], [4, True]])
def test_experiment_rank_validation(ranks):
    with pytest.raises(ValidationError):
        ExperimentConfig(training=SFTConfig(dataset_version=seed_snapshot()['metadata']['version']), ranks=ranks)


def reference_runs():
    path = Path(__file__).parents[2]/'ml/training/results/lora-v1'
    experiment = json.loads((path/'experiment.json').read_text())
    return experiment, [json.loads((path/'runs'/row['run_id']/'run.json').read_text()) for row in experiment['runs']]


def test_real_rank_comparison_and_protocol_guards():
    experiment, runs = reference_runs()
    assert compare_ranks(runs) == experiment['comparison']
    assert [r['model_parameters']['trainable'] for r in runs] == [270336, 1081344]
    assert all(r['comparison']['delta']['micro_f1'] < 0 for r in runs)
    assert all(r['training_resources']['samples'] > 1 for r in runs)
    assert runs[1]['benchmark']['summary']['labels']['micro']['f1'] == 0
    for change in ['config', 'architecture', 'resources', 'status', 'dataset']:
        wrong = copy.deepcopy(runs)
        if change == 'config': wrong[1]['config']['epochs'] += 1
        elif change == 'architecture': wrong[1]['model_parameters']['architecture_sha256'] = 'changed'
        elif change == 'resources': wrong[1]['training_resources'] = None
        elif change == 'status': wrong[1]['status'] = 'failed'
        else: wrong[1]['dataset']['logical_messages_sha256']['train'] = 'changed'
        assert not compare_ranks(wrong)['comparable']


def test_worker_crash_retains_failure_and_does_not_start_next_rank(tmp_path, monkeypatch):
    import recipetriage_ml.training.lora_experiment as module
    snapshot = seed_snapshot(); starts = []
    class CrashedProcess:
        returncode = 137
        def __init__(self, args, **kwargs): starts.append(args)
        def poll(self): return 137
    monkeypatch.setattr(module.subprocess, 'Popen', CrashedProcess)
    config = ExperimentConfig(training=SFTConfig(dataset_version=snapshot['metadata']['version']))
    with pytest.raises(RuntimeError, match='exit 137'):
        module.run_experiment(config, snapshot, root=tmp_path)
    assert len(starts) == 1
    saved = json.loads(next(tmp_path.glob('*/experiment.json')).read_text())
    assert saved['status'] == 'failed' and saved['runs'][0]['status'] == 'failed'
    assert next(tmp_path.glob('*/runs/*/snapshot.json')).exists()
