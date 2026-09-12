import copy
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from recipetriage_ml.training.config import SFTConfig
from recipetriage_ml.training.data import encode_example, prepare_snapshot, seed_snapshot, verify_snapshot
from recipetriage_ml.training.runs import Journal, compare, evaluate, new_training_run
from recipetriage_ml.data.pipeline import build_dataset, chat_messages, digest
from recipetriage_ml.evaluation.benchmark import descriptor
from recipetriage_ml.evaluation.base_provider import render_prompt, MODEL_REVISION
from recipetriage_ml.inference.contracts import Generation


class CharacterTokenizer:
    eos_token_id = 0
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def test_prompt_answer_eos_and_padding_masks():
    torch = pytest.importorskip('torch')
    pytest.importorskip('trl')
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    examples = verify_snapshot(seed_snapshot())['train'][:2]
    rows = [encode_example(row, CharacterTokenizer(), 10000) for row in examples]
    for example, row in zip(examples, rows):
        messages = chat_messages(example)
        prefix = len(render_prompt(messages[:-1]))
        assert row['labels'][:prefix] == [-100] * prefix
        assert row['labels'][prefix:-1] == [ord(char) for char in messages[-1]['content']]
        assert row['labels'][-1] == 0  # EOS remains supervised even though PAD uses the same ID.
    batch = DataCollatorForLanguageModeling(pad_token_id=0)(rows)
    assert torch.all(batch['labels'][batch['attention_mask'] == 0] == -100)
    for i, row in enumerate(rows):
        assert batch['labels'][i, len(row['labels']) - 1].item() == 0


def test_no_silent_truncation_or_token_boundary_drift():
    example = verify_snapshot(seed_snapshot())['train'][0]
    with pytest.raises(ValueError, match='no truncation'):
        encode_example(example, CharacterTokenizer(), 128)
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text)
            if not text.endswith('Assistant:\n'):
                ids[0] += 1
            return ids
    with pytest.raises(ValueError, match='boundary'):
        encode_example(example, MergingTokenizer(), 10000)


def test_snapshot_tampering_and_benchmark_leakage_rejected():
    snapshot = seed_snapshot()
    for field in ['content', 'metadata']:
        bad = copy.deepcopy(snapshot)
        if field == 'content': bad['files']['train.chat.jsonl'] += '\n'
        else: bad['metadata']['counts']['train'] = 99
        with pytest.raises(ValueError): verify_snapshot(bad)
    from recipetriage_ml.evaluation.benchmark import load_benchmark
    records = json.loads(snapshot['files']['raw.json'])
    leaked = load_benchmark().cases[0].model_dump(exclude={'category', 'challenge'})
    records.append(leaked)
    with pytest.raises(ValueError, match='Benchmark leakage'):
        verify_snapshot(build_dataset(records))


def test_same_logical_dataset_and_no_test_export(tmp_path):
    snapshot = seed_snapshot()
    splits, provenance = prepare_snapshot(snapshot, tmp_path)
    for split in ['train', 'validation']:
        local = [json.loads(line)['messages'] for line in (tmp_path / f'{split}.chat.jsonl').read_text().splitlines()]
        managed = [json.loads(line)['messages'] for line in (tmp_path / f'{split}.fireworks.jsonl').read_text().splitlines()]
        assert [[{key: value for key, value in message.items() if key != 'weight'} for message in row] for row in managed] == local
        assert all(message['weight'] == int(message['role'] == 'assistant') for row in managed for message in row)
        assert digest(local) == provenance['logical_messages_sha256'][split]
    assert not list(tmp_path.glob('test*'))
    assert provenance['test_used_for_training_or_selection'] is False


@pytest.mark.parametrize('override', [{'epochs': 0}, {'learning_rate': float('nan')}, {'lora_rank': 7}, {'batch_size': True}, {'sequence_length': 32}, {'unexpected': 1}])
def test_hyperparameters_reject_invalid_values(override):
    with pytest.raises(ValidationError):
        SFTConfig(dataset_version=seed_snapshot()['metadata']['version'], **override)


class Dummy:
    def generate(self, messages, *args):
        assert len(messages) == 2
        return Generation(raw_output='{"labels":["unclear"],"explanation":"Fixture response."}', model='fixture', finish_reason='stop')


def test_automatic_evaluation_and_comparison_use_full_frozen_benchmark(tmp_path):
    config = SFTConfig(dataset_version=seed_snapshot()['metadata']['version'])
    journal = Journal(new_training_run(config), tmp_path)
    before = evaluate(journal, Dummy(), descriptor('hf-base'), 'baseline')
    identity = {**descriptor('hf-base'), 'model': 'local-adapter', 'base_model': descriptor('hf-base')['model'],
                'base_revision': MODEL_REVISION, 'revision': 'weights-hash', 'training_stage': 'sft'}
    after = evaluate(journal, Dummy(), identity, 'benchmark')
    result = compare(before, after)
    assert result['comparable'] and result['delta']['micro_f1'] == 0
    assert result['claim'].startswith('No measured')
    assert len(after['rows']) == 10 and after['model_config']['training_stage'] == 'sft'
    saved = json.loads((journal.directory / 'run.json').read_text())
    assert saved['baseline'] == before and saved['benchmark'] == after
    assert (journal.directory / 'benchmark.json').exists()
    for change in ['protocol', 'base_model', 'status']:
        wrong = copy.deepcopy(after)
        if change == 'protocol': wrong['protocol_sha256'] = 'different'
        elif change == 'base_model': wrong['model_config']['base_model'] = 'another-model'
        else: wrong['status'] = 'blocked'
        assert compare(before, wrong)['comparable'] is False
    no_responses = copy.deepcopy(after)
    no_responses['summary']['returned_responses'] = 0
    no_responses['summary']['json_validity'] = None
    assert compare(before, no_responses)['comparable'] is False


def test_real_reference_run_metrics_and_adapter_hash():
    path = Path(__file__).parents[2] / 'ml/training/results/sft-v1/local'
    if not path.exists(): pytest.skip('Reference artifacts not packaged in this environment')
    run = json.loads((path / 'run.json').read_text())
    assert run['comparison'] == compare(run['baseline'], run['benchmark'])
    assert run['comparison']['delta']['micro_f1'] < 0
    assert run['optimizer_steps'] == 9
