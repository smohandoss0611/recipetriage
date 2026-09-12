"""Readable LoRA SFT: checked data -> masked labels -> SFTTrainer -> fixed benchmark.

Run from the repository root with ``python -m recipetriage_ml.training.sft``.
The existing base checkpoint is frozen; only small adapters receive updates.
"""
import csv
import gc
import json
import math
import os
from pathlib import Path
from importlib.metadata import version
from recipetriage_ml.evaluation.base_provider import MODEL_ID, MODEL_REVISION, FORMAT_VERSION, render_prompt
from recipetriage_ml.inference.contracts import Generation, ProviderError
from recipetriage_ml.inference.hf_provider import _generation_lock
from .config import SFTConfig
from .data import encode_example, file_hash, load_snapshot, prepare_snapshot, seed_snapshot
from .runs import Journal, compare, evaluate, new_training_run, now


class TrainedProvider:
    def __init__(self, model, tokenizer, identity):
        self.model, self.tokenizer, self.identity = model, tokenizer, identity

    def prepare(self):
        self.model.eval()
        return {'already_loaded': True, 'device': 'cpu', 'dtype': 'float32', 'checkpoint_sha256': self.identity['revision']}

    def generate(self, messages, temperature=0, max_new_tokens=128):
        import torch
        if not _generation_lock.acquire(blocking=False):
            raise ProviderError('Local model is busy; retry after training or inference completes')
        try:
            inputs = self.tokenizer(render_prompt(messages), return_tensors='pt', add_special_tokens=False)
            count = inputs.input_ids.shape[1]
            if count + max_new_tokens > 4096:
                raise ValueError('Input plus output exceeds 4096 tokens')
            options = {'max_new_tokens': max_new_tokens, 'do_sample': temperature > 0,
                       'pad_token_id': self.tokenizer.eos_token_id, 'max_time': 120}
            if temperature > 0:
                options['temperature'] = temperature
            with torch.inference_mode():
                output = self.model.generate(**inputs, **options)
            ids = output[0, count:]
            eos = self.model.generation_config.eos_token_id
            stopped = bool(len(ids)) and ids[-1].item() in (eos if isinstance(eos, list) else [eos])
            reason = 'stop' if stopped else 'length' if len(ids) >= max_new_tokens else 'time_limit'
            return Generation(raw_output=self.tokenizer.decode(ids, skip_special_tokens=True),
                              model=self.identity['model'], revision=self.identity['revision'], finish_reason=reason,
                              input_tokens=count, output_tokens=len(ids))
        finally:
            _generation_lock.release()


def run_local(config, snapshot, run=None, root=None, on_update=None, reference_baseline=None):
    if config.provider != 'local':
        raise ValueError('run_local requires provider=local')
    if run and run['status'] not in {'queued'}:
        raise ValueError('Use a new local run; existing training evidence cannot be overwritten')
    journal = Journal(run or new_training_run(config), root, on_update)
    model = trainer = None
    locked = False
    try:
        journal.write(status='preparing', phase='validate-data')
        if snapshot['metadata']['version'] != config.dataset_version:
            raise ValueError('Requested dataset version does not match snapshot')
        splits, data = prepare_snapshot(snapshot, journal.directory / 'data')
        journal.write(dataset=data)

        # Heavy dependencies stay optional for dataset/metrics/API-only users.
        import torch
        from datasets import Dataset
        from .lora import attach_adapter, save_adapter, TrainingResources
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
        from trl import SFTConfig as TrainerConfig, SFTTrainer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        from recipetriage_ml.evaluation.base_provider import BaseProvider, resources
        from recipetriage_ml.evaluation.benchmark import descriptor

        torch.set_num_threads(int(os.getenv('HF_CPU_THREADS', '4')))
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        encoded = {name: [encode_example(row, tokenizer, config.sequence_length) for row in splits[name]]
                   for name in ['train', 'validation']}
        journal.run['token_audit'] = {name: [{'recipe_id': recipe.recipe.id, 'tokens': len(row['input_ids']),
                                             'supervised_tokens': sum(v != -100 for v in row['labels'])}
                                            for recipe, row in zip(splits[name], rows)] for name, rows in encoded.items()}
        journal.run['runtime'] = {name: version(name) for name in ['torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate']}
        journal.write(status='baseline', phase='benchmark-base-before-training')
        # Fresh before-run on the same machine, model, formatter and generation settings.
        if reference_baseline is None:
            before = evaluate(journal, BaseProvider(), descriptor('hf-base'), 'baseline')
        else:
            from recipetriage_ml.evaluation.benchmark import RunConfig, new_run
            expected = new_run(RunConfig(provider='hf-base', temperature=0, max_new_tokens=128, reasoning='disabled'))
            if reference_baseline['protocol_sha256'] != expected['protocol_sha256'] or reference_baseline['model_config'] != descriptor('hf-base'):
                raise ValueError('Reference baseline does not match the fixed model and benchmark protocol')
            before = reference_baseline
            journal.run['baseline'] = before
            journal.artifact('baseline.json', before)
            journal.write()
        if before['status'] != 'completed' or not before['summary']['returned_responses']:
            raise ValueError('Base benchmark could not complete; no adapter training started')

        if not _generation_lock.acquire(blocking=False):
            raise ProviderError('Local model is busy; start training after inference finishes')
        locked = True
        # Remove cached base references before adapting it; normal inference must never use this adapter accidentally.
        inspection = None
        if config.method == 'qlora':
            resources.cache_clear(); gc.collect()
            from .qlora import load_nf4
            model, inspection, quantization_audit = load_nf4(config.gradient_checkpointing)
            journal.run['quantization'] = quantization_audit
            journal.artifact('quantization.json', quantization_audit)
        elif reference_baseline is not None:
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False)
        else:
            _, model = resources()
        resources.cache_clear()
        model.config.use_cache = False
        model, parameter_audit, architecture = attach_adapter(model, rank=config.lora_rank,
            alpha=config.lora_alpha, dropout=config.lora_dropout, policy=config.lora_target_policy, inspection=inspection, layers=config.lora_layers)
        journal.run['model_parameters'] = parameter_audit
        journal.artifact('parameter_report.json', parameter_audit)
        journal.artifact('architecture.json', architecture)

        class Progress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    values = {key: value for key, value in logs.items() if isinstance(value, (int, float))}
                    if any(not math.isfinite(value) for value in values.values()):
                        raise ValueError('Non-finite training metric; stopping instead of hiding it')
                    journal.run['history'].append({'step': state.global_step, 'epoch': state.epoch, **values})
                    journal.write()

            def on_save(self, args, state, control, **kwargs):
                journal.run['checkpoints'] = sorted(path.name for path in Path(args.output_dir).glob('checkpoint-*') if path.is_dir())
                journal.write()

        args = TrainerConfig(
            output_dir=str(journal.directory / 'checkpoints'), use_cpu=True, fp16=False, bf16=False,
            num_train_epochs=config.epochs, learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim='adamw_torch', weight_decay=0.0, lr_scheduler_type='constant', warmup_steps=0,
            max_grad_norm=1.0, seed=config.seed, data_seed=config.seed,
            gradient_checkpointing=config.gradient_checkpointing, gradient_checkpointing_kwargs={'use_reentrant': False},
            eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
            logging_steps=1, report_to='none', disable_tqdm=True, dataloader_pin_memory=False,
            max_length=config.sequence_length, packing=False, completion_only_loss=True,
            dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        )
        # Labels already mark prompt tokens -100. TRL pads labels with -100 too.
        collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
        trainer = SFTTrainer(model=model, args=args,
                             processing_class=tokenizer, data_collator=collator,
                             train_dataset=Dataset.from_list(encoded['train']),
                             eval_dataset=Dataset.from_list(encoded['validation']), callbacks=[Progress()])
        model = trainer.model
        journal.write(status='training', phase='initial-validation')
        initial = trainer.evaluate()
        journal.run['initial_validation_loss'] = initial['eval_loss']
        journal.write(phase='optimizer-updates')

        # SFTTrainer performs forward, masked loss, backward, accumulation, clipping,
        # optimizer.step(), zero_grad(), validation and checkpointing for this call.
        with TrainingResources() as resources_used:
            trainer.train()
        journal.run['training_resources'] = resources_used.result
        selected = trainer.state.best_model_checkpoint
        journal.run['selected_checkpoint'] = Path(selected).name if selected else None
        journal.run['best_validation_loss'] = trainer.state.best_metric
        journal.run['optimizer_steps'] = trainer.state.global_step
        save_adapter(model, tokenizer, journal.directory / 'adapter')
        adapter_sha = file_hash(journal.directory / 'adapter' / 'adapter_model.safetensors')
        identity = {'model': f"{MODEL_ID}+recipetriage-lora/{journal.run['run_id']}", 'revision': adapter_sha,
                    'base_model': MODEL_ID, 'base_revision': MODEL_REVISION,
                    'training_stage': 'supervised-fine-tuned-' + config.method, 'prompt_format': FORMAT_VERSION,
                    'quantization': 'nf4-double-float32' if config.method == 'qlora' else 'none',
                    'dataset_version': config.dataset_version}
        journal.artifact('adapter-provenance.json', identity)
        journal.run['artifacts']['adapter'] = 'adapter'
        save_history(journal)
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        _generation_lock.release()
        locked = False
        journal.write(status='evaluating', phase='benchmark-selected-checkpoint')
        after = evaluate(journal, TrainedProvider(model, tokenizer, identity), identity, 'benchmark')
        comparison = compare(before, after)
        journal.write(status='completed' if after['status'] == 'completed' and after['summary']['returned_responses'] else 'evaluation-blocked',
                      phase='finished', comparison=comparison, finished_at=now())
    except Exception as exc:
        # Stack trace is available to CLI/API logs; public state never embeds HTTP headers.
        error = str(exc) if isinstance(exc, (ValueError, ProviderError)) else f'{type(exc).__name__}: local training failed; inspect runner logs'
        journal.write(status='failed', error=error, finished_at=now())
        raise
    finally:
        if locked:
            _generation_lock.release()
        del trainer, model
        gc.collect()
    return journal.run


def save_history(journal):
    keys = ['step', 'epoch', 'loss', 'eval_loss', 'grad_norm', 'learning_rate']
    with (journal.directory / 'loss_by_step.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(journal.run['history'])
    journal.run['artifacts']['loss_by_step.csv'] = 'loss_by_step.csv'


def load_adapter(directory):
    """Reload only against the exact base revision, including older adapter configs without it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    directory = Path(directory)
    identity = json.loads((directory / 'adapter-provenance.json').read_text())
    if identity['base_model'] != MODEL_ID or identity['base_revision'] != MODEL_REVISION:
        raise ValueError('Unexpected adapter base model/revision')
    if file_hash(directory / 'adapter' / 'adapter_model.safetensors') != identity['revision']:
        raise ValueError('Adapter weight checksum mismatch')
    if (directory / 'adapter' / 'adapter-manifest.json').exists():
        from .lora import load_adapter as load_verified_adapter
        model, tokenizer = load_verified_adapter(directory / 'adapter')
        return TrainedProvider(model, tokenizer, identity)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, directory / 'adapter', is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(directory / 'adapter', trust_remote_code=False)
    return TrainedProvider(model, tokenizer, identity)


def main():
    import argparse
    from dotenv import load_dotenv
    load_dotenv('.env')
    os.environ.setdefault('HF_HOME', str(Path('.cache/huggingface').resolve()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON SFTConfig file')
    parser.add_argument('--dataset', type=Path, help='Versioned dataset directory; defaults to checked teaching seed')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    snapshot = load_snapshot(args.dataset) if args.dataset else seed_snapshot()
    config = SFTConfig.model_validate_json(args.config.read_text()) if args.config else SFTConfig(dataset_version=snapshot['metadata']['version'])
    run = run_local(config, snapshot, root=args.output)
    print(json.dumps({'run_id': run['run_id'], 'status': run['status'], 'comparison': run['comparison']}, indent=2))


if __name__ == '__main__':
    main()
