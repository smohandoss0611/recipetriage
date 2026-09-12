# Fixed experiments, QLoRA and failure analysis

This phase adds a PostgreSQL experiment registry, MLflow metric export, CLI/config-driven matrices, real bitsandbytes NF4 training, and two evaluation pages for failure analysis and title counterfactuals. Previous datasets, benchmark cases and lessons remain available.

## What is fixed and what changes

Every matrix run uses Dataset v1 `v1-63135eef80fb9be90547673e7f13d4142eb0eb29852cf098e672b44fca20158f`: five train, one validation and one test example. The same ten-case RecipeTriage-Bench-v1 hash remains `173e79cc1968dd1207a7aafd051dad40d87ff551c7081a17b896c9556a698448`. No synthetic training recipes were added.

The matrix was declared before running it:

- LoRA: learning rates 0.00005 and 0.0002 crossed with ranks 4 and 16 (four runs).
- QLoRA: learning rate 0.0002 and rank 4, matched against that LoRA setting (one additional run).

All other settings match: Qwen/Qwen2.5-0.5B pinned revision `060db6499f32faf8b98477b0a26969ef7d8b9987`; actual inspected query/value paths; alpha/r=2; dropout=0; seed=42; three epochs; sequence limit 768; microbatch 1; gradient accumulation 2; CPU float32 computation with four threads; AdamW; constant LR; no warmup or weight decay; gradient clipping norm 1; activation checkpointing enabled. There are nine optimizer updates per run. Checkpoints are chosen by validation loss only. Benchmark decoding is greedy with 128 output tokens. These parameters have the same meanings explained in SFT.md and LORA.md.

Each training configuration runs in a fresh, sequential process. One separate process runs the unchanged unquantized base benchmark, which is then shared byte-for-byte as the reference baseline. Keeping that model out of QLoRA's process avoids leaving its float32 weights in the training process's allocator cache. Each selected adapter receives its own full post-training benchmark.

## Actual results

Experiment: `e914b33c-cd67-4aaf-a4c1-4059c881ab30`, completed on the Apple M4 Mac CPU on 2026-09-09.

| Method | Learning rate | Rank | Trainable parameters | Training seconds | Peak RSS MiB | Micro-F1 | Schema-valid outputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 0.00005 | 4 | 270,336 | 25.91 | 3,128.44 | 9.09% | 20% |
| LoRA | 0.00005 | 16 | 1,081,344 | 25.91 | 3,044.67 | 34.48% | 30% |
| LoRA | 0.0002 | 4 | 270,336 | 27.04 | 3,066.67 | 41.38% | 30% |
| LoRA | 0.0002 | 16 | 1,081,344 | 25.44 | 2,978.33 | 0% | 0% |
| QLoRA NF4 | 0.0002 | 4 | 270,336 | 31.78 | 1,875.42 | 23.53% | 40% |

The base benchmark Micro-F1 is 48.48%. **None of the trained configurations improved on it.** All five adapters scored 0% exact label-set match. JSON syntax validity and application-schema validity are different: an object with only an explanation can be valid JSON but still have no usable prediction.

One-factor comparisons:

- At rank 4, raising LR from 0.00005 to 0.0002 helped on these cases: Micro-F1 rose 32.29 percentage points.
- At rank 16, that same LR increase hurt: Micro-F1 fell 34.48 points.
- At LR 0.00005, increasing rank from 4 to 16 helped by 25.39 points.
- At LR 0.0002, increasing rank hurt by 41.38 points.
- At matched LR 0.0002/rank 4, NF4 reduced Micro-F1 by 17.85 points. Peak process memory was about 38.85% lower and training time about 17.54% higher in this pair.

These interactions show why “higher rank” and “lower learning rate” are not universal rules. This is one repetition per configuration on ten provisional cases. No winning rank is selected automatically, and the best matrix score is still below the base. Repeated observation makes this benchmark exploratory evidence; do not tune repeatedly against it and then claim untouched test performance.

Memory is total process resident set size sampled every 20 ms during `trainer.train()`. It includes base weights, activations, optimizer state, libraries and allocator caches. The time includes optimizer updates, epoch validation and checkpoint writes. It excludes model loading, initial validation, before/after benchmarks and final adapter export. Sampling can miss short peaks. These CPU measurements do not measure GPU VRAM or establish a universal speed/memory ordering.

## QLoRA implementation

LoRA trains A/B while retaining ordinary frozen base weights. QLoRA stores the frozen linear weights using 4-bit NF4 and trains the same higher-precision adapters. NF4 is a nonuniform codebook; double quantization compresses scale information. Computation still uses a supported floating-point type. This CPU learning path uses float32 for both methods, so the matched comparison changes base storage, not the compute dtype. It uses regular AdamW for the adapters, not an 8-bit or paged optimizer.

`ml/training/qlora.py` creates:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_quant_storage=torch.uint8,
)
```

The original architecture is inspected on the meta device before loading. The loader checks that all 168 Transformer linear projections really use NF4, double quantization and the expected logical shapes. The output head, tied embeddings and norms stay float32. `prepare_model_for_kbit_training` freezes the base before PEFT attaches the inspected adapter targets. The parameter report distinguishes original logical parameters from packed storage elements. The adapter manifest records quantization settings so loading cannot silently switch the trained adapter onto an unquantized base.

The `gradient_checkpointing` boolean passes through config, PEFT preparation and SFTTrainer. It saves activations by recomputing them during backward. Both methods in a matrix must use the same choice. This differs from saving checkpoints on disk.

bitsandbytes 0.50.2 provides a macOS ARM CPU wheel, and the installed backend passed both a small NF4 backward probe and a real Qwen NF4 backward pass here. The application probes its own runtime and exposes failures rather than pretending to quantize. See the [versioned bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation) and [Transformers quantization guide](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes).

## Registry and MLflow

PostgreSQL is the application registry:

- `experiment_registry`: matrix name, dataset version, benchmark hash and full config/status/comparison payload.
- `training_runs`: each child run's parameters, losses, timings, benchmark and MLflow receipt.
- `failure_analyses`: versioned analysis of an immutable benchmark source, including its content hash.
- `failure_categories`: queryable `(source_run_id, case_id, category)` records with an index on category.
- `diagnostic_runs`: title counterfactuals and a separately identified red-team suite.

A shared PostgreSQL advisory lock serializes new experiments, training, benchmarks and diagnostics. Completed evidence cannot be overwritten by progress updates. Interrupted workers are marked explicitly on restart. Imported experiment summaries/configurations must match their full child journals, and benchmark metrics are validated by the existing importer.

`ml/training/tracking.py` uses the MLflow client to export hyperparameters, loss/gradient history, final benchmark metrics, trainable count, time, memory and evidence artifacts. All five actual training runs synced successfully; `ml/training/results/experiments-v1/mlflow-export.json` contains metadata retrieved back from MLflow to verify the export. MLflow's run/parameter/metric/artifact concepts are documented in [MLflow Experiment Tracking](https://mlflow.org/docs/latest/ml/tracking/).

By default, the client uses a local SQLite tracking database under `TRAINING_RUNS_DIR/tracking` and artifacts under `TRAINING_RUNS_DIR/mlflow-artifacts`. PostgreSQL remains the RecipeTriage registry; this separate SQLite file is only MLflow's default local store. Set `MLFLOW_TRACKING_URI` to an existing MLflow server if desired. No W&B account or hosted tracker is required. Secrets and environment contents are not logged. A tracking failure is displayed separately and does not erase successful training/benchmark evidence.

The actual run receipts refer to the tracking database used for this execution. For a new machine, re-export a delivered run to your own MLflow store without modifying its original journal:

```sh
python -m recipetriage_ml.training.tracking ml/training/results/experiments-v1/runs/771d20b1-77df-446b-b34d-b8259ddbc585/run.json
```

## Exact commands

For an existing environment, install the added dependencies after the earlier requirements:

```sh
python -m pip install -r backend/requirements.lock
python -m pip install -r ml/requirements-hf.lock -r ml/requirements-sft.lock
python -m pip install -r ml/requirements-experiments.lock
python -m pip install -e ./ml -e ./backend
python -m recipetriage_ml.training.qlora --probe
```

On macOS/Linux, activate Python 3.12 with `source .venv/bin/activate`. On PowerShell, use `.\.venv\Scripts\Activate.ps1`, or replace `python` with `.\.venv\Scripts\python.exe` without activating. CPU execution is intentional; do not interpret these runs as CUDA benchmarks.

Run the committed matrix configuration locally:

```sh
python -m recipetriage_ml.training.experiments --config ml/configs/experiments-v1.json --output ml/training/matrix-runs
```

Generate/edit a fresh config or submit it to the application registry:

```sh
python -m recipetriage_ml.training.experiments --write-config matrix.json
python -m recipetriage_ml.training.experiments --config matrix.json --api http://localhost:8000
```

The API submission returns a run ID immediately; follow **Training Lab → Experiments** for progress. The default committed config includes the four LoRA cells plus the matched QLoRA cell. The UI's ordinary matrix form creates the LR × rank grid; the **QLoRA Training** tab creates a matched two-method comparison. Both expose gradient checkpointing and optional MLflow export. Full configuration JSON is visible before starting.

Rebuild and import delivered evidence without retraining:

```sh
docker compose up --build --wait --wait-timeout 180
docker compose exec backend python -m app.experiments /app/ml/training/results/experiments-v1/experiment.json
docker compose exec backend python -m app.analysis /app/ml/evaluation/results/diagnostics-v1/base.json /app/ml/evaluation/results/diagnostics-v1/lora.json /app/ml/evaluation/results/diagnostics-v1/qlora.json
```

Open http://localhost:8080. Training Lab contains Experiments, QLoRA Training, LoRA Training and SFT Monitor. Evaluation Lab contains Failure Analysis, Shortcut Tests and Baseline. Small screens use the Workspace dropdown.

Selected adapters are included in the source ZIP and host project, but excluded from the Docker image to keep it smaller. Newly trained Docker adapters live in `/training`. A model cache and optimizer checkpoint directories are not included in the portable source ZIP.
