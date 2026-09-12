# Measured deployment candidates

The supported formats are Hugging Face float32 safetensors and bitsandbytes NF4 safetensors. The merge script validates the saved full-precision LoRA artifact, uses PEFT safe merge and writes a new immutable directory. NF4 conversion quantizes all 168 inspected Transformer linear projections with double quantization and float32 CPU compute. Embeddings and other non-quantized components explain why total storage is not exactly one eighth of FP32. GGUF, AWQ and GPTQ are not implemented or claimed as tested formats.

The NF4 training adapter is kept separate; the merge script explicitly rejects direct merging of an NF4 adapter. The demonstrated standalone merge uses the full-precision late-six LoRA run. It then quantizes that merged model. This avoids silently treating quantized training weights as an equivalent full-precision training run.

All three candidates were loaded in fresh processes and evaluated with the identical ten-case RecipeTriage-Bench-v1, temperature 0, 128 completion tokens and the original prompt format. Each also ran the unchanged 12-row title-only suite. Results include full raw responses and identities.

| Candidate | Size MiB | Peak RAM MiB | Mean ms | p95 ms | Micro F1 | Macro F1 | Schema valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base-fp32 | 1899.8 | 2293.6 | 4028 | 5152 | 48.48% | 36.50% | 40% |
| late-six-merged-fp32 | 1899.8 | 2327.8 | 3745 | 4944 | 43.24% | 32.65% | 50% |
| late-six-merged-nf4 | 711.1 | 1193.7 | 20764 | 23260 | 0.00% | 0.00% | 0% |

Measurements are one sequential CPU run on an Apple M4 Mac, four PyTorch threads. RAM is sampled total process RSS during benchmark generation; it is not tensor-only memory or a claim about peak loading memory. Model loading time is saved separately and excluded from generation latency. VRAM is N/A because no CUDA GPU was used. Cold disk caches, scheduling, hardware and repetitions can change these observations. Raw measurements are in `ml/deployment/results/v1/`.

Under default operational limits (Micro F1 ≥ 0.48, schema validity ≥ 0.40, RAM ≤ 4096 MiB, storage ≤ 3072 MiB, p95 ≤ 15000 ms), the recommended candidate is **base-fp32**. The selector first rejects constraint failures, then maximizes Micro F1, schema validity and finally speed. Size is a constraint only. NF4 is smaller but fails quality and latency limits. None meets the stricter registry promotion gate. No production pointer was changed.

Merged FP32 exactly reproduced every raw response from the source adapter's ten-case benchmark. All ten prompt token-ID sequences in all three exports matched the original pinned tokenizer. Transformers emitted a Mistral-regex warning when loading these local Qwen tokenizers; it was retained in logs and checked with those exact token comparisons. No tokenizer behavior was changed to hide the warning.

## Reproduce exports and measurements

Install the pinned host HF/SFT/experiment requirements first (see SFT.md and EXPERIMENTS.md). From the root, use new output directories. These commands perform real inference and require several GB of disk:

```sh
.venv/bin/python -m recipetriage_ml.deployment.merge_model --base --output ml/deployment/artifacts/new-base-fp32
.venv/bin/python -m recipetriage_ml.deployment.merge_model --adapter ml/analysis/results/layers-v1/runs/b530230c-018f-4ae5-a3cc-5f6bbf713fd3 --output ml/deployment/artifacts/new-merged-fp32
.venv/bin/python -m recipetriage_ml.deployment.quantize --source ml/deployment/artifacts/new-merged-fp32 --output ml/deployment/artifacts/new-merged-nf4
.venv/bin/python -m recipetriage_ml.deployment.benchmark --model ml/deployment/artifacts/new-base-fp32 --name new-base-fp32 --output ml/deployment/results/v1/new-base-fp32.json
.venv/bin/python -m recipetriage_ml.deployment.benchmark --model ml/deployment/artifacts/new-merged-fp32 --name new-merged-fp32 --output ml/deployment/results/v1/new-merged-fp32.json
.venv/bin/python -m recipetriage_ml.deployment.benchmark --model ml/deployment/artifacts/new-merged-nf4 --name new-merged-nf4 --output ml/deployment/results/v1/new-merged-nf4.json
```

PowerShell uses `.venv\Scripts\python.exe`. The scripts set CPU loading explicitly; CUDA/Metal results cannot be inferred from these measurements. Set `HF_HOME` to your existing model cache as described in LEARNING.md to avoid a duplicate download. The full-source archive contains scripts, results and small adapters, but excludes the multi-GB standalone exports. The user's local project contains the measured exports under `ml/deployment/artifacts/`.

To make newly saved result JSON visible to the container UI, rebuild the backend. Compose mounts `ml/deployment/artifacts` read-only at `/deployment`. Registration records a candidate only:

```sh
docker compose up -d --build --wait backend
docker compose exec backend python -m app.registry --deployment /app/ml/deployment/results/v1/base-fp32.json --artifact-directory /deployment/base-fp32
docker compose exec backend python -m app.registry --deployment /app/ml/deployment/results/v1/merged-fp32.json --artifact-directory /deployment/merged-fp32
docker compose exec backend python -m app.registry --deployment /app/ml/deployment/results/v1/merged-nf4.json --artifact-directory /deployment/merged-nf4
```

For exports you regenerated, substitute the new filenames/directories and register their corresponding new evidence. Old evidence cannot be used with changed artifact checksums. These Docker commands work in PowerShell too. The source archive alone can display saved comparisons, but registering or serving an export requires its actual files.

## Major files

`merge_model.py` verifies and merges FP32 LoRA or exports the pinned base; `quantize.py` converts a verified export to NF4 and audits coverage; `benchmark.py` verifies exports, measures performance and selects among feasible candidates. `gates.py` supplies stricter production criteria. `backend/app/deployment.py` serves saved comparisons, `registry.py` owns immutable models and stage transitions, and `frontend/src/DeploymentStudio.jsx` exposes operational constraints, metrics, promotion and rollback.

The notebook's layer ranking is a diagnostic result, not a reason to automatically deploy its winner. The benchmark and operational limits determine the recommendation, and the independent registry gate still blocks promotion.
