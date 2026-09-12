# LoRA from first principles, with a real rank experiment

## 1. What a Transformer projection does

A Transformer represents each token as a vector of hidden features. A linear projection mixes these features:

$$y = Wx+b,\qquad W\in\mathbb R^{d_{out}\times d_{in}}.$$

For example, one feature might help represent cooking time and another ingredients. Actual learned features are distributed; they are not neatly named recipe fields. Attention makes queries, keys and values with separate projections. It compares queries with keys to decide which tokens to attend to, then combines values. Another projection mixes the attention output. Feed-forward (MLP) projections expand, gate and contract the hidden representation. PyTorch stores `nn.Linear.weight` as `[out_features, in_features]`; for a batch of row vectors its computation is `X @ W.T + b`.

With $W=\begin{bmatrix}1&2\\0&1\end{bmatrix}$, $x=\begin{bmatrix}2\\1\end{bmatrix}$ and zero bias, the projection returns $Wx=[4,1]^T$.

## 2. Why full fine-tuning costs memory

Full fine-tuning updates every weight. With float32 Adam, each trainable parameter can require four bytes for its weight, four for its gradient and eight for two optimizer moments: approximately 16 bytes before activations and temporary buffers. The inspected base has 494,032,768 unique parameters, so that simplified total is about 7.36 GiB. It is an accounting example, not a measured full-fine-tuning run; dtype, optimizer and sharding change the total.

Backpropagation also needs intermediate activations. Freezing weights avoids storing their parameter gradients and optimizer state, but it does not remove the base model or all backward computation. Gradients still flow through frozen operations to reach earlier adapters.

## 3. Learn a small change instead

LoRA starts from:

$$W'=W+\Delta W,\qquad \Delta W=BA.$$

Choose a small rank $r$. Let $A\in\mathbb R^{r\times d_{in}}$ compress the input into $r$ directions and $B\in\mathbb R^{d_{out}\times r}$ expand that update into the output space. Then $BA$ has the same shape as $W$, but rank at most $r$. A rank is a capacity constraint, not a number of labels or attention heads. This is the central low-rank adaptation idea in the [original LoRA paper](https://arxiv.org/abs/2106.09685).

For rank 1, choose $A=[1,-1]$ and $B=[0.1,0.2]^T$. Then:

$$BA=\begin{bmatrix}0.1&-0.1\\0.2&-0.2\end{bmatrix},\quad Ax=1,\quad BAx=[0.1,0.2]^T.$$

Without extra scaling, our tiny projection becomes $W'x=[4.1,1.2]^T$. There are $r(d_{in}+d_{out})$ adapter parameters instead of $d_{in}d_{out}$ weight parameters. In this deliberately tiny 2×2 rank-1 example both counts are four; savings arise for large matrices with small rank. For an 896×896 projection, rank 4 needs 7,168 parameters versus 802,816 for the original weight.

## 4. Alpha, dropout and initialization

The implementation uses standard PEFT LoRA:

$$y=Wx+b+\frac{\alpha}{r}BA\,D_p(x).$$

`alpha` scales the adapter contribution; it is distinct from the optimizer's learning rate. `D_p` is dropout on the adapter input: in training it randomly zeros entries and scales retained entries by $1/(1-p)$; in evaluation it is the identity. The frozen base branch does not receive this adapter dropout. Our experiment uses `dropout=0` and `alpha=2r`, so both ranks use the same scaling of 2. That controls one factor, but does not make their gradients or effective update sizes identical.

PEFT's default initialization gives A random values and B zeros, so the initial update is exactly zero. At the first backward pass A's gradient is zero because B is zero; B can receive a nonzero gradient. Once B changes, both matrices can learn. We use `bias='none'`, `modules_to_save=None`, and `use_rslora=False`. Rank-stabilized LoRA uses a different scaling rule and is outside this experiment. See the [versioned PEFT LoRA API](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora).

`model.train()` enables dropout and training behavior. `model.eval()` disables dropout; it does not by itself disable gradients. The inference provider also uses `torch.inference_mode()`.

## 5. Inspect first, then select targets

The chosen checkpoint remains the existing benchmark's pretrained **Qwen/Qwen2.5-0.5B**, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. This keeps the before/after experiment tied to the same base. Playground's instruct checkpoint is a different model.

The inspector instantiates `Qwen2ForCausalLM` from that pinned configuration on PyTorch's meta device, ties its embedding/output weights, and enumerates actual `nn.Linear` modules. Training repeats the inspection on the loaded weights. It found 24 blocks, hidden size 896, intermediate size 4864, 14 query heads and 2 key/value heads. The shared key/value heads explain their narrower matrices.

| Actual suffix within each block | Weight shape [out, in] | Copies | Function | Default target |
|---|---:|---:|---|---|
| `self_attn.q_proj` | 896 × 896 | 24 | Queries | Yes |
| `self_attn.k_proj` | 128 × 896 | 24 | Keys | No |
| `self_attn.v_proj` | 128 × 896 | 24 | Values | Yes |
| `self_attn.o_proj` | 896 × 896 | 24 | Attention output | No |
| `mlp.gate_proj` | 4864 × 896 | 24 | MLP gate | No |
| `mlp.up_proj` | 4864 × 896 | 24 | MLP expansion | No |
| `mlp.down_proj` | 896 × 4864 | 24 | MLP contraction | No |
| `lm_head` outside the blocks | 151936 × 896 | 1 | Vocabulary logits | Excluded; tied to token embeddings |

The default adapts query and value projections to keep this exercise small. It is a controlled choice, not a claim that these are universally best. The UI also offers all attention projections or attention plus MLP. A policy must resolve to real linear modules in every block. For example, the full path `model.layers.0.self_attn.q_proj` is verified, and 48 full paths are supplied to PEFT. Unsupported architectures fail explicitly rather than reusing Qwen names.

PEFT can compress a long full-path target list into equivalent suffixes in the saved config. The loader resolves these saved suffixes against the exact base tree before loading; it rejects unknown names and matches outside the eligible projections. The run's `parameter_report.json` retains all full target paths regardless of PEFT's serialization.

For the default query/value policy:

$$N_{trainable}=24r[(896+896)+(896+128)]=67,584r.$$

| Rank | Alpha | Trainable parameters | Frozen parameters | Trainable share of base + adapter |
|---:|---:|---:|---:|---:|
| 4 | 8 | 270,336 | 494,032,768 | 0.0547% |
| 16 | 32 | 1,081,344 | 494,032,768 | 0.2184% |

The report counts `requires_grad=True` tensors after insertion, checks the formula, and rejects any trainable tensor outside A/B. Unit tests take an actual optimizer step on a tiny Qwen model and verify that base weights remain byte-for-byte unchanged. Tiny random matrices in tests are not synthetic recipe training data.

## 6. Frozen weights, gradients and checkpoints

`requires_grad=False` freezes W and the base biases. A/B remain trainable during SFT. Only these parameters receive optimizer updates; the base is still used in every forward pass. Higher rank increases update capacity and adapter storage, but is not a guarantee of better labels.

The selected `adapter/` export contains `adapter_model.safetensors`, `adapter_config.json`, tokenizer files and our checksum manifest. It does not contain the original 494M base weights. Loading requires the exact base checkpoint plus the adapter. The manifest checks all exported files and pins the base revision. Checksums detect accidental changes relative to the manifest; they are not a signed claim of authenticity.

An adapter export supports inference. A full training-resume checkpoint additionally needs optimizer, scheduler, RNG and trainer state; the trainer writes those in `checkpoints/checkpoint-*`. The curated downloadable experiment keeps the selected adapters and evidence, not optimizer checkpoint copies. Read the [PEFT checkpoint format](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/checkpoint).

## 7. Controlled experiment and actual results

Experiment `cccedca9-e768-4df5-b3d3-b04d8b837387`, run on this Mac's CPU on 2026-09-09, used ranks 4 then 16 in separate, fresh Python processes. Both used the same immutable 5/1/1 teaching split, seed 42, q/v targets, alpha/r=2, dropout 0, three epochs, learning rate 0.0002, microbatch 1, accumulation 2, sequence length 768, float32 and four CPU threads. Each completed nine optimizer updates. AdamW, constant LR, no warmup/weight decay, clipping norm 1, non-reentrant activation checkpointing and assistant-only target masks match the earlier SFT lesson. Epoch checkpoints were selected solely by minimum validation loss.

The unchanged ten-case RecipeTriage-Bench-v1 ran before and after **each** training run with greedy decoding and 128 output tokens. Both fresh base baselines reproduced Micro-F1 48.48%, macro-F1 36.50%, exact match 0%, and syntactically valid JSON 50%.

| Rank | Trainable | Training seconds | Start RSS MiB | Peak RSS MiB | Micro-F1 | Macro-F1 | Exact match | Valid JSON | Best validation loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 270,336 | 29.74 | 2,032.28 | 2,966.30 | 41.38% | 28.16% | 0% | 60% | 2.3731 |
| 16 | 1,081,344 | 29.52 | 2,090.27 | 2,941.23 | 0% | 0% | 0% | 80% | 1.9261 |

**Neither rank improved classification versus the base.** Rank 16 often generated an `explanation` without the required `labels`; syntactically valid JSON can still violate the response schema. Missing or unusable label sets score as empty predictions. This is why its JSON validity increased while F1 fell to zero. Its lower validation loss is teacher-forced token prediction on one recipe, not evidence of better free-generation triage.

Memory is process resident set size sampled every 20 ms during `trainer.train()`. It includes base weights, activations, libraries, gradients, optimizer state and allocator caches. The timing includes epoch validation/checkpoint writes and excludes initial validation, both benchmarks and the final adapter export. Sampling can miss short peaks. The approximately 25 MiB lower peak for rank 16 is not a measured adapter-memory saving: four times the trainable parameters does not imply four times total RSS, and one run per rank does not establish a performance ordering. OS caching and allocator behavior differ across fresh processes.

This seed has provisional labels, only one validation recipe, and no training examples for `dessert` or `have-most-of-this`. The benchmark has ten provisional cases and is already repeatedly observed for teaching. No best rank is selected automatically, and these results cannot establish population quality. Do not repeatedly tune against this benchmark and then call it an untouched test set.

Both exported adapters were loaded into fresh base models and reproduced the first recorded benchmark answer exactly under greedy decoding. All parameters were frozen after inference loading. Each result is saved as `reload_verification.json`; this is a one-case persistence check, not a second full benchmark.

## 8. Run and inspect the complete code

Docker starts the existing application:

```sh
docker compose up --build --wait --wait-timeout 180
docker compose exec backend python -m app.lora /app/ml/training/results/lora-v1/experiment.json
```

The import is idempotent and makes no training or hosted requests. Open http://localhost:8080, choose **Training Lab → LoRA Training**, and inspect the saved two-rank comparison. **SFT Monitor** remains available in the second tab. Choose at least two ranks and click **Run rank experiment** only to launch another local experiment. The server saves live progress and raw benchmark evidence in PostgreSQL, with adapter/checkpoint artifacts in the `/training` Docker volume. It serializes experiments, SFT runs and benchmark jobs with the same queue lock.

For Python 3.12 host setup, follow SFT.md's installation commands, including the pinned HF/SFT requirements and editable `ml`/`backend` packages. From the repository root with that environment activated:

```sh
python -m recipetriage_ml.training.lora --inspect architecture-inspection.json
python -m recipetriage_ml.training.lora_experiment --output ml/training/lora-runs
```

The default experiment runs ranks 4 and 16. To edit a complete validated configuration:

```sh
python -c "from recipetriage_ml.training.lora_experiment import ExperimentConfig; from recipetriage_ml.training.config import SFTConfig; from recipetriage_ml.training.data import seed_snapshot; print(ExperimentConfig(training=SFTConfig(dataset_version=seed_snapshot()['metadata']['version'])).model_dump_json(indent=2))" > lora-config.json
python -m recipetriage_ml.training.lora_experiment --config lora-config.json --output ml/training/lora-runs
```

PowerShell uses the same module commands after `.\.venv\Scripts\Activate.ps1`, or substitute `.\.venv\Scripts\python.exe` for `python` without activation. For macOS/Linux activate with `source .venv/bin/activate`. All workers use CPU, so CUDA and Metal are unnecessary. First use downloads the pinned base; later offline runs can use `HF_HUB_OFFLINE=1` with the same populated `HF_HOME`. On PowerShell set `$env:HF_HUB_OFFLINE = '1'` instead of a POSIX command prefix.

Load a delivered adapter from the repository root:

```python
from pathlib import Path
from recipetriage_ml.training.lora import load_adapter

adapter_dir = Path('ml/training/results/lora-v1/runs/f343f65d-96b6-4977-9787-d039607553d1/adapter')
model, tokenizer = load_adapter(adapter_dir)
# model is in eval mode and its parameters are frozen for inference.
```

The training module's `load_adapter` accepts the adapter directory itself. The compatibility wrapper `recipetriage_ml.training.sft.load_adapter(run_directory)` accepts its parent run directory and returns a RecipeTriage `TrainedProvider` with recorded provenance and prompt formatting. Large adapter files are included in the source download but excluded from Docker image build context; newly trained adapters live in the mounted volume.

Tests:

```sh
python -m pytest -q
cd frontend
npm test
npm run build
cd ..
docker compose --profile test run --build --rm backend-tests
```

The last command includes PostgreSQL tests. Host tests skip them unless `RUN_DB_TESTS=1` and PostgreSQL is available. No test sends hosted requests or trains on generated recipes.

## 9. Major files to read

| File | Responsibility |
|---|---|
| `ml/training/lora.py` | Inspect actual modules, select target paths, attach PEFT, count/audit parameters, save/load verified adapters, sample training resources. |
| `ml/training/architecture_qwen.json` | Complete pinned-model inspection: all 169 linear modules, shapes, policies and architecture hash. |
| `ml/training/config.py` | Validated SFT/LoRA knobs and resolved fixed settings; alpha defaults to twice the rank. |
| `ml/training/sft.py` | Existing masked SFTTrainer loop now calls inspected LoRA helpers; logs actual memory, exports the selected adapter and benchmarks it. |
| `ml/training/lora_experiment.py` | Validates rank comparisons, launches isolated sequential workers, persists failures/progress, checks comparability and writes CSV/JSON. |
| `ml/training/data.py` | Revalidates immutable snapshots, rejects leakage and overlength inputs, constructs supervised token masks. |
| `ml/training/runs.py` | Atomic per-run journals, frozen benchmark execution and before/after scoring. |
| `backend/app/lora.py` | Architecture and experiment API, background workers, PostgreSQL persistence, evidence download/import, restart recovery. |
| `backend/app/migrate.py` | Adds the `lora_experiments` JSONB table without deleting existing data. |
| `backend/app/training.py`, `backend/app/benchmarks.py` | Shared queue gates now also prevent overlap with an active rank experiment. |
| `backend/app/recover_benchmarks.py` | Marks interrupted experiments/runs explicitly on process startup. |
| `frontend/src/LoRATraining.jsx` | Architecture explorer, rank/configuration controls, live progress, result table, loss history and evidence links. |
| `frontend/src/TrainingLab.jsx` | Tabs and lazy loading for LoRA Training and the earlier SFT Monitor. |
| `tests/ml/test_lora.py` | Mathematical counts, frozen weight updates, dropout, round-trip/integrity, compressed targets, comparison guards and worker crashes. |
| `tests/backend/test_lora_api.py` | Architecture API, PostgreSQL import/immutability, queue exclusion, failure cleanup and restart behavior. |
| `frontend/src/LoRATraining.test.jsx` | Navigation-independent page behavior, rank controls, request construction, zero scores and explicit errors. |
| `ml/training/results/lora-v1/` | Actual experiment CSV/JSON, both full journals, raw benchmarks, parameter reports, selected adapters and reload verification. |

## Check your understanding

1. If W is 896×896 and rank is 4, what are A's and B's shapes, and how many trainable values do they contain?
2. Why can gradients flow through W even though W stays frozen?
3. What changes if rank increases while alpha stays fixed? How does this experiment control that factor?
4. Why can rank 16 produce more valid JSON but zero label F1?
5. Why does an adapter checkpoint need its original base model, and what extra state would resuming training require?
