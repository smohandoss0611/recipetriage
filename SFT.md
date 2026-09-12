# Supervised fine-tuning: lesson, code and measured results

## Streamlit configuration checks

In **Training Lab → SFT Monitor**, choose a **Training provider**, select **Check configuration**, and click **Submit training action**. Both `local` and `fireworks` are supported by `/api/v1/training/preflight`; local checks do not require a Fireworks key or account.

Local checking validates the Pydantic configuration, immutable dataset contents, known benchmark overlap and installed training-package metadata. It returns split counts, teaching-draft warnings and resolved parameters without creating a run, loading model weights, downloading a tokenizer or submitting provider requests. Package presence does not prove runtime compatibility: training still checks imports, tokenized lengths, answer masking, model loading, LoRA targets and memory; QLoRA additionally checks actual CPU NF4 capability. A passed configuration check is not a completed training run.

Managed checking retains its Fireworks account/model/dataset/deployment-shape checks. Choose **Start training** only when you intend to submit the separate training action. To use a saved dataset, set `dataset_version` in the training JSON to the version returned by Dataset Studio; the default JSON starts from the bundled teaching seed.


SFT updates a pretrained language model using examples of the answer we want it to produce. For RecipeTriage, an instruction and recipe are the input; the desired assistant answer is JSON containing `labels` and `explanation`. Instruction tuning is SFT on instruction/answer examples. It changes weights; inference uses those weights to generate an answer.

## The causal language-model objective

Let a token sequence be $x_1,\ldots,x_T$. A causal model predicts each next token from the preceding tokens:

$$p_\theta(x_1,\ldots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).$$

Here $\theta$ denotes trainable parameters. In this exercise only LoRA adapters are trainable. The original base weights stay frozen. The supervised loss is:

$$\mathcal L(\theta)=-\frac{1}{|M|}\sum_{t\in M}\log p_\theta(x_t\mid x_{<t}),$$

where $M$ contains assistant-answer token positions, including EOS. We feed the correct previous answer tokens during training (*teacher forcing*). At inference, the model instead conditions on its own generated answer tokens.

For a tiny two-token answer with correct-token probabilities 0.8 and 0.5, loss is $-(\log 0.8+\log 0.5)/2\approx0.458$. Higher probability on the correct tokens gives lower loss. This is token cross-entropy over the vocabulary; it is not a separate binary classifier loss for each recipe label.

`input_ids` contains the whole prompt and answer. The training tensor named `labels` contains target **token IDs**, which differ from the recipe's semantic labels such as `weekend-project`. We replace prompt and padding targets with `-100`, which means “ignore this target in cross-entropy.” Masked prompt tokens still provide context and participate in the computation producing the answer.

```text
input:   [system instruction] [recipe] [Assistant:] [{"labels": ...}] [EOS] [PAD]
labels:  [-100 ...          ] [-100 ] [-100 ...  ] [answer token IDs] [EOS] [-100]
```

The causal model shifts predictions and labels internally: the logit at position $t-1$ predicts the token at $t$. We do not shift them a second time. `encode_example()` checks the prompt/answer token boundary, includes EOS, and rejects overlength sequences. The collator pads targets with `-100`, even though PAD and EOS share a token ID here. Padding is identified by position, so the actual EOS target is preserved.

## From your explicit loop to SFTTrainer

In the previous lesson you called forward, loss, `backward()`, `optimizer.step()` and `zero_grad()` directly. `SFTTrainer.train()` schedules those same operations, creates batches, accumulates gradients, clips them, logs progress, evaluates, and saves checkpoints. It does not choose appropriate examples or prove model quality. The pinned interface is documented in [TRL 0.24 SFTTrainer](https://huggingface.co/docs/trl/v0.24.0/sft_trainer). Our code still owns the versioned dataset, target masks, hyperparameters and held-out benchmark.

Read `ml/training/sft.py` in order: validate snapshot → encode and inspect target masks → run base benchmark → attach LoRA → construct trainer → initial validation → train → export best validation checkpoint → run the unchanged benchmark → compare actual scores.

For a frozen matrix $W$, LoRA learns a low-rank update $\Delta W=(\alpha/r)BA$. Only $A$ and $B$ receive optimizer updates. We adapt attention's query and value projections, `q_proj` and `v_proj`. This is supervised fine-tuning, although most parameters remain frozen.

## Every training parameter

The editable values live in `ml/training/config.py`; each run records editable and fixed settings. The shipped local experiment used the defaults below. None were selected by searching benchmark scores.

| Parameter | Used value | Meaning and tradeoff |
|---|---|---|
| Dataset | Immutable Dataset v1, `v1-63135e…0158f` | Five training recipes, one validation recipe, one untouched test recipe. Hashes and split IDs are saved. |
| Base model | `Qwen/Qwen2.5-0.5B` | Same pretrained base as the local baseline. Revision is pinned to `060db6499f32faf8b98477b0a26969ef7d8b9987`. |
| Prompt format | `plain-role-completion-v1` | Same system/user/assistant prefix before and after SFT; matches the existing base benchmark. |
| Epochs | 3 | Three passes through the five training examples. More passes can memorize them. |
| Learning rate | 0.0002 | Scale of adapter updates. Larger values learn faster but can destabilize or erase useful behavior. |
| Scheduler / warmup | Constant / 0 steps | Learning rate stays fixed throughout this tiny run. Warmup normally ramps it up initially. |
| Microbatch size | 1 | One recipe per forward/backward pass, keeping memory manageable. |
| Gradient accumulation | 2 | Accumulate two microbatches before stepping; gradients are cleared at update boundaries. |
| Effective batch | Usually 2 | $B_{effective}=B_{micro}\times accumulation\times devices$. One CPU process here. The last update each epoch has only one recipe. |
| Optimizer steps | 9 | $3\times\lceil5/(1\times2)\rceil=9$ for this configuration and trainer. An optimizer step differs from a forward pass. |
| Sequence length | 768 tokens | Maximum prompt + answer + EOS length. Actual training lengths: 350–421 tokens; validation: 403. No silent truncation. |
| Packing | Off | Each recipe remains a separate example, simplifying masks and inspection. |
| Objective | Assistant-only causal cross-entropy | Prompt and padding targets are ignored. Answer JSON and EOS are supervised. |
| Optimizer | PyTorch AdamW | Adaptive first/second gradient moments; optimizer defaults: beta1=0.9, beta2=0.999, epsilon=1e-8. |
| Weight decay | 0 | No additional parameter shrinkage in this demonstration. |
| Gradient clipping | Maximum norm 1 | Bounds the update input to reduce spikes. Logged gradient norm is measured before clipping. |
| LoRA rank | 8 | Capacity of the small update matrices. This run trains 540,672 of 494,573,440 parameters, including adapters. |
| LoRA alpha | 16 | Scaling multiplier; $\alpha/r=2$. |
| LoRA targets / bias | `q_proj`, `v_proj` / frozen | Adapt those attention projections; do not train base biases or embeddings. |
| LoRA dropout | 0 | No adapter dropout in this minimal exercise. Regularization would need validation on more data. |
| Seed | 42 | Local initialization and data-order seed. Exact cross-platform bitwise equality is not promised. |
| Device / dtype | CPU / float32 | Works without CUDA on macOS, Windows and Linux. Not optimized for training speed. Four CPU threads by default (`HF_CPU_THREADS`). |
| Activation checkpointing | On, non-reentrant | Recomputes activations during backward to save memory. Distinct from saving model checkpoints to disk. |
| Validation | Initially and after every epoch; batch 1 | Teacher-forced loss on the one validation recipe. No gradients or weight updates from validation. |
| Checkpoint selection | Lowest validation loss | Select among trained epoch checkpoints, never using benchmark scores. Initial untrained validation is recorded separately. |
| Checkpoint retention | Up to 2 trainer checkpoints plus exported adapter | Trainer checkpoints contain resume state. `adapter/` contains selected adapter weights and tokenizer. |
| Logging | Every optimizer step | Saves loss, gradient norm, learning rate and epoch; also evaluation loss. No external experiment tracker. |
| Benchmark decoding | Temperature 0, 128 new tokens, reasoning disabled | Same fixed ten cases and scoring policy before/after. These are inference settings, not training settings. |

With batches containing different numbers of answer tokens, token-loss reduction and accumulation details matter. We use the pinned Transformers/TRL implementation and retain its logged values; a large effective batch is not a guarantee of identical floating-point results to one physically large batch.

Validation loss measures prediction of known answer tokens. It does not directly measure complete JSON validity or correct label sets during free generation. A falling training loss with rising validation loss is one sign of overfitting. A single validation recipe cannot reliably diagnose generalization. The training split has zero `dessert` and `have-most-of-this` examples; training cannot fix that coverage gap by itself.

## Run locally

From the repository root, with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.lock
python -m pip install -r ml/requirements-hf.lock -r ml/requirements-sft.lock
python -m pip install -e ./backend -e ./ml
python -m recipetriage_ml.training.sft
```

On Windows PowerShell, use `py -3.12 -m venv .venv`, then `\.venv\Scripts\Activate.ps1` with a leading dot: `.\.venv\Scripts\Activate.ps1`. If activation is restricted, run `.\.venv\Scripts\python.exe` in place of `python`; no shell policy change is required. The training code explicitly uses CPU, so no macOS Metal or NVIDIA configuration is needed. On Linux the Dockerfile installs the CPU PyTorch wheel from the official CPU index before other requirements.

The CLI sets a project-local Hugging Face cache unless `HF_HOME` is already defined. First use downloads the pinned model. The local run directory is `ml/training/runs/<UUID>/`; it contains `run.json`, masked-token counts, logical dataset exports, checkpoint directories, selected adapter, `loss_by_step.csv`, and complete `baseline.json` / `benchmark.json` evidence. Adapter provenance and weight checksum are recorded. `load_adapter(run_directory)` always loads the exact base revision and verifies adapter bytes.

For editable JSON configuration:

```sh
python -c "import json; from recipetriage_ml.training.data import seed_snapshot; from recipetriage_ml.training.config import SFTConfig; print(json.dumps(SFTConfig(dataset_version=seed_snapshot()['metadata']['version']).model_dump(), indent=2))" > sft-config.json
python -m recipetriage_ml.training.sft --config sft-config.json
```

To use a later exported version, add `--dataset path/to/v1-HASH` and set the same `dataset_version` in the config. The runner rebuilds the snapshot and verifies its bytes, split metadata, and benchmark source/body/group overlap. The test split is not exported to the trainers.

## Run in the application

```sh
docker compose up --build --wait --wait-timeout 180
docker compose exec backend sh -c 'python -m app.training /app/ml/training/results/sft-v1/*/run.json'
```

Open **http://localhost:8080 → Training Lab**. The import displays the three bundled actual records without training or paid inference. Select the local completed run to see its loss table, checkpoint and benchmark regression. **Start local SFT** starts a fresh run using the displayed parameters. Opening the page only reads data.

Training runs initiated through the UI persist to PostgreSQL's `training_runs` table. A separate `training_artifacts` Docker volume keeps adapter/checkpoint files and file journals. `docker compose down` preserves both. `docker compose down --volumes` deletes the database, training artifacts and model cache.

To copy actual container training files to your machine:

```sh
docker compose cp backend:/training ./training-artifacts
```

Host CLI results can be imported into a host API database with `python -m app.training path/to/run.json`. To import into Docker, first `docker compose cp path/to/run.json backend:/tmp/training-run.json`, then `docker compose exec backend python -m app.training /tmp/training-run.json`. Importing journals does not import adapter bytes; copy those separately when needed. The bundled reference adapter is shipped in the source archive but excluded from the application image.

The API responds 202 and runs work outside the request handler using a background worker thread. One API process and a PostgreSQL queue gate serialize training and benchmark jobs. Restarted local work is marked interrupted; its files remain for inspection. This teaching deployment is not a distributed job queue. Do not start competing CLI jobs alongside an API training run.

## Fireworks path and current limitation

Use **Check Fireworks support** before starting a managed run. It makes read-only account/model requests. The current seed has 5/1/1 examples. Fireworks documents at least three examples per uploaded dataset, so its validation split is too small. The initial working-copy check of `accounts/fireworks/models/qwen3p7-plus` also found no reported LoRA tuning support. At the time of that preflight, the GitHub project’s `.env` selected `accounts/fireworks/models/llama-v3p3-70b-instruct`, which does report supervised LoRA support. Its current check is still blocked by the small validation split and missing deployment shape. The current inference default is Nemotron Lightning 3.5; managed training requires a separately verified `FIREWORKS_SFT_MODEL` and support preflight. Both saved preflights are **blocked**; no datasets were uploaded, no managed training job or deployment was created, and no managed post-SFT scores exist. We did not duplicate recipes, move the test example or generate synthetic rows to bypass this constraint. [Dataset format and limits](https://docs.fireworks.ai/fine-tuning/fine-tuning-models), [model capability fields](https://docs.fireworks.ai/api-reference/get-model).

When you have a sufficiently large hand-reviewed dataset version, configure these server-side values in `.env`:

```dotenv
FIREWORKS_API_KEY=your-existing-key
FIREWORKS_ACCOUNT_ID=your-account-id
FIREWORKS_SFT_MODEL=accounts/fireworks/models/a-currently-tunable-model
FIREWORKS_SFT_DEPLOYMENT_SHAPE=accounts/fireworks/deploymentShapes/a-compatible-shape
```

The last two values are placeholders, not advertised available models. Select a currently supported model and compatible shape from your Fireworks account. An account ID can be discovered automatically only if the key resolves to exactly one account. Recreate the API after changing its environment: `docker compose up -d backend`.

```sh
python -m recipetriage_ml.training.fireworks_sft --preflight-only
python -m recipetriage_ml.training.fireworks_sft --config sft-config.json --dataset path/to/v1-HASH
python -m recipetriage_ml.training.fireworks_sft --resume path/to/run.json --dataset path/to/v1-HASH
```

For managed training, the config must have `"provider": "fireworks"`. The supported path uploads separate train/validation JSONL files, submits SFT, polls the saved job ID, then automatically evaluates the output model. Both providers receive exactly the same logical messages and split membership. Fireworks message weights are 0 for system/user and 1 for assistant; test rows are never uploaded. Fireworks uses its registered chat renderer, so tokenization and rendered tokens can differ. Managed sequence handling must be inspected in the provider's rendered samples; our local no-truncation check does not certify a different provider tokenizer.

Shared settings sent to Fireworks: epochs, learning rate, maximum context length, LoRA rank, weight decay 0, constant scheduler and warmup 0. `batchSizeSamples` receives the local microbatch × accumulation product. We omit deprecated `batchSize`, `gradientAccumulationSteps` and custom `jinjaTemplate`. Hardware, microbatch scheduling, exact optimizer details, LoRA alpha/targets/dropout, seed, clipping and checkpoint retention remain provider controlled. Its final output model is used; we do not claim it uses our local minimum-validation-loss checkpoint policy. The submitted JSON and provider job state are retained. [Current SFT request fields](https://docs.fireworks.ai/api-reference/create-supervised-fine-tuning-job).

Before and after managed training, the runner creates one temporary **preemptible** deployment using the selected shape, waits at most ten minutes for capacity, runs the fixed benchmark, and requests deletion in a `finally` block. It explicitly routes requests to that deployment. Capacity can be unavailable or reclaimed; failures remain visible. A saved deployment name permits cleanup after a crash. Cleanup errors are displayed and can be retried by resuming monitoring. Inspect the Fireworks console after forcibly stopping the host while an evaluation deployment is active. [Preemptible evaluation](https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models), [deployment API](https://docs.fireworks.ai/api-reference/create-deployment).

Monitoring yields after 30 minutes if a remote job is still active; **Resume monitoring** reuses the saved job ID instead of creating another job. A host restart does not cancel remote training. Signed metrics/log URLs and API keys are excluded from public journals. Remote loss files are available through the provider console when supplied; the UI records progress and does not substitute invented local loss values.

Compare each tuned model with its own before-SFT baseline. Comparing a local 0.5B model to a different hosted model measures different systems; it does not isolate the effect of SFT. The previously saved 94.1% hosted baseline is not the baseline for the local adapter.

## Actual run on 2026-09-09

Local training run: `30f1b50c-52b3-4336-ae9e-b4d7d2586e08`. Nine optimizer updates; selected checkpoint 9 based only on validation loss. Initial validation loss **2.8582**, selected **2.1284**. Complete raw outputs, per-label metrics and latency live under `ml/training/results/sft-v1/local/`.

| Metric | Before SFT | After SFT |
|---|---:|---:|
| Micro-F1 | 48.48% | 38.71% |
| Macro-F1, all seven labels | 36.50% | 16.36% |
| Exact label-set match | 0/10 | 0/10 |
| JSON validity | 50% | 90% |
| Usable complete schema-valid responses | 4/10 | 8/10 |
| Mean latency, all attempts | 4,235 ms | 2,823 ms |
| p95 latency, ten attempts | 4,975 ms | 5,229 ms |

**No measured classification improvement.** Micro-F1 decreased by 9.78 percentage points while formatting improved. These ten provisional cases and one repetition do not establish general accuracy or stable latency. We did not tune additional runs against these benchmark outcomes.

The saved adapter was reloaded against the pinned base, its checksum verified, and its first benchmark response matched the saved output exactly. PyTorch/TRL completed training without an exception. The console records tokenizer PAD/BOS alignment and PEFT's offline vocabulary-config warning; the vocabulary was not resized. Test runs also emit existing Starlette/httpx deprecation warnings. See `VERIFICATION.md` for the final application checks.

## Check your understanding

1. Why are prompt tokens available to the model even when their labels are `-100`?
2. With five examples, microbatch 1, accumulation 2 and three epochs, why are there nine optimizer steps?
3. How can validation loss decrease while RecipeTriage Micro-F1 decreases?
4. What is the difference between activation checkpointing and a saved model checkpoint?
5. Why would copying the validation recipe three times make the managed experiment less trustworthy?
