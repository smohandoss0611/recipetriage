# Prompt 1 — Tokenization and inference

This lesson covers the inference and inspection phase. The repository also implements Dataset Engineering v1 and a separate pretrained base-model comparison in Evaluation Lab; read DATASET_V1.md and [BENCHMARK.md](BENCHMARK.md) for those lessons. No training, fine-tuning, or weight updates are implemented.

The current Fireworks default is `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`; earlier model IDs and experiment results below are historical. Baseline can disable hosted reasoning; single-model Playground now disables it to reserve the small completion budget for JSON.

## Concepts, connected to one recipe

A pretrained language model has learned patterns by predicting text during training. An instruct model has received additional training to follow requests. We use the existing Qwen2.5-0.5B-Instruct checkpoint, rather than teaching a model from scratch.

The tokenizer converts `Easy ravioli takes 70 minutes.` into vocabulary pieces. A token can be a word, part of a word, punctuation, a digit, or a byte-level piece. Each token has a token ID: an integer address in this tokenizer's vocabulary. The locally observed sequence has 10 tokens, with IDs `[36730, 435, 6190, 14225, 4990, 220, 22, 15, 4420, 13]`. Those IDs are specific to the pinned Qwen tokenizer; they are not universal word codes.

The chat template adds the model's expected message-role markers and assistant-generation prefix. We supply a system message containing classification instructions and a user message containing recipe JSON. The local provider calls `apply_chat_template`; Fireworks receives structured messages and applies its hosted model's formatting. Do not manually paste Qwen's special markers into messages sent to a different model.

A tensor is a numeric array. For one recipe, input IDs have shape `[1, input_token_count]`. The model outputs logits with shape `[1, input_token_count, vocabulary_size]`. Logits are raw scores, not probabilities. Select the final position's logits to predict the next token.

Softmax maps logits to a probability distribution. At positive temperature T, `softmax(logits / T)` sharpens the distribution when T is smaller and flattens it when T is larger. The notebook computes this demonstration in float64: finite-precision accumulation over a large vocabulary can make a float32 sum slightly different from one. Next-token probability is not a calibrated probability that a recipe label is correct.

Decoding repeatedly chooses a token and appends it to the sequence. Greedy decoding chooses the highest-scoring token; sampling chooses according to the distribution. Our API interprets temperature 0 as greedy decoding. This is not a literal division by zero. Positive temperature enables sampling. Greedy decoding reduces variability, but hardware/library differences can still affect output; it is not a cross-platform determinism guarantee.

Inference uses existing weights. Training computes a loss against targets, backpropagates gradients, and updates weights with an optimizer. `model.eval()` disables training-specific layer behavior, while `torch.inference_mode()` disables gradient tracking. Neither trains a model.

`max_new_tokens` caps output length, excluding the input. A 383-token input with a 128-token budget allows at most 128 additional tokens, not 128 total. The provider can stop earlier. A cap or time limit may truncate JSON. We expose `finish_reason`, preserve raw output, and reject incomplete generations even if their partial text happens to parse. The hosted API calls its output-budget field `max_tokens`; the adapter maps our setting to that name.

## Use the app

Run Compose as described in README.md, then open http://localhost:8080:

1. Click **Data Lab**, then the **Token Inspector** tab.
2. Edit the text and click **Inspect tokens**. See vocabulary pieces, token IDs, count, and exact rendered input.
3. Enable the chat-template checkbox and inspect again. The extra count comes from system instructions and formatting markers.
4. Click **Playground** to run the source-linked ravioli summary. Select Local Hugging Face or Hosted Fireworks.

The inspector uses the real pinned tokenizer. It does not guess token boundaries or display word counts as token counts. Editing its input clears the old result. Its counts describe Qwen, not the different Fireworks model. The local model loads lazily on first inference; the tokenizer loads independently. First use requires network access for model artifacts. The model cache persists in a Docker volume.

## Local notebook and CLI

From the repository root, using Python 3.12 on macOS/Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-test.lock
.venv/bin/python -m pip install -r ml/requirements-hf.lock
.venv/bin/python -m pip install -e './ml[notebook]' -e ./backend
export HF_HOME="$PWD/.cache/huggingface"
```

On Linux CPU-only machines, install the CPU wheel **before** the HF lock file to avoid downloading CUDA packages:

```sh
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

The Dockerfile does this automatically. The macOS lock uses the native PyTorch wheel; local generation deliberately uses CPU float32 for this lesson. A recent macOS compatible with the pinned PyTorch wheel is required; use Docker if your host cannot install it. No GPU is needed. Model weights are roughly 1 GB; allow several GB more for libraries and runtime memory.

Open `ml/notebooks/01_tokenizer.ipynb` in a notebook-capable editor such as VS Code. Select `.venv/bin/python` as the kernel. The notebook begins with a kernel-specific local-package setup cell. Outputs are cleared in this revised copy; running the lesson performs real local inference. Jupyter UI itself is not bundled as an app dependency; the notebook extras install kernel/execution support.

Run the controlled title experiment:

```sh
.venv/bin/python -m recipetriage_ml.inference.experiment --provider hf
.venv/bin/python -m recipetriage_ml.inference.experiment --provider fireworks
.venv/bin/python -m recipetriage_ml.inference.experiment --provider both
```

Each command writes UUID-named JSON files under `ml/experiments/`. Repeated runs preserve earlier results. The runner loads `.env`. It exits nonzero if any title is blocked or produces invalid/incomplete output, after saving the evidence. It does not silently skip failures.

PowerShell equivalents for setup and experiment execution:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend/requirements-test.lock
.venv\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r ml/requirements-hf.lock
.venv\Scripts\python.exe -m pip install -e './ml[notebook]' -e ./backend
$env:HF_HOME = "$PWD/.cache/huggingface"
.venv\Scripts\python.exe -m recipetriage_ml.inference.experiment --provider hf
```

## Fixing `No module named recipetriage_ml`

`recipetriage_ml` is our local project package, not a package to download by name from PyPI. Having PyTorch installed does not install it. A notebook can also use a different Python environment from your terminal.

Run this in a notebook cell (it works from the repository root or `ml/notebooks`):

```python
from pathlib import Path
import sys
import subprocess

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "ml" / "pyproject.toml").is_file())
print("Installing into:", sys.executable)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(ROOT / "ml")])
```

Then **restart the notebook kernel and run all cells**. An editable installation can create startup `.pth` registrations that an already-running kernel has not loaded. The revised notebook also has a fallback setup cell that installs a regular local package copy if it is missing. Both approaches require the complete extracted project and Python 3.12+.

If pip reports that your Python is too old, select the project's Python 3.12 kernel. If root discovery fails, open the notebook from within the extracted `recipetriage` directory. Do not install an unrelated package named `ml`, and do not paste machine-specific `sys.path` workarounds into the lesson.

## Fireworks configuration

Edit the local `.env`, not frontend code:

```dotenv
FIREWORKS_API_KEY=your_actual_key
FIREWORKS_MODEL=accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b
```

Select a model available to your Fireworks account; the example model is documented by Fireworks, but live access is not verified without a key. Restart the backend to apply Compose environment changes:

```sh
docker compose up -d backend
```

For a host-run API, restart Uvicorn; application startup reads `.env`. The provider uses a bounded HTTP request and reports missing credentials, non-success HTTP responses, and malformed provider envelopes. The key is never sent to React. No automatic retries are performed, avoiding unexpected duplicate charges. Real Fireworks calls are billed by the provider.

## Endpoint contract

`POST /api/v1/triage` exists directly at port 8000 and through the frontend proxy at port 8080, with the **same path**. Example request:

```json
{
  "recipe": {
    "title": "Traditional Ravioli",
    "ingredients": ["flour", "eggs", "ricotta"],
    "instructions": ["Prepare dough, rest it, roll and fill the pasta, then boil."],
    "equipment": ["pasta machine", "pot"],
    "time_minutes": 70,
    "pantry_items": null
  },
  "provider": "hf",
  "temperature": 0,
  "max_new_tokens": 128
}
```

This short request is an API-shape illustration; the experiment uses the fuller source-linked summary in `ml/examples/ravioli.json`.

- A completed provider response returns HTTP 200, including `raw_output`, `prediction`, `valid_json`, `error`, model identity, local revision where available, finish reason, token counts, and latency.
- A schema-valid answer can still be factually wrong. The parser does not rewrite wrong labels; doing so would hide model behavior from shortcut testing.
- Invalid or truncated model output returns HTTP 200 with `valid_json=false`, `prediction=null`, and the raw response/error. This means the provider operation completed, not that classification succeeded.
- Invalid request fields return 422. Missing providers/keys, local load failures, and upstream failures return 503 with safe error details. Local exceptions are logged server-side.
- Local inference allows one generation at a time per worker, a 4096-token input-plus-output limit, at most 256 new tokens per API request, and a generation time budget. The time budget is checked between decoding steps and excludes initial model loading; it is not a hard process deadline. This learning deployment uses one API worker.
- `POST /api/v1/tokens` accepts `{"text":"Easy ravioli","chat_template":false}`.
- `GET /api/v1/learning-example` returns the source-linked recipe fixture.
- Existing `/health` and frontend `/api/health` still check API/database readiness, not model readiness.

## The title experiment and its limits

The base recipe is a short paraphrase of David Turner's **Homemade Cheese Ravioli** from King Arthur Baking. The source lists 70 minutes total. Our fixture follows the main food-processor/pasta-machine method; the source also describes manual alternatives. It includes an explicit ahead-of-time freezing option. Source URL and proposed expected labels are stored outside the model input.

Four variants change only the title: Easy Ravioli, Traditional Ravioli, Quick Ravioli, and Simple Ravioli. Ingredients, steps, equipment, pantry context and time remain identical. Each record stores the exact messages, common recipe-body hash, model/revision, prompt version, decoding settings, software versions, raw response, validation status, finish reason, and latency.

Observed local run: all four answers were schema-valid and all incorrectly included `weeknight-30min`, despite 70 minutes. Label flip rate versus Traditional was 0/3. This is **consistent error**, not evidence of correctness. One recipe cannot establish general title robustness. Proposed expected labels are an educational rubric, not human-verified training labels. The Fireworks result file records four blocked attempts due to a missing key; it contains no hosted predictions.

## Major new files

| File | Purpose |
| --- | --- |
| `ml/notebooks/01_tokenizer.ipynb` | Executable teaching sequence: tokens, chat template, tensors, actual logits, softmax, temperature, and generation. |
| `ml/inference/contracts.py` | Pydantic recipe/prediction contracts and provider-result types. |
| `ml/inference/prompts.py` | One versioned system prompt and shared structured user input. |
| `ml/inference/parsing.py` | Full-response JSON parser; rejects unknown/duplicate labels, duplicate keys, missing fields, prose, fences and malformed JSON. |
| `ml/inference/hf_provider.py` | Pinned tokenizer/model, chat-template application, lazy caches, CPU inference, concurrency/input limits, and token inspector. |
| `ml/inference/fireworks_provider.py` | Hosted HTTP adapter, key handling, timeout, output-budget mapping, and safe failures. |
| `ml/inference/service.py` | Shared orchestration, timing, parsing, and preserving raw model behavior. |
| `ml/inference/experiment.py` | Controlled title-only variants, saved per-run evidence, and transparent flip-rate denominator. |
| `ml/examples/ravioli.json` | Source-linked recipe summary and separate proposed labels/provenance. |
| `ml/experiments/*.json` | Saved local experiment and blocked hosted comparison. |
| `ml/pyproject.toml` | Installs these files as `recipetriage_ml`; optional HF/notebook extras. |
| `ml/requirements-hf.lock` | Exact inference-library versions, separate from the backend's minimal dependencies. |
| `backend/app/inference.py` | Validated triage/token endpoints and safe HTTP error mapping. |
| `frontend/src/TokenInspector.jsx` | Text editor, template option, token table, count, and rendered prompt. |
| `frontend/src/InferencePlayground.jsx` | Recipe editor, provider/generation settings, raw response and validity display. |
| `tests/ml/test_inference.py` | Prompt/JSON/title-control/provider contract tests without real network or model downloads. |
| `tests/backend/test_inference_api.py` | API validation and error behavior. |
| `frontend/src/TokenInspector.test.jsx` | Token/ID rendering and visible error-state tests. |

Docker now installs CPU inference libraries and keeps model downloads in `model_cache`. Nginx and Vite preserve the `/api/v1/` prefix while retaining the old health proxy. GitHub Actions discovers the new tests through the updated root pytest configuration; it does not make paid Fireworks calls or download model weights in unit tests.

## References

- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Hugging Face generation parameters](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Fireworks chat completions](https://docs.fireworks.ai/api-reference/post-chatcompletions)
- [Fireworks Llama 3.3 model](https://fireworks.ai/models/fireworks/llama-v3p3-70b-instruct)
- [Source ravioli recipe](https://www.kingarthurbaking.com/recipes/homemade-cheese-ravioli-recipe)

Before moving on, explain inference versus training in your own words.
