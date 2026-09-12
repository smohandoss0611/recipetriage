# RecipeTriage-Bench-v1: baseline lesson and run guide

This phase evaluates unchanged models. It adds a fixed, source-linked benchmark, metrics, saved evidence, and **Evaluation Lab → Baseline**. It does not train or fine-tune a model. The benchmark contains ten individually assembled recipe summaries with provisional labels; a human has not yet reviewed the answer key.

## Learn the metrics with a recipe

Suppose a recipe should receive `{dessert, needs-special-equipment}`, but the model returns `{dessert, meal-prep}`. The correct dessert prediction is one true positive (TP). The extra meal-prep label is one false positive (FP). The missing equipment label is one false negative (FN).

| Metric | Calculation | Meaning in this example |
|---|---|---|
| Precision | TP / (TP + FP) | 1/2: half the labels we predicted were correct. |
| Recall | TP / (TP + FN) | 1/2: we found half the required labels. |
| F1 | 2TP / (2TP + FP + FN) | 1/2: balances precision and recall. |
| Exact label-set match | Correct complete sets / recipes | 0/1: the full set is wrong. Order does not matter. |

**Multi-label evaluation** allows several labels per recipe. Each recipe-label pair is a separate yes/no decision. **Micro averaging** pools TP, FP and FN across labels before calculating scores. **Macro averaging** calculates each label's score first, then gives every label equal weight. Macro F1 can expose weak performance on rare labels that pooled scores obscure. [Metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html).

RecipeTriage fixes the vocabulary at seven labels and uses zero for undefined precision/recall/F1 denominators. Macro F1 averages all seven, including labels with no support inside a category. A category containing only one supported label can therefore have perfect exact match but macro F1 below one. The per-label support column makes that policy visible. Overall micro F1 and exact match are easier starting points for comparing these tiny categories.

**JSON validity** measures whether the entire final text parses as JSON. `{"labels":["instant"]}` is valid JSON but fails our application schema: the label is unknown and the explanation is missing. Schema validity additionally requires the exact Prediction contract, known unique labels, and the multi-label rules. Markdown fences, extracted substrings, repaired JSON, or partial answers are not silently accepted.

**Latency** is elapsed wall-clock time for a provider call, including output transfer and provider-side work. The runner records each attempt, then mean, median (p50), and nearest-rank p95. Local tokenizer/model loading is measured separately as setup time. With ten observations, nearest-rank p95 is the slowest observation; it is not a stable estimate of production tail latency. CPU execution and hosted hardware/network conditions differ.

## What other benchmarks measure

| Term | What it measures | Why it is not our primary measure |
|---|---|---|
| Perplexity | Exponentiated average negative log probability of reference tokens; lower means the model predicts that text more confidently. | Requires token probabilities and depends on tokenization and context handling. It does not directly establish correct recipe labels or usable JSON. [Transformers explanation](https://huggingface.co/docs/transformers/perplexity). |
| ROUGE | Overlap between generated text and reference summaries, using variants based on n-grams or subsequences. | Different explanations can be equally valid; matching wording does not establish matching labels. [Original paper](https://aclanthology.org/W04-1013/). |
| GLUE | A suite of language-understanding tasks such as entailment and sentiment. | Broad language understanding is useful background, but the tasks and labels differ from recipe triage. [GLUE paper](https://arxiv.org/abs/1804.07461). |
| GSM8K | Grade-school mathematical word problems. | Arithmetic skill does not directly measure recipe classification or the JSON contract. [Dataset paper](https://arxiv.org/abs/2110.14168). |
| MMLU | Multiple-choice knowledge and reasoning questions across many subjects. | Broad subject knowledge is different from following our seven-label rubric. [MMLU paper](https://arxiv.org/abs/2009.03300). |

Those measures are not implemented here. Our primary evidence is per-label precision/recall/F1, micro and macro F1, exact match, valid usable responses, and latency. Explanation quality still requires review; this evaluator does not score explanation truthfulness.

## Why freeze the benchmark

A **baseline** is a recorded reference result before an intervention. A **base model** describes a model's training stage. They are different concepts: both a pretrained model and a hosted instruction-following model can supply baseline results.

The fixed benchmark keeps the recipes, answer key, label policy, prompt, and scoring policy stable. If both model and test change, a score difference cannot be attributed to the model alone. Each run stores hashes of those inputs plus decoding settings, runtime information, exact logical messages, returned text, and errors. The base model's plain-text rendered prompt is also saved.

**Contamination** occurs when evaluation examples or answers influence model development, including training on the benchmark or repeatedly choosing prompts based on its answers. We audit exact normalized recipe bodies and source URLs against the seven bundled Dataset v1 examples; no overlap was found. This is not a semantic paraphrase detector and does not audit user-created PostgreSQL dataset versions. These are public recipes, so overlap with either provider's pretraining data is unknown.

Keep `ml/evaluation/` out of future training inputs. Once these examples guide development, retain a separate untouched test collection for a credible final estimate. A fixed checksum prevents accidental edits, not deliberate tampering or contamination. Correcting an annotation requires a new benchmark version and an explanation; do not change a target to match the model output.

## The fixed cases

Each row stores a Recipe, proposed labels, rationale, review status, category, challenge, source URL, and notes about interpretation. Only the Recipe's inference fields enter model messages. Gold labels, category, challenge, provenance, and rationale remain outside the request.

| ID | Category | Recipe/source | Proposed labels |
|---|---|---|---|
| bench-001 | normal | [Stovetop Popcorn](https://foodhero.org/recipes/stovetop-popcorn) | weeknight-30min |
| bench-002 | normal | [Hazelnut Thumbprint Cookies](https://foodhero.org/recipes/hazelnut-thumbprint-cookies) | dessert |
| bench-003 | misleading-title | [Quick Brown Bread](https://foodhero.org/recipes/quick-brown-bread) | weekend-project |
| bench-004 | special-equipment | [Peach Yogurt Smoothie](https://foodhero.org/recipes/peach-yogurt-smoothie) | weeknight-30min, needs-special-equipment |
| bench-005 | special-equipment | [Banana Ice Cream](https://www.nutrition.gov/recipes/banana-ice-cream) | dessert, needs-special-equipment |
| bench-006 | multi-label | [Salsa Chicken](https://foodhero.org/recipes/salsa-chicken) | needs-special-equipment, meal-prep, have-most-of-this |
| bench-007 | multi-label | [Fruit Yogurt Popsicles](https://foodhero.org/recipes/fruit-yogurt-popsicles) | dessert, needs-special-equipment, meal-prep |
| bench-008 | ambiguous | [Herbed Yogurt Sauce](https://foodhero.org/recipes/herbed-yogurt-sauce) | unclear |
| bench-009 | hard | [Hummus without Tahini](https://foodhero.org/recipes/hummus-no-tahini) | weeknight-30min, have-most-of-this |
| bench-010 | hard | [Turnip Pancakes](https://foodhero.org/recipes/turnip-pancakes) | weekend-project |

The summaries are individually assembled from published recipes, not an automatically generated synthetic corpus. Pantry lists in two cases are explicit teaching scenario assumptions, not facts about your pantry. Missing time is null, not zero. Passive freezing and slow cooking do not alone establish a weekend project. Quick Brown Bread's project label is a provisional rubric judgment about the complete mixing/baking/cooling task; review it before treating this as a trusted answer key. No source directions or labels were changed after observing model predictions.

Category counts are 2 normal, 1 misleading-title, 2 special-equipment, 2 multi-label, 1 ambiguous, and 2 hard. Categories are mutually exclusive reporting groups even when a case tests several behaviors. Label supports are 3 weeknight, 2 weekend, 4 equipment, 2 meal-prep, 3 dessert, 2 pantry, and 1 unclear.

Frozen canonical-content SHA-256:

```text
173e79cc1968dd1207a7aafd051dad40d87ff551c7081a17b896c9556a698448
```

This is the hash of validated canonical JSON, not the raw file bytes. `load_benchmark()` verifies it at runtime. The case browser is read-only; Dataset Studio edits the separate training dataset.

## Scoring failures honestly

- JSON validity and schema validity divide by **returned responses**, including empty returned text. With no responses, both rates are null, displayed as a dash.
- A usable response must parse, satisfy the Prediction schema, and finish with `stop`. A complete-looking object returned with `length` is still unusable.
- Unusable answers count as an empty predicted set for end-to-end label scoring. Every required label becomes a false negative, and exact match fails. We do not credit a label extracted from malformed text.
- Usable response rate divides by all ten benchmark cases. Always read this coverage alongside precision.
- A blocked or partially attempted run receives no aggregate label score. Missing credentials or HTTP 404 do not become a misleading accuracy of zero.
- If all cases were attempted but none produced usable answers, end-to-end F1 and exact match are zero. That diagnoses this complete system/configuration, not the model's capability under every possible configuration.
- HTTP 401/403/404/429 stops further calls in that run. The attempted error and unattempted cases remain visible. There are no hidden retries or fallback model changes inside a run.

## Actual baseline results

Runs below were executed on September 9, 2026. Both final runs use temperature 0, output budget 128, one pass in file order, and the same benchmark/prompt/scoring hashes. Hosted reasoning was explicitly disabled. Local execution used CPU float32 with four threads.

| Provider/model | Micro F1 | Macro F1 | Exact match | JSON / schema valid | Usable | Mean / p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Local Qwen2.5-0.5B base | 48.5% | 36.5% | 0/10 | 50% / 40% | 4/10 | 4.850 / 6.167 s |
| Fireworks Qwen3.7 Plus | 94.1% | 95.1% | 9/10 | 100% / 100% | 10/10 | 0.605 / 0.803 s |

The local model is **Qwen/Qwen2.5-0.5B**, pinned to revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. It is pretrained, distinct from the Instruct model in Playground. Because the base model has no instruction chat contract, the runner uses a recorded plain-role completion format without few-shot examples. [Base model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).

The hosted model is `accounts/fireworks/models/qwen3p7-plus`. The account's model API confirmed availability. A pinned hosted revision was not exposed, so revision is null. Provider-native chat formatting, model sizes/training, and hardware differ: this is a comparison of two systems on the same task, not a controlled test isolating one training method. Temperature zero is not a guarantee of identical output across hosted revisions or hardware.

The successful hosted run missed one recipe: Salsa Chicken. It predicted weekend-project instead of needs-special-equipment, retaining the correct meal-prep and pantry labels. This gives 16 TP, 1 FP and 1 FN across the benchmark, so micro F1 is 32/34 = 94.1%. It is useful error-analysis evidence, not justification to rewrite the target.

Local setup took 5.417 s in the final run; hosted setup took 0.019 ms because it makes no warmup request. Per-request latency excludes that separate setup measurement. Setup with uncached local weights would include download time. One ten-case run does not support a general speed or production-quality claim.

All five evidence files are retained under `ml/evaluation/results/`:

| Run ID | What happened |
|---|---|
| `6bd9eeec-3fa8-4089-9b91-1b9d3b4cfc70` | Final local base comparison, reasoning-disabled protocol. |
| `bfec561c-3348-4422-9f75-d3aa57b93e5b` | Final Fireworks comparison, reasoning explicitly disabled. |
| `81b8ba36-9063-4df2-a2a9-8ba526551a9c` | Earlier local base run with provider-default protocol; same quality scores, different timing. |
| `051e640d-412d-4c55-a210-e116b692f43d` | Configured Llama 3.3 model returned HTTP 404; stopped after one attempt, no aggregate label score. |
| `6d6d783a-65e6-488d-8c72-586d7dbd4997` | Qwen3.7 Plus with default reasoning returned ten empty final answers, each ending at the 128-token limit; zero usable answers. |

The last diagnostic is consistent with reasoning consuming the small output budget before final text. We retained finish reasons and token counts, but did not store a separate reasoning body. Fireworks documents separate reasoning/final fields and a `reasoning_effort` control; the rerun sends `none`. This mode change is recorded in the protocol, and both providers were rerun. [Reasoning guide](https://docs.fireworks.ai/guides/reasoning), [chat API parameters](https://docs.fireworks.ai/api-reference/post-chatcompletions).

The final pair shares protocol hash `3502623cbf2d294c3db4989da3950c3b65184caec96291e03fbe39e64d6ba2bf`. Earlier default-mode runs use `8219bcce46149c66e50c80fe7c7bfdf17fc5a12b90a189ffdad1afcc4530f38b`. Compare matching protocol IDs. The UI warns when multiple protocols are present.

## Run and inspect

From the repository root, preserve an existing `.env`. On first setup only, copy `.env.example` to `.env` (`cp` on macOS/Linux; `Copy-Item` in PowerShell). Set your Fireworks key privately in that file and choose a model available to your account. The default now matches the model used in the successful run. Model availability can change. [Availability guide](https://docs.fireworks.ai/faq-new/models-inference/how-to-check-if-a-model-is-available-on-serverless).

```sh
docker compose up --build --wait --wait-timeout 180
docker compose exec backend sh -c 'python -m app.benchmarks /app/ml/evaluation/results/*.json'
```

The second command imports the bundled historical evidence into a fresh PostgreSQL database. It makes no model calls. Importing identical evidence twice is safe; different content with the same run ID is rejected. The container's `sh` expands the wildcard, so the command works from macOS, Linux, or PowerShell.

Open http://localhost:8080/ and click **Evaluation Lab**:

1. **Compare runs:** inspect model, protocol, validity, scores, coverage and timing. Select a provider and click **Run benchmark** only when you intend a new ten-case run.
2. **Benchmark cases:** read the fixed recipe input, proposed target labels, challenge and source notes.
3. **Run details:** inspect per-label and category metrics, setup time, raw responses, errors, exact prompts and downloadable JSON.
4. Reload the page: saved runs remain in PostgreSQL. No hosted calls are triggered by opening the page or downloading evidence.

For a fresh CLI comparison inside Docker:

```sh
docker compose exec backend python -m recipetriage_ml.evaluation --provider both --reasoning disabled --temperature 0 --max-new-tokens 128 --output /tmp/benchmark-results
docker compose cp backend:/tmp/benchmark-results ./benchmark-results
docker compose exec backend sh -c 'python -m app.benchmarks /tmp/benchmark-results/*.json'
```

The CLI exits 1 if a run is blocked or has any unusable responses, **after saving evidence**. A nonzero exit is expected for the observed local base run; inspect the results instead of hiding the error. Low F1 alone does not set a nonzero exit. The CLI checkpoints after every case. An interrupted run remains partial; start a new run rather than appending to old evidence. Copy `/tmp` results before recreating the container.

For host execution, first install the Python 3.12 ML/HF dependencies described in LEARNING.md, including editable `./ml` and `./backend` packages. Then on macOS/Linux:

```sh
export HF_HOME="$PWD/.cache/huggingface"
export HF_CPU_THREADS=4
.venv/bin/python -m recipetriage_ml.evaluation --provider both --reasoning disabled --temperature 0 --max-new-tokens 128 --env-file .env --output benchmark-results
.venv/bin/python -m app.migrate
.venv/bin/python -m app.benchmarks benchmark-results/*.json
```

PowerShell equivalents after the same dependency setup:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_CPU_THREADS = '4'
.venv\Scripts\python.exe -m recipetriage_ml.evaluation --provider both --reasoning disabled --temperature 0 --max-new-tokens 128 --env-file .env --output benchmark-results
.venv\Scripts\python.exe -m app.migrate
Get-ChildItem benchmark-results\*.json | ForEach-Object { .venv\Scripts\python.exe -m app.benchmarks $_.FullName }
```

`--env-file` selects a private credential file for the benchmark runner. Existing process environment variables take precedence. No API keys are written to evidence or delivered in the source archive. The benchmark API requires PostgreSQL; the file-based CLI does not. Local first use downloads a separate base-model checkpoint of roughly 1 GB, even if Playground's instruct checkpoint is already cached.

The UI's reasoning control applies to benchmark runs. Playground retains its existing request settings; a reasoning model may exhaust its small budget there. A provider that does not support `reasoning_effort=none` should use provider-default and record that different protocol.

## Major files and responsibilities

| File | Responsibility |
|---|---|
| `ml/evaluation/bench_v1.json`, `bench_v1.sha256` | Fixed source-linked cases and canonical-content checksum. |
| `ml/evaluation/metrics.py` | Pure label metrics, exact match, output validity, coverage and latency aggregation. No provider calls. |
| `ml/evaluation/benchmark.py` | Pydantic benchmark/config validation, contamination audit, prompt/protocol hashes, execution, per-case evidence and atomic file checkpoints. |
| `ml/evaluation/base_provider.py` | Lazy loading of pinned pretrained Qwen, plain-role prompt rendering, CPU inference and separate setup measurement. |
| `ml/evaluation/__main__.py` | CLI environment loading, provider selection, progress, files and meaningful exit status. |
| `ml/evaluation/results/*.json` | Five real immutable run records, including diagnostic failures. |
| `ml/inference/fireworks_provider.py` | Hosted adapter; optional reasoning control, bounded timeout, safe errors and raw final text. |
| `ml/inference/prompts.py`, `parsing.py` | Existing shared inference instruction and strict Prediction parser used by both systems. |
| `ml/pyproject.toml` | Packages the evaluation module and benchmark assets in the installable ML distribution. |
| `backend/app/benchmarks.py` | Start/list/detail/download API, PostgreSQL checkpoints, import validation and immutable completed evidence. |
| `backend/app/migrate.py` | Idempotently adds the `benchmark_runs` JSONB table alongside dataset storage. |
| `backend/app/recover_benchmarks.py` | Marks old active jobs interrupted when the single API process restarts. |
| `backend/app/main.py`, `backend/Dockerfile` | Mount the router; run migration/recovery before serving the Docker API. |
| `frontend/src/Baseline.jsx`, `App.jsx` | Comparison, fixed cases, detailed evidence, progress polling and Evaluation Lab navigation. |
| `tests/ml/test_evaluation.py` | Hand-calculated metrics, malformed answers, truncation, coverage, frozen data, protocol settings and mocked provider failures. |
| `tests/backend/test_benchmarks.py` | Secret-free public configuration, import tamper checks, real PostgreSQL persistence, successful background execution, immutable history and conflicts. |
| `frontend/src/Baseline.test.jsx` | Visible scores/errors, details/downloads, protocol warnings, frozen cases and submitted run settings. |
| `.env.example`, `docker-compose.yml`, `.github/workflows/ci.yml` | Private local configuration template, service wiring, and automated tests/API smoke checks. |

Product UI stays in `frontend/`; HTTP serving and persistence stay in `backend/`; provider-independent evaluation lives in `ml/`. The CLI and web worker call the same runner, so their scoring rules cannot drift through separate implementations.

The web app uses FastAPI background tasks with one Uvicorn worker. PostgreSQL serializes enqueue requests and stores progress. It is not a durable distributed task queue: a restart interrupts work, which is explicitly marked on startup. Do not run the recovery command while another API process is actively benchmarking. For host development, run `python -m app.recover_benchmarks` once after stopping the old server and before starting its replacement. Scaling to multiple workers needs a separate job queue and worker ownership scheme.

## Check your understanding

1. A recipe's true labels are `{dessert, meal-prep}`, but the model returns `{dessert, needs-special-equipment}`. What are TP, FP, FN, precision, recall, F1 and exact match?
2. Why could micro F1 look good while the rare `unclear` label performs badly? Which table would you inspect?
3. How can an answer be valid JSON but unusable by RecipeTriage? Why do we also record the finish reason?
4. Why are the HTTP 404 run and the ten truncated-response run reported differently? What do their scores tell you about the model itself?
5. Before changing the prompt or training anything, which benchmark/protocol artifacts must stay fixed, and how would you keep a future final test set uncontaminated?
