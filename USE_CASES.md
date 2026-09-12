# RecipeTriage use cases

All 15 use cases have code paths. Availability of a trained model is separate from implementation: reviewed-data retraining and real DPO still require your explicit record-level decisions. GRPO and the measured deployment candidates have not beaten the baseline, and no model is in production.

## Acceptance map

| ID | Use case and page | What to verify | Current evidence or constraint |
| --- | --- | --- | --- |
| UC-01 | Add recipe — Recipes → Add recipe | Choose text, message, public URL or screenshot; normalize; inspect/edit Recipe JSON; save | Fireworks message extraction and live schema.org URL extraction verified. Real Tesseract OCR is tested in Docker, and the complete screenshot/OCR/Fireworks intake endpoint was exercised. Unknown fields remain null/empty; extraction can still make mistakes. |
| UC-02 | Auto-triage — Recipes | Leave automatic triage selected when saving, or choose Run triage on a saved recipe | Raw output, parsed labels, model identity and errors persist. Invalid/truncated responses stay visible. Predictions do not approve labels. |
| UC-03 | Human correction — Recipes → Review or correct labels | Select labels, enter reviewer and rationale, then explicitly save | Creates a revision-bound verified TrainingExample. Editing recipe evidence clears its current review/prediction while retaining history. |
| UC-04 | Recipe library — Recipes | Search by title, ingredient or equipment; filter by label; inspect source and preparation | Seven original recipes are saved as awaiting review. Library entries and saved dataset snapshots are separate collections. |
| UC-05 | Dataset curation — Data Lab and Recipes | Review records, validate labels, build/save a version, inspect splits and JSONL | Recipes exports only current explicit reviews. Synthetic augmentation also freezes the existing holdout sets. Both reject benchmark leakage. |
| UC-06 | Synthetic hard examples — Data Lab → Synthetic Review | Generate proposals, inspect source/model/prompt, edit and Approve/Reject | Six real Fireworks proposals plus seven originals are pending. Nothing is automatically appended to train.jsonl. |
| UC-07 | Baseline — Evaluation Lab | Compare fixed benchmark results and inspect per-label metrics, JSON/schema validity, exact match and latency | Local/Fireworks results and provider failures are saved. Ten examples are a provisional teaching benchmark. |
| UC-08 | SFT/LoRA/QLoRA — Training Lab | Inspect architecture, parameters, configuration, training history and automatic benchmark | Local measured SFT, rank and QLoRA experiments exist. Managed Fireworks SFT has a preflight path; account/model prerequisites can block it. |
| UC-09 | Experiment tracking — Training Lab → Experiments | Compare fixed-data learning-rate/rank runs; use CLI JSON configs | PostgreSQL registry, local evidence and optional MLflow integration. Dataset and benchmark identities are recorded. |
| UC-10 | Preference labeling — Alignment Lab → Preference Pairs | Read both candidates and explicitly choose A/B/Tie/Neither | Five real pairs are pending. Ties/neither do not provide chosen/rejected DPO examples. |
| UC-11 | DPO/GRPO — Alignment Lab | Inspect prerequisites, reward components and after-training benchmark | DPO implementation tested with a tiny random model; real task DPO awaits human A/B choices. Real GRPO probe produced zero reward and no improvement. |
| UC-12 | Failure analysis — Evaluation Lab → Failure Analysis | Inspect systematic-error categories and prioritized collection recommendations | Rule-based failure groups support hypotheses, not causal claims. Failures persist with run identity. |
| UC-13 | Shortcut testing — Evaluation Lab → Shortcut Tests | Compare title-only variants; inspect eligible denominators, flips and misleading-title accuracy | Body fields are asserted unchanged. Ravioli overlaps the original seed, so current scores are diagnostic rather than an untouched generalization test. |
| UC-14 | Model playground — Playground → Compare models | Select 2–6 models and compare the same recipe/settings; reopen Saved comparison | Fireworks, base, instruct, SFT, LoRA and QLoRA catalog. Real DPO/production appear only when artifacts/gates permit. Independent errors persist alongside successful responses. |
| UC-15 | Deployment — Deployment | Compare merged/quantized exports; inspect registry gates; promote only a passing candidate | Merge/NF4 scripts, operational selection, serving pointer and rollback exist. All current candidates fail strict production gates; production requests return 503. |

## Try the product workflow

From the repository root, with your existing `.env`:

```sh
docker compose up --build --wait --wait-timeout 180
```

Open http://localhost:8080. **Recipes** now opens the saved library. Search for `pasta machine` to find ravioli, then choose **Open in Playground**. The prefilled inference object contains recipe evidence only; labels and source annotations are excluded.

For intake, choose **Add recipe**, select an input type and click **Normalize recipe**. Review the extracted JSON before **Save recipe**. With automatic triage enabled the recipe is saved first; the selected model's response appears asynchronously. A busy/unavailable provider does not discard the saved recipe. Exact duplicate content returns the existing record.

Use **Review or correct labels** only after checking the recipe yourself. Enter your reviewer name and reasoning. With at least three independent current reviews, **Create dataset from verified recipes** creates an immutable version and exposes its JSONL ZIP. The server checks label constraints, duplicate groups, nonempty splits and benchmark overlap. It does not launch training.

This library export creates a new independently split collection. To improve the existing experiment dataset while preserving its validation/test membership, use **Data Lab → Synthetic Review → Build next version** instead. Reviewing a library entry does not silently approve a separate synthetic-review queue record.

## Intake providers and limits

Set `FIREWORKS_API_KEY` in `.env` for free-text/message extraction and hosted triage. `FIREWORKS_RECIPE_MODEL` selects the model for intake and the comparison catalog, defaulting to `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`. It is independent of the earlier single-model `FIREWORKS_MODEL` setting. The UI displays the actual selected model; there is no silent provider substitution.

```sh
docker compose up -d backend
```

Keys stay on the backend. Fireworks receives pasted text or extracted OCR text only when normalization needs the model, and receives a recipe when explicitly selected for generation. Complete Recipe JSON validates without a model call. Complete schema.org Recipe JSON-LD pages can be normalized locally. The Fireworks schema envelope is validated again by Pydantic; a valid shape is not evidence that extraction is correct. See the provider's [structured-output documentation](https://docs.fireworks.ai/structured-responses/structured-response-formatting).

Public URL imports accept HTTP(S), ports 80/443, at most four requests including redirects, 2 MB responses and 12-second socket timeouts. All resolved IPs must be public; each redirect is validated and the connection pins a validated IP. Login-only, blocked, compressed or unsupported pages return an explicit paste-text alternative. The loader does not run JavaScript or use your browser login. Multiple recipes in one structured page require selecting/pasting one recipe yourself.

Screenshots accept PNG/JPEG/WebP up to 4 MB and 12 megapixels. OCR is local Tesseract with a 25-second timeout and the default English language pack. Images are temporary; source text, image hash and normalization provenance are retained. OCR is not general visual reasoning and can misread small/handwritten text. Source text has a 24,000-character normalization limit.

Docker includes Pillow and Tesseract. For host development:

```sh
.venv/bin/python -m pip install -r backend/requirements-intake.lock
.venv/bin/python -m app.migrate
.venv/bin/python -m app.library --seed
```

On macOS with Homebrew, install OCR with `brew install tesseract`; Debian/Ubuntu use `sudo apt-get install tesseract-ocr`. On Windows use the Docker backend, or install Tesseract separately and make `tesseract` available on PATH; Python commands use `.venv\Scripts\python.exe`. Host OCR tests are skipped when the executable is absent, with an explicit reason. Docker runs the real OCR test.

## Install delivered adapters for the comparison catalog

The full-source archive contains the small reference adapters. They are excluded from the Docker build context and live in the persistent training volume at runtime. From the extracted repository root with the backend running, these commands work in macOS/Linux shells and PowerShell:

```sh
docker compose exec backend mkdir -p /training/references/sft-v1 /training/references/lora-v1 /training/references/qlora-v1
docker compose cp ml/training/results/sft-v1/local/. backend:/training/references/sft-v1
docker compose cp ml/training/results/experiments-v1/runs/771d20b1-77df-446b-b34d-b8259ddbc585/. backend:/training/references/lora-v1
docker compose cp ml/training/results/experiments-v1/runs/041818c9-7e00-4802-b53d-f7d567616e2b/. backend:/training/references/qlora-v1
docker compose exec --user root backend chown -R appuser:appuser /training/references/sft-v1 /training/references/lora-v1 /training/references/qlora-v1
```

The `chown` runs inside Linux containers even on Windows/macOS; it makes copied checkpoints readable by the existing non-root application user. Refresh the page after copying. Host mode resolves the delivered source directories directly. Adapter loading verifies weight checksums and the pinned base revision. A completed DPO checkpoint must be registered before it appears as an available comparison candidate.

**SFT** identifies the earlier instructional SFT run, which itself used LoRA. The separate **LoRA** and **QLoRA** options select later measured experiment checkpoints; these names identify different runs, not mutually exclusive optimization concepts.

Comparison preparation and generation timing are reported separately. Local models load sequentially to limit memory use; first preparation can include weight downloads. Hosted generation includes network time. Different generated lengths and provider hardware still affect latency. Use the fixed benchmark and deployment measurements for quality/operational decisions; one side-by-side recipe is not a benchmark or improvement claim.

The saved live comparison and intake evidence are under `ml/inference/results/use-cases-v1/`. These drafts/results are diagnostic evidence, not approved training examples. See VERIFICATION.md for failures and the final test counts.

## Major files

| File | Responsibility |
| --- | --- |
| `frontend/src/Recipes.jsx` | Saved library and dataset browsing, search/filter, recipe details and verified export. |
| `frontend/src/RecipeIntake.jsx` | Text/URL/message/screenshot input, extraction preview, explicit save and optional auto-triage. |
| `frontend/src/RecipeActions.jsx` | Named human corrections, revision edits, triage and immutable history display. |
| `frontend/src/Playground.jsx`, `ModelComparison.jsx` | Single-model and comparison tabs, explicit model choices, polling and saved result cards. |
| `backend/app/library.py` | Intake drafts, deduplicated saved recipes, revision-bound prediction/review persistence and reviewed export. |
| `backend/app/intake.py` | Public URL retrieval, schema.org extraction and bounded local OCR. |
| `ml/inference/normalization.py` | Provider-independent Recipe validation/extraction contract and source/prompt/model provenance. |
| `backend/app/playground.py` | Server-resolved model catalog, verified adapter loading, sequential inference and comparison jobs. |
| `backend/app/migrate.py` | Idempotent creation of intake/library/prediction/review tables alongside existing lifecycle tables. |
| `tests/backend/test_recipe_intake.py`, `test_recipe_library.py` | URL/OCR/schema tests and isolated PostgreSQL product workflows; fixture reviews never touch user records. |
| `frontend/src/RecipeProduct.test.jsx` | Library search, clean Playground handoff, intake/save, explicit review and partial comparison failures. |

`recipe_intakes`, `library_recipes`, `library_predictions` and `library_reviews` store the new product records. Existing `learning_jobs` stores comparison and triage jobs. Existing `dataset_versions` stores reviewed exports. Original dataset files, synthetic decisions, preference choices and the production pointer retain their existing boundaries.

This remains a localhost learning deployment. The code paths do not establish production model quality or public-service readiness. See [VERIFICATION.md](VERIFICATION.md) for measured outcomes and [MODEL_REGISTRY.md](MODEL_REGISTRY.md) for quality gates and rollback.
