# Single-server deployment — 2026-09-12

## 2026-09-12 — Streamlit user guide

Added `USER_GUIDE.md`, a 16-page printable PDF under `docs/user-guide/`, and a companion starter JSON file. The guide follows the current Streamlit labels and controls across all seven workspaces, including temporary drafts versus saved versions, explicit human review/preferences, shared-worker conflicts, training progress and gated promotion. README links to all three files. It describes current behavior rather than claiming new training or model improvement.

Verified all 24 guide Markdown references resolve locally. The seven sample records are byte-for-byte copies of the existing seed file; schema validation and an in-memory dataset build passed with 5/1/1 splits and `teaching-draft` status. No approval flags or source provenance were changed. Rendered and visually reviewed all 16 PDF pages, checked text bounds and verified 14 outline entries and 14 distinct contents destinations. The contents links have separate clickable titles and page numbers; an initial assertion expecting one annotation per entry was corrected to count destinations. No application services, database records, jobs or credentials were changed for this documentation task.

## 2026-09-12 — Review errors and active training progress

Synthetic Review now displays individual quality errors and category guidance, offers only Edit/Reject when automated checks fail, tolerates failed generations without a draft, and reloads the saved revision after a human decision. An edit remains pending and never implies approval. Saved training runs fetch their full detail instead of displaying list summaries, exposing the phase, benchmark case counts and loss history. Rejected SFT and generic job submissions retain the saved-results section. Standalone benchmark manifests are not mistaken for after-training progress.

**61 Streamlit tests and 3 synthetic-quality tests passed.** New UI coverage checks blocked approvals, explicit edit followed by a separate approval on the new revision, missing drafts, full progress reads, progress retained after HTTP 409 and benchmark/result distinction. These tests use fixtures and do not modify human review records. A read-only AppTest inside the running UI made eight GET requests against the actual backend, confirmed visible title-quality errors and full training progress, and prohibited all mutations. Streamlit emitted its expected bare-mode ScriptRunContext warning during that check; it did not fail.

Applied the UI source without restarting either service; source hashes, container/process identities and private configuration preservation were verified. Rebuilt only the Streamlit image for future starts. No backend changes, forced unlocks, live reviews or job submissions were performed. The existing LoRA run `9ca0eb96-d05f-4004-8c66-563f7cf4f241` completed with both 10-case benchmarks. A subsequent run `fc585549-856e-405a-bc11-414e47e6c320` was active during verification. HTTP 409 was the expected shared-worker guard, not evidence of a failed or stuck job. This UI change makes no claim about model improvement. The full backend/ML suite was not rerun.

## 2026-09-12 — Local SFT configuration preflight

The Streamlit default Local provider exposed Check configuration, but the backend rejected every non-Fireworks preflight with HTTP 422. `/api/v1/training/preflight` now dispatches by provider. Local checks validate the request schema, immutable dataset snapshot, known benchmark overlap and installed package metadata; they report resolved settings, split counts, draft warnings and explicitly deferred runtime checks. No model weights, tokenizer downloads, training jobs or Fireworks requests are needed. Managed checks retain their provider-specific constraints. The Streamlit result shows pass/fail and issues directly instead of treating every completed HTTP request as a successful configuration.

**55 Streamlit tests and 12 targeted training tests passed** (two unrelated PostgreSQL integration tests were deselected). Coverage includes Local with one example per split and no Fireworks client, missing dependencies, altered datasets, unknown versions, invalid parameters, QLoRA package checks, provider dispatch and both UI provider choices without submitting training. The targeted host tests emitted existing Starlette/httpx and AnyIO deprecation warnings; no tests failed.

Both Docker images built. After confirming no active training, benchmark, generation, LoRA or experiment jobs, only the backend was recreated. Streamlit source was applied without restarting its container; its process/session and private configuration hashes were preserved. A live request against the latest saved teaching dataset returned HTTP 200 with `provider=local`, `supported=true` and 5/1/1 counts. Saved dataset versions and training-run records were identical before and after the request. The check does not prove runtime imports, token lengths, model loading, LoRA targets, NF4 capability or sufficient memory; these remain explicit training-time checks. No training or managed-provider operation ran. The full backend/ML suite was not rerun for this focused change.


## 2026-09-12 — Direct access, tabs and readable chat preview

- Removed the RecipeTriage workspace-password screen and legacy login module at the user’s request. Existing private configuration files were preserved; new setup no longer prompts for a workspace password. Anyone who can reach the interface can use its shared features. Local-check mode stays bound to `127.0.0.1`, and server-side API credentials remain mandatory.
- Added stateful workspace/page tabs across all seven workspaces and 23 pages. Only the active page executes, following [Streamlit’s tab API](https://docs.streamlit.io/develop/api-reference/layout/st.tabs). Replaced checkbox controls with named radio choices. Chat preview always requests model formatting, shows the three message roles and formatted text directly, and retains raw JSON/downloads.
- **53 Streamlit/UI/deployment tests passed.** Checks cover every page, no jobs on navigation, explicit human preferences, draft retention, visible errors and missing API credentials. The first revised run failed because Streamlit 1.63 AppTest omits stateful tab blocks from its simulated browser state. The test helper now includes their actual string-valued widget protocol; browser clicks independently confirmed workspace/page switching.
- An isolated loopback browser preview using bundled public fixtures confirmed direct access, both tab rows, Chat Template Preview without a checkbox, and Training Lab refresh choices. This preview could not submit live changes. A fresh AppTest inside the actual running Streamlit container connected to the real backend and opened directly without login. A separate live request without an API token returned HTTP 401 as expected.
- The Streamlit Docker image rebuilt successfully. Source was also applied to the existing running UI container without restarting it; container ID, process ID, start time and private env-file hashes were unchanged, and health remained healthy. Dataset draft session keys were retained. No review, training, generation, dataset save or model promotion was submitted.
- For the earlier chat-preview change, a real tokenizer response rendered all three messages and the formatted template. Streamlit’s code widget trims trailing whitespace; the initial strict equality assertion was corrected to compare surrounding-whitespace-normalized text. Full backend/ML tests were not rerun for this UI change. The in-app browser’s trust error for the local TLS certificate still limits direct browser verification of the live URL; no certificate check was bypassed.

- Added standalone `compose.server.yml`: Caddy, Streamlit, the existing authenticated FastAPI cloud image, and PostgreSQL. Only the gateway publishes ports; database networking is internal. The original local React stack uses its separate Compose project and remains unchanged.
- **52 Streamlit/deployment tests passed**. Added coverage for the fixed private Docker API origin, required token, login despite local secrets, password length/hash handling, setup injection/overwrite protection, and service recovery after a backup failure. GitHub Actions now also validates the single-server Compose configuration; remote Actions has not run.
- Built and started all four services under the isolated project `recipetriage-server-smoke`. Compose validation and all startup checks passed. Actual HTTPS returned the Streamlit page and health response; the test client verified Caddy's local certificate using an explicit CA file. HTTP redirected to HTTPS. A real WSS connection successfully negotiated Streamlit's WebSocket protocol through Caddy. No OS trust settings were changed.
- The containerized Streamlit client authenticated to FastAPI, checked database health and read the seven seed examples. Unauthenticated API reads and generation requests returned 401. Streamlit retained the password hash exactly, received neither PostgreSQL credentials nor the Fireworks key, and ran as UID 1000. Docker inspection confirmed only Caddy published host ports.
- The maintenance backup stopped only the isolated UI/backend, streamed the PostgreSQL custom dump and training/deployment archive, wrote checksums, and brought services back healthy. Both hashes verified. Restoring the full dump into a separate empty test database succeeded with `pg_restore --exit-on-error`; its marker row was retained. The artifact archive extracted successfully into a disposable container with the QLoRA adapter present.
- Removed and recreated every smoke container without deleting its volumes. The database marker, reference adapter and original local TLS CA remained usable. The temporary test project, its volumes and generated test credentials were then removed. Original local services on 8080, 8000, 8501 and 5050 returned HTTP 200 throughout final checks.
- One diagnostic incorrectly asserted that a default `docker compose exec` shell would inherit the dropped API process UID. The cloud image starts as root for volume setup, so default exec shells use root even though PID 1 drops privileges. The corrected check inspected `/proc/1/status` and explicitly selected `--user appuser`, confirming UID 1000. No application permission fix was needed. Docker build-time pip root/update notices remain visible.

**Public deployment remains pending server access and a real domain.** This verification used localhost certificates, not a public ACME certificate or a purchased cloud host. No GitHub push, paid resource creation, model generation, training, human review or model promotion occurred. Existing records and large deployment artifacts were not uploaded. The single-server setup does not require a GitHub remote or a separate public backend URL. See [SINGLE_SERVER.md](SINGLE_SERVER.md) for setup, backup, migration and operating limits.

---

# Streamlit interface and cloud preparation — 2026-09-12

- Added a native Streamlit interface with seven workspaces and 23 pages, using the existing FastAPI contracts. The local preview at `http://localhost:8501` reads the existing PostgreSQL-backed application. React, FastAPI, PostgreSQL and local MLflow remain available on their original ports.
- **32 Streamlit tests passed**, including all pages, no writes on navigation, explicit preference submission, login gating, API authentication headers and visible/redacted errors. Dependency consistency passed. A separate read-only check opened all **23 pages against the actual API**, including populated saved results, with mutations prohibited by the test transport. Browser inspection confirmed the running library and navigation.
- **53 backend API tests passed in Docker**, using isolated PostgreSQL test schemas. The focused host run passed six tests and skipped the database-dependent check; the subsequent Docker run provided actual database coverage. Existing Starlette/AnyIO deprecation warnings remain visible. The complete ML suite was not rerun for this interface/deployment change. A separate Streamlit GitHub Actions job was added; remote Actions has not run.
- Built the `cloud` Docker target and started it against a separate temporary database and named volume. Public health returned 200; unauthenticated recipe, documentation and generation requests returned 401. Authorized requests returned seven unreviewed seeds and available delivered SFT/LoRA/QLoRA references. A validated JSON intake returned 201 without a Fireworks call. The process ran as UID 1000, and artifact paths resolved into `/data`.
- Restarted that isolated cloud container: health recovered, reference adapters remained available, and its intake record persisted. Removed only the temporary smoke container, its volume and its separate database after verification. No existing application records or model artifacts were deleted.
- The first cloud startup failed because delivered adapter files had restrictive source permissions. The Docker build now grants read access to packaged ML files, and reference installation uses atomic directory copies. The rebuilt image passed startup and restart checks. A later source-inspection command referenced a nonexistent `backend/app/health.py`; the existing health route is in `backend/app/main.py` and required no repair.
- Packaging excludes real `.env` and Streamlit secrets, virtual environments, caches, database-transfer files and large deployment artifacts. Existing human-review, preference and model-promotion gates remain in the backend. No new approvals, preference choices, training, Fireworks calls or promotions were made during this phase.

**Not yet deployed publicly.** Streamlit sign-in and its connected RecipeTriage repository were verified in the browser, but new source has not been pushed. FastAPI/PostgreSQL hosting and private cloud secrets remain to be configured. The optional Render blueprint has been prepared, not provisioned or account-validated; it proposes paid resources and requires cost review. Current database records and large model exports have not been uploaded. See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for exact commands, migration guidance and operating limits.

---

# Local MLflow dashboard — 2026-09-10

Added optional Compose profile `tracking` with the official MLflow 3.16.0 image, matching the backend client. Dashboard HTTP 200 at `http://localhost:5050`; backend connects over `http://mlflow:5000`. Only loopback is published. Metadata and proxied artifact uploads persist in the dedicated `mlflow_data` volume. Existing API and PostgreSQL health remained OK after backend recreation.

Imported the five bundled historical learning-rate/rank experiments without running training or calling Fireworks. All five are FINISHED, each has nine loss points, and all five downloaded `evidence/run.json` files match their source SHA-256 hashes. Browser checks confirmed the five-run table, parameters, benchmark scores and model metric charts. Source journals and human decisions were not modified. The original creation time is tagged; the built-in run Duration describes export time, while `training_seconds` retains measured training time.

Four focused tracking tests passed: HTTP/HTTPS artifact locations belong to the server, direct SQLite retains local artifacts, repeated exports resolve the existing run ID, and a tracking failure preserves training evidence. Compose configuration validated and both containers reached healthy status. MLflow startup emits an upstream Starlette WSGI deprecation warning; it did not prevent startup or artifact access. An initial local port-inspection command used invalid `lsof` options; the corrected check and actual bind verified port 5050 availability. README documents startup, import, stop, persistent storage and Docker commands usable from macOS/Linux/PowerShell. The full project suite was not rerun for this isolated tracking change.

---

# Fireworks configuration repair — 2026-09-10

The user-reported single-model failure was reproduced: the saved Llama 3.3 model setting returned upstream 404 (API 503). Replacing it with yesterday's Qwen 3.7 model also returned 404. The authenticated Fireworks serverless catalog returned 200 and listed neither old model. A GLM 5.3 Flash probe rejected disabled reasoning; a Qwen 3.8 Max probe returned prose outside the required JSON format. These were not successful compatibility checks.

The current replacement, `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`, passed direct triage and schema-based recipe-extraction checks with the existing key. Updated `FIREWORKS_MODEL`, `FIREWORKS_RECIPE_MODEL` and `FIREWORKS_SYNTHETIC_MODEL` in the local `.env`, along with example/Compose/provider defaults. Credentials were unchanged. Single-model inference now disables hosted reasoning like the comparison path; HTTP errors distinguish model access, authentication, billing and rate/capacity failures without exposing secrets or automatically retrying.

After the Docker rebuild/recreation, the actual frontend-proxied `POST /api/v1/triage` returned **HTTP 200, valid_json=true, finish_reason=stop**, in 594 ms on the teaching ravioli recipe. The catalog reports the new model for comparison too. Saved request/response evidence is in `ml/inference/results/fixes/fireworks-config-2026-09-10.json`. **34 focused inference/provider/API tests passed**, including status-specific diagnostics, no automatic retries, secret redaction, configured model selection and disabled reasoning. Older benchmark evidence remains unchanged; this repair does not establish the replacement model's benchmark quality or managed fine-tuning support.

---

# Verification — product use cases, 2026-09-10

- Local folder updated; Docker frontend/backend/PostgreSQL healthy at ports 8080/8000/5432. Recipes now has a saved library, four intake methods, explicit label correction, prediction history and reviewed dataset export. Playground shows persistent side-by-side comparisons.
- Final Docker backend/ML suite: **182 passed, zero skipped**. Frontend: **34 passed**. Final Vite build and backend `pip check` passed. The existing ten HTTP smoke endpoints returned 200 JSON through the frontend proxy.
- Live published oatmeal URL normalized from schema.org Recipe JSON-LD without a model call. A recipe message normalized through Fireworks. A rendered image of the same existing seed text completed local Tesseract OCR and Fireworks normalization through the actual intake endpoint. These created unreviewed intake drafts only.
- Two real five-model ravioli comparisons are saved in PostgreSQL. The final run is `d5310af5-94cb-485a-9717-65548a0976eb`, also exported under `ml/inference/results/use-cases-v1/comparison.json`. All selected providers returned outputs. Fireworks matched the provisional ravioli label set; SFT returned wrong labels; base and LoRA responses were truncated; QLoRA returned an incomplete response with `time_limit`. Invalid/raw outputs and finish reasons remain visible. One recipe is not a benchmark or an improvement claim.
- Comparison generation latency now excludes explicit model preparation; preparation and total timings are also saved. Hosted generation includes network time. The earlier comparison remains intact as legacy timing evidence. A short incomplete output is not a faster successful answer.
- Browser checks verified the populated Recipes page, equipment search, clean selected-recipe handoff, all four intake choices, disabled verified export without reviews, unavailable DPO/production choices and saved comparison cards. The final Recipes layout was visually inspected.
- Seven library recipes remain unreviewed. All 13 curation records are pending, all five preference pairs have no assigned choice, and all five registered models remain candidates. No human decisions, new training examples, retraining or production promotion were fabricated.

## Error found during final verification

The first complete suite passed. Repeating it while the live comparison was running exposed four older queue tests sharing the application's schema: they received the correct HTTP 409 busy response instead of their expected 202. Added `tests/conftest.py` to place test connections in a unique temporary PostgreSQL schema, preserving individual schemas used by newer tests. The corrected full suite passed all 182 tests. Its only cleanup drops that temporary test schema; application data is outside it. Three existing library deprecation/test-fixture warnings remain visible.

Detailed acceptance steps and platform-specific setup alternatives are in [USE_CASES.md](USE_CASES.md). The earlier lifecycle verification below remains historical evidence; its test counts describe the earlier phase.

---

# Verification — human learning and model lifecycle, 2026-09-09

## Final checks

- Final Docker backend/ML suite: **164 passed**, zero skipped, including PostgreSQL integration and an actual random tiny-model two-step DPO update/reference-freezing test. No hosted calls are made by the test suite. Frontend: **29 passed**; Vite build passed. Host and backend `pip check` passed.
- Docker Compose rebuild/start completed with frontend, backend and PostgreSQL healthy. HTTP verification through port 8080 passed health, reviews, preferences, alignment options/results, model gates and deployment comparison. A production-provider request correctly returned **503** because no candidate passed promotion.
- Six actual Fireworks proposals are persisted as pending, together with seven original recipes awaiting review. Five actual Fireworks/current-model preference pairs are persisted with no choices. Synthetic content, prompt/model provenance, automated checks and raw provider errors remain available. No generated example was added to train.jsonl.
- The GRPO probe completed two steps and automatic benchmark/shortcut evaluation. All four sampled outputs reached the token limit, rewards and gradients were zero, and task results did not improve: Micro-F1 **23.53% before and after**, misleading-title accuracy **0/3 before and after**. Training took **97.91 seconds**. The saved result and reward components are visible in Alignment Lab. It remains a candidate only.
- Real LoRA subset comparison completed with identical 67,584-parameter budgets: early-six **29.87 s / 19.05% Micro-F1**, late-six **16.35 s / 43.24%**. Neither beat the **48.48%** base. Notebook 11 executed all cells using saved results and its generated plot was visually inspected. Hidden-state comparisons assert identical token hashes and token counts; descriptive statistics are not causal attribution.
- Exported and freshly loaded the base FP32, merged late-six FP32 and merged NF4 models; each ran the full fixed benchmark and title suite. FP32 merge exactly reproduced all ten source-adapter raw benchmark responses. Every exported tokenizer produced identical IDs to the pinned original on all ten benchmark prompts.
- Deployment Micro-F1: base **48.48%**, merged FP32 **43.24%**, merged NF4 **0%**. NF4 reduced storage from about **1900 MiB to 711 MiB**, but increased CPU mean latency from about **3745 ms to 20764 ms** for the merged candidate. The default quality/operational constraints select the base, not the smallest export. No model meets the separate strict production gate.
- Five actual versions registered in PostgreSQL as **candidate**; zero staging/production versions and no stage events were created in the user's data. Integration tests use isolated schemas for promotion, file tampering, single production pointer and rollback. Current production supplies the per-label regression reference when present; otherwise the frozen base does.
- Live browser checks covered the pending review queue, corrected revision number, explicit-only A/B/Tie/Neither controls, disabled DPO without choices, GRPO reward/results, failed registry gates and disabled promotion, and the full deployment table. Setting p95 ≤ 1000 ms correctly returned no feasible candidate; changing constraints performed no training or promotion. Narrow layout was visually inspected.
- The user's tokenizer notebook remains SHA256 `d181756bbb8f7ab630068802b1f89c8c9c27bf854b5074ef10769a1a145fb59e`. The `.env` file was preserved. Full-source packaging includes earlier code, selected adapters, notebook outputs and new result evidence; it excludes API keys, virtual environments, caches, optimizer checkpoints and the 4.4 GiB standalone export directory. Those full exports remain in the user's local project.

## Pending human decisions and limitations

Reviewed-data QLoRA and real DPO have **not** run. Their code, UI and tests are implemented; 13 explicit review decisions and usable human A/B choices are still pending. The user's “Tie” reply has not been assigned to a recipe or all five pairs because its scope is unconfirmed. Ties cannot supply a chosen/rejected DPO pair. No review decisions were invented to bypass these constraints.

Seven seed recipes and the ten-case benchmark are provisional. The ravioli shortcut source appears in the original training seed, so the three-source shortcut score is a diagnostic with known source overlap. Repeated inspection of the benchmark further limits generalization claims. No measured improvement is claimed for GRPO or these deployment candidates. Authentication, multiuser authorization, service resource isolation and backup automation remain prerequisites for exposing this localhost teaching app publicly.

GitHub Actions is configured with the new tests and HTTP smoke checks but was not run remotely. Only the local folder was updated; no commit, push, public deployment or automatic model promotion was performed.

## Errors found and corrected

- The optional analysis notebook initially failed because matplotlib was missing. Added pinned analysis dependencies and reran every cell successfully. The notebook kernel still printed its loopback TCP warning and a sandbox `sysctl` permission error during subprocess cleanup; execution completed with no failed cells.
- Docker-copied adapter files initially retained ownership/permissions that prevented the non-root application user from reading them. Corrected ownership in the training volume and documented the command. New standalone export scripts produce readable artifact files, and the existing exports were corrected without changing content checksums.
- Fireworks output's optional model revision overwrote the synthetic review revision. The generator now stores `model_revision` separately; an idempotent migration repairs only untouched pending null-revision records. The six saved journals were repaired with the original text, prompts and statuses preserved, and the repair is documented. A regression test covers generation and migration.
- Corrected a deployment constraint selection that was previously reset by UI refresh; UI tests and a live infeasible-limit check now verify the selected constraints remain effective.
- Historical Fireworks connection/404 errors remain in the synthetic evidence. Transformers' Mistral-regex warning on local Qwen tokenizer exports remains in logs; the ten-case token-ID parity check passed without modifying tokenization. Existing Starlette/AnyIO and tiny-test PEFT warnings remain visible. No warning was hidden by fabricating results or silently upgrading the ML stack.

## Earlier checkpoint — experiments, QLoRA and failure analysis, 2026-09-09

## Current phase checks

- Completed the fixed four-cell LoRA learning-rate × rank matrix and a matched fifth QLoRA run on this Mac's CPU. All five runs used the same Dataset v1 snapshot, pinned base revision and ten-case benchmark. No synthetic training data were added. Raw journals, selected adapters, model/quantization audits, losses, timings and every benchmark response are included under `ml/training/results/experiments-v1/`.
- Actual Micro-F1: LoRA LR 0.00005/r4 **9.09%**, LR 0.00005/r16 **34.48%**, LR 0.0002/r4 **41.38%**, LR 0.0002/r16 **0%**; matched QLoRA LR 0.0002/r4 **23.53%**. The unchanged base scored **48.48%**, so none improved classification. All adapters had 0/10 exact label sets. See EXPERIMENTS.md for the full comparison and one-factor effects.
- bitsandbytes 0.50.2 passed an actual CPU NF4 backward probe. The model audit verified all 168 Transformer linear projections use NF4 with double quantization. The selected QLoRA adapter reloaded into a fresh quantized base, reproduced benchmark case bench-001 exactly and had every parameter frozen for inference. The reload verification and adapter checksum are included.
- Matched rank-4 measurements: LoRA **27.04 s / 3066.67 MiB peak RSS**; QLoRA **31.78 s / 1875.42 MiB**. These are one-run CPU process measurements during training, not GPU VRAM or a general performance ranking. Model setup and benchmark inference are outside the measured interval.
- All five runs exported successfully to MLflow. `mlflow-export.json` contains actual metadata retrieved from its tracking store. Portable evidence includes the receipts; new machines can re-export to their own store with the documented CLI. Tracking failures have separate status and do not overwrite model results.
- Ran all 12 controlled title variants and the separate three-probe red-team suite on the base, matched LoRA and matched QLoRA. Title tests enforce identical time, ingredients, equipment, instructions and pantry, including raw text rather than only normalized values. Base had no usable pairs, so its flip rate is undefined. Both adapters had 0 flips in 3 usable pairs, with 6/9 pairs excluded. All three models scored 0/3 on the predeclared misleading Quick-title cases. This does not demonstrate title robustness. See FAILURE_ANALYSIS.md.
- All three models retained the correct usable answer on 0/3 red-team probes; none returned the exact requested attack marker alone. Invalid outputs are not automatically counted as evidence of following an injection. Raw responses and separate marker-presence fields remain available.
- Imported the experiment, all five child runs, six failure analyses and three diagnostic runs into PostgreSQL. Import validation recomputes metrics from raw outputs, checks counterfactual invariance and rejects altered immutable evidence. Failure category rows and prioritized collection recommendations are persisted/queryable.
- Final Docker backend/ML suite: **147 passed, zero skipped**, including real PostgreSQL integration. Frontend: **26 passed**; Vite production build passed with lazy page chunks and no chunk-size advisory. Backend `pip check` reported no broken requirements.
- Docker Compose rebuilt successfully; PostgreSQL, backend and frontend are healthy. New experiment-options and failure-source HTTP smoke assertions passed through Nginx. GitHub Actions includes them but was not executed remotely.
- Live browser verification: Failure Analysis displays the imported categories, exact original misleading-title denominator, raw evidence and priorities. Shortcut Tests displays undefined-rate handling, pair exclusions, all 12 inputs and the separate red-team panel. Experiments displays all five results and synced MLflow receipts; QLoRA Training shows a successful runtime NF4 probe and matched checkpointing controls. Page navigation did not start additional training or inference runs. Narrow layouts use the Workspace selector.
- The user's `.env` and tokenizer notebook were preserved. Source packaging includes selected adapter exports and actual evidence, with checksum verification and a scan against configured API keys. It excludes secrets, environments, model caches and optimizer/runtime directories.

Remaining limitations: seven provisional seed recipes, one validation recipe, ten provisional benchmark cases, three counterfactual bodies and one execution per configuration. The benchmark has been repeatedly inspected; it supports exploratory diagnosis, not an untouched final test claim. A single API worker runs background jobs; restart recovery marks unfinished work interrupted. Saved reference adapters are included on the host and in the ZIP but excluded from Docker images. Starlette/httpx TestClient and AnyIO BlockingPortal deprecation warnings remain visible; they were not suppressed. No managed training, public deployment or remote GitHub Actions run occurred in this phase.

## Earlier checkpoint — Supervised Fine-Tuning

## SFT phase checks

- Local CPU Transformers 4.57.6 + TRL 0.24.0 + PEFT 0.17.1 training completed: 3 epochs, 9 optimizer updates, 540,672 trainable adapter parameters. Validation loss 2.858231 → 2.128406; selected checkpoint 9 by validation loss.
- Automatic before/after RecipeTriage-Bench-v1 runs completed on all 10 unchanged cases. Micro-F1 48.48% → 38.71%, macro-F1 36.50% → 16.36%, exact sets 0/10 → 0/10, JSON validity 50% → 90%. No classification improvement is claimed. Raw responses, metrics, runtime, configuration and dataset hashes are retained.
- The selected adapter was reloaded against the pinned base revision, checksum verified, and its first benchmark response reproduced exactly.
- Backend/ML Docker suite: **117 passed**, including six real PostgreSQL integration tests. Coverage includes masking prompt/padding while retaining EOS, token-boundary checks, overlength rejection, snapshot tampering, benchmark leakage, equal logical messages across providers, parameter validation, managed job lifecycle, automatic benchmark invocation, cleanup on failure, immutable evidence, restart recovery and queue conflicts. Tests make no paid model calls.
- Frontend: **18 tests passed**. Verified default form submission, displayed regression, no training on page open, managed preflight blocking, explicit API/gateway errors and recovery after successful refresh. Production build passed.
- Docker rebuilt successfully; frontend, backend and PostgreSQL are healthy. `python -m pip check` passes in the final backend image and the local virtual environment. Fixed an inherited backend pin to ML 0.2.0 to match the new ML 0.5.0 package.
- Actual run journals imported into PostgreSQL; downloaded local and initial Fireworks evidence matched source JSON. Current project Fireworks preflight was imported too. API health, training options and model configuration were checked through Nginx on port 8080.
- Browser verification: Training Lab opens the real Training Monitor; defaults, dataset counts, selected checkpoint, loss table, measured benchmark regression, provider selection and read-only preflight results are visible. A transient gateway error during backend recreation exposed stale error display; polling now clears that error after a successful refresh, and a regression test covers recovery.
- Managed live training **was not run**. The working-copy Qwen3.7 Plus preflight reported no supervised LoRA support. The user's GitHub `.env` retains Llama3.3 70B, which reports supervised LoRA support; its check is blocked by one validation recipe (documented minimum 3) and a missing compatible deployment shape. No datasets, managed jobs or remote deployments were created. No synthetic or duplicated recipes were added to bypass constraints.
- The API key and user notebooks were preserved. Full-source ZIP packaging excludes secrets, virtual environments, model caches and runtime job directories; selected reference adapter and actual evidence are included with portable readable file permissions.

Remaining limitations: tiny unreviewed seed and benchmark; missing training-label coverage; no statistical generalization claim. Managed lifecycle is covered with mocked API tests, but a successful live managed fine-tune/evaluation requires a supported dataset and deployment configuration. Remote metrics remain in the provider console when available. Single API worker; interrupted local work requires a new run, and remote jobs require monitoring to resume after a restart. Trainer checkpoints remain in the original local run directory; only the selected adapter is bundled.

Warnings were not treated as passing checks: initial UI syntax and learning-rate form-validation failures were corrected before final verification. The final frontend build retains Vite's advisory about a roughly 509 kB JavaScript chunk (about 155 kB gzip). The test stack emits existing Starlette/httpx deprecations. Actual training logs retain tokenizer PAD/BOS alignment and PEFT's offline vocabulary-config warning; no vocabulary resize was performed. GitHub Actions was updated, including a training-options smoke check and package consistency check, but was not executed remotely here.

## Earlier baseline verification

## Baseline phase checks

- Docker backend/ML suite: **92 passed**, including four real PostgreSQL integration tests. New coverage includes hand-calculated label metrics, JSON/schema distinctions, truncated output, incomplete-run scoring, frozen cases, contamination audit, mocked provider errors, import evidence checks, immutable history, enqueue conflicts, and successful background execution with per-case persistence.
- Frontend: **12 tests passed**. Comparison, error display, run details, downloads, fixed cases, protocol warnings, and request settings are covered alongside the existing UI tests.
- Production frontend build passed. Docker Compose rebuilt and all three services became healthy. The final backend image includes the updated provider default and all five real result files.
- The documented container import command succeeded for all five bundled runs, including repeated idempotent imports. No hosted calls occur during import.
- Live Nginx/API checks passed after backend recreation: database health, ten-case frozen manifest, configured provider metadata, five saved runs, and downloadable evidence equal to the original JSON files. Every imported file passed raw-output parsing and metric recomputation checks.
- In-app browser checks confirmed the comparison table, per-label/category details, Salsa Chicken's raw mismatch, fixed-case tab navigation, and all six categories. The mixed-protocol warning and blocked-run dashes were visible. Browser inspection did not start additional paid runs.
- GitHub Actions includes the new benchmark-manifest smoke check. It has not been executed remotely here.

## Actual model execution

The final pair used the same frozen ten cases, temperature 0, 128 output tokens, and a reasoning-disabled protocol. Local execution was the pretrained Qwen2.5-0.5B checkpoint on CPU, not the earlier Instruct checkpoint.

| Run | Micro F1 | Macro F1 | Exact sets | JSON / schema valid | Usable | Mean / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Local base `6bd9eeec` | 48.5% | 36.5% | 0/10 | 50% / 40% | 4/10 | 4.850 / 6.167 s |
| Fireworks Qwen3.7 Plus `bfec561c` | 94.1% | 95.1% | 9/10 | 100% / 100% | 10/10 | 0.605 / 0.803 s |

The earlier local run, hosted HTTP 404 attempt, and hosted default-reasoning run with ten truncated empty final answers are preserved too. The paired CLI exited 1 because the base model had unusable answers; the benchmark files were saved successfully. No fine-tuning or weight updates occurred.

## Errors, warnings and limits

- The originally configured Llama model returned HTTP 404 for this account. A live authenticated model-list/metadata check identified Qwen3.7 Plus as available. The working default was changed and the old error kept. The hosted alias has no pinned revision in our evidence.
- Default reasoning exhausted the 128-token budget without final text. The explicit mode change is recorded under a different protocol ID. Both final providers were rerun with matching protocol metadata.
- Tests emitted two dependency deprecation warnings: Starlette's current httpx TestClient integration and AnyIO's BlockingPortal alias. They are recorded warnings, not failures; dependency migration was not part of this phase.
- Docker's build-time pip emitted its standard root-install warning. Runtime application containers use the configured non-root user.
- This Mac's Docker credential helper previously reported a Keychain error. Verification used an isolated Docker configuration under `/tmp` and the local Docker Desktop socket. That workaround is not required on machines with a functioning helper; see README for ordinary commands.
- Benchmark labels remain provisional and human-unreviewed. Ten cases and one repetition do not establish production accuracy or stable tail latency. Exact Dataset v1 body/source overlap is zero; pretraining overlap and semantic paraphrase overlap are unknown.
- Single-worker background execution stores progress in PostgreSQL but is not a distributed task queue. Restart recovery marks unfinished jobs interrupted. API keys are excluded from evidence and deliverable archives.

See [BENCHMARK.md](BENCHMARK.md) for the full lesson, cases, file map, run IDs, scoring policy and platform alternatives.

# Earlier checkpoint: Dataset Engineering v1, 2026-09-09

## Current phase checks

- Docker backend suite: **66 passed**, including real PostgreSQL integration. Coverage includes invalid labels, duplicates/conflicts, missing fields, source/group leakage, deterministic splits, JSONL/chat formatting, file hashes, malformed imports, migration idempotence, immutable persistence, and ZIP downloads, plus existing inference/health tests.
- Frontend suite: **8 passed**, including Dataset Studio build/save, invalidation after edits, retained invalid imports, malformed object-valued field rendering, chat navigation, and existing Token Inspector/navigation checks.
- Final Vite build passed locally and in Docker. Compose rebuilt and reported PostgreSQL, backend, and frontend healthy.
- CLI produced seven recipes as **5 train / 1 validation / 1 test** with seed 42. Host and container CLI commands produced the same version and exported file bytes.
- Initial version: `v1-63135eef80fb9be90547673e7f13d4142eb0eb29852cf098e672b44fca20158f`, status `teaching-draft`, zero reviewed records.
- Actual HTTP checks through Nginx passed: health, seed/preview, saved version retrieval, ZIP integrity, every exported file hash, invalid-label rejection (422), and real Qwen chat-template rendering with completed assistant target.
- Saved seed snapshot remained available after backend/container recreation. API ZIP contents match the bundled CLI files; independent metadata creation timestamps differ intentionally.
- Browser DOM checks confirmed Dataset Studio navigation/build/save, Label Editor policies/source/review controls, JSONL file preview, and loading the persisted version after recreation. The final Chat Template Preview displayed the actual Qwen role markers and complete assistant target.
- Dataset engineering does not require model generation or a Fireworks key. No synthetic recipe generator, fine-tuning, or model weight updates were added.

## Current limitations and observed errors

- The seed contains individually assembled summaries of published recipes and assistant-proposed annotations. They remain visibly unreviewed. It lacks dessert and pantry examples; a seven-row set cannot provide representative model-quality estimates. Metadata exposes these gaps.
- Stratification is a deterministic grouped greedy approximation. Exact normalized-body and source-family checks do not catch every paraphrase; inspect groups and use `group_id` for known variants. Public recipe pretraining contamination is not tested.
- Unsaved UI drafts are component-local; save a version before reload or leaving Data Lab. PostgreSQL versions persist in the named volume. Draft exports are allowed with explicit warnings, and marking a record reviewed records a user assertion rather than independently certifying it.
- The first new frontend validation test failed because the testing library interpreted JSON brackets as keyboard syntax. Changed the test to paste the JSON; final tests pass. A subsequent import-rendering regression test also passes.
- This turn initially lacked Docker socket access. After the requested sandbox permission was granted, the real Docker/PostgreSQL checks passed. No check was reported as passing while blocked.
- Browser tabs changed/closed during verification; checks resumed in a fresh app tab. No application data was lost: the saved version reloaded from PostgreSQL.
- Starlette TestClient and AnyIO deprecation warnings remain visible. Docker build-time pip root notices remain visible; application containers run as a non-root user.
- The workflow includes the new dataset tests and an HTTP seed smoke check. GitHub Actions has not been executed remotely. No public deployment or live Fireworks call was performed in this phase.

## Earlier checkpoint — tokenization and inference

## Passed

- Python tests with PostgreSQL enabled: **29 passed**. The tests cover the old health endpoint plus prompt construction, strict JSON parsing, title-only interventions, Fireworks request/response/error contracts through a mock transport, API input validation, and token endpoint behavior.
- Docker test image: **29 passed** against the real PostgreSQL service.
- Frontend tests: **4 passed**, covering navigation, actual returned token/ID/count rendering, invalidation on edits, and visible backend errors.
- Vite build: passed locally and inside Docker.
- Docker Compose build/start/readiness: all three services healthy.
- Python dependency consistency: no broken requirements reported.
- Actual tokenizer through Nginx `/api/v1/tokens`: returned 10 tokens for `Easy ravioli takes 70 minutes.`.
- Actual local Qwen inference through Nginx `/api/v1/triage`: completed with `finish_reason=stop`, 383 input tokens and 123 output tokens on the Traditional variant. The observed request including cold model loading took about 92 seconds. This is a single observation, not a latency benchmark.
- Notebook: all five code cells executed; no cell error outputs remain. Output includes real vocabulary pieces, tensors, logits, float64 softmax probabilities, entropy changes with temperature, and model generation.
- Local four-title experiment: four schema-valid responses, saved in `ml/experiments/ravioli-hf-fd9d40cf-d4da-4452-a8c0-5a3eb9059a96.json`.

## Model findings and incomplete external verification

All four local title variants incorrectly predicted `weeknight-30min` for the 70-minute recipe. The zero label-flip rate represents consistent error, not correctness. Raw answers are retained without label repair.

No Fireworks API key was configured. The adapter was tested with a mock transport, but **live Fireworks inference and a hosted comparison were not run**. The corresponding experiment file records the configuration blocker for all four titles. Run the documented experiment command after configuring `.env` to produce real hosted evidence. The experiment command intentionally returned exit status 1 for the blocked hosted cases after saving both result files.

GitHub Actions was configured but not executed remotely. No public deployment or paid hosted calls were made. Browser DOM/visual QA was not performed; frontend interaction tests and actual HTTP delivery/inference were checked.

## Errors and warnings, retained honestly

- Network and Docker socket permissions were required for package/model downloads and container operations.
- Docker's macOS credential helper previously returned `Keychain Error. (-67674)`. Builds used an isolated temporary public-registry Docker configuration. Saved credentials and the user's Docker configuration were not changed.
- The first notebook run failed an overly tight float32 probability-sum assertion (observed sum about 1.00007). The demonstration now explicitly uses float64 softmax and explains finite precision; re-execution passed. Model weights remain float32.
- The notebook kernel printed a local TCP transport warning and a sandbox `sysctl` permission error during subprocess cleanup. These are notebook-runtime environment diagnostics; the final notebook completed, saved its outputs, and had no failed cells.
- Transformers warned that sampling-related defaults from the model's generation configuration are ignored during greedy decoding. The experiment uses `do_sample=False`; temperature/top-p/top-k do not drive token sampling in those runs. This warning was not suppressed.
- Existing Starlette/AnyIO TestClient deprecation warnings remain visible. They did not fail tests. pip also printed cache-permission/root-install notices; the runtime application containers run as non-root users.

Local environment: macOS ARM64, Python 3.12, PyTorch 2.14.0, Transformers 4.57.6, Node 24.4.1. Docker uses Python 3.12, the CPU PyTorch wheel, Node 22, PostgreSQL 17 and unprivileged Nginx. Qwen model and tokenizer are pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`.

## Notebook import fix — 2026-09-09

A user kernel could import PyTorch but not the local `recipetriage_ml` package. Added a first setup cell that locates the complete checkout, verifies Python 3.12+, displays the kernel interpreter, and installs the local package into that interpreter when missing. Added editable-install/restart troubleshooting. Cleared notebook outputs to avoid presenting the revised cell structure as already executed. The previous lesson's successful execution remains recorded above; the setup-only revision does not change inference code.


## LoRA phase verified on 2026-09-09

- Real ranks 4 and 16 trained in sequential fresh Python processes against the pinned Qwen base and the unchanged seed snapshot. Nine optimizer steps per rank; both complete before/after ten-case benchmarks are retained in `ml/training/results/lora-v1/`.
- Rank 4: 270,336 trainable parameters, 29.74 s training, 2966.30 MiB peak sampled process RSS, 41.38% Micro-F1. Rank 16: 1,081,344 trainable parameters, 29.52 s, 2941.23 MiB, 0% Micro-F1. Both fresh base runs scored 48.48%; neither adapter improved classification. See LORA.md for scope and limitations.
- Both selected adapter exports loaded successfully into fresh pinned base models and exactly reproduced benchmark case `bench-001`. Every parameter was frozen after inference loading. Saved checkpoint checksums and reload evidence are included.
- `docker compose up --build --wait --wait-timeout 180`: frontend, backend and PostgreSQL healthy. The source build contains the required HF and SFT lockfiles.
- Docker backend/ML suite: **131 passed**, including all PostgreSQL integration tests; no skipped tests in that run. Frontend suite: **22 passed**; Vite production build passed with lazy training-page chunks and no chunk-size advisory. `pip check`: no broken requirements.
- Completed experiment and child runs imported into PostgreSQL twice to verify idempotence. API integration tests check immutable evidence, queue exclusion, failed worker cleanup and restart interruption state.
- Live browser: narrow layouts use a workspace dropdown; desktop retains the left navigation. Training Lab tabs, inspected architecture, default query/value estimates, target-policy changes, saved comparison and evidence links checked. Targeting all attention projections doubled the rank-4 estimate to 540,672 as expected. Opening the page did not start another training run.
- The existing tokenizer notebook was preserved: SHA256 `d181756bbb8f7ab630068802b1f89c8c9c27bf854b5074ef10769a1a145fb59e`.
- Errors encountered and fixed: PEFT compresses long target lists to suffixes when saving; the loader now validates those suffixes against the real module tree. Initial pytest collection collided on duplicate `test_lora.py` names; the API test was renamed. A request-validation unit test needed a dependency override to remain database-independent. All resulting tests passed.
- Remaining output warnings: Starlette's httpx TestClient compatibility and AnyIO BlockingPortal alias are deprecated. They do not fail the tests. No dependency versions were changed to hide them. Docker pip root-install/update notices are build-time notices; the application itself runs as `appuser`.
- This phase made no managed training calls and did not push a commit or run remote GitHub Actions. CI now includes an architecture/parameter-count API smoke check for a future push.
