# RecipeTriage AI

React + Vite + Material UI, FastAPI + Pydantic, PostgreSQL + SQLAlchemy, Docker Compose, pytest/Vitest and GitHub Actions.

**Start here:** the [User Guide](USER_GUIDE.md) explains where to click, what each result means and what to do next in the Streamlit app. A [printable PDF](docs/user-guide/RecipeTriage_User_Guide.pdf) and [copy-and-paste starter dataset](docs/user-guide/sample-recipes.json) are included. It covers all seven workspaces, the Data Lab and Alignment Lab flows, training progress and common errors.

Current phase: **Recipe product workflows and the complete learning/model lifecycle**. The [UC-01–UC-15 acceptance map](USE_CASES.md) covers implementation, exact setup commands and remaining model/data prerequisites. Read [HUMAN_LEARNING.md](HUMAN_LEARNING.md) for synthetic review, preferences, DPO and GRPO; [ADVANCED_ANALYSIS.md](ADVANCED_ANALYSIS.md) for layer experiments; [DEPLOYMENT.md](DEPLOYMENT.md) for measured export/quantization results; and [MODEL_REGISTRY.md](MODEL_REGISTRY.md) for versioning, lineage, staging, production gates and rollback.

Data Lab now includes Synthetic Review. Alignment Lab includes Preference Pairs, DPO Training, GRPO Experiment and Reviewed QLoRA. Deployment includes Model Registry and deployment comparisons. Training and Evaluation retain the earlier experiments, QLoRA, LoRA, SFT, failure analysis, shortcut and baseline pages. Recipes now provides text/message/URL/screenshot intake, a saved searchable library, triage, explicit human correction and verified dataset export. Playground includes persistent side-by-side model comparison.

All independent implementation paths are present. Human-dependent work remains pending: explicitly review the 13 queued records and choose each preference pair before reviewed-data QLoRA or real DPO training. A general “yes” is not a record-level approval; a tie is not an A/B preference. No generated example has been automatically added to training, no synthetic review has been fabricated, and no model has been promoted.

Earlier lessons remain available: [BENCHMARK.md](BENCHMARK.md) covers metrics and the fixed comparison, [DATASET_V1.md](DATASET_V1.md) covers Dataset Engineering, and [LEARNING.md](LEARNING.md) covers tokenization and inference. See [TRAINING_FUNDAMENTALS.md](TRAINING_FUNDAMENTALS.md) for the explicit PyTorch loop and [VERIFICATION.md](VERIFICATION.md) for execution checks and limitations.

## Streamlit navigation

For **$0 Streamlit hosting with application logic in the same process**, use [STREAMLIT_FREE.md](STREAMLIT_FREE.md) and entrypoint `cloud/streamlit_app.py`. It needs a hosted PostgreSQL free tier, but no separate API host. Data curation and saved evidence work there; training, paid providers and model serving remain in the local deployment.

The Streamlit app opens directly with tabs for all seven workspaces and their pages. There is no app sign-in screen or checkbox control. Chat Template Preview always requests the formatted model template and shows System, User and Assistant messages directly. Other binary settings use named choices, such as Plain text / Chat template and Save only / Save and run triage. Navigation does not submit reviews or start jobs.

The single-server local-check address is https://localhost:8543/ while running. Anyone who can reach this interface has access to its shared features; local-check mode remains bound to `127.0.0.1`. Backend API authentication remains required in server deployments. Existing private configuration files are preserved and old workspace-password settings are ignored.

Training Lab’s **Check configuration** supports Local and Fireworks separately. Local checking needs no Fireworks credentials and creates no training job. See [SFT.md](SFT.md) for what the check covers and which checks happen when training starts.

## Run with Docker

To host **Streamlit, FastAPI and PostgreSQL together on one server**, use [SINGLE_SERVER.md](SINGLE_SERVER.md). It includes a standalone Compose deployment, automatic HTTPS, private service networking, setup and backup commands.

For the full **Streamlit interface plus separately hosted FastAPI/PostgreSQL**, see [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md). The local Streamlit preview runs on port 8501 and reuses the existing API. The React/Docker instructions below continue to work.

Install/start Docker Desktop on macOS/Windows (Linux containers), or Docker Engine with Compose v2+ on Linux. From this repository root:

```sh
cp .env.example .env
docker compose config --quiet
docker compose up --build --wait --wait-timeout 180
```

Copy `.env.example` only on first setup. PowerShell uses `Copy-Item .env.example .env`; the Docker commands are identical. This phase has a larger backend image because it includes CPU PyTorch and Transformers. First local inference downloads roughly 1 GB of pinned model weights; first token inspection downloads only tokenizer artifacts. Allow several GB of disk and runtime memory.

- App: http://localhost:8080 — start in **Recipes**, then explore the learning labs.
- API docs: http://localhost:8000/docs
- Database readiness: http://localhost:8000/health or http://localhost:8080/api/health
- Triage API: `POST http://localhost:8000/api/v1/triage` (also available on port 8080).

```sh
docker compose ps
docker compose logs --tail=100 backend frontend postgres
curl --fail http://localhost:8000/health
```

PowerShell: `Invoke-RestMethod http://localhost:8000/health`, or use `curl.exe --fail`. Expected health JSON is `{"status":"ok","database":"ok"}`. This endpoint checks PostgreSQL, not model readiness.

Fireworks is optional: set `FIREWORKS_API_KEY` in `.env`, choose an accessible `FIREWORKS_MODEL`, then run `docker compose up -d backend` to recreate the backend with updated settings. For recipe intake and multi-model comparison, `FIREWORKS_RECIPE_MODEL` is a separate setting; see USE_CASES.md. Keys remain server-side. Missing keys produce an explicit error, not a fake result.

## Fireworks model-access errors

The single-model Playground and baseline runner use `FIREWORKS_MODEL`. Recipe intake and multi-model comparison use `FIREWORKS_RECIPE_MODEL`; synthetic generation uses `FIREWORKS_SYNTHETIC_MODEL`. All examples currently default to `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`. Updating `.env.example` does not change an existing `.env`.

If one page succeeds while another returns HTTP 404, inspect the corresponding model setting in `.env`. On 2026-09-10, both `llama-v3p3-70b-instruct` and the previously working `qwen3p7-plus` returned 404. Neither was in the current serverless model list. The same key successfully called `nemotron-lightning-3p5-30b-a3b` for triage and structured recipe extraction. Use Fireworks’ [current serverless catalog](https://docs.fireworks.ai/faq-new/models-inference/how-to-check-if-a-model-is-available-on-serverless) to select an available model, then run `docker compose up -d backend` to recreate the container with the new environment, then run inference again. `docker compose restart` alone does not apply changed Compose environment values. Your API key does not need to be pasted into chat or the frontend.

Errors now distinguish model/deployment/access failures (404), authentication/access (401/403), billing (402), and rate/capacity limits (429), following the [Fireworks error-code reference](https://docs.fireworks.ai/guides/inference-error-codes). Single-model triage disables reasoning so its small completion budget is available for the JSON answer, matching the comparison path. Existing benchmark error records remain historical evidence; they are not a new connectivity test.

## Local MLflow dashboard (optional, free software)

MLflow's [self-hosted server](https://mlflow.org/docs/latest/self-hosting/) displays parameters, loss curves, benchmark metrics and evidence files. It is separate from the RecipeTriage app and its PostgreSQL registry. Starting it or importing saved results does not train a model or call Fireworks.

In your existing `.env`, set:

```dotenv
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=RecipeTriage-fixed-benchmark
MLFLOW_PORT=5050
```

From the repository root, with Docker running:

```sh
docker compose --profile tracking up -d --wait mlflow
docker compose up -d --build --wait backend
```

Open **http://localhost:5050**, select **Model training → Experiments → RecipeTriage-fixed-benchmark**. Open a run's **Overview** for parameters and final scores, **Model metrics** for loss curves, or **Artifacts** for saved reports. The first startup downloads the MLflow image and creates its database. An empty installation has no experiments until you export a run. To import the five bundled, previously measured experiments:

```sh
docker compose exec backend sh -c 'python -m recipetriage_ml.training.tracking /app/ml/training/results/experiments-v1/runs/*/run.json'
```

The import copies historical evidence, leaves source journals and PostgreSQL records unchanged, and matches existing runs by their RecipeTriage run ID on subsequent imports. The dashboard's import timestamps are not new training dates; the `source_created_at` tag records each original date. The built-in **Duration** measures the export; use the **training_seconds** metric for measured training time. Source journals may contain MLflow IDs from an earlier tracking store. The `recipetriage_run_id` tag joins each imported result to the RecipeTriage registry. These five runs use the original tiny provisional dataset; importing them does not improve the model or approve data.

For future experiment matrices, leave **Training Lab → Experiments → Export results to MLflow** checked. Docker clients use `http://mlflow:5000`; your browser and host Python clients use `http://localhost:5050`. Host training must explicitly set `MLFLOW_TRACKING_URI` to that localhost URL in its environment. Blank URI retains the earlier local SQLite export mode, whose files are not automatically imported into this dashboard.

```sh
docker compose logs --tail=50 mlflow
docker compose stop mlflow
docker compose --profile tracking up -d --wait mlflow
```

The named `mlflow_data` volume persists metadata and artifact files across stops and container recreation. Avoid `docker compose down -v` if you want to retain data; it deletes this volume and the app's other named volumes. The dashboard port is published only on loopback. If 5050 is occupied, change `MLFLOW_PORT` and recreate `mlflow`; the internal backend URI remains `http://mlflow:5000`. These Docker commands work on macOS, Linux and PowerShell; `sh` and the wildcard run inside the Linux backend container. No host Python or MLflow installation is needed.

## LoRA rank comparison

Import the bundled measured experiment without starting training:

```sh
docker compose exec backend python -m app.lora /app/ml/training/results/lora-v1/experiment.json
```

Open **Training Lab → LoRA Training**. Read [LORA.md](LORA.md) before interpreting time, process memory or benchmark scores. Clicking **Run rank experiment** launches new local training; opening the page does not. Adapter exports are inference checkpoints; full optimizer checkpoints stay in the training artifact volume.

## Baseline comparison

Import the five bundled historical runs into your database without making model calls:

```sh
docker compose exec backend sh -c 'python -m app.benchmarks /app/ml/evaluation/results/*.json'
```

Open **Evaluation Lab → Compare runs**. The final matching-protocol pair scored 48.5% micro F1 / 0 of 10 exact matches locally and 94.1% micro F1 / 9 of 10 on Fireworks. The ten-case answer key is provisional and unreviewed; these are teaching results. Earlier HTTP 404 and token-budget failures remain visible. Read BENCHMARK.md before interpreting scores.

To run both providers again, including new hosted requests:

```sh
docker compose exec backend python -m recipetriage_ml.evaluation --provider both --reasoning disabled --temperature 0 --max-new-tokens 128 --output /tmp/benchmark-results
docker compose cp backend:/tmp/benchmark-results ./benchmark-results
docker compose exec backend sh -c 'python -m app.benchmarks /tmp/benchmark-results/*.json'
```

These Docker commands also work in PowerShell. The CLI saves results before exiting 1 for blocked runs or unusable answers; inspect those saved files even when the command reports failure. Alternatively, click **Run benchmark** in Evaluation Lab to run one selected provider and persist progress directly in PostgreSQL. The local base model downloads separately from Playground's instruct model on first use.

## Tests

```sh
cd frontend
npm ci
npm test
npm run build
cd ..
docker compose up -d --wait postgres
docker compose --profile test run --build --rm backend-tests
```

Frontend host tests require Node 22.12+ and npm. Docker uses Node 22. The backend test command includes PostgreSQL integration tests in temporary schemas, isolated from the running app; provider tests use mocks and incur no hosted calls; a tiny random-model DPO test exercises real optimizer updates. GitHub Actions runs the tests and container smoke checks after you push this repository. A remote GitHub run has not been performed here.

## Host development

Install Python 3.12 and Node 22.12+. From the root:

```sh
docker compose stop backend frontend
docker compose up -d --wait postgres
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-test.lock -r backend/requirements-intake.lock
.venv/bin/python -m pip install -e ./ml -e ./backend
.venv/bin/python -m pytest -q
.venv/bin/python -m app.migrate
.venv/bin/python -m app.library --seed
.venv/bin/python -m app.recover_benchmarks
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

This minimal host install runs unit tests and the Fireworks provider. Screenshot OCR also needs the Tesseract executable; Docker includes it, and USE_CASES.md gives host alternatives. For the **local HF provider, inspector, and notebook**, also follow the HF dependency installation in LEARNING.md. Then, in a second terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open the Vite URL printed in the terminal, normally http://localhost:5173. Vite proxies the API to localhost:8000. The host API loads `.env` at startup; restart it after changing credentials.

PowerShell uses `py -3.12 -m venv .venv` and `.venv\Scripts\python.exe` instead of `python3.12` and `.venv/bin/python`. On macOS/Linux, use `python3` only if it reports Python 3.12. Docker is the alternative when host runtimes are unavailable.

Host integration test with PostgreSQL running:

```sh
RUN_DB_TESTS=1 .venv/bin/python -m pytest -q
```

PowerShell:

```powershell
$env:RUN_DB_TESTS = '1'
.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:RUN_DB_TESTS
```

Without this variable, database integration tests are intentionally skipped. Docker's test service sets it automatically. Run recovery only after stopping the previous API process; it marks its unfinished benchmark jobs interrupted.

## Build the dataset

With the backend running, the following works on macOS, Linux, and PowerShell:

```sh
docker compose exec backend python -m recipetriage_ml.data /app/ml/data/seed.json --output /tmp/dataset-exports --seed 42
docker compose cp backend:/tmp/dataset-exports ./dataset-exports
```

The CLI refuses to overwrite an existing version directory. The bundled initial export is already in `ml/datasets/`. To edit and save versions in PostgreSQL, use **Data Lab → Dataset Studio → Build preview → Save version**. A fresh database starts without saved snapshots. All seed labels are unreviewed proposals; exports clearly retain that status. Builds need no model or hosted API key. See DATASET_V1.md for host commands and the review workflow.

## Architecture and operations

`frontend/` is product UI, `backend/` is API serving and database integration, and `ml/` is independently testable learning/inference code. The ML package owns training logic; API background workers persist progress and keep start requests short. `infra/nginx.conf` serves compiled frontend files and proxies API calls. `tests/` contains backend and ML tests; React tests sit next to their components.

Pydantic validates input/output shape, dataset constraints, and the fixed benchmark. SQLAlchemy owns request-scoped database sessions. PostgreSQL stores immutable dataset snapshots in `dataset_versions` and benchmark progress/evidence in `benchmark_runs`, and training journals in `training_runs`; an idempotent migration runs at container startup. Completed benchmark evidence cannot be overwritten through the application. The recipe library has separate intake, recipe, prediction and review tables; model comparisons use the shared learning-job journal. Host API development runs migration, starter-library import and interrupted-job recovery explicitly. File-based evidence lives in `ml/experiments/` and `ml/evaluation/results/`.

Compose starts PostgreSQL before the API and the API before the frontend, using health checks. Non-root application containers expose host ports on loopback only. Internal database DNS is `postgres`; host development uses `localhost`. Configuration lives in `.env` (ignored by Git and Docker build context); `.env.example` documents the fields. Lock files record resolved dependencies. Image tags are not digest-pinned release artifacts.

Stop without deleting database or model-cache data:

```sh
docker compose down
```

**Destructive reset:** `docker compose down --volumes` deletes PostgreSQL data, cached models and saved training artifacts. Do not use it as the first response to a startup error. Changing PostgreSQL credentials in `.env` does not change users inside an already initialized database.

For port conflicts, change `FRONTEND_PORT`, `BACKEND_PORT`, or `POSTGRES_PORT` in `.env`. Container ports remain fixed. If a host-run API moves from port 8000, update the Vite proxy target too. If Docker cannot connect, start its daemon and inspect `docker info`. If image pulls report a macOS Keychain error, fix/unlock Docker's credential helper and retry. Network failures and model-load failures appear in logs; they are not passing checks.

This is a local learning deployment. Public deployment still needs a target-specific configuration, authentication, TLS, managed secrets, monitoring and resource limits. Local SFT creates a separate adapter; Playground compares available checkpoints without changing the production pointer. The managed SFT path uses temporary provider deployments only when its prerequisites pass. The bundled Fireworks preflight is blocked before resource creation.


## Model rollback

Rollback restores an immutable archived version that previously served in production. In **Deployment → Model Registry**, select that version, supply an actor and reason, and click **Rollback**. The server checks its files and current quality gates and atomically updates the single production pointer; failed checks leave the active model unchanged. Verify `/health` and a Playground request using **Registered production** afterward. No delivered candidate passes the defaults, so there is currently no production version to roll back.

The [complete rollback procedure](MODEL_REGISTRY.md#rollback-procedure) includes exact curl and PowerShell commands, required artifact backups and the stage-event audit. Never use `docker compose down --volumes` when you want to retain reviews, experiment history and model versions.
