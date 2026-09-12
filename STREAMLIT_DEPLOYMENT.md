# RecipeTriage on Streamlit Community Cloud

For the alternative you selected—**everything on one server**—use [SINGLE_SERVER.md](SINGLE_SERVER.md). That setup runs Streamlit in Docker and does not require Community Cloud or a separate public backend URL. The instructions below remain available for the earlier split-hosting option.

This is the full API-connected interface, not a standalone demo. `streamlit_app.py` exposes the seven workspaces; FastAPI retains inference, PostgreSQL persistence, human-review checks, training jobs and model promotion gates. The React interface remains available through the original Docker Compose setup.

## Reviewing data and following training

In **Data Lab → Synthetic Review**, quality errors appear above the draft. A blocked proposal offers Edit or Reject. For a misleading-title proposal, the category check requires a title starting with Quick, Easy or Simple and a known total time over 30 minutes. Keep recipe facts accurate and review the labels yourself; passing this check does not establish annotation quality. Choose Edit, enter your reviewer name, correct the draft JSON and record your decision. The page reloads the saved revision with fresh checks. Edits stay pending; approval requires a separate explicit decision. No records enter the improved dataset merely by being edited or generated.

In **Training Lab → SFT Monitor → Saved runs**, select a run and choose **Every 5 seconds** or **Refresh now**. The selected run shows its phase, recorded before/after benchmark cases, available loss history and raw evidence. This monitor also includes individual local LoRA runs. HTTP 409 stating that a training or benchmark worker is active means another run holds the shared worker. Wait for completion and inspect its result before submitting another job. **Start training creates a new run; it does not refresh an existing one.** Do not restart the backend to clear this guard while a real job is active.

## Deployment layout

```text
Browser → Streamlit Community Cloud (Python UI with workspace and page tabs)
        → HTTPS + private bearer token → FastAPI Docker host
                                      → PostgreSQL
                                      → persistent /data storage
                                      → local CPU models or Fireworks
```

Streamlit runs only `streamlit` and `httpx` plus their dependencies. It does not install PyTorch, run Docker Compose, host PostgreSQL, store model checkpoints, or connect to your Mac's localhost services. [Community Cloud deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Local preview

Keep the existing backend/PostgreSQL running. From the repository root:

```sh
python3.12 -m venv .venv-streamlit
.venv-streamlit/bin/python -m pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with this local-only configuration:

```toml
[backend]
url = "http://127.0.0.1:8000"

```

Then run:

```sh
.venv-streamlit/bin/python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://localhost:8501. The preview uses your existing database. Buttons that save, approve, generate, train or promote have real effects; opening pages only reads data. Stop the preview with Ctrl+C. On Windows use `py -3.12 -m venv .venv-streamlit` and `.venv-streamlit\Scripts\python.exe` for subsequent Python commands. Docker commands are the same in PowerShell.

## Backend hosting proposal

The host must run the `cloud` target in `backend/Dockerfile`, provide PostgreSQL, publish HTTPS and mount persistent storage at `/data`. The container listens on `$PORT` (8000 by default). Its public `/health` checks the database; all other routes, including API documentation, require `Authorization: Bearer <API_AUTH_TOKEN>`.

Required environment:

| Variable | Meaning |
| --- | --- |
| `APP_ENV=cloud` | Fail startup if API authentication is missing or too short |
| `API_AUTH_TOKEN` | Random token of at least 32 characters; shared only with the Streamlit server |
| `DATABASE_URL` | PostgreSQL connection URL; provider SSL query settings are preserved |
| `FIREWORKS_API_KEY` | Server-side Fireworks key, if hosted inference is used |
| `FIREWORKS_MODEL` | Accessible Fireworks model; example defaults are in the deployment proposal |
| `FIREWORKS_RECIPE_MODEL` | Model used for intake and comparisons |
| `FIREWORKS_SYNTHETIC_MODEL` | Model used for synthetic proposals and preference generation |

Managed SFT also requires separately configured `FIREWORKS_ACCOUNT_ID`, `FIREWORKS_SFT_MODEL` and `FIREWORKS_SFT_DEPLOYMENT_SHAPE`. An inference model is not automatically a supported fine-tuning model.

The optional [Render blueprint](infra/render.yaml) proposes a 4 CPU / 8 GB backend, a 1 GB PostgreSQL instance and a 30 GB persistent disk. At the pricing checked on 2026-09-12, the selected compute plans list approximately **$175/month for the backend and $19/month for PostgreSQL**, plus storage and any other applicable usage charges. This is a proposal, not an approved purchase. Review the current [Render pricing](https://render.com/pricing) and deployment estimate before creating resources. Resource sizing is an initial estimate; larger training settings need measurement. The UI's free hosting does not make backend compute or Fireworks free.

For Render, create a Blueprint from the approved GitHub repository, choose `infra/render.yaml`, and review all resources and costs. The blueprint creates an internal-only PostgreSQL connection, generates the API token and requests your Fireworks key as a secret. Copy the generated API token privately to Streamlit secrets. Automatic backend deploys are disabled so a push will not interrupt a running training job. This configuration follows the [Blueprint reference](https://render.com/docs/blueprint-spec); it still needs account-side validation before provisioning.

The Docker cloud target can also be deployed to an existing Linux server or another Docker host. It starts one API worker because model locks and the current job runner assume one process. It links `/training`, `/models` and `/deployment` to persistent subdirectories under `/data` and drops root privileges before migrations and API startup. Three small delivered SFT/LoRA/QLoRA reference adapters are installed once using atomic copies. Base weights are downloaded by the backend when first used; first inference can take longer than a warmed request. If a request times out, inspect saved jobs before submitting it again.

## Publish the Streamlit interface

1. Put the reviewed source into the GitHub repository you intend to deploy. This local project currently has no configured remote; repository selection and publishing must be completed before Community Cloud can read it. Exclude `.env`, `.streamlit/secrets.toml`, caches, virtual environments and database/artifact transfer archives. Do not blindly upload the entire local directory.
2. Sign in at https://share.streamlit.io/ and connect the appropriate GitHub account.
3. Choose **Create app → Deploy a public app from GitHub**. Select the repository, its actual branch, and **`streamlit_app.py`** as the entrypoint. Choose **Python 3.12** in Advanced settings. Root `requirements.txt` contains the UI dependencies. Your connected account lists `smohandoss0611/recipetriage`; the new local files still need to be published there before deployment.
4. In Streamlit Advanced settings, paste values using [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example):

   ```toml
   [backend]
   url = "https://YOUR-ACTUAL-BACKEND-HOST"
   token = "THE-BACKEND-API_AUTH_TOKEN"


   [links]
   mlflow = ""
   ```

5. Review app visibility before selecting **Deploy**. Streamlit inherits initial visibility from the GitHub repository; prefer a private app for this operator workspace. [Sharing controls](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).
6. Open the generated `.streamlit.app` URL, click **Check backend connection**, and use the tabs to verify the saved library and experiment pages. RecipeTriage has no additional app sign-in screen. No public URL is claimed until this succeeds.

Anyone who can reach the interface can use the shared recipes and administrative features. Use Community Cloud’s app-visibility controls to limit who can open a hosted workspace. Reviewer names remain explicit, self-reported entries. The backend API token stays server-side and is still required for cloud API requests. Existing `[access]` settings are ignored; no workspace password or local-login bypass is required.

## Data and model lineage during migration

A new hosted database starts with the seven original unreviewed seed recipes. The current eight-recipe library, pending reviews, preferences, run records and model registry are not automatically uploaded. Decide whether to migrate those records before opening the deployment to users.

For a complete migration, use a database backup and restore into an **empty destination database before starting the API**, then transfer the matching `/training` and `/deployment` artifacts to `/data/training` and `/data/deployment`. Retain all IDs, review revisions, hashes and checkpoint files together. Back up a live local database without modifying it:

```sh
mkdir -p cloud-transfer
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner --no-acl -f /tmp/recipetriage.dump'
docker compose cp postgres:/tmp/recipetriage.dump cloud-transfer/recipetriage.dump
```

PowerShell uses `New-Item -ItemType Directory -Force cloud-transfer` for the first command. `docker compose cp` avoids shell-specific binary-redirection issues. The dump contains application data; keep `cloud-transfer` private and ignored. Restore with the destination provider's supported PostgreSQL 17 tooling and private connection. Do not restore over an initialized or populated destination. If the host already started, create a fresh empty target database or review a data-only migration plan first. No destructive restore is performed by this package.

For a fresh deployment that only needs bundled historical learning evidence, run these in the backend host's shell after migrations:

```sh
python -m app.experiments /app/ml/training/results/experiments-v1/experiment.json
python -m app.lora /app/ml/training/results/lora-v1/experiment.json
```

These are imports of historical results, not new training. Large merged/quantized exports remain excluded from Git and Docker builds; restore them separately before registering or promoting their corresponding candidates. The deployment measurements UI is historical evidence and does not assert that those large artifacts exist on the new host. DPO still needs real human A/B preferences, and reviewed QLoRA still needs approved data. Deployment does not waive either gate.

MLflow on your Mac at port 5050 is not available to the cloud API. Leave `MLFLOW_TRACKING_URI` blank to retain local tracking files on the backend's persistent disk, or configure a separately hosted authenticated MLflow server and an HTTPS dashboard link. A second public MLflow server is not created by this proposal.

## Tests and operating limits

```sh
.venv-streamlit/bin/python -m pip install pytest
.venv-streamlit/bin/python -m pytest -q tests/streamlit
docker compose --profile test run --build --rm backend-tests
```

The GitHub Actions workflow includes a separate Streamlit test job. UI tests verify navigation without implicit writes, explicit preference choices, direct app access, backend credential requirements, draft retention across tabs and visible API errors. Backend tests cover authentication and cloud database URLs. Do not use live human-review or promotion buttons as smoke tests.

Training still uses the existing single-process background runner, not a durable distributed queue. Deploys or crashes can interrupt jobs; saved records identify interrupted work, but jobs do not automatically resume. Keep one backend instance, avoid deployments during training, and measure resource use before larger runs. This deployment preparation does not establish production reliability or model quality.

## Feature locations

| Workspace | Streamlit pages and behavior |
| --- | --- |
| Recipes | Four intake sources, normalization review, library search/labels, triage, revision edits, explicit correction, history, verified export |
| Data Lab | Dataset Studio, draft Label Editor, JSONL Preview, Chat Template Preview, Synthetic Review |
| Training Lab | SFT preflight/start/resume, LoRA rank/layer configuration, QLoRA comparison, experiment matrix, saved results and loss curves |
| Alignment Lab | Two-candidate preference choices, DPO, GRPO reward components, approved-data QLoRA, saved jobs |
| Evaluation Lab | Fixed baseline, per-run evidence, failure analysis, collection priorities, shortcut/red-team results |
| Playground | Recipe inference, model comparison and token inspection |
| Deployment | Size/memory/latency/quality comparison, registry gates, promotion, archival and rollback |

Advanced training configuration remains editable JSON, with backend schema validation. Merge/quantization scripts and advanced analysis notebooks remain host-side tools as in the original project; see DEPLOYMENT.md and ADVANCED_ANALYSIS.md.
