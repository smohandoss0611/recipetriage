# RecipeTriage on free Streamlit hosting

Use **`cloud/streamlit_app.py`** to run the Streamlit UI and RecipeTriage's application services together. No separate FastAPI host, backend URL, Docker server, API token or running Mac is required for this profile. A hosted PostgreSQL database is still required for durable records.

If you already deployed **`streamlit_app.py`**, keep that app and URL: the same `[app] mode = "embedded"` secrets below enable this profile after the updated source is pushed. Both entrypoints install the same lightweight dependencies.

This profile makes no Fireworks calls and starts no model-training or inference workers. It does not silently replace model predictions with rules or fabricated outputs. The original full React/FastAPI and single-server deployments remain available for those operations.

## Deploy

1. Publish this source to `smohandoss0611/recipetriage` on the branch you select in Streamlit. Local changes must be pushed before the cloud can read them.
2. Create a **Free** PostgreSQL project at [Neon](https://console.neon.tech/signup), or use an existing hosted PostgreSQL database. Use a new empty database for a fresh workspace. Do not upgrade the plan or select paid resources. In Neon, choose PostgreSQL 17 and copy the **direct** connection URL from the Connect dialog; retain `sslmode=require` and any `channel_binding` parameter. Direct connections avoid transaction-pooler restrictions on migration settings.
3. Open [Streamlit deployment](https://share.streamlit.io/deploy). Select:

   | Field | Value |
   | --- | --- |
   | Repository | `smohandoss0611/recipetriage` |
   | Branch | The branch containing these changes |
   | Main file path | `cloud/streamlit_app.py` |
   | Python | `3.12` |

4. In **Advanced settings → Secrets**, paste the following with your actual private database URL:

   ```toml
   [app]
   mode = "embedded"

   [database]
   url = "postgresql://USER:PASSWORD@YOUR-POSTGRES-HOST/DBNAME?sslmode=require"
   ```

   Keep this URL out of Git and chat. No `[backend]` section or Fireworks key is needed. [Template](cloud/secrets.toml.example).
5. Select **Deploy**. Streamlit installs `cloud/requirements.txt`, the dependency file next to this entrypoint. It includes the existing Python application packages, PostgreSQL driver and tokenizer dependencies, without PyTorch, TRL or PEFT.
6. Check **Check backend connection**, then **Recipes → Library**. A new database receives the seven existing unreviewed seed recipes. No human decisions, new model results or local database records are uploaded automatically.
7. In **Data Lab → Dataset Studio**, preview and save a dataset version. Reload the app and load that saved version. Download its complete dataset ZIP for local training or backup.

There is no RecipeTriage sign-in page. Everyone able to open this workspace shares its data and can record reviews; use Streamlit's app visibility controls for a personal workspace. Reviewer names are self-reported, not authenticated identities.

## What works here

| Workspace | Available in the free profile | Requires the existing local deployment |
| --- | --- | --- |
| Recipes | Structured entry, validated drafts, library search, editing, explicit label correction, history, verified dataset export | URL/screenshot/message extraction by a model; model triage |
| Data Lab | Validation/cleaning, splits, saved versions, JSONL/ZIP downloads, tokenizer chat formatting, existing-candidate review, original seed review | Fireworks synthetic generation |
| Training Lab | Saved runs/loss history, architecture estimates, local configuration download | SFT, LoRA, QLoRA, experiment matrices, resume |
| Alignment Lab | Existing preference pairs, explicit A/B/Tie/Neither choices, saved results, reward definitions | New candidate generation, DPO/GRPO, reviewed QLoRA |
| Evaluation Lab | Saved benchmark scores, failure analysis of saved results, collection priorities | New model benchmarks and shortcut/red-team runs |
| Playground | Real token inspection and chat templates, saved comparisons | Model inference and new comparisons |
| Deployment | Historical size/latency/quality comparisons, registry and gate inspection | Artifact verification, promotion, rollback, model serving |

The tokenizer downloads only its pinned tokenizer files on first use; no model weights are loaded. Files cached locally can disappear when Streamlit restarts. Recipes, reviews, preferences and dataset snapshots are stored in PostgreSQL. Browser-session drafts remain temporary until saved.

An empty cloud database has no locally trained run history or generated preference pairs. This deployment does not migrate your Mac's database or checkpoint files. Historical deployment measurements bundled in source are labeled as historical, not newly measured cloud performance.

## Cost and operating limits

[Streamlit Community Cloud](https://streamlit.io/cloud) is free. [Neon's Free plan](https://neon.com/pricing), checked September 12, 2026, includes 0.5 GB of database storage and 100 CU-hours per project each month, with no credit card required. Stay on Free and monitor its limits; this does not promise unlimited storage or uninterrupted service. Other providers have their own limits.

The app closes database connections after requests and uses manual history refresh so an idle Neon database can suspend. Waking a suspended database can make the first action slower. There is no heartbeat or background polling to keep either service awake. Streamlit can hibernate inactive apps and has resource limits; this is a learning deployment, not an always-on training server. [Streamlit limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Local check of this exact profile

From the repository root on macOS/Linux:

```sh
python3.12 -m venv .venv-cloud
.venv-cloud/bin/python -m pip install -r cloud/requirements.txt
.venv-cloud/bin/python -m pip check
```

Add the `[app]` and `[database]` sections above to your local ignored `.streamlit/secrets.toml`. Preserve any existing `[backend]` settings for the other mode. For local PostgreSQL only, a `localhost` connection does not require SSL. Cloud hosting must use a public database hostname with SSL.

```sh
.venv-cloud/bin/python -m streamlit run cloud/streamlit_app.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://localhost:8502`. Stop with Ctrl+C. On Windows use `py -3.12 -m venv .venv-cloud`, then `.venv-cloud\Scripts\python.exe` instead of `.venv-cloud/bin/python`. Keep the working directory at the repository root for dependency installation.

To return to the existing API-connected UI, remove `[app].mode` or set it to `"remote"`, retain `[backend]`, and run the root `streamlit_app.py`. The single-server Docker environment continues to select its private backend automatically.

## Train locally with the exported dataset

1. Download the complete dataset ZIP from Data Lab and extract it into `dataset-exports/cloud-version` in your local repository. Keep `metadata.json` together with all exported files.
2. Download `training-config.json` from **Training Lab → Export a configuration for local training**. Choose the same saved dataset version. Review every training parameter before running it.
3. Use the existing ML training environment (the lightweight `.venv-cloud` deliberately lacks training dependencies):

   ```sh
   python -m recipetriage_ml.training.sft --config training-config.json --dataset dataset-exports/cloud-version --output ml/training/runs
   ```

   `python` here must be the interpreter in your configured ML environment; see [SFT.md](SFT.md) for installation. Do not run this command on Streamlit Community Cloud. This command performs real local training and automatically benchmarks afterward. It does not submit a Fireworks training job when `provider` is `local`.

Local results are not automatically synchronized into a different cloud database. Keep dataset version IDs, explicit reviews and benchmark provenance intact when planning a later transfer.

## Technical files

- `cloud/streamlit_app.py`: separate cloud entrypoint, defaulting to embedded mode.
- `cloud/requirements.txt`: lightweight deployment dependencies; the full training locks remain separate.
- `streamlit_ui/embedded.py`: calls the application through HTTPX's in-process ASGI transport. This opens no network port and starts no Uvicorn service. Connections are per request; one initialized runtime is cached per database configuration.
- `backend/app/embedded.py`: selects the existing read/curation handlers, adds provider-free structured intake, initializes PostgreSQL schema and seeds, and omits compute/provider/promotion endpoints. Hidden or disabled buttons are not the enforcement boundary: those operations also cannot be called through this runtime.
- `streamlit_ui/app.py`: retains all workspace tabs, disables unsupported submissions, and exposes downloads for local training.
- `tests/cloud/`: checks the real service contracts, no-provider/no-training boundaries, PostgreSQL persistence and all cloud UI pages. GitHub Actions uses a disposable PostgreSQL service.

The existing FastAPI validators, session handling, immutable dataset exports and human-review checks are reused rather than reimplemented. [HTTPX transport documentation](https://www.python-httpx.org/advanced/transports/).
