# Model versions, promotion and rollback

A **model version** is an immutable combination of base revision, adapter/export checksum, tokenizer and prompt format. Changing any of these can change predictions. A filename such as best-model is not enough to reproduce a result.

The **model registry** is a PostgreSQL catalog linking that identity to training configuration, dataset lineage, immutable evaluation rows and file checksums. A dataset lineage records which reviewed content version and preference decisions produced the weights. An experiment registry answers “what did we try?”; the model registry answers “what exact artifact is eligible to serve?”

Stages have distinct meanings: **candidate** is a recorded result; **staging** has passed evaluation for pre-production inspection; **production** is the single version selected by the serving pointer; **archived** is inactive but retained. A transaction and a unique PostgreSQL index enforce at most one production pointer. Replacing production archives the previous version and records actor, reason, evidence hash, previous pointer and gate result. Nothing in training automatically promotes a model.

A **quality gate** is an executable rule, not a favorable summary written by the model. The server recomputes metrics from all 10 fixed benchmark responses and all 12 title variants, checks model identities, inputs, answer keys and hashes, and fails closed on missing or changed evidence. It then applies:

| Metric | Default threshold |
| --- | --- |
| Macro F1 | at least 0.45 |
| JSON validity | at least 0.95 |
| Schema validity | at least 0.95 |
| Misleading-title exact-label accuracy | at least 0.67 |
| Usable counterfactual pair coverage | 1.0 |
| Prediction-flip rate | at most 0.0 |
| Important-label F1 regression | at most 0.05 absolute per label |

All seven labels are important by default. With production present, the nonregression reference is the current production benchmark; initially it is the frozen base benchmark. On a three-example misleading-title subset, 2/3 is below 0.67, so the default requires 3/3. Invalid outputs count as wrong, and invalid-to-invalid does not count as robustness. These thresholds are deliberately stricter than the operational comparison table. None of the delivered local candidates passes them.

Configure thresholds in `.env` with `MODEL_GATE_JSON`, for example a complete JSON object copied from `GET /api/v1/models/gate-config`. Blank uses defaults. Recreate the backend after edits. Do not lower gates simply to make a weak candidate pass. Repeatedly choosing parameters on the same tiny benchmark risks benchmark overfitting; obtain a separately reviewed untouched test set before real deployment decisions.

## Promotion

Open Deployment → Model Registry. Inspect the lineage, raw evidence and failed gates. Enter an actor and a reason. A candidate must first pass **Stage** and then **Promote**. The server validates again even if a client bypasses a disabled UI control. File checksums are revalidated so a changed checkpoint cannot inherit old scores.

The production serving path is `POST /api/v1/triage` with `provider: "production"`. Playground offers **Registered production**. Every request resolves the current pointer; a small cache reuses only the immutable selected model. If none is promoted, the API returns 503 with an explicit message. Existing local/Fireworks provider choices remain separate experiments.

## Rollback procedure

1. Inspect `GET /api/v1/models/history` and select the archived version that served successfully before the current version.
2. Restore its original artifacts at the registered path if necessary. Do not replace bytes under an existing version ID.
3. Open Deployment → Model Registry and use **Rollback** with an actor and reason. The destination must have previously been production and must satisfy current gates, including nonregression against current production. A failed check leaves the pointer unchanged.
4. Check `/health`, request triage with `provider: "production"`, and verify the response's model revision. Record operational observations. The next request resolves the restored pointer; no retraining is needed.

Equivalent API commands (macOS/Linux; replace the example UUID with an actual archived production version):

```sh
curl --fail http://localhost:8000/api/v1/models/history
curl --fail -X POST http://localhost:8000/api/v1/models/00000000-0000-0000-0000-000000000000/rollback \
  -H 'Content-Type: application/json' \
  --data '{"stage":"production","actor":"your-name","reason":"Restore the previous verified release"}'
```

PowerShell avoids shell quoting differences:

```powershell
$body = @{ stage = 'production'; actor = 'your-name'; reason = 'Restore the previous verified release' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/api/v1/models/00000000-0000-0000-0000-000000000000/rollback' -ContentType 'application/json' -Body $body
```

This procedure does not bypass evaluation for an emergency rollback. Preserve both PostgreSQL and artifact-volume backups; a database pointer alone cannot restore lost weights. `docker compose down` preserves named volumes; `down --volumes` destroys them and is only appropriate for disposable test environments.

## Implementation and deployment scope

`backend/app/migrate.py` creates model_registry, model_stage_events and the single-production index. `registry.py` registers immutable evidence and performs transactional stage changes. `ml/deployment/gates.py` recomputes quality checks. `backend/app/serving.py` resolves the production version. `frontend/src/DeploymentStudio.jsx` displays models, constraints, checks and rollback. The GitHub Actions workflow runs unit, integration and UI smoke tests on push/PR; no remote workflow has been run or GitHub push made in this delivery.

The app is bound to localhost and uses reviewer/actor names as audit attribution. It does not authenticate separate human identities. Add authentication, authorization, backup automation, resource isolation and operational monitoring before exposing promotion/review endpoints to other users or the public internet. The registry is implemented and tested; the supplied toy models are not production-quality models.

Check your understanding: What changes a model version? Why keep a staged model separate from production? Why can zero prediction flips hide a bad model? Which files and database records are required for rollback? Why can perfect training performance still fail the evaluation gate?


Shortcut contamination limitation: the ravioli diagnostic source also appears in the original training seed. Its title variants are held out as variants, but that source recipe is not an unseen evaluation source. The bread and turnip cases come from the frozen benchmark. Report the combined three-source shortcut score as a diagnostic, not an independent generalization estimate. A future benchmark version should add entirely new reviewed recipe families without changing this frozen comparison.
