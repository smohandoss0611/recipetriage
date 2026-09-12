# Human review and isolated alignment

## What is implemented

Data Lab → Synthetic Review contains Fireworks proposals, automated checks, original response/provenance, editable drafts, and explicit Approve/Reject actions. Editing creates a new revision and returns the record to pending. Approval binds a reviewer, timestamp and content hash to that exact revision. An already approved record is immutable. Generated JSON is never appended to train.jsonl.

Six difficult proposals were generated with `accounts/fireworks/models/qwen3p7-plus`: four misleading-title and two ambiguous cases. Seven original seed recipes also require explicit review before augmentation. Automated checks verify structure, label policy, time/pantry consistency and exact/lexical overlap with existing data and evaluation cases. These checks cannot certify factual plausibility or semantic correctness. Review the instructions, timing assumptions, labels and rationale yourself.

Click **Build approved dataset version** after reviewing all seven original recipes and approving at least one proposal. The server rebuilds the version solely from stored approvals. Original validation/test recipes, labels and rationales remain frozen; additions enter training only after group-overlap checks. If a holdout annotation needs correction, create a separately reviewed evaluation protocol rather than silently changing this comparison. The original Dataset v1 remains unchanged.

The reviewed QLoRA action reuses the only previously tested QLoRA configuration (rank 4, alpha 8, LR 0.0002, 3 epochs, batch 1, accumulation 2, sequence length 768, seed 42, query/value targets in all 24 blocks, NF4 double quantization, float32 CPU compute). “Best” means best of the configurations actually tested; only one QLoRA configuration has been measured. It starts from the same base, then automatically runs the unchanged benchmark and title-only suite, recording before/after correctness, uncertainty, JSON validity, flip rate and coverage. No reviewed-data retraining has been performed while approvals are pending.

## Preferences and DPO

Alignment Lab → Preference Pairs displays a recipe and raw responses A and B, along with quality diagnostics and provenance. Five real pairs from Fireworks and the current QLoRA checkpoint are available. Presentation order is randomized independently of quality. Enter your reviewer name and choose Prefer A, Prefer B, Tie or Neither. Opening a page or a successful JSON check never records a choice. Decisions are immutable and bound to the pair hash; create a new pair for a new judgment.

Only explicit A/B choices with nonempty, different and complete responses can enter DPO. Tie/Neither are retained as human evidence but excluded from the preference objective. At least three independent recipe groups are required; one group becomes validation. Only original train recipes are eligible. A benchmark recipe or a changed holdout cannot enter through this endpoint. Sequence overflows fail explicitly rather than silently truncating.

For a chosen response y+ and rejected response y−, DPO minimizes:

    L = −log sigmoid(β [(log πθ(y+|x) − log πref(y+|x))
                         − (log πθ(y−|x) − log πref(y−|x))])

The current QLoRA SFT adapter initializes the trainable policy. A second copy is frozen as the reference, and its weights are checked after training. The default TRL sigmoid objective uses beta 0.1, learning rate 0.000005, 2 optimizer steps (allowed 1–10), batch size 1, gradient accumulation 2, maximum sequence length 1024 and seed 42. AdamW uses zero weight decay, constant LR, no warmup and gradient clipping at norm 1. Gradient checkpointing is enabled with non-reentrant recomputation. CPU computation is float32 over a frozen NF4 base. No fp16/bf16 or GPU speedup is implied. Validation runs before and after; adapters and immutable run/evaluation evidence are saved. The benchmark and shortcut suite run automatically.

DPO has an actual two-step random tiny-model mechanics test proving policy updates and reference freezing. This fixture is not a RecipeTriage training result. Real DPO remains pending explicit usable human preferences; do not infer them from an automated score or a general “yes.”

## Transparent GRPO probe

The small TRL GRPO path stays separate from production. The measurable reward is:

    R = 0.55 F1(labels) + 0.10 valid_JSON + 0.10 complete_schema
        + 0.15 correct_uncertainty + 0.10 correct_title_consistency

Label F1 uses the declared recipe labels. Uncertainty rewards `unclear` exactly when the provisional target calls for it; arbitrary hedging is not rewarded. The title-consistency term requires both the original and counterfactual responses to be correct, complete and schema-valid. Consistently wrong answers earn no robustness bonus. Incomplete originals cannot earn label, schema, uncertainty or consistency rewards; the JSON syntax component is scored separately and truncated completions are masked from policy updates. The counterfactual changes only Traditional to Quick and holds ingredients, equipment, time, pantry and instructions constant. It uses the same current weights and a greedy response. These label-based rewards are fallible proxies, not proof of human preference.

For two sampled completions, group-relative advantage is approximately:

    Ai = (Ri − mean(Rgroup)) / (std(Rgroup) + epsilon)

TRL applies a clipped policy update using these advantages. Identical rewards provide no ranking signal. This isolated probe uses group size 2, temperature 0.8, 128 completion tokens, top-p 1, 2 optimizer steps, the same 5 seed training recipes, learning rate 0.000005 and the shared optimizer settings above. GRPO beta is explicitly 0 (no KL penalty); the shared configuration's beta field is for DPO only. Truncated completions are masked. Benchmark recipes never supply training rewards.

Actual result: 97.91 seconds of training, about 1273.09 MiB sampled peak process RSS. All four sampled outputs reached the token limit and had zero reward. Reported loss and gradient norms were zero. Benchmark Micro-F1 stayed 23.53%, and misleading-title accuracy stayed 0/3. There is no measured gain to justify GRPO or production promotion. The GRPO page shows reward components, raw outputs and the unchanged behavior comparison.

## Run commands

From the repository root with Docker running, import delivered generation evidence without making new model calls:

```sh
docker compose exec backend python -m app.curation /app/ml/data/results/synthetic-v1/fireworks-proposals.json
docker compose exec backend python -m app.alignment /app/ml/alignment/results/preferences-v1/pairs.json
```

Imports accept pending records only and are idempotent. Use Synthetic Review → Add original seed to review if not already present. For new hosted generation, use the UI button; it makes bounded Fireworks requests using `FIREWORKS_SYNTHETIC_MODEL` and the server-only API key. Historical connection/404 failures are retained in the results directory.

Install the delivered reference checkpoint into the persistent training volume once:

```sh
docker compose exec backend mkdir -p /training/references/qlora-v1
docker compose cp ml/training/results/experiments-v1/runs/041818c9-7e00-4802-b53d-f7d567616e2b/. backend:/training/references/qlora-v1/
docker compose exec --user root backend chown -R appuser:appuser /training/references/qlora-v1
```

These commands also work in PowerShell. The copy must contain adapter/, adapter-provenance.json, run.json, benchmark.json and baseline.json. Source archives include the small adapters; optimizer checkpoints and large full-model exports are omitted. The backend image excludes adapter weights so that model versions stay explicit runtime artifacts.

For a new CLI DPO run, download explicit choices from the UI, then use an installed host environment:

```sh
.venv/bin/python -m recipetriage_ml.alignment.dpo --pairs preference-pairs.json --output ml/alignment/runs
.venv/bin/python -m recipetriage_ml.alignment.grpo --output ml/alignment/runs
```

Both CLIs accept `--config path.json` and `--source path/to/reference-run`. Host model caching can be set with `HF_HOME`; see LEARNING.md. PowerShell uses `.venv\Scripts\python.exe`. UI jobs persist in PostgreSQL; CLI runs persist JSON journals on disk and can be imported with `python -m app.alignment --run DIRECTORY`. The two-step defaults are teaching probes, not validated training schedules.

## Major files

| File | Responsibility |
| --- | --- |
| ml/data/synthetic_fireworks.py, data/prompts/synthetic-v1.txt | Hosted proposal generation, bounded requests, checks and original provenance |
| ml/data/approved.py, data/schemas.py | Content-bound approval and frozen-holdout dataset construction |
| backend/app/curation.py | Transactional review queue and approved version endpoint |
| ml/alignment/preferences.py | Candidate/choice schemas, input checks and preference splits |
| ml/alignment/dpo.py, grpo.py, runner.py | CLI entry points and the isolated TRL training/evaluation sequence |
| ml/alignment/rewards.py, evaluation/behaviors.py | Reward components and actual behavior comparisons |
| backend/app/alignment.py, jobs.py | Human choice endpoints, persistent job status and training dispatch |
| frontend/src/ReviewQueue.jsx, AlignmentLab.jsx | Review, Preference Pair, DPO, GRPO and reviewed QLoRA interfaces |

References: [TRL DPO 0.24](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer), [TRL GRPO 0.24](https://huggingface.co/docs/trl/v0.24.0/en/grpo_trainer), [Fireworks structured output](https://docs.fireworks.ai/structured-responses/structured-response-formatting).


Shortcut contamination limitation: the ravioli diagnostic source also appears in the original training seed. Its title variants are held out as variants, but that source recipe is not an unseen evaluation source. The bread and turnip cases come from the frozen benchmark. Report the combined three-source shortcut score as a diagnostic, not an independent generalization estimate. A future benchmark version should add entirely new reviewed recipe families without changing this frozen comparison.
