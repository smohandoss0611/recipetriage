# Features to verify

Start at http://localhost:8080. On a narrow window use the **Workspace** dropdown; on desktop use the left navigation.

| Page | Action | Expected result |
| --- | --- | --- |
| Recipes | Search `pasta machine`, then filter by label | Matching saved recipes, preparation details and unreviewed label status |
| Recipes → Add recipe | Normalize text/message/URL/screenshot and inspect/edit Recipe JSON | Source and model provenance; unknown evidence preserved; explicit Save recipe required |
| Recipes → Run triage | Select an available model and run | Raw/structured prediction appears asynchronously; no automatic review |
| Recipes → Review or correct labels | Inspect recipe, enter your name and rationale, then save chosen labels | Revision-bound verified example; editing recipe afterward invalidates this review |
| Recipes | Create dataset from verified recipes | Disabled without enough reviewed groups; only current reviews enter the version |
| Recipes → Open in Playground | Open a selected recipe | Same ingredients/time/equipment/instructions; no answer labels in the input |
| Playground → Compare models | Select 2–6 models; compare; reopen Saved comparison | Independent raw/structured results, preparation and generation timing; DPO unavailable until trained |
| Data Lab → Synthetic Review | Open the queue | 13 pending records, no automatic approvals |
| Synthetic Review | Inspect a draft, source/model/prompt, labels and rationale | Generated content is clearly marked; automated checks are not human approval |
| Synthetic Review | Enter reviewer, edit a record and save | Revision increases and remains pending; approve the saved revision separately |
| Synthetic Review | Explicitly approve/reject after inspecting each record | Decision and content hash persist; pending/rejected items cannot enter the improved dataset |
| Synthetic Review | Build next version after all seven originals and at least one new proposal are approved | New immutable dataset; original validation/test membership and labels stay fixed |
| Alignment Lab → Preference Pairs | Read the recipe and both responses; choose A/B/Tie/Neither explicitly | Choice is recorded only after your action; ties/neither are excluded from DPO |
| Alignment Lab → DPO Training | Check prerequisites | Run button is disabled without at least three A/B choices; server also validates completion and independent groups |
| Alignment Lab → GRPO Experiment | Inspect saved result and reward components | Measured zero reward and unchanged F1/shortcut accuracy are visible |
| Alignment Lab → Reviewed QLoRA | Select an approved augmented version | Same previously tested QLoRA configuration; automatic benchmark and shortcut comparison after training |
| Deployment → Model Registry | Inspect each candidate | Five candidate records, failed quality gates, no production model |
| Deployment → Deployment Comparison | Compare sizes, RAM, latency and F1 | Base recommended under defaults; NF4 is smaller but fails quality and latency limits |
| Deployment Comparison | Set p95 limit to 1000 and compare | No candidate meets every limit; no training or promotion occurs |
| Playground → Single model → Registered production | Send a recipe before any model passes promotion | Explicit 503: no model has passed the production gate |
| Notebook 11 | Run saved-results cells | Parameter/time/benchmark table and before/after layer plot, with causal limitations |

Opening pages never starts paid generation or training. Actual human decisions cannot be supplied by the automated test suite. README and MODEL_REGISTRY.md contain the rollback procedure; existing toy candidates cannot be promoted by weakening the UI alone.

See [USE_CASES.md](USE_CASES.md) for all 15 acceptance cases, adapter installation, exact limits and platform alternatives.
