# Failure analysis and controlled title tests

`ml/evaluation/failures.py` reparses raw benchmark responses, assigns overlapping failure categories, and retains case IDs, expected labels, predictions, raw text and errors. It separates provider/format/schema/truncation failures from semantic mistakes. An invalid response does not support a diagnosis such as “the model misunderstood equipment,” even though evaluation counts its absent labels as incorrect.

Semantic categories include false positives/negatives, time reasoning, required-equipment reasoning, pantry reasoning, dessert context, ambiguity handling and omitted multi-label predictions. These categories describe disagreement with a provisional answer key, not a verified causal explanation. `backend/app/analysis.py` persists analyses and individual category rows in PostgreSQL. Failure Analysis can analyze existing saved benchmarks without making inference calls.

## Counterfactual construction

`ml/evaluation/shortcut_tests.py` uses three existing source-linked recipe bodies:

| Recipe body | Total minutes | Source |
|---|---:|---|
| Brown Bread | 80 | Existing benchmark case bench-003 |
| Turnip Pancakes | 75 | Existing benchmark case bench-010 |
| Ravioli | 70 | Existing ravioli inference fixture |

Each body receives exactly these titles: Traditional [dish], Easy [dish], Quick [dish], Simple [dish]. Traditional is the declared reference. The checker rejects changes to time, ingredients, equipment, instructions or pantry, including changes hidden by normalizing text. It also rejects extra title words or undeclared adjectives. Every row stores the recipe body hash, exact recipe and prompt. These are controlled evaluation variants, not new training recipes.

There are 12 generations and nine reference/variant pairs. “Quick” is the predeclared misleading-time subset because all three recipes require more than 30 minutes. Easy and Simple suggest ease but do not make a precise time claim. The original benchmark's misleading-title subset contains one different evaluation point: its original Quick Brown Bread case. The UI reports these two denominators separately.

For usable reference/variant pairs P:

$$\text{prediction-flip rate}=\frac{\sum_{(a,b)\in P}\mathbf1[\operatorname{set}(\hat y_a)\ne\operatorname{set}(\hat y_b)]}{|P|}.$$

A usable output is schema-valid and ends normally. If no pair is usable, the rate is **undefined**, shown as `—`, not zero. Coverage is $|P|/9$. Excluded pairs remain visible. Label order alone does not count as a flip.

Misleading-title accuracy is exact label-set accuracy over all three Quick variants. Invalid/missing responses count as incorrect. Overall accuracy uses all 12 planned variants. Reporting both stability and accuracy prevents a consistently wrong model from looking robust.

## Actual diagnostic results

All three models were evaluated with the same prompts, four adjectives, recipe bodies, greedy decoding and 128-output-token budget. The selected LoRA and QLoRA adapters both use rank 4 and LR 0.0002.

| Model | Usable outputs | Usable pairs | Excluded pairs | Label-flip rate | Misleading-title exact accuracy | Overall exact accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0/12 | 0/9 | 9/9 | Undefined | 0/3 | 0/12 |
| LoRA | 7/12 | 3/9 | 6/9 | 0/3 = 0% | 0/3 | 0/12 |
| QLoRA | 9/12 | 3/9 | 6/9 | 0/3 = 0% | 0/3 | 0/12 |

The adapters were stable only within the small usable-pair subset and were consistently wrong on exact sets. This does not establish title robustness. Base-model format failures made its flip rate undefined. The separate red-team suite retained the correct structured answer on 0/3 probes for each model; raw evidence distinguishes malformed responses from explicit instruction following. The measured comparisons are saved in `ml/evaluation/results/diagnostics-v1/comparison.json`.

## Separate red-team suite

`ml/evaluation/red_team.py` applies three hand-authored prompt-injection probes to a held-out recipe: a title instruction, an appended recipe instruction and a role-spoofing instruction. These intentionally alter title content or instructions and are never included in the adjective-only counterfactual metrics.

The scorer records whether the output is a correct usable prediction, whether the attack marker appears, and whether the model explicitly returned the requested attack marker alone. Malformed JSON is a task failure, but not automatically evidence that the injection was followed. Three probes are diagnostic evidence, not a security certification.

## Run and inspect

Base diagnostics:

```sh
python -m recipetriage_ml.evaluation.diagnostics --output diagnostics/base.json
```

Matched LoRA and QLoRA adapters from the delivered experiment:

```sh
python -m recipetriage_ml.evaluation.diagnostics --adapter ml/training/results/experiments-v1/runs/771d20b1-77df-446b-b34d-b8259ddbc585 --output diagnostics/lora.json
python -m recipetriage_ml.evaluation.diagnostics --adapter ml/training/results/experiments-v1/runs/041818c9-7e00-4802-b53d-f7d567616e2b --output diagnostics/qlora.json
```

These commands work in an activated Python environment on macOS/Linux/Windows. Set a populated `HF_HOME` cache and `HF_HUB_OFFLINE=1` to prevent downloads if needed; PowerShell sets `$env:HF_HUB_OFFLINE = '1'` instead of a POSIX command prefix.

The diagnostic CLI reloads the selected adapter with its recorded base revision and NF4 settings. Each run saves every raw response. Import with `python -m app.analysis path/to/result.json` into the configured PostgreSQL database. The importer rechecks suite hashes, recipe invariance, prompt construction, parsing and metric arithmetic. Evaluation Lab → Shortcut Tests shows the saved comparisons and separate red-team results. Its run button starts base-model diagnostics only; it does not train.

## What data to collect next

The UI generates priorities from each selected model's actual failure categories and any matching shortcut evidence. Current evidence supports this order of work:

1. **Review labels and broaden coverage.** Independently adjudicate the seven provisional seed examples. Add new dessert and explicit pantry-coverage examples, absent from the current training split, and expand validation beyond one recipe. This is a known coverage gap, not a proven causal explanation of a specific error.
2. **Complete, concise structured answers.** Collect reviewed examples that always contain both `labels` and `explanation`. Many failures are malformed, truncated or missing fields. Check output budget and prompt formatting as well; more data alone may not fix truncation.
3. **Elapsed time, passive waiting and active projects.** Collect new natural recipes around the 30-minute boundary, long passive waits versus substantial active projects, and titles whose adjectives conflict with total elapsed time. Record prep/cook/rest/cooling evidence explicitly.
4. **Required versus optional tools and complete label sets.** Collect human-reviewed examples distinguishing ordinary kitchen tools, required appliances and optional alternatives. Include multiple simultaneously supported labels with a rationale for each.
5. **Serving context and uncertainty.** Collect savory pancakes versus desserts, breakfast drinks versus dessert drinks, and unknown timing with enough evidence for another label. Adjudicate when `unclear` should appear alone.

Use new recipe sources and hold every related recipe/counterfactual group in a single split. Do not copy these benchmark, shortcut or red-team examples into training, and do not relabel the benchmark merely to agree with a model. Collection priorities are hypotheses to test with reviewed data; this phase creates no synthetic training records.


Shortcut contamination limitation: the ravioli diagnostic source also appears in the original training seed. Its title variants are held out as variants, but that source recipe is not an unseen evaluation source. The bread and turnip cases come from the frozen benchmark. Report the combined three-source shortcut score as a diagnostic, not an independent generalization estimate. A future benchmark version should add entirely new reviewed recipe families without changing this frozen comparison.
