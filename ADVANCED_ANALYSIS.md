# Layer diagnostics

Open `ml/notebooks/11_layer_adaptation_diagnostics.ipynb`. It reads saved measurements by default and has been executed with those results. The optional training cell is off. Install the existing HF/SFT/experiment dependencies first (SFT.md and EXPERIMENTS.md), then:

```sh
.venv/bin/python -m pip install -r ml/requirements-analysis.lock
.venv/bin/python -m pip install --no-deps -e ./ml
```

PowerShell uses `.venv\Scripts\python.exe`. Select this interpreter as the notebook kernel. The setup cell prints the project and interpreter and installs the local package if missing; a complete checkout is required.

`ml/analysis/layers.py` inspects actual Qwen module paths, chooses the first six or last six blocks, and trains each subset from the same pinned base. Both use q/v rank-4 adapters, alpha 8, LR 0.0002, 3 epochs, batch 1, accumulation 2 and seed 42 against unchanged Dataset v1 and RecipeTriage-Bench-v1. Two original train recipes supply calibration activations; holdout activations are not used to select targets.

| Subset | Trainable parameters | Training seconds | Micro F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Blocks 0–5 | 67,584 | 29.87 | 19.05% | 12.86% |
| Blocks 18–23 | 67,584 | 16.35 | 43.24% | 32.65% |
| Unadapted base, contextual reference | 0 | — | 48.48% | 36.50% |

Late-six scored higher than early-six in this single run. Neither exceeded the base. The same hardware ran the experiments sequentially, but one timing observation can reflect caching and scheduling. Do not generalize a speed or causal claim from this table.

Before/after records include mean, population standard deviation, RMS and maximum absolute hidden-state values with identical token hashes and eval mode. Adapter diagnostics include A/B norms and relative update norm. A downstream frozen layer can change activations because its input changed. An RMS shift does not identify a semantic feature or prove it causes a label. The notebook explains stronger follow-up evidence and saves `ml/analysis/results/layers-v1/layer-diagnostics.png`.

For a new measured run:

```sh
.venv/bin/python -m recipetriage_ml.analysis.layers --output ml/analysis/runs/new-layer-comparison
```

Use a new output directory. The saved adapters, layer-statistic JSON, run configurations and full benchmark outputs are in `ml/analysis/results/layers-v1/`.
