# RecipeTriage ML learning code

This package implements tokenization, inference, Dataset Engineering v1, RecipeTriage-Bench-v1 evaluation, and an educational PyTorch training loop. Local SFT trains a separate LoRA adapter; existing serving providers remain configured independently. Managed SFT checks provider and dataset support explicitly. See `LEARNING.md`, `DATASET_V1.md`, `BENCHMARK.md`, and `TRAINING_FUNDAMENTALS.md`, and `SFT.md` at the repository root.

- `notebooks/01_tokenizer.ipynb`: executable lesson using the real Qwen tokenizer and next-token logits.
- `notebooks/04_training_fundamentals.ipynb`: self-contained lesson on forward passes, loss, gradients, SGD, modes, batches and epochs.
- `training/`: the explicit toy loop plus local/managed SFT, shared hyperparameters, checked dataset exports, immutable journals, saved adapters, and benchmark comparisons.
- `inference/`: prompt, schema, parsing, local/hosted providers, and experiment runner.
- `data/`: shared dataset schemas, seven individually assembled source-linked examples, validation, cleaning, deduplication, grouped splitting, chat exports, and version metadata.
- `datasets/`: immutable Dataset v1 export files with unreviewed teaching annotations and hashes.
- `examples/ravioli.json`: source-linked recipe summary and proposed evaluation rubric.
- `experiments/`: immutable per-run results for later shortcut testing.
- `evaluation/`: fixed source-linked benchmark, metrics, true pretrained base provider, CLI, and five real baseline/diagnostic runs. Never use these cases as training input.
- `pyproject.toml`: installable `recipetriage_ml` package; heavy HF/notebook dependencies are optional for pure unit tests.
- `requirements-hf.lock` and `requirements-sft.lock`: pinned inference and SFT libraries for local execution and Docker.
