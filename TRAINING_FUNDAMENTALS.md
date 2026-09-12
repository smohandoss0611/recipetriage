# 04 — Understand a PyTorch update

Start with [ml/notebooks/04_training_fundamentals.ipynb](ml/notebooks/04_training_fundamentals.ipynb). It is self-contained, shows all executable code, and needs PyTorch plus a notebook kernel. It does not import `recipetriage_ml`, download a model, call Fireworks, or use the benchmark as training data.

The readable script is [ml/training/fundamentals.py](ml/training/fundamentals.py). It trains a four-parameter linear classifier on four handwritten time examples, with four separate validation examples. This is a numerical teaching exercise. A simple elapsed-time rule would be better than machine learning for this artificial task; the point is to see parameter updates.

## The equations and the example

We encode **class 0 = more than 30 minutes**, **class 1 = at most 30 minutes**. Training times are `[10, 20, 40, 60]`, with labels `[1, 1, 0, 0]`. Validation times are `[15, 25, 45, 55]`, with the same class ordering. A fixed scaling rule is `x = (minutes - 30) / 30`; we do not fit preprocessing on validation examples.

For a batch of size B, the forward pass is:

```text
X: [B, 1]       W: [2, 1]       b: [2]
z = X Wᵀ + b                    logits: [B, 2]
p[i,k] = exp(z[i,k]) / Σj exp(z[i,j])
L = -(1/B) Σi log(p[i, y[i]])    mean cross-entropy
```

For the notebook's first example, 10 minutes gives x = -2/3 and target y = 1. All four parameters start at zero. Thus logits are `[0, 0]`, probabilities are `[0.5, 0.5]`, and loss is `-ln(0.5) = 0.693147`.

The one-example derivatives are:

```text
∂L/∂z[k] = p[k] - indicator(k == y)
∂L/∂W[k] = (p[k] - indicator(k == y)) × x
∂L/∂b[k] = p[k] - indicator(k == y)
```

So the weight gradient is `[-1/3, +1/3]ᵀ` and bias gradient `[+1/2, -1/2]`. Backpropagation calculates these using the chain rule. It does not itself update parameters. Plain SGD with learning rate η = 0.2 applies:

```text
θ_new = θ_old - η × gradient
W_new = [+1/15, -1/15]ᵀ
b_new = [-0.1, +0.1]
```

The correct class becomes more probable. The notebook checks the manual calculation against autograd and measures loss again after the update. Zero initialization is chosen for transparent linear-model arithmetic; it is not a general deep-network initialization strategy.

Cross-entropy takes **raw logits** and integer class labels in this example. Softmax is shown only to inspect probabilities. This two-class objective is not a drop-in replacement for RecipeTriage's seven-label problem. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## The loop, line by line

```python
model.train()                          # Training behavior for mode-sensitive layers.
optimizer.zero_grad(set_to_none=True)  # Clear gradients from previous work.
logits = model(x_batch)                # Forward pass.
loss = loss_fn(logits, y_batch)        # Cross-entropy against correct labels.
loss_before = loss.item()              # Detached scalar for reporting.
loss.backward()                       # Chain rule computes parameter gradients.
norms = gradient_norms(model)          # Inspect their size before updating.
optimizer.step()                      # SGD changes the parameters.
with torch.no_grad():
    loss_after = loss_fn(model(x_batch), y_batch).item()
```

A **gradient** gives local sensitivity of loss to a parameter. Its L2 norm is `sqrt(sum(g_i²))`; the norm summarizes magnitude but hides signs. The script prints weight and bias norms separately.

The **optimizer** defines how gradients change parameters; this lesson uses SGD without momentum. The **learning rate** scales each update. Large rates can overshoot and increase loss; small rates usually progress slowly. A decrease after every update is observed at the supplied settings, not guaranteed for arbitrary settings.

Gradients accumulate in PyTorch. Two backward passes on the same examples and unchanged model produce twice the gradient. The dedicated demonstration recomputes the forward graph for its second backward pass and verifies that doubling. `zero_grad(set_to_none=True)` then makes stored gradients `None`; it does not reset weights. `optimizer.step()` does not clear gradients. With `set_to_none=False`, existing gradient tensors are zeroed instead. [PyTorch zero_grad](https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.zero_grad.html).

A **batch** is a group processed together. An **epoch** visits every training example once. A **step** here is one optimizer update. With no accumulation and no dropped final batch, `steps_per_epoch = ceil(N / batch_size)`: four examples, batch size two and twenty epochs produce forty steps.

`train()` and `eval()` select layer behavior. Dropout changes behavior; our main linear layer does not, so a separate Dropout probe shows the difference. `eval()` does not disable gradients. `no_grad()` disables recording of new operations; it neither changes model mode nor detaches existing tensors. Evaluation measurements use both. [PyTorch autograd modes](https://docs.pytorch.org/docs/main/notes/autograd.html).

**Overfitting** means learning details or noise specific to training examples that do not generalize. Falling training loss while validation loss rises is a common signal. The default small linear example improves on both sets; it does not demonstrate actual overfitting or establish real-world accuracy. More epochs do not inevitably cause overfitting. The notebook asks you to reason about this distinction before advancing.

## Measured results

The reference CPU run used twenty epochs, batch size two, learning rate 0.2, float32, and shuffle seed 7:

| Measurement | Before | After |
|---|---:|---:|
| Full training-set loss, step 0 → 40 | 0.693147 | 0.233313 |
| Full validation-set loss, step 0 → 40 | 0.693147 | 0.265931 |
| First batch loss, before/after its one update | 0.693147 | 0.559341 |
| Weight gradient norm, one/two accumulated backward passes | 0.412479 | 0.824958 |

The complete measured table is in [loss_by_step.md](ml/training/results/reference/loss_by_step.md) and [loss_by_step.csv](ml/training/results/reference/loss_by_step.csv). It contains the initial measurement and forty updates. Before/after batch loss and gradient norms are blank at step zero because no update has happened. Full-set losses use fixed examples, which makes them easier to compare across steps than losses from changing batches.

The notebook's single-example demonstration and the CLI's first two-example batch are different calculations; their post-update losses need not match. The model in each experiment starts fresh. A fixed seed fixes the batch order here, but bitwise reproducibility across hardware and PyTorch versions is not promised.

## Run commands

Use your existing project Python 3.12 environment. Create one only if needed. On macOS/Linux, from the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.14.0
.venv/bin/python -m pip install -e './ml[training,notebook]'
.venv/bin/python -m recipetriage_ml.training.fundamentals --output training-results
```

On Linux CPU-only machines, replace the PyTorch installation line with:

```sh
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.0
.venv\Scripts\python.exe -m pip install -e './ml[training,notebook]'
.venv\Scripts\python.exe -m recipetriage_ml.training.fundamentals --output training-results
```

Open `ml/notebooks/04_training_fundamentals.ipynb` in your notebook editor and choose the project `.venv` Python kernel. Run all cells in order. The first cell prints the actual interpreter and PyTorch version. If importing torch fails, install it in that kernel's environment; `%pip install torch==2.14.0` in a notebook cell targets the active environment. Restart the kernel after installation if necessary.

If using JupyterLab instead of an existing notebook editor:

```sh
.venv/bin/python -m pip install jupyterlab
.venv/bin/python -m jupyterlab ml/notebooks/04_training_fundamentals.ipynb
```

PowerShell substitutes `.venv\Scripts\python.exe`. This additional notebook UI installation requires network access; the training computations themselves do not.

The notebook writes to `training-results/notebook/`. The CLI's `--output` chooses its folder; rerunning replaces that folder's two table files. Use different output folders when comparing learning rates:

```sh
.venv/bin/python -m recipetriage_ml.training.fundamentals --learning-rate 0.02 --output training-results/lr-002
.venv/bin/python -m recipetriage_ml.training.fundamentals --learning-rate 2.0 --output training-results/lr-2
```

Docker alternative after copying this phase into your project, with Docker running:

```sh
docker compose build backend
docker compose run --name recipetriage-training-lesson --no-deps backend python -m recipetriage_ml.training.fundamentals --output /tmp/training-results
docker cp recipetriage-training-lesson:/tmp/training-results ./training-results
docker rm recipetriage-training-lesson
```

These commands work on macOS, Linux, and PowerShell. The stopped lesson container is kept until its table files are copied to your machine, then removed. Choose another container name if that name already exists. This lesson does not need PostgreSQL or a running API.

## Files and verification

- `ml/training/fundamentals.py`: toy data, model, explicit loop, accumulation/mode demonstrations, table exporter and CLI.
- `ml/training/__init__.py`: marks the educational module as a package.
- `ml/notebooks/04_training_fundamentals.ipynb`: equations, live single-step inspection, visible loop, measured table and understanding checkpoint.
- `ml/training/results/reference/`: the actual CLI reference table, kept separately from your experiments.
- `ml/pyproject.toml`: registers the training package and lightweight optional PyTorch dependency group.
- `tests/ml/test_training_fundamentals.py`: checks manual cross-entropy/SGD math, gradient accumulation, mode behavior, reproducibility, held-out labels not affecting learning, partial batches, and saved measurements.

Four targeted tests pass. All eleven notebook code cells executed successfully in a fresh PyTorch kernel, and all 41 measured table rows exactly matched the CLI reference. Install the existing backend test requirements to run the tests:

```sh
.venv/bin/python -m pip install -r backend/requirements-test.lock
.venv/bin/python -m pytest -q tests/ml/test_training_fundamentals.py
```

This phase changes only toy-model weights. The serving models, benchmark, dataset versions and web application remain as before. General background: [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

Verification environment note: the first Jupyter launch required local socket permission. Its normal shutdown then logged a macOS sandbox error when inspecting child processes, after all notebook cells had succeeded. Re-execution with immediate cleanup of the owned verification kernel completed successfully. The kernel also emitted a local TCP transport warning. These environment messages are separate from the notebook's computations; no cells or errors were skipped.

## Explain before moving on

1. Which line calculates predictions, which calculates gradients, and which changes weights?
2. Why do gradients double without zero_grad? Does clearing gradients reset learned weights?
3. Why must loss be recomputed after an optimizer step?
4. How many steps are there in twenty epochs of four examples with batches of two? Does eval mode disable gradient tracking?
5. What might falling training loss and rising validation loss mean?

SFTTrainer is intentionally deferred until you can explain this loop in your own words.
