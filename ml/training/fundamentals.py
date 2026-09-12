"""Read this small CPU training loop before using a training framework.

Run: python -m recipetriage_ml.training.fundamentals --output training-results
Only a four-parameter toy classifier is trained. No benchmark data or LLM is used.
"""
import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn


def tiny_data():
    """Handwritten teaching examples: class 0 = over 30 min; class 1 = <=30 min."""
    train_minutes = torch.tensor([[10.0], [20.0], [40.0], [60.0]])
    valid_minutes = torch.tensor([[15.0], [25.0], [45.0], [55.0]])
    train_labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    valid_labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    # A fixed, documented scaling rule; no statistics are learned from validation.
    return (train_minutes - 30) / 30, train_labels, (valid_minutes - 30) / 30, valid_labels


def make_model():
    # One input, two logits: two weights plus two biases = four trainable numbers.
    # Zero initialization is intentional for this linear lesson, not a deep-network recipe.
    with torch.random.fork_rng():
        model = nn.Linear(1, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def gradient_norms(model):
    # ||g||_2 = sqrt(sum_i g_i^2). None means no gradient tensor is stored.
    return {name: None if parameter.grad is None else parameter.grad.norm().item()
            for name, parameter in model.named_parameters()}


def demonstrate_zero_grad():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_fn = nn.CrossEntropyLoss()
    x, y, _, _ = tiny_data()
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    first = {name: p.grad.clone() for name, p in model.named_parameters()}
    first_norms = gradient_norms(model)
    # Recompute the forward pass to create a new graph. No optimizer step occurs here.
    loss_fn(model(x), y).backward()
    second_norms = gradient_norms(model)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, 2 * first[name])
    optimizer.zero_grad(set_to_none=True)
    cleared = gradient_norms(model)
    assert all(value is None for value in cleared.values())
    print('After one backward:', first_norms)
    print('After two backwards WITHOUT zero_grad:', second_norms)
    print('After zero_grad(set_to_none=True):', cleared)
    return first_norms, second_norms, cleared


def demonstrate_modes():
    # Dropout makes train/eval behavior visible; the main linear model has no dropout.
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(16, requires_grad=True)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        dropout.train()
        training_output = dropout(x)
        dropout.eval()
        # Add zero so there is a real operation even when eval Dropout is identity.
        evaluation_output = dropout(x) + 0
        with torch.no_grad():
            untracked_output = dropout(x) + 0
    print('train() Dropout output:', training_output.detach().tolist())
    print('eval() Dropout output:', evaluation_output.detach().tolist())
    print('eval() still tracks gradients:', evaluation_output.requires_grad)
    print('eval() + no_grad tracks gradients:', untracked_output.requires_grad)
    return training_output, evaluation_output, untracked_output


def train(epochs=20, batch_size=2, learning_rate=0.2, verbose=True):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be positive and finite')
    x_train, y_train, x_valid, y_valid = tiny_data()
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # Give it raw logits, not softmax probabilities.
    generator = torch.Generator().manual_seed(7)
    history = []
    step = 0
    model.eval()
    with torch.no_grad():
        initial_train = loss_fn(model(x_train), y_train).item()
        initial_valid = loss_fn(model(x_valid), y_valid).item()
    history.append(dict(step=0, epoch=0, batch_size=0, loss_before=None, loss_after=None,
                        weight_grad_norm=None, bias_grad_norm=None,
                        train_loss=initial_train, validation_loss=initial_valid))

    for epoch in range(1, epochs + 1):
        indices = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(x_train), batch_size):
            batch = indices[start:start + batch_size]
            x_batch, y_batch = x_train[batch], y_train[batch]
            model.train()                          # Set training behavior.
            optimizer.zero_grad(set_to_none=True)  # Clear the PREVIOUS gradients.
            logits = model(x_batch)                # Forward pass: [batch, 2] logits.
            loss = loss_fn(logits, y_batch)        # Mean cross-entropy for this batch.
            loss_before = loss.item()
            loss.backward()                       # Backpropagation computes gradients.
            norms = gradient_norms(model)          # Inspect gradients BEFORE updating.
            optimizer.step()                      # SGD updates parameters using gradients.
            with torch.no_grad():
                loss_after = loss_fn(model(x_batch), y_batch).item()  # SAME batch, new forward.
            model.eval()                           # Evaluation behavior, separate from no_grad.
            with torch.no_grad():
                train_loss = loss_fn(model(x_train), y_train).item()
                valid_loss = loss_fn(model(x_valid), y_valid).item()
            step += 1
            history.append(dict(step=step, epoch=epoch, batch_size=len(batch),
                                loss_before=loss_before, loss_after=loss_after,
                                weight_grad_norm=norms['weight'], bias_grad_norm=norms['bias'],
                                train_loss=train_loss, validation_loss=valid_loss))
            if verbose:
                print(f'step={step:02d} epoch={epoch:02d} batch={len(batch)} '
                      f'loss {loss_before:.6f} -> {loss_after:.6f} '
                      f'grad norms: weight={norms["weight"]:.6f}, bias={norms["bias"]:.6f} '
                      f'validation={valid_loss:.6f}')
    return model, history


def history_table(history):
    columns = list(history[0])
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in history:
        values = ['—' if row[key] is None else f'{row[key]:.6f}' if isinstance(row[key], float)
                  else str(row[key]) for key in columns]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def save_history(history, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'loss_by_step.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output / 'loss_by_step.md').write_text('# Measured loss by optimizer step\n\n' + history_table(history), encoding='utf-8')
    print('Saved:', output / 'loss_by_step.csv', 'and', output / 'loss_by_step.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.2)
    parser.add_argument('--output', type=Path, default=Path('training-results'))
    args = parser.parse_args()
    demonstrate_zero_grad()
    demonstrate_modes()
    _, history = train(args.epochs, args.batch_size, args.learning_rate)
    save_history(history, args.output)
    print(f'Full training loss: {history[0]["train_loss"]:.6f} -> {history[-1]["train_loss"]:.6f}')


if __name__ == '__main__':
    main()
