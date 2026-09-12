import math
import pytest

torch = pytest.importorskip('torch', reason='Educational training requires the optional PyTorch dependency')
from recipetriage_ml.training.fundamentals import (
    demonstrate_modes, demonstrate_zero_grad, make_model, save_history, tiny_data, train,
)


def test_cross_entropy_gradient_and_sgd_match_hand_calculation():
    model = make_model()
    x = torch.tensor([[1.0]])
    y = torch.tensor([1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    assert loss.item() == pytest.approx(math.log(2))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.5], [-0.5]]))
    torch.testing.assert_close(model.bias.grad, torch.tensor([0.5, -0.5]))
    optimizer.step()
    torch.testing.assert_close(model.weight, torch.tensor([[-0.1], [0.1]]))
    torch.testing.assert_close(model.bias, torch.tensor([-0.1, 0.1]))
    after = torch.nn.functional.cross_entropy(model(x), y).item()
    assert after == pytest.approx(math.log1p(math.exp(-0.4)))


def test_accumulation_and_mode_demonstrations():
    first, second, cleared = demonstrate_zero_grad()
    for name in first:
        assert second[name] == pytest.approx(2 * first[name])
        assert cleared[name] is None
    train_output, eval_output, untracked = demonstrate_modes()
    assert (train_output == 0).any() and (train_output == 2).any()
    torch.testing.assert_close(eval_output, torch.ones(16))
    assert eval_output.requires_grad and not untracked.requires_grad


def test_training_is_reproducible_improves_and_keeps_validation_out_of_backward(tmp_path):
    _, first = train(verbose=False)
    _, second = train(verbose=False)
    assert first == second
    assert len(first) == 41 and first[-1]['step'] == 40
    assert first[-1]['train_loss'] < first[0]['train_loss']
    assert first[-1]['validation_loss'] < first[0]['validation_loss']
    assert all(row['loss_after'] < row['loss_before'] for row in first[1:])
    # Changing validation labels must not change the learned parameters.
    from unittest.mock import patch
    original = tiny_data()
    baseline, _ = train(epochs=2, verbose=False)
    with patch('recipetriage_ml.training.fundamentals.tiny_data', return_value=(*original[:3], 1 - original[3])):
        changed, _ = train(epochs=2, verbose=False)
    for a, b in zip(baseline.parameters(), changed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    save_history(first, tmp_path)
    import csv
    with (tmp_path / 'loss_by_step.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 41 and int(rows[-1]['step']) == 40
    assert (tmp_path / 'loss_by_step.md').read_text().startswith('# Measured loss')


def test_partial_batch_step_count():
    _, history = train(epochs=2, batch_size=3, verbose=False)
    assert [row['batch_size'] for row in history[1:]] == [3, 1, 3, 1]
    assert history[-1]['step'] == 4
