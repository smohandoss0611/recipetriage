"""Check that a dashboard can access the artifacts exported by its clients."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipetriage_ml.training.tracking import export_run


@pytest.mark.parametrize('uri,server_managed', [
    ('http://mlflow:5000', True),
    ('https://tracking.example.test', True),
    ('', False),
])
def test_export_uses_server_artifacts_for_http(monkeypatch, tmp_path, uri, server_managed):
    mlflow = pytest.importorskip('mlflow')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    monkeypatch.setenv('TRAINING_RUNS_DIR', str(tmp_path))
    root = Path(__file__).parents[2] / 'ml/training/results/experiments-v1/runs'
    source = next(root.glob('*/run.json'))
    run = json.loads(source.read_text())
    original = copy.deepcopy(run)
    (tmp_path / 'run.json').write_text(source.read_text())
    client = Mock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = '1'
    client.search_runs.return_value = []
    client.create_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id='tracked'))
    monkeypatch.setattr(mlflow, 'MlflowClient', Mock(return_value=client))

    assert export_run(run, tmp_path)['status'] == 'synced'
    kwargs = client.create_experiment.call_args.kwargs
    if server_managed:
        assert 'artifact_location' not in kwargs
        assert not (tmp_path / 'mlflow-artifacts').exists()
    else:
        assert kwargs['artifact_location'] == (tmp_path / 'mlflow-artifacts').as_uri()
    client.log_artifact.assert_called_once_with('tracked', str(tmp_path / 'run.json'), artifact_path='evidence')
    assert run == original

    # Importing a saved journal again must resolve the existing run, not create another.
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id='1')
    client.search_runs.return_value = [client.create_run.return_value]
    assert export_run(run, tmp_path)['run_id'] == 'tracked'
    assert client.create_run.call_count == 1
