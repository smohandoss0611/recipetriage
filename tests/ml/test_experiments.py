import copy
import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from recipetriage_ml.training.experiments import default_matrix,MatrixConfig,compare_matrix


def reference():
    root=Path(__file__).parents[2]/'ml/training/results/experiments-v1';experiment=json.loads((root/'experiment.json').read_text())
    return experiment,[json.loads((root/'runs'/row['run_id']/'run.json').read_text()) for row in experiment['runs']]


def test_matrix_rejects_leakage_budget_changes_and_duplicates():
    config=default_matrix();assert len(config.runs)==5
    for field,value in [('dataset_version','v1-'+'0'*64),('seed',43),('gradient_checkpointing',False)]:
        raw=config.model_dump();raw['runs'][1][field]=value
        with pytest.raises(ValidationError,match='fixed'):MatrixConfig.model_validate(raw)
    raw=config.model_dump();raw['runs'][1]=raw['runs'][0]
    with pytest.raises(ValidationError,match='Duplicate'):MatrixConfig.model_validate(raw)


def test_real_matrix_comparison_changes_one_factor_at_a_time():
    experiment,runs=reference();assert compare_matrix(runs)==experiment['comparison']
    effects=experiment['comparison']['effects'];assert len(effects)==5
    assert {effect['factor'] for effect in effects}=={'rank','learning_rate','method'}
    assert all(run['tracking']['status']=='synced' for run in runs)
    assert all(run['comparison']['delta']['micro_f1']<0 for run in runs)
    changed=copy.deepcopy(runs);changed[1]['dataset']['logical_messages_sha256']['train']='changed'
    assert not compare_matrix(changed)['comparable']
    changed=copy.deepcopy(runs);changed[1]['status']='failed'
    assert not compare_matrix(changed)['comparable']


def test_nf4_configuration_probe_and_real_quantization_audit():
    torch=pytest.importorskip('torch');pytest.importorskip('bitsandbytes')
    from recipetriage_ml.training.qlora import quantization_config,capability
    config=quantization_config()
    assert config.load_in_4bit and config.bnb_4bit_quant_type=='nf4'
    assert config.bnb_4bit_use_double_quant and config.bnb_4bit_compute_dtype==torch.float32
    assert capability()['supported'],capability().get('error')
    _,runs=reference();qlora=runs[-1]
    assert qlora['quantization']['quantized_module_count']==168
    assert qlora['model_parameters']['trainable']==270336
    assert qlora['model_parameters']['frozen']==494032768
    assert all(row['storage_dtype']=='torch.uint8' for row in qlora['quantization']['quantized_modules'].values())
    assert qlora['parameters']['gradient_checkpointing'] is True


def test_mlflow_failure_does_not_erase_training_evidence(monkeypatch,tmp_path):
    mlflow=pytest.importorskip('mlflow')
    from recipetriage_ml.training.tracking import export_run
    _,runs=reference();original=copy.deepcopy(runs[0])
    def fail(*args,**kwargs):raise RuntimeError('test service unavailable')
    monkeypatch.setattr(mlflow,'MlflowClient',fail)
    assert export_run(runs[0],tmp_path)['status']=='failed'
    assert runs[0]==original
