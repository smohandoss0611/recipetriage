import json
import pytest
from app.registry import verify_artifacts
from recipetriage_ml.training.data import file_hash


def test_adapter_provenance_is_part_of_registered_identity(tmp_path):
    path=tmp_path/'adapter-provenance.json';identity={'model':'fixture','revision':'fixture-checksum'}
    path.write_text(json.dumps(identity));(tmp_path/'weights.bin').write_bytes(b'fixture weights')
    model={'format':'peft-fp32','artifact_directory':str(tmp_path),'identity':identity,
           'artifact_files':{'weights.bin':file_hash(tmp_path/'weights.bin')}}
    verify_artifacts(model)
    path.write_text(json.dumps({**identity,'model':'changed'}))
    with pytest.raises(ValueError,match='provenance changed'):verify_artifacts(model)
    path.unlink()
    with pytest.raises(ValueError,match='provenance changed'):verify_artifacts(model)
