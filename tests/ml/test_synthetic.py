import copy
import json
import pytest
from recipetriage_ml.data.synthetic_fireworks import quality_checks, generate_batch
from recipetriage_ml.data.schemas import TrainingExample
from recipetriage_ml.data.pipeline import build_dataset
from recipetriage_ml.training.data import seed_snapshot
from recipetriage_ml.inference.contracts import Generation


def draft():
    return {'recipe':{'title':'Quick Stuffed Leeks','ingredients':['leeks','millet','sage'],
        'instructions':['Prepare and stuff the leeks for 35 minutes.','Bake for 25 minutes.'],
        'equipment':['oven'],'time_minutes':60,'pantry_items':None},
        'labels':['weekend-project'],'rationale':'Stuffing and baking take 60 minutes with substantial active work.',
        'category':'misleading-title','challenge_notes':'Quick is misleading.'}


def test_generator_never_approves_or_exports(tmp_path):
    class Provider:
        model='mock'
        def generate(self,*args):return Generation(raw_output=json.dumps(draft()),model=self.model,finish_reason='stop')
    result=generate_batch(1,Provider())
    assert result['items'][0]['status']=='pending' and result['items'][0]['review'] is None
    assert result['training_exported'] is False
    assert result['items'][0]['prompt_sha256']
    assert result['items'][0]['revision']==1
    assert result['items'][0]['model_revision'] is None


def test_checks_reject_bad_labels_duplicates_and_truncation():
    row=draft();assert quality_checks(row)['passed']
    row['labels']=['weeknight-30min'];assert not quality_checks(row)['passed']
    row=draft();assert not quality_checks(row,others=[row['recipe']])['passed']
    assert not quality_checks(row,finish_reason='length')['passed']


def test_unapproved_synthetic_cannot_enter_generic_exports():
    rows=seed_snapshot()['examples'];rows[0]['recipe']['source_type']='synthetic'
    with pytest.raises(ValueError,match='explicit human approval'):build_dataset(rows)
    assert seed_snapshot()['metadata']['version']=='v1-63135eef80fb9be90547673e7f13d4142eb0eb29852cf098e672b44fca20158f'
