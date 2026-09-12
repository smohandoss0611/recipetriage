import copy
import hashlib
import json
from importlib.resources import files
import pytest
from pydantic import ValidationError
from recipetriage_ml.data.schemas import Recipe, TrainingExample
from recipetriage_ml.data.pipeline import (as_drafts, build_dataset, chat_messages, export_jsonl, fingerprint,
    group_examples, load_raw, load_recipes, render_chat, split_examples, validate_clean)
from recipetriage_ml.inference.parsing import parse_prediction
from recipetriage_ml.inference.prompts import build_messages


@pytest.fixture
def records():
    return json.loads(files('recipetriage_ml').joinpath('data/seed.json').read_text())


def test_seed_contracts_and_no_invented_provenance(records):
    values,issues,duplicates = validate_clean(records)
    assert len(values)==7 and not issues and not duplicates
    assert all(x.recipe.source_type=='published' and not x.reviewed and x.reviewed_by is None for x in values)
    assert not any(x.recipe.title.startswith(('Easy Ravioli','Quick Ravioli','Simple Ravioli')) for x in values)


@pytest.mark.parametrize('labels', [['invented'],[],['meal-prep','meal-prep'],['unclear','meal-prep'],['weeknight-30min'],['have-most-of-this']])
def test_invalid_labels_rejected(records,labels):
    records[0]['labels']=labels
    with pytest.raises(ValidationError): TrainingExample.model_validate(records[0])


@pytest.mark.parametrize('field',['title','ingredients','instructions','equipment','time_minutes','id','source_uri'])
def test_missing_recipe_fields(records,field):
    del records[0]['recipe'][field]
    with pytest.raises(ValidationError): TrainingExample.model_validate(records[0])


def test_missing_targets_not_filled(records):
    recipe=records[0]['recipe']
    assert load_recipes(json.dumps([recipe]))[0].id==recipe['id']
    draft=as_drafts([recipe])[0]
    assert draft['labels']==[]
    _,issues,_=validate_clean([draft])
    assert issues


def test_cleaning_normalizes_whitespace_not_facts(records):
    records[0]['recipe']['title']='  Overnight   Oatmeal \n'
    records[0]['recipe']['ingredients'][0]=' rolled   oats '
    parsed=TrainingExample.model_validate(records[0])
    assert parsed.recipe.title=='Overnight Oatmeal'
    assert parsed.recipe.ingredients[0]=='rolled oats'
    assert parsed.recipe.time_minutes==375


def test_duplicate_body_removed_despite_new_title_and_id(records):
    duplicate=copy.deepcopy(records[0]); duplicate['recipe'].update(id='z-copy',title='Renamed oatmeal')
    duplicate['recipe']['ingredients'].reverse()
    duplicate['recipe']['instructions']=[x.upper() for x in duplicate['recipe']['instructions']]
    parsed,issues,duplicates=validate_clean(records+[duplicate])
    assert len(parsed)==7 and not issues
    assert duplicates[0]['removed']=='z-copy'
    result=build_dataset(records+[duplicate])
    assert sum(result['metadata']['counts'].values())==7
    assert len(json.loads(result['files']['raw.json']))==8


def test_conflicting_duplicate_blocks_export(records):
    duplicate=copy.deepcopy(records[0]); duplicate['recipe']['id']='z-copy'; duplicate['labels']=['unclear']
    _,issues,_=validate_clean(records+[duplicate])
    assert any(x['kind']=='duplicate_conflict' for x in issues)
    with pytest.raises(ValueError,match='duplicate_conflict'): build_dataset(records+[duplicate])


def test_reused_id_with_different_body_is_error(records):
    records[1]['recipe']['id']=records[0]['recipe']['id']
    with pytest.raises(ValueError,match='id_conflict'): build_dataset(records)


def test_review_identity_required(records):
    records[0]['reviewed']=True
    with pytest.raises(ValidationError,match='reviewed_by'): TrainingExample.model_validate(records[0])


def test_pantry_matching_threshold(records):
    records[1]['labels']=['have-most-of-this']
    records[1]['recipe']['pantry_items']=records[1]['recipe']['ingredients'][:5]
    with pytest.raises(ValidationError,match='80%'): TrainingExample.model_validate(records[1])
    records[1]['recipe']['pantry_items']=records[1]['recipe']['ingredients'][:6]
    assert TrainingExample.model_validate(records[1]).labels==['have-most-of-this']


def test_split_completeness_no_id_or_body_leakage(records):
    examples,_,_=validate_clean(records)
    splits=split_examples(examples)
    assert sorted(len(x) for x in splits.values())==[1,1,5]
    assert sorted(x.recipe.id for rows in splits.values() for x in rows)==sorted(x.recipe.id for x in examples)
    names=list(splits)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            assert not ({fingerprint(x) for x in splits[a]} & {fingerprint(x) for x in splits[b]})


def test_source_and_manual_groups_do_not_cross_splits(records):
    # Different recipes sharing a source are grouped, not deleted.
    records[1]['recipe']['source_uri']=records[0]['recipe']['source_uri']+'?tracking=1'
    records[2]['recipe']['group_id']='related-family'
    records[3]['recipe']['group_id']='related-family'
    result=build_dataset(records)
    allocation={id:name for name,ids in result['metadata']['split_ids'].items() for id in ids}
    assert allocation['seed-001']==allocation['seed-002']
    assert allocation['seed-003']==allocation['seed-004']


def test_removed_duplicate_source_alias_preserves_grouping(records):
    alias=copy.deepcopy(records[0]); alias['recipe']['id']='z-alias'
    alias['recipe']['source_uri']=records[1]['recipe']['source_uri']
    result=build_dataset(records+[alias])
    allocation={id:name for name,ids in result['metadata']['split_ids'].items() for id in ids}
    assert allocation['seed-001']==allocation['seed-002']


def test_deterministic_splits_and_version_under_input_permutation(records):
    a=build_dataset(records,seed=42)
    b=build_dataset(list(reversed(records)),seed=42)
    assert a['metadata']['split_ids']==b['metadata']['split_ids']
    assert a['metadata']['version']==b['metadata']['version']
    assert a['files']==b['files']
    allocations={json.dumps(build_dataset(records,seed=n)['metadata']['split_ids'],sort_keys=True) for n in range(10)}
    assert len(allocations)>1


def test_independent_groups_required(records):
    for row in records: row['recipe']['group_id']='same-family'
    with pytest.raises(ValueError,match='three independent'): build_dataset(records)


@pytest.mark.parametrize('ratios',[(0.8,0.2,0),(0.7,0.2,0.2),(float('nan'),0.5,0.5)])
def test_invalid_ratios(records,ratios):
    with pytest.raises(ValueError): build_dataset(records,ratios=ratios)


def test_jsonl_and_chat_targets_match_inference_contract(records):
    result=build_dataset(records)
    for split in ['train','validation','test']:
        raw=load_raw(result['files'][split+'.jsonl'],'jsonl')
        chats=load_raw(result['files'][split+'.chat.jsonl'],'jsonl')
        assert len(raw)==len(chats)
        for row,chat in zip(raw,chats):
            example=TrainingExample.model_validate(row)
            messages=chat['messages']
            assert messages[:2]==build_messages(example.recipe.inference_recipe())
            assert [x['role'] for x in messages]==['system','user','assistant']
            target=parse_prediction(messages[-1]['content'])
            assert target.labels==example.labels and target.explanation==example.rationale
            assert not ({'labels','rationale','annotation_notes','reviewed_by','source_notes','source_uri','id'} & json.loads(messages[1]['content']).keys())
    for name,checksum in result['metadata']['file_sha256'].items():
        assert hashlib.sha256(result['files'][name].encode()).hexdigest()==checksum


def test_render_chat_uses_complete_supervised_conversation(records):
    example=TrainingExample.model_validate(records[0])
    class Tokenizer:
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            assert not tokenize and not add_generation_prompt
            assert messages[-1]['role']=='assistant'
            return 'rendered'
    assert render_chat(example,Tokenizer())=='rendered'


def test_version_changes_with_annotation(records):
    before=build_dataset(records)['metadata']['version']
    records[0]['rationale']+=' Portions are prepared ahead.'
    assert build_dataset(records)['metadata']['version']!=before


def test_bad_json_and_jsonl_are_visible():
    with pytest.raises(ValueError,match='line 2'): load_raw('{}\nnot json','jsonl')
    with pytest.raises(ValueError,match='Duplicate JSON key'): load_raw('[{"a":1,"a":2}]')
    with pytest.raises(ValueError,match='Non-finite'): load_raw('[NaN]')
    with pytest.raises(ValueError,match='array'): load_raw('{}')
