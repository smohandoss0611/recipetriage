import copy
import json
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ElementTree
from streamlit_ui.client import API, APIError

ROOT = Path(__file__).parents[2]
PAGES = {
    'Recipes': ['Library', 'Add recipe'],
    'Data Lab': ['Dataset Studio', 'Label Editor', 'JSONL Preview', 'Chat Template Preview', 'Synthetic Review'],
    'Training Lab': ['SFT Monitor', 'LoRA Training', 'QLoRA Training', 'Experiments'],
    'Alignment Lab': ['Preference Pairs', 'DPO Training', 'GRPO Experiment', 'Reviewed QLoRA'],
    'Evaluation Lab': ['Baseline', 'Failure Analysis', 'Shortcut Tests'],
    'Playground': ['Triage', 'Model Comparison', 'Token Inspector'],
    'Deployment': ['Measured Candidates', 'Model Registry'],
}


@pytest.fixture
def workspace(monkeypatch):
    # Streamlit 1.63 AppTest treats stateful tabs as blocks rather than widgets,
    # omitting their browser state on reruns. Include the same string-valued
    # tab state sent by the browser; otherwise every simulated click resets it.
    original_states = ElementTree.get_widget_states
    def widget_states(tree):
        state = original_states(tree)
        for block in tree.get('tab_container'):
            identity = block.proto.id
            if identity:
                state.widgets.add(id=identity, string_value=tree.session_state[identity])
        return state
    monkeypatch.setattr(ElementTree, 'get_widget_states', widget_states)
    fixtures = json.loads((Path(__file__).parent / 'fixtures.json').read_text())
    mutations = []
    def request(self, method, path, body=None, **kwargs):
        if method != 'GET':
            mutations.append((method, path, body))
            return {'status': 'queued', 'run_id': 'test-run'}
        return copy.deepcopy(fixtures[path])
    monkeypatch.setattr(API, 'request', request)
    app = AppTest.from_file(str(ROOT / 'streamlit_app.py'), default_timeout=15)
    app.secrets['backend'] = {'url': 'http://127.0.0.1:8000'}
    return app, fixtures, mutations


def open_page(app, section, page=None):
    # AppTest does not yet provide Tab.click(); stateful tabs expose their
    # selected labels in public session state, just like browser tab changes.
    app.session_state['workspace'] = section
    app.run()
    if page is not None:
        app.session_state['page_' + section] = page
        app.run()
    return app


@pytest.mark.parametrize('section,page', [(section, page) for section, pages in PAGES.items() for page in pages])
def test_all_workspaces_render_without_starting_jobs(workspace, section, page):
    app, _, mutations = workspace
    app.run()
    open_page(app, section, page)
    assert not app.exception
    assert not mutations
    assert not app.checkbox
    assert {section, page}.issubset({tab.label for tab in app.tabs})
    assert not any(x.label.endswith(' page') or x.label == 'Recipe workspace' for x in app.selectbox)


def test_preference_choice_is_never_preselected_or_sent_on_open(workspace):
    app, fixtures, mutations = workspace
    recipe = fixtures['/api/v1/datasets/seed']['examples'][0]['recipe']
    fixtures['/api/v1/alignment/pairs'] = [{
        'pair_id': 'test-pair', 'pair_sha256': 'a' * 64, 'recipe': recipe, 'status': 'pending', 'decision': None,
        'candidates': {key: {'raw_output': '{"labels":["meal-prep"]}', 'provider': key, 'finish_reason': 'stop'} for key in ['a', 'b']},
    }]
    open_page(app.run(), 'Alignment Lab')
    choice = next(radio for radio in app.radio if radio.label == 'Your preference')
    assert choice.value is None
    assert not mutations
    choice.set_value('tie')
    next(x for x in app.text_input if x.label == 'Your reviewer name').set_value('Test Reviewer')
    next(x for x in app.button if x.label == 'Save my explicit preference').click().run()
    assert mutations[-1][2]['choice'] == 'tie'
    assert mutations[-1][2]['pair_sha256'] == 'a' * 64


def test_remote_app_opens_without_workspace_sign_in(workspace):
    app, _, mutations = workspace
    app.secrets['backend'] = {'url': 'https://api.example', 'token': 'private-token'}
    app.secrets['access'] = {'password_hash': 'legacy-setting-is-ignored'}
    app.run()
    assert not app.exception
    assert app.tabs
    assert not app.error
    assert not any('Sign in' in title.value for title in app.title)
    assert not mutations


def test_backend_failure_is_displayed_without_crashing(workspace, monkeypatch):
    app, _, _ = workspace
    def failed(*args, **kwargs):
        raise APIError('Backend temporarily unavailable')
    monkeypatch.setattr(API, 'request', failed)
    app.run()
    assert not app.exception
    assert any('temporarily unavailable' in error.value for error in app.error)


def test_single_server_opens_directly_and_retains_private_api_configuration(workspace, monkeypatch):
    app, _, mutations = workspace
    monkeypatch.setenv('RECIPETRIAGE_DEPLOYMENT', 'single-server')
    monkeypatch.setenv('API_AUTH_TOKEN', 't' * 48)
    monkeypatch.setenv('WORKSPACE_PASSWORD_HASH', 'legacy-setting-is-ignored')
    app.run()
    assert not app.exception
    assert app.session_state['workspace'] == 'Recipes'
    assert not any('Sign in' in title.value for title in app.title)
    assert not mutations


def test_single_server_still_rejects_missing_backend_token(workspace, monkeypatch):
    app, _, mutations = workspace
    monkeypatch.setenv('RECIPETRIAGE_DEPLOYMENT', 'single-server')
    monkeypatch.delenv('API_AUTH_TOKEN', raising=False)
    app.run()
    assert not app.exception
    assert any('Docker mode' in error.value for error in app.error)
    assert not app.tabs
    assert not mutations


def test_tab_navigation_preserves_dataset_draft_and_does_not_load_hidden_labs(workspace, monkeypatch):
    app, fixtures, mutations = workspace
    reads = []
    original = API.request
    def request(self, method, path, body=None, **kwargs):
        reads.append(path)
        return original(self, method, path, body, **kwargs)
    monkeypatch.setattr(API, 'request', request)
    app.run()
    draft = copy.deepcopy(fixtures['/api/v1/datasets/seed']['examples'])
    draft[0]['rationale'] = 'Annotation I am still editing'
    draft_json = json.dumps(draft)
    app.session_state['_dataset_raw'] = draft_json
    open_page(app, 'Data Lab', 'Chat Template Preview')
    open_page(app, 'Recipes', 'Library')
    open_page(app, 'Data Lab', 'Label Editor')
    assert not app.exception
    assert app.session_state['_dataset_raw'] == draft_json
    assert not any('/training/' in path or '/alignment/' in path for path in reads)
    assert not mutations


def test_chat_preview_renders_formatted_text_without_a_checkbox(workspace, monkeypatch):
    app, _, mutations = workspace
    preview = {'messages': [
        {'role': 'system', 'content': 'Use the RecipeTriage label definitions.'},
        {'role': 'user', 'content': '{"title": "Overnight Oatmeal"}'},
        {'role': 'assistant', 'content': '{"labels": ["meal-prep"]}'},
    ], 'rendered_text': 'system: rules\nuser: recipe\nassistant: expected answer\n'}
    original = API.request
    def request(self, method, path, body=None, **kwargs):
        if path == '/api/v1/datasets/chat-preview':
            mutations.append((method, path, body))
            return copy.deepcopy(preview)
        return original(self, method, path, body, **kwargs)
    monkeypatch.setattr(API, 'request', request)
    open_page(app.run(), 'Data Lab', 'Chat Template Preview')
    assert not app.checkbox and not mutations
    next(x for x in app.button if x.label == 'Preview chat template').click().run()
    assert not app.exception
    assert mutations[-1][2]['render_model'] is True
    assert len(mutations) == 1
    assert [x.value.strip() for x in app.code] == [x['content'] for x in preview['messages']] + [preview['rendered_text'].strip()]


@pytest.mark.parametrize('provider,supported', [('local', True), ('fireworks', False)])
def test_sft_configuration_check_uses_selected_provider_without_starting_training(workspace, monkeypatch, provider, supported):
    app, _, mutations = workspace
    original = API.request
    def request(self, method, path, body=None, **kwargs):
        if path == '/api/v1/training/preflight':
            mutations.append((method, path, body))
            return {'provider': provider, 'supported': supported,
                    'issues': [] if supported else ['Choose a compatible managed training model.'],
                    'counts': {'train': 1, 'validation': 1, 'test': 1},
                    'dataset_status': 'teaching-draft', 'deferred_checks': ['Tokenized sequence lengths']}
        return original(self, method, path, body, **kwargs)
    monkeypatch.setattr(API, 'request', request)
    open_page(app.run(), 'Training Lab', 'SFT Monitor')
    next(x for x in app.selectbox if x.label == 'Training provider').set_value(provider)
    next(x for x in app.button if x.label == 'Submit training action').click().run()
    assert not app.exception
    assert len(mutations) == 1 and mutations[0][1] == '/api/v1/training/preflight'
    assert mutations[0][2]['provider'] == provider
    if supported:
        assert any('Local configuration check passed' in x.value for x in app.success)
    else:
        assert any('compatible managed training model' in x.value for x in app.error)
        assert not app.success


def review_candidate(fixtures):
    recipe = copy.deepcopy(fixtures['/api/v1/datasets/seed']['examples'][0]['recipe'])
    recipe.update(title='Stuffed Peppers', time_minutes=55)
    return {
        'candidate_id': 'test-candidate', 'status': 'pending', 'revision': 1,
        'category': 'misleading-title', 'review': None,
        'draft': {'recipe': recipe, 'labels': ['weekend-project'],
                  'category': 'misleading-title', 'rationale': 'Review this annotation.',
                  'challenge_notes': 'The title should contrast with the cooking time.'},
        'quality': {'passed': False, 'errors': [
            'Misleading-title proposal requires a declared adjective and known time >30 minutes.'
        ], 'warnings': []},
    }


def test_blocked_review_explains_errors_and_requires_edit_or_reject(workspace):
    app, fixtures, mutations = workspace
    fixtures['/api/v1/curation/candidates'] = [review_candidate(fixtures)]
    open_page(app.run(), 'Data Lab', 'Synthetic Review')
    assert not app.exception
    assert any('declared adjective' in item.value for item in app.error)
    assert any('Quick, Easy or Simple' in item.value for item in app.info)
    decision = next(item for item in app.selectbox if item.label == 'Your decision')
    assert decision.options == ['edit', 'reject']
    assert decision.value is None
    assert not mutations


def test_review_edit_reloads_revision_without_implicitly_approving(workspace, monkeypatch):
    app, fixtures, mutations = workspace
    row = review_candidate(fixtures)
    fixtures['/api/v1/curation/candidates'] = [row]
    original = API.request

    def request(self, method, path, body=None, **kwargs):
        if method == 'POST' and path == '/api/v1/curation/candidates/test-candidate/review':
            mutations.append((method, path, copy.deepcopy(body)))
            assert body['revision'] == row['revision']
            if body['action'] == 'edit':
                row.update(draft=body['draft'], revision=2,
                           quality={'passed': True, 'errors': [], 'warnings': []})
            else:
                assert body['action'] == 'approve'
                row.update(status='approved', review={'reviewer': body['reviewer']})
            return copy.deepcopy(row)
        return original(self, method, path, body, **kwargs)

    monkeypatch.setattr(API, 'request', request)
    open_page(app.run(), 'Data Lab', 'Synthetic Review')
    edited = copy.deepcopy(row['draft'])
    edited['recipe']['title'] = 'Quick Stuffed Peppers'
    next(x for x in app.selectbox if x.label == 'Your decision').set_value('edit')
    next(x for x in app.text_input if x.label == 'Reviewer name').set_value('Test Reviewer')
    next(x for x in app.text_area if x.label.startswith('Edited draft JSON')).set_value(json.dumps(edited))
    next(x for x in app.button if x.label == 'Record my decision').click().run()
    assert not app.exception
    assert len(mutations) == 1 and mutations[0][2]['action'] == 'edit'
    assert row['status'] == 'pending' and row['review'] is None
    assert any('revision 2' in x.value for x in app.success)
    decision = next(x for x in app.selectbox if x.label == 'Your decision')
    assert 'approve' in decision.options and decision.value is None
    decision.set_value('approve')
    next(x for x in app.text_input if x.label == 'Reviewer name').set_value('Test Reviewer')
    next(x for x in app.button if x.label == 'Record my decision').click().run()
    assert not app.exception
    assert len(mutations) == 2 and mutations[1][2]['revision'] == 2
    assert row['status'] == 'approved'


def test_failed_generation_with_no_draft_still_renders_review(workspace):
    app, fixtures, mutations = workspace
    row = review_candidate(fixtures)
    row.update(draft=None, quality={'passed': False, 'errors': ['Generation was truncated.']})
    fixtures['/api/v1/curation/candidates'] = [row]
    open_page(app.run(), 'Data Lab', 'Synthetic Review')
    assert not app.exception
    assert any('truncated' in x.value for x in app.error)
    assert 'Failed generation' in next(x for x in app.selectbox if x.label == 'Candidate').options[0]
    assert not mutations


@pytest.mark.parametrize('submit_conflicting_job', [False, True])
def test_training_monitor_fetches_full_progress_even_after_worker_conflict(workspace, monkeypatch, submit_conflicting_job):
    app, fixtures, mutations = workspace
    summary = {'run_id': 'active-run', 'status': 'baseline'}
    fixtures['/api/v1/training/runs'] = [summary]
    fixtures['/api/v1/training/runs/active-run'] = {
        **summary, 'phase': 'benchmark-base-before-training',
        'baseline': {'status': 'running', 'rows': [{}, {}]},
        'history': [{'step': 1, 'loss': 2.5, 'eval_loss': 2.6}],
    }
    reads = []
    original = API.request

    def request(self, method, path, body=None, **kwargs):
        if method == 'POST' and path == '/api/v1/training/runs':
            mutations.append((method, path, body))
            raise APIError('HTTP 409: A training or benchmark worker is active; inspect its progress first')
        reads.append(path)
        return original(self, method, path, body, **kwargs)

    monkeypatch.setattr(API, 'request', request)
    open_page(app.run(), 'Training Lab', 'SFT Monitor')
    if submit_conflicting_job:
        next(x for x in app.selectbox if x.label == 'Action').set_value('Start training')
        next(x for x in app.button if x.label == 'Submit training action').click().run()
        assert len(mutations) == 1
        assert any('HTTP 409' in x.value for x in app.error)
    else:
        assert not mutations
    assert not app.exception
    assert '/api/v1/training/runs/active-run' in reads
    assert any('Current phase: benchmark base before training' in x.value for x in app.caption)
    assert any('Before-training benchmark: 2 cases recorded' in x.value for x in app.caption)


def test_benchmark_manifest_is_not_shown_as_after_training_progress(workspace):
    app, fixtures, mutations = workspace
    fixtures['/api/v1/benchmarks/runs'] = [{
        'run_id': 'benchmark-run', 'status': 'completed',
        'benchmark': {'version': 'RecipeTriage-Bench-v1'}, 'rows': [],
    }]
    open_page(app.run(), 'Evaluation Lab', 'Baseline')
    assert not app.exception and not mutations
    assert not any('After-training benchmark' in x.value for x in app.caption)
