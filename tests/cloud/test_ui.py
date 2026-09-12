import copy
import json
from pathlib import Path

import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ElementTree

from streamlit_ui.app import PAGES
from streamlit_ui.embedded import EmbeddedAPI

ROOT = Path(__file__).parents[2]


@pytest.fixture
def cloud_ui(monkeypatch):
    original = ElementTree.get_widget_states
    def states(tree):
        result = original(tree)
        for block in tree.get('tab_container'):
            if block.proto.id:
                result.widgets.add(id=block.proto.id, string_value=tree.session_state[block.proto.id])
        return result
    monkeypatch.setattr(ElementTree, 'get_widget_states', states)
    fixtures = json.loads((ROOT/'tests/streamlit/fixtures.json').read_text())
    writes = []
    def request(self, method, path, body=None, **kwargs):
        if method != 'GET':
            writes.append((path, body))
            return {'status': 'saved'}
        return copy.deepcopy(fixtures[path])
    monkeypatch.setattr(EmbeddedAPI, 'request', request)
    monkeypatch.setattr('streamlit_ui.embedded.embedded_api', lambda url: EmbeddedAPI(None))
    app = AppTest.from_file(str(ROOT/'cloud/streamlit_app.py'), default_timeout=15)
    app.secrets['database'] = {'url': 'postgresql://u:test-only@localhost/cloud_test'}
    return app, writes


@pytest.mark.parametrize('section,page', [(s,p) for s,pages in PAGES.items() for p in pages])
def test_cloud_tabs_render_without_jobs_or_checkbox_controls(cloud_ui, section, page):
    app, writes = cloud_ui
    app.session_state['workspace'] = section
    app.run()
    app.session_state['page_' + section] = page
    app.run()
    assert not app.exception
    assert not writes
    assert not app.checkbox
    assert {section,page}.issubset({tab.label for tab in app.tabs})
    blocked_labels = {
        'Generate proposals with Fireworks', 'Run triage for this revision',
        'Submit training action', 'Resume run', 'Start rank comparison', 'Start experiment matrix',
        'Generate candidate pairs', 'Start DPO experiment', 'Start GRPO experiment',
        'Retrain QLoRA and evaluate shortcut accuracy', 'Run fixed benchmark',
        'Run shortcut and red-team diagnostics', 'Run inference', 'Start model comparison',
        'Apply stage change subject to quality gates',
    }
    for button in app.button:
        if button.label in blocked_labels:
            assert button.disabled, button.label


def test_cloud_entrypoint_explains_missing_database_without_api_url():
    app = AppTest.from_file(str(ROOT/'cloud/streamlit_app.py')).run()
    assert not app.exception
    assert any('database.url' in error.value for error in app.error)
    assert not any('backend.url' in item.value for item in app.info)


def test_cloud_add_recipe_uses_structured_fields(cloud_ui):
    app, writes = cloud_ui
    app.session_state['workspace'] = 'Recipes'
    app.run()
    app.session_state['page_Recipes'] = 'Add recipe'
    app.run()
    assert not app.exception and not writes
    assert any(item.label == 'Title' for item in app.text_input)
    assert not any(item.label == 'Source' for item in app.selectbox)
    assert not app.get('file_uploader')
