"""RecipeTriage's seven workspaces, backed by the existing FastAPI contracts."""
import base64
import copy
import json
from urllib.parse import quote, urlsplit

import streamlit as st

from .client import API, APIError
from .settings import configuration

PAGES = {
    'Recipes': ['Library', 'Add recipe'],
    'Data Lab': ['Dataset Studio', 'Label Editor', 'JSONL Preview', 'Chat Template Preview', 'Synthetic Review'],
    'Training Lab': ['SFT Monitor', 'LoRA Training', 'QLoRA Training', 'Experiments'],
    'Alignment Lab': ['Preference Pairs', 'DPO Training', 'GRPO Experiment', 'Reviewed QLoRA'],
    'Evaluation Lab': ['Baseline', 'Failure Analysis', 'Shortcut Tests'],
    'Playground': ['Triage', 'Model Comparison', 'Token Inspector'],
    'Deployment': ['Measured Candidates', 'Model Registry'],
}
SECTIONS = list(PAGES)
LABELS = ['weeknight-30min', 'weekend-project', 'needs-special-equipment', 'meal-prep', 'dessert', 'have-most-of-this', 'unclear']
RECIPE_FIELDS = ['title', 'ingredients', 'instructions', 'equipment', 'time_minutes', 'pantry_items']


def encoded(value):
    return quote(str(value), safe='')


def pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def read_json(text):
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError('Invalid JSON. Check commas, quotes and brackets before submitting.') from exc


def remember(key, value):
    st.session_state['_result_' + key] = value
    st.success('Request completed. The backend response is shown below.')


def result(key):
    value = st.session_state.get('_result_' + key)
    if value is not None:
        show_evidence(value, key)


def show_evidence(value, key):
    if isinstance(value, dict):
        if value.get('error'):
            st.error(str(value['error']))
        if value.get('status'):
            st.caption('Status: ' + str(value['status']))
        if value.get('phase'):
            st.caption('Current phase: ' + str(value['phase']).replace('-', ' '))
        for name, label in [('baseline', 'Before-training benchmark'), ('benchmark', 'After-training benchmark')]:
            benchmark = value.get(name)
            if isinstance(benchmark, dict) and isinstance(benchmark.get('rows'), list):
                st.caption(f"{label}: {len(benchmark['rows'])} cases recorded · {benchmark.get('status', 'in progress')}")
        history = value.get('history')
        if isinstance(history, list):
            points = [{k: row.get(k) for k in ['step', 'loss', 'eval_loss']} for row in history if isinstance(row, dict) and 'step' in row]
            if points:
                st.line_chart(points, x='step', y=['loss', 'eval_loss'])
        rows = value.get('rows') or value.get('candidates')
        if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
            st.dataframe(rows, hide_index=True)
    st.json(value, expanded=False)
    st.download_button('Download evidence JSON', pretty(value), file_name='recipetriage-evidence.json', mime='application/json', key=key + '_download')


def json_action(api, title, path, initial, key, *, method='POST', help_text=None):
    with st.form(key):
        if help_text:
            st.caption(help_text)
        body = st.text_area('Configuration JSON', pretty(initial), height=240, key=key + '_json')
        submitted = st.form_submit_button(title, disabled=api.embedded and path != '/api/v1/deployment/select')
    if submitted:
        try:
            remember(key, api.request(method, path, read_json(body)))
        except APIError as exc:
            # Keep rendering saved progress after a rejected job submission.
            st.error(str(exc))
    result(key)


def saved_runs(api, path, key):
    st.subheader('Saved runs')
    refresh = st.radio('Refresh saved runs', ['Manual'] if api.embedded else ['Manual', 'Every 5 seconds'], horizontal=True, key=key + '_refresh_mode')

    @st.fragment(run_every=5 if refresh == 'Every 5 seconds' else None)
    def render():
        try:
            st.button('Refresh now', key=key + '_refresh')
            rows = api.get(path)
            if not rows:
                st.info('No saved runs yet.')
                return
            def caption(row):
                identity = row.get('run_id') or row.get('experiment_id') or row.get('source_run_id', '')
                return f"{row.get('name') or row.get('kind') or 'Run'} · {row.get('status', 'saved')} · {identity[:8]}"
            selected = st.selectbox('Saved result', range(len(rows)), format_func=lambda i: caption(rows[i]), key=key + '_selected')
            row = rows[selected]
            if path == '/api/v1/training/runs':
                row = api.get(path + '/' + encoded(row['run_id']))
            if isinstance(row.get('runs'), list) and row['runs']:
                fields = ['method', 'learning_rate', 'rank', 'status', 'trainable_parameters', 'training_seconds', 'peak_rss_mib', 'micro_f1', 'exact_match', 'schema_validity']
                st.dataframe([{k: x.get(k) for k in fields} for x in row['runs']], hide_index=True)
            show_evidence(row, key + '_evidence')
        except APIError as exc:
            st.error(str(exc))
    render()


def recipe_inputs(initial, key):
    title = st.text_input('Title', initial.get('title', ''), max_chars=2000, key=key + '_title')
    ingredients = st.text_area('Ingredients — one per line', '\n'.join(initial.get('ingredients', [])), key=key + '_ingredients')
    instructions = st.text_area('Instructions — one step per line', '\n'.join(initial.get('instructions', [])), key=key + '_instructions')
    equipment = st.text_area('Equipment — one item per line', '\n'.join(initial.get('equipment', [])), key=key + '_equipment')
    minutes = st.number_input('Total elapsed minutes — 0 means unknown', min_value=0, max_value=10080,
                              value=initial.get('time_minutes') or 0, key=key + '_minutes')
    pantry_status = st.radio('Pantry inventory', ['Unknown', 'Known'],
                             index=1 if initial.get('pantry_items') is not None else 0,
                             horizontal=True, key=key + '_pantry_status')
    pantry = st.text_area('Pantry items — one per line', '\n'.join(initial.get('pantry_items') or []), key=key + '_pantry')
    lines = lambda text: [line.strip() for line in text.splitlines() if line.strip()]
    return dict(title=title.strip(), ingredients=lines(ingredients), instructions=lines(instructions),
                equipment=lines(equipment), time_minutes=int(minutes) or None,
                pantry_items=lines(pantry) if pantry_status == 'Known' else None)


def show_recipe(recipe):
    st.subheader(recipe['title'])
    st.caption(f"Total elapsed time: {recipe.get('time_minutes') or 'unknown'} minutes · Equipment: {', '.join(recipe.get('equipment', [])) or 'not specified'}")
    ingredients, instructions = st.columns([1, 2])
    with ingredients:
        st.write('**Ingredients**')
        for item in recipe['ingredients']:
            st.write('• ' + item)
    with instructions:
        st.write('**Instructions**')
        for index, step in enumerate(recipe['instructions'], 1):
            st.write(f'{index}. {step}')
    with st.expander('Recipe fields and source details'):
        st.json(recipe)


def model_choices(api):
    rows = api.get('/api/v1/playground/models')['models']
    available = {x['key']: x['title'] for x in rows if x['available']}
    with st.expander('Model availability'):
        if api.embedded:
            st.info('Model inference is disabled in this free cloud deployment. Use the local application to run models. Recipe editing, human review and dataset tools remain available here.')
        st.dataframe([{'Model': x['title'], 'Status': 'Available' if x['available'] else 'Unavailable',
                       **({} if api.embedded else {'Details': '' if x['available'] else x.get('reason', '')})}
                      for x in rows], hide_index=True)
    return available


def recipes(api, view):
    models = model_choices(api)
    if view == 'Add recipe':
        if api.embedded:
            st.info('Enter the recipe fields directly. This free profile validates your input without calling a paid extraction model.')
            with st.form('manual_intake'):
                recipe = recipe_inputs({}, 'manual_intake')
                submit = st.form_submit_button('Create recipe draft')
            if submit:
                st.session_state['_recipe_draft'] = api.post('/api/v1/recipes/intake', {'kind': 'text', 'content': pretty(recipe)})
        else:
            with st.form('intake'):
                kind = st.selectbox('Source', ['text', 'message', 'url', 'screenshot'])
                content = st.text_area('Recipe text, message or public URL', max_chars=24000)
                upload = st.file_uploader('Screenshot (PNG, JPEG or WebP; at most 4 MB)', type=['png', 'jpg', 'jpeg', 'webp'])
                submit = st.form_submit_button('Create recipe draft')
            if submit:
                if kind == 'screenshot':
                    if upload is None or upload.size > 4 * 1024 * 1024:
                        raise ValueError('Choose a screenshot no larger than 4 MB.')
                    content = base64.b64encode(upload.getvalue()).decode()
                draft = api.post('/api/v1/recipes/intake', {'kind': kind, 'content': content})
                st.session_state['_recipe_draft'] = draft
        draft = st.session_state.get('_recipe_draft')
        if draft:
            st.subheader('Check the normalized recipe')
            with st.form('save_' + draft['draft_id']):
                recipe = recipe_inputs(draft['recipe'], draft['draft_id'])
                save_mode = st.radio('After saving', ['Save only'] if api.embedded else ['Save only', 'Save and run triage'], horizontal=True)
                model = st.selectbox('Triage model', list(models), format_func=models.get, disabled=api.embedded)
                save = st.form_submit_button('Save recipe')
            if save:
                remember('saved_recipe', api.post('/api/v1/recipes', {'draft_id': draft['draft_id'], 'recipe': recipe, 'auto_triage': save_mode == 'Save and run triage', 'model_key': model or 'fireworks'}))
                del st.session_state['_recipe_draft']
            result('saved_recipe')
        return
    st.button('Refresh library')
    data = api.get('/api/v1/recipes')
    search = st.text_input('Search recipes, ingredients or equipment').casefold()
    selected_labels = st.multiselect('Filter labels (all selected labels must match)', LABELS)
    rows = [row for row in data['examples'] if search in pretty(row['recipe']).casefold()
            and set(selected_labels).issubset(row['labels'])]
    st.caption(f"{len(rows)} matching recipes · {sum(x['reviewed'] for x in data['examples'])} reviewed")
    if not rows:
        st.info('No matching recipes. Add a recipe or change the filters.')
        return
    selection = st.selectbox('Recipe', range(len(rows)), format_func=lambda i: rows[i]['recipe']['title'])
    row = rows[selection]
    identity = row['library_id']; revision = row['revision']; key = identity + '_' + str(revision)
    st.write('Labels: ' + (', '.join(row['labels']) or 'No labels yet'))
    st.write(row['rationale'])
    st.caption('Human verified' if row['reviewed'] else 'Unreviewed')
    if row.get('triage_job'):
        st.json(row['triage_job'])
    show_recipe(row['recipe'])
    with st.expander('Run triage'):
        with st.form('triage_' + key):
            model = st.selectbox('Model', list(models), format_func=models.get, key=key + '_model')
            submit = st.form_submit_button('Run triage for this revision', disabled=api.embedded)
        if submit:
            remember('library_triage', api.post(f'/api/v1/recipes/{encoded(identity)}/triage', {'revision': revision, 'model_key': model}))
        result('library_triage')
    with st.expander('Correct and verify labels'):
        with st.form('review_' + key):
            labels = st.multiselect('Correct labels', LABELS, row['labels'], key=key + '_labels')
            reviewer = st.text_input('Your reviewer name', key=key + '_reviewer')
            rationale = st.text_area('Evidence supporting these labels', row['rationale'], key=key + '_rationale')
            submit = st.form_submit_button('Save my verified correction')
        if submit:
            remember('library_review', api.post(f'/api/v1/recipes/{encoded(identity)}/review', dict(revision=revision, labels=labels, reviewer=reviewer, rationale=rationale)))
        result('library_review')
    with st.expander('Edit recipe content'):
        with st.form('edit_' + key):
            recipe = recipe_inputs(row['recipe'], 'edit_' + key)
            actor = st.text_input('Your name', key='actor_' + key)
            submit = st.form_submit_button('Save a new recipe revision')
        if submit:
            remember('recipe_edit', api.request('PUT', f'/api/v1/recipes/{encoded(identity)}', dict(revision=revision, recipe=recipe, actor=actor)))
        result('recipe_edit')
    if st.button('Load prediction and review history'):
        remember('recipe_history', api.get(f'/api/v1/recipes/{encoded(identity)}/history'))
    result('recipe_history')
    if st.button('Build dataset from verified library recipes'):
        remember('library_dataset', api.post('/api/v1/recipes/dataset-version'))
    result('library_dataset')


def dataset_lab(api, view):
    if view == 'Synthetic Review':
        return synthetic_review(api)
    seed = api.get('/api/v1/datasets/seed')
    st.session_state.setdefault('_dataset_raw', pretty(seed['examples']))
    if view == 'Label Editor':
        rows = read_json(st.session_state['_dataset_raw'])
        i = st.selectbox('Example to edit', range(len(rows)), format_func=lambda i: rows[i]['recipe']['title'])
        row = rows[i]
        st.json(row['recipe'])
        with st.form('label_editor_' + str(i)):
            labels = st.multiselect('Labels', LABELS, row.get('labels', []))
            rationale = st.text_area('Rationale', row.get('rationale', ''))
            save = st.form_submit_button('Update draft annotation')
        if save:
            # Editing annotations invalidates any prior human-review claim.
            rows[i] = {**row, 'labels': labels, 'rationale': rationale, 'reviewed': False, 'reviewed_by': None}
            st.session_state['_dataset_raw'] = pretty(rows)
            st.success('Draft updated for this session. Validate and save a dataset version in Dataset Studio.')
        st.caption('Use Synthetic Review for generated records; draft edits do not approve data.')
        return
    if view == 'Chat Template Preview':
        rows = read_json(st.session_state['_dataset_raw'])
        i = st.selectbox('Example', range(len(rows)), format_func=lambda i: rows[i]['recipe']['title'])
        st.caption('Preview includes the training conversation and the local model’s formatted chat template.')
        if st.button('Preview chat template'):
            st.session_state['_result_chat_preview'] = api.post('/api/v1/datasets/chat-preview', {'example': rows[i], 'render_model': True})
            st.success('Chat preview ready. Read the messages and formatted text below.')
        preview = st.session_state.get('_result_chat_preview')
        if preview is not None:
            st.subheader('Training conversation')
            st.caption('The assistant message is the expected answer from your dataset. This preview formats the example without training or generating a prediction.')
            role_names = {'system': 'System — RecipeTriage rules', 'user': 'User — recipe input', 'assistant': 'Assistant — expected answer'}
            for message in preview.get('messages', []):
                st.markdown('**' + role_names.get(message['role'], message['role']) + '**')
                content = message['content']
                st.code(content, language='json' if content.lstrip().startswith(('{', '[')) else 'text', wrap_lines=True)
            if preview.get('rendered_text'):
                st.subheader('Formatted model chat template')
                st.caption('The special markers identify the roles and message boundaries expected by the tokenizer.')
                st.code(preview['rendered_text'], language='text', wrap_lines=True)
            else:
                st.info('Click Preview chat template to include the model-specific formatting in this saved preview.')
            with st.expander('Raw JSON and model details'):
                st.json(preview)
            st.download_button('Download evidence JSON', pretty(preview), file_name='recipetriage-chat-preview.json', mime='application/json', key='chat_preview_download')
        return
    if view == 'Dataset Studio':
        with st.expander('Label definitions'):
            st.json(seed['policy'])
        with st.form('dataset_studio'):
            raw = st.text_area('Raw dataset', st.session_state['_dataset_raw'], height=320)
            fmt = st.selectbox('Input format', ['json', 'jsonl'])
            split_seed = st.number_input('Split seed', 0, 4294967295, 42)
            ratios = st.text_input('Train, validation, test fractions', '0.7, 0.15, 0.15')
            action = st.selectbox('Action', ['Validate and clean', 'Preview split and export', 'Save dataset version'])
            submit = st.form_submit_button('Apply dataset action')
        if submit:
            body = dict(raw=raw, format=fmt, seed=int(split_seed), ratios=[float(x.strip()) for x in ratios.split(',')])
            operation = {'Validate and clean': 'validate', 'Preview split and export': 'preview', 'Save dataset version': 'versions'}[action]
            response = api.post('/api/v1/datasets/' + operation, body)
            st.session_state['_dataset_raw'] = pretty(response.get('drafts') or response.get('examples') or read_json(raw))
            if 'files' in response:
                st.session_state['_dataset_export'] = response
            remember('dataset', response)
        result('dataset')
    versions = api.get('/api/v1/datasets/versions')
    if versions:
        selected = st.selectbox('Saved dataset version', [x['version'] for x in versions])
        if st.button('Load saved dataset'):
            st.session_state['_dataset_export'] = api.get('/api/v1/datasets/versions/' + encoded(selected))
    snapshot = st.session_state.get('_dataset_export')
    if snapshot:
        st.json(snapshot['metadata'], expanded=False)
        name = st.selectbox('Export file', list(snapshot['files']))
        st.code(snapshot['files'][name], language='json')
        st.download_button('Download selected file', snapshot['files'][name], file_name=name)
        import io
        from zipfile import ZipFile, ZIP_DEFLATED
        archive = io.BytesIO()
        with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
            for filename, content in snapshot['files'].items():
                output.writestr(filename, content)
            output.writestr('metadata.json', pretty(snapshot['metadata']))
        st.download_button('Download complete dataset ZIP', archive.getvalue(), file_name='recipetriage-' + snapshot['metadata']['version'] + '.zip', mime='application/zip')
    elif view == 'JSONL Preview':
        st.info('Preview a split in Dataset Studio or load a saved version first.')


def synthetic_review(api):
    st.caption('Edits remain pending. Approval requires your explicit decision on the displayed revision.')
    if api.embedded:
        st.info('Review existing candidates or queue the original seed recipes. Fireworks generation is disabled in this free profile.')
    notice = st.session_state.pop('_candidate_review_notice', None)
    if notice:
        st.success(notice)
    with st.expander('Generate or prepare review candidates'):
        with st.form('synthetic_generate'):
            count = st.number_input('Number of Fireworks proposals', 1, 6, 6)
            submit = st.form_submit_button('Generate proposals with Fireworks', disabled=api.embedded)
        if submit:
            remember('synthetic_generation', api.post('/api/v1/curation/generate', {'count': count}))
        if st.button('Queue original seed recipes for review'):
            remember('seed_review', api.post('/api/v1/curation/seed-review'))
        result('synthetic_generation')
    st.button('Refresh review queue')
    rows = api.get('/api/v1/curation/candidates')
    if not rows:
        st.info('The review queue is empty.')
        return
    def candidate_title(row):
        draft = row.get('draft')
        recipe = draft.get('recipe') if isinstance(draft, dict) else None
        title = recipe.get('title') if isinstance(recipe, dict) else None
        return f"{title or 'Failed generation'} · {row['status']}"
    i = st.selectbox('Candidate', range(len(rows)), format_func=lambda i: candidate_title(rows[i]))
    row = rows[i]; key = row['candidate_id'] + '_' + str(row['revision'])
    st.caption(f"Revision {row['revision']} · {row.get('category') or 'Original seed'}")
    quality = row.get('quality') or {}
    if quality.get('passed'):
        st.info('Automated checks passed. Review the recipe and labels before deciding whether to approve.')
    else:
        st.error('Approval is blocked by quality checks. Save corrections with Edit, or choose Reject.')
        for error in quality.get('errors', []):
            st.error(error)
        if row.get('category') == 'misleading-title':
            st.info('This test category needs a title beginning with Quick, Easy or Simple and a total time over 30 minutes. The title should sound easy or fast while the recipe facts support the labels. Keep the actual time and recipe details accurate.')
    st.json(row.get('draft') or {}, expanded=True)
    with st.expander('Quality check details and warnings'):
        st.json(quality)
    with st.expander('Provenance and review history'):
        st.json(row)
    if row['status'] != 'approved':
        with st.form('candidate_' + key):
            decisions = ['approve', 'reject', 'edit'] if quality.get('passed') else ['edit', 'reject']
            action = st.selectbox('Your decision', decisions, index=None, placeholder='Choose a decision')
            reviewer = st.text_input('Reviewer name')
            notes = st.text_area('Review notes')
            draft = st.text_area('Edited draft JSON — used only for Edit', pretty(row.get('draft') or {}), height=260)
            submit = st.form_submit_button('Record my decision')
        if submit:
            if action is None:
                raise ValueError('Choose Approve, Reject or Edit explicitly.')
            body = dict(revision=row['revision'], action=action, reviewer=reviewer, notes=notes)
            if action == 'edit':
                body['draft'] = read_json(draft)
            reviewed = api.post(f"/api/v1/curation/candidates/{encoded(row['candidate_id'])}/review", body)
            if action == 'edit':
                notice = f"Edits saved as revision {reviewed['revision']}. The example is still pending; inspect the updated checks before approving."
            else:
                notice = f"Your {action} decision was saved for revision {reviewed['revision']}."
            st.session_state['_candidate_review_notice'] = notice
            # Reload the saved revision before another human decision, so Edit
            # followed by Approve cannot accidentally submit the old revision.
            st.rerun()
    if st.button('Build improved dataset from approved records'):
        remember('approved_dataset', api.post('/api/v1/curation/build-version'))
    result('approved_dataset')


def training_lab(api, view):
    options = api.get('/api/v1/training/options')
    default = options['defaults']
    if api.embedded:
        st.info('Training runs in your local deployment. Here you can inspect saved evidence and prepare a local training configuration.')
        with st.expander('Export a configuration for local training'):
            versions = api.get('/api/v1/datasets/versions')
            if versions:
                version = st.selectbox('Dataset version for local training', [row['version'] for row in versions])
                method = st.selectbox('Local training method', ['sft', 'lora', 'qlora'])
                config = {**default, 'dataset_version': version, 'provider': 'local', 'method': method}
                st.download_button('Download local training config', pretty(config), file_name='training-config.json', mime='application/json')
                st.caption('Download the matching complete dataset ZIP in Data Lab. Run the training command from STREAMLIT_FREE.md on your Mac.')
            else:
                st.info('Save a dataset version in Data Lab first.')
    else:
        st.caption('Jobs run on the backend. Opening this page starts no training. Benchmark evidence is saved after training.')
    if view == 'SFT Monitor':
        with st.form('sft'):
            provider = st.selectbox('Training provider', ['local', 'fireworks'])
            raw = st.text_area('Training parameters JSON', pretty(default), height=300)
            action = st.selectbox('Action', ['Check configuration', 'Start training'])
            st.caption('Check configuration validates settings for the selected provider. Start training submits a separate job.')
            submit = st.form_submit_button('Submit training action', disabled=api.embedded)
        if submit:
            config = {**read_json(raw), 'provider': provider}
            if action == 'Check configuration':
                st.session_state.pop('_sft_configuration_check', None)
                check = api.post('/api/v1/training/preflight', config)
                st.session_state['_sft_configuration_check'] = {'provider': provider, 'raw': raw, 'response': check}
            else:
                try:
                    remember('sft', api.post('/api/v1/training/runs', config))
                except APIError as exc:
                    st.error(str(exc))
        checked = st.session_state.get('_sft_configuration_check')
        if checked and checked['provider'] == provider and checked['raw'] == raw:
            check = checked['response']
            if check.get('supported'):
                st.success(provider.capitalize() + ' configuration check passed.')
            else:
                st.error('Configuration needs changes before training.')
            for issue in check.get('issues', []):
                st.error(issue)
            counts = check.get('counts', {})
            if counts:
                st.caption(f"Dataset: {counts.get('train', 0)} train / {counts.get('validation', 0)} validation / {counts.get('test', 0)} test examples.")
            if check.get('dataset_status') == 'teaching-draft':
                st.warning('This dataset contains unreviewed teaching examples. Its results are for learning.')
            if check.get('deferred_checks'):
                st.caption('Training will still check: ' + '; '.join(check['deferred_checks']) + '.')
            with st.expander('Configuration check details'):
                show_evidence(check, 'sft_configuration_check')
        result('sft')
        saved_runs(api, '/api/v1/training/runs', 'training')
        with st.expander('Resume a saved training run'):
            with st.form('resume'):
                identifier = st.text_input('Training run ID')
                resume = st.form_submit_button('Resume run', disabled=api.embedded)
            if resume:
                remember('resume', api.post('/api/v1/training/runs/' + encoded(identifier) + '/resume'))
            result('resume')
    elif view == 'LoRA Training':
        with st.expander('Inspected model architecture and valid targets'):
            st.json(api.get('/api/v1/training/lora/architecture'), expanded=False)
        json_action(api, 'Start rank comparison', '/api/v1/training/lora/experiments', {'training': default, 'ranks': [4, 16]}, 'lora', help_text='Every rank uses the same dataset and training budget. Edit training.lora_layers for a layer-subset experiment.')
        saved_runs(api, '/api/v1/training/lora/experiments', 'lora_runs')
    else:
        config = api.get('/api/v1/experiments/options')
        if view == 'QLoRA Training':
            st.json(config['qlora'])
            runs = [{**default, 'method': method, 'lora_rank': 4} for method in ['lora', 'qlora']]
            initial = {'name': 'Matched LoRA and QLoRA', 'runs': runs, 'mlflow': True}
        else:
            initial = config['defaults']
        json_action(api, 'Start experiment matrix', '/api/v1/experiments', initial, 'matrix')
        saved_runs(api, '/api/v1/experiments', 'experiments')
    with st.expander('Parameter definitions and resolved training objective'):
        st.json(options['parameters'])


def alignment_lab(api, view):
    options = api.get('/api/v1/alignment/options')
    if api.embedded:
        st.info('Explicit choices for existing pairs are saved here. Candidate generation and alignment training run in the local deployment.')
    if view == 'Preference Pairs':
        if st.button('Generate candidate pairs', disabled=api.embedded):
            remember('pair_generation', api.post('/api/v1/alignment/pairs/generate'))
        result('pair_generation')
        st.button('Refresh pairs')
        rows = api.get('/api/v1/alignment/pairs')
        if rows:
            i = st.selectbox('Preference recipe', range(len(rows)), format_func=lambda i: f"{rows[i]['recipe']['title']} · {rows[i]['status']}")
            row = rows[i]
            show_recipe(row['recipe'])
            for column, name in zip(st.columns(2), ['a', 'b']):
                with column:
                    candidate = row['candidates'][name]
                    st.subheader('Candidate ' + name.upper())
                    st.code(candidate['raw_output'], language='json')
                    st.caption(f"{candidate['provider']} · {candidate['finish_reason']}")
            if row['decision'] is None:
                with st.form('choice_' + row['pair_id']):
                    choice = st.radio('Your preference', ['a', 'b', 'tie', 'neither'], index=None, horizontal=True)
                    reviewer = st.text_input('Your reviewer name')
                    notes = st.text_area('Why did you choose this?')
                    submit = st.form_submit_button('Save my explicit preference')
                if submit:
                    if choice is None:
                        raise ValueError('Choose A, B, Tie or Neither explicitly.')
                    remember('choice', api.post(f"/api/v1/alignment/pairs/{encoded(row['pair_id'])}/choice", dict(choice=choice, reviewer=reviewer, notes=notes, pair_sha256=row['pair_sha256'])))
                result('choice')
            else:
                st.json(row['decision'])
            st.download_button('Download preference records', pretty(rows), file_name='preference-pairs.json')
        else:
            st.info('No preference pairs yet.')
    elif view == 'Reviewed QLoRA':
        versions = [x for x in api.get('/api/v1/datasets/versions') if x.get('status') == 'reviewed' and x.get('holdouts_frozen')]
        if not versions:
            st.info('Build an improved dataset from explicit human approvals in Synthetic Review first.')
        else:
            with st.form('retrain'):
                version = st.selectbox('Approved dataset version', [x['version'] for x in versions])
                submit = st.form_submit_button('Retrain QLoRA and evaluate shortcut accuracy', disabled=api.embedded)
            if submit:
                remember('retrain', api.post('/api/v1/alignment/retrain-approved', {'dataset_version': version}))
            result('retrain')
    else:
        method = 'dpo' if view == 'DPO Training' else 'grpo'
        st.caption('Human A/B decisions are required for DPO. Ties and Neither are retained but excluded from DPO training.')
        if method == 'grpo':
            st.write('Reward components')
            st.json(options['reward_weights'])
            st.caption('Small diagnostic experiment. It cannot automatically change the production model.')
        json_action(api, 'Start ' + method.upper() + ' experiment', '/api/v1/alignment/train/' + method, options['defaults'], method)
    saved_runs(api, '/api/v1/alignment/jobs', 'alignment_jobs')


def evaluation_lab(api, view):
    if api.embedded:
        st.info('Inspect saved benchmark results and analyze their failures here. New model benchmarks and shortcut runs require the local deployment.')
    if view == 'Baseline':
        with st.expander('Fixed benchmark and provider availability'):
            st.json(api.get('/api/v1/benchmarks/manifest'))
            st.json(api.get('/api/v1/benchmarks/providers'))
        with st.form('benchmark'):
            provider = st.selectbox('Provider', ['hf-base', 'fireworks'])
            tokens = st.number_input('Maximum new tokens', 1, 256, 128)
            submit = st.form_submit_button('Run fixed benchmark', disabled=api.embedded)
        if submit:
            remember('benchmark', api.post('/api/v1/benchmarks/runs', dict(provider=provider, temperature=0, max_new_tokens=tokens, reasoning='disabled')))
        result('benchmark')
        rows = api.get('/api/v1/benchmarks/runs')
        table = []
        for row in rows:
            summary = row.get('summary') or {}; labels = summary.get('labels') or {}
            table.append({'Run': row['run_id'], 'Model': row.get('model_config', {}).get('model'), 'Status': row['status'],
                          'Macro F1': labels.get('macro', {}).get('f1'), 'Micro F1': labels.get('micro', {}).get('f1'),
                          'Exact match': labels.get('exact_match'), 'JSON validity': summary.get('json_validity')})
        if table:
            st.dataframe(table, hide_index=True)
        saved_runs(api, '/api/v1/benchmarks/runs', 'benchmark_runs')
    elif view == 'Failure Analysis':
        sources = api.get('/api/v1/analysis/sources')
        if sources:
            with st.form('failures'):
                source = st.selectbox('Benchmark source', [x['run_id'] for x in sources])
                submit = st.form_submit_button('Analyze failures')
            if submit:
                remember('failures', api.post('/api/v1/analysis/failures', {'source_run_id': source}))
            result('failures')
            if st.button('Load collection priorities for selected source'):
                remember('priorities', api.get('/api/v1/analysis/collection-priorities/' + encoded(source)))
            result('priorities')
        saved_runs(api, '/api/v1/analysis/failures', 'failure_history')
    else:
        st.write('Title adjectives change; time, ingredients, equipment and instructions remain fixed.')
        st.caption('Inspect prediction-flip rate, misleading-title accuracy, pair coverage and red-team results together.')
        if st.button('Run shortcut and red-team diagnostics', disabled=api.embedded):
            remember('diagnostics', api.post('/api/v1/analysis/diagnostics'))
        result('diagnostics')
        saved_runs(api, '/api/v1/analysis/diagnostics', 'diagnostic_runs')


def playground(api, view):
    if view == 'Token Inspector':
        with st.form('tokens'):
            text = st.text_area('Text to tokenize', 'Easy traditional ravioli', max_chars=10000)
            token_format = st.radio('Tokenization format', ['Plain text', 'Chat template'], horizontal=True)
            submit = st.form_submit_button('Inspect tokens')
        if submit:
            remember('tokens', api.post('/api/v1/tokens', {'text': text, 'chat_template': token_format == 'Chat template'}))
        value = st.session_state.get('_result_tokens')
        if value:
            st.metric('Token count', value['token_count'])
            st.dataframe([{'Token': token, 'Token ID': identifier} for token, identifier in zip(value['tokens'], value['token_ids'])], hide_index=True)
            st.code(value['rendered_text'])
        return
    if api.embedded:
        st.info('Model generation is disabled in this free profile. Token Inspector uses only the tokenizer; saved comparisons remain available.')
    seed = api.get('/api/v1/datasets/seed')['examples']
    selected = st.selectbox('Start from a seed recipe', range(len(seed)), format_func=lambda i: seed[i]['recipe']['title'])
    models = model_choices(api)
    with st.form('playground_' + view):
        recipe = recipe_inputs(seed[selected]['recipe'], 'play_' + view + str(selected))
        if view == 'Triage':
            provider = st.selectbox('Provider', ['fireworks', 'hf', 'production'])
        else:
            selected_models = st.multiselect('Models to compare', list(models), default=list(models)[:2], format_func=models.get)
        temperature = st.slider('Temperature', 0.0, 2.0, 0.0, 0.1)
        tokens = st.number_input('Maximum new tokens', 1, 256, 128)
        submit = st.form_submit_button('Run inference' if view == 'Triage' else 'Start model comparison', disabled=api.embedded)
    if submit:
        body = dict(recipe=recipe, temperature=temperature, max_new_tokens=tokens)
        if view == 'Triage':
            remember('playground', api.post('/api/v1/triage', {**body, 'provider': provider}))
        else:
            remember('playground', api.post('/api/v1/playground/comparisons', {**body, 'models': selected_models}))
    result('playground')
    if view == 'Model Comparison':
        saved_runs(api, '/api/v1/playground/comparisons', 'comparisons')


def deployment(api, view):
    if api.embedded:
        st.info('Compare saved measurements here. Promotion and rollback require the local deployment where model artifacts are stored and verified.')
    if view == 'Measured Candidates':
        data = api.get('/api/v1/deployment/results')
        st.dataframe(data['candidates'], hide_index=True)
        st.caption('Historical measurements; selection does not promote or deploy a model. Merge and quantization commands run on the backend host; see DEPLOYMENT.md.')
        json_action(api, 'Compare candidates against constraints', '/api/v1/deployment/select', data['constraints'], 'deployment_selection')
    else:
        st.subheader('Evaluation quality gates')
        st.json(api.get('/api/v1/models/gate-config'))
        models = api.get('/api/v1/models')
        if models:
            selected = st.selectbox('Registered model', range(len(models)), format_func=lambda i: f"{models[i]['name']} · {models[i]['stage']}")
            model = models[selected]
            st.json(model, expanded=False)
            with st.form('stage_' + model['model_id']):
                action = st.selectbox('Registry action', ['staging', 'production', 'archived', 'rollback'], index=None)
                actor = st.text_input('Your name')
                reason = st.text_area('Reason for this change')
                submit = st.form_submit_button('Apply stage change subject to quality gates', disabled=api.embedded)
            if submit:
                if action is None:
                    raise ValueError('Choose a stage change explicitly.')
                path = '/api/v1/models/' + encoded(model['model_id']) + ('/rollback' if action == 'rollback' else '/stage')
                remember('model_stage', api.post(path, dict(stage='production' if action == 'rollback' else action, actor=actor, reason=reason)))
            result('model_stage')
        else:
            st.info('No registered model artifacts on this backend yet.')
        with st.expander('Promotion and rollback history'):
            st.json(api.get('/api/v1/models/history'))


def main(default_mode=None):
    st.set_page_config(page_title='RecipeTriage AI', page_icon='🍲', layout='wide')
    backend, links = configuration(default_mode=default_mode)
    st.title('RecipeTriage AI')
    if not backend.get('url') and backend.get('transport') != 'embedded':
        st.info('Configure backend.url and backend.token in Streamlit secrets. See STREAMLIT_DEPLOYMENT.md.')
        st.stop()
    try:
        if backend.get('transport') == 'embedded':
            from .embedded import embedded_api
            api = embedded_api(backend.get('database_url', ''))
            st.caption('Streamlit runs the application logic directly · PostgreSQL stores saved records · No paid provider calls')
        else:
            api = API(backend['url'], backend.get('token', ''), docker_network=backend.get('transport') == 'docker')
        with st.container(horizontal=True):
            if st.button('Check backend connection'):
                st.success('Connected: ' + str(api.get('/health')['status']))
            mlflow = links.get('mlflow', '')
            if mlflow and urlsplit(mlflow).scheme == 'https':
                st.link_button('Open MLflow', mlflow)
        renderers = dict(zip(SECTIONS, [recipes, dataset_lab, training_lab, alignment_lab, evaluation_lab, playground, deployment]))
        # Stateful tabs render only the active page: hidden labs do not load
        # models or poll the API. Drafts and results remain in session state.
        for section, tab in zip(SECTIONS, st.tabs(SECTIONS, key='workspace', on_change='rerun')):
            if tab.open:
                with tab:
                    st.header(section)
                    labels = PAGES[section]
                    for page, panel in zip(labels, st.tabs(labels, key='page_' + section, on_change='rerun')):
                        if panel.open:
                            with panel:
                                renderers[section](api, page)
    except (APIError, ValueError) as exc:
        st.error(str(exc))
