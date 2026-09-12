import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import seed from '../../ml/data/seed.json';
import App from './App';
import Recipes from './Recipes';
import RecipeIntake from './RecipeIntake';
import RecipeActions from './RecipeActions';
import ModelComparison from './ModelComparison';

const fields = ['title', 'ingredients', 'instructions', 'equipment', 'time_minutes', 'pantry_items'];
const input = Object.fromEntries(fields.map(key => [key, seed[0].recipe[key]]));
const examples = seed.map((item, index) => ({ ...item, library_id: 'library-' + index, revision: 1, prediction: null, review: null }));
const labels = ['weeknight-30min', 'weekend-project', 'needs-special-equipment', 'meal-prep', 'dessert', 'have-most-of-this', 'unclear'];
const modelRows = [{ key: 'fireworks', title: 'Fireworks', model: 'hosted-fixture', available: true }, { key: 'base', title: 'Local base', model: 'local-fixture', available: true }, { key: 'dpo', title: 'DPO', available: false, reason: 'No completed DPO checkpoint' }];
const response = data => ({ ok: true, json: async () => data });
afterEach(() => vi.unstubAllGlobals());
function mock(extra) {
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    const custom = extra?.(url, options); if (custom) return custom;
    if (url === '/api/v1/recipes') return response({ examples, labels, policy: Object.fromEntries(labels.map(label => [label, 'Check the recipe evidence.'])) });
    if (url.endsWith('datasets/versions') || url.endsWith('playground/comparisons')) return response([]);
    if (url.endsWith('playground/models')) return response({ models: modelRows, default_models: ['fireworks', 'base'] });
    if (url.endsWith('learning-example')) return response({ recipe: input });
    throw new Error('Unexpected request ' + url);
  }));
}

it('browses saved recipes, searches ingredients and shows an explicit empty state', async () => {
  mock(); const user = userEvent.setup(); render(<Recipes onTryRecipe={vi.fn()} onManageDatasets={vi.fn()} />);
  await screen.findByText('7 of 7 recipes');
  await user.type(screen.getByRole('textbox', { name: 'Search recipes' }), 'ricotta');
  expect(screen.getByText('1 of 7 recipes')).toBeInTheDocument();
  expect(within(screen.getByRole('article', { name: 'Recipe details' })).getByRole('heading', { name: 'Homemade Cheese Ravioli' })).toBeInTheDocument();
  await user.clear(screen.getByRole('textbox', { name: 'Search recipes' }));
  await user.type(screen.getByRole('textbox', { name: 'Search recipes' }), 'no such ingredient');
  expect(screen.getByText('No recipes match your search and label filter.')).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Clear filters' }));
  expect(screen.getByText('7 of 7 recipes')).toBeInTheDocument();
  expect(fetch.mock.calls.every(([, options]) => !options.method)).toBe(true);
});

it('opens the selected library recipe in Playground without leaking labels or fetching a default recipe', async () => {
  mock(); const user = userEvent.setup(); render(<App />);
  await user.click(await screen.findByRole('button', { name: 'Homemade Cheese Ravioli', exact: true }));
  await user.click(screen.getByRole('button', { name: 'Open in Playground' }));
  const editor = await screen.findByRole('textbox', { name: 'Comparison Recipe JSON' });
  const value = JSON.parse(editor.value);
  expect(value.title).toBe('Homemade Cheese Ravioli');
  expect(Object.keys(value).sort()).toEqual([...fields].sort());
  expect(fetch.mock.calls.some(([url]) => url.endsWith('learning-example'))).toBe(false);
  expect(fetch.mock.calls.every(([, options]) => !options.method)).toBe(true);
});

it('normalizes a pasted message, lets the user edit it, then explicitly saves with auto-triage', async () => {
  const onSaved = vi.fn();
  mock((url, options) => {
    if (url.endsWith('recipes/intake')) return response({ draft_id: 'intake-fixture', recipe: input, source_text: 'Fixture message', source: { kind: 'message' }, normalization: { method: 'fixture' } });
    if (url === '/api/v1/recipes' && options.method === 'POST') return response({ ...examples[0], already_saved: false });
  });
  const user = userEvent.setup(); render(<RecipeIntake onSaved={onSaved} onCancel={vi.fn()} />);
  await user.click(screen.getByRole('combobox', { name: 'Input type' }));
  await user.click(screen.getByRole('option', { name: 'Pasted message' }));
  await user.type(screen.getByRole('textbox', { name: 'Recipe text or message' }), 'Fixture message');
  await user.click(screen.getByRole('button', { name: 'Normalize recipe' }));
  const editor = await screen.findByRole('textbox', { name: 'Normalized Recipe JSON' });
  await user.clear(editor); await user.paste(JSON.stringify({ ...input, title: 'Edited oatmeal' }));
  await user.click(screen.getByRole('button', { name: 'Save recipe', exact: true }));
  expect(onSaved).toHaveBeenCalledOnce();
  const create = fetch.mock.calls.find(([url, options]) => url === '/api/v1/recipes' && options.method === 'POST');
  expect(JSON.parse(create[1].body)).toEqual({ draft_id: 'intake-fixture', recipe: { ...input, title: 'Edited oatmeal' }, auto_triage: true, model_key: 'fireworks' });
});

it('stores human corrections only after an explicit named review', async () => {
  const changed = vi.fn(); mock(url => url.endsWith('/review') ? response({ reviewed: true }) : null);
  const user = userEvent.setup(); render(<RecipeActions item={examples[0]} labels={labels} policy={{}} onChanged={changed} />);
  await user.click(screen.getByRole('button', { name: 'Review or correct labels' }));
  expect(screen.getByRole('button', { name: 'Verify corrected labels' })).toBeDisabled();
  expect(fetch.mock.calls.every(([, options]) => !options.method)).toBe(true);
  await user.type(screen.getByRole('textbox', { name: 'Reviewer name' }), 'Test fixture reviewer');
  await user.click(screen.getByRole('button', { name: 'Verify corrected labels' }));
  expect(changed).toHaveBeenCalledOnce();
  const body = JSON.parse(fetch.mock.calls.find(([url]) => url.endsWith('/review'))[1].body);
  expect(body).toMatchObject({ revision: 1, reviewer: 'Test fixture reviewer', labels: seed[0].labels });
});

it('retains independent model results and makes missing DPO explicit', async () => {
  const config = { recipe: input, models: ['fireworks', 'base'], temperature: 0, max_new_tokens: 128 };
  const output = { valid_json: true, prediction: { labels: ['meal-prep'], explanation: 'Fixture response.' }, raw_output: '{"labels":["meal-prep"]}', finish_reason: 'stop', latency_ms: 12, output_tokens: 8 };
  mock((url, options) => url.endsWith('playground/comparisons') && options.method === 'POST' ? response({ run_id: 'comparison-fixture', status: 'completed', config, result: { status: 'partial', rows: [{ key: 'fireworks', title: 'Fireworks', result: output }, { key: 'base', title: 'Local base', error: 'Fixture provider failed' }] } }) : null);
  const user = userEvent.setup(); render(<ModelComparison initialRecipe={input} />);
  expect(await screen.findByText('No completed DPO checkpoint')).toBeInTheDocument();
  expect(screen.getByRole('checkbox', { name: 'DPO', exact: true })).toBeDisabled();
  expect(fetch.mock.calls.every(([, options]) => !options.method)).toBe(true);
  await user.click(screen.getByRole('button', { name: 'Compare selected models' }));
  expect(await screen.findByText('Fixture provider failed')).toBeInTheDocument();
  expect(within(screen.getByRole('region', { name: 'Fireworks comparison result' })).getByText('Fixture response.')).toBeInTheDocument();
  expect(JSON.parse(fetch.mock.calls.find(([, options]) => options.method === 'POST')[1].body)).toEqual(config);
});
