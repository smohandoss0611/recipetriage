import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import ExperimentStudio from './ExperimentStudio';
import FailureAnalysis from './FailureAnalysis';
import ShortcutTests from './ShortcutTests';

const defaults = { name: 'test', runs: [{ provider: 'local', dataset_version: 'v1-' + 'a'.repeat(64), method: 'lora', epochs: 3, learning_rate: 0.00005, lora_rank: 4, lora_alpha: null, lora_dropout: 0, lora_target_policy: 'query-value', batch_size: 1, gradient_accumulation_steps: 2, sequence_length: 768, seed: 42, gradient_checkpointing: true }] };
function mock(responses) { vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => ({ ok: true, json: async () => responses(url, options) }))); }
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
it('submits a fixed learning-rate/rank matrix without launching on open', async () => {
  mock(url => url.endsWith('/options') ? { defaults, qlora: { supported: true, bitsandbytes: '0.50.2' } } : []);
  const user = userEvent.setup(); render(<ExperimentStudio />);
  const button = await screen.findByRole('button', { name: 'Run experiment matrix' });
  expect(fetch.mock.calls.every(([, options]) => options?.method !== 'POST')).toBe(true);
  await user.click(button);
  const config = JSON.parse(fetch.mock.calls.find(([, options]) => options?.method === 'POST')[1].body);
  expect(config.runs).toHaveLength(4);
  expect(new Set(config.runs.map(row => row.dataset_version)).size).toBe(1);
  expect(new Set(config.runs.map(row => row.learning_rate))).toEqual(new Set([0.00005, 0.0002]));
});
it('QLoRA compares matched settings and allows checkpointing to be disabled', async () => {
  mock(url => url.endsWith('/options') ? { defaults, qlora: { supported: true, bitsandbytes: '0.50.2' } } : []);
  const user = userEvent.setup(); render(<ExperimentStudio qlora />);
  const button = await screen.findByRole('button', { name: 'Run LoRA and QLoRA' });
  await user.click(screen.getByRole('checkbox', { name: 'Gradient checkpointing for every run' })); await user.click(button);
  const config = JSON.parse(fetch.mock.calls.find(([, options]) => options?.method === 'POST')[1].body);
  expect(config.runs.map(row => row.method)).toEqual(['lora', 'qlora']);
  expect(config.runs.every(row => !row.gradient_checkpointing && row.lora_rank === 4 && row.learning_rate === 0.0002)).toBe(true);
});
it('failure analysis retains malformed output and shows collection priorities', async () => {
  const analysis = { source_run_id: 'one', model_config: { training_stage: 'test' }, failed_cases: 1, total_cases: 1, category_counts: { schema_violation: 1 }, misleading_title_accuracy: 0, misleading_title_correct: 0, misleading_title_total: 1, collection_priorities: [], findings: [{ case_id: 'bench-003', title: 'Quick Bread', expected_labels: ['weekend-project'], predicted_labels: null, failure_categories: ['schema_violation'], raw_output: '{"explanation":"No labels"}', error: 'labels required' }] };
  const priorities = [{ priority: 1, topic: 'Complete structured answers', action: 'Collect reviewed JSON answers.', basis: 'observed failures', observed_cases: 1, case_ids: ['bench-003'], split_rule: 'Keep benchmark held out.' }];
  mock(url => url.endsWith('/failures') ? [analysis] : url.includes('collection-priorities') ? priorities : []);
  render(<FailureAnalysis />);
  expect(await screen.findByRole('table', { name: 'Failure cases' })).toHaveTextContent('Unusable / missing response');
  expect(await screen.findByText('1. Complete structured answers')).toBeInTheDocument();
});
it('undefined flip rate stays undefined when every pair is excluded', async () => {
  const run = { run_id: 'one', status: 'completed', model_config: { training_stage: 'base' }, shortcut: { rows: [], summary: { prediction_flip_rate: null, flipped_pairs: 0, eligible_pairs: 0, planned_pairs: 9, excluded_pairs: 9, pair_coverage: 0, misleading_title_accuracy: 0, misleading_correct: 0, misleading_total: 3, exact_set_accuracy: 0, usable_rows: 0, planned_rows: 12 } } };
  mock(() => [run]); render(<ShortcutTests />);
  expect(await screen.findByText('Prediction-flip rate: — (0/0 usable pairs).')).toBeInTheDocument();
  expect(screen.getByText(/9\/9 excluded pairs/)).toBeInTheDocument();
});
