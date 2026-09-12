import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import LoRATraining from './LoRATraining';

const defaults = { provider: 'local', dataset_version: 'v1-' + 'a'.repeat(64), epochs: 3, learning_rate: 0.0002, sequence_length: 768, batch_size: 1, gradient_accumulation_steps: 2, seed: 42, lora_rank: 8 };
const architecture = { model: 'Qwen/Qwen2.5-0.5B', layers: 24, base_parameters: 494032768, attention_heads: 14, kv_heads: 2,
  policies: { 'query-value': ['q_proj', 'v_proj'], attention: ['q_proj', 'k_proj', 'v_proj', 'o_proj'] },
  projections: [{ name: 'model.layers.0.self_attn.q_proj', role: 'q_proj', weight_shape: [896, 896], in_features: 896, out_features: 896, eligible: true }, { name: 'lm_head', role: 'lm_head', weight_shape: [151936, 896], eligible: false }] };
let history, reject;
beforeEach(() => {
  history = []; reject = false;
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    let value;
    if (url.endsWith('/architecture')) value = architecture;
    else if (url.endsWith('/options')) value = { defaults, seed_dataset: { version: defaults.dataset_version, counts: { train: 5, validation: 1, test: 1 }, status: 'teaching-draft' } };
    else if (url.endsWith('/datasets/versions')) value = [];
    else if (options.method === 'POST') value = reject ? { detail: 'Another training worker is active' } : { experiment_id: 'new' };
    else value = history;
    return { ok: !(options.method === 'POST' && reject), json: async () => value };
  }));
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });
it('shows real shape inventory and estimates without launching training', async () => {
  render(<LoRATraining />);
  expect(await screen.findByRole('button', { name: 'Run rank experiment' })).toBeEnabled();
  expect(screen.getByText('Excluded: tied output head')).toBeInTheDocument();
  expect(screen.getByText('896 × 896')).toBeInTheDocument();
  expect(fetch.mock.calls.every(([, options]) => options?.method !== 'POST')).toBe(true);
});
it('requires two ranks and submits a controlled local experiment', async () => {
  const user = userEvent.setup(); render(<LoRATraining />);
  const button = await screen.findByRole('button', { name: 'Run rank experiment' });
  await user.click(screen.getByRole('checkbox', { name: /Rank 16/ })); expect(button).toBeDisabled();
  await user.click(screen.getByRole('checkbox', { name: /Rank 8/ })); await user.click(button);
  const body = JSON.parse(fetch.mock.calls.find(([, options]) => options?.method === 'POST')[1].body);
  expect(body.ranks).toEqual([4, 8]);
  expect(body.training).toMatchObject({ provider: 'local', lora_alpha: null, lora_dropout: 0, lora_target_policy: 'query-value', seed: 42 });
});
it('renders zero scores as zero and explains measured memory', async () => {
  history = [{ experiment_id: 'saved', status: 'completed', config: { ranks: [4, 16] }, comparison: { claim: 'No measured improvement.' }, runs: [{ run_id: 'rank16', rank: 16, alpha: 32, status: 'completed', micro_f1: 0, macro_f1: 0, exact_match: 0, json_validity: 0.8, peak_rss_mib: 2941.2, training_seconds: 29.5, trainable_parameters: 1081344 }] }];
  render(<LoRATraining />);
  const table = await screen.findByRole('table', { name: 'Rank experiment results' });
  expect(within(table).getAllByText('0.00%')).toHaveLength(3);
  expect(within(table).getByText('80.00%')).toBeInTheDocument();
  expect(screen.getByText(/total process RSS sampled/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Download comparison evidence' })).toHaveAttribute('href', '/api/v1/training/lora/experiments/saved/download');
});
it('surfaces queue errors without pretending training started', async () => {
  reject = true; const user = userEvent.setup(); render(<LoRATraining />);
  await user.click(await screen.findByRole('button', { name: 'Run rank experiment' }));
  expect(await screen.findByText('Another training worker is active')).toBeInTheDocument();
});
