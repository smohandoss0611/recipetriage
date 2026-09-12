import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import TrainingMonitor from './TrainingMonitor';

const version = 'v1-' + 'a'.repeat(64);
const defaults = { provider: 'local', dataset_version: version, epochs: 3, learning_rate: 0.0002, sequence_length: 768, batch_size: 1, gradient_accumulation_steps: 2, seed: 42, lora_rank: 8, fireworks_model: null, fireworks_deployment_shape: null };
const seed = { version, counts: { train: 5, validation: 1, test: 1 }, status: 'teaching-draft' };
const run = { run_id: 'one', config: defaults, status: 'completed', phase: 'finished', parameters: {}, dataset: seed, managed: {},
  history: [{ step: 1, epoch: 1, loss: 2.4, grad_norm: 3.5, learning_rate: 0.0002 }],
  comparison: { comparable: true, before: { micro_f1: 0.485 }, after: { micro_f1: 0.387 }, delta: { micro_f1: -0.098 }, claim: 'No measured Micro-F1 improvement on these ten fixed cases.' } };
let history;
beforeEach(() => {
  history = [];
  vi.stubGlobal('fetch', vi.fn(async (url, options = {}) => {
    let value;
    if (url.endsWith('/options')) value = { defaults, seed_dataset: seed, fireworks_configured: true };
    else if (url.endsWith('/datasets/versions')) value = [];
    else if (url.endsWith('/preflight')) value = { supported: false, issues: ['Validation requires at least three examples.'] };
    else if (url.endsWith('/runs/one')) value = run;
    else if (options.method === 'POST') { history = [run]; value = { run_id: 'one', status: 'queued' }; }
    else value = history;
    return { ok: true, json: async () => value };
  }));
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

it('opens without starting training or making hosted mutation calls', async () => {
  render(<TrainingMonitor />);
  expect(await screen.findByRole('button', { name: 'Start local SFT' })).toBeEnabled();
  expect(screen.getByText(/5 train \/ 1 validation \/ 1 test/)).toBeInTheDocument();
  expect(fetch.mock.calls.every(([, options]) => options?.method !== 'POST')).toBe(true);
});

it('submits explicit hyperparameters then shows measured regression and gradient history', async () => {
  const user = userEvent.setup(); render(<TrainingMonitor />);
  await user.click(await screen.findByRole('button', { name: 'Start local SFT' }));
  expect(await screen.findByText(/No measured Micro-F1 improvement/)).toBeInTheDocument();
  expect(screen.getByText('-9.8')).toBeInTheDocument();
  expect(screen.getByRole('table', { name: 'Loss by optimizer step' })).toBeInTheDocument();
  const request = fetch.mock.calls.find(([url, options]) => url.endsWith('/training/runs') && options?.method === 'POST');
  expect(JSON.parse(request[1].body)).toEqual(defaults);
});

it('preflight reports provider limits and does not submit a managed job', async () => {
  const user = userEvent.setup(); render(<TrainingMonitor />);
  await user.click(await screen.findByRole('combobox', { name: 'Training provider' }));
  await user.click(screen.getByRole('option', { name: 'Managed · Fireworks' }));
  await user.click(screen.getByRole('button', { name: 'Check Fireworks support' }));
  expect(await screen.findByText('Validation requires at least three examples.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Start managed SFT' })).toBeDisabled();
  expect(fetch.mock.calls.filter(([, options]) => options?.method === 'POST')).toHaveLength(1);
});

it('displays API errors rather than a success state', async () => {
  const user = userEvent.setup(); render(<TrainingMonitor />);
  await screen.findByRole('button', { name: 'Start local SFT' });
  fetch.mockImplementationOnce(async () => ({ ok: false, json: async () => ({ detail: 'Worker is already busy' }) }));
  await user.click(screen.getByRole('button', { name: 'Start local SFT' }));
  expect(await screen.findByText('Worker is already busy')).toBeInTheDocument();
  expect(screen.queryByText(/Fixed benchmark comparison/)).not.toBeInTheDocument();
});

it('reports a non-JSON gateway response with its HTTP status', async () => {
  const user = userEvent.setup(); render(<TrainingMonitor />);
  await screen.findByRole('button', { name: 'Start local SFT' });
  fetch.mockImplementationOnce(async () => ({ ok: false, status: 502, json: async () => { throw new Error('HTML response'); } }));
  await user.click(screen.getByRole('button', { name: 'Start local SFT' }));
  expect(await screen.findByText('API returned HTTP 502 without JSON. Check the backend connection.')).toBeInTheDocument();
});

it('clears a refresh error when the backend recovers', async () => {
  let refresh;
  const originalInterval = globalThis.setInterval;
  vi.spyOn(globalThis, 'setInterval').mockImplementation((callback, delay) => {
    if (delay === 3000) { refresh = callback; return 123; }
    return originalInterval(callback, delay);
  });
  render(<TrainingMonitor />);
  await screen.findByRole('button', { name: 'Start local SFT' });
  fetch.mockImplementationOnce(async () => ({ ok: false, status: 502, json: async () => { throw new Error('HTML'); } }));
  await act(async () => { await refresh(); });
  expect(screen.getByText(/HTTP 502/)).toBeInTheDocument();
  await act(async () => { await refresh(); });
  expect(screen.queryByText(/HTTP 502/)).not.toBeInTheDocument();
});
