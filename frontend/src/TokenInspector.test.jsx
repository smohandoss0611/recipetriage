import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import TokenInspector from './TokenInspector';

afterEach(() => vi.unstubAllGlobals());
it('shows actual returned vocabulary tokens, IDs and count', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({model:'test-model',revision:'test',tokens:['Easy','Ġrav'],token_ids:[12,34],token_count:2,rendered_text:'Easy ravioli'}) }));
  const user = userEvent.setup();
  render(<TokenInspector />);
  await user.click(screen.getByRole('button', {name:'Inspect tokens'}));
  expect(await screen.findByText('2 tokens')).toBeInTheDocument();
  expect(screen.getByText('12')).toBeInTheDocument();
  expect(screen.getByText('34')).toBeInTheDocument();
  await user.type(screen.getByLabelText('Text to tokenize'), ' edit');
  expect(screen.queryByText('2 tokens')).not.toBeInTheDocument();
});
it('shows backend errors instead of fabricating tokens', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ok:false,json:async()=>({detail:'Tokenizer unavailable'})}));
  render(<TokenInspector />);
  await userEvent.click(screen.getByRole('button',{name:'Inspect tokens'}));
  expect(await screen.findByRole('alert')).toHaveTextContent('Tokenizer unavailable');
});
