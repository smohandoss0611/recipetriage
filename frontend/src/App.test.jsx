import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
beforeEach(() => vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ok:true,json:async()=>({recipe:{title:'Ravioli'}})})));
afterEach(() => vi.unstubAllGlobals());
import App from './App';

const labels = ['Recipes', 'Data Lab', 'Training Lab', 'Alignment Lab', 'Evaluation Lab', 'Playground', 'Deployment'];
describe('application shell', () => {
  it('renders all required navigation destinations and the default page', () => {
    render(<App />);
    const nav = within(screen.getByRole('navigation', { name: 'Main navigation' }));
    labels.forEach(label => expect(nav.getByRole('button', { name: label })).toBeInTheDocument());
    expect(screen.getByRole('heading', { level: 1, name: 'Recipes' })).toBeInTheDocument();
  });
  it('switches between every workspace', async () => {
    const user = userEvent.setup();
    render(<App />);
    for (const label of labels) {
      const button = screen.getByRole('button', { name: label });
      await user.click(button);
      expect(screen.getByRole('heading', { level: 1, name: label })).toBeInTheDocument();
      expect(button).toHaveAttribute('aria-current', 'page');
    }
  });
});
