import { useEffect, useState } from 'react';
import { Alert, Button, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material';

export default function InferencePlayground({ initialRecipe = null }) {
  const [recipe, setRecipe] = useState(() => initialRecipe ? JSON.stringify(initialRecipe, null, 2) : '');
  const [provider, setProvider] = useState('hf');
  const [temperature, setTemperature] = useState('0');
  const [maxTokens, setMaxTokens] = useState('128');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let active = true;
    if (initialRecipe) {
      setRecipe(JSON.stringify(initialRecipe, null, 2));
      setResult(null);
      setError('');
      return () => { active = false; };
    }
    fetch('/api/v1/learning-example').then(async response => {
      if (!response.ok) throw new Error('Could not load ravioli example');
      return response.json();
    }).then(data => { if (active) setRecipe(JSON.stringify(data.recipe, null, 2)); }).catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [initialRecipe]);
  async function run(event) {
    event.preventDefault(); setError(''); setResult(null); setBusy(true);
    try {
      const response = await fetch('/api/v1/triage', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ recipe: JSON.parse(recipe), provider, temperature: Number(temperature), max_new_tokens: Number(maxTokens) }) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  return <Stack component="form" onSubmit={run} spacing={3}>
    <Typography component="h2" variant="h5">Inference Playground</Typography>
    <Typography>Run the same recipe through either provider. Fireworks sends the recipe to its hosted API; its key stays on the server. Local inference can take longer on first use.</Typography>
    <TextField label="Recipe JSON" multiline minRows={10} value={recipe} disabled={busy} onChange={e => { setRecipe(e.target.value); setResult(null); }} fullWidth />
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
      <TextField select label="Provider" value={provider} disabled={busy} onChange={e => { setProvider(e.target.value); setResult(null); }} sx={{ minWidth: 220 }}><MenuItem value="hf">Local Hugging Face</MenuItem><MenuItem value="fireworks">Hosted Fireworks</MenuItem><MenuItem value="production">Registered production version</MenuItem></TextField>
      <TextField label="Temperature" type="number" value={temperature} disabled={busy} onChange={e => { setTemperature(e.target.value); setResult(null); }} slotProps={{ htmlInput: { min: 0, max: 2, step: 0.1 } }} />
      <TextField label="Max new tokens" type="number" value={maxTokens} disabled={busy} onChange={e => { setMaxTokens(e.target.value); setResult(null); }} slotProps={{ htmlInput: { min: 1, max: 256, step: 1 } }} />
    </Stack>
    <Button type="submit" variant="contained" disabled={busy || !recipe || !temperature || !maxTokens}>{busy ? 'Running inference…' : 'Run inference'}</Button>
    {error && <Alert severity="error">{error}</Alert>}
    {result && <><Alert severity={result.valid_json ? 'success' : 'warning'}>{result.valid_json ? 'Schema-valid response. Check the labels yourself: valid JSON does not prove correctness.' : `Invalid or incomplete response: ${result.error}`}</Alert><Paper component="pre" variant="outlined" sx={{ p: 2, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(result, null, 2)}</Paper></>}
  </Stack>;
}
