import { useEffect, useState } from 'react';
import { Alert, Box, Button, Checkbox, Chip, FormControlLabel, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material';
import { api, numeric } from './experimentApi';

export default function ModelComparison({ initialRecipe = null }) {
  const [recipe, setRecipe] = useState(() => initialRecipe ? JSON.stringify(initialRecipe, null, 2) : '');
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState([]);
  const [temperature, setTemperature] = useState('0');
  const [tokens, setTokens] = useState('128');
  const [job, setJob] = useState(null);
  const [history, setHistory] = useState([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const running = ['queued', 'running'].includes(job?.status);
  useEffect(() => {
    let active = true;
    Promise.all([api('playground/models'), api('playground/comparisons')]).then(([catalog, previous]) => {
      if (!active) return;
      setModels(catalog.models); setHistory(previous);
      const defaults = catalog.default_models.filter(key => catalog.models.some(row => row.key === key && row.available));
      setSelected(defaults.length >= 2 ? defaults : catalog.models.filter(row => row.available).slice(0, 2).map(row => row.key));
    }).catch(err => { if (active) setError(err.message); });
    if (!initialRecipe) api('learning-example').then(data => { if (active) setRecipe(JSON.stringify(data.recipe, null, 2)); }).catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [initialRecipe]);
  useEffect(() => { if (initialRecipe) { setRecipe(JSON.stringify(initialRecipe, null, 2)); setJob(null); } }, [initialRecipe]);
  useEffect(() => {
    if (!running) return;
    let active = true;
    const timer = setInterval(async () => {
      try {
        const updated = await api('playground/comparisons/' + job.run_id);
        if (!active) return;
        setJob(updated);
        if (!['queued', 'running'].includes(updated.status)) setHistory(await api('playground/comparisons'));
      } catch (err) { if (active) setError(err.message); }
    }, 2000);
    return () => { active = false; clearInterval(timer); };
  }, [job?.run_id, running]);
  async function run() {
    setSending(true); setError(''); setJob(null);
    try {
      if (!temperature.trim() || !tokens.trim()) throw new Error('Enter temperature and the completion token budget');
      setJob(await api('playground/comparisons', { recipe: JSON.parse(recipe), models: selected, temperature: Number(temperature), max_new_tokens: Number(tokens) }));
    } catch (err) { setError(err.message); } finally { setSending(false); }
  }
  return <Stack spacing={3}>
    <Typography component="h2" variant="h5">Compare models</Typography>
    <Typography>Send the same recipe to selected models and compare their raw and structured responses. Fireworks receives the recipe through its hosted API; local models run sequentially.</Typography>
    {error && <Alert severity="error">{error}</Alert>}
    <TextField label="Comparison Recipe JSON" multiline minRows={10} value={recipe} onChange={event => { setRecipe(event.target.value); setJob(null); }} disabled={sending || running} />
    <Box component="fieldset" sx={{ border: 0, p: 0, m: 0 }}><legend>Models to compare</legend>
      {models.map(row => <Box key={row.key} sx={{ mb: 1 }}>
        <FormControlLabel control={<Checkbox checked={selected.includes(row.key)} disabled={!row.available || running || sending} onChange={event => setSelected(previous => event.target.checked ? [...previous, row.key] : previous.filter(key => key !== row.key))} />} label={row.title} />
        <Typography variant="body2" color="text.secondary" sx={{ overflowWrap: 'anywhere' }}>{row.available ? row.model : row.reason}</Typography>
      </Box>)}
    </Box>
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
      <TextField label="Comparison temperature" value={temperature} type="number" onChange={event => setTemperature(event.target.value)} disabled={sending || running} slotProps={{ htmlInput: { min: 0, max: 2, step: 0.1 } }} />
      <TextField label="Comparison max new tokens" value={tokens} type="number" onChange={event => setTokens(event.target.value)} disabled={sending || running} slotProps={{ htmlInput: { min: 1, max: 256, step: 1 } }} />
    </Stack>
    <Button variant="contained" disabled={sending || running || selected.length < 2 || selected.length > 6 || !recipe.trim()} onClick={run}>{sending || running ? 'Comparing models…' : 'Compare selected models'}</Button>
    <Typography variant="body2">Choose 2–6 models. Missing DPO checkpoints stay unavailable until human preference training completes. This comparison does not add training data or promote a model.</Typography>
    {!!history.length && <TextField select label="Saved comparison" value={history.some(row => row.run_id === job?.run_id) ? job.run_id : ''} disabled={sending || running} onChange={event => {
      const previous = history.find(row => row.run_id === event.target.value); setJob(previous);
      if (previous) { setRecipe(JSON.stringify(previous.config.recipe, null, 2)); setSelected(previous.config.models); setTemperature(String(previous.config.temperature)); setTokens(String(previous.config.max_new_tokens)); }
    }}>
      <MenuItem value="">Choose saved results</MenuItem>
      {history.map(row => <MenuItem key={row.run_id} value={row.run_id}>{row.config.recipe.title} · {row.run_id.slice(0, 8)} · {row.status}</MenuItem>)}
    </TextField>}
    {job && <>
      <Typography component="h3" variant="h6">{job.config.recipe.title} · {job.result?.status ?? job.status}</Typography>
      {job.error && <Alert severity="error">{job.error}</Alert>}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, minmax(0, 1fr))' }, gap: 2 }}>
        {job.config.models.map(key => {
          const row = job.result?.rows?.find(item => item.key === key);
          const output = row?.result;
          return <Paper key={key} variant="outlined" component="section" aria-label={`${models.find(item => item.key === key)?.title ?? key} comparison result`} sx={{ p: 2, minWidth: 0 }}>
            <Stack spacing={2}>
              <Typography component="h4" variant="h6">{row?.title ?? models.find(item => item.key === key)?.title ?? key}</Typography>
              {!row && <Typography>{running ? 'Waiting for this model…' : 'No response recorded'}</Typography>}
              {row?.error && <Alert severity="error">{row.error}</Alert>}
              {output && <>
                <Alert severity={output.valid_json ? 'info' : 'warning'}>{output.valid_json ? 'Valid structured output; correctness needs review.' : output.error}</Alert>
                <Stack direction="row" flexWrap="wrap" gap={1}>{output.prediction?.labels?.map(label => <Chip key={label} label={label} size="small" />)}</Stack>
                <Typography>{output.prediction?.explanation}</Typography>
                <Typography variant="body2">{output.latency_scope ? 'Generation' : 'Provider latency (legacy timing)'}: {numeric(output.latency_ms)} ms · Finish: {output.finish_reason} · Output tokens: {output.output_tokens ?? 'not reported'}</Typography>
                {output.model_load_ms != null && <Typography variant="body2">Model preparation: {numeric(output.model_load_ms)} ms · Total: {numeric(output.total_ms)} ms. Hosted generation includes network time.</Typography>}
                <Box component="pre" sx={{ p: 1, bgcolor: 'background.default', overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{output.raw_output}</Box>
                <Box component="details"><Typography component="summary">Model identity and complete response</Typography><Box component="pre" sx={{ overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(output, null, 2)}</Box></Box>
              </>}
            </Stack>
          </Paper>;
        })}
      </Box>
    </>}
  </Stack>;
}
