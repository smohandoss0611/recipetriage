import { useEffect, useState } from 'react';
import { Alert, Box, Button, Checkbox, FormControlLabel, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material';
import { api } from './experimentApi';

export default function RecipeActions({ item, labels, policy, onChanged }) {
  const [mode, setMode] = useState('');
  const [chosen, setChosen] = useState([]);
  const [reviewer, setReviewer] = useState('');
  const [rationale, setRationale] = useState('');
  const [edited, setEdited] = useState('');
  const [model, setModel] = useState('fireworks');
  const [models, setModels] = useState([]);
  const [history, setHistory] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { setMode(''); setError(''); setHistory(null); }, [item.library_id, item.revision]);
  useEffect(() => {
    let active = true;
    api('playground/models').then(data => { if (active) { setModels(data.models); setModel(data.models.find(row => row.available)?.key ?? ''); } }).catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, []);
  async function act(operation) {
    setBusy(true); setError('');
    try { await operation(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  const running = ['queued', 'running'].includes(item.triage_job?.status);
  const output = item.prediction?.result;
  return <Stack spacing={2}>
    {error && <Alert severity="error">{error}</Alert>}
    {running && <Alert severity="info">Triage is running. The library refreshes when the prediction is ready.</Alert>}
    {item.triage_job?.error && <Alert severity="error">{item.triage_job.error}</Alert>}
    {output && <Box component="details"><Typography component="summary">Model prediction and raw JSON</Typography>
      <Alert severity={output.valid_json ? 'info' : 'warning'}>{output.valid_json ? 'Structured prediction; human verification is still separate.' : output.error}</Alert>
      <Paper component="pre" sx={{ p: 2, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(output, null, 2)}</Paper>
    </Box>}
    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
      <Button disabled={busy} variant="outlined" onClick={() => { setMode('review'); setChosen(item.labels); setRationale(item.rationale); }}>Review or correct labels</Button>
      <Button disabled={busy} onClick={() => {
        const { title, ingredients, instructions, equipment, time_minutes, pantry_items = null } = item.recipe;
        setEdited(JSON.stringify({ title, ingredients, instructions, equipment, time_minutes, pantry_items }, null, 2)); setMode('edit');
      }}>Edit recipe</Button>
      <Button disabled={busy} onClick={() => setMode('triage')}>Run triage</Button>
      <Button disabled={busy} onClick={() => act(async () => setHistory(await api(`recipes/${item.library_id}/history`)))}>View recipe history</Button>
    </Stack>
    {mode === 'review' && <>
      <Typography component="h3" variant="h6">Verify the labels for revision {item.revision}</Typography>
      <Box component="fieldset" sx={{ border: 0, p: 0, m: 0 }}><legend>Correct labels</legend>
        {labels.map(value => <FormControlLabel key={value} control={<Checkbox checked={chosen.includes(value)} disabled={busy} onChange={event => setChosen(previous => event.target.checked ? [...previous, value] : previous.filter(label => label !== value))} />} label={value} />)}
      </Box>
      <Box component="details"><Typography component="summary">Label definitions</Typography>{labels.map(value => <Typography key={value} variant="body2" sx={{ my: 1 }}><strong>{value}</strong>: {policy[value]}</Typography>)}</Box>
      <TextField label="Reviewer name" value={reviewer} onChange={event => setReviewer(event.target.value)} disabled={busy} />
      <TextField label="Reason for verified labels" value={rationale} onChange={event => setRationale(event.target.value)} multiline disabled={busy} />
      <Button variant="contained" disabled={busy || !chosen.length || !reviewer.trim() || !rationale.trim()} onClick={() => act(async () => {
        await api(`recipes/${item.library_id}/review`, { revision: item.revision, labels: chosen, reviewer, rationale }); setMode(''); onChanged();
      })}>Verify corrected labels</Button>
      <Typography variant="body2">This explicit action records your review and a verified training example. Dataset export is a separate action.</Typography>
    </>}
    {mode === 'edit' && <>
      <TextField label="Edited Recipe JSON" value={edited} onChange={event => setEdited(event.target.value)} multiline minRows={10} disabled={busy} />
      <TextField label="Editor name" value={reviewer} onChange={event => setReviewer(event.target.value)} disabled={busy} />
      <Alert severity="info">Changing recipe content creates a new revision and clears its current verification. Earlier predictions and reviews remain in history.</Alert>
      <Button variant="contained" disabled={busy || !reviewer.trim()} onClick={() => act(async () => {
        await api(`recipes/${item.library_id}`, { revision: item.revision, recipe: JSON.parse(edited), actor: reviewer }, 'PUT'); setMode(''); onChanged();
      })}>Save recipe revision</Button>
    </>}
    {mode === 'triage' && <>
      <TextField select label="Triage model" value={model} onChange={event => setModel(event.target.value)} disabled={busy || running}>
        {models.map(row => <MenuItem key={row.key} value={row.key} disabled={!row.available}>{row.title}{!row.available ? ' · unavailable' : ''}</MenuItem>)}
      </TextField>
      <Button disabled={busy || running || !model} onClick={() => act(async () => { await api(`recipes/${item.library_id}/triage`, { revision: item.revision, model_key: model }); setMode(''); onChanged(); })}>Predict labels</Button>
    </>}
    {mode && <Button disabled={busy} onClick={() => setMode('')}>Cancel</Button>}
    {history && <Box component="details" open><Typography component="summary">Saved predictions, edits and human reviews</Typography><Paper component="pre" sx={{ p: 2, maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(history, null, 2)}</Paper></Box>}
  </Stack>;
}
