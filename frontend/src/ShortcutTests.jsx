import { useEffect, useState } from 'react';
import { Alert, Box, Button, LinearProgress, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material';
import { api, percent } from './experimentApi';
export default function ShortcutTests() {
  const [runs, setRuns] = useState([]), [selected, setSelected] = useState(''), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => { let alive = true; const refresh = async () => { try { const rows = await api('analysis/diagnostics'); if (!Array.isArray(rows)) throw new Error('Unexpected diagnostic response'); if (alive) { setRuns(rows); setSelected(previous => previous || rows[0]?.run_id || ''); } } catch (e) { if (alive) setError(e.message); } }; refresh(); const timer = setInterval(refresh, 3000); return () => { alive = false; clearInterval(timer); }; }, []);
  const run = runs.find(row => row.run_id === selected), shortcut = run?.shortcut, score = shortcut?.summary;
  const start = async () => { setBusy(true); setError(''); try { const value = await api('analysis/diagnostics', {}); setRuns(await api('analysis/diagnostics')); setSelected(value.run_id); } catch (e) { setError(e.message); } finally { setBusy(false); } };
  return <Stack spacing={3}>
    <Box><Typography component="h2" variant="h5">Shortcut Tests</Typography><Typography color="text.secondary">Traditional, Easy, Quick and Simple: only the title adjective changes. Time, ingredients, equipment, instructions and pantry stay identical.</Typography></Box>
    {error && <Alert severity="error">{error}</Alert>}
    <Alert severity="info">Flip rate compares each usable variant with its usable Traditional reference. Invalid pairs are excluded and counted. Accuracy requires the exact label set; invalid or missing answers count as incorrect.</Alert>
    <Button variant="contained" disabled={busy || runs.some(row => ['queued', 'running'].includes(row.status))} onClick={start} sx={{ alignSelf: 'flex-start' }}>Run base-model diagnostics</Button>
    <Typography color="text.secondary">Runs 12 title variants and a separate three-case prompt-injection suite. No training. Saved adapter diagnostics can also be run with the CLI and imported.</Typography>
    {!runs.length ? <Typography>No saved diagnostics yet.</Typography> : <TextField select label="Diagnostic run" value={selected} onChange={e => setSelected(e.target.value)}>{runs.map(row => <MenuItem key={row.run_id} value={row.run_id}>{row.model_config?.training_stage || 'base'} · {row.status} · {row.run_id.slice(0, 8)}</MenuItem>)}</TextField>}
    {run && <>
      <Button component="a" href={`/api/v1/analysis/diagnostics/${run.run_id}/download`} sx={{ alignSelf: 'flex-start' }}>Download diagnostic evidence</Button>
      {['queued', 'running'].includes(run.status) && <LinearProgress aria-label="Diagnostics running" />}{run.error && <Alert severity="error">{run.error}</Alert>}
      {score && <Paper variant="outlined" sx={{ p: 3 }}><Stack spacing={1}>
        <Typography component="h3" variant="h6">Counterfactual results</Typography>
        <Typography>Prediction-flip rate: {percent(score.prediction_flip_rate)} ({score.flipped_pairs}/{score.eligible_pairs} usable pairs).</Typography>
        <Typography>Pair coverage: {percent(score.pair_coverage)} · {score.excluded_pairs}/{score.planned_pairs} excluded pairs.</Typography>
        <Typography>Misleading-title accuracy: {percent(score.misleading_title_accuracy)} ({score.misleading_correct}/{score.misleading_total} Quick-title cases).</Typography>
        <Typography>Overall exact-set accuracy: {percent(score.exact_set_accuracy)} · {score.usable_rows}/{score.planned_rows} usable outputs.</Typography>
        <Typography color="text.secondary">Quick is the predeclared misleading-time adjective for these 70–80 minute recipes. Easy and Simple do not assert a precise duration.</Typography>
      </Stack></Paper>}
      {shortcut && <>
        <TableContainer component={Paper} variant="outlined"><Table size="small" aria-label="Title counterfactual predictions"><TableHead><TableRow>{['Recipe', 'Title', 'Minutes', 'Labels', 'Usable', 'Evidence'].map(label => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead><TableBody>{shortcut.rows.map(row => <TableRow key={`${row.case_id}-${row.adjective}`}><TableCell>{row.case_id}</TableCell><TableCell>{row.recipe.title}</TableCell><TableCell>{row.recipe.time_minutes}</TableCell><TableCell>{row.usable ? row.prediction.labels.join(', ') : 'Unusable response'}</TableCell><TableCell>{row.usable ? 'Yes' : 'No'}</TableCell><TableCell><Box component="details"><Typography component="summary">Input and raw output</Typography><Typography color="error">{row.error}</Typography><Box component="pre" sx={{ maxWidth: 420, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify({ body_sha256: row.body_sha256, recipe: row.recipe, raw_output: row.raw_output }, null, 2)}</Box></Box></TableCell></TableRow>)}</TableBody></Table></TableContainer>
        <Alert severity="warning">{shortcut.limitations}</Alert>
      </>}
      {run.red_team && <Paper variant="outlined" sx={{ p: 3 }}><Stack spacing={2}>
        <Typography component="h3" variant="h6">Separate red-team diagnostics</Typography><Typography>{run.red_team.passed}/{run.red_team.total} attacks retained the correct structured answer.</Typography>
        <Alert severity="info">{run.red_team.suite.note} {run.red_team.limitations}</Alert>
        {run.red_team.rows.map(row => <Box component="details" key={row.case_id}><Typography component="summary">{row.case_id} · {row.passed ? 'Passed' : 'Failed'}</Typography><Typography>Explicit attack instruction followed: {row.explicit_instruction_following ? 'Yes' : 'No'} · Marker present: {row.marker_present ? 'Yes' : 'No'}</Typography><Box component="pre" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{row.raw_output || row.error}</Box></Box>)}
      </Stack></Paper>}
    </>}
  </Stack>;
}
