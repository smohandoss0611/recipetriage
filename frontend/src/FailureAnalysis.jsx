import { useEffect, useState } from 'react';
import { Alert, Box, Button, Chip, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material';
import { api, percent } from './experimentApi';
export default function FailureAnalysis() {
  const [analyses, setAnalyses] = useState([]), [sources, setSources] = useState([]), [selected, setSelected] = useState(''), [source, setSource] = useState('');
  const [error, setError] = useState(''), [busy, setBusy] = useState(false), [category, setCategory] = useState('all'), [priorities, setPriorities] = useState([]);
  useEffect(() => { let alive = true; Promise.all([api('analysis/failures'), api('analysis/sources')]).then(([rows, origins]) => {
    if (!alive) return; if (!Array.isArray(rows) || !Array.isArray(origins)) throw new Error('Unexpected analysis response');
    setAnalyses(rows); setSources(origins); setSelected(rows[0]?.source_run_id || ''); setSource(origins[0]?.run_id || '');
  }).catch(e => alive && setError(e.message)); return () => { alive = false; }; }, []);
  const analysis = analyses.find(row => row.source_run_id === selected);
  useEffect(() => { let alive = true; setPriorities(analysis?.collection_priorities || []); if (analysis) api(`analysis/collection-priorities/${analysis.source_run_id}`).then(rows => alive && setPriorities(rows)).catch(e => alive && setError(e.message)); return () => { alive = false; }; }, [analysis]);
  const create = async () => { setBusy(true); setError(''); try { const row = await api('analysis/failures', { source_run_id: source }); setAnalyses(previous => [row, ...previous.filter(item => item.source_run_id !== row.source_run_id)]); setSelected(row.source_run_id); } catch (e) { setError(e.message); } finally { setBusy(false); } };
  const visible = (analysis?.findings || []).filter(row => category === 'all' || row.failure_categories.includes(category));
  return <Stack spacing={3}>
    <Box><Typography component="h2" variant="h5">Failure Analysis</Typography><Typography color="text.secondary">Separate response-format failures from incorrect labels. Categories are persisted in PostgreSQL and can overlap.</Typography></Box>
    {error && <Alert severity="error">{error}</Alert>}
    {!!sources.length && <Paper variant="outlined" sx={{ p: 3 }}><Stack spacing={2}><TextField select label="Benchmark source" value={source} onChange={e => setSource(e.target.value)}>{sources.map(row => <MenuItem key={row.run_id} value={row.run_id}>{row.model_config.training_stage} · {row.run_id.slice(0, 8)} · {row.status}</MenuItem>)}</TextField><Button disabled={busy || !source} onClick={create} sx={{ alignSelf: 'flex-start' }}>Analyze saved benchmark</Button></Stack></Paper>}
    {!analyses.length && <Typography>No saved analyses yet. Choose a completed benchmark; analysis makes no model calls.</Typography>}
    {!!analyses.length && <TextField select label="Failure analysis" value={selected} onChange={e => { setSelected(e.target.value); setCategory('all'); }}>{analyses.map(row => <MenuItem key={row.source_run_id} value={row.source_run_id}>{row.model_config.training_stage} · {row.source_run_id.slice(0, 8)} · {row.failed_cases}/{row.total_cases} failed</MenuItem>)}</TextField>}
    {analysis && <>
      <Alert severity="info">{analysis.limitations}</Alert>
      <Typography>Misleading-title benchmark accuracy: {percent(analysis.misleading_title_accuracy)} ({analysis.misleading_title_correct}/{analysis.misleading_title_total} exact label sets). This original benchmark subset differs from the counterfactual suite.</Typography>
      <Stack direction="row" useFlexGap spacing={1} sx={{ flexWrap: 'wrap' }}>{Object.entries(analysis.category_counts).map(([key, count]) => <Chip key={key} label={`${key.replaceAll('_', ' ')}: ${count}`} />)}</Stack>
      <TextField select label="Failure category" value={category} onChange={e => setCategory(e.target.value)}><MenuItem value="all">All cases</MenuItem>{Object.keys(analysis.category_counts).map(key => <MenuItem key={key} value={key}>{key.replaceAll('_', ' ')}</MenuItem>)}</TextField>
      <TableContainer component={Paper} variant="outlined"><Table size="small" aria-label="Failure cases"><TableHead><TableRow>{['Case', 'Expected labels', 'Predicted labels', 'Categories', 'Evidence'].map(label => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead><TableBody>{visible.map(row => <TableRow key={row.case_id}><TableCell>{row.case_id}<br />{row.title}</TableCell><TableCell>{row.expected_labels.join(', ')}</TableCell><TableCell>{row.predicted_labels?.join(', ') || 'Unusable / missing response'}</TableCell><TableCell>{row.failure_categories.join(', ') || 'Correct exact set'}</TableCell><TableCell><Box component="details"><Typography component="summary">Raw response</Typography><Typography color="error">{row.error}</Typography><Box component="pre" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxWidth: 420 }}>{row.raw_output ?? 'No output recorded'}</Box></Box></TableCell></TableRow>)}</TableBody></Table></TableContainer>
      <Paper variant="outlined" sx={{ p: 3 }}><Typography component="h3" variant="h6">What data to collect next</Typography><Stack spacing={2} sx={{ mt: 2 }}>{priorities.map(row => <Box key={row.priority}><Typography fontWeight={700}>{row.priority}. {row.topic}</Typography><Typography>{row.action}</Typography><Typography variant="body2" color="text.secondary">Evidence: {row.basis}{row.observed_cases !== null ? ` · ${row.observed_cases} affected cases/pairs` : ''}. {row.case_ids.join(', ')} {row.split_rule}</Typography></Box>)}</Stack></Paper>
    </>}
  </Stack>;
}
