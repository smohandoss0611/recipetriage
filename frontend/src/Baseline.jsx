import { useEffect, useState } from 'react';
import { Alert, Box, Button, Chip, LinearProgress, MenuItem, Paper, Stack, Tab, Tabs, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material';

const API = '/api/v1/benchmarks';
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const seconds = value => value == null ? '—' : `${(value / 1000).toFixed(3)} s`;
const pending = run => ['queued', 'running'].includes(run.status);
const pretty = value => JSON.stringify(value, null, 2);
const Code = ({ children }) => <Paper component="pre" variant="outlined" sx={{ p: 2, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: 450, fontSize: '0.8rem' }}>{children}</Paper>;
async function request(path, body) {
  const response = await fetch(API + path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : pretty(result.detail ?? result));
  return result;
}

export default function Baseline() {
  const [page, setPage] = useState(0);
  const [benchmark, setBenchmark] = useState(null);
  const [providers, setProviders] = useState([]);
  const [provider, setProvider] = useState('hf-base');
  const [temperature, setTemperature] = useState('0');
  const [budget, setBudget] = useState('128');
  const [reasoning, setReasoning] = useState('disabled');
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState('');
  const [detail, setDetail] = useState(null);
  const [caseId, setCaseId] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const active = runs.some(pending);

  useEffect(() => {
    let current = true;
    Promise.all([request('/manifest'), request('/providers'), request('/runs')]).then(([spec, available, history]) => {
      if (!current) return;
      if (!Array.isArray(spec.cases) || !Array.isArray(history)) throw new Error('Invalid benchmark response');
      setBenchmark(spec); setProviders(available); setRuns(history); setSelected(history[0]?.run_id ?? '');
    }).catch(err => { if (current) setError(err.message); });
    return () => { current = false; };
  }, []);

  useEffect(() => {
    let current = true;
    if (!selected) { setDetail(null); return; }
    setDetail(null);
    request('/runs/' + selected).then(result => {
      if (current) { setDetail(result); setCaseId(result.rows[0]?.case_id ?? ''); }
    }).catch(err => { if (current) setError(err.message); });
    return () => { current = false; };
  }, [selected]);

  useEffect(() => {
    if (!active) return;
    let current = true;
    let inFlight = false;
    const timer = setInterval(async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const history = await request('/runs');
        const result = selected ? await request('/runs/' + selected) : null;
        if (current) {
          setRuns(history);
          if (result) { setDetail(result); setCaseId(old => old || result.rows[0]?.case_id || ''); }
        }
      } catch (err) { if (current) setError(err.message); }
      finally { inFlight = false; }
    }, 2000);
    return () => { current = false; clearInterval(timer); };
  }, [active, selected]);

  async function refresh() {
    setError('');
    try {
      const history = await request('/runs'); setRuns(history);
      setProviders(await request('/providers'));
      if (selected) setDetail(await request('/runs/' + selected));
      else setSelected(history[0]?.run_id ?? '');
    } catch (err) { setError(err.message); }
  }
  async function start() {
    setBusy(true); setError('');
    try {
      if (!budget.trim() || !temperature.trim()) throw new Error('Enter an output budget and temperature.');
      const created = await request('/runs', { provider, max_new_tokens: Number(budget), temperature: Number(temperature), reasoning });
      setSelected(created.run_id); setPage(2);
      setRuns(await request('/runs'));
      setDetail(await request('/runs/' + created.run_id));
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  const row = detail?.rows.find(item => item.case_id === caseId);
  const metrics = detail?.summary.labels;
  const eligible = runs.filter(run => run.status === 'completed' && run.summary.scorable);
  const mixedProtocols = new Set(eligible.map(run => run.protocol_sha256)).size > 1;
  const providerInfo = providers.find(item => item.id === provider);

  return <Stack spacing={3}>
    <Typography variant="h5" component="h2">Baseline</Typography>
    <Typography>Compare unchanged models on RecipeTriage-Bench-v1. Inspect label quality, structured answers, and response time before considering training.</Typography>
    {error && <Alert severity="error" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{error}</Alert>}
    <Alert severity="warning">The answer key is provisional and unreviewed. Ten cases demonstrate evaluation; they do not establish production accuracy.</Alert>
    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
      <Chip label={`${benchmark?.cases.length ?? 0} fixed cases`} /><Chip label="6 categories" /><Chip label="No fine-tuning" />
      <Button onClick={refresh} disabled={busy}>Refresh results</Button>
    </Stack>
    <Tabs value={page} onChange={(_, value) => setPage(value)} variant="scrollable" aria-label="Baseline pages">
      {['Compare runs', 'Benchmark cases', 'Run details'].map(label => <Tab label={label} key={label} />)}
    </Tabs>
    {page === 0 && <>
      <Paper variant="outlined" sx={{ p: 2 }}><Stack spacing={2}>
        <Typography variant="h6" component="h3">Run a baseline</Typography>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
          <TextField select label="Benchmark provider" value={provider} onChange={e => setProvider(e.target.value)} disabled={busy || active} sx={{ minWidth: 260 }}>
            <MenuItem value="hf-base">Local pretrained Qwen base</MenuItem><MenuItem value="fireworks">Hosted Fireworks</MenuItem>
          </TextField>
          <TextField label="Output token budget" type="number" value={budget} onChange={e => setBudget(e.target.value)} disabled={busy || active} slotProps={{ htmlInput: { min: 1, max: 256 } }} />
          <TextField label="Temperature" type="number" value={temperature} onChange={e => setTemperature(e.target.value)} disabled={busy || active} slotProps={{ htmlInput: { min: 0, max: 2, step: 0.1 } }} />
          <TextField select label="Reasoning mode" value={reasoning} onChange={e => setReasoning(e.target.value)} disabled={busy || active} sx={{ minWidth: 230 }}>
            <MenuItem value="disabled">Disable hosted reasoning</MenuItem><MenuItem value="provider-default">Provider default</MenuItem>
          </TextField>
        </Stack>
        {providerInfo && <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>{providerInfo.model} · {providerInfo.configured ? 'Configured' : 'Key missing on server'}</Typography>}
        <Typography variant="body2">Each run evaluates all ten cases. Hosted reasoning can consume the output budget before returning a final answer; its mode is recorded in the protocol. The local base model uses plain text completion. Fireworks uses your configured API account; local inference uses CPU.</Typography>
        <Button variant="contained" onClick={start} disabled={busy || active || !benchmark}>Run benchmark</Button>
        {active && <Typography role="status">A benchmark is running. Progress is saved after every case.</Typography>}
      </Stack></Paper>
      {mixedProtocols && <Alert severity="warning">These runs use different evaluation settings. Compare only rows with the same protocol ID.</Alert>}
      <Typography variant="h6" component="h3">Saved runs</Typography>
      {!runs.length ? <Alert severity="info">No benchmark runs saved yet.</Alert> : <Paper variant="outlined" sx={{ overflowX: 'auto' }}>
        <Table size="small" aria-label="Baseline comparison"><TableHead><TableRow>
          {['Model / protocol', 'Status', 'Micro F1', 'Macro F1', 'Exact match', 'JSON / schema validity', 'Usable / total', 'Mean / p95 time', 'Inspect'].map(name => <TableCell key={name}>{name}</TableCell>)}
        </TableRow></TableHead><TableBody>{runs.map(run => <TableRow key={run.run_id} selected={run.run_id === selected}>
          <TableCell sx={{ maxWidth: 220, overflowWrap: 'anywhere' }}>{run.model_config.model}<Typography variant="caption" component="div">{run.protocol_sha256.slice(0, 12)} · {run.run_id.slice(0, 8)}</Typography></TableCell>
          <TableCell>{run.status}</TableCell><TableCell>{percent(run.summary.labels?.micro.f1)}</TableCell><TableCell>{percent(run.summary.labels?.macro.f1)}</TableCell><TableCell>{percent(run.summary.labels?.exact_match)}</TableCell>
          <TableCell>{percent(run.summary.json_validity)} / {percent(run.summary.schema_validity)}</TableCell><TableCell>{run.summary.usable_responses} / {run.summary.expected_cases}</TableCell>
          <TableCell>{seconds(run.summary.latency_all_attempts.mean_ms)} / {seconds(run.summary.latency_all_attempts.p95_ms)}</TableCell>
          <TableCell><Button onClick={() => { setSelected(run.run_id); setPage(2); }}>View {run.run_id.slice(0, 8)}</Button></TableCell>
        </TableRow>)}</TableBody></Table>
      </Paper>}
      <Typography variant="body2" color="text.secondary">JSON and schema rates use returned responses as their denominator. Label scores include failed attempts as missing predictions. Blocked or unfinished runs have no aggregate label score. Macro F1 averages all seven labels, including labels absent in a category.</Typography>
    </>}
    {page === 1 && benchmark && <>
      <Typography sx={{ overflowWrap: 'anywhere' }}>Frozen benchmark hash: {benchmark.benchmark_sha256}</Typography>
      <Typography>Dataset v1 source/body overlaps: {benchmark.contamination_audit.dataset_v1_source_or_body_overlap.length}. Pretraining overlap: {benchmark.contamination_audit.pretraining_overlap}.</Typography>
      {benchmark.cases.map(item => <Paper component="details" variant="outlined" key={item.recipe.id} sx={{ p: 2 }}>
        <Typography component="summary" sx={{ cursor: 'pointer' }}>{item.recipe.id} · {item.recipe.title} · {item.category}</Typography>
        <Stack spacing={1} sx={{ mt: 2 }}><Typography>{item.challenge}</Typography><Typography>Expected labels: {item.labels.join(', ')}</Typography><Typography>{item.rationale}</Typography>
          <Button component="a" href={item.recipe.source_uri} target="_blank" rel="noreferrer">Read source</Button><Code>{pretty(item.recipe)}</Code>
        </Stack>
      </Paper>)}
    </>}
    {page === 2 && <>
      <TextField select label="Run to inspect" value={selected} onChange={e => setSelected(e.target.value)} disabled={!runs.length}>
        {!runs.length && <MenuItem value="">No runs</MenuItem>}
        {runs.map(run => <MenuItem value={run.run_id} key={run.run_id}>{run.config.provider} · {run.status} · {run.run_id.slice(0, 8)}</MenuItem>)}
      </TextField>
      {detail && <>
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap"><Chip label={detail.status} /><Chip label={`${detail.rows.length} / ${detail.benchmark.cases.length} cases recorded`} />
          <Button component="a" href={`${API}/runs/${detail.run_id}/download`}>Download run JSON</Button></Stack>
        {pending(detail) && <LinearProgress aria-label="Benchmark progress" variant="determinate" value={100 * detail.rows.length / detail.benchmark.cases.length} />}
        {detail.error && <Alert severity="error">{detail.error}</Alert>}
        <Typography sx={{ overflowWrap: 'anywhere' }}>{detail.model_config.model} · {detail.model_config.training_stage}</Typography>
        <Typography>Provider setup: {seconds(detail.setup_latency_ms)}. Request latency excludes that setup.</Typography>
        {metrics ? <>
          <Typography variant="h6" component="h3">Per-label metrics</Typography>
          <Paper variant="outlined" sx={{ overflowX: 'auto' }}><Table size="small" aria-label="Per-label metrics"><TableHead><TableRow>{['Label', 'Support', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'].map(name => <TableCell key={name}>{name}</TableCell>)}</TableRow></TableHead>
            <TableBody>{Object.entries(metrics.per_label).map(([label, value]) => <TableRow key={label}><TableCell>{label}</TableCell>{['support', 'tp', 'fp', 'fn'].map(key => <TableCell key={key}>{value[key]}</TableCell>)}{['precision', 'recall', 'f1'].map(key => <TableCell key={key}>{percent(value[key])}</TableCell>)}</TableRow>)}</TableBody>
          </Table></Paper>
          <Typography variant="h6" component="h3">Category metrics</Typography>
          <Paper variant="outlined" sx={{ overflowX: 'auto' }}><Table size="small" aria-label="Category metrics"><TableHead><TableRow>{['Category', 'Cases', 'Micro F1', 'Macro F1', 'Exact match'].map(name => <TableCell key={name}>{name}</TableCell>)}</TableRow></TableHead><TableBody>
            {Object.entries(detail.summary.categories).map(([name, value]) => <TableRow key={name}><TableCell>{name}</TableCell><TableCell>{value.samples}</TableCell><TableCell>{percent(value.micro.f1)}</TableCell><TableCell>{percent(value.macro.f1)}</TableCell><TableCell>{percent(value.exact_match)}</TableCell></TableRow>)}
          </TableBody></Table></Paper>
        </> : <Alert severity="info">Aggregate label scores are available only when all cases have been attempted. Missing scores are not zero scores.</Alert>}
        {!!detail.rows.length && <TextField select label="Case to inspect" value={caseId} onChange={e => setCaseId(e.target.value)}>{detail.rows.map(item => <MenuItem key={item.case_id} value={item.case_id}>{item.case_id} · {item.category} · {item.usable ? 'usable' : 'failed'}</MenuItem>)}</TextField>}
        {row && <>
          <Typography>Expected: {row.expected_labels.join(', ')}. Predicted: {row.prediction?.labels.join(', ') || 'No valid label set'}.</Typography>
          <Typography>JSON: {row.json_valid ? 'valid' : 'invalid or absent'} · Schema: {row.schema_valid ? 'valid' : 'invalid or absent'} · Finish: {row.finish_reason ?? 'no response'} · Time: {seconds(row.latency_ms)}</Typography>
          {row.error && <Alert severity="error" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{row.error}</Alert>}
          <Typography variant="h6" component="h3">Raw model answer</Typography><Code>{row.raw_output ?? 'No answer returned.'}</Code>
          <Box component="details"><Typography component="summary">Exact input messages</Typography><Code>{pretty(row.messages)}</Code>{row.rendered_prompt && <Code>{row.rendered_prompt}</Code>}</Box>
        </>}
        <Box component="details"><Typography component="summary">Full run evidence and settings</Typography><Code>{pretty(detail)}</Code></Box>
      </>}
    </>}
  </Stack>;
}
