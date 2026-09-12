import { useEffect, useMemo, useState } from 'react';
import { Alert, Box, Button, Checkbox, Chip, FormControlLabel, LinearProgress, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material';

const endpoint = '/api/v1/training/lora';
const number = (value, digits = 1) => typeof value === 'number' ? value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
const percent = value => typeof value === 'number' ? `${number(value * 100, 2)}%` : '—';
async function api(url, body) {
  const response = await fetch(url, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let result;
  try { result = await response.json(); } catch { throw new Error(`HTTP ${response.status}: backend did not return JSON.`); }
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result));
  return result;
}
const fields = [
  ['epochs', 'Epochs', 1, 20, 1, 'Passes through the training split.'],
  ['learning_rate', 'Learning rate', 0.000001, 0.01, 'any', 'Constant AdamW update scale.'],
  ['sequence_length', 'Sequence length', 128, 2048, 1, 'Prompt + JSON answer + EOS. Overlength examples fail.'],
  ['batch_size', 'Microbatch size', 1, 4, 1, 'Examples per forward/backward pass.'],
  ['gradient_accumulation_steps', 'Gradient accumulation', 1, 32, 1, 'Microbatches before one optimizer update.'],
  ['seed', 'Seed', 0, 4294967295, 1, 'Same initialization/shuffle seed for each rank.'],
  ['lora_dropout', 'LoRA dropout', 0, 0.5, 0.05, 'Drop inputs to the adapter branch only during training.'],
];

export default function LoRATraining() {
  const [architecture, setArchitecture] = useState(null);
  const [config, setConfig] = useState(null);
  const [versions, setVersions] = useState([]);
  const [ranks, setRanks] = useState([4, 16]);
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const [refreshError, setRefreshError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let alive = true;
    Promise.all([api(`${endpoint}/architecture`), api('/api/v1/training/options'), api('/api/v1/datasets/versions'), api(`${endpoint}/experiments`)]).then(([report, settings, datasets, history]) => {
      if (!alive) return;
      if (!report.projections || !settings.defaults || !settings.seed_dataset || !Array.isArray(datasets) || !Array.isArray(history)) throw new Error('Unexpected LoRA configuration response from the API');
      setArchitecture(report);
      setConfig({ ...settings.defaults, provider: 'local', lora_alpha: null, lora_target_policy: 'query-value', lora_dropout: 0, fireworks_model: null, fireworks_deployment_shape: null });
      setVersions([settings.seed_dataset, ...datasets.filter(row => row.version !== settings.seed_dataset.version)]);
      setExperiments(history); setSelected(history[0]?.experiment_id || '');
    }).catch(e => alive && setError(e.message));
    const timer = setInterval(async () => {
      try {
        const history = await api(`${endpoint}/experiments`);
        if (!Array.isArray(history)) throw new Error('Unexpected experiment history response');
        if (alive) { setExperiments(history); setRefreshError(''); }
      } catch (e) { if (alive) setRefreshError(e.message); }
    }, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  const projections = useMemo(() => {
    const groups = new Map();
    for (const row of architecture?.projections || []) {
      const key = `${row.role}:${row.weight_shape}`;
      if (!groups.has(key)) groups.set(key, { ...row, count: 0 });
      groups.get(key).count += 1;
    }
    return [...groups.values()];
  }, [architecture]);
  const selectedPaths = architecture?.projections.filter(row => row.eligible && architecture.policies[config?.lora_target_policy]?.includes(row.role)) || [];
  const parameterFactor = selectedPaths.reduce((sum, row) => sum + row.in_features + row.out_features, 0);
  const experiment = experiments.find(row => row.experiment_id === selected);
  const active = experiments.some(row => ['queued', 'running'].includes(row.status));
  const dataset = versions.find(row => row.version === config?.dataset_version);
  const update = (key, value) => setConfig(previous => ({ ...previous, [key]: value }));
  const start = async event => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const result = await api(`${endpoint}/experiments`, { training: config, ranks: [...ranks].sort((a, b) => a - b) });
      setExperiments(await api(`${endpoint}/experiments`)); setSelected(result.experiment_id);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };
  return <Stack spacing={3}>
    <Box><Typography component="h2" variant="h5">LoRA Training</Typography><Typography color="text.secondary">Learn a small update to a frozen model. Compare predeclared ranks on the same recipes and benchmark.</Typography></Box>
    {error && <Alert severity="error">{error}</Alert>}
    {refreshError && <Alert severity="warning">{refreshError} Showing last saved evidence; retrying automatically.</Alert>}
    {!config && !error && <LinearProgress aria-label="Loading LoRA configuration" />}
    {architecture && <Paper variant="outlined" sx={{ p: 3 }}><Stack spacing={2}>
      <Typography component="h3" variant="h6">Inspected model architecture</Typography>
      <Typography>{architecture.model} · {architecture.layers} Transformer blocks · {number(architecture.base_parameters, 0)} frozen base parameters</Typography>
      <Typography color="text.secondary">Qwen uses {architecture.attention_heads} query heads and {architecture.kv_heads} shared key/value heads. That is why the key and value matrices are narrower than the query matrix. These shapes come from the instantiated model tree; every training run checks it again.</Typography>
      <TableContainer><Table size="small" aria-label="Inspected Transformer projections"><TableHead><TableRow>{['Projection', 'Weight [out, in]', 'Count', 'Selected'].map(label => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead><TableBody>
        {projections.map(row => <TableRow key={row.role}><TableCell>{row.role}</TableCell><TableCell>{row.weight_shape.join(' × ')}</TableCell><TableCell>{row.count}</TableCell><TableCell>{row.eligible ? architecture.policies[config?.lora_target_policy]?.includes(row.role) ? 'Yes' : 'No' : 'Excluded: tied output head'}</TableCell></TableRow>)}
      </TableBody></Table></TableContainer>
      <Box component="details"><Typography component="summary" sx={{ cursor: 'pointer' }}>Show {selectedPaths.length} exact target module paths</Typography><Box component="pre" sx={{ maxHeight: 220, overflow: 'auto', fontSize: 12 }}>{selectedPaths.map(row => row.name).join('\n')}</Box><Typography variant="caption">Base revision: {architecture.revision}</Typography></Box>
    </Stack></Paper>}
    {config && <Paper variant="outlined" sx={{ p: 3 }} component="form" onSubmit={start}><Stack spacing={3}>
      <Typography component="h3" variant="h6">Controlled rank experiment</Typography>
      <TextField select label="Dataset version" value={config.dataset_version} onChange={e => update('dataset_version', e.target.value)}>{versions.map(row => <MenuItem key={row.version} value={row.version}>{row.version.slice(0, 19)}… · {row.counts.train}/{row.counts.validation}/{row.counts.test} · {row.status}</MenuItem>)}</TextField>
      <Alert severity="warning">{dataset?.counts.train} train / {dataset?.counts.validation} validation / {dataset?.counts.test} test recipes. The teaching seed has provisional labels and limited coverage. Test and benchmark answers never enter training or checkpoint selection.</Alert>
      <TextField select label="Target projections" value={config.lora_target_policy} onChange={e => update('lora_target_policy', e.target.value)}>
        <MenuItem value="query-value">Query + value</MenuItem><MenuItem value="attention">All attention projections</MenuItem><MenuItem value="attention-and-mlp">Attention + MLP projections</MenuItem>
      </TextField>
      <Box><Typography>Choose at least two ranks. Alpha = 2 × rank keeps the scaling factor alpha/r = 2.</Typography><Stack direction="row" useFlexGap sx={{ flexWrap: 'wrap' }}>{[4, 8, 16, 32].map(rank => <FormControlLabel key={rank} control={<Checkbox checked={ranks.includes(rank)} onChange={e => setRanks(previous => e.target.checked ? [...previous, rank] : previous.filter(value => value !== rank))} />} label={`Rank ${rank} · ${number(rank * parameterFactor, 0)} trainable`} />)}</Stack></Box>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr', xl: '1fr 1fr 1fr' }, gap: 2 }}>{fields.map(([key, label, min, max, step, help]) => <TextField key={key} required type="number" label={label} value={config[key]} inputProps={{ min, max, step }} helperText={help} onChange={e => update(key, e.target.value === '' ? '' : Number(e.target.value))} />)}</Box>
      <Typography>Effective batch: {config.batch_size * config.gradient_accumulation_steps} examples per update; the final update can be smaller. Each rank starts a fresh CPU process sequentially.</Typography>
      <Box component="details"><Typography component="summary" sx={{ cursor: 'pointer' }}>Equations and fixed settings</Typography><Typography sx={{ mt: 1 }}>y = Wx + b + (alpha/r)BA·dropout(x). W and bias stay frozen. A is [rank, input]; B is [output, rank]. Trainable count per projection = rank × (input + output). B starts at zero, so the initial model output is unchanged.</Typography><Typography sx={{ mt: 1 }}>CPU float32; AdamW; constant learning rate; no warmup or weight decay; gradient clipping at 1; activation checkpointing. Mask prompt and padding labels with −100. Save/evaluate each epoch, choose lowest validation loss, keep at most two trainer checkpoints. Benchmark the selected adapter with greedy decoding and 128 output tokens on ten fixed cases.</Typography></Box>
      <Button type="submit" variant="contained" disabled={busy || active || ranks.length < 2} sx={{ alignSelf: 'flex-start' }}>{busy ? 'Starting…' : 'Run rank experiment'}</Button>
      {ranks.length < 2 && <Typography color="error">Choose at least two ranks.</Typography>}
      {active && <Typography color="text.secondary">An experiment is active. Progress and failures appear below.</Typography>}
    </Stack></Paper>}
    <Paper variant="outlined" sx={{ p: 3 }}><Stack spacing={2}>
      <Typography component="h3" variant="h6">Saved rank comparisons</Typography>
      {!experiments.length ? <Typography>No experiments yet. Opening this page does not start training.</Typography> : <TextField select label="LoRA experiment" value={selected} onChange={e => setSelected(e.target.value)}>{experiments.map(row => <MenuItem key={row.experiment_id} value={row.experiment_id}>Ranks {row.config.ranks.join(', ')} · {row.status} · {row.experiment_id.slice(0, 8)}</MenuItem>)}</TextField>}
      {experiment && <>
        <Stack direction="row" spacing={2} sx={{ alignItems: 'center' }}><Chip label={experiment.status} color={experiment.status === 'completed' ? 'success' : 'default'} /><Button component="a" href={`${endpoint}/experiments/${experiment.experiment_id}/download`}>Download comparison evidence</Button></Stack>
        {['queued', 'running'].includes(experiment.status) && <LinearProgress aria-label="Rank experiment active" />}
        {experiment.error && <Alert severity="error">{experiment.error}</Alert>}
        {experiment.comparison && <Alert severity="info">{experiment.comparison.claim}</Alert>}
        <TableContainer><Table size="small" aria-label="Rank experiment results"><TableHead><TableRow>{['Rank / alpha', 'Status', 'Trainable', 'Train time (s)', 'Peak RSS (MiB)', 'Micro-F1', 'Macro-F1', 'Exact match', 'Valid JSON', 'Best val loss'].map(label => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead><TableBody>{experiment.runs.map(row => <TableRow key={row.run_id}>
          <TableCell>{row.rank} / {row.alpha}</TableCell><TableCell>{row.status}</TableCell><TableCell>{number(row.trainable_parameters, 0)}</TableCell><TableCell>{number(row.training_seconds)}</TableCell><TableCell>{number(row.peak_rss_mib)}</TableCell><TableCell>{percent(row.micro_f1)}</TableCell><TableCell>{percent(row.macro_f1)}</TableCell><TableCell>{percent(row.exact_match)}</TableCell><TableCell>{percent(row.json_validity)}</TableCell><TableCell>{number(row.best_validation_loss, 4)}</TableCell>
        </TableRow>)}</TableBody></Table></TableContainer>
        <Typography color="text.secondary">Memory is total process RSS sampled every 20 ms during training, including base weights and activations. Short peaks may be missed. Time includes epoch validation and checkpoint writes, excludes both benchmarks. One run per rank cannot establish a speed or memory advantage.</Typography>
        {experiment.runs.map(row => <Box component="details" key={row.run_id}><Typography component="summary" sx={{ cursor: 'pointer' }}>Rank {row.rank} · {row.phase} · loss and baseline evidence</Typography><Stack spacing={2} sx={{ pt: 2 }}>
          {row.error && <Alert severity="error">{row.error}</Alert>}
          <Typography>Start RSS: {number(row.start_rss_mib)} MiB · Peak increase: {number(row.peak_rss_increase_mib)} MiB · Optimizer steps: {row.optimizer_steps ?? '—'}</Typography>
          {row.baseline_comparison && <Alert severity="info">{row.baseline_comparison.claim} Base Micro-F1: {percent(row.baseline_comparison.before?.micro_f1)} → adapter: {percent(row.baseline_comparison.after?.micro_f1)}.</Alert>}
          <Button component="a" href={`/api/v1/training/runs/${row.run_id}/download`} sx={{ alignSelf: 'flex-start' }}>Download rank {row.rank} full run</Button>
          <TableContainer><Table size="small" aria-label={`Rank ${row.rank} loss history`}><TableHead><TableRow>{['Step', 'Train loss', 'Validation loss', 'Gradient norm'].map(label => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead><TableBody>{(row.history || []).filter(item => item.loss !== undefined || item.eval_loss !== undefined).map((item, index) => <TableRow key={index}><TableCell>{item.step}</TableCell><TableCell>{number(item.loss, 4)}</TableCell><TableCell>{number(item.eval_loss, 4)}</TableCell><TableCell>{number(item.grad_norm, 4)}</TableCell></TableRow>)}</TableBody></Table></TableContainer>
        </Stack></Box>)}
      </>}
    </Stack></Paper>
  </Stack>;
}
