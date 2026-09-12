import { useEffect, useState } from 'react';
import { Alert, Box, Button, Checkbox, Chip, FormControlLabel, LinearProgress, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material';
import { api, numeric, percent } from './experimentApi';

export default function ExperimentStudio({ qlora = false }) {
  const [options, setOptions] = useState(null), [history, setHistory] = useState([]), [selected, setSelected] = useState('');
  const [rates, setRates] = useState(qlora ? '0.0002' : '0.00005, 0.0002'), [ranks, setRanks] = useState(qlora ? [4] : [4, 16]);
  const [checkpointing, setCheckpointing] = useState(true), [tracking, setTracking] = useState(true);
  const [error, setError] = useState(''), [refreshError, setRefreshError] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => {
    let alive = true;
    Promise.all([api('experiments/options'), api('experiments')]).then(([settings, rows]) => {
      if (!alive) return;
      if (!settings.defaults?.runs || !Array.isArray(rows)) throw new Error('Unexpected experiment registry response');
      setOptions(settings); setHistory(rows); setSelected(rows[0]?.experiment_id || '');
    }).catch(e => alive && setError(e.message));
    const timer = setInterval(async () => {
      try { const rows = await api('experiments'); if (!Array.isArray(rows)) throw new Error('Invalid registry response'); if (alive) { setHistory(rows); setRefreshError(''); } }
      catch (e) { if (alive) setRefreshError(e.message); }
    }, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  const values = rates.split(',').map(value => Number(value.trim()));
  const valid = values.length > 0 && values.every(value => value > 0 && value <= 0.01) && new Set(values).size === values.length && ranks.length > 0;
  const common = options?.defaults.runs[0];
  const runs = !common || !valid ? [] : qlora ? ['lora', 'qlora'].map(method => ({ ...common, method, learning_rate: values[0], lora_rank: ranks[0], gradient_checkpointing: checkpointing })) : values.flatMap(learning_rate => ranks.map(lora_rank => ({ ...common, method: 'lora', learning_rate, lora_rank, gradient_checkpointing: checkpointing })));
  const config = { name: qlora ? 'Matched LoRA vs QLoRA comparison' : 'Learning-rate × rank matrix', runs, mlflow: tracking };
  const active = history.some(row => ['queued', 'running'].includes(row.status));
  const experiment = history.find(row => row.experiment_id === selected);
  const submit = async event => {
    event.preventDefault(); setError(''); setBusy(true);
    try { const value = await api('experiments', config); setHistory(await api('experiments')); setSelected(value.experiment_id); }
    catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  return <Stack spacing={3}>
    <Box><Typography component="h2" variant="h5">{qlora ? 'QLoRA Training' : 'Experiment Registry'}</Typography><Typography color="text.secondary">{qlora ? 'Compare frozen float32 weights with NF4 storage. Both methods train the same LoRA adapters on CPU.' : 'Compare a predeclared matrix. PostgreSQL keeps the evidence; MLflow receives parameters, losses and final metrics.'}</Typography></Box>
    {error && <Alert severity="error">{error}</Alert>}{refreshError && <Alert severity="warning">{refreshError} Last saved records remain visible.</Alert>}
    {!options && !error && <LinearProgress aria-label="Loading experiment settings" />}
    {options && <Paper component="form" onSubmit={submit} variant="outlined" sx={{ p: 3 }}><Stack spacing={2}>
      {qlora && <Alert severity={options.qlora.supported ? 'info' : 'error'}>{options.qlora.supported ? `CPU NF4 forward/backward probe passed · bitsandbytes ${options.qlora.bitsandbytes}. 4-bit NF4 + double quantization; float32 computation; embeddings and norms stay float32.` : options.qlora.error}</Alert>}
      <Typography>Fixed dataset: {common.dataset_version.slice(0, 20)}… · Same seed, target modules, training budget and ten-case benchmark for every run.</Typography>
      <TextField required label={qlora ? 'Learning rate' : 'Learning rates (comma separated)'} value={rates} onChange={e => setRates(e.target.value)} helperText="Positive values up to 0.01. Lower is not automatically better." />
      {qlora ? <TextField select label="Matched LoRA rank" value={ranks[0]} onChange={e => setRanks([Number(e.target.value)])}>{[4, 8, 16, 32].map(rank => <MenuItem key={rank} value={rank}>{rank}</MenuItem>)}</TextField> : <Stack direction="row" useFlexGap sx={{ flexWrap: 'wrap' }}>{[4, 8, 16, 32].map(rank => <FormControlLabel key={rank} label={`Rank ${rank}`} control={<Checkbox checked={ranks.includes(rank)} onChange={e => setRanks(previous => e.target.checked ? [...previous, rank].sort((a, b) => a - b) : previous.filter(value => value !== rank))} />} />)}</Stack>}
      <FormControlLabel label="Gradient checkpointing for every run" control={<Checkbox checked={checkpointing} onChange={e => setCheckpointing(e.target.checked)} />} />
      <Typography variant="body2" color="text.secondary">Checkpointing recomputes activations during backward to save memory. This is separate from saving adapter checkpoints to disk.</Typography>
      <FormControlLabel label="Export results to MLflow" control={<Checkbox checked={tracking} onChange={e => setTracking(e.target.checked)} />} />
      <Typography>{runs.length} runs · {common.epochs} epochs · microbatch {common.batch_size} · accumulation {common.gradient_accumulation_steps} · alpha/r = 2 · seed {common.seed}</Typography>
      <Alert severity="warning">One validation recipe and ten provisional benchmark cases support learning, not a general model ranking. Changing settings after viewing this benchmark makes the experiment exploratory.</Alert>
      <Box component="details"><Typography component="summary">Full configuration for CLI or review</Typography><Box component="pre" sx={{ overflow: 'auto', maxHeight: 320, fontSize: 12 }}>{JSON.stringify(config, null, 2)}</Box></Box>
      <Button type="submit" variant="contained" disabled={busy || active || !valid || runs.length < 2 || runs.length > 8 || (qlora && (values.length !== 1 || !options.qlora.supported))} sx={{ alignSelf: 'flex-start' }}>{busy ? 'Queuing…' : qlora ? 'Run LoRA and QLoRA' : 'Run experiment matrix'}</Button>
      {(!valid || runs.length < 2 || runs.length > 8) && <Typography color="error">Choose 2–8 unique combinations with valid learning rates.</Typography>}
    </Stack></Paper>}
    <Paper variant="outlined" sx={{ p: 3 }}><Stack spacing={2}>
      <Typography component="h3" variant="h6">Saved comparisons</Typography>
      {!history.length ? <Typography>No experiments yet. Opening the page starts no training.</Typography> : <TextField select label="Experiment" value={selected} onChange={e => setSelected(e.target.value)}>{history.map(row => <MenuItem key={row.experiment_id} value={row.experiment_id}>{row.name} · {row.status}</MenuItem>)}</TextField>}
      {experiment && <>
        <Stack direction="row" spacing={1}><Chip label={experiment.status} /><Button component="a" href={`/api/v1/experiments/${experiment.experiment_id}/download`}>Download experiment JSON</Button></Stack>
        {['queued', 'running'].includes(experiment.status) && <LinearProgress aria-label="Matrix running" />}
        <Typography>{experiment.phase}</Typography>{experiment.error && <Alert severity="error">{experiment.error}</Alert>}
        {experiment.comparison && <Alert severity="info">{experiment.comparison.claim}</Alert>}
        <TableContainer><Table size="small" aria-label="Experiment comparison"><TableHead><TableRow>{['Method', 'Learning rate', 'Rank', 'Status', 'Trainable', 'Train s', 'Peak RSS MiB', 'Micro-F1', 'Exact match', 'Schema valid', 'MLflow'].map(label => <TableCell key={label}>{label}</TableCell>)}</TableRow></TableHead><TableBody>{experiment.runs.map(row => <TableRow key={row.run_id}><TableCell>{row.method}</TableCell><TableCell>{row.learning_rate}</TableCell><TableCell>{row.rank}</TableCell><TableCell>{row.status}</TableCell><TableCell>{numeric(row.trainable_parameters, 0)}</TableCell><TableCell>{numeric(row.training_seconds)}</TableCell><TableCell>{numeric(row.peak_rss_mib)}</TableCell><TableCell>{percent(row.micro_f1)}</TableCell><TableCell>{percent(row.exact_match)}</TableCell><TableCell>{percent(row.schema_validity)}</TableCell><TableCell>{row.tracking?.status || '—'}</TableCell></TableRow>)}</TableBody></Table></TableContainer>
        <Typography color="text.secondary">Peak RSS is total process memory sampled every 20 ms during training. Time includes epoch validation and checkpoint writes. Base loading and benchmarks are outside this interval. One run per setting cannot establish speed or memory superiority.</Typography>
        {experiment.comparison?.effects?.map((effect, index) => <Typography key={index}>{effect.factor}: {effect.from_value} → {effect.to_value}: {effect.effect} ({(100 * effect.micro_f1_delta).toFixed(2)} percentage points Micro-F1). Runs {effect.from_run.slice(0, 8)} → {effect.to_run.slice(0, 8)}.</Typography>)}
        {experiment.runs.map(row => <Box component="details" key={row.run_id}><Typography component="summary">{row.method} · lr {row.learning_rate} · rank {row.rank} · run {row.run_id.slice(0, 8)}</Typography><Stack spacing={1} sx={{ pt: 1 }}>
          {row.error && <Alert severity="error">{row.error}</Alert>}{row.tracking?.error && <Alert severity="warning">{row.tracking.error}</Alert>}
          <Typography>MLflow run: {row.tracking?.run_id || 'Not synced'} · Best validation loss: {numeric(row.best_validation_loss, 4)}</Typography>
          <Typography>{row.baseline_comparison?.claim}</Typography><Button component="a" href={`/api/v1/training/runs/${row.run_id}/download`}>Download full run evidence</Button>
          <Box component="pre" sx={{ overflow: 'auto', maxHeight: 250, fontSize: 12 }}>{JSON.stringify(row.history, null, 2)}</Box>
        </Stack></Box>)}
      </>}
    </Stack></Paper>
  </Stack>;
}
