import { useEffect, useState } from 'react';
import { Alert, Box, Button, Chip, Divider, LinearProgress, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material';

const fields = [
  ['epochs', 'Epochs', 'Full passes through the training split.', 1, 20, 1],
  ['learning_rate', 'Learning rate', 'Adapter update scale. Default 0.0002; constant schedule.', 0.000001, 0.01, 'any'],
  ['sequence_length', 'Sequence length', 'Prompt + answer + EOS tokens. Local overlength examples fail instead of truncating.', 128, 2048, 1],
  ['batch_size', 'Microbatch size', 'Examples per local forward/backward pass.', 1, 4, 1],
  ['gradient_accumulation_steps', 'Gradient accumulation', 'Microbatches accumulated before an optimizer update.', 1, 32, 1],
  ['seed', 'Seed', 'Local initialization and shuffle seed. Managed reproducibility is provider controlled.', 0, 4294967295, 1],
];
const activeStates = ['queued', 'preparing', 'baseline', 'training', 'evaluating', 'submitting', 'monitoring'];
const number = value => typeof value === 'number' ? value.toFixed(4) : '—';
const percent = value => typeof value === 'number' ? `${(100 * value).toFixed(1)}%` : '—';
async function api(path, body) {
  const response = await fetch(`/api/v1/${path}`, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`API returned HTTP ${response.status} without JSON. Check the backend connection.`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

export default function TrainingMonitor() {
  const [options, setOptions] = useState(null);
  const [config, setConfig] = useState(null);
  const [versions, setVersions] = useState([]);
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState('');
  const [run, setRun] = useState(null);
  const [error, setError] = useState('');
  const [refreshError, setRefreshError] = useState('');
  const [busy, setBusy] = useState(false);
  const [check, setCheck] = useState(null);

  useEffect(() => {
    let alive = true;
    Promise.all([api('training/options'), api('datasets/versions'), api('training/runs')]).then(([settings, datasets, history]) => {
      if (!alive) return;
      if (!settings.defaults || !settings.seed_dataset || !Array.isArray(datasets) || !Array.isArray(history)) throw new Error('Unexpected training configuration response from the API');
      setOptions(settings);
      setConfig({ ...settings.defaults, fireworks_model: settings.fireworks_model || null, fireworks_deployment_shape: settings.fireworks_deployment_shape || null });
      setVersions([settings.seed_dataset, ...datasets.filter(row => row.version !== settings.seed_dataset.version)]);
      setRuns(history);
      setSelected(history[0]?.run_id || '');
    }).catch(e => alive && setError(e.message));
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    setRun(null);
    const refresh = async () => {
      try {
        const [history, detail] = await Promise.all([api('training/runs'), selected ? api(`training/runs/${selected}`) : Promise.resolve(null)]);
        if (alive) { setRuns(history); setRun(detail); setRefreshError(''); }
      } catch (e) { if (alive) setRefreshError(e.message); }
    };
    if (selected) refresh();
    const timer = setInterval(refresh, 3000);
    return () => { alive = false; clearInterval(timer); };
  }, [selected]);

  const update = (key, value) => { setConfig(current => ({ ...current, [key]: value })); setCheck(null); };
  const submit = async (action) => {
    setBusy(true); setError('');
    try {
      if (action === 'preflight') setCheck(await api('training/preflight', config));
      else {
        const result = await api(action === 'resume' ? `training/runs/${selected}/resume` : 'training/runs', action === 'resume' ? {} : config);
        setSelected(result.run_id);
        setRun(await api(`training/runs/${result.run_id}`));
        setRuns(await api('training/runs'));
      }
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };
  const currentDataset = versions.find(row => row.version === config?.dataset_version);
  const workerActive = runs.some(row => activeStates.includes(row.status));

  return <Stack spacing={3}>
    <Box><Typography component="h2" variant="h5">Training Monitor</Typography>
      <Typography color="text.secondary">Supervised fine-tuning with a fixed dataset and a measured before-and-after benchmark.</Typography></Box>
    {error && <Alert severity="error">{error}</Alert>}
    {refreshError && <Alert severity="warning">{refreshError} The last loaded record is retained; refreshing automatically.</Alert>}
    {!options && !error && <LinearProgress aria-label="Loading training configuration" />}
    {config && <Paper variant="outlined" sx={{ p: 3 }} component="form" onSubmit={event => { event.preventDefault(); submit('start'); }}>
      <Stack spacing={3}>
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1fr 2fr' }, gap: 2 }}>
          <TextField select label="Training provider" value={config.provider} onChange={e => update('provider', e.target.value)}>
            <MenuItem value="local">Local · Qwen 0.5B + LoRA</MenuItem><MenuItem value="fireworks">Managed · Fireworks</MenuItem>
          </TextField>
          <TextField select label="Dataset version" value={config.dataset_version} onChange={e => update('dataset_version', e.target.value)}>
            {versions.map(row => <MenuItem key={row.version} value={row.version}>{row.version.slice(0, 19)}… · {row.counts.train}/{row.counts.validation}/{row.counts.test} · {row.status}</MenuItem>)}
          </TextField>
        </Box>
        <Alert severity="warning">{currentDataset?.counts.train} train / {currentDataset?.counts.validation} validation / {currentDataset?.counts.test} test recipes. {currentDataset?.status === 'teaching-draft' ? 'Annotations are provisional. This is a learning run.' : 'Check coverage before interpreting scores.'} Test recipes and benchmark answers are excluded from training and checkpoint selection.</Alert>
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr', xl: '1fr 1fr 1fr' }, gap: 2 }}>
          {fields.map(([key, label, help, min, max, step]) => <TextField key={key} type="number" required label={label} value={config[key]}
            onChange={e => update(key, e.target.value === '' ? '' : Number(e.target.value))} helperText={help} inputProps={{ min, max, step }} />)}
          <TextField select label="LoRA rank" value={config.lora_rank} helperText="Adapter capacity. Local alpha = 2 × rank; targets q_proj and v_proj." onChange={e => update('lora_rank', Number(e.target.value))}>
            {[4, 8, 16, 32].map(value => <MenuItem key={value} value={value}>{value}</MenuItem>)}
          </TextField>
        </Box>
        <Typography>Effective batch: <strong>{config.batch_size * config.gradient_accumulation_steps}</strong> examples per optimizer update (local single device). The last update in an epoch can be smaller.</Typography>
        {config.provider === 'fireworks' && <>
          <TextField label="Fireworks tunable model" value={config.fireworks_model || ''} onChange={e => update('fireworks_model', e.target.value || null)} helperText="Full accounts/…/models/… resource. Hosted inference availability does not imply tuning support." />
          <TextField label="Fireworks deployment shape" value={config.fireworks_deployment_shape || ''} onChange={e => update('fireworks_deployment_shape', e.target.value || null)} helperText="Compatible accounts/…/deploymentShapes/… resource for temporary preemptible evaluation." />
          <Alert severity="info">Fireworks receives batchSizeSamples = {config.batch_size * config.gradient_accumulation_steps}. It controls microbatching, optimizer details, tokenization and checkpoints; the local seed is not sent. Managed training and evaluation use your account resources. Temporary evaluation deployments are deleted afterwards.</Alert>
          <Button sx={{ alignSelf: 'flex-start' }} onClick={() => submit('preflight')} disabled={busy || !options.fireworks_configured}>Check Fireworks support</Button>
          {!options.fireworks_configured && <Alert severity="warning">Configure FIREWORKS_API_KEY in the server .env file.</Alert>}
          {check && <Alert severity={check.supported ? 'success' : 'warning'}>{check.supported ? 'Provider prerequisites passed. No datasets uploaded and no training started.' : check.issues.join(' ')}</Alert>}
        </>}
        <Box component="details"><Typography component="summary" sx={{ cursor: 'pointer' }}>Fixed settings and parameter meanings</Typography>
          <Typography sx={{ mt: 1 }}>Local: CPU float32, AdamW, constant learning rate, no warmup or weight decay, gradient clipping at norm 1. LoRA dropout 0. Prompt and padding labels are −100; JSON answer and EOS tokens contribute to loss. No sequence packing. Activation checkpointing trades speed for memory. Evaluate and save each epoch; retain up to two checkpoints and export the one with lowest validation loss. Benchmark: greedy decoding, 128 output tokens, ten unchanged cases.</Typography>
        </Box>
        <Button type="submit" variant="contained" disabled={busy || workerActive || (config.provider === 'fireworks' && (!options.fireworks_configured || check?.supported === false))} sx={{ alignSelf: 'flex-start' }}>
          {busy ? 'Working…' : config.provider === 'local' ? 'Start local SFT' : 'Start managed SFT'}
        </Button>
        {workerActive && <Typography color="text.secondary">A worker is active. Follow its progress below.</Typography>}
      </Stack>
    </Paper>}

    <Paper variant="outlined" sx={{ p: 3 }}>
      <Stack spacing={2}>
        <Typography component="h2" variant="h6">Saved training runs</Typography>
        {!runs.length ? <Typography>No runs yet. Opening this page does not start training.</Typography> : <TextField select label="Training run" value={selected} onChange={e => setSelected(e.target.value)}>
          {runs.map(row => <MenuItem key={row.run_id} value={row.run_id}>{row.config.provider} · {row.status} · {row.run_id.slice(0, 8)}</MenuItem>)}
        </TextField>}
        {run && <>
          <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap', alignItems: 'center' }}>
            <Chip label={run.status} color={run.status === 'completed' ? 'success' : 'default'} /><Typography>{run.phase}</Typography>
            <Button component="a" href={`/api/v1/training/runs/${run.run_id}/download`}>Download run evidence</Button>
            {run.config.provider === 'fireworks' && ['waiting', 'evaluation-blocked'].includes(run.status) && <Button disabled={busy || workerActive} onClick={() => submit('resume')}>Resume monitoring</Button>}
          </Stack>
          {activeStates.includes(run.status) && <LinearProgress aria-label="Training worker active" />}
          {run.error && <Alert severity="warning">{run.error}</Alert>}
          {run.managed?.job && <Typography>Managed job: {run.managed.job.state} · {run.managed.job.jobProgress?.percent ?? '—'}% · {run.managed.job_name}</Typography>}
          {run.managed?.cleanup_error && <Alert severity="error">Deployment cleanup needs attention: {run.managed.active_deployment}. Resume monitoring to retry cleanup.</Alert>}
          {run.config.provider === 'fireworks' && <Typography color="text.secondary">Remote loss curves and checkpoints are available in the Fireworks console when provided. This monitor records job progress; it does not invent missing loss values.</Typography>}
          <Typography>Optimizer steps: {run.optimizer_steps ?? run.history?.at(-1)?.step ?? '—'} · Initial validation loss: {number(run.initial_validation_loss)} · Best: {number(run.best_validation_loss)} · Selected: {run.selected_checkpoint || '—'}</Typography>
          {!!run.history?.length && <TableContainer><Table size="small" aria-label="Loss by optimizer step"><TableHead><TableRow>{['Step', 'Epoch', 'Training loss', 'Validation loss', 'Gradient norm', 'Learning rate'].map(value => <TableCell key={value}>{value}</TableCell>)}</TableRow></TableHead><TableBody>
            {run.history.filter(row => row.loss !== undefined || row.eval_loss !== undefined).map((row, index) => <TableRow key={index}><TableCell>{row.step}</TableCell><TableCell>{number(row.epoch)}</TableCell><TableCell>{number(row.loss)}</TableCell><TableCell>{number(row.eval_loss)}</TableCell><TableCell>{number(row.grad_norm)}</TableCell><TableCell>{row.learning_rate ?? '—'}</TableCell></TableRow>)}
          </TableBody></Table></TableContainer>}
          {run.comparison && <>
            <Divider /><Typography component="h3" variant="h6">Fixed benchmark comparison</Typography>
            <Alert severity="info">{run.comparison.claim}</Alert>
            {run.comparison.comparable && <TableContainer><Table size="small" aria-label="Benchmark before and after"><TableHead><TableRow>{['Metric', 'Before SFT', 'After SFT', 'Change (percentage points)'].map(value => <TableCell key={value}>{value}</TableCell>)}</TableRow></TableHead><TableBody>
              {Object.keys(run.comparison.before).map(key => <TableRow key={key}><TableCell>{key.replaceAll('_', ' ')}</TableCell><TableCell>{percent(run.comparison.before[key])}</TableCell><TableCell>{percent(run.comparison.after[key])}</TableCell><TableCell>{(100 * run.comparison.delta[key]).toFixed(1)}</TableCell></TableRow>)}
            </TableBody></Table></TableContainer>}
          </>}
          {['baseline', 'benchmark'].filter(key => run[key]).map(key => <Box component="details" key={key}><Typography component="summary" sx={{ cursor: 'pointer' }}>{key === 'baseline' ? 'Before SFT' : 'After SFT'} · {run[key].rows.length}/10 cases · raw evidence, per-label scores and latency</Typography>
            <Box component="pre" sx={{ p: 2, maxHeight: 420, overflow: 'auto', bgcolor: '#f3f6fb', fontSize: 12 }}>{JSON.stringify(run[key], null, 2)}</Box>
          </Box>)}
          <Box component="details"><Typography component="summary" sx={{ cursor: 'pointer' }}>Recorded configuration and dataset provenance</Typography><Box component="pre" sx={{ maxHeight: 350, overflow: 'auto', fontSize: 12 }}>{JSON.stringify({ parameters: run.parameters, dataset: run.dataset, token_audit: run.token_audit, checkpoints: run.checkpoints }, null, 2)}</Box></Box>
        </>}
      </Stack>
    </Paper>
  </Stack>;
}
