import { useEffect, useState } from 'react';
import { Alert, Box, Button, Checkbox, Chip, FormControlLabel, MenuItem, Paper, Stack, Tab, Tabs, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material';
import TokenInspector from './TokenInspector';
import ReviewQueue from './ReviewQueue';

const pages = ['Dataset Studio', 'Label Editor', 'JSONL Preview', 'Chat Template Preview', 'Token Inspector', 'Synthetic Review'];
const pretty = value => JSON.stringify(value, null, 2);
// Invalid imports remain editable; never render raw objects as React children.
const display = (value, fallback = '') => value == null ? fallback : typeof value === 'object' ? pretty(value) : String(value);
const API = '/api/v1/datasets';
async function request(path, body) {
  const response = await fetch(API + path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : pretty(data.detail ?? data));
  return data;
}
const Code = ({ children }) => <Paper component="pre" variant="outlined" sx={{ p: 2, maxHeight: 480, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: '0.875rem' }}>{children}</Paper>;

export default function DataLab() {
  const [page, setPage] = useState(0);
  const [catalog, setCatalog] = useState(null);
  const [examples, setExamples] = useState([]);
  const [raw, setRaw] = useState('');
  const [format, setFormat] = useState('json');
  const [rawDirty, setRawDirty] = useState(false);
  const [selected, setSelected] = useState(0);
  const [seed, setSeed] = useState('42');
  const [ratios, setRatios] = useState(['0.70','0.15','0.15']);
  const [report, setReport] = useState(null);
  const [result, setResult] = useState(null);
  const [saved, setSaved] = useState('');
  const [history, setHistory] = useState([]);
  const [chat, setChat] = useState(null);
  const [file, setFile] = useState('train.jsonl');
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState('');

  function invalidate() { setResult(null); setSaved(''); setChat(null); setReport(null); setError(''); }
  function adopt(payload) {
    setExamples(payload.examples); setRaw(payload.files?.['raw.json'] ?? pretty(payload.examples));
    setFormat('json'); setRawDirty(false); setSelected(0); setChat(null); setReport(null);
    setSeed(String(payload.metadata?.seed ?? 42));
    setRatios((payload.metadata?.requested_ratios ?? [0.7,0.15,0.15]).map(String));
    setResult(payload.metadata ? payload : null); setSaved(payload.metadata?.version ?? '');
  }
  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const data = await request('/seed');
        if (!active) return;
        if (!Array.isArray(data.examples)) throw new Error('Dataset seed response is invalid');
        setCatalog(data); adopt(data);
        const versions = await request('/versions');
        if (!active) return;
        setHistory(versions);
        if (versions.length) {
          const latest = await request('/versions/' + versions[0].version);
          if (active) adopt(latest);
        }
      } catch (err) { if (active) setError(err.message); }
      finally { if (active) setBusy(false); }
    }
    load(); return () => { active = false; };
  }, []);

  function body() {
    if (seed.trim() === '' || ratios.some(x => x.trim() === '')) throw new Error('Enter a seed and all three split ratios.');
    return { raw, format, seed: Number(seed), ratios: ratios.map(Number) };
  }
  async function act(operation) {
    setBusy(true); setError('');
    try { await operation(); } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  async function validate() {
    invalidate();
    await act(async () => {
      const data = await request('/validate', body());
      setReport(data); setExamples(data.drafts); setSelected(0); setRawDirty(false);
    });
  }
  async function build() {
    invalidate();
    await act(async () => { const data = await request('/preview', body()); setResult(data); setExamples(data.examples); setSelected(0); setRawDirty(false); });
  }
  async function save() {
    await act(async () => {
      const data = await request('/versions', body());
      setResult(data); setSaved(data.metadata.version);
      setHistory(await request('/versions'));
    });
  }
  function edit(fields) {
    invalidate();
    const next = examples.map((row,i) => i === selected ? { ...row, ...fields,
      ...(!('reviewed' in fields) && !('reviewed_by' in fields) ? { reviewed: false, reviewed_by: null } : {}) } : row);
    setExamples(next); setRaw(pretty(next)); setFormat('json'); setRawDirty(false);
  }
  async function importFile(event) {
    const inputFile = event.target.files?.[0];
    event.target.value = '';
    if (!inputFile) return;
    invalidate();
    if (inputFile.size > 2_000_000) { setError('Files must be 2 MB or smaller.'); return; }
    await act(async () => {
      setRaw(await inputFile.text()); setFormat(inputFile.name.toLowerCase().endsWith('.jsonl') ? 'jsonl' : 'json'); setRawDirty(true);
    });
  }
  const validRows = examples.filter(row => row && typeof row === 'object' && row.recipe && typeof row.recipe === 'object');
  const example = examples[selected];
  const labels = Array.isArray(example?.labels) ? example.labels : [];
  const selector = <TextField select label="Recipe to inspect" value={String(selected)} onChange={e => { setSelected(Number(e.target.value)); setChat(null); }} disabled={busy || rawDirty || !validRows.length} fullWidth>
    {examples.map((row,i) => <MenuItem key={i} value={String(i)}>{display(row?.recipe?.id, `Row ${i+1}`)} · {display(row?.recipe?.title, 'Invalid record — edit raw JSON')}</MenuItem>)}
  </TextField>;
  return <Stack spacing={3}>
    <Tabs value={page} onChange={(_,value) => setPage(value)} variant="scrollable" scrollButtons="auto" aria-label="Data Lab pages">{pages.map(label => <Tab key={label} label={label} />)}</Tabs>
    {page === 5 ? <ReviewQueue /> : page === 4 ? <TokenInspector /> : <>
      <Typography component="h2" variant="h5">{pages[page]}</Typography>
      {error && <Alert severity="error" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{error}</Alert>}
      {busy && <Typography role="status">Working…</Typography>}
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        <Chip label={`${examples.length} records`} /><Chip label={`${examples.filter(x => x?.reviewed).length} reviewed`} />
        <Chip label={saved ? 'Saved version' : 'Unsaved draft'} color={saved ? 'success' : 'default'} />
        <Button variant="contained" onClick={build} disabled={busy || !raw.trim()}>Build preview</Button>
        <Button variant="outlined" onClick={save} disabled={busy || !result || rawDirty}>Save version</Button>
        {saved && <Button component="a" href={API+'/versions/'+saved+'/download'}>Download version ZIP</Button>}
      </Stack>
      {!saved && <Typography variant="body2" color="text.secondary">Save a version before reloading or leaving Data Lab. Unsaved edits stay only in this workspace view.</Typography>}
      {rawDirty && <Alert severity="info">Raw input changed. Choose Validate &amp; load in Dataset Studio before editing individual labels.</Alert>}
      {page === 0 && <>
        <Typography>Review seven real recipes, inspect validation results, and create reproducible exports. No generated recipes or training jobs are used. Labels start as proposals that need your review.</Typography>
        <Stack direction={{xs:'column',md:'row'}} spacing={2}>
          <Button onClick={() => { invalidate(); adopt(catalog); }} disabled={busy || !catalog}>Load original seed</Button>
          <Button component="label" disabled={busy}>Import JSON / JSONL<input hidden type="file" accept=".json,.jsonl,application/json" onChange={importFile} /></Button>
          <TextField select label="Saved version" value={saved && history.some(x=>x.version===saved) ? saved : ''} onChange={e => act(async () => adopt(await request('/versions/'+e.target.value)))} disabled={busy || !history.length} sx={{minWidth:240}}>
            <MenuItem value="" disabled>Select a saved snapshot</MenuItem>{history.map(x=><MenuItem key={x.version} value={x.version}>{x.version.slice(0,15)} · {x.status}</MenuItem>)}
          </TextField>
        </Stack>
        <Stack direction={{xs:'column',md:'row'}} spacing={2}>
          <TextField label="Split seed" type="number" value={seed} disabled={busy} onChange={e=>{setSeed(e.target.value);invalidate();}} />
          {['Train ratio','Validation ratio','Test ratio'].map((label,i)=><TextField key={label} label={label} type="number" value={ratios[i]} disabled={busy} onChange={e=>{setRatios(ratios.map((x,j)=>i===j?e.target.value:x));invalidate();}} slotProps={{htmlInput:{min:0.01,max:0.98,step:0.01}}} />)}
        </Stack>
        <TextField select label="Raw format" value={format} disabled={busy} onChange={e=>{setFormat(e.target.value);setRawDirty(true);invalidate();}}><MenuItem value="json">JSON array</MenuItem><MenuItem value="jsonl">JSONL records</MenuItem></TextField>
        <TextField label="Raw recipes / training examples" value={raw} onChange={e=>{setRaw(e.target.value);setRawDirty(true);invalidate();}} disabled={busy} multiline minRows={6} maxRows={14} fullWidth />
        <Button onClick={validate} disabled={busy || !raw.trim()} variant="outlined">Validate &amp; load</Button>
        {report && <><Alert severity={report.issues.length?'warning':'success'}>{report.examples.length} valid unique records · {report.duplicates.length} duplicate copies · {report.issues.length} issues. Invalid records block version creation.</Alert>{report.issues.length>0 && <Code>{pretty(report.issues)}</Code>}{report.duplicates.length>0 && <Code>{pretty(report.duplicates)}</Code>}</>}
        <Paper variant="outlined" sx={{overflowX:'auto'}}><Table size="small" aria-label="Dataset recipes"><TableHead><TableRow><TableCell>Recipe</TableCell><TableCell>Total minutes</TableCell><TableCell>Proposed labels</TableCell></TableRow></TableHead><TableBody>{validRows.map((row,i)=><TableRow key={i}><TableCell>{display(row.recipe.title, 'Missing title')}</TableCell><TableCell>{display(row.recipe.time_minutes, 'Unknown')}</TableCell><TableCell>{Array.isArray(row.labels)?row.labels.map(x=>display(x)).join(', '):'Invalid labels'}</TableCell></TableRow>)}</TableBody></Table></Paper>
        {catalog && <Box component="details"><Typography component="summary">Recipe and TrainingExample schemas</Typography><Code>{pretty({Recipe:catalog.recipe_schema,TrainingExample:catalog.schema})}</Code></Box>}
      </>}
      {page === 1 && <>
        <Typography>Check the source and recipe evidence. Label or rationale edits clear prior review status. Save a version to persist your changes.</Typography>
        {selector}
        {example?.recipe && <>
          <Code>{pretty(example.recipe)}</Code>
          {/^https?:\/\//.test(example.recipe.source_uri ?? '') && <Button component="a" href={example.recipe.source_uri} target="_blank" rel="noreferrer">Open source recipe</Button>}
          <Paper variant="outlined" sx={{p:2}}>{(catalog?.labels ?? []).map(label=><Box key={label}><FormControlLabel control={<Checkbox checked={labels.includes(label)} disabled={busy || rawDirty} onChange={e=>edit({labels:e.target.checked?[...labels,label]:labels.filter(x=>x!==label)})} />} label={label} /><Typography variant="body2" color="text.secondary" sx={{ml:4,mb:1}}>{catalog.policy[label]}</Typography></Box>)}</Paper>
          <Button onClick={()=>edit({labels:[]})} disabled={busy || rawDirty}>Clear labels</Button>
          <TextField label="Recipe-grounded explanation (assistant target)" value={example.rationale ?? ''} disabled={busy || rawDirty} onChange={e=>edit({rationale:e.target.value})} multiline minRows={2} />
          <TextField label="Review notes (excluded from model input and target)" value={example.annotation_notes ?? ''} disabled={busy || rawDirty} onChange={e=>edit({annotation_notes:e.target.value})} multiline />
          <TextField label="Reviewer name" value={example.reviewed_by ?? ''} disabled={busy || rawDirty} onChange={e=>edit({reviewed_by:e.target.value || null})} />
          <FormControlLabel control={<Checkbox checked={example.reviewed === true} disabled={busy || rawDirty} onChange={e=>edit({reviewed:e.target.checked,reviewed_by:e.target.checked?example.reviewed_by:null})} />} label="I have checked this recipe and its labels" />
        </>}
      </>}
      {page === 2 && (result ? <>
        <TextField select label="Export file" value={file} onChange={e=>setFile(e.target.value)}>{Object.keys(result.files).map(name=><MenuItem key={name} value={name}>{name}</MenuItem>)}</TextField>
        <Typography>These are the exact UTF-8 file contents included in the saved version. Each JSONL line is an independent JSON object.</Typography>
        <Code>{result.files[file] ?? ''}</Code>
      </> : <Alert severity="info">Build a preview to inspect train, validation, and test exports.</Alert>)}
      {page === 3 && <>
        <Typography>Inspect the three training messages, then optionally render them using the pinned Qwen tokenizer. Rendering downloads tokenizer files only and never runs a model.</Typography>
        {selector}
        <Stack direction="row" spacing={2}>
          <Button variant="outlined" disabled={busy || rawDirty || !example?.recipe} onClick={()=>act(async()=>{setChat(null);setChat(await request('/chat-preview',{example,render_model:false}));})}>Preview messages</Button>
          <Button variant="outlined" disabled={busy || rawDirty || !example?.recipe} onClick={()=>act(async()=>{setChat(null);setChat(await request('/chat-preview',{example,render_model:true}));})}>Render Qwen chat template</Button>
        </Stack>
        {chat && <><Code>{pretty(chat.messages)}</Code>{chat.rendered_text && <><Typography sx={{overflowWrap:'anywhere'}}>{chat.model} · {chat.revision}</Typography><Code>{chat.rendered_text}</Code></>}</>}
      </>}
      {result && <>
        <Typography component="h3" variant="h6">Version metadata</Typography>
        <Typography sx={{overflowWrap:'anywhere'}}>{result.metadata.version}</Typography>
        <Typography>{Object.entries(result.metadata.counts).map(([name,count])=>`${name}: ${count}`).join(' · ')}</Typography>
        <Paper variant="outlined" sx={{overflowX:'auto'}}><Table size="small" aria-label="Label distribution"><TableHead><TableRow><TableCell>Label</TableCell><TableCell>Train</TableCell><TableCell>Validation</TableCell><TableCell>Test</TableCell></TableRow></TableHead><TableBody>{(catalog?.labels ?? []).map(label=><TableRow key={label}><TableCell>{label}</TableCell>{['train','validation','test'].map(name=><TableCell key={name}>{result.metadata.label_distribution[name][label]}</TableCell>)}</TableRow>)}</TableBody></Table></Paper>
        <Alert severity="warning"><Box component="ul" sx={{m:0,pl:2}}>{result.metadata.warnings.map(message=><li key={message}>{message}</li>)}</Box></Alert>
        <Box component="details"><Typography component="summary">Full metadata, groups and file hashes</Typography><Code>{pretty(result.metadata)}</Code></Box>
      </>}
    </>}
  </Stack>;
}
