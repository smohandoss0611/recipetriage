import { useEffect, useState } from 'react';
import { Alert, Box, Button, Chip, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material';
import { api } from './experimentApi';
const pretty=x=>JSON.stringify(x,null,2);

export default function ReviewQueue() {
  const [items,setItems]=useState([]), [id,setId]=useState(''), [reviewer,setReviewer]=useState(''), [notes,setNotes]=useState('');
  const [draft,setDraft]=useState(''), [error,setError]=useState(''), [busy,setBusy]=useState(false), [result,setResult]=useState(null);
  async function refresh(){const rows=await api('curation/candidates');setItems(rows);setId(old=>old||rows[0]?.candidate_id||'');}
  useEffect(()=>{refresh().catch(e=>setError(e.message));const timer=setInterval(()=>refresh().catch(e=>setError(e.message)),5000);return()=>clearInterval(timer);},[]);
  const selected=items.find(x=>x.candidate_id===id);
  useEffect(()=>{setDraft(pretty(selected?.draft??{}));setNotes('');},[selected?.candidate_id,selected?.revision]);
  async function act(fn){setBusy(true);setError('');try{await fn();await refresh();}catch(e){setError(e.message);}finally{setBusy(false);}}
  async function decide(action){await act(async()=>{await api(`curation/candidates/${id}/review`,{action,revision:selected.revision,reviewer,notes,...(action==='edit'?{draft:JSON.parse(draft)}:{})});});}
  const edited=draft!==pretty(selected?.draft??{});
  return <Stack spacing={2}>
    <Typography variant="h5" component="h2">Synthetic Review Queue</Typography>
    <Alert severity="info">Generation proposes data; only your explicit approval can authorize a training export. Edits return to pending review. The original seed also needs review before creating the improved version.</Alert>
    {error&&<Alert severity="error">{error}</Alert>}
    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
      <Button disabled={busy} onClick={()=>act(async()=>{await api('curation/generate',{count:6});})}>Generate 6 Fireworks proposals</Button>
      <Button disabled={busy} onClick={()=>act(()=>api('curation/seed-review',{}))}>Queue original seed for review</Button>
      <Button disabled={busy} onClick={()=>act(()=>refresh())}>Refresh queue</Button>
      <Chip label={`${items.filter(x=>x.status==='approved').length} approved · ${items.filter(x=>x.status==='pending').length} pending`} />
    </Stack>
    {!items.length&&<Typography>No proposals yet. Generation never starts merely by opening this page.</Typography>}
    {!!items.length&&<>
      <TextField select label="Candidate to review" value={id} onChange={e=>setId(e.target.value)}>
        {items.map(x=><MenuItem key={x.candidate_id} value={x.candidate_id}>{x.draft?.recipe?.title??'Failed generation'} · {x.kind} · {x.status}</MenuItem>)}
      </TextField>
      <Typography>{selected?.category??'Original seed'} · revision {selected?.revision} · {selected?.model??'Published source'}</Typography>
      <Alert severity={selected?.quality?.passed?'info':'warning'}>{selected?.quality?.passed?'Automated checks passed; this is not a human approval.':(selected?.quality?.errors??[]).join('\n')}</Alert>
      <TextField label="Recipe, proposed labels and rationale" value={draft} onChange={e=>setDraft(e.target.value)} multiline minRows={12} maxRows={25} disabled={busy||selected?.status==='approved'} />
      <TextField label="Reviewer name" value={reviewer} onChange={e=>setReviewer(e.target.value)} />
      <TextField label="Review notes" value={notes} onChange={e=>setNotes(e.target.value)} multiline />
      <Stack direction="row" spacing={1}>
        <Button variant="contained" disabled={busy||!reviewer.trim()||edited||!selected?.quality?.passed||selected?.status==='approved'} onClick={()=>decide('approve')}>Approve displayed revision</Button>
        <Button disabled={busy||!reviewer.trim()||selected?.status==='approved'} onClick={()=>decide('reject')}>Reject</Button>
        <Button disabled={busy||!reviewer.trim()||!edited||selected?.status==='approved'} onClick={()=>decide('edit')}>Save edits for review</Button>
      </Stack>
      {edited&&<Alert severity="info">Save your edits first; approval is a separate explicit action on the saved revision.</Alert>}
      <Box component="details"><Typography component="summary">Original output, provenance and review history</Typography><Paper component="pre" sx={{p:2,overflow:'auto',maxHeight:400,whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{pretty(selected)}</Paper></Box>
    </>}
    <Button variant="outlined" disabled={busy} onClick={()=>act(async()=>setResult(await api('curation/build-version',{})))}>Build next version from approved examples</Button>
    <Typography variant="body2">All seven original records and at least one synthetic proposal must be approved. Original validation/test recipes and labels stay fixed. Rejected and pending items cannot enter this version.</Typography>
    {result&&<Alert severity="success" sx={{overflowWrap:'anywhere'}}>Saved {result.metadata.version} · {pretty(result.metadata.counts)}. QLoRA retraining is available in Alignment Lab.</Alert>}
  </Stack>;
}
