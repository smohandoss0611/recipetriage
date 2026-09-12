import { useEffect,useState } from 'react';
import { Alert,Box,Button,MenuItem,Paper,Stack,Tab,Tabs,Table,TableHead,TableBody,TableRow,TableCell,TextField,Typography } from '@mui/material';
import { api,percent,numeric } from './experimentApi';
const pretty=x=>JSON.stringify(x,null,2);
const Code=({children})=><Paper component="pre" sx={{p:2,maxHeight:350,overflow:'auto',whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{children}</Paper>;

export default function DeploymentStudio(){
  const [tab,setTab]=useState('registry'),[models,setModels]=useState([]),[history,setHistory]=useState([]),[gate,setGate]=useState(null),[id,setId]=useState('');
  const [actor,setActor]=useState(''),[reason,setReason]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[deployment,setDeployment]=useState(null),[constraints,setConstraints]=useState(''),[selection,setSelection]=useState(null);
  async function refresh(){const [m,h,g,d]=await Promise.all([api('models'),api('models/history'),api('models/gate-config'),api('deployment/results')]);setModels(m);setHistory(h);setGate(g);setDeployment(d);setId(old=>old||m[0]?.model_id||'');setConstraints(old=>old||pretty(d.constraints));}
  useEffect(()=>{refresh().catch(e=>setError(e.message));},[]);
  async function act(fn){setBusy(true);setError('');try{await fn();await refresh();}catch(e){setError(e.message);}finally{setBusy(false);}}
  const model=models.find(x=>x.model_id===id);
  const rows=deployment?.candidates??[];
  return <Stack spacing={3}>
    <Tabs value={tab} onChange={(_,v)=>setTab(v)} aria-label="Deployment pages"><Tab label="Model Registry" value="registry"/><Tab label="Deployment Comparison" value="comparison"/></Tabs>
    {error&&<Alert severity="error" sx={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{error}</Alert>}
    <Button disabled={busy} onClick={()=>act(()=>refresh())}>Refresh deployment evidence</Button>
    {tab==='registry'?<>
      <Typography component="h2" variant="h5">Model Registry</Typography>
      <Typography>Candidate → staging → production. Promotion requires complete benchmark evidence, configured quality thresholds, shortcut coverage and per-label nonregression. Replaced production versions are archived for rollback.</Typography>
      <Box component="details"><Typography component="summary">Configured evaluation thresholds</Typography><Code>{pretty(gate)}</Code></Box>
      {!models.length&&<Alert severity="info">No registered candidates yet. Completed experiments can be imported without promoting them.</Alert>}
      {!!models.length&&<><TextField select label="Model version" value={id} onChange={e=>setId(e.target.value)}>{models.map(x=><MenuItem key={x.model_id} value={x.model_id}>{x.name} · {x.stage}</MenuItem>)}</TextField>
      {model&&<><Typography>Stage: {model.stage} · Format: {model.format}</Typography>
        <Alert severity={model.gate.passed?'success':'warning'}>{model.gate.passed?'Evaluation gate passed. Promotion still requires an explicit action.':model.gate.failures.join('\n')}</Alert>
        <Box component="details"><Typography component="summary">Dataset lineage and checkpoint identity</Typography><Code>{pretty({identity:model.identity,dataset_lineage:model.dataset_lineage,evidence_sha256:model.evidence_sha256,training_config:model.training_config})}</Code></Box>
        <TextField label="Actor / reviewer" value={actor} onChange={e=>setActor(e.target.value)}/><TextField label="Promotion or rollback reason" value={reason} onChange={e=>setReason(e.target.value)} multiline/>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {model.stage==='candidate'&&<Button disabled={busy||!actor.trim()||!reason.trim()||!model.gate.passed} onClick={()=>act(()=>api(`models/${id}/stage`,{stage:'staging',actor,reason}))}>Promote to staging</Button>}
          {model.stage==='staging'&&<Button variant="contained" disabled={busy||!actor.trim()||!reason.trim()||!model.gate.passed} onClick={()=>act(()=>api(`models/${id}/stage`,{stage:'production',actor,reason}))}>Promote to production</Button>}
          {['candidate','staging'].includes(model.stage)&&<Button disabled={busy||!actor.trim()||!reason.trim()} onClick={()=>act(()=>api(`models/${id}/stage`,{stage:'archived',actor,reason}))}>Archive candidate</Button>}
          {model.stage==='archived'&&history.some(x=>x.model_id===id&&x.to==='production')&&<Button disabled={busy||!actor.trim()||!reason.trim()||!model.gate.passed} onClick={()=>act(()=>api(`models/${id}/rollback`,{stage:'production',actor,reason}))}>Roll back production to this version</Button>}
        </Stack>
      </>}
      </>}
      <Typography component="h3" variant="h6">Stage history</Typography><Code>{pretty(history)}</Code>
    </>:<>
      <Typography component="h2" variant="h5">Deployment Comparison</Typography>
      <Typography>Supported exports: merged Hugging Face float32 safetensors and bitsandbytes NF4. Measurements are CPU process RAM; VRAM is not applicable. GGUF/AWQ/GPTQ are not claimed or silently substituted.</Typography>
      <Paper sx={{overflowX:'auto'}}><Table size="small" aria-label="Deployment candidates"><TableHead><TableRow>{['Candidate','Size MiB','Peak RAM MiB','VRAM','Mean ms','p95 ms','Micro-F1','Macro-F1','Schema valid'].map(x=><TableCell key={x}>{x}</TableCell>)}</TableRow></TableHead><TableBody>{rows.map(x=><TableRow key={x.name}><TableCell>{x.name}</TableCell><TableCell>{numeric(x.size_mib)}</TableCell><TableCell>{numeric(x.peak_ram_mib)}</TableCell><TableCell>{x.vram_mib==null?'N/A (CPU)':numeric(x.vram_mib)}</TableCell><TableCell>{numeric(x.mean_ms)}</TableCell><TableCell>{numeric(x.p95_ms)}</TableCell><TableCell>{percent(x.micro_f1)}</TableCell><TableCell>{percent(x.macro_f1)}</TableCell><TableCell>{percent(x.schema_validity)}</TableCell></TableRow>)}</TableBody></Table></Paper>
      <TextField label="Operational constraints (JSON)" value={constraints} onChange={e=>setConstraints(e.target.value)} multiline minRows={6}/>
      <Button disabled={busy||!rows.length} onClick={()=>act(async()=>{const d=await api('deployment/select',JSON.parse(constraints));setSelection(d);})}>Compare against these constraints</Button>
      <Alert severity="info">Recommendation: {(selection??deployment?.selection)?.recommended??'No candidate meets every limit'}. This does not promote a model or choose the smallest model automatically.</Alert>
      <Code>{pretty(selection??deployment?.selection)}</Code>
      <Typography>Run the documented merge, quantize and benchmark commands to add measured candidates. Rechecking constraints uses saved evidence and performs no training.</Typography>
    </>}
  </Stack>;
}
