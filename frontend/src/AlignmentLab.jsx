import { useEffect,useState } from 'react';
import { Alert,Box,Button,Chip,MenuItem,Paper,Stack,Tab,Tabs,TextField,Typography,Table,TableHead,TableBody,TableRow,TableCell } from '@mui/material';
import { api,percent,numeric } from './experimentApi';
const pretty=x=>JSON.stringify(x,null,2);
const Code=({children})=><Paper component="pre" sx={{p:2,maxHeight:380,overflow:'auto',whiteSpace:'pre-wrap',overflowWrap:'anywhere',fontSize:13}}>{children}</Paper>;

export default function AlignmentLab(){
  const [tab,setTab]=useState('pairs'),[pairs,setPairs]=useState([]),[id,setId]=useState(''),[options,setOptions]=useState(null),[jobs,setJobs]=useState([]);
  const [reviewer,setReviewer]=useState(''),[notes,setNotes]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const [versions,setVersions]=useState([]),[version,setVersion]=useState(''),[lr,setLr]=useState('0.000005'),[steps,setSteps]=useState('2');
  async function refresh(){const [p,o,j,v]=await Promise.all([api('alignment/pairs'),api('alignment/options'),api('alignment/jobs'),api('datasets/versions')]);setPairs(p);setOptions(o);setJobs(j);setVersions(v.filter(x=>x.holdouts_frozen));setId(old=>old||p[0]?.pair_id||'');}
  useEffect(()=>{refresh().catch(e=>setError(e.message));const timer=setInterval(()=>refresh().catch(e=>setError(e.message)),5000);return()=>clearInterval(timer);},[]);
  async function act(fn){setBusy(true);setError('');try{await fn();await refresh();}catch(e){setError(e.message);}finally{setBusy(false);}}
  const selected=pairs.find(x=>x.pair_id===id),chosen=pairs.filter(x=>x.status==='chosen').length;
  const active=jobs.some(x=>['queued','running'].includes(x.status));
  function train(method){return act(()=>api('alignment/train/'+method,{...options.defaults,learning_rate:Number(lr),max_steps:Number(steps)}));}
  return <Stack spacing={3}>
    <Tabs value={tab} onChange={(_,v)=>setTab(v)} variant="scrollable" aria-label="Alignment pages">
      <Tab label="Preference Pairs" value="pairs"/><Tab label="DPO Training" value="dpo"/><Tab label="GRPO Experiment" value="grpo"/><Tab label="Reviewed QLoRA" value="reviewed"/>
    </Tabs>
    {error&&<Alert severity="error" sx={{whiteSpace:'pre-wrap',overflowWrap:'anywhere'}}>{error}</Alert>}
    {tab==='pairs'?<>
      <Typography component="h2" variant="h5">Preference Pairs</Typography>
      <Alert severity="info">Choose based on the recipe and your preferences. There is no selected or recommended answer. Only an explicit choice is stored; ties and “neither” are excluded from DPO.</Alert>
      <Button disabled={busy||active||!options?.reference_available} onClick={()=>act(()=>api('alignment/pairs/generate',{}))}>Generate 5 candidate pairs</Button>
      <Chip label={`${chosen} explicit a/b choices · ${pairs.filter(x=>x.status==='pending').length} pending`}/>
      <TextField select label="Preference recipe" value={id} onChange={e=>{setId(e.target.value);setNotes('');}}>{pairs.map(x=><MenuItem key={x.pair_id} value={x.pair_id}>{x.recipe.title} · {x.status}</MenuItem>)}</TextField>
      {selected&&<><Code>{pretty(selected.recipe)}</Code>
        <Stack direction={{xs:'column',lg:'row'}} spacing={2}>{['a','b'].map(key=><Paper variant="outlined" key={key} sx={{p:2,flex:1,minWidth:0}}>
          <Typography component="h3" variant="h6">Response {key.toUpperCase()}</Typography><Code>{selected.candidates[key].raw_output}</Code>
          <Typography variant="body2">JSON: {selected.candidates[key].quality.json_valid?'valid':'invalid'} · Schema: {selected.candidates[key].quality.schema_valid?'valid':'invalid'} · Finish: {selected.candidates[key].finish_reason}</Typography>
          <Box component="details"><Typography component="summary">Candidate provenance</Typography><Code>{pretty(selected.candidates[key])}</Code></Box>
        </Paper>)}</Stack>
        <TextField label="Reviewer name" value={reviewer} onChange={e=>setReviewer(e.target.value)}/>
        <TextField label="Why do you prefer this response?" value={notes} onChange={e=>setNotes(e.target.value)} multiline/>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>{[['a','Prefer A'],['b','Prefer B'],['tie','Tie'],['neither','Neither is acceptable']].map(([choice,label])=><Button key={choice} variant="outlined" disabled={busy||!reviewer.trim()||!!selected.decision} onClick={()=>act(()=>api(`alignment/pairs/${id}/choice`,{choice,reviewer,notes,pair_sha256:selected.pair_sha256}))}>{label}</Button>)}</Stack>
        {selected.decision&&<Alert severity="success">Recorded {selected.decision.choice} by {selected.decision.reviewer}. The exact response pair and decision are immutable.</Alert>}
      </>}
    </>:tab==='reviewed'?<>
      <Typography component="h2" variant="h5">Retrain QLoRA on approved data</Typography>
      <Typography>Uses the previously measured QLoRA configuration: rank 4, learning rate 0.0002, three epochs, batch 1, accumulation 2, NF4, and gradient checkpointing. Only the reviewed dataset changes; original validation/test membership stays fixed.</Typography>
      <TextField select label="Approved augmented dataset" value={version} onChange={e=>setVersion(e.target.value)}>{versions.map(x=><MenuItem key={x.version} value={x.version}>{x.version.slice(0,20)} · {x.counts.train} train</MenuItem>)}</TextField>
      {!versions.length&&<Alert severity="info">Complete the Synthetic Review queue and build the next version first.</Alert>}
      <Button variant="contained" disabled={busy||active||!version} onClick={()=>act(()=>api('alignment/retrain-approved',{dataset_version:version}))}>Retrain and measure shortcut accuracy</Button>
    </>:<>
      <Typography component="h2" variant="h5">{tab==='dpo'?'DPO Training':'GRPO Experiment'}</Typography>
      {tab==='dpo'?<><Typography>DPO learns from your chosen/rejected answers. It starts from the current QLoRA adapter and retains a separate frozen copy as reference. No model judge creates preference labels.</Typography><Alert severity="info">At least three independent, complete a/b choices are required; one recipe group is held out for validation. Currently {chosen} a/b choices.</Alert></>:<>
        <Typography>This is a small isolated reward experiment on the existing provisional training split. It does not replace production and is not automatically preferable to SFT or DPO.</Typography>
        <Code>{'R = 0.55 × label F1 + 0.10 × valid JSON + 0.10 × valid completed schema\n  + 0.15 × appropriate unclear + 0.10 × correct title-pair consistency'}</Code>
        <Typography>Consistency requires both title variants to match the label set; stable wrong answers earn no bonus. Every title intervention holds the recipe body constant. Two sampled responses per prompt, two steps by default, no KL term in this bounded CPU probe.</Typography>
      </>}
      <Stack direction={{xs:'column',sm:'row'}} spacing={2}><TextField label="Learning rate" value={lr} onChange={e=>setLr(e.target.value)} type="number"/><TextField label="Optimizer steps (1–10)" value={steps} onChange={e=>setSteps(e.target.value)} type="number"/></Stack>
      <Button variant="contained" disabled={busy||active||!options?.reference_available||(tab==='dpo'&&chosen<3)} onClick={()=>train(tab)}>Run {tab.toUpperCase()} and benchmark</Button>
      <Alert severity="warning">The benchmark and title suite have provisional labels and have been inspected repeatedly. Compare actual labels, complete output, unclear behavior and shortcut coverage; loss/reward alone does not establish improvement.</Alert>
    </>}
    {options&&!options.reference_available&&<Alert severity="warning">The reference adapter is not installed in this runtime. The README provides the copy command; no substitute model is used.</Alert>}
    <Typography component="h2" variant="h6">Saved learning jobs</Typography>
    {jobs.map(job=><Paper key={job.run_id} variant="outlined" sx={{p:2}}><Stack spacing={1}>
      <Typography>{job.kind} · {job.status} · {job.run_id.slice(0,8)}</Typography>
      {job.error&&<Alert severity="error">{job.error}</Alert>}
      {(job.result?.comparison?.after||job.result?.behavior_comparison?.after)&&(()=>{const c=job.result.behavior_comparison??job.result.comparison;return <Paper sx={{overflowX:'auto'}}><Table size="small" aria-label="Alignment behavior comparison"><TableHead><TableRow><TableCell>Behavior</TableCell><TableCell>Before</TableCell><TableCell>After</TableCell></TableRow></TableHead><TableBody>{['micro_f1','macro_f1','json_validity','schema_validity'].map(k=><TableRow key={k}><TableCell>{k}</TableCell><TableCell>{percent(c.before[k])}</TableCell><TableCell>{percent(c.after[k])}</TableCell></TableRow>)}<TableRow><TableCell>Misleading-title accuracy</TableCell><TableCell>{percent(c.before.shortcut?.misleading_title_accuracy)}</TableCell><TableCell>{percent(c.after.shortcut?.misleading_title_accuracy)}</TableCell></TableRow><TableRow><TableCell>Unclear F1</TableCell><TableCell>{percent(c.before.unclear?.f1)}</TableCell><TableCell>{percent(c.after.unclear?.f1)}</TableCell></TableRow></TableBody></Table></Paper>;})()}
      {job.result?.reward_components&&<Box component="details"><Typography component="summary">Reward components for every sampled response</Typography><Code>{pretty(job.result.reward_components)}</Code></Box>}
      <Box component="details"><Typography component="summary">Full parameters, lineage and results</Typography><Code>{pretty(job)}</Code></Box>
    </Stack></Paper>)}
  </Stack>;
}
