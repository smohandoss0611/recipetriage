import { render,screen,waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach,expect,test,vi } from 'vitest';
import ReviewQueue from './ReviewQueue';
import AlignmentLab from './AlignmentLab';
import DeploymentStudio from './DeploymentStudio';

afterEach(()=>vi.unstubAllGlobals());
const reply=data=>({ok:true,status:200,json:async()=>data});

test('opening review does not approve, and approval names the displayed revision',async()=>{
  const user=userEvent.setup();let item={candidate_id:'candidate-1',kind:'synthetic',revision:2,status:'pending',draft:{recipe:{title:'Quick Leeks'},labels:['weekend-project'],rationale:'60 minutes'},quality:{passed:true,errors:[]},history:[]};
  const fetch=vi.fn(async(url,options)=>{
    if(url.endsWith('/review')){item={...item,status:'approved'};return reply(item);}
    return reply([item]);
  });vi.stubGlobal('fetch',fetch);
  render(<ReviewQueue/>);await screen.findByDisplayValue(/Quick Leeks/);
  expect(fetch.mock.calls.some(([,o])=>o?.method==='POST')).toBe(false);
  expect(screen.getByRole('button',{name:'Approve displayed revision'})).toBeDisabled();
  await user.type(screen.getByLabelText('Reviewer name'),'Reviewer');
  await user.click(screen.getByRole('button',{name:'Approve displayed revision'}));
  await waitFor(()=>expect(fetch.mock.calls.some(([url,o])=>url.endsWith('/review')&&JSON.parse(o.body).action==='approve')).toBe(true));
  const body=JSON.parse(fetch.mock.calls.find(([u])=>u.endsWith('/review'))[1].body);
  expect(body.revision).toBe(2);expect(body.reviewer).toBe('Reviewer');expect(body.draft).toBeUndefined();
});

test('preference has no default choice; explicit tie is stored and DPO stays disabled',async()=>{
  const user=userEvent.setup();const candidate={raw_output:'{"labels":["unclear"],"explanation":"Unknown timing"}',finish_reason:'stop',quality:{json_valid:true,schema_valid:true}};
  let pair={pair_id:'pair-1',pair_sha256:'hash',recipe:{title:'Millet'},candidates:{a:candidate,b:candidate},status:'pending',decision:null};
  const fetch=vi.fn(async(url,options)=>{
    if(url.endsWith('/choice')){pair={...pair,status:'tie',decision:{choice:'tie',reviewer:'Reviewer'}};return reply(pair);}
    if(url.endsWith('/pairs'))return reply([pair]);
    if(url.endsWith('/options'))return reply({reference_available:true,defaults:{max_steps:2,learning_rate:.000005}});
    return reply([]);
  });vi.stubGlobal('fetch',fetch);render(<AlignmentLab/>);
  await screen.findByRole('heading',{name:'Response A'});
  expect(fetch.mock.calls.some(([,o])=>o?.method==='POST')).toBe(false);
  await user.type(screen.getByLabelText('Reviewer name'),'Reviewer');await user.click(screen.getByRole('button',{name:'Tie',exact:true}));
  await screen.findByText(/Recorded tie by Reviewer/);
  const body=JSON.parse(fetch.mock.calls.find(([u])=>u.endsWith('/choice'))[1].body);expect(body.choice).toBe('tie');
  await user.click(screen.getByRole('tab',{name:'DPO Training'}));expect(screen.getByRole('button',{name:'Run DPO and benchmark'})).toBeDisabled();
});

test('failed registry gate disables promotion and deployment constraints retain their result',async()=>{
  const user=userEvent.setup();const result={candidates:[{name:'small',size_mib:100,peak_ram_mib:900,micro_f1:.2}],constraints:{max_size_mib:200},selection:{recommended:null}};
  const fetch=vi.fn(async(url,options)=>{
    if(url.endsWith('/models'))return reply([{model_id:'m1',name:'Candidate',stage:'candidate',format:'peft-nf4',gate:{passed:false,failures:['Insufficient coverage']}}]);
    if(url.endsWith('/history'))return reply([]);if(url.endsWith('/gate-config'))return reply({min_macro_f1:.45});
    if(url.endsWith('/select'))return reply({recommended:'small',constraints:{max_size_mib:200}});
    return reply(result);
  });vi.stubGlobal('fetch',fetch);render(<DeploymentStudio/>);
  await screen.findByText('Insufficient coverage');await user.type(screen.getByLabelText('Actor / reviewer'),'Reviewer');await user.type(screen.getByLabelText('Promotion or rollback reason'),'Check');
  expect(screen.getByRole('button',{name:'Promote to staging'})).toBeDisabled();
  await user.click(screen.getByRole('tab',{name:'Deployment Comparison'}));await user.click(screen.getByRole('button',{name:'Compare against these constraints'}));
  await screen.findByText(/Recommendation: small/);
  expect(fetch.mock.calls.some(([u,o])=>u.includes('/stage')&&o?.method==='POST')).toBe(false);
});
