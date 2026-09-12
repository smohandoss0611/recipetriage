import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import benchmark from '../../ml/evaluation/bench_v1.json';
import Baseline from './Baseline';

const labels = ['weeknight-30min','weekend-project','needs-special-equipment','meal-prep','dessert','have-most-of-this','unclear'];
const score = { micro:{f1:0.5}, macro:{f1:0.4}, exact_match:0.2,
  per_label:Object.fromEntries(labels.map(label=>[label,{support:1,tp:1,fp:1,fn:0,precision:0.5,recall:1,f1:2/3}])) };
const run = {run_id:'11111111-1111-4111-8111-111111111111',status:'completed',config:{provider:'hf-base'},
  model_config:{model:'Qwen/Qwen2.5-0.5B',training_stage:'pretrained-base'},protocol_sha256:'a'.repeat(64),setup_latency_ms:20,
  benchmark, summary:{scorable:true,expected_cases:10,usable_responses:4,labels:score,json_validity:0.5,schema_validity:0.4,
    latency_all_attempts:{mean_ms:1500,p95_ms:2000},categories:{normal:{...score,samples:2}}},
  rows:[{case_id:'bench-001',category:'normal',expected_labels:['weeknight-30min'],prediction:null,json_valid:false,schema_valid:false,usable:false,
    raw_output:'Unformatted response retained',finish_reason:'length',latency_ms:1200,error:'Incomplete response',messages:[{role:'user',content:'Recipe input'}]}]};
afterEach(()=>vi.unstubAllGlobals());
function setup(history=[run], custom) {
  const fetch = vi.fn(async(url, options={})=>{
    const replacement=custom?.(url,options);
    if(replacement) return replacement;
    let data;
    if(url.endsWith('/manifest')) data={...benchmark,benchmark_sha256:'b'.repeat(64),contamination_audit:{dataset_v1_source_or_body_overlap:[],pretraining_overlap:'unknown'}};
    else if(url.endsWith('/providers')) data=[{id:'hf-base',model:'Qwen/Qwen2.5-0.5B',configured:true},{id:'fireworks',model:'Hosted model',configured:true}];
    else if(url.endsWith('/runs')) data=history;
    else data=history.find(item=>url.endsWith(item.run_id));
    return {ok:true,json:async()=>data};
  });
  vi.stubGlobal('fetch',fetch); render(<Baseline />); return fetch;
}
it('shows comparison scores and preserves raw failures in run details',async()=>{
  setup(); const user=userEvent.setup();
  expect(await screen.findByRole('table',{name:'Baseline comparison'})).toHaveTextContent('50.0%');
  await user.click(screen.getByRole('button',{name:'View 11111111'}));
  expect(await screen.findByRole('table',{name:'Per-label metrics'})).toHaveTextContent('needs-special-equipment');
  expect(screen.getByText('Unformatted response retained')).toBeInTheDocument();
  expect(screen.getByRole('link',{name:'Download run JSON'})).toHaveAttribute('href','/api/v1/benchmarks/runs/'+run.run_id+'/download');
});
it('does not present blocked runs as zero model scores',async()=>{
  const blocked={...run,status:'blocked',error:'API key missing',rows:[],summary:{...run.summary,scorable:false,labels:null,json_validity:null,schema_validity:null,usable_responses:0,categories:{}}};
  setup([blocked]); const user=userEvent.setup();
  const table=await screen.findByRole('table',{name:'Baseline comparison'});
  expect(table).toHaveTextContent('blocked'); expect(table).not.toHaveTextContent('0.0%');
  await user.click(screen.getByRole('button',{name:'View 11111111'}));
  expect(await screen.findByText(/Missing scores are not zero scores/)).toBeInTheDocument();
});
it('warns on mixed protocols and exposes frozen benchmark cases',async()=>{
  setup([run,{...run,run_id:'22222222-2222-4222-8222-222222222222',protocol_sha256:'c'.repeat(64)}]);
  const user=userEvent.setup(); expect(await screen.findByText(/different evaluation settings/)).toBeInTheDocument();
  await user.click(screen.getByRole('tab',{name:'Benchmark cases'}));
  expect(screen.getByText(/bench-003 · Quick Brown Bread/)).toBeInTheDocument();
});
it('submits bounded run settings without the benchmark answer key and shows API errors',async()=>{
  const fetch=setup([], (url,options)=>options.method==='POST'?{ok:false,json:async()=>({detail:'Another benchmark is running'})}:null);
  const user=userEvent.setup(); await screen.findByText('No benchmark runs saved yet.');
  await user.click(screen.getByRole('button',{name:'Run benchmark'}));
  expect(await screen.findByText('Another benchmark is running')).toBeInTheDocument();
  const call=fetch.mock.calls.find(([,options])=>options?.method==='POST');
  expect(JSON.parse(call[1].body)).toEqual({provider:'hf-base',temperature:0,max_new_tokens:128,reasoning:'disabled'});
});
