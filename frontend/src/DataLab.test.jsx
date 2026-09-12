import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import seed from '../../ml/data/seed.json';
import DataLab from './DataLab';

const labels = ['weeknight-30min','weekend-project','needs-special-equipment','meal-prep','dessert','have-most-of-this','unclear'];
const metadata = { version:'v1-ui-fixture', status:'teaching-draft', counts:{train:5,validation:1,test:1}, warnings:['Unreviewed teaching draft'], label_distribution:Object.fromEntries(['train','validation','test'].map(name=>[name,Object.fromEntries(labels.map(label=>[label,0]))])) };
const payload = {examples:seed, metadata, files:{'train.jsonl':'{"recipe":{"title":"Overnight Oatmeal"}}\n'}};
afterEach(()=>vi.unstubAllGlobals());
function setup(override) {
  let saved=false;
  vi.stubGlobal('fetch',vi.fn(async(url,options={})=>{
    const custom=override?.(url,options);
    if(custom) return custom;
    let data;
    if(url.endsWith('/seed')) data={examples:seed,labels,policy:Object.fromEntries(labels.map(label=>[label,'Review the recipe evidence.']))};
    else if(url.endsWith('/preview')) data=payload;
    else if(url.endsWith('/versions') && options.method==='POST') {saved=true;data=payload;}
    else if(url.endsWith('/versions')) data=saved?[metadata]:[];
    else if(url.endsWith('/chat-preview')) data={messages:[{role:'system',content:'Instructions'},{role:'user',content:'Recipe'},{role:'assistant',content:'{"labels":["meal-prep"]}'}],rendered_text:null};
    else throw new Error('Unexpected request '+url);
    return {ok:true,json:async()=>data};
  }));
  render(<DataLab />);
}
it('builds and saves a snapshot, then invalidates its export after label edits',async()=>{
  setup(); const user=userEvent.setup();
  await screen.findByText('7 records');
  await user.click(screen.getByRole('button',{name:'Build preview'}));
  await screen.findByText('v1-ui-fixture');
  await user.click(screen.getByRole('button',{name:'Save version'}));
  expect(await screen.findByRole('link',{name:'Download version ZIP'})).toHaveAttribute('href','/api/v1/datasets/versions/v1-ui-fixture/download');
  await user.click(screen.getByRole('tab',{name:'Label Editor'}));
  await user.click(screen.getByRole('checkbox',{name:'meal-prep',exact:true}));
  expect(screen.getByRole('button',{name:'Save version'})).toBeDisabled();
  expect(screen.queryByRole('link',{name:'Download version ZIP'})).not.toBeInTheDocument();
});
it('shows validation failures and preserves the raw input for correction',async()=>{
  setup((url)=>url.endsWith('/validate')?{ok:true,json:async()=>({drafts:[{broken:true}],examples:[],duplicates:[],issues:[{row:1,kind:'invalid',detail:'Missing recipe'}]})}:null);
  const user=userEvent.setup(); await screen.findByText('7 records');
  const raw=screen.getByLabelText('Raw recipes / training examples');
  await user.clear(raw); await user.paste('[{"broken":true}]');
  await user.click(screen.getByRole('button',{name:'Validate & load'}));
  expect(await screen.findByText(/0 valid unique records/)).toBeInTheDocument();
  expect(raw).toHaveValue('[{"broken":true}]');
  expect(screen.getByRole('button',{name:'Save version'})).toBeDisabled();
});
it('opens the chat-message preview and retains the Token Inspector tab',async()=>{
  setup(); const user=userEvent.setup(); await screen.findByText('7 records');
  await user.click(screen.getByRole('tab',{name:'Chat Template Preview'}));
  await user.click(screen.getByRole('button',{name:'Preview messages'}));
  expect(await screen.findByText(/"role": "assistant"/)).toBeInTheDocument();
  await user.click(screen.getByRole('tab',{name:'Token Inspector'}));
  expect(screen.getByRole('button',{name:'Inspect tokens'})).toBeInTheDocument();
});
it('keeps malformed object-valued recipe fields visible instead of crashing',async()=>{
  setup((url)=>url.endsWith('/validate')?{ok:true,json:async()=>({drafts:[{recipe:{id:{bad:true},title:{bad:true},time_minutes:{bad:true}},labels:[]}],examples:[],duplicates:[],issues:[{row:1,kind:'invalid',detail:'Invalid field types'}]})}:null);
  const user=userEvent.setup(); await screen.findByText('7 records');
  await user.click(screen.getByRole('button',{name:'Validate & load'}));
  expect(await screen.findByText(/0 valid unique records/)).toBeInTheDocument();
  expect(screen.getByRole('table',{name:'Dataset recipes'})).toHaveTextContent('"bad": true');
  await user.click(screen.getByRole('tab',{name:'Label Editor'}));
  expect(screen.getByRole('button',{name:'Clear labels'})).toBeInTheDocument();
});
