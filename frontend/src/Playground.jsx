import { useState } from 'react';
import { Stack, Tab, Tabs } from '@mui/material';
import InferencePlayground from './InferencePlayground';
import ModelComparison from './ModelComparison';

export default function Playground({ initialRecipe }) {
  const [tab, setTab] = useState('compare');
  return <Stack spacing={3}>
    <Tabs value={tab} onChange={(_, value) => setTab(value)} aria-label="Playground pages"><Tab label="Compare models" value="compare" /><Tab label="Single model" value="single" /></Tabs>
    {tab === 'compare' ? <ModelComparison initialRecipe={initialRecipe} /> : <InferencePlayground initialRecipe={initialRecipe} />}
  </Stack>;
}
