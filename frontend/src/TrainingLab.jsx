import { lazy, Suspense, useState } from 'react';
import { Box, LinearProgress, Tab, Tabs } from '@mui/material';
const ExperimentStudio = lazy(() => import('./ExperimentStudio'));
const LoRATraining = lazy(() => import('./LoRATraining'));
const TrainingMonitor = lazy(() => import('./TrainingMonitor'));

export default function TrainingLab() {
  const [tab, setTab] = useState('experiments');
  return <>
    <Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" aria-label="Training Lab pages" sx={{ mb: 3 }}>
      <Tab label="Experiments" value="experiments" id="experiments-tab" aria-controls="training-panel" />
      <Tab label="QLoRA Training" value="qlora" id="qlora-tab" aria-controls="training-panel" />
      <Tab label="LoRA Training" value="lora" id="lora-tab" aria-controls="training-panel" />
      <Tab label="SFT Monitor" value="sft" id="sft-tab" aria-controls="training-panel" />
    </Tabs>
    <Box role="tabpanel" id="training-panel" aria-labelledby={`${tab}-tab`}>
      <Suspense fallback={<LinearProgress aria-label="Loading training page" />}>
        {tab === 'experiments' ? <ExperimentStudio key="matrix" /> : tab === 'qlora' ? <ExperimentStudio key="qlora" qlora /> : tab === 'lora' ? <LoRATraining /> : <TrainingMonitor />}
      </Suspense>
    </Box>
  </>;
}
