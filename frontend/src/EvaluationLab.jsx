import { lazy, Suspense, useState } from 'react';
import { Box, LinearProgress, Tab, Tabs } from '@mui/material';
const Baseline = lazy(() => import('./Baseline'));
const FailureAnalysis = lazy(() => import('./FailureAnalysis'));
const ShortcutTests = lazy(() => import('./ShortcutTests'));
export default function EvaluationLab() {
  const [tab, setTab] = useState('failures');
  return <><Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" aria-label="Evaluation Lab pages" sx={{ mb: 3 }}>
    <Tab label="Failure Analysis" value="failures" /><Tab label="Shortcut Tests" value="shortcuts" /><Tab label="Baseline" value="baseline" />
  </Tabs><Box><Suspense fallback={<LinearProgress />}>{tab === 'failures' ? <FailureAnalysis /> : tab === 'shortcuts' ? <ShortcutTests /> : <Baseline />}</Suspense></Box></>;
}
