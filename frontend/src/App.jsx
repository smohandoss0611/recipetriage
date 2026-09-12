import { useState } from 'react';
import DataLab from './DataLab';
import EvaluationLab from './EvaluationLab';
import TrainingLab from './TrainingLab';
import { lazy, Suspense } from 'react';
const AlignmentLab=lazy(()=>import('./AlignmentLab'));
const DeploymentStudio=lazy(()=>import('./DeploymentStudio'));
const Recipes=lazy(()=>import('./Recipes'));
const Playground=lazy(()=>import('./Playground'));
import { Box, Chip, CssBaseline, Drawer, List, ListItemButton, ListItemText, MenuItem, TextField, ThemeProvider, Typography, createTheme } from '@mui/material';

export const sections = ['Recipes', 'Data Lab', 'Training Lab', 'Alignment Lab', 'Evaluation Lab', 'Playground', 'Deployment'];
const theme = createTheme({
  palette: { primary: { main: '#1559bc' }, background: { default: '#f3f6fb' } },
  typography: { fontFamily: 'Inter, system-ui, sans-serif' },
  shape: { borderRadius: 12 },
});

export default function App() {
  const [active, setActive] = useState('Recipes');
  const [playgroundRecipe, setPlaygroundRecipe] = useState(null);
  function tryRecipe(recipe) {
    setPlaygroundRecipe(recipe);
    setActive('Playground');
  }
  return <ThemeProvider theme={theme}>
    <CssBaseline />
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Drawer variant="permanent" sx={{ display: { xs: 'none', sm: 'block' }, width: { xs: 0, sm: 248 }, flexShrink: 0, '& .MuiDrawer-paper': { width: 248, boxSizing: 'border-box', bgcolor: '#112441', color: 'white' } }}>
        <Typography sx={{ px: 2, py: 3, fontWeight: 750, fontSize: '1.15rem' }}>RecipeTriage AI</Typography>
        <Box component="nav" aria-label="Main navigation">
          <List sx={{ px: 1 }}>{sections.map(section => <ListItemButton key={section} selected={active === section} aria-current={active === section ? 'page' : undefined} onClick={() => setActive(section)} sx={{ borderRadius: 1, mb: 0.5, '&.Mui-selected': { bgcolor: '#244b7f' }, '&.Mui-selected:hover': { bgcolor: '#2b588f' } }}>
            <ListItemText primary={section} primaryTypographyProps={{ fontSize: '0.95rem' }} />
          </ListItemButton>)}</List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, minWidth: 0, p: { xs: 2, md: 5 } }}>
        <TextField select label="Workspace" value={active} onChange={event => setActive(event.target.value)} fullWidth sx={{ display: { xs: 'flex', sm: 'none' }, mb: 3 }}>
          {sections.map(section => <MenuItem key={section} value={section}>{section}</MenuItem>)}
        </TextField>
        <Chip label="Reviewed learning & model lifecycle" size="small" color="primary" variant="outlined" />
        <Typography component="h1" variant="h4" sx={{ mt: 2, mb: 3, fontWeight: 700, overflowWrap: 'anywhere' }}>{active}</Typography>
        {active === 'Alignment Lab' ? <Suspense fallback={<Typography>Loading alignment…</Typography>}><AlignmentLab/></Suspense> : active === 'Deployment' ? <Suspense fallback={<Typography>Loading deployment…</Typography>}><DeploymentStudio/></Suspense> : active === 'Data Lab' ? <DataLab /> : active === 'Training Lab' ? <TrainingLab /> : active === 'Evaluation Lab' ? <EvaluationLab /> : active === 'Playground' ? <Suspense fallback={<Typography>Loading playground…</Typography>}><Playground initialRecipe={playgroundRecipe} /></Suspense> : <Suspense fallback={<Typography>Loading recipes…</Typography>}><Recipes onTryRecipe={tryRecipe} onManageDatasets={() => setActive('Data Lab')} /></Suspense>}
      </Box>
    </Box>
  </ThemeProvider>;
}
