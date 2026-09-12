import { useEffect, useState } from 'react';
import { Alert, Box, Button, Chip, CircularProgress, Link, List, ListItem, ListItemButton, ListItemText, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material';
import { api } from './experimentApi';
import RecipeIntake from './RecipeIntake';
import RecipeActions from './RecipeActions';

function sourceLink(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch { return null; }
}

export default function Recipes({ onTryRecipe, onManageDatasets }) {
  const [version, setVersion] = useState('library');
  const [versions, setVersions] = useState([]);
  const [snapshot, setSnapshot] = useState(null);
  const [selectedId, setSelectedId] = useState('');
  const [query, setQuery] = useState('');
  const [label, setLabel] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [adding, setAdding] = useState(false);
  const [message, setMessage] = useState('');
  const [exported, setExported] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError('');
    const path = version === 'library' ? 'recipes' : `datasets/versions/${encodeURIComponent(version)}`;
    Promise.all([api(path), api('datasets/versions')]).then(([data, history]) => {
      if (!active) return;
      if (!Array.isArray(data.examples) || !Array.isArray(history)) throw new Error('Recipe collection response is invalid');
      setSnapshot(data);
      setVersions(history);
      setSelectedId(previous => data.examples.some(item => item.recipe.id === previous) ? previous : data.examples[0]?.recipe.id ?? '');
    }).catch(err => {
      if (active) { setError(err.message); setSnapshot(null); }
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [version, reload]);

  const examples = snapshot?.examples ?? [];
  const pending = version === 'library' && examples.some(item => ['queued', 'running'].includes(item.triage_job?.status));
  useEffect(() => {
    if (!pending) return;
    let active = true;
    const timer = setInterval(() => api('recipes').then(data => { if (active) setSnapshot(data); }).catch(err => { if (active) setError(err.message); }), 3000);
    return () => { active = false; clearInterval(timer); };
  }, [pending]);
  const labels = [...new Set(examples.flatMap(item => item.labels))].sort();
  const search = query.trim().toLocaleLowerCase();
  const visible = examples.filter(item => {
    const text = [item.recipe.title, ...item.recipe.ingredients, ...item.recipe.equipment].join(' ').toLocaleLowerCase();
    return text.includes(search) && (!label || item.labels.includes(label));
  });
  const selected = visible.find(item => item.recipe.id === selectedId) ?? visible[0];
  const recipe = selected?.recipe;
  const source = sourceLink(recipe?.source_uri);

  function openRecipe() {
    // The inference request contains recipe evidence only, never the proposed answer labels.
    const { title, ingredients, instructions, equipment, time_minutes, pantry_items = null } = recipe;
    onTryRecipe({ title, ingredients, instructions, equipment, time_minutes, pantry_items });
  }

  if (adding) return <RecipeIntake onCancel={() => setAdding(false)} onSaved={result => {
    setAdding(false); setVersion('library'); setSelectedId(result.recipe.id); setQuery(''); setLabel(''); setReload(value => value + 1);
    setMessage(result.warning || (result.already_saved ? 'This recipe is already in your library.' : 'Recipe saved. Any automatic triage will appear here when ready.'));
  }} />;

  return <Stack spacing={3}>
    <Typography>Browse your recipe collection, inspect the ingredients and preparation, and try a recipe in Playground.</Typography>
    {message && <Alert severity="info" onClose={() => setMessage('')}>{message}</Alert>}
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
      <Button variant="contained" onClick={() => setAdding(true)}>Add recipe</Button>
      <TextField select label="Recipe collection" value={version} disabled={loading} onChange={event => {
        setVersion(event.target.value); setQuery(''); setLabel(''); setSelectedId('');
      }} sx={{ flex: 1, minWidth: 0 }}>
        <MenuItem value="library">My saved recipes</MenuItem>
        {versions.map(item => <MenuItem key={item.version} value={item.version}>
          {item.version.slice(0, 15)} · {item.status}
        </MenuItem>)}
      </TextField>
      <Button disabled={loading} onClick={() => setReload(value => value + 1)}>Refresh recipes</Button>
      <Button onClick={onManageDatasets}>Manage datasets</Button>
    </Stack>
    {loading ? <Stack direction="row" spacing={2} alignItems="center" role="status"><CircularProgress size={20} /><Typography>Loading recipes…</Typography></Stack> : error ? <Alert severity="error" action={<Button color="inherit" onClick={() => setReload(value => value + 1)}>Retry</Button>}>{error}</Alert> : <>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
        <TextField label="Search recipes" placeholder="Title, ingredient or equipment" value={query} onChange={event => setQuery(event.target.value)} fullWidth />
        <TextField select label="Filter by label" value={label} onChange={event => setLabel(event.target.value)} sx={{ minWidth: { sm: 230 } }}>
          <MenuItem value="">All labels</MenuItem>
          {labels.map(value => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
      </Stack>
      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        <Chip label={`${visible.length} of ${examples.length} recipes`} />
        <Chip label={`${examples.filter(item => item.reviewed).length} reviewed`} variant="outlined" />
      </Stack>
      {examples.some(item => !item.reviewed) && <Alert severity="info">Some labels are proposed annotations awaiting human review. Opening or trying a recipe does not approve it.</Alert>}
      {version === 'library' && <Stack spacing={1}>
        <Button disabled={exporting || examples.filter(item => item.reviewed).length < 3} onClick={async () => {
          setExporting(true); setError('');
          try { setExported(await api('recipes/dataset-version', {})); } catch (err) { setError(err.message); } finally { setExporting(false); }
        }}>Create dataset from verified recipes</Button>
        <Typography variant="body2" color="text.secondary">Requires at least three independent human-reviewed recipes. Unverified predictions stay out of this export.</Typography>
        {exported && <Alert severity="success" sx={{ overflowWrap: 'anywhere' }}>Created {exported.metadata.version}. <Link href={`/api/v1/datasets/versions/${encodeURIComponent(exported.metadata.version)}/download`}>Download dataset JSONL ZIP</Link></Alert>}
      </Stack>}
      {!visible.length ? <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography>{examples.length ? 'No recipes match your search and label filter.' : 'This collection has no recipes.'}</Typography>
        {(query || label) && <Button onClick={() => { setQuery(''); setLabel(''); }}>Clear filters</Button>}
      </Paper> : <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'minmax(220px, 1fr) minmax(0, 2fr)' }, gap: 3, alignItems: 'start' }}>
        <Paper variant="outlined" sx={{ overflow: 'hidden', minWidth: 0 }}>
          <List aria-label="Recipe collection entries" disablePadding>
            {visible.map(item => <ListItem key={item.recipe.id} disablePadding divider>
              <ListItemButton selected={item.recipe.id === recipe.id} aria-label={item.recipe.title} aria-pressed={item.recipe.id === recipe.id} onClick={() => setSelectedId(item.recipe.id)} sx={{ py: 2 }}>
                <ListItemText primary={item.recipe.title} secondary={`${item.recipe.time_minutes == null ? 'Time unknown' : `${item.recipe.time_minutes} min total`} · ${item.reviewed ? 'Reviewed' : 'Awaiting review'}`} slotProps={{ primary: { fontWeight: 600 }, secondary: { sx: { mt: 0.5 } } }} />
              </ListItemButton>
            </ListItem>)}
          </List>
        </Paper>
        <Paper component="article" aria-label="Recipe details" variant="outlined" sx={{ p: { xs: 2, md: 3 }, minWidth: 0, overflowWrap: 'anywhere' }}>
          <Stack spacing={2}>
            <Typography component="h2" variant="h5">{recipe.title}</Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Chip label={recipe.time_minutes == null ? 'Total time unknown' : `${recipe.time_minutes} min total`} />
              <Chip variant="outlined" label={selected.reviewed ? 'Reviewed' : 'Awaiting review'} />
              {recipe.source_type === 'synthetic' && <Chip variant="outlined" label="Synthetic source" color="warning" />}
            </Stack>
            <Box>
              <Typography component="h3" variant="subtitle1" fontWeight={600}>{selected.reviewed ? 'Reviewed labels' : 'Proposed labels'}</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>{selected.labels.map(value => <Chip size="small" key={value} label={value} color="primary" variant="outlined" />)}</Stack>
              <Typography sx={{ mt: 1 }} color="text.secondary">{selected.rationale}</Typography>
              {selected.reviewed && <Typography variant="body2">Reviewed by {selected.reviewed_by}</Typography>}
            </Box>
            <Button variant="contained" onClick={openRecipe}>Open in Playground</Button>
            {selected.library_id && <RecipeActions item={selected} labels={snapshot.labels} policy={snapshot.policy} onChanged={() => setReload(value => value + 1)} />}
            <Box>
              <Typography component="h3" variant="h6">Ingredients</Typography>
              <Box component="ul" sx={{ pl: 3, my: 1 }}>{recipe.ingredients.map((item, index) => <li key={index}>{item}</li>)}</Box>
            </Box>
            <Box>
              <Typography component="h3" variant="h6">Instructions</Typography>
              <Box component="ol" sx={{ pl: 3, my: 1 }}>{recipe.instructions.map((item, index) => <Box component="li" key={index} sx={{ mb: 1 }}>{item}</Box>)}</Box>
            </Box>
            <Box>
              <Typography component="h3" variant="h6">Equipment</Typography>
              <Typography>{recipe.equipment.length ? recipe.equipment.join(', ') : 'None listed'}</Typography>
            </Box>
            <Box>
              <Typography component="h3" variant="h6">Pantry information</Typography>
              <Typography>{recipe.pantry_items == null ? 'Not provided' : recipe.pantry_items.length ? recipe.pantry_items.join(', ') : 'No pantry items listed'}</Typography>
            </Box>
            <Box>
              <Typography component="h3" variant="h6">Source</Typography>
              {source ? <Link href={source} target="_blank" rel="noopener noreferrer">View original recipe</Link> : <Typography variant="body2">{recipe.source_uri}</Typography>}
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{recipe.source_notes}</Typography>
            </Box>
          </Stack>
        </Paper>
      </Box>}
    </>}
  </Stack>;
}
