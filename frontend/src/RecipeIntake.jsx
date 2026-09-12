import { useEffect, useState } from 'react';
import { Alert, Box, Button, Checkbox, FormControlLabel, MenuItem, Stack, TextField, Typography } from '@mui/material';
import { api } from './experimentApi';

export default function RecipeIntake({ onSaved, onCancel }) {
  const [kind, setKind] = useState('text');
  const [content, setContent] = useState('');
  const [image, setImage] = useState('');
  const [draft, setDraft] = useState(null);
  const [normalized, setNormalized] = useState('');
  const [models, setModels] = useState([]);
  const [model, setModel] = useState('fireworks');
  const [auto, setAuto] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    api('playground/models').then(data => {
      if (!active) return;
      setModels(data.models);
      if (!data.models.some(item => item.key === 'fireworks' && item.available)) setModel(data.models.find(item => item.available)?.key ?? '');
    }).catch(err => { if (active) setError(err.message); });
    return () => { active = false; };
  }, []);
  function changeSource(value) { setContent(value); setDraft(null); setError(''); }
  async function readScreenshot(file) {
    setDraft(null); setError('');
    if (!file || !['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 4_000_000) {
      setContent(''); setImage(''); setError('Choose a PNG, JPEG or WebP screenshot no larger than 4 MB.'); return;
    }
    const reader = new FileReader();
    reader.onload = () => { setImage(reader.result); setContent(String(reader.result).split(',')[1]); setKind('screenshot'); };
    reader.onerror = () => setError('The screenshot could not be read. Try another file.');
    reader.readAsDataURL(file);
  }
  async function act(operation) {
    setBusy(true); setError('');
    try { await operation(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  return <Stack spacing={3} onPaste={event => {
    const file = [...(event.clipboardData?.files ?? [])].find(item => item.type.startsWith('image/'));
    if (file && !busy) { event.preventDefault(); readScreenshot(file); }
  }}>
    <Typography component="h2" variant="h5">Add a recipe</Typography>
    <Typography>Paste recipe text, a message or a public URL, or upload/paste a screenshot. Check the extracted fields before saving.</Typography>
    <Alert severity="info">Text extraction uses Fireworks. Screenshot OCR runs locally, then the extracted text is sent to Fireworks. Complete Recipe JSON and supported structured recipe pages can be read without a model call.</Alert>
    {error && <Alert severity="error">{error}</Alert>}
    <TextField select label="Input type" value={kind} disabled={busy} onChange={event => { setKind(event.target.value); changeSource(''); setImage(''); }}>
      <MenuItem value="text">Recipe text or JSON</MenuItem><MenuItem value="message">Pasted message</MenuItem><MenuItem value="url">Recipe URL</MenuItem><MenuItem value="screenshot">Screenshot</MenuItem>
    </TextField>
    {kind === 'screenshot' ? <>
      <Button component="label" variant="outlined" disabled={busy}>Choose screenshot<input hidden type="file" accept="image/png,image/jpeg,image/webp" aria-label="Screenshot file" onChange={event => readScreenshot(event.target.files?.[0])} /></Button>
      <Typography variant="body2">PNG, JPEG or WebP · up to 4 MB and 12 megapixels · you can also paste an image here.</Typography>
      {image && <Box component="img" src={image} alt="Recipe screenshot to extract" sx={{ maxHeight: 360, maxWidth: '100%', objectFit: 'contain' }} />}
    </> : <TextField label={kind === 'url' ? 'Recipe URL' : 'Recipe text or message'} multiline={kind !== 'url'} minRows={kind === 'url' ? undefined : 8} value={content} onChange={event => changeSource(event.target.value)} disabled={busy} placeholder={kind === 'url' ? 'https://…' : 'Include the recipe title, ingredients, instructions and any timing or equipment information.'} />}
    <Stack direction="row" spacing={2}>
      <Button variant="contained" disabled={busy || !content.trim()} onClick={() => act(async () => {
        const data = await api('recipes/intake', { kind, content }); setDraft(data); setNormalized(JSON.stringify(data.recipe, null, 2));
      })}>{busy && !draft ? 'Extracting recipe…' : 'Normalize recipe'}</Button>
      <Button disabled={busy} onClick={onCancel}>Back to library</Button>
    </Stack>
    {draft && <>
      <Typography component="h3" variant="h6">Check the normalized recipe</Typography>
      <Typography variant="body2">Correct extraction mistakes here. Unknown timing or pantry information should stay null. Saving does not verify predicted labels.</Typography>
      <TextField label="Normalized Recipe JSON" value={normalized} onChange={event => setNormalized(event.target.value)} multiline minRows={12} disabled={busy} />
      <Box component="details"><Typography component="summary">Extracted source text and provenance</Typography><Box component="pre" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: 350, overflow: 'auto' }}>{JSON.stringify({ source_text: draft.source_text, source: draft.source, normalization: draft.normalization }, null, 2)}</Box></Box>
      <FormControlLabel control={<Checkbox checked={auto} onChange={event => setAuto(event.target.checked)} disabled={busy} />} label="Automatically triage after saving" />
      {auto && <TextField select label="Triage model" value={model} onChange={event => setModel(event.target.value)} disabled={busy}>
        {models.map(item => <MenuItem key={item.key} value={item.key} disabled={!item.available}>{item.title}{!item.available ? ' · unavailable' : ''}</MenuItem>)}
      </TextField>}
      <Button variant="contained" disabled={busy || (auto && !model)} onClick={() => act(async () => {
        const result = await api('recipes', { draft_id: draft.draft_id, recipe: JSON.parse(normalized), auto_triage: auto, model_key: model || 'fireworks' });
        onSaved(result);
      })}>{busy ? 'Saving…' : 'Save recipe'}</Button>
    </>}
  </Stack>;
}
