import { useState } from 'react';
import { Alert, Box, Button, Checkbox, Chip, FormControlLabel, Paper, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from '@mui/material';

export default function TokenInspector() {
  const [text, setText] = useState('Easy ravioli takes 70 minutes.');
  const [chat, setChat] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  function reset() { setResult(null); setError(''); }
  async function inspect(event) {
    event.preventDefault(); reset(); setBusy(true);
    try {
      const response = await fetch('/api/v1/tokens', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, chat_template: chat }) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  return <Stack spacing={3}>
    <Typography component="h2" variant="h5">Token Inspector</Typography>
    <Typography>See how Qwen2.5-0.5B-Instruct represents your text. Tokens are model vocabulary pieces, not necessarily words.</Typography>
    <Paper variant="outlined" sx={{ p: 3 }} component="form" onSubmit={inspect}>
      <TextField label="Text to tokenize" multiline minRows={3} fullWidth value={text} disabled={busy} onChange={e => { setText(e.target.value); reset(); }} slotProps={{ htmlInput: { maxLength: 10000 } }} />
      <FormControlLabel control={<Checkbox checked={chat} disabled={busy} onChange={e => { setChat(e.target.checked); reset(); }} />} label="Include RecipeTriage chat template and assistant generation marker" />
      <Box><Button variant="contained" type="submit" disabled={busy || !text.trim()}>{busy ? 'Loading tokenizer…' : 'Inspect tokens'}</Button></Box>
      <Typography sx={{ mt: 1 }} variant="body2">First use downloads the tokenizer; it does not run the model or contact Fireworks.</Typography>
    </Paper>
    {error && <Alert severity="error">{error}</Alert>}
    {result && <>
      <Stack direction="row" spacing={2} flexWrap="wrap"><Chip color="primary" label={`${result.token_count} tokens`} /><Typography sx={{ overflowWrap: 'anywhere' }}>{result.model}</Typography></Stack>
      <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>Tokenizer revision: {result.revision}</Typography>
      <Paper variant="outlined" sx={{ maxHeight: 440, overflow: 'auto' }}>
        <Table stickyHeader size="small" aria-label="Token vocabulary pieces and IDs"><TableHead><TableRow><TableCell>Position (0-based)</TableCell><TableCell>Token piece</TableCell><TableCell>Token ID</TableCell></TableRow></TableHead>
          <TableBody>{result.tokens.map((token, i) => <TableRow key={i}><TableCell>{i}</TableCell><TableCell sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(token)}</TableCell><TableCell>{result.token_ids[i]}</TableCell></TableRow>)}</TableBody>
        </Table>
      </Paper>
      <Typography variant="body2">Characters such as Ġ and Ċ are tokenizer vocabulary representations of spaces/newlines. Individual pieces may represent bytes; decode the whole sequence to reconstruct the text.</Typography>
      <Typography component="h3" variant="h6">Exact text sent to the tokenizer</Typography>
      <Paper component="pre" variant="outlined" sx={{ p: 2, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: 400, overflow: 'auto' }}>{result.rendered_text}</Paper>
    </>}
  </Stack>;
}
