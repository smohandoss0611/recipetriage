export async function api(path, body, method = 'POST') {
  const response = await fetch(`/api/v1/${path}`, body === undefined ? {} : { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let data;
  try { data = await response.json(); } catch { throw new Error(`HTTP ${response.status}: backend returned no JSON`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
export const percent = value => typeof value === 'number' ? `${(100 * value).toFixed(2)}%` : '—';
export const numeric = (value, digits = 2) => typeof value === 'number' ? value.toLocaleString('en-US', { maximumFractionDigits: digits }) : '—';
