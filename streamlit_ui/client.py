"""Server-side API transport. Never send the backend token to browser widgets."""
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx


class APIError(RuntimeError):
    pass


@dataclass
class API:
    url: str
    token: str = field(default='', repr=False)
    transport: object = field(default=None, repr=False)
    docker_network: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        parsed = urlsplit(self.url)
        local = parsed.hostname in {'localhost', '127.0.0.1', '::1'}
        internal = self.docker_network and self.url.rstrip('/') == 'http://backend:8000'
        if self.docker_network and (not internal or len(self.token) < 32):
            raise ValueError('Docker mode requires the fixed private backend origin and a token of at least 32 characters.')
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Set backend.url to an HTTP(S) API origin without credentials or query parameters.')
        if not local and not internal and parsed.scheme != 'https':
            raise ValueError('A remote backend requires HTTPS.')
        if not local and not self.token:
            raise ValueError('A remote backend requires backend.token in Streamlit secrets.')
        self.url = self.url.rstrip('/')

    def request(self, method, path, body=None, *, binary=False):
        if not path.startswith('/') or path.startswith('//') or '..' in path or '#' in path or '?' in path:
            raise ValueError('API paths must be fixed relative paths.')
        headers = {'Authorization': 'Bearer ' + self.token} if self.token else {}
        try:
            # Mutations are never retried: a timeout can happen after a job was queued.
            with httpx.Client(timeout=httpx.Timeout(150, connect=15), follow_redirects=False,
                              transport=self.transport) as client:
                response = client.request(method, self.url + path, json=body, headers=headers)
        except httpx.RequestError as exc:
            raise APIError('The backend could not be reached or the request timed out. Refresh saved records before resubmitting a job.') from exc
        if response.status_code >= 300:
            if response.status_code in {401, 403}:
                raise APIError('Backend access denied. Check the private API token in deployment secrets.')
            try:
                detail = response.json().get('detail', 'Request failed')
                if isinstance(detail, list):
                    detail = '; '.join(str(x.get('msg', 'Invalid input')) for x in detail)
            except (ValueError, AttributeError):
                detail = 'Backend returned an unexpected error. Check backend logs.'
            detail = str(detail)[:2000]
            if self.token:
                detail = detail.replace(self.token, '[redacted]')
            raise APIError(f'HTTP {response.status_code}: {detail}')
        if binary:
            return response.content
        try:
            return response.json()
        except ValueError as exc:
            raise APIError('Backend returned an invalid JSON response.') from exc

    def get(self, path):
        return self.request('GET', path)

    def post(self, path, body=None):
        return self.request('POST', path, body)
