"""Bounded public URL retrieval and local screenshot OCR for recipe intake."""
import base64
import http.client
import io
import ipaddress
import json
import math
import re
import shutil
import socket
import ssl
import subprocess
import tempfile
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urljoin
from recipetriage_ml.inference.contracts import Recipe
from recipetriage_ml.inference.normalization import normalize

MAX_PAGE_BYTES = 2_000_000
MAX_IMAGE_BYTES = 4_000_000


def public_target(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use a public HTTP(S) recipe URL without credentials')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    if port not in {80, 443}:
        raise ValueError('Recipe URLs must use port 80 or 443')
    host = parsed.hostname.encode('idna').decode('ascii')
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError('Recipe URL hostname could not be resolved') from exc
    ips = sorted({row[4][0] for row in addresses}, key=lambda value: ':' in value)
    if not ips or any(not ipaddress.ip_address(value).is_global for value in ips):
        raise ValueError('Recipe URLs cannot access local, private or reserved network addresses')
    return parsed, host, port, ips[0]


def fetch_public_page(url):
    """Pin the validated public IP; redirects are resolved and revalidated independently."""
    for _ in range(4):
        parsed, host, port, ip = public_target(url)
        connection = http.client.HTTPConnection(host, port, timeout=12)
        try:
            connection.sock = socket.create_connection((ip, port), timeout=12)
            if parsed.scheme == 'https':
                connection.sock = ssl.create_default_context().wrap_socket(connection.sock, server_hostname=host)
            path = parsed.path or '/'
            if parsed.query: path += '?' + parsed.query
            connection.request('GET', path, headers={'User-Agent': 'RecipeTriage/1.0 recipe-import', 'Accept': 'text/html,application/ld+json', 'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                target = response.getheader('Location')
                if not target: raise ValueError('Recipe URL returned a redirect without a destination')
                url = urljoin(url, target)
                continue
            if response.status != 200:
                raise ValueError(f'Recipe site returned HTTP {response.status}; paste its recipe text instead')
            if response.getheader('Content-Encoding', 'identity').lower() not in {'identity', ''}:
                raise ValueError('Recipe site returned unsupported compressed content; paste its text instead')
            kind = response.getheader('Content-Type', '').lower()
            if not any(value in kind for value in ['text/html', 'application/ld+json', 'application/json', 'text/plain']):
                raise ValueError('Recipe URL must return text or HTML')
            raw = response.read(MAX_PAGE_BYTES + 1)
            if len(raw) > MAX_PAGE_BYTES: raise ValueError('Recipe page exceeds the 2 MB intake limit')
            return raw.decode('utf-8', errors='replace'), url
        except (OSError, http.client.HTTPException) as exc:
            raise ValueError('Recipe page could not be downloaded; paste its text instead') from exc
        finally:
            connection.close()
    raise ValueError('Recipe URL has too many redirects')


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(); self.text = []; self.jsonld = []; self.hidden = 0; self.script = None
    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.script = [] if dict(attrs).get('type', '').split(';')[0] == 'application/ld+json' else None
        if tag in {'script', 'style', 'noscript'}: self.hidden += 1
    def handle_endtag(self, tag):
        if tag == 'script' and self.script is not None:
            try: self.jsonld.append(json.loads(''.join(self.script)))
            except ValueError: pass
            self.script = None
        if tag in {'script', 'style', 'noscript'}: self.hidden = max(0, self.hidden - 1)
    def handle_data(self, data):
        if self.script is not None: self.script.append(data)
        elif not self.hidden and data.strip(): self.text.append(data.strip())


def text_only(value):
    parser = PageText(); parser.feed(str(value)); return unescape(' '.join(parser.text)).strip()


def recipe_nodes(value):
    if isinstance(value, list):
        return [node for item in value for node in recipe_nodes(item)]
    if not isinstance(value, dict): return []
    types = value.get('@type', [])
    if isinstance(types, str): types = [types]
    if any(str(kind).split('/')[-1] == 'Recipe' for kind in types): return [value]
    return [node for item in value.values() if isinstance(item, (dict, list)) for node in recipe_nodes(item)]


def instructions(value):
    if isinstance(value, str): return [text_only(line) for line in value.splitlines() if text_only(line)]
    if isinstance(value, list): return [step for item in value for step in instructions(item)]
    if isinstance(value, dict): return instructions(value.get('text') or value.get('itemListElement') or [])
    return []


def duration(value):
    match = re.fullmatch(r'P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?', str(value))
    if not match or not any(match.groups()): return None
    days, hours, minutes, seconds = [int(x or 0) for x in match.groups()]
    return math.ceil(days * 1440 + hours * 60 + minutes + seconds / 60) or None


def from_page(html):
    parser = PageText(); parser.feed(html)
    nodes = recipe_nodes(parser.jsonld)
    if len(nodes) > 1: raise ValueError('This page contains multiple recipes; paste the specific recipe you want')
    if nodes:
        node = nodes[0]
        try:
            recipe = Recipe(title=text_only(node.get('name', '')),
                ingredients=[text_only(item) for item in node.get('recipeIngredient', [])],
                instructions=instructions(node.get('recipeInstructions')), equipment=[],
                time_minutes=duration(node.get('totalTime')), pantry_items=None)
            return recipe, {'method': 'schema-org-recipe', 'input_sha256': sha256(html.encode()).hexdigest(),
                            'note': 'Equipment and pantry were not inferred; inspect the extracted recipe before saving.'}, json.dumps(node)
        except ValueError:
            # Partial structured data is still evidence for the normalizer; it is not silently accepted.
            text = json.dumps(node, ensure_ascii=False)
    else:
        text = '\n'.join(parser.text)
    recipe, provenance = normalize(text[:24000])
    provenance['source_truncated'] = len(text) > 24000
    return recipe, provenance, text[:24000]


def screenshot_text(encoded):
    try: raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc: raise ValueError('Screenshot must contain valid base64 image data') from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES: raise ValueError('Screenshot limit is 4 MB')
    executable = shutil.which('tesseract')
    if not executable: raise ValueError('Screenshot OCR requires Tesseract. Use the Docker backend or paste the recipe text.')
    from PIL import Image
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'} or image.width * image.height > 12_000_000:
                raise ValueError('Use a PNG, JPEG or WebP screenshot under 12 megapixels')
            image.verify()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Screenshot is not a readable supported image') from exc
    with tempfile.TemporaryDirectory(prefix='recipetriage-ocr-') as folder:
        path = Path(folder) / 'input-image'; path.write_bytes(raw)
        try:
            result = subprocess.run([executable, str(path), 'stdout', '--psm', '6'], capture_output=True, text=True, timeout=25, check=False)
        except subprocess.TimeoutExpired as exc: raise ValueError('Screenshot OCR timed out; crop the recipe or paste its text') from exc
    if result.returncode or not result.stdout.strip(): raise ValueError('No recipe text could be read; use a clearer screenshot or paste text')
    if len(result.stdout) > 24000: raise ValueError('Screenshot contains too much text; crop to one recipe')
    return result.stdout, {'method': 'tesseract-ocr', 'image_sha256': sha256(raw).hexdigest(), 'image_bytes': len(raw)}
