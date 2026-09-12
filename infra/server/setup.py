"""Create private first-install configuration using only the Python standard library."""
import argparse
from getpass import getpass
import os
from pathlib import Path
import re
import secrets
import shutil

MODEL = 'accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b'


def write_env(path, values):
    if any('\n' in value or '\r' in value or '\x00' in value for value in values.values()):
        raise ValueError('Configuration values must contain a single line.')
    # Compose env_file format: raw deliberately preserves $, quotes and backslashes.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(''.join(f'{key}={value}\n' for key, value in values.items()))


def create_configuration(destination, domain, fireworks_key='', *, local_check=False):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Configuration already exists; refusing to overwrite credentials or reset database access.')
    domain = domain.strip().lower()
    label = r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?'
    if not local_check and (len(domain) > 253 or not re.fullmatch(rf'{label}(?:\.{label})+', domain)
                            or domain.endswith(('.local', '.localhost', '.internal', '.example'))
                            or domain == 'example.com' or re.fullmatch(r'[0-9.]+', domain)):
        raise ValueError('Enter your public DNS hostname only, without https://, a path or a port.')
    api_token, db_password = secrets.token_urlsafe(48), secrets.token_urlsafe(48)
    destination.mkdir(mode=0o700, parents=True)
    try:
        write_env(destination/'compose.env', dict(APP_DOMAIN='localhost' if local_check else domain,
                  SERVER_BIND='127.0.0.1' if local_check else '0.0.0.0',
                  HTTP_PORT='8580' if local_check else '80', HTTPS_PORT='8543' if local_check else '443'))
        write_env(destination/'postgres.env', dict(POSTGRES_USER='recipetriage', POSTGRES_DB='recipetriage', POSTGRES_PASSWORD=db_password))
        write_env(destination/'backend.env', dict(APP_ENV='cloud', API_AUTH_TOKEN=api_token,
                  POSTGRES_USER='recipetriage', POSTGRES_DB='recipetriage', POSTGRES_PASSWORD=db_password,
                  POSTGRES_HOST='postgres', POSTGRES_PORT='5432', FIREWORKS_API_KEY=fireworks_key,
                  FIREWORKS_MODEL=MODEL, FIREWORKS_RECIPE_MODEL=MODEL, FIREWORKS_SYNTHETIC_MODEL=MODEL,
                  MLFLOW_TRACKING_URI='', MLFLOW_EXPERIMENT_NAME='RecipeTriage-fixed-benchmark'))
        write_env(destination/'streamlit.env', dict(RECIPETRIAGE_DEPLOYMENT='single-server',
                  API_AUTH_TOKEN=api_token, MLFLOW_DASHBOARD_URL=''))
    except BaseException:
        # Only this invocation's newly-created directory is removed on partial failure.
        shutil.rmtree(destination)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', help='Your DNS hostname; omit to enter it interactively.')
    parser.add_argument('--local-check', action='store_true', help='Use localhost TLS and loopback ports for an isolated test.')
    args = parser.parse_args()
    destination = Path(__file__).resolve().parents[2]/'.server'
    if destination.exists():
        raise SystemExit('Existing .server directory preserved. Follow SINGLE_SERVER.md to change configuration.')
    domain = 'localhost' if args.local_check else (args.domain or input('Public hostname: '))
    key = getpass('Fireworks API key (Enter to leave hosted inference disabled): ')
    try:
        create_configuration(destination, domain, key, local_check=args.local_check)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print('Created private .server configuration. Credentials were not printed. See SINGLE_SERVER.md for startup commands.')


if __name__ == '__main__':
    main()
