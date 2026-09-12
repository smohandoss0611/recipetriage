"""Create a consistent database/artifact backup during explicit maintenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def backup(root, project='recipetriage-server'):
    if not re.fullmatch(r'recipetriage-server(?:-[a-z0-9-]+)?', project):
        raise ValueError('Use the single-server project name, not the local React stack.')
    root = Path(root).resolve()
    compose = ['docker', 'compose', '--env-file', str(root/'.server/compose.env'), '-f',
               str(root/'compose.server.yml'), '-p', project]
    running = subprocess.check_output(compose + ['ps', '--status', 'running', '-q', 'backend', 'streamlit'], text=True)
    if len(running.splitlines()) != 2:
        raise RuntimeError('Backend and Streamlit must both be running before entering maintenance.')
    destination = root/'.server/backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True, mode=0o700)
    try:
        subprocess.run(compose + ['stop', 'streamlit', 'backend'], check=True)
        operations = {
            'database.dump': ['exec', '-T', 'postgres', 'pg_dump', '-U', 'recipetriage', '-d', 'recipetriage',
                              '--format=custom', '--no-owner', '--no-acl'],
            'artifacts.tar.gz': ['run', '-T', '--rm', '--no-deps', '--user', 'root', 'backend',
                                 'tar', '-C', '/data', '-czf', '-', 'training', 'deployment'],
        }
        manifest = {}
        for name, command in operations.items():
            path = destination/name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as handle:
                subprocess.run(compose + command, check=True, stdout=handle)
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            manifest[name] = {'sha256': digest, 'bytes': path.stat().st_size}
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    finally:
        # Always attempt recovery, including after a failed/partial backup.
        subprocess.run(compose + ['up', '-d', '--no-deps', '--wait', '--wait-timeout', '180', 'backend', 'streamlit'], check=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maintenance', action='store_true', help='Allow stopping UI/backend while taking the backup.')
    parser.add_argument('--project', default='recipetriage-server', help='Override only for an isolated test deployment.')
    args = parser.parse_args()
    if not args.maintenance:
        raise SystemExit('Wait for training to finish, then pass --maintenance. This backup briefly stops the UI and backend.')
    print(backup(Path(__file__).resolve().parents[2], args.project))
