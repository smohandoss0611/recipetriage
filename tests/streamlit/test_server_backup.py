import subprocess

import pytest

pytest.importorskip('streamlit')
from infra.server.backup import backup


def test_backup_failure_still_restarts_services_and_does_not_publish_manifest(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'backend-id\nstreamlit-id\n')
    def run(command, **kwargs):
        commands.append(command)
        if 'pg_dump' in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        backup(tmp_path)
    assert 'stop' in commands[0]
    assert 'up' in commands[-1] and '--wait' in commands[-1]
    assert commands[-1][-2:] == ['backend', 'streamlit']
    assert not list(tmp_path.rglob('manifest.json'))


def test_backup_rejects_unrelated_project_and_stopped_services(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='single-server'):
        backup(tmp_path, 'recipetriage')
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'backend-only\n')
    with pytest.raises(RuntimeError, match='both be running'):
        backup(tmp_path)
    assert not (tmp_path/'.server').exists()
