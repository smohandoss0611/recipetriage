"""Single-host cloud startup with persistent data and existing model paths."""
import os
from pathlib import Path
import subprocess
import sys
import shutil

from app.auth import configured_token


def main():
    os.environ['APP_ENV'] = 'cloud'
    configured_token()
    root = Path('/data')
    root.mkdir(exist_ok=True)
    for name in ('training', 'models', 'deployment'):
        target = root / name
        target.mkdir(exist_ok=True)
        link = Path('/') / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f'{link} must point into the persistent /data volume.')
        elif link.exists():
            # Never remove data to establish the cloud volume mapping.
            if not link.is_dir() or any(link.iterdir()):
                raise RuntimeError(f'{link} is not empty; move it into /data before cloud startup.')
            link.rmdir()
            link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    if os.getuid() == 0:
        import pwd
        account = pwd.getpwnam('appuser')
        for path in [root, root/'training', root/'models', root/'deployment']:
            os.chown(path, account.pw_uid, account.pw_gid)
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    os.environ['TRAINING_RUNS_DIR'] = '/training'
    os.environ['HF_HOME'] = '/models'
    references = {
        'sft-v1': 'sft-v1/local',
        'lora-v1': 'experiments-v1/runs/771d20b1-77df-446b-b34d-b8259ddbc585',
        'qlora-v1': 'experiments-v1/runs/041818c9-7e00-4802-b53d-f7d567616e2b',
    }
    for name, relative in references.items():
        source = Path('/app/ml/training/results') / relative
        target = Path('/training/references') / name
        if target.exists() and not (target/'adapter/adapter_model.safetensors').is_file():
            raise RuntimeError(f'Incomplete reference at {target}; restore it from the delivered adapter before startup.')
        if not target.exists() and (source/'adapter/adapter_model.safetensors').is_file():
            pending = target.with_name('.' + name + '.installing')
            if pending.exists():
                shutil.rmtree(pending)
            shutil.copytree(source, pending)
            pending.rename(target)
    for module in ('app.migrate', 'app.library', 'app.recover_benchmarks'):
        command = [sys.executable, '-m', module]
        if module == 'app.library':
            command.append('--seed')
        subprocess.run(command, check=True)
    port = int(os.getenv('PORT', '8000'))
    os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0',
                            '--port', str(port), '--workers', '1'])


if __name__ == '__main__':
    main()
