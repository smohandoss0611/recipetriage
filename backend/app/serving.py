"""Read the production pointer per request; cache only its immutable model version."""
from functools import lru_cache
from threading import Lock
from app.registry import production

_load_lock=Lock()


@lru_cache(maxsize=1)
def load_version(model_id,directory,format):
    if format.startswith('peft-'):
        from recipetriage_ml.training.sft import load_adapter
        return load_adapter(directory)
    from recipetriage_ml.deployment.benchmark import load_export
    return load_export(directory)


def provider(session):
    model=production(session)
    with _load_lock:return load_version(model['model_id'],model['artifact_directory'],model['format'])
