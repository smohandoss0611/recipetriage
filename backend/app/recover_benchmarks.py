"""Run at API process startup, never as a side effect of opening a test client."""
from app.config import Settings
from app.db import build_engine
from app.benchmarks import recover_interrupted

if __name__ == '__main__':
    engine = build_engine(Settings())
    try:
        recover_interrupted(engine)
        from app.training import recover_interrupted as recover_training
        recover_training(engine)
        from app.lora import recover_interrupted as recover_lora
        recover_lora(engine)
        from app.experiments import recover_interrupted as recover_experiments
        recover_experiments(engine)
        from app.analysis import recover_interrupted as recover_analysis
        recover_analysis(engine)
        from app.jobs import recover
        recover(engine)
    finally:
        engine.dispose()
