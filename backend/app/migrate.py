"""Idempotent PostgreSQL migration for immutable dataset snapshots."""
from sqlalchemy import text
from app.config import Settings
from app.db import build_engine


def migrate(engine):
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(74291001)"))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS dataset_versions (
            version TEXT PRIMARY KEY, payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id UUID PRIMARY KEY, payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS training_runs (
            run_id UUID PRIMARY KEY, payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS lora_experiments (
            experiment_id UUID PRIMARY KEY, payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""))


        connection.execute(text("""CREATE TABLE IF NOT EXISTS experiment_registry (
            experiment_id UUID PRIMARY KEY, name TEXT NOT NULL, dataset_version TEXT NOT NULL,
            benchmark_sha256 TEXT NOT NULL, payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"""))
        connection.execute(text("CREATE INDEX IF NOT EXISTS experiment_registry_dataset_idx ON experiment_registry(dataset_version,benchmark_sha256)"))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS diagnostic_runs (
            run_id UUID PRIMARY KEY,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now(),updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"""))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS failure_analyses (
            source_run_id UUID PRIMARY KEY,source_sha256 TEXT NOT NULL,analysis_version TEXT NOT NULL,
            payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS failure_categories (
            source_run_id UUID REFERENCES failure_analyses(source_run_id) ON DELETE CASCADE,
            case_id TEXT NOT NULL,category TEXT NOT NULL,PRIMARY KEY(source_run_id,case_id,category))"""))
        connection.execute(text("CREATE INDEX IF NOT EXISTS failure_category_idx ON failure_categories(category)"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS learning_jobs (run_id UUID PRIMARY KEY,kind TEXT NOT NULL,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS review_candidates (candidate_id UUID PRIMARY KEY,kind TEXT NOT NULL,status TEXT NOT NULL,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS review_status_idx ON review_candidates(status)"))
        # Early generator journals mixed the provider revision with the review revision.
        # Repair only untouched pending proposals; preserve all review history and content.
        connection.execute(text("""UPDATE review_candidates
            SET payload=payload || jsonb_build_object('model_revision',payload->'revision','revision',1)
            WHERE kind='synthetic' AND status='pending'
              AND payload->>'revision' IS NULL AND COALESCE(payload->'history','[]'::jsonb)='[]'::jsonb
              AND payload->>'review' IS NULL"""))
        connection.execute(text("CREATE TABLE IF NOT EXISTS preference_pairs (pair_id UUID PRIMARY KEY,status TEXT NOT NULL,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS model_registry (model_id UUID PRIMARY KEY,stage TEXT NOT NULL CHECK(stage IN ('candidate','staging','production','archived')),payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS one_production_model ON model_registry(stage) WHERE stage='production'"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS model_stage_events (event_id UUID PRIMARY KEY,model_id UUID REFERENCES model_registry(model_id),actor TEXT NOT NULL,action TEXT NOT NULL,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS recipe_intakes (draft_id UUID PRIMARY KEY,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS library_recipes (recipe_id UUID PRIMARY KEY,content_sha256 TEXT UNIQUE NOT NULL,payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now(),updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS library_predictions (prediction_id UUID PRIMARY KEY,recipe_id UUID REFERENCES library_recipes(recipe_id),payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS library_reviews (review_id UUID PRIMARY KEY,recipe_id UUID REFERENCES library_recipes(recipe_id),payload JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now())"))


if __name__ == "__main__":
    engine = build_engine(Settings())
    try:
        migrate(engine)
    finally:
        engine.dispose()
