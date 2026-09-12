"""Keep every integration test outside the running application's PostgreSQL schema."""
import os
from uuid import uuid4
import pytest


@pytest.fixture(scope='session', autouse=True)
def isolated_database_schema():
    if os.getenv('RUN_DB_TESTS') != '1':
        yield
        return
    from sqlalchemy import Engine, event, text
    from app.config import Settings
    from app.db import build_engine
    primary = build_engine(Settings())
    schema = 'test_suite_' + uuid4().hex
    with primary.begin() as db:
        db.execute(text('CREATE SCHEMA ' + schema))

    def use_test_schema(dialect, connection_record, arguments, parameters):
        if dialect.name != 'postgresql':
            return
        options = parameters.get('options', '')
        # Newer tests already use individual schemas; retain their tighter isolation.
        if 'search_path=' not in options:
            parameters['options'] = options + ' -c search_path=' + schema

    event.listen(Engine, 'do_connect', use_test_schema)
    try:
        yield
    finally:
        event.remove(Engine, 'do_connect', use_test_schema)
        with primary.begin() as db:
            db.execute(text('DROP SCHEMA ' + schema + ' CASCADE'))
        primary.dispose()
