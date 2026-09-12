from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db import get_session
from app.main import app


@pytest.fixture
def client():
    # No lifespan: isolate the HTTP contract without opening a database.
    session = Mock()
    app.dependency_overrides[get_session] = lambda: session
    try:
        yield TestClient(app), session
    finally:
        app.dependency_overrides.clear()


def test_health_success(client):
    http, session = client
    response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}
    assert str(session.execute.call_args.args[0]) == "SELECT 1"


def test_health_returns_503_when_database_is_unavailable(client):
    http, session = client
    session.execute.side_effect = OperationalError("SELECT 1", {}, Exception("private database details"))
    response = http.get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert "private" not in response.text
