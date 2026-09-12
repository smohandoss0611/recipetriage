import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_DB_TESTS") != "1", reason="Set RUN_DB_TESTS=1 with PostgreSQL running")
def test_real_postgres_readiness():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "ok", "database": "ok"}
