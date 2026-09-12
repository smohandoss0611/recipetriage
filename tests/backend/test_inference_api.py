from fastapi.testclient import TestClient
from app.main import app
from app import inference
from recipetriage_ml.inference.contracts import ProviderError
from recipetriage_ml.inference.experiment import fixture

client = TestClient(app)


def test_triage_endpoint(monkeypatch):
    monkeypatch.setattr(inference, "triage", lambda *args: {"valid_json": True, "prediction": {"labels": ["weekend-project"], "explanation":"70 minutes"}})
    response = client.post('/api/v1/triage', json={"recipe": fixture()["recipe"]})
    assert response.status_code == 200
    assert response.json()['prediction']['labels'] == ['weekend-project']


def test_invalid_request():
    response = client.post('/api/v1/triage', json={"recipe": {"title": "Empty"}})
    assert response.status_code == 422
    response = client.post('/api/v1/triage', json={"recipe":fixture()["recipe"], "max_new_tokens":0})
    assert response.status_code == 422


def test_provider_unavailable(monkeypatch):
    def fail(*args):
        raise ProviderError('FIREWORKS_API_KEY is not configured on the server')
    monkeypatch.setattr(inference, 'triage', fail)
    response = client.post('/api/v1/triage', json={"recipe":fixture()["recipe"],"provider":"fireworks"})
    assert response.status_code == 503
    assert 'not configured' in response.json()['detail']


def test_token_endpoint_uses_chat_option(monkeypatch):
    def inspect(text, messages):
        assert messages[1]['content'] == text
        return {"token_ids":[1],"tokens":["x"],"token_count":1}
    monkeypatch.setattr(inference, 'inspect_tokens', inspect)
    response = client.post('/api/v1/tokens', json={"text":"x","chat_template":True})
    assert response.status_code == 200
    assert response.json()['token_count'] == 1
