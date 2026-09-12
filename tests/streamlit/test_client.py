import httpx
import pytest

pytest.importorskip('streamlit')
from streamlit_ui.client import API, APIError


def test_authentication_is_server_side_and_redirects_are_not_followed():
    seen = []
    def handler(request):
        seen.append(request)
        assert request.headers['authorization'] == 'Bearer private-token'
        return httpx.Response(302, headers={'Location': 'https://different.example/'})
    api = API('https://api.example', 'private-token', httpx.MockTransport(handler))
    with pytest.raises(APIError, match='HTTP 302'):
        api.get('/api/v1/models')
    assert len(seen) == 1
    assert 'private-token' not in repr(api)


def test_backend_validation_errors_are_visible_without_secrets_or_retries():
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(422, json={'detail': 'Invalid labels private-token'})
    api = API('https://api.example', 'private-token', httpx.MockTransport(handler))
    with pytest.raises(APIError, match='Invalid labels') as error:
        api.post('/api/v1/recipes', {})
    assert 'private-token' not in str(error.value)
    assert len(seen) == 1


@pytest.mark.parametrize('url,token', [('http://api.example', 'token'), ('https://api.example', ''), ('https://user:password@api.example', 'token')])
def test_remote_configuration_requires_secure_transport_and_token(url, token):
    with pytest.raises(ValueError):
        API(url, token)


def test_docker_mode_uses_only_the_fixed_private_service_and_keeps_authentication():
    def handler(request):
        assert str(request.url) == 'http://backend:8000/health'
        assert request.headers['authorization'] == 'Bearer ' + 't' * 48
        return httpx.Response(200, json={'status': 'ok'})
    api = API('http://backend:8000', 't' * 48, httpx.MockTransport(handler), docker_network=True)
    assert api.get('/health')['status'] == 'ok'
    with pytest.raises(ValueError, match='HTTPS'):
        API('http://backend:8000', 't' * 48)


@pytest.mark.parametrize('url,token', [
    ('http://api.example', 't' * 48), ('http://backend:8000.evil.example', 't' * 48),
    ('http://backend:8000/extra', 't' * 48), ('http://backend:8000', ''),
    ('http://backend:8000', 'short'), ('http://localhost:8000', 't' * 48),
])
def test_private_docker_setting_is_not_a_general_insecure_transport_bypass(url, token):
    with pytest.raises(ValueError, match='Docker mode'):
        API(url, token, docker_network=True)
