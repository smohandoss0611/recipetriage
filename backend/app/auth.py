"""Private API boundary for a trusted Streamlit server or other operator client."""
import hmac
import os

from starlette.responses import JSONResponse


def configured_token():
    token = os.getenv('API_AUTH_TOKEN', '')
    if os.getenv('APP_ENV', 'local') == 'cloud' and len(token) < 32:
        raise RuntimeError('Cloud startup requires an API_AUTH_TOKEN with at least 32 characters.')
    return token


class APITokenMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or (scope.get('path') == '/health' and scope.get('method') in {'GET', 'HEAD'}):
            return await self.app(scope, receive, send)
        try:
            expected = configured_token()
        except RuntimeError:
            return await JSONResponse({'detail': 'API authentication is not configured.'}, status_code=503)(scope, receive, send)
        if expected:
            headers = dict(scope.get('headers', []))
            supplied = headers.get(b'authorization', b'')
            if not hmac.compare_digest(supplied, ('Bearer ' + expected).encode()):
                return await JSONResponse({'detail': 'Authentication required.'}, status_code=401,
                                          headers={'WWW-Authenticate': 'Bearer'})(scope, receive, send)
        return await self.app(scope, receive, send)
