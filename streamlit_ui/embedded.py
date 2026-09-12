"""Server-side, in-process transport; no HTTP listener, subprocess or API token."""
import asyncio
import atexit
import logging
import traceback

import httpx
import streamlit as st

from .client import API, APIError

logger = logging.getLogger(__name__)


class EmbeddedAPI(API):
    embedded = True

    def __init__(self, application):
        super().__init__('http://localhost')
        self.application = application

    def request(self, method, path, body=None, *, binary=False):
        self.validate_path(path)

        async def dispatch():
            transport = httpx.ASGITransport(app=self.application, raise_app_exceptions=True)
            async with httpx.AsyncClient(transport=transport, base_url=self.url) as client:
                return await client.request(method, path, json=body)

        try:
            response = asyncio.run(dispatch())
        except Exception as exc:
            # Do not expose DB connection strings, SQL payloads or user data.
            # Stack locations aid debugging without logging exception values,
            # which may include connection strings or submitted records.
            logger.error('Embedded operation failed: %s %s (%s)\n%s', method, path, type(exc).__name__, ''.join(traceback.format_tb(exc.__traceback__)))
            from sqlalchemy.exc import SQLAlchemyError
            reason = 'The database operation failed.' if isinstance(exc, SQLAlchemyError) else 'The application operation failed (' + type(exc).__name__ + ').'
            retry = 'Refresh saved records before resubmitting; a write may already have completed.' if method != 'GET' else 'You can retry loading this page.'
            raise APIError(reason + ' Check Streamlit logs. ' + retry) from None
        return self.read_response(response, binary=binary)


@st.cache_resource(show_spinner='Connecting to PostgreSQL…')
def embedded_api(database_url):
    if not database_url:
        raise ValueError('Set database.url in Streamlit secrets to your PostgreSQL connection URL. See STREAMLIT_FREE.md.')
    from app.embedded import build_runtime, database_settings
    try:
        database_settings(database_url)
    except ValueError:
        raise ValueError('Set a valid PostgreSQL database.url with a host and database name. Remote connections require ?sslmode=require. See STREAMLIT_FREE.md.') from None
    try:
        application = build_runtime(database_url)
    except Exception as exc:
        from sqlalchemy.engine import make_url
        detail = str(exc).replace(database_url, '[database URL redacted]')
        password = make_url(database_url).password
        if password:
            detail = detail.replace(password, '[password redacted]')
        logger.error('Embedded startup failed (%s): %s', type(exc).__name__, detail)
        raise APIError('Could not initialize PostgreSQL. Check database.url, SSL, network access and schema permissions in Streamlit secrets, then retry.') from None
    atexit.register(application.state.engine.dispose)
    return EmbeddedAPI(application)
