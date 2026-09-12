"""Choose the explicit Docker configuration or existing Community Cloud secrets."""
import os

import streamlit as st


def configuration(default_mode=None):
    if os.getenv('RECIPETRIAGE_DEPLOYMENT') == 'single-server':
        # Service DNS is fixed; this switch does not permit arbitrary HTTP origins.
        return ({'url': 'http://backend:8000', 'token': os.getenv('API_AUTH_TOKEN', ''), 'transport': 'docker'},
                {'mlflow': os.getenv('MLFLOW_DASHBOARD_URL', '')})
    try:
        if st.secrets.get('app', {}).get('mode', default_mode) == 'embedded':
            return ({'transport': 'embedded', 'database_url': st.secrets.get('database', {}).get('url', '')}, {})
        return dict(st.secrets.get('backend', {})), dict(st.secrets.get('links', {}))
    except (FileNotFoundError, KeyError):
        if default_mode == 'embedded':
            return {'transport': 'embedded', 'database_url': ''}, {}
        return {}, {}
