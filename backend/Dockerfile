FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY backend/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY ml/requirements-hf.lock /app/requirements-hf.lock
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-hf.lock
COPY ml/requirements-sft.lock /app/requirements-sft.lock
RUN pip install --no-cache-dir -r requirements-sft.lock
COPY ml/requirements-experiments.lock /app/requirements-experiments.lock
RUN pip install --no-cache-dir -r requirements-experiments.lock
COPY backend/requirements-intake.lock /app/requirements-intake.lock
RUN pip install --no-cache-dir -r requirements-intake.lock \
    && apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY ml /app/ml
RUN chmod -R a+rX /app/ml && pip install --no-cache-dir --no-deps ./ml
COPY backend /app/backend
RUN pip install --no-cache-dir --no-deps ./backend && useradd --create-home appuser && mkdir -p /models /training && chown appuser:appuser /models /training

FROM base AS test
ENV PYTEST_ADDOPTS="-o cache_dir=/tmp/pytest-cache"
COPY backend/requirements-test.lock /app/requirements-test.lock
RUN pip install --no-cache-dir -r requirements-test.lock
COPY --chmod=755 tests /app/tests
COPY --chmod=644 pytest.ini /app/pytest.ini
USER appuser
CMD ["sh", "-c", "python -m app.migrate && pytest -q"]

FROM base AS runtime
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "python -m app.migrate && python -m app.library --seed && python -m app.recover_benchmarks && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]

# Cloud hosts build this final target. Compose explicitly retains target: runtime.
FROM base AS cloud
USER root
ENV APP_ENV=cloud TRAINING_RUNS_DIR=/training HF_HOME=/models HF_CPU_THREADS=4
EXPOSE 8000
CMD ["python", "-m", "app.cloud_start"]
