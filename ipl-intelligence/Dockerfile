# Multi-stage: deps layer cached separately from code; runs as non-root.
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM deps AS runtime
RUN useradd --create-home appuser
COPY --chown=appuser core ./core
COPY --chown=appuser db ./db
COPY --chown=appuser ingest ./ingest
COPY --chown=appuser stats ./stats
COPY --chown=appuser rag ./rag
COPY --chown=appuser agent ./agent
COPY --chown=appuser mcp_server ./mcp_server
COPY --chown=appuser api ./api
COPY --chown=appuser frontend ./frontend
COPY --chown=appuser tests ./tests
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface
EXPOSE 8000 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
