FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    DB_PATH=/srv/data/history.db \
    RAG_INDEX_PATH=/srv/data/rag_index.json

WORKDIR /srv

# Dependencies install as their own layer so code edits do not invalidate them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /srv/data \
    && chown appuser:appuser /srv/data
COPY --chown=appuser:appuser app ./app
USER appuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '3000') + '/health').status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
