FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (defaults to 8080).
# 1 worker keeps each participant's on-disk move/conversation files in a single
# process; gthread threads handle concurrent participants, the streaming
# /api/chat endpoint, and the parallel prompt-judging calls. Generous timeout so
# long chat streams aren't killed.
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --worker-class gthread --workers 1 --threads 16 --timeout 300 app:app
