FROM python:3.11-slim

WORKDIR /app/env

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy entire env (models.py, client.py, data/, server/, pyproject.toml, etc.)
COPY . .

# Install env + dependencies
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
# /app/env on PYTHONPATH so `from models import ...` and `from server.app import ...` resolve
ENV PYTHONPATH="/app/env"
ENV PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ws-ping-interval", "300", "--ws-ping-timeout", "300"]
