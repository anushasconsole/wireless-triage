# ---------------------------------------------------------------------------
# Wireless Triage — Docker image
# Runs both FastAPI (port 8000) and Streamlit UI (port 8501)
# ---------------------------------------------------------------------------

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc supervisor \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models outputs

# Train models at build time
RUN python train_all.py

# Supervisor config to run both services
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
