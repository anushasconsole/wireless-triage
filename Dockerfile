# ---------------------------------------------------------------------------
# Wireless Triage API — Docker image
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runtime directories
RUN mkdir -p data models outputs

# Pre-train models at build time so the container is self-contained.
# Comment out this line if you prefer to mount pre-trained models via a volume.
RUN python train_all.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
