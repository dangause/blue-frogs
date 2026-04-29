# --- Build stage ---
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

# Install CPU-only PyTorch first, then the project, then extras
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir ultralytics

# --- Runtime stage ---
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files needed at runtime
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Data directory is a volume mount point
VOLUME ["/app/data"]

ENTRYPOINT ["python", "scripts/smoke_test.py", "--data-dir", "/app/data", "--output-dir", "/app/data/smoke_output"]
