# Build stage for PyTorch
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install minimal dependencies for downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create wheels directory
WORKDIR /wheels

# Set long timeout and download PyTorch
RUN pip config set global.timeout 3000 && \
    pip3 wheel --wheel-dir=/wheels torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Final stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    awscli \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy Python dependencies files
COPY requirements.txt ./

# Copy PyTorch wheels from builder stage
COPY --from=builder /wheels /wheels

# Install PyTorch from wheels
RUN pip3 install --no-cache-dir /wheels/*.whl

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install DeepSpeed separately since it's optional and large
RUN pip3 install --no-cache-dir deepspeed==0.16.4

# Create model directory
RUN mkdir -p /app/models

# Copy application code (do this last to maximize caching)
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set command
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]