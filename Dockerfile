FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-optional.txt ./

# Install PyTorch with GPU support
RUN pip3 install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Conditionally install DeepSpeed (will be skipped in CI environments without CUDA)
ARG INSTALL_DEEPSPEED=true
RUN if [ "$INSTALL_DEEPSPEED" = "true" ] && [ -f /usr/local/cuda/bin/nvcc ]; then \
    pip3 install --no-cache-dir -r requirements-optional.txt; \
    else \
    echo "Skipping DeepSpeed installation"; \
    fi

# Create model directory
RUN mkdir -p /app/models

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set command
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]