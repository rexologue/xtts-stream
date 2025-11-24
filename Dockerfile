# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    API_PORT=8000

WORKDIR /app

# ------------------------------------------------------------------------------
# Install Python 3.12 + system dependencies for audio processing and build tools
# ------------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      software-properties-common \
      build-essential \
      git \
      curl \
      ffmpeg \
      libsndfile1 \
      libgl1 \
      libglib2.0-0 \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      python3.12-dev \
 && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.12 /usr/bin/python \
 && python3 -m ensurepip \
 && python3 -m pip install --upgrade pip setuptools wheel \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------------------------
COPY requirements_locked.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt && rm -rf /root/.cache/pip /tmp/*

# ------------------------------------------------------------------------------
# Copy the application code
# ------------------------------------------------------------------------------
COPY src ./src

# Runtime will fail fast if XTTS_CONFIG_FILE is not provided
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE ${API_PORT}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "xtts_stream.api.service.balancer"]

