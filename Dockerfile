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

# System dependencies for Python 3.10, audio processing and build tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        curl \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src ./src

# Runtime will fail fast if XTTS_SETTINGS_FILE is not provided
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE ${API_PORT}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "xtts_stream.api.service.app"]
