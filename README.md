# XTTS Stream Inference

XTTS Stream Inference is a focused voice-cloning runtime that keeps only the
modules required to run the XTTS architecture. It bundles the autoregressive
GPT, perceiver resampler, HiFi-GAN decoder, text normalisation helpers, and a
FastAPI streaming service so the full inference pipeline is ready out of the box.

## Features

- Voice cloning utilities for extracting conditioning latents from reference audio.
- Sentence splitting, multilingual normalisation, and voice caching helpers.
- Minimal command-line interface for offline synthesis.
- ElevenLabs-compatible websocket endpoint for real-time streaming.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add the repository to the Python path before running any commands (from the
repository root):

```bash
export PYTHONPATH="$(pwd)/src"
```

The requirements list assumes CUDA-enabled wheels for PyTorch/Torchaudio are
available on your system. Install the appropriate builds for your environment
before running inference.

## Configuration

Service settings and model paths are loaded from a YAML configuration file. Use
`config.example.yaml` as a template, copy it anywhere on your filesystem, and
update the fields to point at your XTTS checkpoint directory and reference
speaker audio. Required artefacts inside the model directory are:

   - `config.json`
   - `model.pth`
   - `dvae.pth`
   - `mel_stats.pth`
   - `vocab.json`

   You can also override the device, default language, and optional filenames in
   the same section.

Always set `XTTS_CONFIG_FILE` to the absolute path of the configuration before
launching any command that touches the balancer or worker processes.

## Usage

1. Download an XTTS checkpoint directory containing:
   - `model.pth`
   - `config.json`
   - `vocab.json`
   - `speakers_xtts.pth` (optional but recommended for speaker libraries)
2. Provide a reference utterance (≥3 seconds) for cloning the target voice.

Run offline inference:

```bash
PYTHONPATH=src python -m xtts_stream.inference.infer_xtts \
  --config /path/to/config.json \
  --checkpoint /path/to/model.pth \
  --tokenizer /path/to/vocab.json \
  --speakers /path/to/speakers_xtts.pth \
  --text "Your text goes here" \
  --language en \
  --reference /path/to/reference.wav \
  --output ./generated.wav
```

Set `--device cpu` if you do not have a GPU available. Advanced sampling
controls (temperature, top-k/p, length penalties, speed, etc.) can be overridden
via CLI flags; see `python -m xtts_stream.inference.infer_xtts --help` for the full
list.

To stream audio chunks while they are generated, add `--stream` (optionally
configuring `--stream-chunk-size` and `--stream-overlap`):

```bash
PYTHONPATH=src python -m xtts_stream.inference.infer_xtts \
  --config /path/to/config.json \
  --checkpoint /path/to/model.pth \
  --tokenizer /path/to/vocab.json \
  --speakers /path/to/speakers_xtts.pth \
  --text "Your text goes here" \
  --language en \
  --reference /path/to/reference.wav \
  --output ./generated.wav \
  --stream
```

### Streaming service

The websocket service now runs as a balancer with multiple worker processes. Each
worker loads a single XTTS instance, and the balancer proxies incoming
ElevenLabs-compatible `stream-input` requests to the first free worker.

The balancer can run in two modes controlled by `service.standalone_mode` in the
YAML configuration:

- `standalone_mode: true` (default) — the balancer exposes the ElevenLabs
  websocket directly and manages its local worker pool exactly as before.
- `standalone_mode: false` — the balancer registers with a central Broker
  service and only accepts websocket traffic that originates from the Broker.
  Client applications should point to the Broker host/port instead of the
  balancer.

When Broker mode is enabled the balancer requires two extra fields in the YAML:
`broker_host` and `broker_port`, which tell it where to register and receive
forwarded websocket connections.

Ensure your configuration file is in place (see the [Configuration](#configuration)
section) and start the service with:

```bash
XTTS_CONFIG_FILE=/absolute/path/to/config.yaml PYTHONPATH=src \
  python -m xtts_stream.api.service.balancer
```

### Broker service

The Broker coordinates multiple balancers (potentially on separate machines)
while keeping the ElevenLabs-compatible API stable for clients. It listens for
client websocket connections, chooses a balancer based on the configured
strategy (`deep`, `wide`, or `random`), and proxies the stream to that balancer.

1. Copy and edit `broker.config.example.yaml`, then set
   `XTTS_BROKER_CONFIG_FILE=/absolute/path/to/broker.config.yaml`.
2. Start the Broker:

```bash
PYTHONPATH=src python -m xtts_stream.api.broker.server
```

Sample Docker assets live under `src/xtts_stream/api/broker/` for running the
Broker independently from GPU-equipped balancer hosts. Each balancer set to
`standalone_mode: false` will automatically register with the Broker on startup
and expose its `/workers/idle` endpoint for capacity checks.

Wrapper classes located under `src/xtts_stream/api/wrappers` provide reusable hooks
for other models. Implement `xtts_stream.api.wrappers.base.StreamingTTSWrapper` for
new backends and import your implementation inside the worker module.

### Docker deployment

The repository contains a `Dockerfile` and `docker-compose.yaml` to simplify running
the websocket API in a container. Build the image and launch the service with:

```bash
# Build the runtime image (CUDA 12.1 + cuDNN runtime)
docker compose build

# Ensure your custom config file is ready and referenced via XTTS_CONFIG_FILE
export XTTS_CONFIG_FILE="$(pwd)/config.yaml"

# Start the websocket API on port 8000 with GPU access
docker compose up
```

The compose file mounts the host path provided via `XTTS_CONFIG_FILE` into
`/app/config.yaml`, exposes port `8000`, and requests all available NVIDIA GPUs.
Adjust the environment variable before launching to point at the desired config
file and add extra `volumes` entries pointing to the XTTS checkpoint directory
and any reference audio files so they are available inside the container.

## Repository layout

- `src/xtts_stream/core/` – XTTS inference stack (autoregressive GPT, vocoder,
  tokenisers, helpers, etc.).
- `src/xtts_stream/inference/` – command-line utilities for offline and dataset
  streaming inference.
- `src/xtts_stream/api/service/` – FastAPI application exposing the ElevenLabs
  compatible websocket endpoint.
- `src/xtts_stream/api/wrappers/` – reusable abstractions for streaming TTS
  engines.
- `src/xtts_stream/api/client/` – reference websocket client mirroring the
  ElevenLabs streaming format.

## Notes

- Sentence splitting uses spaCy language pipelines when `--split-text` is set.
  Install the corresponding spaCy language models as needed.
- Noise reduction (via `noisereduce`) is applied to streaming output chunks.
  Comment the call in `xtts_stream.core.xtts::_apply_noise_reduction` if a raw
  waveform is preferred.

The XTTS architecture was originally created and open-sourced by Coqui.
