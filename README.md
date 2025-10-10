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

Service settings and model paths are loaded from a YAML configuration file.

1. Copy `config.example.yaml` to `config.yaml` (or another filename of your
   choice):

   ```bash
   cp config.example.yaml config.yaml
   ```

2. Edit the file to point at your XTTS checkpoint directory and reference
   speaker audio. Required artefacts inside the model directory are:

   - `config.json`
   - `model.pth`
   - `dvae.pth`
   - `mel_stats.pth`
   - `vocab.json`

   You can also override the device, default language, and optional filenames in
   the same section.

3. By default the service reads `config.yaml` from the repository root. To use a
   custom path, set `XTTS_SETTINGS_FILE` before launching any commands:

   ```bash
   export XTTS_SETTINGS_FILE=/absolute/path/to/your-config.yaml
   ```

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

The websocket service lives in `src/xtts_stream/api/service/app.py` and exposes the
ElevenLabs-compatible `stream-input` protocol. Start it with:

Ensure your configuration file is in place (see the [Configuration](#configuration)
section) and start the service with:

```bash
PYTHONPATH=src python -m xtts_stream.api.service.app
```

Wrapper classes located under `src/xtts_stream/api/wrappers` provide reusable hooks
for other models. Implement `xtts_stream.api.wrappers.base.StreamingTTSWrapper` for
new backends and import your implementation inside the service module.

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
