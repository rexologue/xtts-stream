# XTTS Stream Inference

This repository provides a stripped-down XTTS inference stack derived from the original Coqui implementation.  All training-only code has been removed and the remaining modules were refactored so the project can be used as a focused voice-cloning inference runtime.

## Features

- Autoregressive GPT, perceiver resampler, and HiFi-GAN decoder required for XTTS inference.
- Voice cloning utilities for extracting conditioning latents from reference audio.
- Sentence splitting, multilingual text normalisation, and voice caching helpers.
- Minimal command-line interface (`infer_xtts.py`) for running inference locally.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements list assumes CUDA-enabled wheels for PyTorch/Torchaudio are available on your system.  Install the appropriate builds for your environment before running inference.

## Usage

1. Download an XTTS checkpoint directory containing:
   - `model.pth`
   - `config.json`
   - `vocab.json`
   - `speakers_xtts.pth` (optional but recommended for speaker libraries)
2. Provide a reference utterance (≥3 seconds) for cloning the target voice.

Run inference:

```bash
python infer_xtts.py \
  --config /path/to/config.json \
  --checkpoint /path/to/model.pth \
  --tokenizer /path/to/vocab.json \
  --speakers /path/to/speakers_xtts.pth \
  --text "Your text goes here" \
  --language en \
  --reference /path/to/reference.wav \
  --output ./generated.wav
```

Set `--device cpu` if you do not have a GPU available.  Advanced sampling controls (temperature, top-k/p, length penalties, speed, etc.) can be overridden via CLI flags; see `python infer_xtts.py --help` for the full list.

## Repository Layout

- `xtts.py`, `gpt.py`, `autoregressive.py`, `perceiver_encoder.py`, `xtransformers.py` – core autoregressive stack.
- `hifigan_decoder.py`, `hifigan_generator.py`, `resnet.py` – neural vocoder and speaker encoder.
- `tokenizer.py`, `text/`, `zh_num2words.py` – multilingual tokenisation utilities.
- `generic_utils.py`, `shared_configs.py`, `xtts_config.py`, `xtts_manager.py` – configuration helpers.
- `stream_generator.py`, `helpers.py`, `arch_utils.py`, `transformer.py` – shared utilities for generation.
- `infer_xtts.py` – command-line wrapper for end-to-end inference.

## Notes

- Sentence splitting uses spaCy language pipelines when `--split-text` is set.  Install the corresponding spaCy language models as needed.
- Noise reduction (via `noisereduce`) is applied to streaming output chunks.  Comment the call in `xtts.py::_apply_noise_reduction` if a raw waveform is preferred.
- The repository does not include any training facilities; the only supported workflow is inference using existing XTTS checkpoints.
