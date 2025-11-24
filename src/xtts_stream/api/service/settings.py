"""Strict YAML configuration loader for the streaming service.

All runtime options must be sourced from the provided YAML file. The only
permitted environment variable is ``XTTS_CONFIG_FILE`` which selects the YAML
path. No other overrides are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml


class SettingsError(RuntimeError):
    """Raised when the application configuration is invalid."""


@dataclass
class ServiceSettings:
    """Runtime settings for the public balancer service."""

    host: str
    port: int
    instances: int


@dataclass
class ModelSettings:
    """Model artefact locations and runtime options."""

    directory: Path
    config_path: Path
    checkpoint_path: Path
    tokenizer_path: Optional[Path]
    speaker_wav: Path
    language: str
    device: str = "auto"
    enable_accentizer: bool = True


@dataclass
class Settings:
    """Top-level configuration bundle."""

    service: ServiceSettings
    model: ModelSettings


def _require_path(base: Path, relative: str, *, kind: str) -> Path:
    candidate = (base / relative).expanduser().resolve()
    if kind == "file" and (not candidate.exists() or candidate.is_dir()):
        raise SettingsError(f"Expected file does not exist: {candidate}")
    if kind == "dir" and (not candidate.exists() or not candidate.is_dir()):
        raise SettingsError(f"Expected directory does not exist: {candidate}")
    return candidate


def load_settings(path: Path) -> Settings:
    """Load and validate settings from ``path``."""

    if not path.exists():
        raise SettingsError(f"Configuration file not found: {path}")

    if not torch.cuda.is_available():
        raise SettingsError("CUDA is not available on this machine! Service working is not possible.")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SettingsError(f"Failed to parse configuration file: {path}") from exc

    try:
        service_raw = raw["service"]
        model_raw = raw["model"]
    except KeyError as exc:
        raise SettingsError(f"Missing configuration block: {exc.args[0]}")

    try:
        service = ServiceSettings(
            host=str(service_raw["host"]),
            port=int(service_raw["port"]),
            instances=int(service_raw["instances"]),
        )
    except KeyError as exc:
        raise SettingsError(f"Service configuration is incomplete! Missing {exc.args[0]}")

    if not (1 <= service.port <= 65535):
        raise SettingsError("Invalid service port specification!")
    if service.instances < 1:
        raise SettingsError("instances must be >= 1")

    try:
        model_dir = Path(model_raw["directory"]).expanduser().resolve()
        language = str(model_raw["language"])
    except KeyError as exc:
        raise SettingsError(f"Model configuration is incomplete! Missing {exc.args[0]}")

    if not model_dir.exists() or not model_dir.is_dir():
        raise SettingsError("Invalid model directory specification!")

    device = str(model_raw.get("device", "auto"))
    enable_accentizer = bool(model_raw.get("enable_accentizer", True))

    checkpoint = _require_path(model_dir, "model.pth", kind="file")
    config_path = _require_path(model_dir, "config.json", kind="file")
    tokenizer_path = _require_path(model_dir, "vocab.json", kind="file")
    speaker_rel = model_raw.get("ref_file", "ref.wav")
    speaker_wav = _require_path(model_dir, speaker_rel, kind="file")

    model = ModelSettings(
        directory=model_dir,
        config_path=config_path,
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer_path,
        speaker_wav=speaker_wav,
        language=language,
        device=device,
        enable_accentizer=enable_accentizer,
    )

    return Settings(service=service, model=model)

