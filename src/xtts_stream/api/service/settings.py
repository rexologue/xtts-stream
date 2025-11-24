"""Configuration loader for the streaming service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import yaml
import torch


class SettingsError(RuntimeError):
    """Raised when the application configuration is invalid."""


# ================================
# SERVICE
# ================================
@dataclass
class ServiceSettings:
    """Runtime settings for the FastAPI service."""

    host: str = "0.0.0.0"
    port: int = 8000
    instances: int = 1
    log_path: Optional[Path] = None


# ================================
# MODEL
# ================================
@dataclass
class ModelSettings:
    """Resolved model artefact locations and runtime configuration."""

    directory: Path
    config_path: Path
    checkpoint_path: Path
    tokenizer_path: Path
    reference_wav: Path
    language: str


# ================================
# EXTRA OPTIONS
# ================================
@dataclass
class ExtraSettings:
    """Optional extra features for the runtime service."""

    enable_accentizer: bool = False
    enable_asr_cutting: bool = False


# ================================
# ALL SETTINGS
# ================================
@dataclass
class Settings:
    """Top level application settings."""

    service: ServiceSettings
    model: ModelSettings
    extra: ExtraSettings


# ================================
# LOADERS
# ================================
def load_settings(path: Path) -> Settings:
    """Load settings from the given YAML ``path``."""

    if not torch.cuda.is_available():
        raise SettingsError("CUDA is not available on this machine! Service working is not possible.")

    if not path.exists():
        raise SettingsError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise SettingsError(f"Failed to parse configuration file: {path}") from exc

    try:
        service_data = raw["service"]
    except KeyError:
        raise SettingsError(f"Failde to extract `service` configuration block!")
    try:
        model_data = raw["model"]
    except KeyError:
        raise SettingsError(f"Failde to extract `model` configuration block!")
    try:
        extra_data = raw["extra"]
    except KeyError:
        raise SettingsError(f"Failde to extract `extra` configuration block!")

    # --- service ---
    try:
        host = service_data["host"]
        port = service_data["port"]
        instances = service_data["instances"]
        log_path = service_data["instances"]

    except KeyError as e:
        missing_key = e.args[0]
        raise SettingsError(f"Service configuration is incomplete! Missing {missing_key}")
    
    if port < 1 or port > 65535 or not isinstance(port, int):
        raise SettingsError("Invalid service port specification!")
    if instances < 1 or not isinstance(instances, int):
        raise SettingsError("Invalid service instances specification!")
    if not isinstance(log_path, str):
        raise SettingsError("Invalid service log path specification!")
    
    log_path = Path(log_path).expanduser().resolve()

    if not log_path.is_dir():
        raise SettingsError("Invalid service log path specification!")

    log_path.mkdir(exist_ok=True, parents=True)
    
    service = ServiceSettings(
        host,
        port,
        instances,
        log_path
    )

    # --- model ---
    try:
        directory = model_data["directory"]
        ref_file = model_data["ref_file"]
        language = model_data["language"]

    except KeyError as e:
        missing_key = e.args[0]
        raise SettingsError(f"Model configuration is incomplete! Missing {missing_key}")
    
    directory = Path(directory).expanduser().resolve()
    ref_file = Path(ref_file).expanduser().resolve()

    if not directory.exists() or not directory.is_dir():
        raise SettingsError("Invalid model directory specification!")
    
    if not ref_file.exists() or ref_file.is_dir():
        raise SettingsError("Invalid reference file specification!")
    
    ckp = directory / "model.pth"
    dvae = directory / "dvae.pth"
    stats = directory / "mel_stats.pth"
    vocab = directory / "vocab.json"
    config = directory / "config.json"

    if not ckp.exists() or not dvae.exists() or not stats.exists() or not vocab.exists() or not config.exists():
         raise SettingsError("Invalid model directory structure!")
    
    model = ModelSettings(
        directory,
        config,
        ckp,
        vocab,
        ref_file,
        language
    )

    # --- extra ---
    extra = ExtraSettings(
        enable_accentizer=bool(extra_data.get("enable_accentizer", False)),
        enable_asr_cutting=bool(extra_data.get("enable_asr_cutting", False)),
    )

    return Settings(service=service, model=model, extra=extra)



