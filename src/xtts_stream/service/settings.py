"""Configuration loader for the streaming service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class SettingsError(RuntimeError):
    """Raised when the application configuration is invalid."""


@dataclass
class ServiceSettings:
    """Runtime settings for the FastAPI service."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrency: int = 1


@dataclass
class ResolvedModelSettings:
    """Resolved model artefact locations and runtime configuration."""

    directory: Path
    config_path: Path
    checkpoint_path: Path
    tokenizer_path: Path
    speaker_wav: Path
    device: str
    language: str


@dataclass
class ModelSettings:
    """User provided model configuration as read from YAML."""

    directory: Path
    speaker_wav: Path
    device: str = "auto"
    language: str = "en"
    config: Optional[str] = None
    checkpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    def resolve(self) -> ResolvedModelSettings:
        directory = self.directory
        if not directory.is_dir():
            raise SettingsError("Model directory is not a folder: {0}".format(directory))
        config_path = _resolve_optional(self.config, directory, default="config.json")
        checkpoint_path = _resolve_optional(self.checkpoint, directory, default="model.pth")
        tokenizer_path = _resolve_optional(self.tokenizer, directory, default="vocab.json")

        required: Dict[str, Path] = {
            "config.json": config_path,
            "model.pth": checkpoint_path,
            "dvae.pth": directory / "dvae.pth",
            "mel_stats.pth": directory / "mel_stats.pth",
            "vocab.json": tokenizer_path,
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise SettingsError(
                "Model directory {0} is missing required files: {1}".format(
                    directory, ", ".join(sorted(missing))
                )
            )
        if not self.speaker_wav.is_file():
            raise SettingsError("Speaker WAV file not found: {0}".format(self.speaker_wav))

        return ResolvedModelSettings(
            directory=directory,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            speaker_wav=self.speaker_wav,
            device=self.device,
            language=self.language,
        )


@dataclass
class Settings:
    """Top level application settings."""

    service: ServiceSettings
    model: ResolvedModelSettings


def load_settings(path: Path) -> Settings:
    """Load :class:`Settings` from the given YAML ``path``."""

    if not path.exists():
        raise SettingsError("Configuration file not found: {0}".format(path))

    with path.open("r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise SettingsError("Failed to parse configuration file: {0}".format(path)) from exc

    service_data = _ensure_mapping(raw.get("service", {}), "service")
    model_data = _ensure_mapping(raw.get("model"), "model")

    service = ServiceSettings(
        host=str(service_data.get("host", ServiceSettings.host)),
        port=int(service_data.get("port", ServiceSettings.port)),
        max_concurrency=int(
            service_data.get("max_concurrency", ServiceSettings.max_concurrency)
        ),
    )

    model_settings = _parse_model_settings(model_data, base_dir=path.parent)
    resolved_model = model_settings.resolve()

    return Settings(service=service, model=resolved_model)


def _parse_model_settings(data: Dict[str, Any], *, base_dir: Path) -> ModelSettings:
    directory_str = data.get("directory")
    speaker_str = data.get("speaker_wav")
    if directory_str is None:
        raise SettingsError("`model.directory` must be specified in the configuration file")
    if speaker_str is None:
        raise SettingsError("`model.speaker_wav` must be specified in the configuration file")

    directory = _resolve_path(directory_str, base_dir)
    speaker = _resolve_path(speaker_str, base_dir)

    device = str(data.get("device", "auto"))
    language = str(data.get("language", "en"))

    config_name = data.get("config")
    checkpoint_name = data.get("checkpoint")
    tokenizer_name = data.get("tokenizer")

    return ModelSettings(
        directory=directory,
        speaker_wav=speaker,
        device=device,
        language=language,
        config=config_name,
        checkpoint=checkpoint_name,
        tokenizer=tokenizer_name,
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_optional(value: Optional[str], base_dir: Path, *, default: str) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path
    return (base_dir / default).resolve()


def _ensure_mapping(value: Any, section: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SettingsError("Section `{0}` must be a mapping".format(section))
    return value
