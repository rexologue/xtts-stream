"""Configuration loader for the Broker service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import yaml


BROKER_CONFIG_ENV_VAR = "XTTS_BROKER_CONFIG_FILE"
SUPPORTED_STRATEGIES = {"deep", "wide", "random"}


class BrokerSettingsError(RuntimeError):
    """Raised when the broker configuration is invalid."""


@dataclass
class BrokerServiceSettings:
    host: str
    port: int
    strategy: str = "deep"
    balancer_timeout_seconds: float = 2.0


@dataclass
class BrokerSettings:
    service: BrokerServiceSettings


def _config_path() -> Path:
    if BROKER_CONFIG_ENV_VAR not in os.environ:
        raise BrokerSettingsError(
            "Environment variable XTTS_BROKER_CONFIG_FILE must point to the broker configuration file."
        )
    return Path(os.environ[BROKER_CONFIG_ENV_VAR]).expanduser().resolve(strict=False)


def load_broker_settings(path: Path) -> BrokerSettings:
    if not path.exists():
        raise BrokerSettingsError(f"Configuration file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - formatting errors only happen at runtime
        raise BrokerSettingsError(f"Failed to parse configuration file: {path}") from exc

    try:
        service_raw = raw["service"]
    except KeyError as exc:  # pragma: no cover - validated at runtime
        raise BrokerSettingsError(f"Missing configuration block: {exc.args[0]}")

    try:
        strategy = str(service_raw.get("strategy", "deep"))
        service = BrokerServiceSettings(
            host=str(service_raw["host"]),
            port=int(service_raw["port"]),
            strategy=strategy,
            balancer_timeout_seconds=float(service_raw.get("balancer_timeout_seconds", 2.0)),
        )
    except KeyError as exc:  # pragma: no cover - validated at runtime
        raise BrokerSettingsError(f"Service configuration is incomplete! Missing {exc.args[0]}")

    if not (1 <= service.port <= 65535):
        raise BrokerSettingsError("Invalid service port specification!")
    if service.strategy not in SUPPORTED_STRATEGIES:
        raise BrokerSettingsError(
            f"Unsupported strategy '{service.strategy}'. Supported values: {sorted(SUPPORTED_STRATEGIES)}"
        )
    if service.balancer_timeout_seconds <= 0:
        raise BrokerSettingsError("balancer_timeout_seconds must be positive")

    return BrokerSettings(service=service)


def config_path_from_env() -> Path:
    return _config_path()

