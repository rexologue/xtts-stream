"""High level package for the XTTS streaming service."""

from importlib import import_module
import sys

_LEGACY_MODULES = [
    "arch_utils",
    "autoregressive",
    "base_tts",
    "generic_utils",
    "gpt",
    "gpt_inference",
    "helpers",
    "hifigan_decoder",
    "hifigan_generator",
    "infer_xtts",
    "perceiver_encoder",
    "resnet",
    "shared_configs",
    "stream_generator",
    "stream_ljs_dataset",
    "tokenizer",
    "transformer",
    "xtts",
    "xtts_config",
    "xtts_manager",
    "xtransformers",
    "zh_num2words",
]

for _name in _LEGACY_MODULES:
    module = import_module(f".core.{_name}", __name__)
    sys.modules.setdefault(_name, module)

__all__ = ["core", "service", "wrappers"]
