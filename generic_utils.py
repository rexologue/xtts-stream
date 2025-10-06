import os
import re
import sys
import glob
import fsspec
import random
import logging
import datetime
import warnings
import importlib
import unicodedata
from pathlib import Path
from collections.abc import Callable
from typing import Any, TextIO, TypeVar

import torch
from typing_extensions import TypeIs
from packaging.version import Version

import numpy as np
from scipy import signal

from base_encoder import BaseEncoder
from resnet import ResNetSpeakerEncoder

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def exists(val: _T | None) -> TypeIs[_T]:
    return val is not None


def default(val: _T | None, d: _T | Callable[[], _T]) -> _T:
    if exists(val):
        return val
    return d() if callable(d) else d # type: ignore


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    text = text.replace("vc", "VC")
    text = text.replace("Knn", "KNN")
    return text


def slugify(text: str) -> str:
    """Convert a string (e.g. speaker IDs) into a safe filename base."""
    # Normalize to ASCII (e.g., Zoë -> Zoe)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")

    # Replace unsafe characters with underscores
    safe = re.sub(r"[^\w\-]", "_", ascii_str)

    # Collapse repeated underscores
    return re.sub(r"_+", "_", safe).strip("_")


def find_module(module_path: str, module_name: str) -> type[Any]:
    module_name = module_name.lower()
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def import_class(module_path: str) -> type[Any]:
    """Import a class from a module path.

    Args:
        module_path (str): The module path of the class.

    Returns:
        object: The imported class.
    """
    class_name = module_path.split(".")[-1]
    module_path = ".".join(module_path.split(".")[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_import_path(obj: object) -> str:
    """Get the import path of a class.

    Args:
        obj (object): The class object.

    Returns:
        str: The import path of the class.
    """
    return ".".join([type(obj).__module__, type(obj).__name__])


def format_aux_input(def_args: dict, kwargs: dict) -> dict:
    """Format kwargs to hande auxilary inputs to models.

    Args:
        def_args (Dict): A dictionary of argument names and their default values if not defined in `kwargs`.
        kwargs (Dict): A `dict` or `kwargs` that includes auxilary inputs to the model.

    Returns:
        Dict: arguments with formatted auxilary inputs.
    """
    kwargs = kwargs.copy()
    for name, arg in def_args.items():
        if name not in kwargs or kwargs[name] is None:
            kwargs[name] = arg
    return kwargs


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_user_data_dir(appname: str) -> Path:
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)

def load_fsspec(
    path: str | os.PathLike[Any],
    map_location: str | torch.device | dict[str, str] | None = None,
    *,
    cache: bool = True,
    **kwargs: Any,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/trainer_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = Path(path).exists()
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
    else:
        with fsspec.open(str(path), "rb") as f:
            return torch.load(f, map_location=map_location, weights_only=False, **kwargs)


class ConsoleFormatter(logging.Formatter):
    """Custom formatter that prints logging.INFO messages without the level name.

    Source: https://stackoverflow.com/a/62488520
    """

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


def setup_logger(
    logger_name: str,
    level: int = logging.INFO,
    *,
    formatter: logging.Formatter | None = None,
    stream: TextIO | None = None,
    log_dir: str | os.PathLike[Any] | None = None,
    log_name: str = "log",
) -> None:
    """Set up a logger.

    Args:
        logger_name: Name of the logger to set up
        level: Logging level
        formatter: Formatter for the logger
        stream: Add a StreamHandler for the given stream, e.g. sys.stderr or sys.stdout
        log_dir: Folder to write the log file (no file created if None)
        log_name: Prefix of the log file name
    """
    lg = logging.getLogger(logger_name)
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(levelname)-8s - %(name)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
    lg.setLevel(level)
    if log_dir is not None:
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        log_file = Path(log_dir) / f"{log_name}_{get_timestamp()}.log"
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if stream is not None:
        sh = logging.StreamHandler(stream)
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def is_pytorch_at_least_2_4() -> bool:
    """Check if the installed Pytorch version is 2.4 or higher."""
    return Version(torch.__version__) >= Version("2.4")


def optional_to_str(x: Any | None) -> str:
    """Convert input to string, using empty string if input is None."""
    return "" if x is None else str(x)


def warn_synthesize_config_deprecated() -> None:
    warnings.warn(
        "The `config` argument of synthesize() is deprecated and will be removed soon. You can safely leave it out.",
        DeprecationWarning,
    )


def warn_synthesize_speaker_id_deprecated() -> None:
    warnings.warn(
        "The `speaker_id` argument of synthesize() is deprecated and will be removed soon. Use `speaker` instead.",
        DeprecationWarning,
    )


class AugmentWAV:
    def __init__(self, ap, augmentation_config):
        self.ap = ap
        self.use_additive_noise = False

        if "additive" in augmentation_config.keys():
            self.additive_noise_config = augmentation_config["additive"]
            additive_path = self.additive_noise_config["sounds_path"]
            if additive_path:
                self.use_additive_noise = True
                # get noise types
                self.additive_noise_types = []
                for key in self.additive_noise_config.keys():
                    if isinstance(self.additive_noise_config[key], dict):
                        self.additive_noise_types.append(key)

                additive_files = glob.glob(os.path.join(additive_path, "**/*.wav"), recursive=True)

                self.noise_list = {}

                for wav_file in additive_files:
                    noise_dir = wav_file.replace(additive_path, "").split(os.sep)[0]
                    # ignore not listed directories
                    if noise_dir not in self.additive_noise_types:
                        continue
                    if noise_dir not in self.noise_list:
                        self.noise_list[noise_dir] = []
                    self.noise_list[noise_dir].append(wav_file)

                logger.info(
                    "Using Additive Noise Augmentation: with %d audios instances from %s",
                    len(additive_files),
                    self.additive_noise_types,
                )

        self.use_rir = False

        if "rir" in augmentation_config.keys():
            self.rir_config = augmentation_config["rir"]
            if self.rir_config["rir_path"]:
                self.rir_files = glob.glob(os.path.join(self.rir_config["rir_path"], "**/*.wav"), recursive=True)
                self.use_rir = True

            logger.info("Using RIR Noise Augmentation: with %d audios instances", len(self.rir_files))

        self.create_augmentation_global_list()

    def create_augmentation_global_list(self):
        if self.use_additive_noise:
            self.global_noise_list = self.additive_noise_types
        else:
            self.global_noise_list = []
        if self.use_rir:
            self.global_noise_list.append("RIR_AUG")

    def additive_noise(self, noise_type, audio):
        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

        noise_list = random.sample(
            self.noise_list[noise_type],
            random.randint(
                self.additive_noise_config[noise_type]["min_num_noises"],
                self.additive_noise_config[noise_type]["max_num_noises"],
            ),
        )

        audio_len = audio.shape[0]
        noises_wav = None
        for noise in noise_list:
            noiseaudio = self.ap.load_wav(noise, sr=self.ap.sample_rate)[:audio_len]

            if noiseaudio.shape[0] < audio_len:
                continue

            noise_snr = random.uniform(
                self.additive_noise_config[noise_type]["min_snr_in_db"],
                self.additive_noise_config[noise_type]["max_num_noises"],
            )
            noise_db = 10 * np.log10(np.mean(noiseaudio**2) + 1e-4)
            noise_wav = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio

            if noises_wav is None:
                noises_wav = noise_wav
            else:
                noises_wav += noise_wav

        # if all possible files is less than audio, choose other files
        if noises_wav is None:
            return self.additive_noise(noise_type, audio)

        return audio + noises_wav

    def reverberate(self, audio):
        audio_len = audio.shape[0]

        rir_file = random.choice(self.rir_files)
        rir = self.ap.load_wav(rir_file, sr=self.ap.sample_rate)
        rir = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode=self.rir_config["conv_mode"])[:audio_len]

    def apply_one(self, audio):
        noise_type = random.choice(self.global_noise_list)
        if noise_type == "RIR_AUG":
            return self.reverberate(audio)

        return self.additive_noise(noise_type, audio)


def setup_encoder_model(config: "Coqpit") -> BaseEncoder:
    if config.model_params["model_name"].lower() == "resnet":
        model = ResNetSpeakerEncoder(
            input_dim=config.model_params["input_dim"],
            proj_dim=config.model_params["proj_dim"],
            log_input=config.model_params.get("log_input", False),
            use_torch_spec=config.model_params.get("use_torch_spec", False),
            audio_config=config.audio,
        )
    else:
        msg = f"Model not supported: {config.model_params['model_name']}"
        raise ValueError(msg)
    
    return model # type: ignore