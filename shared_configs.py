import os
import re
import json
import yaml
import fsspec
from typing import Any
from pathlib import Path
from dataclasses import asdict, dataclass, field

from trainer import TrainerConfig
from coqpit import Coqpit, check_argument

from generic_utils import find_module


@dataclass
class BaseAudioConfig(Coqpit):
    """Base config to definge audio processing parameters. It is used to initialize
    ```TTS.utils.audio.AudioProcessor.```

    Args:
        fft_size (int):
            Number of STFT frequency levels aka.size of the linear spectogram frame. Defaults to 1024.

        win_length (int):
            Each frame of audio is windowed by window of length ```win_length``` and then padded with zeros to match
            ```fft_size```. Defaults to 1024.

        hop_length (int):
            Number of audio samples between adjacent STFT columns. Defaults to 1024.

        frame_shift_ms (int):
            Set ```hop_length``` based on milliseconds and sampling rate.

        frame_length_ms (int):
            Set ```win_length``` based on milliseconds and sampling rate.

        stft_pad_mode (str):
            Padding method used in STFT. 'reflect' or 'center'. Defaults to 'reflect'.

        sample_rate (int):
            Audio sampling rate. Defaults to 22050.

        resample (bool):
            Enable / Disable resampling audio to ```sample_rate```. Defaults to ```False```.

        preemphasis (float):
            Preemphasis coefficient. Defaults to 0.0.

        ref_level_db (int): 20
            Reference Db level to rebase the audio signal and ignore the level below. 20Db is assumed the sound of air.
            Defaults to 20.

        do_sound_norm (bool):
            Enable / Disable sound normalization to reconcile the volume differences among samples. Defaults to False.

        log_func (str):
            Numpy log function used for amplitude to DB conversion. Defaults to 'np.log10'.

        do_trim_silence (bool):
            Enable / Disable trimming silences at the beginning and the end of the audio clip. Defaults to ```True```.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        pitch_fmax (float, optional):
            Maximum frequency of the F0 frames. Defaults to ```640```.

        pitch_fmin (float, optional):
            Minimum frequency of the F0 frames. Defaults to ```1```.

        trim_db (int):
            Silence threshold used for silence trimming. Defaults to 45.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        power (float):
            Exponent used for expanding spectrogra levels before running Griffin Lim. It helps to reduce the
            artifacts in the synthesized voice. Defaults to 1.5.

        griffin_lim_iters (int):
            Number of Griffing Lim iterations. Defaults to 60.

        num_mels (int):
            Number of mel-basis frames that defines the frame lengths of each mel-spectrogram frame. Defaults to 80.

        mel_fmin (float): Min frequency level used for the mel-basis filters. ~50 for male and ~95 for female voices.
            It needs to be adjusted for a dataset. Defaults to 0.

        mel_fmax (float):
            Max frequency level used for the mel-basis filters. It needs to be adjusted for a dataset.

        spec_gain (int):
            Gain applied when converting amplitude to DB. Defaults to 20.

        signal_norm (bool):
            enable/disable signal normalization. Defaults to True.

        min_level_db (int):
            minimum db threshold for the computed melspectrograms. Defaults to -100.

        symmetric_norm (bool):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else
            [0, k], Defaults to True.

        max_norm (float):
            ```k``` defining the normalization range. Defaults to 4.0.

        clip_norm (bool):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        stats_path (str):
            Path to the computed stats file. Defaults to None.
    """

    # stft parameters
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    # audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    # silence trimming
    do_trim_silence: bool = True
    trim_db: int = 45
    # rms volume normalization
    do_rms_norm: bool = False
    db_level: float = None
    # griffin-lim params
    power: float = 1.5
    griffin_lim_iters: int = 60
    # mel-spec params
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # f0 params
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)


@dataclass
class BaseDatasetConfig(Coqpit):
    """Base config for TTS datasets.

    Args:
        formatter (str):
            Formatter name that defines used formatter in ```TTS.tts.datasets.formatter```. Defaults to `""`.

        dataset_name (str):
            Unique name for the dataset. Defaults to `""`.

        path (str):
            Root path to the dataset files. Defaults to `""`.

        meta_file_train (str):
            Name of the dataset meta file. Or a list of speakers to be ignored at training for multi-speaker datasets.
            Defaults to `""`.

        ignored_speakers (List):
            List of speakers IDs that are not used at the training. Default None.

        language (str):
            Language code of the dataset. If defined, it overrides `phoneme_language`. Defaults to `""`.

        phonemizer (str):
            Phonemizer used for that dataset's language. By default it uses `DEF_LANG_TO_PHONEMIZER`. Defaults to `""`.

        meta_file_val (str):
            Name of the dataset meta file that defines the instances used at validation.

        meta_file_attn_mask (str):
            Path to the file that lists the attention mask files used with models that require attention masks to
            train the duration predictor.
    """

    formatter: str = ""
    dataset_name: str = ""
    path: str = ""
    meta_file_train: str = ""
    ignored_speakers: list[str] = None
    language: str = ""
    phonemizer: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("formatter", c, restricted=True)
        check_argument("path", c, restricted=True)
        check_argument("meta_file_train", c, restricted=True)
        check_argument("meta_file_val", c, restricted=False)
        check_argument("meta_file_attn_mask", c, restricted=False)


@dataclass
class BaseTrainingConfig(TrainerConfig):
    """Base config to define the basic 🐸TTS training parameters that are shared
    among all the models. It is based on ```Trainer.TrainingConfig```.

    Args:
        model (str):
            Name of the model that is used in the training.

        num_loader_workers (int):
            Number of workers for training time dataloader.

        num_eval_loader_workers (int):
            Number of workers for evaluation time dataloader.
    """

    model: str = None
    # dataloading
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False


def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments but not urls with //
    input_str = re.sub(
        r"(\"(?:[^\"\\]|\\.)*\")|(/\*(?:.|[\\n\\r])*?\*/)|(//.*)", lambda m: m.group(1) or m.group(2) or "", input_str
    )
    return json.loads(input_str)


def register_config(model_name: str) -> type[BaseTrainingConfig]:
    """Find the right config for the given model name.

    Args:
        model_name (str): Model name.

    Raises:
        ModuleNotFoundError: No matching config for the model name.

    Returns:
        type[BaseTrainingConfig]: config class.
    """
    config_class = None
    config_name = model_name + "_config"

    # TODO: fix this
    if model_name == "xtts":
        from xtts_config import XttsConfig

        config_class = XttsConfig
    paths = ["TTS.tts.configs", "TTS.vocoder.configs", "TTS.encoder.configs", "TTS.vc.configs"]
    for path in paths:
        try:
            config_class = find_module(path, config_name)
            if not issubclass(config_class, BaseTrainingConfig):
                msg = f"{config_class} is not a subclass of BaseTrainingConfig."
                raise TypeError(msg)
        except ModuleNotFoundError:
            pass
    if config_class is None:
        raise ModuleNotFoundError(f" [!] Config for {model_name} cannot be found.")
    return config_class


def _process_model_name(config_dict: dict) -> str:
    """Format the model name as expected. It is a band-aid for the old `vocoder` model names.

    Args:
        config_dict (dict): A dictionary including the config fields.

    Returns:
        str: Formatted modelname.
    """
    model_name = config_dict["model"] if "model" in config_dict else config_dict["generator_model"]
    model_name = model_name.replace("_generator", "").replace("_discriminator", "")
    return model_name


def load_config(config_path: str | os.PathLike[Any]) -> BaseTrainingConfig:
    """Import `json` or `yaml` files as TTS configs. First, load the input file as a `dict` and check the model name
    to find the corresponding Config class. Then initialize the Config.

    Args:
        config_path (str): path to the config file.

    Raises:
        TypeError: given config file has an unknown type.

    Returns:
        Coqpit: TTS config object.
    """
    config_path = str(config_path)
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # backwards compat.
            data = read_json_with_comments(config_path)
    else:
        msg = f" [!] Unknown config file type {ext}"
        raise TypeError(msg)
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config


def check_config_and_model_args(config, arg_name, value):
    """Check the give argument in `config.model_args` if exist or in `config` for
    the given value.

    Return False if the argument does not exist in `config.model_args` or `config`.
    This is to patch up the compatibility between models with and without `model_args`.

    TODO: Remove this in the future with a unified approach.
    """
    if getattr(config, "model_args", None) is not None:
        if arg_name in config.model_args:
            return config.model_args[arg_name] == value
    if hasattr(config, arg_name):
        return config[arg_name] == value
    return False


def get_from_config_or_model_args(config, arg_name):
    """Get the given argument from `config.model_args` if exist or in `config`."""
    if getattr(config, "model_args", None) is not None:
        if arg_name in config.model_args:
            return config.model_args[arg_name]
    return config[arg_name]


def get_from_config_or_model_args_with_default(config, arg_name, def_val):
    """Get the given argument from `config.model_args` if exist or in `config`."""
    if getattr(config, "model_args", None) is not None:
        if arg_name in config.model_args:
            return config.model_args[arg_name]
    if hasattr(config, arg_name):
        return config[arg_name]
    return def_val


@dataclass
class GSTConfig(Coqpit):
    """Defines the Global Style Token Module.

    Args:
        gst_style_input_wav (str):
            Path to the wav file used to define the style of the output speech at inference. Defaults to None.

        gst_style_input_weights (dict):
            Defines the weights for each style token used at inference. Defaults to None.

        gst_embedding_dim (int):
            Defines the size of the GST embedding vector dimensions. Defaults to 256.

        gst_num_heads (int):
            Number of attention heads used by the multi-head attention. Defaults to 4.

        gst_num_style_tokens (int):
            Number of style token vectors. Defaults to 10.
    """

    gst_style_input_wav: str = None
    gst_style_input_weights: dict = None
    gst_embedding_dim: int = 256
    gst_use_speaker_embedding: bool = False
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    def check_values(self) -> None:
        """Check config fields."""
        c = asdict(self)
        super().check_values()
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_style_input_wav", c, restricted=False)
        check_argument("gst_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("gst_use_speaker_embedding", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)


@dataclass
class CapacitronVAEConfig(Coqpit):
    """Defines the capacitron VAE Module.

    Args:
        capacitron_capacity (int):
            Defines the variational capacity limit of the prosody embeddings. Defaults to 150.
        capacitron_VAE_embedding_dim (int):
            Defines the size of the Capacitron embedding vector dimension. Defaults to 128.
        capacitron_use_text_summary_embeddings (bool):
            If True, use a text summary embedding in Capacitron. Defaults to True.
        capacitron_text_summary_embedding_dim (int):
            Defines the size of the capacitron text embedding vector dimension. Defaults to 128.
        capacitron_use_speaker_embedding (bool):
            if True use speaker embeddings in Capacitron. Defaults to False.
        capacitron_VAE_loss_alpha (float):
            Weight for the VAE loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        capacitron_grad_clip (float):
            Gradient clipping value for all gradients except beta. Defaults to 5.0
    """

    capacitron_loss_alpha: int = 1
    capacitron_capacity: int = 150
    capacitron_VAE_embedding_dim: int = 128
    capacitron_use_text_summary_embeddings: bool = True
    capacitron_text_summary_embedding_dim: int = 128
    capacitron_use_speaker_embedding: bool = False
    capacitron_VAE_loss_alpha: float = 0.25
    capacitron_grad_clip: float = 5.0

    def check_values(self) -> None:
        """Check config fields."""
        c = asdict(self)
        super().check_values()
        check_argument("capacitron_capacity", c, restricted=True, min_val=10, max_val=500)
        check_argument("capacitron_VAE_embedding_dim", c, restricted=True, min_val=16, max_val=1024)
        check_argument("capacitron_use_speaker_embedding", c, restricted=False)
        check_argument("capacitron_text_summary_embedding_dim", c, restricted=False, min_val=16, max_val=512)
        check_argument("capacitron_VAE_loss_alpha", c, restricted=False)
        check_argument("capacitron_grad_clip", c, restricted=False)


@dataclass
class CharactersConfig(Coqpit):
    """Defines arguments for the `BaseCharacters` or `BaseVocabulary` and their subclasses.

    Args:
        characters_class (str):
            Defines the class of the characters used. If None, we pick ```Phonemes``` or ```Graphemes``` based on
            the configuration. Defaults to None.

        vocab_dict (list[str]):
            Defines the vocabulary dictionary used to encode the characters. Defaults to None.

        pad (str):
            characters in place of empty padding. Defaults to None.

        eos (str):
            characters showing the end of a sentence. Defaults to None.

        bos (str):
            characters showing the beginning of a sentence. Defaults to None.

        blank (str):
            Optional character used between characters by some models for better prosody. Defaults to `_blank`.

        characters (str):
            character set used by the model. Characters not in this list are ignored when converting input text to
            a list of sequence IDs. Defaults to None.

        punctuations (str):
            characters considered as punctuation as parsing the input sentence. Defaults to None.

        phonemes (str):
            characters considered as parsing phonemes. This is only for backwards compat. Use `characters` for new
            models. Defaults to None.

        is_unique (bool):
            remove any duplicate characters in the character lists. It is a bandaid for compatibility with the old
            models trained with character lists with duplicates. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    characters_class: str = None

    # using BaseVocabulary
    vocab_dict: list[str] | None = None

    # using on BaseCharacters
    pad: str = "<PAD>"
    eos: str = None
    bos: str = None
    blank: str = None
    characters: str = None
    punctuations: str = None
    phonemes: str = None
    is_unique: bool = True  # for backwards compatibility of models trained with char sets with duplicates
    is_sorted: bool = True


@dataclass
class BaseTTSConfig(BaseTrainingConfig):
    """Shared parameters among all the TTS models.

    Args:
        audio (BaseAudioConfig):
            Audio processor config object instance.

        model_args:
            Model class arguments.

        _supports_cloning:
            Whether voice cloning is supported. Accessed via `supports_cloning` property.

        use_phonemes (bool):
            enable / disable phoneme use.

        phonemizer (str):
            Name of the phonemizer to use. If set None, the phonemizer will be selected by `phoneme_language`.
            Defaults to None.

        phoneme_language (str):
            Language code for the phonemizer. You can check the list of supported languages by running
            `python TTS/tts/utils/text/phonemizers/__init__.py`. Defaults to None.

        compute_input_seq_cache (bool):
            enable / disable precomputation of the phoneme sequences. At the expense of some delay at the beginning of
            the training, It allows faster data loader time and precise limitation with `max_seq_len` and
            `min_seq_len`.

        text_cleaner (str):
            Name of the text cleaner used for cleaning and formatting transcripts.

        enable_eos_bos_chars (bool):
            enable / disable the use of eos and bos characters.

        test_senteces_file (str):
            Path to a txt file that has sentences used at test time. The file must have a sentence per line.

        phoneme_cache_path (str):
            Path to the output folder caching the computed phonemes for each sample.

        characters (CharactersConfig):
            Instance of a CharactersConfig class.

        batch_group_size (int):
            Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
            length for a more efficient and stable training. If `batch_group_size > 1` then it performs bucketing to
            prevent using the same batches for each epoch.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        min_text_len (int):
            Minimum length of input text to be used. All shorter samples will be ignored. Defaults to 0.

        max_text_len (int):
            Maximum length of input text to be used. All longer samples will be ignored. Defaults to float("inf").

        min_audio_len (int):
            Minimum length of input audio to be used. All shorter samples will be ignored. Defaults to 0.

        max_audio_len (int):
            Maximum length of input audio to be used. All longer samples will be ignored. The maximum length in the
            dataset defines the VRAM used in the training. Hence, pay attention to this value if you encounter an
            OOM error in training. Defaults to float("inf").

        compute_f0 (int):
            (Not in use yet).

        compute_energy (int):
            (Not in use yet).

        compute_linear_spec (bool):
            If True data loader computes and returns linear spectrograms alongside the other data.

        precompute_num_workers (int):
            Number of workers to precompute features. Defaults to 0.

        use_noise_augment (bool):
            Augment the input audio with random noise.

        start_by_longest (bool):
            If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
            Defaults to False.

        shuffle (bool):
            If True, the data loader will shuffle the dataset when there is not sampler defined. Defaults to True.

        drop_last (bool):
            If True, the data loader will drop the last batch if it is not complete. It helps to prevent
            issues that emerge from the partial batch statistics. Defaults to True.

        add_blank (bool):
            Add blank characters between each other two characters. It improves performance for some models at expense
            of slower run-time due to the longer input sequence.

        datasets (List[BaseDatasetConfig]):
            List of datasets used for training. If multiple datasets are provided, they are merged and used together
            for training.

        optimizer (str):
            Optimizer used for the training. Set one from `torch.optim.Optimizer` or `TTS.utils.training`.
            Defaults to ``.

        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`

        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers or
            `TTS.utils.training`. Defaults to ``.

        lr_scheduler_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"warmup": 4000}`.

        test_sentences (List[str]):
            List of sentences to be used at testing. Defaults to '[]'

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).

        use_speaker_weighted_sampler (bool):
            Enable / Disable the batch balancer by speaker. Defaults to ```False```.

        speaker_weighted_sampler_alpha (float):
            Number that control the influence of the speaker sampler weights. Defaults to ```1.0```.

        use_language_weighted_sampler (bool):
            Enable / Disable the batch balancer by language. Defaults to ```False```.

        language_weighted_sampler_alpha (float):
            Number that control the influence of the language sampler weights. Defaults to ```1.0```.

        use_length_weighted_sampler (bool):
            Enable / Disable the batch balancer by audio length. If enabled the
            dataset will be divided into 10 buckets considering the min and max
            audio of the dataset. The sampler weights will be computed forcing
            to have the same quantity of data for each bucket in each training
            batch. Defaults to ```False```.

        length_weighted_sampler_alpha (float):
            Number that control the influence of the length sampler weights. Defaults to ```1.0```.
    """

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    model_args: Coqpit | None = None
    _supports_cloning: bool = False
    # phoneme settings
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = None
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    # vocabulary parameters
    characters: CharactersConfig = None
    add_blank: bool = False
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_energy: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    # dataset
    datasets: list[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    # optimizer
    optimizer: str = "radam"
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=dict)
    # testing
    test_sentences: list[str] | list[list[str]] = field(default_factory=list)
    # evaluation
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    # weighted samplers
    use_speaker_weighted_sampler: bool = False
    speaker_weighted_sampler_alpha: float = 1.0
    use_language_weighted_sampler: bool = False
    language_weighted_sampler_alpha: float = 1.0
    use_length_weighted_sampler: bool = False
    length_weighted_sampler_alpha: float = 1.0

    @property
    def supports_cloning(self) -> bool:
        return self._supports_cloning or (
            Path(get_from_config_or_model_args_with_default(self, "speaker_encoder_model_path", "")).is_file()
            and Path(get_from_config_or_model_args_with_default(self, "speaker_encoder_config_path", "")).is_file()
        )
