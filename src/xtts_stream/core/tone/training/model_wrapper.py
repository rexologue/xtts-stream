"""Module for wrapping the T-one model for fine-tuning."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from tone.nn.model import Tone


class ToneConfig(PretrainedConfig):
    """A `PreTrainedModel` configuration for the T-one model for fine-tuning."""

    def __init__(
        self,
        feature_extraction_params: dict | None = None,
        encoder_params: dict | None = None,
        decoder_params: dict | None = None,
        pad_token_id: int = 34,
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if feature_extraction_params is None:
            feature_extraction_params = {
                "sample_rate": 8000,
                "window_size": 0.02,
                "window_stride": 0.01,
                "n_fft": 160,
                "n_mels": 64,
                "preemphasis_coefficient": 0.97,
            }
        if encoder_params is None:
            encoder_params = {
                "feat_in": 64,
                "n_layers": 16,
                "subsampling_conv_channels": [32, 64],
                "subsampling_kernel_size": [[11, 21], [11, 11]],
                "subsampling_strides": [[1, 1], [3, 1]],
                "ff_expansion_factor": 4,
                "n_heads": 8,
                "conv_kernel_size": 31,
                "dropout": 0.1,
                "dropout_att": 0.1,
                "mhsa_stateless_layers": 14,
                "rope_dim": 32,
                "should_recompute_att_scores": [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                ],
                "mhsa_state_size": 30,
                "chunk_size": 10,
                "d_model": 384,
                "reduction_factor": 2,
                "reduction_kernel_size": 3,
                "reduction_position": 6,
                "upsample_position": 14,
            }
        if decoder_params is None:
            decoder_params = {
                "feat_in": 384,
                "vocabulary": [
                    "а",
                    "б",
                    "в",
                    "г",
                    "д",
                    "е",
                    "ё",
                    "ж",
                    "з",
                    "и",
                    "й",
                    "к",
                    "л",
                    "м",
                    "н",
                    "о",
                    "п",
                    "р",
                    "с",
                    "т",
                    "у",
                    "ф",
                    "х",
                    "ц",
                    "ч",
                    "ш",
                    "щ",
                    "ъ",
                    "ы",
                    "ь",
                    "э",
                    "ю",
                    "я",
                    " ",
                ],
            }
        self.feature_extraction_params = feature_extraction_params
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.pad_token_id = pad_token_id
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

    @property
    def vocab_size(self) -> int:
        """The size of vocabulary from `decoder_params`.

        Returns:
            (int): The token count from decoder vocabulary.

        """
        return len(self.decoder_params["vocabulary"])


class ToneForCTC(PreTrainedModel):
    """A `PreTrainedModel` wrapper for the T-one model for fine-tuning.

    This class adapts the base T-one model to the Hugging Face `transformers`
    API. It includes the necessary `forward` method that computes the
    Connectionist Temporal Classification (CTC) loss, making it compatible
    with the `Trainer` API for fine-tuning.

    The model's configuration is defined by `ToneConfig`.
    """

    config_class = ToneConfig
    base_model_prefix = "tone"
    main_input_name = "input_values"
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_flex_attn = False

    def __init__(self, config: ToneConfig) -> None:
        super().__init__(config)
        self.config = config
        self.tone = Tone(
            feature_extraction_params=config.feature_extraction_params,
            encoder_params=config.encoder_params,
            decoder_params=config.decoder_params,
        )
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the decoder. Please "
                "instantiate the model as follows: with the parameter 'vocabulary' of decoder_params set",
            )

    def forward(
        self,
        input_values: torch.Tensor | None,
        input_lengths: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple | CausalLMOutput:
        """Performs a forward pass and computes the CTC loss if labels are provided.

        Args:
            input_values (torch.Tensor): A batch of audio signals.
            input_lengths (torch.Tensor): The length of each audio signal in the batch.
            attention_mask (torch.Tensor, optional): The attention mask.
                Note: This is not used by the model but included for API compatibility.
            labels (torch.Tensor, optional): A batch of labels for computing the
                CTC loss. Padded values should be -100.
                Note: All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.

        Returns:
            (CausalLMOutput): An output object containing the loss (if labels are
            provided) and the model's output logits.

        """
        assert attention_mask is None, "Training with attention mask not supported for this model"
        log_probs, log_probs_len = self.tone(input_values, input_lengths)
        loss = None
        if labels is not None:
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            with torch.backends.cudnn.flags(enabled=False):
                loss = torch.nn.functional.ctc_loss(
                    log_probs.transpose(0, 1),
                    flattened_targets,
                    log_probs_len,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return CausalLMOutput(loss=loss, logits=log_probs)
