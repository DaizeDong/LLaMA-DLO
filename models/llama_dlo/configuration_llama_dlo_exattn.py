"""LLaMA-DLO-ExAttn model configuration"""
from typing import Union

from transformers import LlamaConfig
from transformers.utils import logging

from .configuration_llama_dlo import LlamaDLOConfig

logger = logging.get_logger(__name__)


class LlamaDLOExAttnConfig(LlamaDLOConfig):
    model_type = "llama_dlo_exattn"

    # 🔍
    def from_llama_config(
            config: LlamaConfig,
            is_dlo: list = None,
            dlo_capacity: Union[float, list] = 0.5,
            dlo_loss_coefficient: float = 1.0,
            dlo_loss_type: str = "self",  # self cos cos-global
            rescale_hidden_states: bool = True,
            scale_factor: float = 2.0,
            scale_gap: float = 0.05,
            gate_init_method: str = "zero",  # zero random
            **kwargs
    ):
        return LlamaDLOExAttnConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            rms_norm_eps=config.rms_norm_eps,
            use_cache=config.use_cache,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pretraining_tp=config.pretraining_tp,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            # 🔍
            is_dlo=is_dlo,
            dlo_capacity=dlo_capacity,
            dlo_loss_coefficient=dlo_loss_coefficient,
            dlo_loss_type=dlo_loss_type,
            rescale_hidden_states=rescale_hidden_states,
            scale_factor=scale_factor,
            scale_gap=scale_gap,
            gate_init_method=gate_init_method,
            **kwargs
        )
