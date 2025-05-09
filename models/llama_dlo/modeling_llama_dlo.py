"""PyTorch LLaMA-DLO model."""
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaPreTrainedModel, LLAMA_ATTENTION_CLASSES, LlamaMLP
from transformers.utils import logging, ModelOutput

from .configuration_llama_dlo import LlamaDLOConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaDLOConfig"


@dataclass
class BaseDLOModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    dlo_losses: Optional[torch.FloatTensor] = None  # 🔍


class LlamaDLODecoderLayer(nn.Module):
    def __init__(self, config: LlamaDLOConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rescale_hidden_states = config.rescale_hidden_states  # 🔍
        self.scale_factor = config.scale_factor  # 🔍 scale the central value of sigmoid score
        self.scale_gap = config.scale_gap  # 🔍 scale the range between the maximum & minimum values of sigmoid scores
        # Final scores: 0.5 * scale_factor + (sigmoid(x) - 0.5) * scale_gap

    def forward(
            self,
            hidden_states: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            topk_mask: Optional[torch.BoolTensor] = None,  # 🔍
            topk_scores: Optional[torch.Tensor] = None,  # 🔍
            calculate_similarity: Optional[bool] = False,  # 🔍
            llama_dlo_model: Optional[LlamaPreTrainedModel] = None,  # 🔍
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual_input = hidden_states

        if topk_mask is None or topk_scores is None:  # Normal
            # Self Attention
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = residual_input + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if calculate_similarity:
                with torch.no_grad():
                    similarities = F.cosine_similarity(residual_input, hidden_states, dim=-1)  # (batch_size, seq_len)
            else:
                similarities = None

        else:  # DLO
            batch_size, seq_len, hidden_size = hidden_states.shape

            # number of topk tokens in each sample & the whole batch
            topk_seq_lens = topk_mask.sum(1).long()  # (batch_size)
            max_topk_seq_len = topk_seq_lens.max().item()

            # print("topk_seq_lens", topk_seq_lens.shape, topk_seq_lens.dtype, topk_seq_lens)
            # print("max_topk_seq_len", max_topk_seq_len)

            if max_topk_seq_len > 0:
                """🔍 Get topk attention mask"""
                # create "attention_mask" for topk tokens
                topk_attention_mask = torch.arange(max_topk_seq_len, device=topk_seq_lens.device)[None, :] < topk_seq_lens[:, None]

                # print("topk_attention_mask", topk_attention_mask.shape, topk_attention_mask.dtype, topk_attention_mask)

                """🔍 Prepare Indices (for acceleration)"""
                # pre-calculate the indices from mask is faster compared to tensor indexing with raw mask
                topk_indices = torch.nonzero(topk_mask).split(1, dim=1)  # 2 * (topk_num_tokens)
                bottom_indices = torch.nonzero(~topk_mask).split(1, dim=1)  # 2 * (bottom_num_tokens)
                forward_topk_indices = torch.nonzero(topk_attention_mask).split(1, dim=1)  # 2 * (topk_num_tokens)

                # print("topk_indices", topk_indices)
                # print("bottom_indices", bottom_indices)
                # print("forward_topk_indices", forward_topk_indices)

                """🔍 Separate Hidden States"""
                # add bottom hidden states
                bottom_hidden_states = torch.zeros((batch_size, seq_len, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
                bottom_hidden_states[bottom_indices[0], bottom_indices[1]] = hidden_states[bottom_indices[0], bottom_indices[1]]

                # select forward hidden states
                temp_hidden_states = torch.zeros((batch_size, max_topk_seq_len, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
                temp_hidden_states[forward_topk_indices[0], forward_topk_indices[1]] = hidden_states[topk_indices[0], topk_indices[1]]  # (batch_size, max_topk_seq_len, hidden_size)
                hidden_states = temp_hidden_states

                # print("bottom_hidden_states", bottom_hidden_states.shape, bottom_hidden_states.dtype, bottom_hidden_states)
                # print("hidden_states", hidden_states.shape, hidden_states.dtype, hidden_states)

                """🔍 Prepare Attention Inputs"""
                # TODO: add cache support
                assert use_cache is False  # 🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍

                # cache_position & causal_mask
                topk_cache_position = torch.arange(0, max_topk_seq_len, device=hidden_states.device)  # TODO: add cache support
                topk_causal_mask = llama_dlo_model._update_causal_mask(topk_attention_mask, hidden_states, topk_cache_position, max_topk_seq_len)

                # consider the original position_ids
                topk_position_ids = torch.zeros((batch_size, max_topk_seq_len), dtype=position_ids.dtype, device=position_ids.device)
                topk_position_ids[forward_topk_indices[0], forward_topk_indices[1]] = position_ids.expand(batch_size, seq_len)[topk_indices[0], topk_indices[1]]  # (batch_size, max_topk_seq_len)

                # print("topk_cache_position", topk_cache_position.shape, topk_cache_position.dtype, topk_cache_position)
                # print("topk_causal_mask", topk_causal_mask.shape, topk_causal_mask.dtype, topk_causal_mask)
                # print("topk_position_ids", topk_position_ids.shape, topk_position_ids.dtype, topk_position_ids)

                """forward"""
                # Self Attention
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=topk_causal_mask,
                    position_ids=topk_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=topk_cache_position,
                    **kwargs,
                )
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)

                """🔍 Aggregate Hidden States"""
                topk_hidden_states = torch.zeros((batch_size, seq_len, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)

                if self.rescale_hidden_states:
                    topk_scores = 0.5 * self.scale_factor + (topk_scores - 0.5) * self.scale_gap  # scale the scores
                    topk_hidden_states[topk_indices[0], topk_indices[1]] = residual[forward_topk_indices[0], forward_topk_indices[1]] + hidden_states[forward_topk_indices[0], forward_topk_indices[1]] * topk_scores[:, None, None]
                else:
                    topk_hidden_states[topk_indices[0], topk_indices[1]] = residual[forward_topk_indices[0], forward_topk_indices[1]] + hidden_states[forward_topk_indices[0], forward_topk_indices[1]]

                hidden_states = bottom_hidden_states + topk_hidden_states

            if calculate_similarity:
                with torch.no_grad():
                    transformed_hidden_states = self.input_layernorm(residual_input)
                    transformed_hidden_states, self_attn_weights, present_key_value = self.self_attn(
                        hidden_states=transformed_hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )
                    transformed_hidden_states = residual_input + transformed_hidden_states
                    residual = transformed_hidden_states
                    transformed_hidden_states = self.post_attention_layernorm(transformed_hidden_states)
                    transformed_hidden_states = self.mlp(transformed_hidden_states)
                    transformed_hidden_states = residual + transformed_hidden_states
                    similarities = F.cosine_similarity(residual_input.float(), transformed_hidden_states.float(), dim=-1)  # (batch_size, seq_len)
            else:
                similarities = None

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (similarities,)

        return outputs


class LlamaDLOPreTrainedModel(LlamaPreTrainedModel):
    config_class = LlamaDLOConfig
    _no_split_modules = ["LlamaDLODecoderLayer"]


class LlamaDLOModel(LlamaDLOPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDLODecoderLayer`]

    Args:
        config: LlamaDLOConfig
    """

    def __init__(self, config: LlamaDLOConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                LlamaDLODecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # 🔍
        self.is_dlo = config.is_dlo
        self.routers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, 1, bias=False) if config.is_dlo[layer_idx] else None  # 🔍
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.dlo_capacity = config.dlo_capacity
        self.dlo_loss_type = config.dlo_loss_type
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

        if config.dlo_loss_type == "cos-global":
            self.dlo_global_avg_capacity = (
                sum([capacity for capacity in config.dlo_capacity if capacity is not None]) / sum(config.is_dlo)
                if isinstance(config.dlo_capacity, list) else
                config.dlo_capacity
            )  # this is for global TopK to calculate the labels of gates

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        # 🔍 zeros init for routers
        if hasattr(self.config, "gate_init_method") and self.config.gate_init_method == "zero":
            for layer_idx in range(self.config.num_hidden_layers):
                if self.config.is_dlo[layer_idx]:
                    self.routers[layer_idx].weight.data.zero_()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_dlo_capacity(self, dlo_capacity):  # 🔍
        self.config.dlo_capacity = dlo_capacity
        self.dlo_capacity = dlo_capacity

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseDLOModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens + inputs_embeds.shape[1])

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        dlo_losses = torch.tensor(0., device=hidden_states.device, dtype=torch.float32)  # 🔍

        if self.training and self.dlo_loss_type == "cos-global":  # 🔍
            sigmoid_logits_cache = []
            similarities_cache = []

        for layer_index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.is_dlo[layer_index]:  # 🔍
                """🔍 Get Indices / Masks for DLO Routing"""
                batch_size, seq_len, hidden_size = hidden_states.shape
                logits = self.routers[layer_index](hidden_states).squeeze(-1)  # (batch_size, seq_len)
                sigmoid_logits = self.sigmoid(logits)  # (batch_size, seq_len)

                # print(batch_size, seq_len, hidden_size)
                # print("logits", logits.shape, logits.dtype, logits)
                # print("sigmoid_logits", sigmoid_logits.shape, sigmoid_logits.dtype, sigmoid_logits)

                if self.training:
                    """[Pretraining | Finetuning] Use the batch-wise TopK to select tokens."""
                    # TODO: The implementation here is problematic as the true topk number should be different for each sample considering the padding tokens.
                    # TODO: However, it is inefficient to use the above implementation as PyTorch doesn't support it well.
                    # TODO: So here I select the topk tokens from all non-padding tokens in a batch-level instead of sample-level.
                    # TODO: Also the scores for padding positions are dropped for correct sorting.

                    # get num_tokens_selected
                    this_layer_dlo_capacity = min(1.0, self.dlo_capacity[layer_index] if isinstance(self.dlo_capacity, list) else self.dlo_capacity)
                    if attention_mask is None:
                        num_tokens_selected = math.ceil(batch_size * seq_len * this_layer_dlo_capacity)
                    else:
                        # for padding tokens, they should be classified into the "drop" class
                        attention_mask = attention_mask.bool()
                        non_padding_token_num = attention_mask.sum().item()
                        num_tokens_selected = math.ceil(non_padding_token_num * this_layer_dlo_capacity)

                    # get topk threshold & topk mask & routing scores
                    if self.training and num_tokens_selected == 0:
                        num_tokens_selected = 1  # for compatibility of gradient flow

                    if num_tokens_selected > 0:
                        if attention_mask is None:
                            topk_sigmoid_logits, _ = sigmoid_logits.flatten().topk(num_tokens_selected, dim=0)
                            threshold = topk_sigmoid_logits[-1]
                            topk_mask = (sigmoid_logits >= threshold)  # (batch_size, seq_len)
                            topk_scores = sigmoid_logits[topk_mask]  # (topk_num)
                        else:
                            # for padding tokens, they should be classified into the "drop" class
                            non_padding_sigmoid_logits = sigmoid_logits[attention_mask]  # (non_padding_token_num)
                            topk_non_padding_sigmoid_logits, _ = non_padding_sigmoid_logits.topk(num_tokens_selected, dim=0)
                            threshold = topk_non_padding_sigmoid_logits[-1]
                            topk_mask = attention_mask & (sigmoid_logits >= threshold)  # (batch_size, seq_len)
                            topk_scores = sigmoid_logits[topk_mask]  # (topk_num)

                        # print(f"layer {layer_index}, num_tokens_selected = {num_tokens_selected}, dlo_capacity = {this_layer_dlo_capacity}, threshold = {threshold}")

                    else:
                        topk_mask = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.bool)
                        topk_scores = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=sigmoid_logits.dtype)

                else:
                    """[Evaluation] Use the binary-classification to select tokens."""
                    # use router logits to select the tokens
                    topk_mask = (sigmoid_logits > 0.5)  # (batch_size, seq_len)
                    topk_scores = sigmoid_logits[topk_mask]  # (topk_num)

                # check if calculate the similarities of hidden features
                if self.training and self.dlo_loss_type in ("cos", "cos-global"):
                    calculate_similarity = True
                else:
                    calculate_similarity = False

                # print("topk_mask", topk_mask.shape, topk_mask.dtype, topk_mask)
                # print("topk_scores", topk_scores.shape, topk_scores.dtype, topk_scores)
                # print("calculate_similarity", calculate_similarity)

                """Normal Forward Process"""
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,  # 🔍
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        topk_mask,  # 🔍
                        topk_scores,  # 🔍
                        calculate_similarity,  # 🔍
                        self,  # 🔍
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        causal_mask=causal_mask,  # 🔍
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        topk_mask=topk_mask,  # 🔍
                        topk_scores=topk_scores,  # 🔍
                        calculate_similarity=calculate_similarity,  # 🔍
                        llama_dlo_model=self,  # 🔍
                    )

                hidden_states = layer_outputs[0]
                similarities = layer_outputs[-1]

                """🔍 DLO Loss"""
                dlo_loss = None

                if self.training:
                    if self.dlo_loss_type == "self":
                        labels = topk_mask.clone().to(sigmoid_logits.dtype)
                        dlo_loss = self.bce_loss(sigmoid_logits, labels)
                    elif self.dlo_loss_type == "cos":
                        with torch.no_grad():
                            if attention_mask is None:
                                topk_similarities, _ = similarities.flatten().topk(num_tokens_selected, dim=0, largest=False)
                                similarity_threshold = topk_similarities[-1]
                                labels = (similarities <= similarity_threshold).to(sigmoid_logits.dtype)  # (batch_size, seq_len)
                            else:
                                # for padding tokens, they should be classified into the "drop" class
                                non_padding_similarities = similarities[attention_mask]  # (non_padding_token_num)
                                topk_similarities, _ = non_padding_similarities.topk(num_tokens_selected, dim=-1, largest=False)
                                similarity_threshold = topk_similarities[-1]
                                labels = (attention_mask & (similarities <= similarity_threshold)).to(sigmoid_logits.dtype)  # (batch_size, seq_len)
                        dlo_loss = self.bce_loss(sigmoid_logits, labels)
                    elif self.dlo_loss_type == "cos-global":
                        # Here we leave the calculation of dlo_loss out of layer iteration, as we need to gather the global similarities.
                        # The dlo_loss here is 0, which it will be reassigned when the iteration on layers is done.
                        sigmoid_logits_cache.append(sigmoid_logits)
                        similarities_cache.append(similarities)
                        dlo_loss = torch.tensor(0., device=hidden_states.device, dtype=hidden_states.dtype)
                else:
                    dlo_loss = torch.tensor(0., device=hidden_states.device, dtype=hidden_states.dtype)

                if dlo_loss is None:
                    raise ValueError("dlo_loss")
                else:
                    dlo_losses += dlo_loss

                # print("dlo_loss", dlo_loss.shape, dlo_loss.dtype, dlo_loss)

            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,  # 🔍
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        causal_mask=causal_mask,  # 🔍
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )

                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 🔍 Post calculation of DLO loss for the "cos-global" setting
        if self.training and self.dlo_loss_type == "cos-global":  # 🔍
            dlo_layer_num = len(sigmoid_logits_cache)
            sigmoid_logits_cache = torch.stack(sigmoid_logits_cache, dim=0)  # (dlo_layer_num, batch_size, seq_len)
            similarities_cache = torch.stack(similarities_cache, dim=0)  # (dlo_layer_num, batch_size, seq_len)

            # print("dlo_layer_num", dlo_layer_num)
            # print("sigmoid_logits_cache", sigmoid_logits_cache.shape)
            # print("similarities_cache", similarities_cache.shape)

            with torch.no_grad():
                if attention_mask is None:
                    global_num_tokens_selected = math.ceil(batch_size * seq_len * dlo_layer_num * self.dlo_global_avg_capacity)
                    topk_similarities, _ = similarities_cache.flatten().topk(global_num_tokens_selected, dim=0, largest=False)
                    similarity_threshold = topk_similarities[-1]
                    labels = (similarities_cache <= similarity_threshold).to(sigmoid_logits_cache.dtype)  # (dlo_layer_num, batch_size, seq_len)
                    capacities = labels.clone().float().reshape(dlo_layer_num, -1).sum(1) / (batch_size * seq_len)
                else:
                    # for padding tokens, they should be classified into the "drop" class
                    global_num_tokens_selected = math.ceil(non_padding_token_num * dlo_layer_num * self.dlo_global_avg_capacity)
                    non_padding_similarities = similarities_cache[attention_mask.expand(dlo_layer_num, batch_size, seq_len)]  # (dlo_layer_num * non_padding_token_num)
                    topk_similarities, _ = non_padding_similarities.topk(global_num_tokens_selected, dim=0, largest=False)
                    similarity_threshold = topk_similarities[-1]
                    labels = (attention_mask.expand(dlo_layer_num, batch_size, seq_len) & (similarities_cache <= similarity_threshold)).to(sigmoid_logits_cache.dtype)  # (dlo_layer_num, batch_size, seq_len)
                    capacities = labels.clone().float().reshape(dlo_layer_num, -1).sum(1) / non_padding_token_num

            dlo_losses = self.bce_loss(sigmoid_logits_cache, labels)  # reassign dlo_losses
            dlo_losses *= dlo_layer_num  # scale the loss

            # print("sigmoid_logits_cache", sigmoid_logits_cache.shape)
            # print("similarity_threshold", similarity_threshold)
            # print("capacities", capacities)
            # print("dlo_losses", dlo_losses)

            # add missing values for non-DLO layers
            padded_capacities = []
            dlo_layer_index = 0
            for i in range(self.config.num_hidden_layers):
                if self.config.is_dlo[i]:
                    padded_capacities.append(capacities[dlo_layer_index].item())
                    dlo_layer_index += 1
                else:
                    padded_capacities.append(None)

            self.set_dlo_capacity(padded_capacities)  # update the layer-wise capacity

            # print("padded_capacities", capacities)
            # print("self.dlo_capacity", self.dlo_capacity)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseDLOModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            dlo_losses=dlo_losses,  # 🔍
        )

    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                : mask_shape[0], : mask_shape[1], offset: mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaDLOForCausalLM(LlamaDLOPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaDLOModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.total_dlo_layers = sum(config.is_dlo)  # 🔍

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_dlo_capacity(self, dlo_capacity):  # 🔍
        self.config.dlo_capacity = dlo_capacity
        self.model.set_dlo_capacity(dlo_capacity)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseDLOModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss += outputs.dlo_losses / self.total_dlo_layers * self.config.dlo_loss_coefficient  # 🔍

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            use_cache=True,
            **kwargs,
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class LlamaDLOForSequenceClassification(LlamaDLOPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaDLOModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class LlamaDLOForQuestionAnswering(LlamaDLOPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->LlamaDLO
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaDLOModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
