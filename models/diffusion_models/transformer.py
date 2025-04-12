# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### modified from https://github.com/huggingface/diffusers/blob/v0.19.3/src/diffusers/models/transformer_temporal.py 

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_utils import ModelMixin
from .util import register_model

@dataclass
class TransformerDenoiserOutput(BaseOutput):
    """
    The output of [`TransformerDenoiser`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size x num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: torch.FloatTensor

@register_model('TransformerDenoiser')
class TransformerDenoiser(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        in_channels: Optional[int] = None,   
        num_layers: int = 24,
        dropout: float = 0.1, 
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,  
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = False,
        ### [0615] Added by YYK
        positional_embeddings: Optional[str] = None, # "sinusoidal"
        num_positional_embeddings: Optional[int] = None,
        num_embeds_ada_norm: Optional[int] = None, # Size of time embedding lookup table for AdaNorm (default: None) 2000?
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.LayerNorm(normalized_shape=in_channels, eps=1e-6, elementwise_affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        assert in_channels <= inner_dim, f'{in_channels} <= {inner_dim} False'

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    ### [0615] Added by YYK
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings,
                    norm_type=norm_type,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        self,
        hidden_states, 
        timestep=None,
        class_labels=None, 
        cross_attention_kwargs=None,
        return_dict: bool = True,
        # added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerDenoiserOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerDenoiserOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        # print("Hidden states shape:", hidden_states.shape)

        batch_size, seq_length, input_dim = hidden_states.shape  

        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if timestep is not None:
                if block.use_ada_layer_norm:
                    timestep_b = timestep.unsqueeze(-1).to(hidden_states.device)
                else:
                    timestep_b = timestep.to(hidden_states.device)
            else:
                timestep_b = timestep

            ## yingqing
            # print(timestep_b)
                
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=None,
                # timestep=timestep,
                timestep=timestep_b,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                # added_cond_kwargs=add_cond_kwargs,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerDenoiserOutput(sample=output)
