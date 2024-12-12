# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# And https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class AttnProcessor(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        num_inference_steps=50,
        end_fusion=10,
        name=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.denoise_step = 0
        self.num_inference_steps=num_inference_steps
        self.end_fusion = end_fusion
        self.name = name

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        self.denoise_step += 1
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.denoise_step <= self.end_fusion:
            assert query.shape[0] == 2
            if "up_blocks.1.attentions.2.transformer_blocks.1" in self.name and self.denoise_step == self.end_fusion:
                print("end now:", self.end_fusion)   
            scale_factor = 1 / torch.sqrt(torch.tensor(head_dim, dtype=query.dtype))
            attn_probs = (torch.matmul(query, key.transpose(-2, -1)) * scale_factor).softmax(dim=-1)
            attn_probs[0] = attn_probs[1]
            hidden_states = torch.matmul(attn_probs, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.denoise_step == self.num_inference_steps:
            self.denoise_step = 0
        return hidden_states


class DualAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for DualCrossAttn for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4,
                 state_dict=None, use_norm=False, layer_idx=None,):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_style = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_style = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        if state_dict is not None:
            # load sdxl text cross attention's weights
            self.to_k_style.load_state_dict(state_dict['k'], strict=True)
            self.to_v_style.load_state_dict(state_dict['v'], strict=True)
        
        self.use_norm = use_norm
        if use_norm:
            self.norm_style = nn.LayerNorm(hidden_size)
        else:
            self.norm_style = lambda x: x        
        
        self.layer_idx = layer_idx
        
        self.denoise_step = 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, style_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for style features
        style_key = self.to_k_style(style_hidden_states)
        style_value = self.to_v_style(style_hidden_states)

        style_key = style_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        style_value = style_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        style_hidden_states = F.scaled_dot_product_attention(
            query, style_key, style_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ style_key.transpose(-2, -1).softmax(dim=-1)
            #print(self.attn_map.shape)

        style_hidden_states = style_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        style_hidden_states = style_hidden_states.to(query.dtype)

        if self.use_norm:
            style_hidden_states = self.norm_style(style_hidden_states)

        hidden_states = hidden_states + self.scale * style_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DualAttnASProcessor(DualAttnProcessor):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale_scalar=None,
        *args,
        **kwargs,
    ):
        self.denoise_step += 1
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = 77
            if encoder_hidden_states.shape[1] > 77:
                encoder_hidden_states, style_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            else:
                assert encoder_hidden_states.shape[1] == 77
                encoder_hidden_states = encoder_hidden_states
                style_hidden_states = None

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        raw_hidden_states = hidden_states
        ########################################
        if style_hidden_states is not None:
            if scale_scalar is not None:
                scale = 1 + scale_scalar[:, self.layer_idx]
                scale = scale[:, None]
                # print(f"Using scale: {scale}")
            else:
                scale = 1.0

            ## for style features
            style_key = self.to_k_style(style_hidden_states)
            style_value = self.to_v_style(style_hidden_states)

            style_key = style_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            style_value = style_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            style_hidden_states = F.scaled_dot_product_attention(
                query, style_key, style_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            with torch.no_grad():
                self.attn_map = query @ style_key.transpose(-2, -1).softmax(dim=-1)
                #print(self.attn_map.shape)

            style_hidden_states = style_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            style_hidden_states = style_hidden_states.to(query.dtype)

            if self.use_norm:
                style_hidden_states = self.norm_style(style_hidden_states)

            hidden_states = hidden_states + scale * self.scale * style_hidden_states

        if hidden_states.shape[0] == 2 and style_hidden_states is not None:
            # pdb.set_trace()
            hidden_states[1] = raw_hidden_states[1]
        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.denoise_step == 50:
            self.denoise_step = 0
        return hidden_states
