import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from seer_attn.kernels.varlen.flash_decode_varlen_left_pad_max_v2 import flash_decode_leftpad
from seer_attn.kernels.varlen.block_sparse_attn_varlen_2d_leftpad import blocksparse_flash_attn_varlen_leftpad
from seer_attn.kernels.varlen.block_sparse_attn_varlen_gqa import blocksparse_flash_attn_varlen_fwd
from seer_attn.kernels.block_sparse_attn import block_sparse_triton_fn
import os
import math

from seer_attn.modules.common import (
    repeat_kv_varlen,
    repeat_kv,
    pad_input,
    _upad_input,
    get_sparse_attn_mask_from_nz_ratio,
    # get_sparse_attn_mask_from_threshold,
)

def get_sparse_attn_mask_from_threshold(x, threshold, use_dense_for_last_block=False):
    dense_mask = x > threshold 
    if use_dense_for_last_block:
        dense_mask[:, :, -128:, :] = True  # 128
    # dense_mask.tril_()
    return dense_mask

def sparse_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    softmax_scale: Optional[float] = None,
    attn_gate_score: Optional[torch.Tensor] = None,
    sparsity_method: Optional[str] = None,
    threshold: Optional[float] = None,
    nz_ratio: Optional[float] = None,
    last_block_dense: Optional[bool] = None,
    block_size: Optional[int] = None,
    num_key_value_groups: Optional[int] = None,
    profile_file: Optional[str] = None,
    **kwargs,
):


    if query_length > 1:
        if sparsity_method == "nz_ratio":
            downsampled_len = math.ceil(key_states.shape[-2] / block_size)
            gate_mask = get_sparse_attn_mask_from_nz_ratio(attn_gate_score, nz_ratio, last_block_dense)
        elif sparsity_method == "threshold":
            gate_mask = get_sparse_attn_mask_from_threshold(attn_gate_score, threshold, last_block_dense)
            if profile_file is not None:
                downsampled_len = gate_mask.shape[-1]
                total_causal_size = ((1 + downsampled_len) * downsampled_len / 2) * gate_mask.shape[0] * gate_mask.shape[1]
                with open(profile_file, "a") as f:
                    f.write(f"{query_length}: {gate_mask.sum().item() / total_causal_size}\n")

        cu_seqlens_k = torch.tensor([0, key_states.shape[1]], device=key_states.device)
        cu_seqlens_q = torch.tensor([0, query_states.shape[1]], device=query_states.device)
        max_seqlen = key_states.shape[1]
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        # print("cu_seqlens_k:", cu_seqlens_k, "cu_seqlens_q:", cu_seqlens_q, "max_seqlen:", max_seqlen)
        # print("gate_mask shape:", gate_mask.shape)
        attn_output, _ = blocksparse_flash_attn_varlen_fwd(
            query_states, 
            key_states, 
            value_states, 
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen=max_seqlen,
            sm_scale=softmax_scale,
            block_mask=gate_mask,
            block_size=block_size,
        )

    else:

        if attention_mask is None:
            cache_seqlens = torch.full(
                (key_states.shape[0],), key_states.shape[1], dtype=torch.int32, device=key_states.device)
        else:
            cache_seqlens = torch.sum(attention_mask.to(torch.int32), dim=-1, dtype=torch.int32) 

        attn_output = flash_decode_leftpad(
            query_states, 
            key_states,
            value_states, 
            cache_seqlens=cache_seqlens, 
            block_size=block_size,
            sm_scale=softmax_scale,
        )

    return attn_output


