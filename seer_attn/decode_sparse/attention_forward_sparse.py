import os
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F

from flash_attn.bert_padding import index_first_axis, unpad_input, pad_input
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
from seer_attn.kernels.varlen.block_sparse_flash_decode_varlen_mask_leftpad import block_sparse_flash_decode_leftpad_gqa_mask
from seer_attn.kernels.varlen.triton_sparse_gqa_decode_varlen_mask import block_sparse_flash_decode_gqa_mask_triton
from seer_attn.kernels.varlen.triton_sparse_gqa_decode_varlen_indice import block_sparse_flash_decode_gqa_indice_triton
from einops import rearrange
import math
import os

def convert_mask2indices(block_mask, max_selected_blocks):
    B, H, num_blocks = block_mask.shape
    # 创建索引矩阵: [0, 1, 2, ..., num_blocks-1]
    indices = torch.arange(num_blocks, dtype=torch.int32, device=block_mask.device).unsqueeze(0).unsqueeze(0)
    indices = indices.expand(B, H, -1)  # 扩展为 [B, H, num_blocks]
    
    # 未选中位置填充极小值（确保排序靠后）
    fill_value = -num_blocks - 1
    adjusted = torch.where(block_mask, indices, fill_value)
    
    # 降序排序（选中的大索引靠前）
    sorted_adj, _ = adjusted.sort(dim=-1, descending=True)
    
    # # 截取前 max_selected_blocks 个
    # selected = sorted_adj[..., :max_selected_blocks]
    
    # 将填充值替换为 -1
    block_indices = torch.where(sorted_adj < 0, -1, sorted_adj)
    
    return block_indices

def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, :query_length]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sparse_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    softmax_scale: Optional[float] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_mask: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    sparse_decode_kernel: Optional[callable] = None,
    max_sel_blocks: Optional[int] = None,
    **kwargs,
):

    if query_length > 1:
        
        # q = query_states.transpose(1, 2)
        # k = key_states.transpose(1, 2)
        # v = value_states.transpose(1, 2)
        # k = repeat_kv(k, q.shape[1] // k.shape[1])
        # v = repeat_kv(v, q.shape[1] // v.shape[1])
        # attention_mask = attention_mask.to(torch.bool)
        # expanded_attn_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)

        # seq_len = q.size(2)
        # causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))  
        
        # effective_mask = expanded_attn_mask & causal_mask[None,:,:]  # (batch_size, seq_len, seq_len)

        # scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))  # (batch_size, heads, seq_len, seq_len)
        # # Apply mask to scores
        # scores = scores.masked_fill(~effective_mask[:, None, :], float('-inf'))

        # # Compute attention weights with softmax
        # all_inf_mask = scores.isinf().all(dim=-1, keepdim=True)  # (batch_size, heads, seq_len, 1)
        # scores_noinf = scores.masked_fill(all_inf_mask, 0.0) 

        # # Compute attention weights with softmax
        # attn_weights = torch.nn.functional.softmax(scores_noinf, dim=-1)
        # attn_weights = torch.where(all_inf_mask, 0.0, attn_weights) 

        # # Compute output
        # attn_output = torch.matmul(attn_weights, v)  # (batch_size, heads, seq_len, hidden_size)
        # attn_output = attn_output.transpose(1, 2)  # ( batch_size, seq_len, heads, hidden_size)

        # assert attention_mask is not None, "Attention mask must be provided for Flash Attention."
        if attention_mask is None:
            attention_mask = torch.ones(
                (query_states.shape[0], query_states.shape[1]), device=query_states.device, dtype=torch.int32
            )
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            softmax_scale=softmax_scale,
            causal=True,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    else:
        query_states = query_states.squeeze(1)
        block_indices = convert_mask2indices(block_mask, max_selected_blocks=max_sel_blocks)
        attn_output = sparse_decode_kernel(query_states, key_states, value_states, block_indices, cache_seqlens)

    return attn_output


