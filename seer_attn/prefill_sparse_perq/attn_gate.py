
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import combinations
from seer_attn.modules.common import apply_rotary_pos_emb, apply_rotary_pos_emb_single, repeat_kv, repeat_kv_varlen
from flash_attn.layers.rotary import apply_rotary_emb_func
import math



def min_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return -F.max_pool3d(-input, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)



# class MultiHeadLinear(nn.Module):
#     def __init__(self, in_channel_size, hidden_size, num_head):
#         super(MultiHeadLinear, self).__init__()
#         self.in_channel = in_channel_size
#         self.hidden_size = hidden_size
#         self.num_head = num_head
#         self.weight = nn.Parameter(torch.Tensor(self.num_head, self.in_channel, self.hidden_size))
    

#     def forward(self, x): # x shape (batch_size, seq_length, head, channel_size)
#         if x.shape[2] < self.num_head:
#             x = repeat_kv_varlen(x, self.num_head // x.shape[2])
#         # print(f"x.shape: {x.shape}, self.weight.shape: {self.weight.shape}")
#         return torch.einsum('bshi, hio->bsho', x, self.weight) # torch.matmul(x, self.weight)
#         # return torch.matmul(x, self.weight) # torch.einsum('bhsi,hio->bhso', x, self.weight)

class HeadPoolingLinear(nn.Module):
    def __init__(self, num_k_head, gqa_group_size, model_hidden_size, gate_hidden_size):
        super(HeadPoolingLinear, self).__init__()
        self.num_k_head = num_k_head
        self.gqa_group_size = gqa_group_size
        self.model_hidden_size = model_hidden_size
        self.gate_hidden_size = gate_hidden_size
        self.weight = nn.Parameter(torch.Tensor(self.num_k_head, gqa_group_size, self.model_hidden_size, self.gate_hidden_size))
        self._init_weight()

    def _init_weight(self):
        init.xavier_uniform_(self.weight)

    def forward(self, x): 
        if x.dim() == 3: ## x shape (seq_length, num_q_head, channel_size)
            x = x.view(x.shape[0], self.num_k_head, self.gqa_group_size, x.shape[2])
            return torch.einsum('skgi,kgio->sko', x, self.weight)
        elif x.dim() == 4: ## x shape (b, seq_length, num_q_head, channel_size)
            x = x.view(x.shape[0], x.shape[1], self.num_k_head, self.gqa_group_size, x.shape[3])
            return torch.einsum('bskgi,kgio->bsko', x, self.weight)
        else:
            raise ValueError("x dim should be 3 or 4")


class MultiHeadLinear(nn.Module):
    def __init__(self, in_channel_size, hidden_size, num_head):
        super(MultiHeadLinear, self).__init__()
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.weight = nn.Parameter(torch.Tensor(self.num_head, self.in_channel, self.hidden_size))
        self._init_weight()
    

    def _init_weight(self):
        init.xavier_uniform_(self.weight)

    def forward(self, x): # x shape (seq_length, head, channel_size)
        # return torch.matmul(x, self.weight) 
        if x.dim() == 3:
            return torch.einsum('shi,hio->sho', x, self.weight)
        elif x.dim() == 4:
            return torch.einsum('bshi,hio->bsho', x, self.weight)
            # return torch.einsum('bhsi,hio->bhso', x, self.weight)
        else:
            raise ValueError("x dim should be 3 or 4")

class AttnGate(nn.Module):
    def __init__(self, 
                 block_size, 
                 model_hidden_size, 
                 gate_hidden_size, 
                 num_k_head, 
                 num_q_head, 
                 q_head_pooling_type, 
                 k_pooling_funcs,
                 use_flash_rope,
                 use_qk_norm,
                ):
        super(AttnGate, self).__init__()
        self.block_size = block_size
        self.model_hidden_size = model_hidden_size   
        self.gate_hidden_size = gate_hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.gqa_group_size = int(num_q_head // num_k_head)
        self.k_pooling_funcs = k_pooling_funcs
        self.use_flash_rope = use_flash_rope
        self.use_qk_norm = use_qk_norm
    

        self.k_dup_size = len(k_pooling_funcs)
        k_in_channel_size = model_hidden_size * self.k_dup_size
        
        self.q_head_pooling_type = q_head_pooling_type
        
        if self.q_head_pooling_type == "Qproj":
            self.attngate_linear_q = HeadPoolingLinear(self.num_k_head, self.gqa_group_size, self.model_hidden_size, self.gate_hidden_size)
        elif self.q_head_pooling_type == "Qavgproj":
            self.attngate_linear_q = MultiHeadLinear(self.model_hidden_size, self.gate_hidden_size, self.num_k_head)
        else:
            self.attngate_linear_q = None
        self.attngate_linear_k = MultiHeadLinear(k_in_channel_size, self.gate_hidden_size, self.num_k_head)

    
    def forward(
            self, 
            q, # [batch_size, seq_length, num_q_head, channel_size]
            k, # [batch_size, seq_length, num_k_head, channel_size]
            attention_mask, 
            block_position_embeddings=None, 
            position_embeddings_gate_q=None,
            use_softmax=True
        ):  
        q_len = q.shape[1]
        if q_len == 1:
            return None

        if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qavg":
            q = F.avg_pool2d(q, kernel_size=[self.gqa_group_size, 1], stride=[self.gqa_group_size, 1])
        if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qproj":
            q = self.attngate_linear_q(q)

        q = q.transpose(1, 2)  # [batch_size, num_q_head, seq_length, channel_size]

        if self.use_qk_norm:
            q = self.attngate_qnorm(q)

        if position_embeddings_gate_q is not None:  
            cos, sin = position_embeddings_gate_q
            q = apply_rotary_pos_emb_single(q, cos, sin, unsqueeze_dim=1)

        k_pooled = [pool_func(k, kernel_size=[self.block_size, 1, 1], stride=[self.block_size, 1, 1], ceil_mode=True) for pool_func in self.k_pooling_funcs]
        k = torch.cat(k_pooled, dim=-1)
        k = self.attngate_linear_k(k)

        k = k.transpose(1, 2)
        if self.use_qk_norm:
            k = self.attngate_knorm(k)

        if block_position_embeddings is not None:
            cos, sin = block_position_embeddings
            k = apply_rotary_pos_emb_single(k, cos, sin, unsqueeze_dim=1)

        

        attn = torch.matmul(q, k.transpose(-1, -2)) * (1 / math.sqrt(self.gate_hidden_size))
        # print("attn", attn, "mask", attention_mask)
        if attention_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attention_mask, -1e9)
        else:
            attn = attn + attention_mask
        if use_softmax:
            attn = F.softmax(attn, dim=-1)
        return attn


POOL_FUNCS = {
    'max': F.max_pool3d,
    'min': min_pool3d,
    'avg': F.avg_pool3d
}


def _create_generic_attngate_class(base_class, suffix, k_pooling_names):
    k_pooling_funcs = [POOL_FUNCS[name] for name in k_pooling_names]
    class_name = f"K{''.join(k_pooling_names)}{suffix}"

    class NewAttnGate(base_class):
        def __init__(self, block_size, model_hidden_size, gate_hidden_size, num_k_head, num_q_head, q_head_pooling_type, use_flash_rope=False, use_qk_norm=False):
            super(NewAttnGate, self).__init__(
                block_size=block_size,
                model_hidden_size=model_hidden_size,
                gate_hidden_size=gate_hidden_size,
                num_k_head=num_k_head,
                num_q_head=num_q_head,
                q_head_pooling_type=q_head_pooling_type,
                k_pooling_funcs=k_pooling_funcs,
                use_flash_rope=use_flash_rope,
                use_qk_norm=use_qk_norm,
            )
    NewAttnGate.__name__ = class_name
    return class_name, NewAttnGate


def generate_combinations():
    new_classes = {}
    pool_types = ['max', 'min', 'avg']

    for k_comb in range(1, 4):
        for k_pooling_comb in combinations(pool_types, k_comb):
            class_name, new_class = _create_generic_attngate_class(AttnGate, '', k_pooling_comb)
            new_classes[class_name] = new_class
    return new_classes


ATTNGATE_CLASSES = generate_combinations()