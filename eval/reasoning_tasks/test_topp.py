import torch

def get_sparse_attn_mask_from_topp(x, p=0.9):

    sorted_weights, sorted_indices = torch.sort(x, dim=-1, descending=True)

    cumulative_weights = torch.cumsum(sorted_weights, dim=-1)

    sorted_mask = (cumulative_weights - sorted_weights) < p

    final_mask = torch.zeros_like(x, dtype=torch.bool)
    
    final_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

    return final_mask

x = torch.tensor([[0.15, 0.04, 0.7, 0.05, 0.06]])

mask = get_sparse_attn_mask_from_topp(x, p=0.9)
print(mask)