from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from transformers.cache_utils import Cache

class KCompressionCache(Cache):

    def __init__(self, num_layers: int, block_size: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.block_size = block_size
        # initialize caches for each layer
        self.k_compressed: Dict[int, Optional[torch.Tensor]] = {}
        self.k_remainder: Dict[int, Optional[torch.Tensor]] = {}
        for layer in range(num_layers):
            self.k_compressed[layer] = None  
            self.k_remainder[layer] = None  

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        # Return a tuple of (k_cache, k_remainder) for a given layer.
        return [(self.k_compressed[layer_idx], self.k_remainder[layer_idx])]

    def batch_select_indices(self, indices: torch.Tensor):
        for layer in range(self.num_layers):
            self.k_compressed[layer] = self.k_compressed[layer][indices, ...]
            self.k_remainder[layer] = self.k_remainder[layer][indices, ...]

    def get_k_remainder(self, layer_idx: int) -> torch.Tensor:
        return self.k_remainder[layer_idx]

    def update(
        self,
        layer_idx: int,
        k: Optional[torch.Tensor] = None,
        k_compressed: Optional[torch.Tensor] = None,
        k_remainder: Optional[torch.Tensor] = None, 
        is_decode: bool = False,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:

        if k_compressed is not None:
            if is_decode:
                self.k_compressed[layer_idx][:, -1:, :, :] = k_compressed
            else:
                self.k_compressed[layer_idx] = k_compressed
                bsz = k_compressed.shape[0]
                self.k_remainder[layer_idx] = torch.zeros(
                    [bsz, self.block_size, k_compressed.shape[2], k_compressed.shape[3]], device=k_compressed.device, dtype=k_compressed.dtype)
                if k_remainder is not None:        
                    self.k_remainder[layer_idx][:, :k_remainder.shape[1],] = k_remainder
                
                if layer_idx == 0:
                    self.remainder_len = k_remainder.shape[1] if k_remainder is not None else 0


        elif k is not None:
            self.k_remainder[layer_idx][:, self.remainder_len:self.remainder_len + 1, :, :] = k
            if layer_idx == 0:
                self.remainder_len += 1
                self.remainder_len %= self.block_size
            if self.remainder_len == 1:
                b, _, h, d = self.k_compressed[layer_idx].shape
                dtype, devcie = self.k_compressed[layer_idx].dtype, self.k_compressed[layer_idx].device
                self.k_compressed[layer_idx] = torch.cat(
                    [self.k_compressed[layer_idx], torch.zeros([b, 1, h, d], device=devcie, dtype=dtype)], dim=1)

        return self.k_compressed[layer_idx]

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`, modified to support right padding.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device.
    """

    is_compileable = True

    def __init__(
        self,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self._dtype = dtype

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (self.max_batch_size, self.max_cache_len, self.hidden_size)
        device = torch.device(device) if device is not None else None
        for idx in range(num_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`, supporting right padding.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments including `seq_lengths` to specify the current sequence lengths.

        Return:
            A tuple containing the updated key and value states.
        """

        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)

        if key_states.size(1) > 1:
            for b in range(key_states.size(0)):
                pos = cache_seqlens[b]
                
                self.key_cache[layer_idx][b, :pos, :] = key_states[b, :pos, :]
                self.value_cache[layer_idx][b, :pos, :] = value_states[b, :pos, :]
                # if layer_idx == 0:
                #     print("pos:", pos)
                #     print("cur_length:", torch.sum(torch.any(self.key_cache[layer_idx][b] != 0, dim=-1)).item())
            if layer_idx == 0:
                self.seq_length = key_states.size(1)
                # print("key_states shape:", key_states.shape, "seq_length:", self.seq_length)

        else:
            for b in range(key_states.size(0)):
                pos = cache_seqlens[b]
                # print("key_states[b] shape:", key_states[b].shape)
                self.key_cache[layer_idx][b, pos-1, :] = key_states[b]
                self.value_cache[layer_idx][b, pos-1, :] = value_states[b]
                # if layer_idx == 0:
                #     print("pos:",pos)
                #     print("cur_length:",torch.sum(torch.any(self.key_cache[layer_idx][b] != 0, dim=-1)).item())
            if layer_idx == 0:
                self.seq_length += 1
                # print("key_states shape:", key_states.shape, "seq_length:", self.seq_length)
        # print("seq_length:", self.seq_length)
        # print("key_cache shape:", self.key_cache[layer_idx][:self.seq_length].shape)
        return self.key_cache[layer_idx][:,:self.seq_length,:], self.value_cache[layer_idx][:,:self.seq_length,:]

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self) -> int:
        """
        Returns the current sequence length of the cache.
        """
        return self.seq_length if hasattr(self, 'seq_length') else 0
