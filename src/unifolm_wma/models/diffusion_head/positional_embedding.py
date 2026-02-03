import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Register a non-persistent buffer to track dtype (won't be saved in state_dict)
        self.register_buffer('_dtype_tracker', torch.zeros(1), persistent=False)

    def forward(self, x):
        device = x.device
        target_dtype = self._dtype_tracker.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(target_dtype)
