import torch
import time
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from functools import partial

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# ===== Op-level timing helpers =====
def _sync_device_op():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _time_start_op():
    _sync_device_op()
    return time.time()

def _time_end_op(start):
    _sync_device_op()
    return time.time() - start

# Op timing accumulators
_OP_TIMES = {
    'spatial_norm': 0.0, 'spatial_proj_in': 0.0, 'spatial_attn_blocks': 0.0, 'spatial_proj_out': 0.0,
    'temporal_norm': 0.0, 'temporal_proj_in': 0.0, 'temporal_attn_blocks': 0.0, 'temporal_proj_out': 0.0,
}
_OP_COUNTS = {k: 0 for k in _OP_TIMES}

def _reset_op_times():
    global _OP_TIMES, _OP_COUNTS
    _OP_TIMES = {k: 0.0 for k in _OP_TIMES}
    _OP_COUNTS = {k: 0 for k in _OP_COUNTS}

def _get_op_times():
    return _OP_TIMES, _OP_COUNTS
# ===== End op-level timing helpers =====

from unifolm_wma.utils.common import (
    checkpoint,
    exists,
    default,
)
from unifolm_wma.utils.basics import zero_module


class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat,
                                           -self.max_relative_position,
                                           self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 relative_position=False,
                 temporal_length=None,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 image_cross_attention_scale=1.0,
                 agent_state_cross_attention_scale=1.0,
                 agent_action_cross_attention_scale=1.0,
                 cross_attention_scale_learnable=False,
                 text_context_len=77):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

        self.relative_position = relative_position
        if self.relative_position:
            assert (temporal_length is not None)
            self.relative_position_k = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.agent_state_cross_attention_scale = agent_state_cross_attention_scale
        self.agent_action_cross_attention_scale = agent_action_cross_attention_scale
        self.text_context_len = text_context_len
        self.agent_state_context_len = agent_state_context_len
        self.agent_action_context_len = agent_action_context_len
        self.cross_attention_scale_learnable = cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_k_as = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_as = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_k_aa = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_aa = nn.Linear(context_dim, inner_dim, bias=False)
            if cross_attention_scale_learnable:
                self.register_parameter('alpha_ctx',
                                        nn.Parameter(torch.tensor(0.)))
                self.register_parameter('alpha_cas',
                                        nn.Parameter(torch.tensor(0.)))
                self.register_parameter('alpha_caa',
                                        nn.Parameter(torch.tensor(0.)))

    def forward(self, x, context=None, mask=None):
        # CrossAttention op timing
        if not hasattr(CrossAttention, '_op_times'):
            CrossAttention._op_times = {'to_qkv': 0.0, 'sdpa': 0.0, 'to_out': 0.0}
            CrossAttention._op_counts = {k: 0 for k in CrossAttention._op_times}
        
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        h = self.heads
        _t0 = _time_start_op()
        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            # assert 1 > 2, ">>> ERROR: should setup xformers and use efficient_forward ..."
            context_agent_state = context[:, :self.agent_state_context_len, :]
            context_agent_action = context[:,
                                           self.agent_state_context_len:self.
                                           agent_state_context_len +
                                           self.agent_action_context_len, :]
            context_ins = context[:, self.agent_state_context_len +
                                  self.agent_action_context_len:self.
                                  agent_state_context_len +
                                  self.agent_action_context_len +
                                  self.text_context_len, :]
            context_image = context[:, self.agent_state_context_len +
                                    self.agent_action_context_len +
                                    self.text_context_len:, :]

            k = self.to_k(context_ins)
            v = self.to_v(context_ins)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
            k_as = self.to_k_as(context_agent_state)
            v_as = self.to_v_as(context_agent_state)
            k_aa = self.to_k_aa(context_agent_action)
            v_aa = self.to_v_aa(context_agent_action)
        else:
            if not spatial_self_attn:
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)

        CrossAttention._op_times['to_qkv'] += _time_end_op(_t0)
        CrossAttention._op_counts['to_qkv'] += 1
        
        # Reshape for SDPA: b n (h d) -> b h n d
        _t0 = _time_start_op()
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # Prepare mask for SDPA if needed
        attn_mask = None
        if exists(mask):
            attn_mask = repeat(mask, 'b i j -> b h i j', h=h)
            attn_mask = attn_mask > 0.5

        # Use SDPA (handles relative position separately)
        if self.relative_position:
            # Fallback to manual for relative position
            q_m = rearrange(q, 'b h n d -> (b h) n d')
            k_m = rearrange(k, 'b h n d -> (b h) n d')
            v_m = rearrange(v, 'b h n d -> (b h) n d')
            len_q, len_k, len_v = q_m.shape[1], k_m.shape[1], v_m.shape[1]
            sim = torch.einsum('b i d, b j d -> b i j', q_m, k_m) * self.scale
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q_m, k2) * self.scale
            sim += sim2
            if attn_mask is not None:
                max_neg_value = -torch.finfo(sim.dtype).max
                attn_mask_m = rearrange(attn_mask, 'b h i j -> (b h) i j')
                sim.masked_fill_(~attn_mask_m, max_neg_value)
            sim = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', sim, v_m)
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2)
            out += out2
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        else:
            # Use optimized SDPA
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, 
                is_causal=False, scale=self.scale
            )
            out = rearrange(out, 'b h n d -> b n (h d)')

        if k_ip is not None and k_as is not None and k_aa is not None:
            ## Use SDPA for cross-attentions
            k_ip = rearrange(k_ip, 'b n (h d) -> b h n d', h=h)
            v_ip = rearrange(v_ip, 'b n (h d) -> b h n d', h=h)
            out_ip = F.scaled_dot_product_attention(
                q, k_ip, v_ip, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=self.scale
            )
            out_ip = rearrange(out_ip, 'b h n d -> b n (h d)')

            k_as = rearrange(k_as, 'b n (h d) -> b h n d', h=h)
            v_as = rearrange(v_as, 'b n (h d) -> b h n d', h=h)
            out_as = F.scaled_dot_product_attention(
                q, k_as, v_as, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=self.scale
            )
            out_as = rearrange(out_as, 'b h n d -> b n (h d)')

            k_aa = rearrange(k_aa, 'b n (h d) -> b h n d', h=h)
            v_aa = rearrange(v_aa, 'b n (h d) -> b h n d', h=h)
            out_aa = F.scaled_dot_product_attention(
                q, k_aa, v_aa, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=self.scale
            )
            out_aa = rearrange(out_aa, 'b h n d -> b n (h d)')

        CrossAttention._op_times['sdpa'] += _time_end_op(_t0)
        CrossAttention._op_counts['sdpa'] += 1
        
        if out_ip is not None and out_as is not None and out_aa is not None:
            if self.cross_attention_scale_learnable:
                out = out + \
                    self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha_ctx) + 1) + \
                    self.agent_state_cross_attention_scale * out_as * (torch.tanh(self.alpha_cas) + 1) + \
                    self.agent_action_cross_attention_scale * out_aa * (torch.tanh(self.alpha_caa) + 1)
            else:
                out = out + \
                    self.image_cross_attention_scale * out_ip + \
                    self.agent_state_cross_attention_scale * out_as + \
                    self.agent_action_cross_attention_scale * out_aa

        _t0 = _time_start_op()
        result = self.to_out(out)
        CrossAttention._op_times['to_out'] += _time_end_op(_t0)
        CrossAttention._op_counts['to_out'] += 1
        return result

    def efficient_forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k, v, out = None, None, None
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        q = self.to_q(x)
        context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            if context.shape[1] == self.text_context_len + self.video_length:
                context_ins, context_image = context[:, :self.text_context_len, :], context[:,self.text_context_len:, :]
                k = self.to_k(context)
                v = self.to_v(context)
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
            elif context.shape[1] == self.agent_state_context_len + self.text_context_len + self.video_length:
                context_agent_state = context[:, :self.agent_state_context_len, :]
                context_ins = context[:, self.agent_state_context_len:self.agent_state_context_len+self.text_context_len, :]
                context_image = context[:, self.agent_state_context_len+self.text_context_len:, :]
                k = self.to_k(context_ins)
                v = self.to_v(context_ins)
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
                k_as = self.to_k_as(context_agent_state)
                v_as = self.to_v_as(context_agent_state)
            else:
                context_agent_state = context[:, :self.agent_state_context_len, :]
                context_agent_action = context[:, self.agent_state_context_len:self.agent_state_context_len+self.agent_action_context_len, :]
                context_ins = context[:, self.agent_state_context_len+self.agent_action_context_len:self.agent_state_context_len+self.agent_action_context_len+self.text_context_len, :]
                context_image = context[:, self.agent_state_context_len+self.agent_action_context_len+self.text_context_len:, :]

                k = self.to_k(context_ins)
                v = self.to_v(context_ins)
                k_ip = self.to_k_ip(context_image)
                v_ip = self.to_v_ip(context_image)
                k_as = self.to_k_as(context_agent_state)
                v_as = self.to_v_as(context_agent_state)
                k_aa = self.to_k_aa(context_agent_action)
                v_aa = self.to_v_aa(context_agent_action)

                attn_mask_aa = self._get_attn_mask_aa(x.shape[0],
                                                      q.shape[1],
                                                      k_aa.shape[1],
                                                      block_size=16).to(k_aa.device)
        else:
            if not spatial_self_attn:
                assert 1 > 2, ">>> ERROR: you should never go into here ..."
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q = q.unsqueeze(3).reshape(b, q.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(b * self.heads, q.shape[1], self.dim_head).contiguous()
        if k is not None:
            k, v = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(),
                (k, v),
            )
            out = xformers.ops.memory_efficient_attention(q,
                                                          k,
                                                          v,
                                                          attn_bias=None,
                                                          op=None)
            out = (out.unsqueeze(0).reshape(
                b, self.heads, out.shape[1],
                self.dim_head).permute(0, 2, 1,
                                       3).reshape(b, out.shape[1],
                                                  self.heads * self.dim_head))

        if k_ip is not None:
            # For image cross-attention
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(
                        ),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q,
                                                             k_ip,
                                                             v_ip,
                                                             attn_bias=None,
                                                             op=None)
            out_ip = (out_ip.unsqueeze(0).reshape(
                b, self.heads, out_ip.shape[1],
                self.dim_head).permute(0, 2, 1,
                                       3).reshape(b, out_ip.shape[1],
                                                  self.heads * self.dim_head))

        if k_as is not None:
            # For agent state cross-attention
            k_as, v_as = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(
                        ),
                (k_as, v_as),
            )
            out_as = xformers.ops.memory_efficient_attention(q,
                                                             k_as,
                                                             v_as,
                                                             attn_bias=None,
                                                             op=None)
            out_as = (out_as.unsqueeze(0).reshape(
                b, self.heads, out_as.shape[1],
                self.dim_head).permute(0, 2, 1,
                                       3).reshape(b, out_as.shape[1],
                                                  self.heads * self.dim_head))
        if k_aa is not None:
            # For agent action cross-attention
            k_aa, v_aa = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(
                        ),
                (k_aa, v_aa),
            )

            attn_mask_aa = attn_mask_aa.unsqueeze(1).repeat(1,self.heads,1,1).reshape(
                    b * self.heads, attn_mask_aa.shape[1], attn_mask_aa.shape[2])
            attn_mask_aa = attn_mask_aa.to(q.dtype)

            out_aa = xformers.ops.memory_efficient_attention(
                q, k_aa, v_aa, attn_bias=attn_mask_aa, op=None)

            out_aa = (out_aa.unsqueeze(0).reshape(
                b, self.heads, out_aa.shape[1],
                self.dim_head).permute(0, 2, 1,
                                       3).reshape(b, out_aa.shape[1],
                                                  self.heads * self.dim_head))
        if exists(mask):
            raise NotImplementedError

        out = 0.0 if out is None else out
        out_ip = 0.0 if out_ip is None else out_ip
        out_as = 0.0 if out_as is None else out_as
        out_aa = 0.0 if out_aa is None else out_aa

        if self.cross_attention_scale_learnable:
            out = out + \
                self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha_ctx) + 1) + \
                self.agent_state_cross_attention_scale * out_as * (torch.tanh(self.alpha_cas) + 1) + \
                self.agent_action_cross_attention_scale * out_aa * (torch.tanh(self.alpha_caa) + 1)

        else:
            out = out + \
                self.image_cross_attention_scale * out_ip + \
                self.agent_state_cross_attention_scale * out_as + \
                self.agent_action_cross_attention_scale * out_aa

        return self.to_out(out)

    def _get_attn_mask_aa(self, b, l1, l2, block_size=16):
        num_token = l2 // block_size
        start_positions = ((torch.arange(b) % block_size) + 1) * num_token
        col_indices = torch.arange(l2)
        mask_2d = col_indices.unsqueeze(0) >= start_positions.unsqueeze(1)
        mask = mask_2d.unsqueeze(1).expand(b, l1, l2)
        attn_mask = torch.zeros_like(mask, dtype=torch.float)
        attn_mask[mask] = float('-inf')
        return attn_mask


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 disable_self_attn=False,
                 attention_cls=None,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 image_cross_attention_scale=1.0,
                 cross_attention_scale_learnable=False,
                 text_context_len=77):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            video_length=video_length,
            agent_state_context_len=agent_state_context_len,
            agent_action_context_len=agent_action_context_len,
            image_cross_attention=image_cross_attention,
            image_cross_attention_scale=image_cross_attention_scale,
            cross_attention_scale_learnable=cross_attention_scale_learnable,
            text_context_len=text_context_len)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, **kwargs):
        # implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  # should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x, ), self.parameters(),
                              self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(),
                          self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        # BasicTransformerBlock op timing
        if not hasattr(BasicTransformerBlock, '_op_times'):
            BasicTransformerBlock._op_times = {'self_attn': 0.0, 'cross_attn': 0.0, 'ff': 0.0}
            BasicTransformerBlock._op_counts = {k: 0 for k in BasicTransformerBlock._op_times}
        
        _t0 = _time_start_op()
        x = self.attn1(self.norm1(x),
                       context=context if self.disable_self_attn else None,
                       mask=mask) + x
        BasicTransformerBlock._op_times['self_attn'] += _time_end_op(_t0)
        BasicTransformerBlock._op_counts['self_attn'] += 1
        
        _t0 = _time_start_op()
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        BasicTransformerBlock._op_times['cross_attn'] += _time_end_op(_t0)
        BasicTransformerBlock._op_counts['cross_attn'] += 1
        
        _t0 = _time_start_op()
        x = self.ff(self.norm3(x)) + x
        BasicTransformerBlock._op_times['ff'] += _time_end_op(_t0)
        BasicTransformerBlock._op_counts['ff'] += 1
        
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 disable_self_attn=False,
                 use_linear=False,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 cross_attention_scale_learnable=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                agent_state_context_len=agent_state_context_len,
                agent_action_context_len=agent_action_context_len,
                image_cross_attention=image_cross_attention,
                cross_attention_scale_learnable=cross_attention_scale_learnable,
            ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        # Op timing: norm
        _t0 = _time_start_op()
        x = self.norm(x)
        _OP_TIMES['spatial_norm'] += _time_end_op(_t0)
        _OP_COUNTS['spatial_norm'] += 1
        # Op timing: proj_in
        _t0 = _time_start_op()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        _OP_TIMES['spatial_proj_in'] += _time_end_op(_t0)
        _OP_COUNTS['spatial_proj_in'] += 1
        # Op timing: attn_blocks
        _t0 = _time_start_op()
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        _OP_TIMES['spatial_attn_blocks'] += _time_end_op(_t0)
        _OP_COUNTS['spatial_attn_blocks'] += 1
        # Op timing: proj_out
        _t0 = _time_start_op()
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        _OP_TIMES['spatial_proj_out'] += _time_end_op(_t0)
        _OP_COUNTS['spatial_proj_out'] += 1
        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 use_linear=False,
                 only_self_att=True,
                 causal_attention=False,
                 causal_block_size=1,
                 relative_position=False,
                 temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert (temporal_length is not None)
            attention_cls = partial(CrossAttention,
                                    relative_position=True,
                                    temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention,
                                    temporal_length=temporal_length)
        if self.causal_attention:
            assert (temporal_length is not None)
            self.mask = torch.tril(
                torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim,
                                  n_heads,
                                  d_head,
                                  dropout=dropout,
                                  context_dim=context_dim,
                                  attention_cls=attention_cls,
                                  checkpoint=use_checkpoint)
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        # Op timing: norm
        _t0 = _time_start_op()
        x = self.norm(x)
        _OP_TIMES['temporal_norm'] += _time_end_op(_t0)
        _OP_COUNTS['temporal_norm'] += 1
        # Op timing: proj_in
        _t0 = _time_start_op()
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        _OP_TIMES['temporal_proj_in'] += _time_end_op(_t0)
        _OP_COUNTS['temporal_proj_in'] += 1

        temp_mask = None
        if self.causal_attention:
            # Slice the from mask map
            temp_mask = self.mask[:, :t, :t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b * h * w)
        else:
            mask = None

        # Op timing: attn_blocks
        _t0 = _time_start_op()
        if self.only_self_att:
            # NOTE: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con',
                                t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # Calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(context[j],
                                       't l con -> (t r) l con',
                                       r=(h * w) // t,
                                       t=t).contiguous()
                    # Note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)

        _OP_TIMES['temporal_attn_blocks'] += _time_end_op(_t0)
        _OP_COUNTS['temporal_attn_blocks'] += 1
        # Op timing: proj_out
        _t0 = _time_start_op()
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h,
                          w=w).contiguous()
        _OP_TIMES['temporal_proj_out'] += _time_end_op(_t0)
        _OP_COUNTS['temporal_proj_out'] += 1

        return x + x_in


class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        # FeedForward op timing
        if not hasattr(FeedForward, '_op_times'):
            FeedForward._op_times = {'geglu': 0.0, 'linear_out': 0.0}
            FeedForward._op_counts = {k: 0 for k in FeedForward._op_times}
        
        _t0 = _time_start_op()
        x = self.net[0](x)  # project_in (GEGLU or Linear+GELU)
        FeedForward._op_times['geglu'] += _time_end_op(_t0)
        FeedForward._op_counts['geglu'] += 1
        
        _t0 = _time_start_op()
        x = self.net[1](x)  # Dropout
        x = self.net[2](x)  # Linear out
        FeedForward._op_times['linear_out'] += _time_end_op(_t0)
        FeedForward._op_counts['linear_out'] += 1
        
        return x


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv,
                            'b (qkv heads c) h w -> qkv b heads c (h w)',
                            heads=self.heads,
                            qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out,
                        'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads,
                        h=h,
                        w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_
