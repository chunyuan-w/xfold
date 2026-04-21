import argparse
import copy
import einops
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch._inductor import config as inductor_config

inductor_config.profiler_mark_wrapper_call = True
inductor_config.cpp.enable_kernel_profile = True
# TODO: after using weight_packed_linear, cpp wrapper will fail but python wrapper works
# inductor_config.cpp_wrapper = True
# inductor_config.max_autotune = True

torch.manual_seed(1234)


def register_fake_ops():
    @torch.library.register_fake("sgl_kernel::flash_attn_varlen_func")
    def _(
        q,
        k,
        v,
        bias,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    ):
        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        head_size_v = v.shape[2]
        
        return torch.empty(num_tokens, num_heads, head_size_v, device=q.device, dtype=q.dtype)

    @torch.library.register_fake("sgl_kernel::weight_packed_linear")
    def _(x, weight, bias, is_vnni):
        # TODO: in upstream, the below one is used, but seems wrong when x.dim > 2. weight dim should be weight.shape[1] if weight is the prepacked one
        # return x.new_empty(x.shape[0], weight.shape[0])
        out_shape = x.shape[:-1] + (weight.shape[1],)
        print(f"fake: x shape = {x.shape}, w shape = {weight.shape}, out shape = {out_shape}", )

        return x.new_empty(out_shape)

    @torch.library.register_fake("sgl_kernel::fused_grid_attention")
    def _(pair, bias, q_w, k_w, v_w, g_w, o_w, num_heads, is_vnni):
        # Output preserves pair's shape, with last dim = output_weight's logical out-dim.
        out_shape = pair.shape[:-1] + (o_w.shape[1],)
        return pair.new_empty(out_shape)

    @torch.library.register_fake("sgl_kernel::fused_grid_attention_v2")
    def _(pair, bias, qkvg_w, o_w, num_heads, is_vnni):
        out_shape = pair.shape[:-1] + (o_w.shape[1],)
        return pair.new_empty(out_shape)

    @torch.library.register_fake("sgl_kernel::fused_grid_attention_v3")
    def _(pair, bias, q_w, k_w, v_w, g_w, o_w, num_heads, is_vnni):
        out_shape = pair.shape[:-1] + (o_w.shape[1],)
        return pair.new_empty(out_shape)


def dot_product_attention_sglang(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    """
    q, k, v: [B, H, N, D]
    bias: [H, N, N]
    """

    B, H, N, D = q.shape

    # TODO：bias not supported in kernel
    # if bias is not None:
    #     # flash_attn_varlen_func currently does not support additive bias directly
    #     # apply bias manually to v (optional)
    #     v = v + bias.unsqueeze(-1)  # broadcast if necessary

    # ---- flatten batch sequences ----
    q_flat = q.transpose(1, 2).reshape(B*N, H, D)  # [T_total, H, D]
    k_flat = k.transpose(1, 2).reshape(B*N, H, D)
    v_flat = v.transpose(1, 2).reshape(B*N, H, D)

    # TODO: exclude this from time benchmark
    # cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, B*N + 1, step=N, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.arange(0, B*N + 1, step=N, dtype=torch.int32, device=q.device)

    # For example, B = 2, N = 4, cu_seqlens_q: [0, 4, 8]
    # BS 0: seq[0]:seq[4]
    # BS 1: seq[4]:seq[8]

    # ---- call fused kernel ----
    # breakpoint()
    # TODO: support non contiguous bias
    bias = bias.contiguous()
    out_flat = torch.ops.sgl_kernel.flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        bias, # comment out to test no bias case
        cu_seqlens_q,
        cu_seqlens_k,
        N,
        N,
        False, # is_causal
    )

    # ---- reshape back ----
    out = out_flat.reshape(B, N, H, D).transpose(1, 2)  # [B, H, N, D]

    return out



def dot_product_attention_sdpa_slice(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,    
):
    # TODO: this is bad for perf, but sdpa does not accept out buffer
    out = torch.zeros_like(q)
    
    # Derive valid length from mask
    if mask is not None:
        mask_index = int(mask.any(dim=0).sum().item())
    else:
        mask_index = N
    
    # Handle bias
    bias = bias.unsqueeze(0)
    bias = bias[:, :, :mask_index, :mask_index]
    
    # TODO: for batch dim (dim 0), do we need to do slice??
    q = q[:mask_index, :, :mask_index, :]
    k = k[:mask_index, :, :mask_index, :]
    v = v[:mask_index, :, :mask_index, :]

    attn_out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias,
        dropout_p=0.0,
        is_causal=False,
    )
    # TODO: this is bad for perf, but sdpa does not accept out buffer
    out[:mask_index,:,:mask_index,:] = attn_out
    
    return out

def dot_product_attention_sdpa_no_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    # Handle bias (additive attention bias)
    if bias is not None:
        bias = bias.unsqueeze(0)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias, # comment out to test no bias case
        dropout_p=0.0,
        is_causal=False,
    )

def dot_product_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    attn_mask = None

    # Handle mask (key padding mask)
    if mask is not None:
        if mask.dim() == 1:
            # (K,) -> (1,1,1,K)
            mask = mask[None, None, None, :]
        elif mask.dim() == 2:
            # (B,K) -> (B,1,1,K)
            mask = mask[:, None, None, :]

        mask = mask.to(dtype=torch.bool)
        attn_mask = mask  # bool mask

    # Handle bias (additive attention bias)
    if bias is not None:
        if attn_mask is None:
            attn_mask = bias
        else:
            # merge bool mask + additive bias
            # masked positions → -inf
            # TODO: -inf or -1e9
            attn_mask = bias.masked_fill(~attn_mask, float("-inf"))
            # attn_mask = bias.masked_fill(~attn_mask, -1e9)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )


def dot_product_attention_torch_no_mask(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    # breakpoint()

    logits = torch.matmul(q, k.transpose(-1, -2))

    # TODO: bias not supported in sglang, comment out for test for now
    if bias is not None:
        logits += bias

    # if mask is not None:
    #     if mask.dim() == 1:
    #         mask = mask[None, None, None, :].to(dtype=torch.bool)
    #     elif mask.dim() == 2:
    #         mask = mask[:, None, None, :].to(dtype=torch.bool)
    #     logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)


def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    # breakpoint()

    logits = torch.matmul(q, k.transpose(-1, -2))

    if bias is not None:
        logits += bias

    if mask is not None:
        if mask.dim() == 1:
            mask = mask[None, None, None, :].to(dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask[:, None, None, :].to(dtype=torch.bool)
        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)


class GridSelfAttentionTorch(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttentionTorch, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = nn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor, small_ops, torch_sdpa):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        # breakpoint()
        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])
        
        # breakpoint()
        # TODO: for debug
        # q = torch.ones_like(q)
        # k = torch.ones_like(k)
        # v = torch.ones_like(v)
        # print(bias)
        # bias = torch.zeros_like(bias)
        # breakpoint()

        if small_ops:
            # sdpa_func = dot_product_attention_torch
            sdpa_func = dot_product_attention_torch_no_mask
        else:
            if torch_sdpa:
                sdpa_func = dot_product_attention_sdpa_no_mask
            else:
                # sdpa_func = dot_product_attention_sdpa
                # sdpa_func = dot_product_attention_sdpa_no_mask
                sdpa_func = dot_product_attention_sglang
        # sdpa_func = dot_product_attention_torch
        # sdpa_func = dot_product_attention_sdpa
        weighted_avg = sdpa_func(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)

        weighted_avg *= torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(self, pair, mask, small_ops=False, torch_sdpa=False):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        # breakpoint()
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)
        # breakpoint()
        if self.transpose:
            pair = pair.permute(1, 0, 2)

        # nonbatched_bias = torch.zeros_like(nonbatched_bias)

        pair = self._attention(pair, mask, nonbatched_bias, small_ops, torch_sdpa)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class LayerNormSGL(nn.Module):
    def __init__(self, m):
        super(LayerNormSGL, self).__init__()
        
        weight = torch.nn.Parameter(
            m.weight.data,
            requires_grad=False,
        )
        self.weight = weight
        self.variance_epsilon = m.eps

    def forward(self, x):
        x_shapes = x.shape
        if len(x_shapes) == 3:
            x = x.view(-1, x.shape[-1])             
        # the output is directly written into x
        torch.ops.sgl_kernel.layernorm_cpu(
            x, self.weight.data, self.variance_epsilon
        )
        if len(x_shapes) == 3:
            x = x.view(x_shapes[0], x_shapes[1], -1)
        return x        


class LinearSGL(nn.Module):
    def __init__(self, m):
        super(LinearSGL, self).__init__()
        if hasattr(m, "bias") and m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.data.float(), requires_grad=False)
        
        weight = m.weight.data
        
        # pack weight
        import sgl_kernel
        
        packed_weight = torch.nn.Parameter(
            torch.ops.sgl_kernel.convert_weight_packed(weight),
            requires_grad=False,
        )
        # packed_weight.__dict__ = weight.__dict__
        self.weight = packed_weight

    def forward(self, x):
        x_shapes = x.shape
        if len(x_shapes) == 3:
            x = x.view(-1, x.shape[-1])        
        output = torch.ops.sgl_kernel.weight_packed_linear(
            x,
            self.weight,
            self.bias if hasattr(self, "bias") else None,
            True,  # is_vnni
        )
        if len(x_shapes) == 3:
            output = output.view(x_shapes[0], x_shapes[1], -1)
        return output

class GridSelfAttentionSGL(nn.Module):
    def __init__(self, m):
        super(GridSelfAttentionSGL, self).__init__()
        self.c_pair = m.c_pair
        self.num_head = m.num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = m.transpose

        # TODO: LayerNormSGL does not support bias
        self.act_norm = copy.deepcopy(m.act_norm)
        # self.act_norm = LayerNormSGL(m.act_norm)
        self.pair_bias_projection = LinearSGL(m.pair_bias_projection)

        # concat the weight of q_projection, k_projection and v_projection
        # TODO: add check: in_features and out_features all equal; no bias for all of them
        qkv_projection = nn.Linear(m.q_projection.in_features, m.q_projection.out_features * 3, bias=m.q_projection.bias)
        concat_weight = torch.nn.Parameter(
            torch.cat([
                m.q_projection.weight,
                m.k_projection.weight,
                m.v_projection.weight
            ], dim=0),
            requires_grad=False,          
        )
        qkv_projection.weight = concat_weight
        self.qkv_projection = LinearSGL(qkv_projection)
        
        # self.q_projection = LinearSGL(m.q_projection)
        # self.k_projection = LinearSGL(m.k_projection)
        # self.v_projection = LinearSGL(m.v_projection)

        self.gating_query = LinearSGL(m.gating_query)
        self.output_projection = LinearSGL(m.output_projection)

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor, small_ops, torch_sdpa):
        # q = self.q_projection(pair)
        # k = self.k_projection(pair)
        # v = self.v_projection(pair)
        
        qkv = self.qkv_projection(pair)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # breakpoint()
        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])
        
        sdpa_func = dot_product_attention_sglang
        weighted_avg = sdpa_func(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        # TODO: wrap view into a module
        x_shapes = pair.shape
        if len(x_shapes) == 3:
            pair = pair.view(-1, pair.shape[-1])
        mul_shapes = weighted_avg.shape
        if len(mul_shapes) == 3:
            weighted_avg = weighted_avg.view(-1, pair.shape[-1])
        weighted_avg = torch.ops.sgl_kernel.weight_packed_linear_sigmoid_mul(
            pair,
            self.gating_query.weight,
            self.gating_query.bias if hasattr(self.gating_query, "bias") else None,
            weighted_avg,
            True,  # inplace
            True,  # is_vnni
        )
        if len(x_shapes) == 3:
            weighted_avg = weighted_avg.view(x_shapes[0], x_shapes[1], -1)
        # gate_values = self.gating_query(pair)
        
        # weighted_avg *= torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(self, pair, mask, small_ops=False, torch_sdpa=False):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        # breakpoint()
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)
        # breakpoint()
        if self.transpose:
            pair = pair.permute(1, 0, 2)

        # nonbatched_bias = torch.zeros_like(nonbatched_bias)

        pair = self._attention(pair, mask, nonbatched_bias, small_ops, torch_sdpa)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class GridSelfAttentionFusedSGL(nn.Module):
    """Calls the single fused CPU kernel (QKV + flash attn + gate + output proj)."""

    def __init__(self, m):
        super().__init__()
        self.c_pair = m.c_pair
        self.num_head = m.num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = m.transpose

        self.act_norm = copy.deepcopy(m.act_norm)
        self.pair_bias_projection = LinearSGL(m.pair_bias_projection)

        import sgl_kernel  # ensure the op library is loaded

        def pack(weight: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(weight.data),
                requires_grad=False,
            )

        self.q_weight = pack(m.q_projection.weight)
        self.k_weight = pack(m.k_projection.weight)
        self.v_weight = pack(m.v_projection.weight)
        self.gating_weight = pack(m.gating_query.weight)
        self.output_weight = pack(m.output_projection.weight)

    def forward(self, pair, mask, small_ops=False, torch_sdpa=False):
        pair = self.act_norm(pair)
        bias = self.pair_bias_projection(pair).permute(2, 0, 1).contiguous()
        if self.transpose:
            pair = pair.permute(1, 0, 2).contiguous()

        # Treat the leading [N, N, C] as [B=N, N, C] so the fused op's [B, N, D]
        # contract matches our grid-self-attention input.
        out = torch.ops.sgl_kernel.fused_grid_attention(
            pair,
            bias,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.gating_weight,
            self.output_weight,
            self.num_head,
            True,  # is_vnni
        )

        if self.transpose:
            out = out.permute(1, 0, 2)
        return out


class GridSelfAttentionFusedSGLv2(nn.Module):
    """A+B split: one QKVG concat GEMM + fused attention-tail + out_proj."""

    def __init__(self, m):
        super().__init__()
        self.c_pair = m.c_pair
        self.num_head = m.num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = m.transpose

        self.act_norm = copy.deepcopy(m.act_norm)
        self.pair_bias_projection = LinearSGL(m.pair_bias_projection)

        import sgl_kernel  # ensure the op library is loaded

        # Build one concat weight [4*D, D] = cat(Q | K | V | G) and pack it so
        # stage A reads pair once per projection-group.
        qkvg_weight_unpacked = torch.cat(
            [
                m.q_projection.weight.data,
                m.k_projection.weight.data,
                m.v_projection.weight.data,
                m.gating_query.weight.data,
            ],
            dim=0,
        )
        self.qkvg_weight = torch.nn.Parameter(
            torch.ops.sgl_kernel.convert_weight_packed(qkvg_weight_unpacked),
            requires_grad=False,
        )
        self.output_weight = torch.nn.Parameter(
            torch.ops.sgl_kernel.convert_weight_packed(m.output_projection.weight.data),
            requires_grad=False,
        )

    def forward(self, pair, mask, small_ops=False, torch_sdpa=False):
        pair = self.act_norm(pair)
        bias = self.pair_bias_projection(pair).permute(2, 0, 1).contiguous()
        if self.transpose:
            pair = pair.permute(1, 0, 2).contiguous()

        out = torch.ops.sgl_kernel.fused_grid_attention_v2(
            pair,
            bias,
            self.qkvg_weight,
            self.output_weight,
            self.num_head,
            True,  # is_vnni
        )

        if self.transpose:
            out = out.permute(1, 0, 2)
        return out


class GridSelfAttentionFusedSGLv3(nn.Module):
    """v1 layout (per-head tiled projections) + v2 attention core (full logits)."""

    def __init__(self, m):
        super().__init__()
        self.c_pair = m.c_pair
        self.num_head = m.num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = m.transpose

        self.act_norm = copy.deepcopy(m.act_norm)
        self.pair_bias_projection = LinearSGL(m.pair_bias_projection)

        import sgl_kernel  # ensure the op library is loaded

        def pack(weight: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(weight.data),
                requires_grad=False,
            )

        self.q_weight = pack(m.q_projection.weight)
        self.k_weight = pack(m.k_projection.weight)
        self.v_weight = pack(m.v_projection.weight)
        self.gating_weight = pack(m.gating_query.weight)
        self.output_weight = pack(m.output_projection.weight)

    def forward(self, pair, mask, small_ops=False, torch_sdpa=False):
        pair = self.act_norm(pair)
        bias = self.pair_bias_projection(pair).permute(2, 0, 1).contiguous()
        if self.transpose:
            pair = pair.permute(1, 0, 2).contiguous()

        out = torch.ops.sgl_kernel.fused_grid_attention_v3(
            pair,
            bias,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.gating_weight,
            self.output_weight,
            self.num_head,
            True,  # is_vnni
        )

        if self.transpose:
            out = out.permute(1, 0, 2)
        return out


def main(use_torch, torch_compile, use_fused, use_fused2, use_fused3):
    c_pair = 128
    num_head = 4

    # TODO: test transpose=True
    if use_torch:
        import sgl_kernel

        # TODO: if we import from sglang, no need to register here but need to register in sglang
        if torch_compile:
            register_fake_ops()

        m_ref = GridSelfAttentionTorch(c_pair=c_pair, num_head=num_head)
        m_ref = m_ref.to(torch.bfloat16)
        m_ref.eval()
        with torch.no_grad():
            # Keep LayerNorm affine params non-trivial in the baseline UT path.
            m_ref.act_norm.weight.copy_(1.0 + 0.1 * torch.randn_like(m_ref.act_norm.weight))
            m_ref.act_norm.bias.copy_(0.2 + 0.1 * torch.randn_like(m_ref.act_norm.bias))

        if use_fused3:
            m = GridSelfAttentionFusedSGLv3(m_ref)
        elif use_fused2:
            m = GridSelfAttentionFusedSGLv2(m_ref)
        elif use_fused:
            m = GridSelfAttentionFusedSGL(m_ref)
        else:
            m = GridSelfAttentionSGL(m_ref)
            
    else:
        # import xfold
        from af3_kernels import GridSelfAttentionCpp
        m = GridSelfAttentionCpp(c_pair=c_pair, num_head=num_head)
        m = m.to(torch.bfloat16)

    # TODO: add correctness check
    m.eval()
    print("done model creation")


    # N_token = 384
    # N_token = 5120
    # N_token = 1024

    # N_token = 8
    # original_N_token = 8
    # original_N_token = 256

    # N_token = 384
    
    # N_token = 1896
    # N_token = 3469
    N_token = 4655
    
    # original_N_token = 384

    # N_token = 2048
    # original_N_token = 1896

    # N_token = 1896
    # original_N_token = 1896

    # TODO: change according to shape
    warmup = 20 if N_token <= 1024 else 10
    measure = 50 if N_token <= 1024 else 20

    pair = torch.randn(N_token, N_token, c_pair, dtype=torch.bfloat16)

    mask = torch.ones(N_token, N_token, dtype=torch.bfloat16)
    
    assert torch.all(mask == 1)
    print("done tensor creation")

    with torch.no_grad():
        if use_torch:
            # TODO: use fused sdpa when size is too large
            # y_small_ops = m(pair, mask, small_ops = True)
            y_ref_sdpa = m_ref(pair, mask, torch_sdpa = True)
            # torch.testing.assert_close(y_small_ops, y_ref_sdpa, atol=1e-2, rtol=1e-2)
            

        if torch_compile:
            m = torch.compile(m)
            # run once to trigger compile
            _ = m(pair, mask)       
        
        if use_torch:
            y_fused_sdpa = m(pair, mask)
            # y_fused_sdpa_slice = m(pair, mask, slice=True)

            # print(y_small_ops[256][0][6])
            # print(y_fused_sdpa[256][0][6])
            
            # print(y_small_ops)
            # print(y_fused_sdpa)

            torch.testing.assert_close(y_ref_sdpa, y_fused_sdpa, atol=1e-2, rtol=1e-2)
            # torch.testing.assert_close(y_fused_sdpa, y_fused_sdpa_slice, atol=1e-2, rtol=1e-2)
        # else:
        #     y_tpp = m(pair, mask)

        # TODO: use which func as ref for xfold kernel?
        #     m_ref = GridSelfAttentionTorch(c_pair=c_pair, num_head=num_head).to(torch.bfloat16).eval()
        #     y_ref = m_ref(pair, mask, small_ops = True)

        #     # Mismatched elements: 1359815 / 2097152 (64.8%)
        #     # Greatest absolute difference: 0.16796875 at index (59, 5, 83) (up to 0.01 allowed)
        #     # Greatest relative difference: 425984.0 at index (24, 59, 27) (up to 0.01 allowed)
                        
        #     torch.testing.assert_close(y_ref, y_tpp, atol=1e-2, rtol=1e-2)
        
        for _ in range(warmup):
            y = m(pair, mask)
        
        record_shapes = True
        
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU], record_shapes=record_shapes) as prof:
            y = m(pair, mask)
        
        print(prof.key_averages(group_by_input_shape=record_shapes).table(sort_by="self_cpu_time_total"))

        run_times = []
        for run_idx in range(measure):
            start = time.perf_counter()
            y = m(pair, mask)
            end = time.perf_counter()
            run_times.append(end - start)

        mean_time = sum(run_times) / len(run_times)
        print("per-run times (s):")
        print(", ".join(f"{run_idx + 1}:{run_time:.6f}" for run_idx, run_time in enumerate(run_times)))
        print(
            "time summary (s): "
            f"avg={mean_time:.6f}, min={min(run_times):.6f}, max={max(run_times):.6f}"
        )

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', action='store_true')
    parser.add_argument('--torch-compile', action='store_true')
    parser.add_argument('--fused', action='store_true',
                        help='use the single fused CPU kernel (QKV + flash attn + gate + output proj)')
    parser.add_argument('--fused2', action='store_true',
                        help='A+B split: concat QKVG GEMM + fused attn-tail + out_proj')
    parser.add_argument('--fused3', action='store_true',
                        help='v1 per-head tiled projections + v2 full-logit attn core (no 22 GB qkvg)')
    args = parser.parse_args()

    main(args.torch, args.torch_compile, args.fused, args.fused2, args.fused3)