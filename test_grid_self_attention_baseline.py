import einops
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch._inductor import config as inductor_config

inductor_config.profiler_mark_wrapper_call = True
inductor_config.cpp.enable_kernel_profile = True

torch_compile = True

torch.manual_seed(1234)


# flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func  # your kernel

# def dot_product_attention_flash(q: torch.Tensor,
#                                 k: torch.Tensor,
#                                 v: torch.Tensor,
#                                 mask: Optional[torch.Tensor] = None,
#                                 bias: Optional[torch.Tensor] = None):
#     """
#     q, k, v: [B, H, N, D]
#     mask: [B, 1, 1, N] or [B, 1, N, N]
#     bias: broadcastable additive bias
#     """

#     B, H, N, D = q.shape

#     # ---- handle mask and bias ----
#     if mask is not None:
#         # convert to boolean and flatten
#         mask = mask.bool()
#         # TODO: SDPA had True=masked; flash_attn_varlen_func does not support mask yet?
#         # If your kernel does not support masks, you may need to manually zero-out values or skip masking
#         # For simplicity, we'll assume mask=None for now

#     if bias is not None:
#         # flash_attn_varlen_func currently does not support additive bias directly
#         # apply bias manually to v (optional)
#         v = v + bias.unsqueeze(-1)  # broadcast if necessary

#     # ---- flatten batch sequences ----
#     q_flat = q.transpose(1, 2).reshape(B*N, H, D)  # [T_total, H, D]
#     k_flat = k.transpose(1, 2).reshape(B*N, H, D)
#     v_flat = v.transpose(1, 2).reshape(B*N, H, D)

#     # cumulative sequence lengths
#     cu_seqlens_q = torch.arange(0, B*N + 1, step=N, dtype=torch.int32, device=q.device)
#     cu_seqlens_k = torch.arange(0, B*N + 1, step=N, dtype=torch.int32, device=q.device)

#     # ---- call fused kernel ----
#     out_flat = flash_attn_varlen_func(
#         q_flat,
#         k_flat,
#         v_flat,
#         cu_seqlens_q,
#         cu_seqlens_k,
#         max_seqlen_q=N,
#         max_seqlen_k=N,
#         is_causal=False,
#     )

#     # ---- reshape back ----
#     out = out_flat.reshape(B, N, H, D).transpose(1, 2)  # [B, H, N, D]

#     return out

def dot_product_attention_sdpa(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    attn_mask = None

    if mask is not None:
        # Convert to SDPA format (True = masked)
        if mask.dim() == 1:
            mask = mask[None, None, None, :]
        elif mask.dim() == 2:
            mask = mask[:, None, None, :]
        attn_mask = ~mask.bool()   # SDPA expects True = mask-out

    if bias is not None:
        # SDPA supports additive bias as attn_mask (float)
        if attn_mask is None:
            attn_mask = bias
        else:
            attn_mask = attn_mask + bias

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False
    )


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

    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor, small_ops):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        # breakpoint()
        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])
        
        # breakpoint()

        # sdpa_func = dot_product_attention_torch if small_ops else dot_product_attention_sdpa
        # sdpa_func = dot_product_attention_torch
        # sdpa_func = dot_product_attention_sdpa
        weighted_avg = dot_product_attention_sdpa(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)

        weighted_avg *= torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    def forward(self, pair, mask, small_ops=True):
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

        pair = self._attention(pair, mask, nonbatched_bias, small_ops)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair



# TODO: add correctness check
# TODO: add bench time

# TODO: test transpose=True

m = GridSelfAttentionTorch().to(torch.bfloat16)
m.eval()

warmup = 5
measure = 5
print("done model creation")
# N_token = 384
# N_token = 5120

# N_token = 1024
N_token = 384

c_pair = 128
pair = torch.randn(N_token, N_token, c_pair, dtype=torch.bfloat16)
mask = torch.randn(N_token, N_token, dtype=torch.bfloat16)
print("done tensor creation")

with torch.no_grad():

    # y_small_ops = m(pair, mask)
    y_fused_sdpa = m(pair, mask, small_ops=False)

    # print(y_small_ops[358][339][67])
    # print(y_fused_sdpa[358][339][67])
    
    # print(y_small_ops[21][234][36])
    # print(y_fused_sdpa[21][234][36])
    # torch.testing.assert_close(y_small_ops, y_fused_sdpa, atol=1e-2, rtol=1e-2)

    if torch_compile:
        m = torch.compile(m)
    
    for _ in range(warmup):
        y = m(pair, mask)
    
        # y_xfold = xfold_m(pair, mask)
    print("done 1st run")

    record_shapes = True
    
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=record_shapes) as prof:
        y = m(pair, mask)
        # y_xfold = xfold_m(pair, mask)
    
    print(prof.key_averages(group_by_input_shape=record_shapes).table(sort_by="self_cpu_time_total"))

    start = time.time()
    for _ in range(measure):
        y = m(pair, mask)
    
        # y_xfold = xfold_m(pair, mask)        
    end = time.time()
    
    print(f"time used: {(end - start) / measure} s")

print("done")