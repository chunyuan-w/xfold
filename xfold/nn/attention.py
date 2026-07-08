# Copyright 2025 Xflops
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import einops
import torch
import torch.nn as nn

from xfold import fastnn
from xfold.fastnn import config as fastnn_config
from af3_kernels import GridSelfAttentionCpp, DistributedGridSelfAttentionCpp
from af3_kernels.tools import profile

try:
    import sgl_kernel  # noqa: F401
    _HAS_SGL_KERNEL = True
except ImportError:
    _HAS_SGL_KERNEL = False


def pack_sgl_weights(module: nn.Module):
    """Pack weights for all LinearSGL layers in a module tree.
    
    MUST call this after loading model weights. This replaces unpacked weights with packed
    weights in-place, enabling efficient kernel execution without runtime weight transformation.
    
    Order: concat fused weights first (on unpacked data), then pack everything.
    
    Args:
        module: The model or submodule to pack weights for.
    """
    # Packing builds throwaway nn.Linear modules in the concat passes below
    # (concat_proj_gate_weights / concat_qkvg_weights); their default kaiming init
    # draws from the global torch RNG. pack_sgl_weights() runs AFTER the per-seed
    # torch.manual_seed() in ModelRunner.run_inference(), so without this guard an SGL
    # run advances the global RNG more than a non-SGL run (e.g. the torch reference)
    # and desyncs every later draw -- specifically:
    #   - xfold.nn.featurization.shuffle_msa() -> gumbel_argsort_sample_idx() ->
    #     gumbel_noise() -> torch.rand(): the per-recycle MSA subsampling, run
    #     num_recycles times inside Evoformer.forward (alphafold3.py).
    #   - AlphaFold3._sample_diffusion() (alphafold3.py): the initial noise
    #     `positions = torch.randn((num_samples,) + mask.shape + (3,))`, plus the
    #     per-step torch.randn() in _sample_step()/random_augmentation().
    # Net effect without the guard: identically-seeded SGL vs reference runs pick
    # different MSA rows and different diffusion noise, masking
    # the bf16 kernel numerics we actually want to measure. Save/restore the RNG state
    # so packing is RNG-neutral and those draws stay bit-identical across runs.
    _rng_state = torch.get_rng_state()
    try:
        # First pass: concat unpacked weights into fused weights (e.g. qkvg_weight).
        for submodule in module.modules():
            if hasattr(submodule, 'concat_qkvg_weights') and callable(submodule.concat_qkvg_weights):
                submodule.concat_qkvg_weights()
            if hasattr(submodule, 'concat_proj_gate_weights') and callable(submodule.concat_proj_gate_weights):
                submodule.concat_proj_gate_weights()
        # Second pass: pack all LinearSGL weights (including the newly created fused ones).
        for submodule in module.modules():
            if hasattr(submodule, 'pack_weight') and callable(submodule.pack_weight):
                submodule.pack_weight()
    finally:
        torch.set_rng_state(_rng_state)


def dot_product_attention_sgl(q: torch.Tensor,
                              k: torch.Tensor,
                              v: torch.Tensor,
                              mask: torch.Tensor | None = None,
                              bias: torch.Tensor | None = None):
    if not _HAS_SGL_KERNEL:
        raise ImportError(
            "sgl_kernel is required when grid_self_attention_implementation='sgl'"
        )

    B, H, N, D = q.shape
    q_flat = q.transpose(1, 2).reshape(B * N, H, D)
    k_flat = k.transpose(1, 2).reshape(B * N, H, D)
    v_flat = v.transpose(1, 2).reshape(B * N, H, D)

    cu_seqlens_q = torch.arange(0, B * N + 1, step=N, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.arange(0, B * N + 1, step=N, dtype=torch.int32, device=q.device)

    # TODO: support bias is None case
    out_flat = torch.ops.sgl_kernel.flash_attn_varlen_func(
        q_flat,
        k_flat,
        v_flat,
        bias.contiguous(),
        cu_seqlens_q,
        cu_seqlens_k,
        N,
        N,
        False,
    )
    return out_flat.reshape(B, N, H, D).transpose(1, 2)


class GridSelfAttentionTorch(nn.Module):
    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttentionTorch, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_bias_projection = nn.Linear(
            self.c_pair, self.num_head, bias=False)

        self.q_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.k_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.v_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.gating_query = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)

    @profile()
    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])

        weighted_avg = fastnn.dot_product_attention(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        gate_values = self.gating_query(pair)

        weighted_avg *= torch.sigmoid(gate_values)
        return self.output_projection(weighted_avg)

    @profile("GridSelfAttention")
    def forward(self, pair, mask):
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        pair = self._attention(pair, mask, nonbatched_bias)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class LayerNormSGL(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super(LayerNormSGL, self).__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)

    def forward(self, x):
        x_shapes = x.shape
        # layernorm_cpu writes into its input. Clone so GSA normalization does not
        # clobber the caller's residual tensor used by pair += attention(pair).
        if len(x_shapes) == 3:
            x = x.reshape(-1, x.shape[-1]).clone()
        else:
            x = x.clone()
        torch.ops.sgl_kernel.layernorm_cpu(
            x, self.weight, self.bias, self.eps
        )
        if len(x_shapes) == 3:
            x = x.view(x_shapes[0], x_shapes[1], -1)
        return x


class LinearSGL(torch.nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(LinearSGL, self).__init__(in_features, out_features, bias, device, dtype)

    def pack_weight(self):
        weight = self.weight.data
        packed_weight = torch.nn.Parameter(
            torch.ops.sgl_kernel.convert_weight_packed(weight),
            requires_grad=False,
        )        
        self.weight = packed_weight
        
        if hasattr(self, "bias") and self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias.data.float(), requires_grad=False)

    def forward(self, x):
        x_shapes = x.shape
        # TODO: reshape introduces extra memory copy here
        # x is from previous TPP triangle multiplication kernel and has been padded. The size is [81,81,64] but stride is [81*128, 64, 1]
        # directly view on this tensor will fail.        
        if len(x_shapes) == 3:
            x = x.reshape(-1, x.shape[-1])
        output = torch.ops.sgl_kernel.weight_packed_linear(
            x,
            self.weight,  # Use packed weight directly
            self.bias if hasattr(self, "bias") else None,
            True,  # is_vnni
        )
        if len(x_shapes) == 3:
            output = output.view(x_shapes[0], x_shapes[1], -1)
        return output


class GridSelfAttentionSGL(nn.Module):

    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super(GridSelfAttentionSGL, self).__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        # Keep the same construction style as GridSelfAttentionTorch.
        # Replace canonical modules with SGL-backed modules so weight loading
        # targets the same attrs used at runtime.
        self.act_norm = LayerNormSGL(self.c_pair)
        self.pair_bias_projection = LinearSGL(self.c_pair, self.num_head, bias=False)

        self.q_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.k_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.v_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        
        # TODO: where to concat qkv? also post process?

        self.gating_query = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.output_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)

    @profile()
    def _attention(self, pair: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor):
        q = self.q_projection(pair)
        k = self.k_projection(pair)
        v = self.v_projection(pair)

        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])

        weighted_avg = dot_product_attention_sgl(q, k, v,
                                                 mask=mask,
                                                 bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

        x_shapes = pair.shape
        if len(x_shapes) == 3:
            pair = pair.reshape(-1, pair.shape[-1])
        mul_shapes = weighted_avg.shape
        if len(mul_shapes) == 3:
            weighted_avg = weighted_avg.reshape(-1, pair.shape[-1])

        weighted_avg = torch.ops.sgl_kernel.weight_packed_linear_sigmoid_mul(
            pair,
            self.gating_query.weight,  # Use packed weight directly
            self.gating_query.bias,
            weighted_avg,
            True,
            True,
        )
        if len(x_shapes) == 3:
            weighted_avg = weighted_avg.reshape(x_shapes[0], x_shapes[1], -1)

        return self.output_projection(weighted_avg)

    @profile("GridSelfAttention")
    def forward(self, pair, mask):
        pair = self.act_norm(pair)
        nonbatched_bias = self.pair_bias_projection(pair).permute(2, 0, 1)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        pair = self._attention(pair, mask, nonbatched_bias)

        if self.transpose:
            pair = pair.permute(1, 0, 2)

        return pair


class GridSelfAttentionFusedSGL(nn.Module):
    """One QKVG concat GEMM + fused attention-tail + out_proj (sgl-kernel fused_grid_attention)."""

    def __init__(self, c_pair: int = 128, num_head: int = 4, transpose: bool = False):
        super().__init__()
        self.c_pair = c_pair
        self.num_head = num_head
        self.qkv_dim = self.c_pair // self.num_head
        self.transpose = transpose

        self.act_norm = LayerNormSGL(self.c_pair)
        self.pair_bias_projection = LinearSGL(self.c_pair, self.num_head, bias=False)

        # Individual projections for weight loading compatibility.
        self.q_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.k_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.v_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.gating_query = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.output_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)

    def concat_qkvg_weights(self):
        """Concat Q/K/V/G weights into a single QKVG LinearSGL.
        
        Must be called before pack_weight() so that the concat is done on unpacked
        weights. pack_sgl_weights() will then pack qkvg_projection.weight via
        the normal LinearSGL.pack_weight() walk.
        """
        qkvg_linear = LinearSGL(self.c_pair, 4 * self.c_pair, bias=False)
        qkvg_linear.weight = torch.nn.Parameter(
            torch.cat(
                [
                    self.q_projection.weight.data,
                    self.k_projection.weight.data,
                    self.v_projection.weight.data,
                    self.gating_query.weight.data,
                ],
                dim=0,
            ),
            requires_grad=False,
        )
        self.qkvg_projection = qkvg_linear
        # Remove individual projections to free memory; they're no longer needed.
        del self.q_projection
        del self.k_projection
        del self.v_projection
        del self.gating_query

    @profile("GridSelfAttention")
    def forward(self, pair, mask):
        # fused_grid_attention has no mask input, so this backend silently
        # ignores `mask`. That is only correct when the mask is all-ones, i.e.
        # features are not bucket-padded. Guard the assumption rather than
        # produce wrong results on padded inputs.
        assert not fastnn_config.pad_to_buckets, (
            "grid_self_attention_implementation='sgl' (GridSelfAttentionFusedSGL) "
            "ignores the attention mask and assumes it is all-ones, which only holds "
            "without bucket padding. Run with --pad_to_buckets=False, or use the "
            "'cpp' grid self attention backend."
        )
        pair = self.act_norm(pair)
        bias = self.pair_bias_projection(pair).permute(2, 0, 1).contiguous()
        if self.transpose:
            pair = pair.permute(1, 0, 2).contiguous()

        out = torch.ops.sgl_kernel.fused_grid_attention(
            pair,
            bias,
            self.qkvg_projection.weight,
            self.output_projection.weight,
            self.num_head,
            True,  # is_vnni
        )

        if self.transpose:
            out = out.permute(1, 0, 2)
        return out


if fastnn_config.grid_self_attention_implementation == "cpp":
    if fastnn_config.grid_self_attention_dist:
        GridSelfAttention = DistributedGridSelfAttentionCpp
    else:
        GridSelfAttention = GridSelfAttentionCpp
elif fastnn_config.grid_self_attention_implementation == "sgl":
    GridSelfAttention = GridSelfAttentionFusedSGL
else:
    GridSelfAttention = GridSelfAttentionTorch


class MSAAttention(nn.Module):
    def __init__(self, c_msa=64, c_pair=128, num_head=8):
        super(MSAAttention, self).__init__()

        self.c_msa = c_msa
        self.c_pair = c_pair
        self.num_head = num_head

        self.value_dim = self.c_msa // self.num_head

        self.act_norm = fastnn.LayerNorm(self.c_msa)
        self.pair_norm = fastnn.LayerNorm(self.c_pair)
        self.pair_logits = nn.Linear(self.c_pair, self.num_head, bias=False)
        self.v_projection = nn.Linear(
            self.c_msa, self.num_head * self.value_dim, bias=False)
        self.gating_query = nn.Linear(self.c_msa, self.c_msa, bias=False)
        self.output_projection = nn.Linear(self.c_msa, self.c_msa, bias=False)

    @profile("MSAAttention")
    def forward(self, msa, msa_mask, pair):
        msa = self.act_norm(msa)
        pair = self.pair_norm(pair)
        logits = self.pair_logits(pair)
        logits = logits.permute(2, 0, 1)

        logits += 1e9 * (torch.max(msa_mask, dim=0).values - 1.0)
        weights = torch.softmax(logits, dim=-1)

        v = self.v_projection(msa)
        v = einops.rearrange(v, 'b k (h c) -> b k h c', h=self.num_head)

        v_avg = torch.einsum('hqk, bkhc -> bqhc', weights, v)
        v_avg = torch.reshape(v_avg, v_avg.shape[:-2] + (-1,))

        gate_values = self.gating_query(msa)
        v_avg *= torch.sigmoid(gate_values)

        return self.output_projection(v_avg)
