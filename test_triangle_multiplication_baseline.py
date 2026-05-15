import argparse
import copy
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
    @torch.library.register_fake("sgl_kernel::weight_packed_linear")
    def _(x, weight, bias, is_vnni):
        out_shape = x.shape[:-1] + (weight.shape[1],)
        print(f"fake: x shape = {x.shape}, w shape = {weight.shape}, out shape = {out_shape}", )
        return x.new_empty(out_shape)

    @torch.library.register_fake("sgl_kernel::fused_triangle_multiplication")
    def _(pair_orig, pair_normed, mask, proj_gate_w,
          center_norm_w, center_norm_b, out_proj_w, gating_w, outgoing, is_vnni):
        # In-place residual: the op returns pair_orig itself (clobbered).
        return pair_orig


class TriangleMultiplicationTorch(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super().__init__()
        self.c_pair = c_pair

        self.left_norm_input = nn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = nn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation = 'ckj,cki->cij'
        if _outgoing is True:
            self.equation = 'cik,cjk->cij'

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token, N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """
        input = pair
        pair = self.left_norm_input(pair)
        input_pair = pair

        projection = self.projection(pair)
        projection = projection.permute(2, 0, 1)
        if mask is not None:
            projection *= mask[None, ...]

        gate = self.gate(pair)
        gate = gate.permute(2, 0, 1)
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])

        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
        pair = torch.einsum(self.equation, a, b)

        pair = pair.permute(1, 2, 0)
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair *= torch.sigmoid(gate_out)
        input = input + pair
        return input


class LayerNormSGL(nn.Module):
    def __init__(self, m):
        super().__init__()
        weight = torch.nn.Parameter(m.weight.data, requires_grad=False)
        self.weight = weight
        if hasattr(m, "bias") and m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.data, requires_grad=False)
        else:
            self.bias = None
        self.variance_epsilon = m.eps

    def forward(self, x):
        x_shapes = x.shape
        if len(x_shapes) == 3:
            x = x.view(-1, x.shape[-1])
        torch.ops.sgl_kernel.layernorm_cpu(
            x, self.weight.data, self.bias, self.variance_epsilon
        )
        if len(x_shapes) == 3:
            x = x.view(x_shapes[0], x_shapes[1], -1)
        return x


class LinearSGL(nn.Module):
    def __init__(self, m):
        super().__init__()
        if hasattr(m, "bias") and m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.data.float(), requires_grad=False)

        import sgl_kernel  # noqa: F401

        weight = m.weight.data
        packed_weight = torch.nn.Parameter(
            torch.ops.sgl_kernel.convert_weight_packed(weight),
            requires_grad=False,
        )
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


class TriangleMultiplicationSGL(nn.Module):
    """SGL drop-in. Uses `weight_packed_linear_sigmoid_mul` on the two
    sigmoid*mul sites (gate over projection, and the output gating Linear)."""

    def __init__(self, m):
        super().__init__()
        self.c_pair = m.c_pair
        self.equation = m.equation

        import sgl_kernel  # noqa: F401

        def pack(w: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(w.data),
                requires_grad=False,
            )

        # NOTE: left_norm_input intentionally uses torch.nn.LayerNorm (not
        # LayerNormSGL) because LayerNormSGL writes its result in-place into
        # the input buffer. TM has a residual at the end (`input + pair`)
        # where `input` aliases the pre-layernorm tensor, so an in-place
        # left-norm corrupts the residual. center_norm is safe to keep on
        # LayerNormSGL because by then `pair` is a fresh tensor from the
        # einsum + permute + .contiguous().
        # TODO: revisit when LayerNormSGL grows an out-of-place variant (or
        # when we have a fused TM kernel that handles the residual itself).
        self.left_norm_input = copy.deepcopy(m.left_norm_input)
        # `projection` stays as a regular packed linear (mask is applied to its
        # output before it's consumed as `post_mul_mat`).
        self.projection = LinearSGL(m.projection)
        # `gate` is consumed by sigmoid_mul; we only need the packed weight.
        # Dropping the proj/gate concat here: site-1 sigmoid_mul fusion (avoids
        # the ~11 GB gate intermediate at N=4655) beats sharing a GEMM with proj.
        self.gate_weight = pack(m.gate.weight)
        self.center_norm = LayerNormSGL(m.center_norm)
        self.output_projection = LinearSGL(m.output_projection)
        self.gating_linear_weight = pack(m.gating_linear.weight)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input = pair
        pair = self.left_norm_input(pair)
        input_pair = pair

        N1, N2, C = pair.shape
        M = N1 * N2
        twoC = 2 * C

        projection = self.projection(pair)            # [N, N, 2C]
        if mask is not None:
            projection = projection * mask.unsqueeze(-1)

        # Site 1: projection *= sigmoid(gate(pair))  fused.
        pair_flat = pair.reshape(M, C)
        proj_flat = projection.reshape(M, twoC)
        proj_flat = torch.ops.sgl_kernel.weight_packed_linear_sigmoid_mul(
            pair_flat,
            self.gate_weight,
            None,
            proj_flat,
            True,  # inplace
            True,  # is_vnni
        )
        projection = proj_flat.view(N1, N2, twoC)

        projection = projection.permute(2, 0, 1)
        projection = projection.reshape(C, 2, *projection.shape[1:])
        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
        pair = torch.einsum(self.equation, a, b)

        pair = pair.permute(1, 2, 0).contiguous()
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)         # [N, N, C]

        # Site 2: pair *= sigmoid(gating_linear(input_pair))  fused.
        pair_flat = pair.reshape(M, C)
        input_pair_flat = input_pair.reshape(M, C)
        pair_flat = torch.ops.sgl_kernel.weight_packed_linear_sigmoid_mul(
            input_pair_flat,
            self.gating_linear_weight,
            None,
            pair_flat,
            True,  # inplace
            True,  # is_vnni
        )
        pair = pair_flat.view(N1, N2, C)

        input = input + pair
        return input


class TriangleMultiplicationFusedSGL(nn.Module):
    """Single fused CPU kernel: pre_einsum + einsum + center_norm + post_einsum,
    matching the v2 grid-attention design (skill: af3-fused-cpu-kernel-perf).

    Wrapper contract (see fused_triangle_multiplication.cpp):
      - left_norm is applied here in Python (out-of-place); the kernel sees
        both pair (= input, also output buffer, clobbered in place) and
        pair_normed (= left_norm(pair)).
      - All weights are pre-packed via convert_weight_packed.
      - The op writes the residual-summed result into `pair` and returns it.
    """

    def __init__(self, m):
        super().__init__()
        self.c_pair = m.c_pair
        self._outgoing = (m.equation == 'cik,cjk->cij')

        import sgl_kernel  # noqa: F401

        def pack(w: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(w.data),
                requires_grad=False,
            )

        # Layer norms stay as out-of-place torch ops (v2 convention).
        self.left_norm_input = copy.deepcopy(m.left_norm_input)
        self.center_norm_weight = torch.nn.Parameter(
            m.center_norm.weight.data.clone(), requires_grad=False)
        if m.center_norm.bias is not None:
            self.center_norm_bias = torch.nn.Parameter(
                m.center_norm.bias.data.clone(), requires_grad=False)
        else:
            self.center_norm_bias = None

        # Concat proj | gate along the output dim -> [4C, C], then pack once.
        proj_gate_unpacked = torch.cat(
            [m.projection.weight.data, m.gate.weight.data], dim=0)  # [4C, C]
        self.proj_gate_weight = pack(proj_gate_unpacked)
        self.out_proj_weight = pack(m.output_projection.weight.data)
        self.gating_weight = pack(m.gating_linear.weight.data)

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Left norm out-of-place; pair stays unchanged for the in-kernel residual.
        pair_normed = self.left_norm_input(pair)
        # Ensure mask is bf16 contiguous as the kernel expects.
        if mask.dtype != pair.dtype:
            mask = mask.to(pair.dtype)
        if not mask.is_contiguous():
            mask = mask.contiguous()
        return torch.ops.sgl_kernel.fused_triangle_multiplication(
            pair,                      # pair_orig (in/out, clobbered in place)
            pair_normed,
            mask,
            self.proj_gate_weight,
            self.center_norm_weight,
            self.center_norm_bias,
            self.out_proj_weight,
            self.gating_weight,
            self._outgoing,
            True,                      # is_vnni
        )


def main(use_torch, torch_compile, use_fused, outgoing):
    c_pair = 128

    if use_torch:
        import sgl_kernel  # noqa: F401

        if torch_compile:
            register_fake_ops()

        m_ref = TriangleMultiplicationTorch(c_pair=c_pair, _outgoing=outgoing)
        m_ref = m_ref.to(torch.bfloat16)
        m_ref.eval()
        with torch.no_grad():
            m_ref.left_norm_input.weight.copy_(1.0 + 0.1 * torch.randn_like(m_ref.left_norm_input.weight))
            m_ref.left_norm_input.bias.copy_(0.2 + 0.1 * torch.randn_like(m_ref.left_norm_input.bias))
            m_ref.center_norm.weight.copy_(1.0 + 0.1 * torch.randn_like(m_ref.center_norm.weight))
            m_ref.center_norm.bias.copy_(0.2 + 0.1 * torch.randn_like(m_ref.center_norm.bias))

        # The pure-torch module is used only for the correctness reference.
        # The benchmarked path is the SGL variant (drop-in or fused).
        if use_fused:
            m = TriangleMultiplicationFusedSGL(m_ref)
        else:
            m = TriangleMultiplicationSGL(m_ref)
    else:
        from af3_kernels import TriangleMultiplicationCpp
        m = TriangleMultiplicationCpp(c_pair=c_pair, _outgoing=outgoing)
        m = m.to(torch.bfloat16)

    m.eval()
    print("done model creation")

    # Keep the same default shape as the grid-attention bench so timings are
    # directly comparable across the two ops on the same machine.
    N_token = 4655

    warmup = 20 if N_token <= 1024 else 10
    measure = 50 if N_token <= 1024 else 20

    pair = torch.randn(N_token, N_token, c_pair, dtype=torch.bfloat16)
    mask = torch.ones(N_token, N_token, dtype=torch.bfloat16)
    assert torch.all(mask == 1)
    print("done tensor creation")

    with torch.no_grad():
        if use_torch:
            y_ref = m_ref(pair, mask)

        if torch_compile:
            m = torch.compile(m)
            _ = m(pair, mask)

        if use_torch:
            y_test = m(pair, mask)
            # Looser tolerance than the GSA bench: TM's einsum reduces over k
            # of length N_token (~4655 here), so bf16 round-on-store accumulates
            # to a few ulps (~0.03 max abs) even with an fp32 brgemm accumulator.
            torch.testing.assert_close(y_ref, y_test, atol=5e-2, rtol=1e-2)

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
    parser.add_argument('--torch', action='store_true',
                        help='use the SGL drop-in (pure torch is reference only). '
                             'Without this flag, benchmarks the xfold C++ kernel.')
    parser.add_argument('--torch-compile', action='store_true')
    parser.add_argument('--fused', action='store_true',
                        help='use the single fused TM CPU kernel '
                             '(sgl_kernel::fused_triangle_multiplication). '
                             'Requires --torch.')
    parser.add_argument('--incoming', action='store_true',
                        help='use the incoming equation (ckj,cki->cij); default is outgoing (cik,cjk->cij)')
    args = parser.parse_args()

    main(args.torch, args.torch_compile, args.fused, not args.incoming)
