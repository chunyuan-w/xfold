# Copyright 2025 Xflops
# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import torch
import torch.nn as nn

from xfold import fastnn
from xfold.fastnn import config as fastnn_config
from af3_kernels import TriangleMultiplicationCpp, DistributedTriangleMultiplicationCpp
from af3_kernels.tools import profile

if fastnn_config.triangle_multiplication_implementation == "sgl":
    import sgl_kernel  # noqa: F401


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
        if len(x_shapes) == 3:
            x = x.view(-1, x.shape[-1])
        output = torch.ops.sgl_kernel.weight_packed_linear(
            x,
            self.weight,
            self.bias if hasattr(self, "bias") else None,
            True,
        )
        if len(x_shapes) == 3:
            output = output.view(x_shapes[0], x_shapes[1], -1)
        return output


class TriangleMultiplicationTorch(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplicationTorch, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation='ckj,cki->cij'
        if _outgoing is True:
            self.equation='cik,cjk->cij'

    @profile("TriangleMultiplication")
    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token]
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
        input += pair
        return input


class TriangleMultiplicationSGL(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super().__init__()
        self.c_pair = c_pair
        self._outgoing = _outgoing

        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = LinearSGL(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = LinearSGL(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = LinearSGL(self.c_pair, self.c_pair, bias=False)
        self.gating_linear = LinearSGL(self.c_pair, self.c_pair, bias=False)

        self.equation = 'ckj,cki->cij'
        if _outgoing is True:
            self.equation = 'cik,cjk->cij'

    def concat_proj_gate_weights(self):
        proj_gate_linear = LinearSGL(self.c_pair, 4 * self.c_pair, bias=False)
        proj_gate_linear.weight = torch.nn.Parameter(
            torch.cat(
                [
                    self.projection.weight.data,
                    self.gate.weight.data,
                ],
                dim=0,
            ),
            requires_grad=False,
        )
        self.proj_gate_projection = proj_gate_linear
        del self.projection
        del self.gate

    @profile("TriangleMultiplication")
    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pair_normed = self.left_norm_input(pair)
        if mask.dtype != pair.dtype:
            mask = mask.to(pair.dtype)
        if not mask.is_contiguous():
            mask = mask.contiguous()
        return torch.ops.sgl_kernel.fused_triangle_multiplication(
            pair,
            pair_normed,
            mask,
            self.proj_gate_projection.weight,
            self.center_norm.weight,
            self.center_norm.bias,
            self.output_projection.weight,
            self.gating_linear.weight,
            self._outgoing,
            True,
        )


if fastnn_config.triangle_multiplication_implementation == "sgl":
    TriangleMultiplication = TriangleMultiplicationSGL
elif fastnn_config.triangle_multiplication_implementation == "cpp":
    if fastnn_config.triangle_multiplication_dist:
        TriangleMultiplication = DistributedTriangleMultiplicationCpp
    else:
        TriangleMultiplication = TriangleMultiplicationCpp
else:
    TriangleMultiplication = TriangleMultiplicationTorch
