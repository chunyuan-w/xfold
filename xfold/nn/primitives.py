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
from af3_kernels.tools import profile


class Transition(nn.Module):

    def __init__(self, c_x: int, num_intermediate_factor: int = 4) -> None:
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_x
        self.input_layer_norm = fastnn.LayerNorm(c_x)
        self.transition1 = nn.Linear(
            c_x, self.num_intermediate_factor * c_x * 2, bias=False)
        self.transition1_weight_t = None # cache transposed weight (cpp/torch/triton)
        self.transition1_weight_packed = None  # cache convert_weight_packed'd weight (sgl)
        self.transition2 = nn.Linear(
            self.num_intermediate_factor * c_x, c_x, bias=False)
        # cache packed transition2 weight for the fused sgl Transition kernel
        self.transition2_weight_packed = None

    def pack_weight(self) -> None:
        if fastnn_config.gated_linear_unit_implementation != "sgl":
            return
        import sgl_kernel  # noqa: F401
        if self.transition1_weight_packed is None:
            self.transition1_weight_packed = torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(self.transition1.weight.data),
                requires_grad=False,
            )
        # The fused transition1+transition2 kernel needs both weights packed and
        # requires the transition2 output channels (c_in) to tile by 32.  When
        # c_in % 32 != 0 we leave transition2_weight_packed = None and fall back
        # to the two-step path in forward().
        if self.transition2_weight_packed is None and self.c_in % 32 == 0:
            self.transition2_weight_packed = torch.nn.Parameter(
                torch.ops.sgl_kernel.convert_weight_packed(self.transition2.weight.data),
                requires_grad=False,
            )

    def _glu_weight(self) -> torch.Tensor:
        if fastnn_config.gated_linear_unit_implementation == "sgl":
            assert self.transition1_weight_packed is not None, (
                "sgl gated_linear_unit requires pre-packed weights; call "
                "pack_sgl_weights(model) after loading weights (it invokes "
                "Transition.pack_weight)."
            )
            return self.transition1_weight_packed
        if self.transition1_weight_t is None:
            self.transition1_weight_t = self.transition1.weight.T.contiguous()
        return self.transition1_weight_t

    @profile("Transition")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer_norm(x)
        # Fused transition1(GLU)+transition2: avoids materializing the wide
        # [M, 4*c_x] gated intermediate (the dominant cost at pair-rep scale,
        # M = N_tok²).  sgl-only and only when transition2 was packed.
        if (
            fastnn_config.gated_linear_unit_implementation == "sgl"
            and self.transition2_weight_packed is not None
            and not torch.is_grad_enabled()
        ):
            import sgl_kernel  # noqa: F401
            x2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.sgl_kernel.fused_transition(
                x2d, self.transition1_weight_packed, self.transition2_weight_packed, True)
            return out.view(*x.shape[:-1], out.shape[-1])
        c = fastnn.gated_linear_unit(x, self._glu_weight())
        return self.transition2(c)


class OuterProductMean(nn.Module):
    def __init__(self, c_msa: int = 64, num_output_channel: int = 128, num_outer_channel: int = 32) -> None:
        super(OuterProductMean, self).__init__()

        self.c_msa = c_msa
        self.num_outer_channel = num_outer_channel
        self.num_output_channel = num_output_channel
        self.epsilon = 1e-3

        self.layer_norm_input = fastnn.LayerNorm(self.c_msa)
        self.left_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)
        self.right_projection = nn.Linear(
            self.c_msa, self.num_outer_channel, bias=False)

        self.output_w = nn.Parameter(
            torch.randn(self.num_outer_channel, self.num_outer_channel, self.num_output_channel))
        self.output_b = nn.Parameter(
            torch.randn(self.num_output_channel))

    @profile("OuterProductMean")
    def forward(self, msa: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        msa = self.layer_norm_input(msa)
        left_act = mask * self.left_projection(msa)
        right_act = mask * self.right_projection(msa)

        left_act = left_act.permute(0, 2, 1)
        act = torch.einsum('acb,ade->dceb', left_act, right_act)
        act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
        act = act.permute(1, 0, 2)

        norm = torch.einsum('abc,adc->bdc', mask, mask)
        return act / (self.epsilon + norm)
