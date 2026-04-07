# Copyright 2025 Xflops

import os

from af3_kernels.tools import USE_DIST

# options: ["torch", "triton", "ipex"]
layer_norm_implementation = "torch"

# options: ["torch", "triton", "cpp"]
dot_product_attention_implementation = "cpp"

# options: ["torch", "triton", "cpp"]
gated_linear_unit_implementation = "cpp"

# options: ["torch", "cpp", "sgl"]
grid_self_attention_implementation = os.environ.get(
	"AF3_GRID_SELF_ATTENTION_IMPL", "cpp"
)

# options: ["torch", ""cpp"]
self_attention_implementation = "cpp"

# options: ["torch", ""cpp"]
triangle_multiplication_implementation = "cpp"

# Whether to use the distributed C++ implementation of the  modules
grid_self_attention_dist = USE_DIST
triangle_multiplication_dist = False # Disabled by default
