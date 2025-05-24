# Copyright 2025 Xflops

from af3_kernels.tools import USE_DIST

# options: ["torch", "triton", "ipex"]
layer_norm_implementation = "ipex"

# options: ["torch", "triton", "cpp"]
dot_product_attention_implementation = "cpp"

# options: ["torch", "triton", "cpp"]
gated_linear_unit_implementation = "cpp"

# options: ["torch", ""cpp"]
grid_self_attention_implementation = "cpp"

# options: ["torch", ""cpp"]
self_attention_implementation = "cpp"

# options: ["torch", ""cpp"]
triangle_multiplication_implementation = "cpp"

# Whether to use the distributed C++ implementation of the  modules
grid_self_attention_dist = USE_DIST
triangle_multiplication_dist = False # Disabled by default
