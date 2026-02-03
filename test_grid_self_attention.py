import time

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

import xfold
from xfold.nn.attention import GridSelfAttentionTorch
from af3_kernels import GridSelfAttentionCpp
from xfold.fastnn import config as fastnn_config

fastnn_config.dot_product_attention_implementation = "torch"

inductor_config.profiler_mark_wrapper_call = True
inductor_config.cpp.enable_kernel_profile = True

torch_compile = True
torch_compile = False

# TODO: add correctness check
# TODO: add bench time

# TODO: test transpose=True

# m = GridSelfAttentionTorch().to(torch.bfloat16)
# m.eval()

warmup = 20
measure = 50
xfold_m = GridSelfAttentionCpp().to(torch.bfloat16)
xfold_m.eval()
print("done model creation")
# N_token = 384
# N_token = 5120
N_token = 2048

c_pair = 128
pair = torch.randn(N_token, N_token, c_pair, dtype=torch.bfloat16)
mask = torch.randn(N_token, N_token, dtype=torch.bfloat16)
print("done tensor creation")

with torch.no_grad():

    if torch_compile:
        m = torch.compile(m)
    
    for _ in range(warmup):
        # y = m(pair, mask)
    
        y_xfold = xfold_m(pair, mask)
        # print(y_xfold.shape)
        # breakpoint()
    print("done 1st run")

    record_shapes = True
    
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=record_shapes) as prof:
        # y = m(pair, mask)
        y_xfold = xfold_m(pair, mask)
    
    print(prof.key_averages(group_by_input_shape=record_shapes).table(sort_by="self_cpu_time_total"))

    start = time.time()
    for _ in range(measure):
        # y = m(pair, mask)
    
        y_xfold = xfold_m(pair, mask)        
    end = time.time()
    
    print(f"time used: {(end - start) / measure} s")

print("done")