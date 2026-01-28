import torch
import torch.nn as nn

import xfold
from xfold.nn.attention import GridSelfAttentionTorch
from af3_kernels import GridSelfAttentionCpp

# TODO: add correctness check
# TODO: add bench time

# TODO: test transpose=True
m = GridSelfAttentionTorch().to(torch.bfloat16)

xfold_m = GridSelfAttentionCpp().to(torch.bfloat16)

N_token = 384
c_pair = 128
pair = torch.randn(N_token, N_token, c_pair, dtype=torch.bfloat16)
mask = torch.randn(N_token, N_token, dtype=torch.bfloat16)

with torch.no_grad():
    y = m(pair, mask)
    
    y_xfold = xfold_m(pair, mask)
    
    record_shapes = True
    
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=record_shapes) as prof:
        y = m(pair, mask)
        # y_xfold = xfold_m(pair, mask)
    
    print(prof.key_averages(group_by_input_shape=record_shapes).table(sort_by="self_cpu_time_total"))


print("done")