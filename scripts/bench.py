# Copyright 2025 Xflops

import torch
import torch.distributed as dist
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import os
import time

os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
os.environ["LOCAL_RANK"] = str(os.environ.get("MPI_LOCALRANKID", 0))
os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))

backend = os.environ.get("USE_BACKEND", "ccl")
dist.init_process_group(backend, init_method="env://")

rank = dist.get_rank()
ws = dist.get_world_size()


def test_op(op_fn, num_iter=1000, warmup=10):
    for _ in range(warmup):
        op_fn()

    dist.barrier()
    start = time.time()
    for _ in range(num_iter):
        op_fn()
    return (time.time() - start) / num_iter


if __name__ == "__main__":
    shape = (1024, 1024)
    numel = shape[0] * shape[1]

    tensor = torch.randn(shape)
    output = [torch.zeros_like(tensor) for _ in range(ws)]
    all_gather_time = test_op(lambda: dist.all_gather(output, tensor))

    tensor = torch.randn(shape) if rank == 0 else torch.zeros(shape)
    broadcast_time = test_op(lambda: dist.broadcast(tensor, 0))

    if rank == 0:
        print(f"AG: {all_gather_time:.6f}s/iter, bandwidth: {numel * 4 * (ws - 1) / all_gather_time / 1e9:.2f} GB/s")
        print(f"Bcast: {broadcast_time:.6f}s/iter, bandwidth: {numel * 4 * (ws - 1) / broadcast_time / 1e9:.2f} GB/s")

    dist.barrier()
    dist.destroy_process_group()
