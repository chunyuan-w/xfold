#!/usr/bin/env python
"""Isolated TriangleMultiplication benchmark: SGL vs CPP/TPP, same env.

Run on the TARGET machine to confirm whether the sgl TM regression is real at
the kernel level (vs the whole-Evoformer delta) and which stage dominates.

IMPORTANT — keep the env identical to the TPP path (no sgl-only crutches):
    export TCMALLOC_AGGRESSIVE_DECOMMIT=false
    export LD_PRELOAD=<path to libtcmalloc.so>          # the production allocator
    # do NOT set TCMALLOC_RELEASE_RATE=0 for the comparison (it's only a private
    # profiling aid; TPP doesn't use it, so it would be an unfair comparison)
    # match the production thread count / NUMA binding (e.g. numactl as you ship)

Usage:
    python bench_tm_sgl_vs_cpp.py --N 2752
    python bench_tm_sgl_vs_cpp.py --N 4655 --iters 2
    # stage breakdown (the [tm] line goes to stderr; RELEASE_RATE=0 OK here as a
    # private measurement aid only):
    SGL_TM_PROFILE=1 python bench_tm_sgl_vs_cpp.py --N 2752 --iters 3
"""
import argparse, time, gc
import torch
import xfold.fastnn.config as cfg

cfg.pad_to_buckets = False  # all-ones mask fast path (kernel skips mask)

from xfold.nn.triangle_multiplication import TriangleMultiplicationSGL
from af3_kernels import TriangleMultiplicationCpp
import sgl_kernel  # noqa: F401


def build_sgl(c):
    m = TriangleMultiplicationSGL(c, _outgoing=True).to(torch.bfloat16).eval()
    m.concat_proj_gate_weights()
    m.proj_gate_projection.pack_weight()
    m.output_projection.pack_weight()
    m.gating_linear.pack_weight()
    return m


def bench(fn, iters, warm):
    for _ in range(warm):
        r = fn(); del r
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn(); del r
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2752)
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warm", type=int, default=2)
    args = ap.parse_args()
    N, c = args.N, args.c
    torch.manual_seed(0)

    x = torch.randn(N, N, c, dtype=torch.bfloat16)
    mask = torch.ones(N, N, dtype=torch.bfloat16)
    ms = build_sgl(c)
    mc = TriangleMultiplicationCpp(c).to(torch.bfloat16).eval()

    # FLOP model: einsum (Stage B) dominates the *theoretical* work = 2*c*N^3;
    # the projection/out/gate GEMMs add 2*N^2*c*(4c + c + 2c).  Report both the
    # einsum-only and total rates so you can see which the kernel is bound by.
    einsum_flops = 2.0 * c * N**3
    proj_flops = 2.0 * (N * N) * c * (4 * c + c + 2 * c)
    total_flops = einsum_flops + proj_flops

    with torch.no_grad():
        ts = bench(lambda: ms(x, None), args.iters, args.warm)
        tc = bench(lambda: mc(x, mask), args.iters, args.warm)

    g = lambda ms_, fl: fl / (ms_ * 1e-3) / 1e9
    print(f"TM  N={N} c={c}  threads={torch.get_num_threads()}")
    print(f"  sgl {ts:9.1f} ms   (einsum {g(ts, einsum_flops):5.0f} GF/s | total {g(ts, total_flops):5.0f} GF/s)")
    print(f"  cpp {tc:9.1f} ms   (einsum {g(tc, einsum_flops):5.0f} GF/s | total {g(tc, total_flops):5.0f} GF/s)")
    print(f"  sgl/cpp = {tc/ts:.2f}x   ({'sgl SLOWER' if ts > tc else 'sgl faster'})")
    print("  (set SGL_TM_PROFILE=1 to get the per-stage [tm] breakdown on stderr)")


if __name__ == "__main__":
    main()
