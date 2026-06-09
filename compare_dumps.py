# Copyright 2025 Xflops
"""Diff two AF3 checkpoint-dump dirs to localize where an SGL run diverges from
the torch reference (same seed/flags).

Produce the dumps by running each pipeline with AF3_DUMP_DIR set, e.g.:

    AF3_DUMP_DIR=dump_torch  <all-torch run>          # reference
    AF3_DUMP_DIR=dump_sgl    <gsa/tm sgl run>

then:

    python compare_dumps.py dump_torch dump_sgl

Checkpoints (written by xfold/alphafold3.py:_dbg_dump):
    recycleNN_pair / recycleNN_single   -- trunk conditioning after each recycle
    diff_stepNNN_pos                     -- sample-0 diffusion trajectory per step
                                            (diff_step-001_pos = scaled initial noise)

Reading the output: rel = ||torch - sgl|| / ||torch||. Through the trunk it should
sit near bf16 noise (~1e-2); the first checkpoint where it jumps and then keeps
growing is where the divergence originates. For the diffusion rows, the step where
rel explodes from ~bf16 to O(1) is the fold-flip / bifurcation step.
"""

import argparse
import os

import torch


def _sort_key(name: str):
    if name.startswith("recycle"):
        idx = int(name[7:9])
        sub = 0 if name.endswith("pair") else 1
        return (0, idx, sub)
    if name.startswith("diff_step"):
        s = name[len("diff_step"):name.rindex("_pos")]
        return (1, int(s), 0)
    return (2, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", help="reference dump dir (e.g. all-torch)")
    ap.add_argument("test_dir", help="dump dir to compare (e.g. sgl)")
    ap.add_argument("--rel-threshold", type=float, default=5e-2,
                    help="rel diff above this is flagged as a divergence (default 5e-2)")
    args = ap.parse_args()

    ref = {f[:-3] for f in os.listdir(args.ref_dir) if f.endswith(".pt")}
    test = {f[:-3] for f in os.listdir(args.test_dir) if f.endswith(".pt")}
    common = sorted(ref & test, key=_sort_key)
    only_ref, only_test = ref - test, test - ref
    if only_ref:
        print(f"[warn] {len(only_ref)} checkpoints only in {args.ref_dir} (e.g. {sorted(only_ref)[:3]})")
    if only_test:
        print(f"[warn] {len(only_test)} checkpoints only in {args.test_dir} (e.g. {sorted(only_test)[:3]})")
    if not common:
        raise SystemExit("No common checkpoints to compare.")

    print(f"{'checkpoint':<22} {'shape':<18} {'max|d|':>11} {'mean|d|':>11} {'rel':>11}  flag")
    print("-" * 90)
    first_flag = None
    eps = 1e-12
    for name in common:
        a = torch.load(os.path.join(args.ref_dir, name + ".pt")).float()
        b = torch.load(os.path.join(args.test_dir, name + ".pt")).float()
        if a.shape != b.shape:
            print(f"{name:<22} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a - b).abs()
        rel = (a - b).norm().item() / (a.norm().item() + eps)
        flag = ""
        if rel > args.rel_threshold:
            flag = "<-- DIVERGED"
            if first_flag is None:
                first_flag = name
                flag = "<-- FIRST DIVERGENCE"
        print(f"{name:<22} {str(tuple(a.shape)):<18} "
              f"{d.max().item():>11.3e} {d.mean().item():>11.3e} {rel:>11.3e}  {flag}")

    print("-" * 90)
    if first_flag is None:
        print(f"No checkpoint exceeded rel>{args.rel_threshold:g} -- runs track to ~bf16 throughout.")
    else:
        print(f"First divergence (rel>{args.rel_threshold:g}) at: {first_flag}")


if __name__ == "__main__":
    main()
