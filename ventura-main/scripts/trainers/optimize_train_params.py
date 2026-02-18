#!/usr/bin/env python3
"""
Compute sensible training hyper-params for a CombinedLoader run.

► Inputs
    --cfg          path to the model YAML (must contain `batch_size`, `max_epochs`)
    --n_big        number of training samples in the large dataset
    --n_small      number of training samples in the small dataset
    --k_min / --k_max   optional bounds for the oversampling factor (default 5–20)

► Outputs   (printed only – the YAML file itself is NOT modified)
    • recommended batch_size_small   (integer ≤ batch_size_big)
    • oversampling factor  k
    • finetune_loss_weight  α  (= 1/k)
    • accumulate_grad_batches (read from YAML)
    • total_iter_length  (= ⌈Nbig/Bbig⌉ · max_epochs / accumulate_grad_batches)
"""
import argparse, math, sys, yaml
from pathlib import Path

def find_batch_small(
    n_big: int,
    n_small: int,
    b_big: int,
    k_min: float = 0.8,
    k_max: float = 1.25,
):
    """
    Choose the *small-set* batch size (b_small ≤ b_big) so that the sampling
    ratio  
        k = (n_big / b_big) ÷ (n_small / b_small)
    falls inside [k_min, k_max].

    Returns
    -------
    b_small : int   – batch size to use for the small loader
    k       : float – resulting oversampling ratio
    """
    # --- closed-form lower / upper bounds for b_small ------------------
    #   k_min ≤ (n_big/b_big)/(n_small/b_small) ≤ k_max
    # ⇒ k_min · n_small · b_big / n_big ≤ b_small ≤ k_max · n_small · b_big / n_big
    lo = math.ceil(k_min * n_small * b_big / n_big)
    hi = math.floor(k_max * n_small * b_big / n_big)

    # pick the smallest valid b_small within [1, b_big]
    if lo <= hi and lo <= b_big:
        b_small = max(1, lo)
        k = n_big * b_small / (n_small * b_big)
        return b_small, k

    # --- fallback: brute-force search (rarely needed) ------------------
    for b_small in range(1, b_big + 1):
        k = n_big * b_small / (n_small * b_big)
        if k_min <= k <= k_max:
            return b_small, k

    # If nothing fits, return 1 and accept that k is outside the bounds.
    k = n_big / (n_small * b_big)
    return 1, k

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, type=Path, help="model YAML file")
    p.add_argument("--n_big",   required=True, type=int, help="large-set size")
    p.add_argument("--n_small", required=True, type=int, help="small-set size")
    p.add_argument("--n_gpus", type=int, default=4, help="number of GPUs (for batch size calc)")
    p.add_argument("--k_min", type=float, default=1.0)
    p.add_argument("--k_max", type=float, default=20.0)
    args = p.parse_args()

    # --- read YAML -------------------------------------------------------
    cfg = yaml.safe_load(args.cfg.read_text())
    b_big = int(cfg["batch_size"])
    max_epochs = int(cfg["max_epochs"])
    accum = int(cfg.get("accumulate_grad_batches", 1))

    # --- search for a good B_small --------------------------------------
    b_small, k = find_batch_small(args.n_big, args.n_small,
                                  b_big, args.k_min, args.k_max)
    alpha = round(1.0 / k, 4)        # finetune_loss_weight
    steps_per_epoch = math.ceil(args.n_big / b_big)
    total_iter = int(steps_per_epoch * max_epochs / accum / args.n_gpus)

    # --- report ----------------------------------------------------------
    print("\n# ===== Recommended settings =====")
    print(f"batch_size_big      : {b_big}")
    print(f"batch_size_small    : {b_small}")
    print(f"oversampling factor k: {k:.2f}")
    print(f"finetune_loss_weight: {alpha}")
    print(f"acc_grad_batches    : {accum}")
    print(f"total_iter_length   : {total_iter}")
    print("# ================================\n")

if __name__ == "__main__":
    main()
