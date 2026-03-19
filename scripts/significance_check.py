#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import itertools
import math
import random
import re
from pathlib import Path

RE_FINAL = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)")


def parse_vals(pattern: str) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        path = Path(p)
        txt = path.read_text(encoding="utf-8", errors="ignore")
        val_loss = None
        val_bpb = None
        for line in txt.splitlines():
            m = RE_FINAL.search(line)
            if m:
                val_loss = float(m.group(1))
                val_bpb = float(m.group(2))
        if val_loss is not None and val_bpb is not None:
            out.append((str(path), val_loss, val_bpb))
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def exact_or_mc_pvalue(baseline: list[float], candidate: list[float], mc_trials: int, seed: int) -> tuple[float, str]:
    """One-sided permutation p-value for improvement in val_loss (nats)."""
    rng = random.Random(seed)
    n0 = len(baseline)
    n1 = len(candidate)
    pooled = baseline + candidate
    observed = mean(baseline) - mean(candidate)  # positive => candidate better

    total_combos = math.comb(n0 + n1, n1)
    if total_combos <= 200_000:
        ge = 0
        total = 0
        idxs = range(n0 + n1)
        for cand_idx in itertools.combinations(idxs, n1):
            cand_set = set(cand_idx)
            c = [pooled[i] for i in cand_set]
            b = [pooled[i] for i in idxs if i not in cand_set]
            diff = mean(b) - mean(c)
            if diff >= observed - 1e-15:
                ge += 1
            total += 1
        return ge / max(total, 1), f"exact ({total} labelings)"

    ge = 0
    for _ in range(mc_trials):
        perm = pooled[:]
        rng.shuffle(perm)
        c = perm[:n1]
        b = perm[n1:]
        diff = mean(b) - mean(c)
        if diff >= observed - 1e-15:
            ge += 1
    return ge / max(mc_trials, 1), f"monte-carlo ({mc_trials} shuffles)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check Parameter Golf significance gate: >=0.005 nats improvement, p<0.01")
    ap.add_argument("--baseline", required=True, help="Glob for baseline logs")
    ap.add_argument("--candidate", required=True, help="Glob for candidate logs")
    ap.add_argument("--mc-trials", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    b = parse_vals(args.baseline)
    c = parse_vals(args.candidate)
    if len(b) < 2 or len(c) < 2:
        print("Need >=2 runs for both baseline and candidate.")
        print(f"Found baseline={len(b)} candidate={len(c)}")
        return 2

    b_loss = [x[1] for x in b]
    c_loss = [x[1] for x in c]
    b_bpb = [x[2] for x in b]
    c_bpb = [x[2] for x in c]

    imp_nats = mean(b_loss) - mean(c_loss)
    imp_bpb = mean(b_bpb) - mean(c_bpb)
    pval, method = exact_or_mc_pvalue(b_loss, c_loss, args.mc_trials, args.seed)

    print(f"baseline_n={len(b)} mean_val_loss={mean(b_loss):.8f} mean_val_bpb={mean(b_bpb):.8f}")
    print(f"candidate_n={len(c)} mean_val_loss={mean(c_loss):.8f} mean_val_bpb={mean(c_bpb):.8f}")
    print(f"improvement_nats={imp_nats:.8f} (needs >= 0.005)")
    print(f"improvement_bpb={imp_bpb:.8f}")
    print(f"p_value={pval:.6g} via {method} (needs < 0.01)")

    pass_gate = (imp_nats >= 0.005) and (pval < 0.01)
    print("gate_pass=YES" if pass_gate else "gate_pass=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
