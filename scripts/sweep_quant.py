#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_quant_v1_configs() -> list[dict[str, str]]:
    """Small, high-signal quantization sweep around the baseline."""
    return [
        {
            "name": "baseline_q",
            "INT8_CLIP_PERCENTILE": "99.99984",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "65536",
            "INT8_KEEP_FLOAT_STORE_DTYPE": "fp16",
            "INT8_PER_ROW_SCALE_DTYPE": "fp16",
            "INT8_PER_TENSOR_SCALE_DTYPE": "fp32",
        },
        {
            "name": "clip_99999",
            "INT8_CLIP_PERCENTILE": "99.999",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "65536",
            "INT8_KEEP_FLOAT_STORE_DTYPE": "fp16",
            "INT8_PER_ROW_SCALE_DTYPE": "fp16",
            "INT8_PER_TENSOR_SCALE_DTYPE": "fp32",
        },
        {
            "name": "clip_999995",
            "INT8_CLIP_PERCENTILE": "99.9995",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "65536",
            "INT8_KEEP_FLOAT_STORE_DTYPE": "fp16",
            "INT8_PER_ROW_SCALE_DTYPE": "fp16",
            "INT8_PER_TENSOR_SCALE_DTYPE": "fp32",
        },
        {
            "name": "keepfloat_32k",
            "INT8_CLIP_PERCENTILE": "99.99984",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "32768",
            "INT8_KEEP_FLOAT_STORE_DTYPE": "fp16",
            "INT8_PER_ROW_SCALE_DTYPE": "fp16",
            "INT8_PER_TENSOR_SCALE_DTYPE": "fp32",
        },
        {
            "name": "keepfloat_96k",
            "INT8_CLIP_PERCENTILE": "99.99984",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "98304",
            "INT8_KEEP_FLOAT_STORE_DTYPE": "fp16",
            "INT8_PER_ROW_SCALE_DTYPE": "fp16",
            "INT8_PER_TENSOR_SCALE_DTYPE": "fp32",
        },
        {
            "name": "scale_bf16",
            "INT8_CLIP_PERCENTILE": "99.99984",
            "INT8_KEEP_FLOAT_MAX_NUMEL": "65536",
            "INT8_KEEP_FLOAT_STORE_DTYPE": "fp16",
            "INT8_PER_ROW_SCALE_DTYPE": "bf16",
            "INT8_PER_TENSOR_SCALE_DTYPE": "bf16",
        },
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run quantization sweeps for parameter-golf")
    p.add_argument("--preset", default="quant_v1", choices=["quant_v1"], help="Sweep preset")
    p.add_argument("--seeds", default="1337", help="Comma-separated seeds, e.g. 1337,2027,31415")
    p.add_argument("--nproc-per-node", type=int, default=8)
    p.add_argument("--max-wallclock-seconds", type=int, default=600)
    p.add_argument("--train-log-every", type=int, default=100)
    p.add_argument("--val-loss-every", type=int, default=200)
    p.add_argument("--run-prefix", default="qsw")
    p.add_argument("--results-dir", default="sweeps")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    if args.preset == "quant_v1":
        configs = build_quant_v1_configs()
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / f"{args.run_prefix}_{args.preset}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"

    base_env = {
        "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
        "TRAIN_LOG_EVERY": str(args.train_log_every),
        "VAL_LOSS_EVERY": str(args.val_loss_every),
        # baseline architecture (explicit for reproducibility)
        "VOCAB_SIZE": "1024",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TIE_EMBEDDINGS": "1",
        "TIED_EMBED_LR": "0.05",
        "TRAIN_BATCH_TOKENS": "524288",
        "TRAIN_SEQ_LEN": "1024",
    }

    rows: list[dict[str, str]] = []

    for cfg in configs:
        cfg_name = cfg["name"]
        cfg_env = {k: v for k, v in cfg.items() if k != "name"}
        for seed in seeds:
            run_id = f"{args.run_prefix}_{cfg_name}_s{seed}_{ts}"
            env = os.environ.copy()
            env.update(base_env)
            env.update(cfg_env)
            env["SEED"] = str(seed)
            env["RUN_ID"] = run_id

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={args.nproc_per_node}",
                "train_gpt.py",
            ]

            print("=" * 88)
            print(f"RUN_ID={run_id}")
            print(f"Config={cfg_name} seed={seed}")
            print("Command:", " ".join(cmd))
            print("Quant env:", " ".join(f"{k}={v}" for k, v in cfg_env.items()))

            if args.dry_run:
                rows.append(
                    {
                        "run_id": run_id,
                        "config": cfg_name,
                        "seed": str(seed),
                        "status": "dry-run",
                        "log_path": f"logs/{run_id}.txt",
                    }
                )
                continue

            rc = subprocess.run(cmd, env=env, cwd=repo_root).returncode
            log_path = repo_root / "logs" / f"{run_id}.txt"
            rows.append(
                {
                    "run_id": run_id,
                    "config": cfg_name,
                    "seed": str(seed),
                    "status": "ok" if rc == 0 else f"exit-{rc}",
                    "log_path": str(log_path),
                }
            )
            if rc != 0:
                print(f"[warn] run failed: {run_id} rc={rc}")

            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["run_id", "config", "seed", "status", "log_path"])
                w.writeheader()
                w.writerows(rows)

    print(f"\nSweep done. Summary: {summary_csv}")
    print("Tip: python scripts/analyze_logs.py --glob 'logs/qsw_*.txt' --top 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
