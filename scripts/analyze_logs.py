#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

RE_FINAL = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)")
RE_TOTAL = re.compile(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes")
RE_MODEL = re.compile(r"Serialized model int8\+zlib:\s*([0-9]+)\s*bytes")
RE_CODE = re.compile(r"Code size:\s*([0-9]+)\s*bytes")
RE_STEP = re.compile(r"step:([0-9]+)/([0-9]+)")
RE_TIME = re.compile(r"train_time:([0-9.]+)ms")
RE_QCFG = re.compile(r"quant_cfg:(.*)$")


def parse_log(path: Path) -> dict:
    final_val_loss = None
    final_val_bpb = None
    total_bytes = None
    model_bytes = None
    code_bytes = None
    final_step = None
    total_steps = None
    train_time_ms = None
    quant_cfg = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if (m := RE_FINAL.search(line)):
            final_val_loss = float(m.group(1))
            final_val_bpb = float(m.group(2))
        if (m := RE_TOTAL.search(line)):
            total_bytes = int(m.group(1))
        if (m := RE_MODEL.search(line)):
            model_bytes = int(m.group(1))
        if (m := RE_CODE.search(line)):
            code_bytes = int(m.group(1))
        if (m := RE_STEP.search(line)):
            final_step = int(m.group(1))
            total_steps = int(m.group(2))
        if (m := RE_TIME.search(line)):
            train_time_ms = float(m.group(1))
        if (m := RE_QCFG.search(line)):
            quant_cfg = m.group(1).strip()

    run_id = path.stem
    if run_id == "train" and path.parent.name:
        run_id = path.parent.name

    return {
        "log_path": str(path),
        "run_id": run_id,
        "final_val_loss": final_val_loss,
        "final_val_bpb": final_val_bpb,
        "total_bytes": total_bytes,
        "model_bytes": model_bytes,
        "code_bytes": code_bytes,
        "final_step": final_step,
        "total_steps": total_steps,
        "train_time_ms": train_time_ms,
        "quant_cfg": quant_cfg,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Parse parameter-golf logs and summarize final metrics")
    p.add_argument("--glob", dest="glob_pat", default="logs/*.txt", help="Glob for log files")
    p.add_argument("--top", type=int, default=20, help="How many top rows to print")
    p.add_argument("--csv", default=None, help="Optional CSV output path")
    p.add_argument("--json", default=None, help="Optional JSON output path")
    args = p.parse_args()

    paths = [Path(x) for x in sorted(glob.glob(args.glob_pat, recursive=True))]
    rows = [parse_log(pth) for pth in paths]
    rows = [r for r in rows if r["final_val_bpb"] is not None]
    rows.sort(key=lambda x: x["final_val_bpb"])

    if not rows:
        print("No parsed runs with final_int8_zlib_roundtrip_exact found.")
        return 1

    print("rank\trun_id\tval_bpb\tval_loss\ttotal_bytes\ttrain_time_ms")
    for i, r in enumerate(rows[: args.top], start=1):
        print(
            f"{i}\t{r['run_id']}\t{r['final_val_bpb']:.8f}\t{r['final_val_loss']:.8f}\t"
            f"{r['total_bytes']}\t{r['train_time_ms']}"
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote CSV: {out}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
