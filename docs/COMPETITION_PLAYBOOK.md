# Parameter Golf: Fast Winning Playbook

This playbook is optimized for **rapid leaderboard improvement** under the 10-minute / 16MB track.

## Immediate objective

- Beat baseline `val_bpb` (currently ~`1.2244`) with a reproducible run.
- Keep submission artifact (`code + int8_zlib model`) below `16,000,000` bytes.

## Core principle

The scored metric is from the **post-quant roundtrip** model:

- `final_int8_zlib_roundtrip_exact val_bpb:...`

So optimize for post-quant performance first, not pre-quant loss alone.

## 48-hour runbook

### Stage A: baseline lock (3 runs)

1. Reproduce baseline config with 3 seeds.
2. Confirm byte accounting + runtime + metric parsing.

### Stage B: quant sweep (6+ runs)

Use:

```bash
python scripts/sweep_quant.py --preset quant_v1 --seeds 1337
python scripts/analyze_logs.py --glob 'logs/qsw_*.txt' --top 20
```

Promote top 2-3 configs to multi-seed validation:

```bash
python scripts/sweep_quant.py --preset quant_v1 --seeds 1337,2027,31415
python scripts/analyze_logs.py --glob 'logs/qsw_*.txt' --csv sweeps/latest_ranked.csv
```

### Stage C: schedule/lr tuning on best quant config

Tune only a few high-leverage variables first:
- `WARMUP_STEPS`
- `WARMDOWN_ITERS`
- `MATRIX_LR`
- `TIED_EMBED_LR`
- `BETA2`

## New quantization knobs in `train_gpt.py`

All env-configurable now:

- `INT8_CLIP_PERCENTILE` (default `99.99984`)
- `INT8_KEEP_FLOAT_MAX_NUMEL` (default `65536`)
- `INT8_KEEP_FLOAT_STORE_DTYPE` (`fp16|bf16|fp32`, default `fp16`)
- `INT8_PER_ROW_SCALE_DTYPE` (`fp16|bf16|fp32`, default `fp16`)
- `INT8_PER_TENSOR_SCALE_DTYPE` (`fp16|bf16|fp32`, default `fp32`)

These are logged in `quant_cfg:` in each run log for reproducibility.

## Submission checklist

- [ ] Improvement threshold met with significance requirement
- [ ] `final_int8_zlib_roundtrip_exact` is the reported score
- [ ] Artifact bytes under cap
- [ ] Runtime under caps (train + eval)
- [ ] PR folder includes README, `submission.json`, logs, and runnable training script
