#!/usr/bin/env bash
set -euo pipefail

# Parameter Golf first campaign launcher (Runpod-friendly)
#
# Usage:
#   bash scripts/run_first_campaign.sh
#   NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600 bash scripts/run_first_campaign.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_SHARDS="${TRAIN_SHARDS:-10}"
SEEDS_BASELINE="${SEEDS_BASELINE:-1337,2027,31415}"
SEEDS_SWEEP="${SEEDS_SWEEP:-1337}"
RUN_PREFIX="${RUN_PREFIX:-campaign1}"

mkdir -p logs sweeps

echo "[campaign] root=$ROOT_DIR"
echo "[campaign] nproc=$NPROC_PER_NODE max_wallclock=$MAX_WALLCLOCK_SECONDS train_shards=$TRAIN_SHARDS"
echo "[campaign] baseline_seeds=$SEEDS_BASELINE sweep_seeds=$SEEDS_SWEEP"

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[error] torchrun not found; activate challenge environment first."
  exit 2
fi

if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ] || [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
  echo "[campaign] downloading cached challenge data/tokenizer..."
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
fi

IFS=',' read -r -a BASE_SEED_ARR <<< "$SEEDS_BASELINE"
for seed in "${BASE_SEED_ARR[@]}"; do
  run_id="${RUN_PREFIX}_baseline_s${seed}_$(date -u +%Y%m%d_%H%M%S)"
  echo "[baseline] RUN_ID=$run_id"
  RUN_ID="$run_id" \
  SEED="$seed" \
  DATA_PATH="./data/datasets/fineweb10B_sp1024/" \
  TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
  VOCAB_SIZE=1024 \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  TRAIN_LOG_EVERY=100 \
  VAL_LOSS_EVERY=200 \
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py

done

python3 scripts/sweep_quant.py \
  --preset quant_v1 \
  --seeds "$SEEDS_SWEEP" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --max-wallclock-seconds "$MAX_WALLCLOCK_SECONDS" \
  --run-prefix "$RUN_PREFIX"

python3 scripts/analyze_logs.py --glob "logs/${RUN_PREFIX}_*.txt" --top 20 --csv "sweeps/${RUN_PREFIX}_ranked.csv" --json "sweeps/${RUN_PREFIX}_ranked.json"

echo "[campaign] done. Ranked results: sweeps/${RUN_PREFIX}_ranked.csv"
