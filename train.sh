#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash train.sh OUTPUT_DIR [SEED]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$1"
SEED="${2:-20260929}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${ROOT}/src/train.py" \
  --output "${OUTPUT_DIR}" \
  --epochs 30 \
  --batch-size 256 \
  --devices 4 \
  --learning-rate 1e-3 \
  --layer-learning-rate 1e-4 \
  --temperature 0.07 \
  --siglip-bias -10.0 \
  --layer-temperature 2.0 \
  --instruction-min 0.65 \
  --metadata-negative-max 0.70 \
  --balance-classes \
  --hard-negative-topk 128 \
  --hard-negative-weight 1.0 \
  --hard-negative-weighting linear \
  --hard-negative-temperature 0.1 \
  --seed "${SEED}"
