#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash retrieve.sh CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="$1"
OUTPUT_DIR="$2"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Use a fresh output directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

# The three modes use the same checkpoint and official test-to-train split.
for MODE in pp pt; do
  echo "RETRIEVAL_START mode=$MODE checkpoint=$CHECKPOINT"
  python "$PROJECT_ROOT/src/retrieve.py" \
    --mode "$MODE" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_DIR/$MODE" \
    --instruction-cosine-min 0.65 \
    --topk 10 \
    2>&1 | tee "$OUTPUT_DIR/$MODE.log"
  echo "RETRIEVAL_DONE mode=$MODE"
done

# Reorder exactly the PP top-10; repeated metadata keep their original PP order.
echo "RETRIEVAL_START mode=rerank checkpoint=$CHECKPOINT"
python "$PROJECT_ROOT/src/retrieve.py" \
  --mode rerank \
  --checkpoint "$CHECKPOINT" \
  --protein-results "$OUTPUT_DIR/pp" \
  --output "$OUTPUT_DIR/rerank" \
  --instruction-cosine-min 0.65 \
  2>&1 | tee "$OUTPUT_DIR/rerank.log"
echo "RETRIEVAL_DONE mode=rerank"

# Collect per-task and macro metrics, plus a compact table for inspection.
python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
results = {mode: json.loads((output / mode / "metrics.json").read_text())
           for mode in ("pp", "pt", "rerank")}
(output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
lines = ["Macro retrieval metrics (%); instruction cosine >= 0.65", "",
         "| Method | MRR@10 | R@1 | R@10 |",
         "|---|---:|---:|---:|"]
for mode, scores in results.items():
    macro = scores["macro"]
    values = " | ".join(f"{100 * macro[key]:.2f}" for key in ("MRR@10", "R@1", "R@10"))
    lines.append(f"| {mode} | {values} |")
table = "\n".join(lines) + "\n"
(output / "summary.md").write_text(table)
print(table, flush=True)
PY
