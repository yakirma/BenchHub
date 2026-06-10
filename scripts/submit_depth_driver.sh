#!/bin/bash
# Sequentially submit depth models to each LB once its materialisation is
# ready and has GT samples. GPU-bound, so one LB at a time.
set -u
DB="$HOME/.dtofbenchmarking/database.db"
PY="$HOME/miniconda3/envs/BenchClient/bin/python"
SUBMIT="$HOME/Git/BenchClient/submit_depth.py"
export BENCHHUB_BASE_URL=http://127.0.0.1:6060

# lb_id : "repo input_field"
declare -A LBS=(
  [83]="stereo-dataset/stereo-dataset cam_00_first_frame"
  [84]="prs-eth/ZuriPano rgb"
  [85]="naufalso/carla_hd rgb"
  [86]="POSE-Lab/IndustryShapes rgb"
)

for lb in 83 84 85 86; do
  set -- ${LBS[$lb]}; repo="$1"; inp="$2"
  echo "=== lb$lb ($repo / $inp): waiting for materialisation ==="
  for _ in $(seq 1 360); do
    st=$(sqlite3 "$DB" "SELECT status FROM leaderboard_materialization WHERE leaderboard_id=$lb" 2>/dev/null)
    [ "$st" = "ready" ] && break
    [ "$st" = "failed" ] && { echo "lb$lb materialisation FAILED"; break; }
    sleep 5
  done
  n=$(sqlite3 "$DB" "SELECT COUNT(DISTINCT sample_name) FROM custom_field WHERE leaderboard_id=$lb AND sample_id IS NULL AND submission_id IS NULL" 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then
    echo "=== lb$lb: $n GT samples — submitting ==="
    "$PY" "$SUBMIT" "$lb" "$repo" "$inp"
  else
    echo "=== lb$lb: 0 GT samples — SKIP (materialisation empty) ==="
  fi
done
echo "ALL_DEPTH_SUBMITS_DONE"
