#!/bin/bash
# Robust depth-submission driver. The depth-metric scoring holds the SQLite
# write lock for long stretches, which (a) 500s concurrent submission INSERTs
# and (b) starves small config updates. So: drain the scoring backlog before
# touching the DB, fix lb83's roles, then submit ONE leaderboard at a time and
# fully drain its scoring before the next — bounding concurrent scoring. The
# submit scripts already retry the final POST on "database is locked".
set -u
DB="$HOME/.dtofbenchmarking/database.db"
PY="$HOME/miniconda3/envs/BenchClient/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BENCHHUB_BASE_URL=http://127.0.0.1:6060

drain() {  # wait until no Pending/Processing submissions (optionally for one lb)
  local where="(processing_status LIKE 'Processing%' OR processing_status='Pending')"
  [ -n "${1:-}" ] && where="leaderboard_id=$1 AND $where"
  for _ in $(seq 1 720); do
    n=$(sqlite3 "$DB" "SELECT COUNT(*) FROM submission WHERE $where" 2>/dev/null)
    [ "${n:-1}" = "0" ] && return 0
    sleep 10
  done
}

echo "=== draining existing scoring backlog ==="; drain
echo "=== fixing lb83 field_roles (stereo pair) ==="
sqlite3 "$DB" -cmd ".timeout 120000" "UPDATE leaderboard SET field_roles_json='{\"cam_01_first_frame\": \"input\", \"cam_05_first_frame\": \"input\", \"cam_01_depth_vis\": \"gt\"}' WHERE id=83;" \
  && echo "  field_roles OK"

# Monocular depth boards: "lb repo input_field"
for spec in "84 prs-eth/ZuriPano rgb" "85 naufalso/carla_hd rgb" "86 POSE-Lab/IndustryShapes rgb"; do
  set -- $spec; lb=$1; repo=$2; inp=$3
  echo "=== SUBMIT mono lb$lb ($repo) ==="
  "$PY" "$HOME/Git/BenchClient/submit_depth.py" "$lb" "$repo" "$inp"
  echo "=== draining lb$lb scoring ==="; drain "$lb"
  echo "=== lb$lb done ==="
done

# Stereo board: cam_01 (left) + cam_05 (right) -> disparity
echo "=== SUBMIT stereo lb83 ==="
"$PY" "$HOME/Git/BenchClient/submit_stereo.py" 83 cam_01_first_frame cam_05_first_frame
drain 83
echo "ALL_DEPTH_V2_DONE"
