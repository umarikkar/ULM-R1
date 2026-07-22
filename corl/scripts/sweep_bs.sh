#!/bin/bash
set -uo pipefail
source /projects/u6gd/umar/env.sh
source /projects/u6gd/umar/miniconda3/etc/profile.d/conda.sh
conda activate corl
cd /projects/u6gd/umar/codes/ULM-R1
export PYTHONPATH=/projects/u6gd/umar/codes/ULM-R1
LOGDIR=/projects/u6gd/umar/codes/ULM-R1/corl/scripts/logs
mkdir -p "$LOGDIR"

run_one() {
  local K=$1 IPS=$2
  local tag="K${K}_IPS${IPS}"
  local memlog="$LOGDIR/mem_$tag.log"
  local runlog="$LOGDIR/run_$tag.log"
  echo ""
  echo "########## $tag ##########"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo; sleep 2; done ) > "$memlog" 2>&1 &
  local mpid=$!
  MAX_STEPS=2 SAVE_STEPS=1000 GROUP_SIZE=$K IMAGES_PER_STEP=$IPS OUT_DIR=/tmp/sweep_$tag \
    bash /projects/u6gd/umar/codes/ULM-R1/corl/scripts/rl_gran_lora.sh > "$runlog" 2>&1
  local rc=$?
  kill $mpid 2>/dev/null; wait $mpid 2>/dev/null
  # Max across all 4 GPUs across all samples
  local peak=$(awk '{for(i=1;i<=NF;i++) if($i+0>m) m=$i+0} END{print m}' "$memlog")
  local oom=$(grep -cE "OutOfMemory|CUDA out of memory" "$runlog")
  local step=$(grep -cE "\[save\]|\[done\]" "$runlog")
  echo "RESULT $tag rc=$rc peak_MiB=$peak oom=$oom saved=$step"
}

for cfg in "8 4" "8 6" "8 8" "12 4" "12 6" "12 8" "16 4" "16 8"; do
  run_one $cfg
done
echo "SWEEP DONE"
