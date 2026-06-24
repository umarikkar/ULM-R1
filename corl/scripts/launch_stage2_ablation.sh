#!/bin/bash
# Launch two parallel Stage-2 SFT runs in detached screens:
#   - GPUs 0-3, perceptual_weight=0
#   - GPUs 4-7, perceptual_weight=0.5
# Each run gets a distinct master_port to avoid "address already in use".

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

LOG_DIR=${PROJECT_ROOT}/logs
mkdir -p "${LOG_DIR}"

STAMP=$(date +%Y%m%d_%H%M%S)

launch_run() {
    local name=$1
    local gpus=$2
    local pw=$3
    local port=$4
    local log="${LOG_DIR}/${STAMP}_${name}.log"

    echo "Launching screen '${name}' on GPUs ${gpus} (pw=${pw}, port=${port})"
    echo "  log: ${log}"
    screen -dmS "${name}" bash -c "
        cd '${PROJECT_ROOT}' && \
        CUDA_VISIBLE_DEVICES=${gpus} \
        perceptual_weight=${pw} \
        perceptual_layers='3,6,9' \
        master_port=${port} \
        bash corl/scripts/corl_sft_stage2.sh 2>&1 | tee '${log}'
    "
}

launch_run sft_pw0   "0,1,2,3,4,5,6,7" 0.25    29500
# launch_run sft_pw05  "4,5,6,7" 0.5  29501

echo
echo "Active screens:"
screen -ls || true
echo
echo "Attach with:  screen -r sft_pw0   |   screen -r sft_pw05"
echo "Tail logs:    tail -f ${LOG_DIR}/${STAMP}_s${name}.log"
