#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
ST_DATASET_ROOT="${ST_DATASET_ROOT:-${DATASET_ROOT:-/data/pudata/VTUAV/test_ST}}"
THREADS="${THREADS:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
TEST_EPOCHS="${TEST_EPOCHS:-30}"
CONFIG="deep_rgbt_256_vtuav"

cd "${SCRIPT_DIR}"

read -r -a test_epochs <<< "${TEST_EPOCHS}"

for epoch in "${test_epochs[@]}"; do
  printf -v epoch_padded "%04d" "${epoch}"
  checkpoint="${SCRIPT_DIR}/output/checkpoints/train/cpetrack/${CONFIG}/CPETrack_ep${epoch_padded}.pth.tar"
  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${SCRIPT_DIR}/output/checkpoints/train/sttrack/${CONFIG}/STTrack_ep${epoch_padded}.pth.tar"
  fi
  results_root="${SCRIPT_DIR}/RGBT_workspace/results/VTUAVST/${CONFIG}_${epoch}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  echo "===== VTUAV-ST tracking: epoch ${epoch} ====="
  CUDA_VISIBLE_DEVICES="${GPU}" python ./RGBT_workspace/test_rgbt_mgpus.py \
    --script_name cpetrack \
    --yaml_name "${CONFIG}" \
    --dataset_name VTUAVST \
    --dataset_root "${ST_DATASET_ROOT}" \
    --epoch "${epoch}" \
    --threads "${THREADS}" \
    --num_gpus "${NUM_GPUS}"

  echo "===== VTUAV-ST MPR/MSR: epoch ${epoch} ====="
  python ./RGBT_workspace/evaluate_vtuav_st_m.py \
    --dataset_root "${ST_DATASET_ROOT}" \
    --results_root "${results_root}"
done
