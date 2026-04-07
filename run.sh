#!/bin/bash
# This script is used to run AlphaFold on a single node.

set -o pipefail

cd $(dirname "$0")
source scripts/env.sh || { \
    echo 'Please place your `env.sh` under `scripts` directory.'; \
    echo 'You can refer to `env.sh.example` for the content of `env.sh`.'; \
    exit 1; \
}

#INPUT_NAME=${1:-37aa_2JO9.json}
export USE_DIST=0
NCORES=72
RANK=0
DB_DIR=/data/af3data
MODEL_DIR=/data/af3data/params
RAW_INPUT_DIR=/home/intel/af3/chunyuan/af3_kernels-dev/data/inputs
PADDED_OUTPUT_ROOT=/home/intel/af3/chunyuan/af3_kernels-dev/data/outputs
UNPADDED_OUTPUT_ROOT=/home/intel/af3/chunyuan/af3_kernels-dev/data/outputs_unpadded

INPUT_NAME=${INPUT_NAME:-amp_81.txt}
# 
# INPUT_NAME=igm_237x8.txt

# TODO: Jackhmmer failed
# INPUT_NAME=titin_34350.txt

# TODO: takes very long, test again
# INPUT_NAME=rydr_5038.txt

# INPUT_NAME=piezo2_2752.txt
# INPUT_NAME=reelin_3469.txt
# INPUT_NAME=lrp2_4655.txt

# TODO: failed to run since size too large
# INPUT_NAME=lcl_6077.txt



# INPUT_NAME=7rce.json
#Protein-DNA-Ion_PDB_7RCE_data.json
LOG_DIR=/home/intel/af3/chunyuan/af3_kernels-dev/data/results/log
HMMER_BIN=/home/intel/af3/chunyuan/af3_kernels-dev/hmmer/bin

# Usage:
#   1) First run the data pipeline and inference from the raw input:
#      RUN_DATA_PIPELINE=True INPUT_JSON_PATH=${RAW_INPUT_DIR}/${INPUT_NAME} bash run.sh
#      (or pipeline-only cache generation):
#      RUN_DATA_PIPELINE=True RUN_INFERENCE=False INPUT_JSON_PATH=${RAW_INPUT_DIR}/${INPUT_NAME} bash run.sh
#   2) Reuse the saved processed JSON on later runs:
#      RUN_DATA_PIPELINE=False INPUT_JSON_PATH=/path/to/<job>/<job>_data.json bash run.sh
RUN_DATA_PIPELINE=${RUN_DATA_PIPELINE:-False}
RUN_INFERENCE=${RUN_INFERENCE:-True}
PAD_TO_BUCKETS=${PAD_TO_BUCKETS:-True}
DEFAULT_RAW_INPUT_JSON=${RAW_INPUT_DIR}/${INPUT_NAME}

if [[ "${PAD_TO_BUCKETS}" == "True" ]]; then
    OUTPUT_DIR=${OUTPUT_DIR:-${PADDED_OUTPUT_ROOT}}
else
    OUTPUT_DIR=${OUTPUT_DIR:-${UNPADDED_OUTPUT_ROOT}}
fi

resolve_processed_input_json() {
    local source_json="$1"
    local output_root="${2:-${OUTPUT_DIR}}"
    python - "$source_json" "$output_root" <<'PY'
import json
import pathlib
import string
import sys

source_json = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])

with source_json.open() as f:
    payload = json.load(f)

name = payload.get("name") or source_json.stem
lower_spaceless_name = name.lower().replace(' ', '_')
allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
sanitised_name = ''.join(ch for ch in lower_spaceless_name if ch in allowed_chars)

print(output_dir / sanitised_name / f'{sanitised_name}_data.json')
PY
}

if [[ "${RUN_DATA_PIPELINE}" == "True" ]]; then
    INPUT_JSON_PATH=${INPUT_JSON_PATH:-${DEFAULT_RAW_INPUT_JSON}}
else
    if [[ -z "${PROCESSED_INPUT_JSON:-}" ]]; then
        PROCESSED_INPUT_JSON=$(resolve_processed_input_json "${DEFAULT_RAW_INPUT_JSON}" "${PADDED_OUTPUT_ROOT}")
    fi
    INPUT_JSON_PATH=${INPUT_JSON_PATH:-${PROCESSED_INPUT_JSON}}
fi

if [[ -d "${HMMER_BIN}" ]]; then
    export PATH="${HMMER_BIN}:${PATH}"
fi

mkdir -p "${LOG_DIR}"

#export USE_OPENMP=1
#export OMP_NUM_THREADS=${NCORES}
#echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

echo "Running AlphaFold on single node for ${INPUT_JSON_PATH}, NCORES=${NCORES}, RANK=${RANK}, run_data_pipeline=${RUN_DATA_PIPELINE}, run_inference=${RUN_INFERENCE}, pad_to_buckets=${PAD_TO_BUCKETS}, output_dir=${OUTPUT_DIR}"
#numactl -N 0
time numactl -N 0 python -m torch.backends.xeon.run_cpu --ninstances 1 --ncores-per-instance ${NCORES} --rank ${RANK} run_alphafold.py \
    --db_dir=${DB_DIR} \
    --jackhmmer_n_cpu=${NCORES} \
    --nhmmer_n_cpu=${NCORES} \
    --run_data_pipeline=${RUN_DATA_PIPELINE} \
    --run_inference=${RUN_INFERENCE} \
    --pad_to_buckets=${PAD_TO_BUCKETS} \
    --json_path=${INPUT_JSON_PATH} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} 2>&1 \
    | tee -a ${LOG_DIR}/${INPUT_NAME}.log





# time numactl -N 0 python -m torch.backends.xeon.run_cpu \
#     --ninstances 1 \
#     --ncores-per-instance ${NCORES} \
#     --rank ${RANK} \
#     run_alphafold.py \
#     --db_dir=/data \
#     --jackhmmer_n_cpu=${NCORES} \
#     --nhmmer_n_cpu=${NCORES} \
#     --run_data_pipeline=False \
#     --run_inference=True \
#     --json_path=${JSON_PATH}/${INPUT_NAME} \
#     --model_dir=${MODEL_DIR} \
#     --output_dir=${OUTPUT_DIR} \
#     2>&1 | tee -a ${LOG_DIR}/${INPUT_NAME}.log &

# PY_PID=$!

# # # collect RSS (memory usage)
# LOG_PREFIX=../chunyuan_profile/${INPUT_NAME}
# MEM_LOG=${LOG_PREFIX}_rss.log

# while true; do
#     date +%s
#     grep -E 'MemTotal|MemFree|MemAvailable|Buffers|Cached' /proc/meminfo
#     sleep 1
# done > $MEM_LOG &

# MEM_MONITOR_PID=$!
# echo "Memory monitor PID: $MEM_MONITOR_PID"

# wait $PY_PID
# kill $MEM_MONITOR_PID