#!/bin/bash
# This script is used to run AlphaFold on a single node.

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
MODEL_DIR=/data/params
JSON_PATH=/workspace/inputs
#/workspace/outputs/Protein-DNA-Ion_PDB_7RCE

# INPUT_NAME=amp_81.txt

# INPUT_NAME=igm_237x8.txt

# TODO: Jackhmmer failed
# INPUT_NAME=titin_34350.txt

# TODO: takes very long, test again
# INPUT_NAME=rydr_5038.txt

# INPUT_NAME=piezo2_2752.txt
INPUT_NAME=reelin_3469.txt
# INPUT_NAME=lrp2_4655.txt

# TODO: failed to run since size too large
# INPUT_NAME=lcl_6077.txt



# INPUT_NAME=7rce.json
#Protein-DNA-Ion_PDB_7RCE_data.json
OUTPUT_DIR=/workspace/outputs
LOG_DIR=/workspace/results/log

#export USE_OPENMP=1
#export OMP_NUM_THREADS=${NCORES}
#echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

echo "Running AlphaFold on single node for ${INPUT_NAME}, NCORES=${NCORES}, RANK=${RANK}"
#numactl -N 0
time numactl -N 0 python -m torch.backends.xeon.run_cpu --ninstances 1 --ncores-per-instance ${NCORES} --rank ${RANK} run_alphafold.py \
    --db_dir=/data \
    --jackhmmer_n_cpu=${NCORES} \
    --nhmmer_n_cpu=${NCORES} \
    --run_data_pipeline=False \
    --run_inference=True \
    --json_path=${JSON_PATH}/${INPUT_NAME} \
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