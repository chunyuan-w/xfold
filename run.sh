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
INPUT_NAME=7rce.json
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
    --run_data_pipeline=True \
    --run_inference=True \
    --json_path=${JSON_PATH}/${INPUT_NAME} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} 2>&1 \
    | tee -a ${LOG_DIR}/${INPUT_NAME}.log
