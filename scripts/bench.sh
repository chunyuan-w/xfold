#!/bin/bash
# This script is used to bench the performance of oneCCL on multiple nodes using MPI.

cd $(dirname $0)
source env.sh || { \
    echo 'Please place your `env.sh` under `scripts` directory.'; \
    echo 'You can refer to `env.sh.example` for the content of `env.sh`.'; \
    exit 1; \
}

HOSTS=${1:-}
if [ ${NNODES} -gt 1 ]; then
    MPIRUN_ARGS="-hosts ${HOSTS} -ppn ${PPN} -n ${NNODE_MPI}"
else
    MPIRUN_ARGS="-ppn ${PPN} -n ${NNODE_MPI}"
fi

export OMP_NUM_THREADS=${AVAIL_THREADS}
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
echo "Running oneCCL benchmark, PPN=${PPN}, NNODES=${NNODES}, MASTER_PORT=${MASTER_PORT}, MASTER_ADDR=${MASTER_ADDR}, AVAIL_THREADS=${AVAIL_THREADS}"
time mpirun ${MPIRUN_ARGS} python bench.py 
