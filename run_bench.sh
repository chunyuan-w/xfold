# For torch kernel
# conda activate chunyuan
# TORCH_LOGS="+inductor,+output_code"  TORCHINDUCTOR_FREEZING=1  ./run_bench.sh torch compile 2>&1 | tee ../chunyuan_profile/op_bench_2048_torch_0203.log

# For xfold kernel, in docker:
# ./run_bench.sh 2>&1 | tee ../chunyuan_profile/op_bench_2048_xfold_0203.log

# export KMP_BLOCKTIME=1
# export KMP_TPAUSE=0
# export KMP_SETTINGS=1
# export KMP_AFFINITY="granularity=fine,compact,1,0"
# export KMP_FORKJOIN_BARRIER_PATTERN="dist,dist"
# export KMP_PLAIN_BARRIER_PATTERN="dist,dist"
# export KMP_REDUCTION_BARRIER_PATTERN="dist,dist"

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

if [ "$1" == "torch" ]; then
    ARGS="$ARGS --torch"
    echo "### running torch kernel"

    if [ "$2" == "compile" ]; then
        ARGS="$ARGS --torch-compile"
        echo "### running torch compile"
    elif [ "$2" == "fused" ]; then
        ARGS="$ARGS --fused"
        echo "### running fused sgl kernel (single QKV+attn+gate+out_proj op)"
    elif [ "$2" == "fused2" ]; then
        ARGS="$ARGS --fused2"
        echo "### running fused sgl kernel v2 (QKVG concat + fused attn-tail + out_proj)"
    elif [ "$2" == "fused3" ]; then
        ARGS="$ARGS --fused3"
        echo "### running fused sgl kernel v3 (per-head tiled proj + full-logit attn core)"
    else
        echo "### running eager"
    fi

else
    ARGS=""
    echo "### running xfold kernel"
fi

# Bind to the physical cores of NUMA node 0 (drop hyperthread siblings so each
# OpenMP thread gets its own core). Derived automatically from lscpu so this
# script works across machines with different NUMA / SMT layouts.
NUMA_NODE=${NUMA_NODE:-0}
PHYS_CORES=$(lscpu -p=CPU,Core,Node | awk -F, -v n=$NUMA_NODE \
    '!/^#/ && $3==n { if (!seen[$2]++) printf("%s%s", sep, $1); sep="," }')
echo "### binding to NUMA node $NUMA_NODE physical cores: $PHYS_CORES"

numactl --physcpubind=$PHYS_CORES --membind=$NUMA_NODE python -u test_grid_self_attention_baseline.py $ARGS

# numactl --physcpubind=$PHYS_CORES --membind=$NUMA_NODE python -u test_grid_self_attention.py