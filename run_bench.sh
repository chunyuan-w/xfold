# For torch kernel
# conda activate chunyuan
# TORCH_LOGS="+inductor,+output_code"  TORCHINDUCTOR_FREEZING=1  ./run_bench.sh torch fused2 2>&1 | tee ../chunyuan_profile/op_bench_2048_torch_0203.log

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

# Select which op bench to run. Default keeps the original grid-self-attention
# behavior; set OP=tm (or triangle_mul) to bench TriangleMultiplication.
OP=${OP:-grid_self_attention}
case "$OP" in
    grid_self_attention|gsa)
        SCRIPT=test_grid_self_attention_baseline.py
        ;;
    triangle_multiplication|triangle_mul|tm)
        SCRIPT=test_triangle_multiplication_baseline.py
        ;;
    *)
        echo "### unknown OP=$OP (expected one of: grid_self_attention/gsa, triangle_multiplication/triangle_mul/tm)" >&2
        exit 2
        ;;
esac
echo "### op bench script: $SCRIPT"

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
    elif [ "$2" == "fused4" ]; then
        ARGS="$ARGS --fused4"
        echo "### running fused sgl kernel v4 (B-tiled qkvg scratch, L3-sized)"
    elif [ "$2" == "fused5" ]; then
        ARGS="$ARGS --fused5"
        echo "### running fused sgl kernel v5 (per-b parallel, per-thread qkvg_row)"
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

numactl --physcpubind=$PHYS_CORES --membind=$NUMA_NODE python -u $SCRIPT $ARGS

# numactl --physcpubind=$PHYS_CORES --membind=$NUMA_NODE python -u test_grid_self_attention.py