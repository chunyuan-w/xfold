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
    else
        echo "### running eager"
    fi

else
    ARGS=""
    echo "### running xfold kernel"
fi

numactl --physcpubind=0-71 --membind=0 python -u test_grid_self_attention_baseline.py $ARGS

# numactl --physcpubind=0-71 --membind=0 python -u test_grid_self_attention.py