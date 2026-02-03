# For torch kernel
# conda activate chunyuan
# TORCH_LOGS="+inductor,+output_code"  TORCHINDUCTOR_FREEZING=1  ./run_bench.sh torch compile 2>&1 | tee ../chunyuan_profile/op_bench_2048_torch_0203.log

# For xfold kernel, in docker:
# ./run_bench.sh 2>&1 | tee ../chunyuan_profile/op_bench_2048_xfold_0203.log

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

if [ "$1" == "torch" ]; then
    ARGS="$ARGS --torch"
    echo "### running torch kernel"

    if [ "$2" == "compile" ]; then
        ARGS="$ARGS --torch-compile"
        echo "### running torch compile"
    else
        ARGS=""
        echo "### running eager"
    fi

else
    ARGS=""
    echo "### running xfold kernel"
fi

numactl --physcpubind=0-71 --membind=0 python -u test_grid_self_attention_baseline.py $ARGS

# numactl --physcpubind=0-71 --membind=0 python -u test_grid_self_attention.py