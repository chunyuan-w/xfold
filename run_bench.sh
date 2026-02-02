# TODO: add tcmallo, iomp
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
numactl --physcpubind=0-71 --membind=0 python -u test_grid_self_attention_baseline.py

# numactl --physcpubind=0-71 --membind=0 python -u test_grid_self_attention.py