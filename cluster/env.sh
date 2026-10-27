# Cluster locations for the probes (source this before running). Data from Parthib's shared paths are linked under data/.
export CAVLA_ROOT=/data/wesleyferreiramaia/CAVLA
export CAVLA_ALPAMAYO_INTEGRATION=/data/wesleyferreiramaia/cavla_deps/carlamayo-integration
export CAVLA_R1_SRC=/data/wesleyferreiramaia/cavla_deps/alpamayo_r1_src
export PYTHONPATH=$CAVLA_ROOT/src CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=$HOME/.cache/huggingface
