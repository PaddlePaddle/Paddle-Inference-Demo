#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. download model
if [ ! -d ernie_model_4 ]; then
    wget http://paddle-inference-dist.bj.bcebos.com/tensorrt_test/ernie_model_4.tar.gz
    tar xzf ernie_model_4.tar.gz
fi

# 2. run
python infer_ernie_varlen.py --model_dir=./ernie_model_4/  --run_mode=trt_fp16