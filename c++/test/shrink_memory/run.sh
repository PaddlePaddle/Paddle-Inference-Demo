#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d mobilenetv1 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz
    tar xzf mobilenetv1.tgz
fi

# 3. run
./build/single_thread_test --model_file mobilenetv1/inference.pdmodel --params_file mobilenetv1/inference.pdiparams --use_gpu