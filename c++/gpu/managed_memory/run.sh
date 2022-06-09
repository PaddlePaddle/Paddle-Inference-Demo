#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d resnet50 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar xzf resnet50.tgz
fi

# 3. run
./build/managed_memory --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
