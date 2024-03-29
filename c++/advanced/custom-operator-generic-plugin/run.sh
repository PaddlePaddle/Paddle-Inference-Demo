#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d custom_gap_infer_model ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/custom_operator/custom_gap_infer_model.tgz
    tar xzf custom_gap_infer_model.tgz
fi

# 3. run
./build/custom_gap_op_test