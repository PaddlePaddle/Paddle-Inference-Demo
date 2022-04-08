#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d MobileNetV2 ]; then
    wget http://paddle-inference-dist.bj.bcebos.com/MobileNetV2.inference.model.tar.gz
    tar xzf MobileNetV2.inference.model.tar.gz
fi

# 3. run
./build/onnxruntime_mobilenet_demo --model_file MobileNetV2/inference.pdmodel --params_file MobileNetV2/inference.pdiparams
