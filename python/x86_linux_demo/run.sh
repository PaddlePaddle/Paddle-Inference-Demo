#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. download model
if [ ! -d mobilenetv1 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/mobilenetv1.tgz
    tar xzf mobilenetv1.tgz 
fi

# 2. download data
if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi


# 3. run
python model_test.py --model_file=./mobilenetv1/inference.pdmodel --params_file=./mobilenetv1/inference.pdiparams --img_path ILSVRC2012_val_00000247.jpeg