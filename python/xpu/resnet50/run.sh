#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. download model
if [ ! -d resnet50 ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar xzf resnet50.tgz 
fi

# 2. download data
if [ ! -f ILSVRC2012_val_00000247.jpeg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg
fi


# 3. run
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams