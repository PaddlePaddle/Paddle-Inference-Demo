#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. download model
if [ ! -d elmo ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/elmo.tgz
    tar xzf elmo.tgz 
fi

# 2. download data
if [ ! -d elmo_data ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/elmo/elmo_data.tgz
    tar xzf elmo_data.tgz
fi


# 3. run
python infer.py --model_file=./elmo/inference.pdmodel --params_file=./elmo/inference.pdiparams