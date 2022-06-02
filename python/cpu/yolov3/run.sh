#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. download model
if [ ! -d yolov3_r50vd_dcn_270e_coco ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
    tar xzf yolov3_r50vd_dcn_270e_coco.tgz
fi

# 2. download data
if [ ! -f kite.jpg ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg
fi


# 3. run
python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams --use_gpu=1