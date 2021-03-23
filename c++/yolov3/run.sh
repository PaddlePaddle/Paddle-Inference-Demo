#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d yolov3_r50vd_dcn_270e_coco ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
    tar xzf yolov3_r50vd_dcn_270e_coco.tgz
fi

# 3. run
./build/yolov3_test -model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams