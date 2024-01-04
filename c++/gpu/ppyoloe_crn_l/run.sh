#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d ppyoloe_crn_l ]; then
    wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar
    tar -xvf ppyoloe_crn_l_300e_coco.tar
fi

# 3. run
./build/ppyoloe_crn_l --model_file ppyoloe_crn_l_300e_coco/model.pdmodel --params_file ppyoloe_crn_l_300e_coco/model.pdiparams
