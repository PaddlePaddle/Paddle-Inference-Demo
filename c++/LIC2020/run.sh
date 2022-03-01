#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. compile
bash ${work_path}/compile.sh

# 2. download model
if [ ! -d lic_model/ ]; then
    wget -q https://paddle-inference-dist.bj.bcebos.com/lic_model.tgz
    tar xzf lic_model.tgz
fi

# 3. run
./build/demo -model_file lic_model//model --params_file lic_model//params
