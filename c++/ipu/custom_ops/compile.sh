#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "${work_path}/../../lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/c++/lib"
  exit 1
fi

# 2. compile
mkdir -p build
cd build

DEMO_NAME=custom_op_test

WITH_IPU=ON
WITH_SHARED_PHI=ON

LIB_DIR=${work_path}/../../lib/paddle_inference


cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_IPU=${WITH_IPU} \
  -DWITH_STATIC_LIB=OFF \
  -DWITH_SHARED_PHI=${WITH_SHARED_PHI} \
  -DCUSTOM_OPERATOR_FILES="custom_relu_op.cc;custom_relu_op_ipu.cc"

make -j
