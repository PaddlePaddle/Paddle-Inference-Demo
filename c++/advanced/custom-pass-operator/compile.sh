#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "${work_path}/../../lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/lib"
  exit 1
fi

# 2. check CMakeLists exists
if [ ! -f "${work_path}/CMakeLists.txt" ]; then
  cp -a "${work_path}/../../lib/CMakeLists.txt" "${work_path}/"
fi

# 3. compile
mkdir -p build
cd build
rm -rf *

# same with the resnet50_test.cc
DEMO_NAME=custom_pass_test

WITH_MKL=ON
WITH_GPU=ON

LIB_DIR=${work_path}/../../lib/paddle_inference
CUDA_LIB=/usr/local/cuda/lib64

CUSTOM_OPERATOR_FILES="custom_relu_op_pass/custom_relu_op.cc;custom_relu_op_pass/custom_relu_op.cu;"
CUSTOM_PASS_FILES="custom_relu_op_pass/custom_relu_pass.cc"

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DCUDA_LIB=${CUDA_LIB} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCUSTOM_PASS_FILES=${CUSTOM_PASS_FILES} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES} \

make -j
