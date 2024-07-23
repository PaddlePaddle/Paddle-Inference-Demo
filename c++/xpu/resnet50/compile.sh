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
  sed -i 's/project(cpp_inference_demo CXX C CUDA)/project(cpp_inference_demo CXX C)/g' ${work_path}/CMakeLists.txt
fi

# 3. compile
mkdir -p build
cd build
rm -rf *

# same with the resnet50_test.cc
DEMO_NAME=resnet50_test

WITH_MKL=ON
WITH_ARM=OFF
WITH_XPU=ON
WITH_SHARED_PHI=ON

LIB_DIR=${work_path}/../../lib/paddle_inference

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_ARM=${WITH_ARM} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_XPU=${WITH_XPU} \
  -DWITH_STATIC_LIB=OFF \
  -DWITH_SHARED_PHI=${WITH_SHARED_PHI}

if [ "$WITH_ARM" == "ON" ];then
  make TARGET=ARMV8 -j
else
  make -j
fi
