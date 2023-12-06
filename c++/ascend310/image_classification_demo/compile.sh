#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))
LIB_DIR=${work_path}/../../lib/paddle_inference

# 1. check paddle_inference exists
if [ ! -d "${LIB_DIR}" ]; then
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

DEMO_NAME=demo

WITH_MKL=OFF
WITH_GPU=OFF
USE_TENSORRT=OFF
WITH_SHARED_PHI=ON

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_SHARED_PHI=${WITH_SHARED_PHI}

make -j
