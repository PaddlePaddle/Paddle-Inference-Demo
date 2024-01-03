#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "${work_path}/../../lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/lib"
  exit 1
fi

# 2. compile
mkdir -p build
cd build
rm -rf *

DEMO_NAME=tensorrt_precision_debug

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON
WITH_SHARED_PHI=ON

LIB_DIR=${work_path}/../../lib/paddle_inference
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6

WITH_ROCM=OFF
ROCM_LIB=/opt/rocm/lib

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_ROCM=${WITH_ROCM} \
  -DROCM_LIB=${ROCM_LIB} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DWITH_SHARED_PHI=${WITH_SHARED_PHI}
  
make -j
