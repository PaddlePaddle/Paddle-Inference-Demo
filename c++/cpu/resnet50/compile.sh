#!/bin/bash
set +x
set -e

work_path=${PWD}

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
DEMO_NAME=resnet50_test

WITH_MKL=ON
WITH_ONNXRUNTIME=ON
WITH_GPU=OFF
WITH_ARM=OFF
WITH_MIPS=OFF
WITH_SW=OFF
WITH_XPU=OFF
USE_TENSORRT=OFF

LIB_DIR=${work_path}/../../lib/paddle_inference
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-7.1.3.4

WITH_ROCM=OFF
ROCM_LIB=/opt/rocm/lib

WITH_NPU=OFF
ASCEND_DIR=/usr/local/Ascend

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_ROCM=${WITH_ROCM} \
  -DROCM_LIB=${ROCM_LIB} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DWITH_ARM=${WITH_ARM} \
  -DWITH_MIPS=${WITH_MIPS} \
  -DWITH_SW=${WITH_SW} \
  -DWITH_XPU=${WITH_XPU} \
  -DWITH_NPU=${WITH_NPU} \
  -DASCEND_DIR=${ASCEND_DIR}

if [ "$WITH_ARM" == "ON" ];then
  make TARGET=ARMV8 -j
else
  make -j
fi

