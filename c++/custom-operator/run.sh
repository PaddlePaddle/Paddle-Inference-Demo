mkdir -p build
cd build
rm -rf *

DEMO_NAME=custom_op_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

LIB_DIR=/shixiaowei02/Paddle-custom-op-src/Paddle/build/paddle_inference_install_dir
CUDNN_LIB=/usr/local/cudnn/lib64
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/root/work/nvidia/TensorRT-6.0.1.5.cuda-10.1.cudnn7.6-OSS7.2.1
CUSTOM_OPERATOR_FILES="custom_relu_op.cc;custom_relu_op.cu;custom_relu_op_dup.cc"


cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES}

make -j
