mkdir -p build
cd build
rm -rf *

DEMO_NAME=model_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

LIB_DIR=/work/Paddle/build/paddle_inference_install_dir
CUDNN_LIB=/path/to/cudnn/lib
CUDA_LIB=/path/to/cuda/lib
TENSORRT_ROOT=/path/to/trt/root/dir

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
