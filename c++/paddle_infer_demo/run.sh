mkdir -p build
cd build
rm -rf *

#DEMO_NAME=hrnet_test
DEMO_NAME=yolov3_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON
WITH_ONNXRUNTIME=ON

LIB_DIR=/pr/Paddle/build/paddle_inference_install_dir
MODEL_DIR=/paddle_infer_demo/yolov3_r34_float/

#echo $MODEL_DIR
CUDNN_LIB=/pr/nvidia/cudnn-8.1/lib64
CUDA_LIB=/usr/local/cuda-10.2/lib64

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME}

make -j

./${DEMO_NAME} --model_file=${MODEL_DIR}/model.pdmodel --params_file=${MODEL_DIR}/model.pdiparams --batch_size=1
