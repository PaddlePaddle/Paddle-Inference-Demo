#!/bin/bash
rm -rf build && mkdir build && cd build

DEMO_NAME=model_test

LIB_DIR=/home/li/repo/Paddle/build_c5a6ae4/paddle_inference_install_dir

#LIB_DIR=/home/lidanqing/Paddle-Inference-Demo/c++/x86_gru_int8/paddle_inference
cmake .. -DPADDLE_LIB=${LIB_DIR} -DDEMO_NAME=${DEMO_NAME} && make -j

#MODEL_DIR=/home/lidanqing/Paddle-Inference-Demo/c++/x86_gru_int8/GRU_infer_model
#DATA_DIR=/home/lidanqing/Paddle-Inference-Demo/c++/x86_gru_int8/test_eval_1w.bin

MODEL_DIR=/home/li/repo/Paddle-Inference-Demo/c++/x86_gru_int8/GRU_eval_int8
# MODEL_DIR=/home/li/repo/Paddle-Inference-Demo/c++/x86_gru_int8/GRU_eval_int8
DATA_DIR=/home/li/repo/Paddle-Inference-Demo/c++/x86_gru_int8/test_eval_1w.bin
default_num_threads=1
default_with_accuracy=true
num_threads=${3:-$default_num_threads}
with_accuracy_layer=${4:-$default_with_accuracy}

#cgdb --args ./${DEMO_NAME} \
./${DEMO_NAME} \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_DIR} \
    --num_threads=${num_threads} \
    --with_accuracy_layer=${with_accuracy_layer}
