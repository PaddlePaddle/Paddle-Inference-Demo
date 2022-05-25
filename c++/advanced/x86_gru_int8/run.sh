#!/bin/bash
rm -rf build && mkdir build && cd build

DEMO_NAME=model_test

LIB_DIR=$HOME/test/Paddle-Inference-Demo/c++/x86_gru_int8/paddle_inference
cmake .. -DPADDLE_LIB=${LIB_DIR} -DDEMO_NAME=${DEMO_NAME} -DWITH_ONNXRUNTIME=ON && make -j && cd ..

MODEL_DIR=GRU_eval_int8
DATA_DIR=dataloader/data/test_eval_1000.bin
default_num_threads=1
default_with_accuracy=true
num_threads=${3:-$default_num_threads}
with_accuracy_layer=${4:-$default_with_accuracy}

#cgdb --args ./build/${DEMO_NAME} \
./build/${DEMO_NAME} \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_DIR} \
    --num_threads=${num_threads} \
    --with_accuracy_layer=${with_accuracy_layer}
