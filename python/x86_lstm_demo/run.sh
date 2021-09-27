#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. download model FP32 model for post-training quantization
if [ ! -d lstm_fp32_model ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/lstm/lstm_fp32_model.tar.gz
    tar xzf lstm_fp32_model.tar.gz
fi

# 2. download model quant-aware model
if [ ! -d lstm_quant ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/lstm/lstm_quant.tar.gz
    tar xzf lstm_quant.tar.gz
fi

# 3. download data
if [ ! -f quant_lstm_input_data ]; then
    wget https://paddle-inference-dist.bj.bcebos.com/int8/unittest_model_data/quant_lstm_input_data.tar.gz
    tar xzf quant_lstm_input_data.tar.gz
fi

# 4. download save_quant_model file
if [ ! -f save_quant_model.py ]; then
    wget https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py
fi

declare -i omp_num_threads=1
export OMP_NUM_THREADS=${omp_num_threads}

# run fp32 model
 python3 model_test.py \
 --model_path=${work_path}/lstm_fp32_model \
 --data_path=${work_path}/quant_lstm_input_data \
 --use_analysis=False \
 --num_threads=${omp_num_threads}

# run ptq int8 model
 python3 model_test.py \
 --model_path=${work_path}/lstm_fp32_model \
 --data_path=${work_path}/quant_lstm_input_data \
 --use_ptq=True \
 --num_threads=${omp_num_threads}

# save quant2 int8 model
python3 save_quant_model.py \
--quant_model_path=${work_path}/lstm_quant \
--int8_model_save_path=${work_path}/quant_saved_model_int8

# run quant2 int8 model
python3 model_test.py \
--model_path=${work_path}/quant_saved_model_int8 \
--data_path=${work_path}/quant_lstm_input_data \
--use_analysis=True \
--num_threads=${omp_num_threads}