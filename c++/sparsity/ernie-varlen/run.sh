#!/bin/bash
./build/ernie_varlen_test --model ./models/QNLI-large-2.0 --data ./data/QNLI/dev.inference.dynamic --mode trt-fp16 --batch_size 4 —num_labels 2 --seq_lens 0 --min_graph 5 --ignore_copy 0
