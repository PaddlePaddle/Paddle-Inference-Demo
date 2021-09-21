## LSTM INT8 prediction example on X86 Linux

### 1. Prepare the prediction library
Please download the Paddle Inference Library version 2.1.3 or higher.

### 2. Run the LSTM model tests
The `run.sh` script downloads all models and files and runs all tests. 
With this file you can test:
- FP32 model with native config
- INT8 model generated with post-training quantization (PTQ) with OneDNN
- INT8 model generated from quant-aware training model (QAT) with OneDNN

In this script you can set the number of threads by setting the variable `omp_num_threads`. 

### Results from CLX 6248 machine:

#### Performance

|    FPS    |   FP32  | INT8 PTQ | INT8 QUANT2 | INT8_PTQ/FP32 | INT8 QUANT2/FP32 |
|:---------:|:-------:|:--------:|:-----------:|:-------------:|:----------------:|
|  1 thread | 4774.80 |  4997.06 |     7099.02 |          1.05 |             1.49 |
| 4 threads | 6293.63 |  6756.11 |     7973.21 |          1.07 |             1.27 |


#### Accuracy

|   ACC   |  FP32 | INT8 PTQ | INT8 QUANT2 |
|:-------:|:-----:|:--------:|:-----------:|
|  HX_ACC | 0.933 |    0.922 |       0.925 |
| CTC_ACC | 0.999 |    1.000 |       1.000 |
