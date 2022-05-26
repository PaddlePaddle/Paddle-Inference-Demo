## LSTM INT8 prediction example on X86 Linux

### 1. Prepare the PaddlePaddle release/2.2

```
git clone https://github.com/PaddlePaddle/Paddle.git
git checkout release/2.2
```
cd /paddle/repo/Paddle/build
```
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_GPU=OFF \
      -DWITH_AVX=ON \
      -DWITH_DISTRIBUTE=OFF \
      -DWITH_MKLDNN=ON \
      -DON_INFER=ON \
      -DWITH_TESTING=ON \
      -DWITH_INFERENCE_API_TEST=ON \
      -DWITH_NCCL=OFF \
      -DWITH_PYTHON=ON \
      -DPY_VERSION=3.7 \
      -DWITH_LITE=OFF .. \
```
`make -j12`

`cd python/dist`

`pip install paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl`

export PYTHONPATH=/paddle/repo/Paddle/build/paddle/:/paddle/repo/Paddle/build/python


## 2. Prepare the LSTM test files
```
git clone https://github.com/Paddle/Paddle-Inference-Demo.git
cd Paddle-Inference-Demo/python/x86_lstm_demo
```

### 3. Run the LSTM tests

- The `run.sh` is test script, in it you set the threads number by setting variable `omp_num_threads`.
- To run the test, do
```
bash run.sh
```
You will get results as follows:
- First part is FP32 results, 
- Second part is INT8 post-training results, 
- Third part is INT8 quant-aware results.
```
...
...

--- Running analysis [ir_graph_to_program_pass]
I0924 11:22:13.081768   375 analysis_predictor.cc:699] ======= optimize end =======
I0924 11:22:13.100350   375 device_context.cc:641] oneDNN v2.3.2
FPS 4803.7807358377595, HX_ACC 0.9330669330669331, CTC_ACC 0.9989680082559339

...
...

--- Running analysis [ir_graph_to_program_pass]
I0924 11:22:23.296375   386 mkldnn_quantizer.cc:598] == optimize 2 end ==
FPS 7155.112160899545, HX_ACC 0.9200799200799201, CTC_ACC 1.0

...
...

--- Running analysis [ir_graph_to_program_pass]
I0924 11:22:32.275900   408 analysis_predictor.cc:699] ======= optimize end =======
I0924 11:22:32.280690   408 device_context.cc:641] oneDNN v2.3.2
FPS 7191.383517594144, HX_ACC 0.9250749250749251, CTC_ACC 1.0

```

In this script you can set the number of threads by setting the variable `omp_num_threads`. 

### Results from CLX 6271 machine:

#### Performance

|    FPS    |   FP32  | INT8 PTQ | INT8 QUANT2 | INT8_PTQ/FP32 | INT8 QUANT2/FP32 |
|:---------:|:-------:|:--------:|:-----------:|:-------------:|:----------------:|
|  1 thread | 4895.65 |  7166.44 |     7190.55 |          1.46 |             1.47 |
| 4 threads | 6370.86 |  8026.94 |     7942.51 |          1.26 |             1.25 |


#### Accuracy

|   ACC   |  FP32 | INT8 PTQ | INT8 QUANT2 |
|:-------:|:-----:|:--------:|:-----------:|
|  HX_ACC | 0.933 |    0.920 |       0.925 |
| CTC_ACC | 0.999 |    1.000 |       1.000 |
