# X86 Linux 上部署 GRU INT8 C++预测示例

## 1 流程解析

### 1.1 准备预测库

请在[推理库下载文档](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)下载manylinux_cpu_avx_mkl_gcc82预测库，建议版本2.0.0以上

### 1.2 产出 INT8 预测模型

使用Paddle训练结束后，得到预测模型。通过quant-aware 或者 post-quantization 得到quant gru模型。
本示例准备了 gru_quant 模型，可以从[链接](https://paddle-inference-dist.cdn.bcebos.com/int8/QAT_models/GRU_quant_acc.tar.gz)下载，或者wget下载。然后使用 [save_quant_model.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py) 将 quant 模型转化为可以在 CPU 上获得加速的INT8模型，命令如下：

```
wget https://paddle-inference-dist.cdn.bcebos.com/int8/QAT_models/GRU_quant_acc.tar.gz
tar -xzvf GRU_quant_acc.tar.gz
gru_quant_model=/full/path/to/GRU_quant_acc
int8_gru_save_path=/full/path/to/int8/gru/folder
python3.6 path/to/Paddle/python/paddle/fluid/contrib/slim/tests/save_quant_model.py --quant_model_path ${gru_quant_model} --int8_model_save_path ${int8_gru_save_path} --ops_to_quantize "multi_gru"
```

转化后的INT8 模型保存在了 `${int8_gru_save_path}`所在位置

### 1.3 准备预测数据

**Note:**
- 如果为了验证gru int8性能，你可以直接从[这里](https://paddle-inference-dist.cdn.bcebos.com/gru/GRU_eval_data.tar.gz)下载已经转化好的 binary 数据, 下载好后直接跳过 1.3
- 如果要预测自己的tsv数据，根据如下步骤读取数据和转化。我们为了减少python overhead和性能最大化，暂时使用转化的bin作输入数据。你也可以写自己的c++ dataloader或者python dataloader. 读取转化数据为binary data 如下：

```
cd path/to/Paddle-Inference-Demo/c++/x86_gru_int8/dataloader

# 下载官方tsv数据，或者准备自己的tsv数据。
python downloads.py dataset

# 下载conf字典
wget https://paddle-inference-dist.cdn.bcebos.com/gru/lexical_analysis-conf-modelsrepo-v1.8.0.tar.gz
tar -xzvf lexical_analysis-conf-modelsrepo-v1.8.0.tar.gz

# Need to use data generator in paddlepaddle
pip install paddlepaddle==2.0.0 

# 转化数据
bash run_reader.sh #或者使用如下命令：
python2 my_reader.py \
    --batch_size 1 \
    --word_emb_dim 128 \
    --grnn_hidden_dim 128 \
    --bigru_num 2 \
    --test_data ./data/test.tsv \ 
    --word_dict_path ./conf/word.dic \
    --label_dict_path ./conf/tag.dic \
    --word_rep_dict_path ./conf/q2b.dic
```
修改参数：
- **test_data：** tsv格式测试数据。如果需要测试性能，官方下载的1000例的可能太小，测试性能建议复制到10000例，因为oneDNN 使用cache存储，数据太少性能不突出。


### 1.4 设置Config

根据预测部署的实际情况，设置Config。Config默认是使用CPU预测，设置开启MKLDNN加速、设置CPU的线程数、开启IR优化、开启内存优化。

```
config.SetModel(FLAGS_model_dir); // Load no-combined model
config.SetCpuMathLibraryNumThreads(FLAGS_threads);
config.EnableMKLDNN();
config.SwitchIrOptim();
config.EnableMemoryOptim();
```
**Note:**
- 如果在 VNNI支持的 CPU 上预测保存好的 INT8 模型，则以下无需设置，因为 **1.2** 准备的INT8 模型经过量化和fusion优化。
```
# 预测保存好的int8模型，以下3行可以删除
config.EnableMKLDNN();
config.SwitchIrOptim();
config.EnableMemoryOptim();
```
- 如果只测试FP32模型，则上述设置全部打开。

## 2 编译运行示例

2.1 编译示例

- 文件`model_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- 文件`CMakeLists.txt` 为编译构建文件。
- 脚本`run.sh` 包含了第三方库、预编译库的信息配置。
- 打开 `run.sh` 文件，设置 LIB_DIR 为准备的预测库路径，比如 `LIB_DIR=/work/Paddle/build/paddle_inference_install_dir`。
- 运行 `sh run.sh`,程序开始编译运行。

2.2 性能与精度

* Accuracy
  
|  Model  | FP32    | INT8   | Accuracy diff|
|---------|---------|--------|--------------|
|accuracy | 0.89326 |0.89323 |  -0.00007    |

* Performance of 6271

| Performance configuration  | Naive fp32        | int8 | Int8/Native fp32 |
|----------------------------|-------------------|------|------------------|
| bs 1, thread 1             | 1108              | 1393 | 1.26             |
| repeat 1, bs 50, thread 1  | 2175              | 3199 | 1.47             |
| repeat 10, bs 50, thread 1 | 2165              | 3334 | 1.54             |
