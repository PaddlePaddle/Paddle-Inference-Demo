# 预测示例 (R)

本章节包含2部分内容：(1) [运行 R 示例程序](#id1)；(2) [R 预测程序开发说明](#id6)。

## 运行 R 示例程序

### 1. 安装 R 预测环境

**方法1:** Paddle Inference 的 R 语言预测依赖 Paddle Python环境，请先根据 [官方主页-快速安装](https://www.paddlepaddle.org.cn/install/quick) 页面进行自行安装或编译，当前支持 pip/conda 安装，docker镜像 以及源码编译等多种方式来准备 Paddle Inference Python 环境。之后需要安装 R 运行paddle预测所需要的库

```bash
Rscript -e 'install.packages("reticulate", repos="https://cran.rstudio.com")'
```

**方法2:** 将 [Paddle/r/Dockerfile](https://github.com/PaddlePaddle/Paddle/blob/develop/r/Dockerfile) 下载到本地，使用以下命令构建 Docker 镜像，启动 Docker 容器：

```bash
# 构建 Docker 镜像
docker build -t paddle-rapi:latest .

# 启动 Docker 容器
docker run --rm -it paddle-rapi:latest bash
```

### 2. 准备预测部署模型

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 预测格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可将 `inference.pdmodel` 文件重命名为 `__model__`，然后通过模型可视化工具 Netron 打开。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

### 3. 准备预测部署程序

将以下代码保存为 `r_demo.r` 文件，并添加可执行权限：

```r
#!/usr/bin/env Rscript

library(reticulate) # call Python library
use_python("/opt/python3.7/bin/python")

np <- import("numpy")
paddle_infer <- import("paddle.inference")

predict_run_resnet50 <- function() {
    # 创建 config
    config <- paddle_infer$Config("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
    
    # 根据 config 创建 predictor
    predictor <- paddle_infer$create_predictor(config)

    # 获取输入的名称
    input_names <- predictor$get_input_names()
    input_handle <- predictor$get_input_handle(input_names[1])

    # 设置输入
    input_data <- np$random$randn(as.integer(1 * 3 * 318 * 318))
    input_data <- np_array(input_data, dtype="float32")$reshape(as.integer(c(1, 3, 318, 318)))
    input_handle$reshape(as.integer(c(1, 3, 318, 318)))
    input_handle$copy_from_cpu(input_data)

    # 运行predictor
    predictor$run()

    # 获取输出
    output_names <- predictor$get_output_names()
    output_handle <- predictor$get_output_handle(output_names[1])
    output_data <- output_handle$copy_to_cpu()
    output_data <- np_array(output_data)$reshape(as.integer(-1))
    print(paste0("Output data size is: ", output_data$size))
    print(paste0("Output data shape is: ", output_data$shape))
}

if (!interactive()) {
    predict_run_resnet50()
}
```

```r
# use_python 中指定 python 可执行文件路径
use_python("/opt/python3.7/bin/python")
```

### 4. 执行预测程序

```bash
# 将本章节第2步中下载的模型文件夹移动到当前目录
./r_demo.r
```

成功执行之后，得到的预测输出结果如下：

```bash
# 程序输出结果如下
--- Running analysis [ir_graph_build_pass]
W1202 07:44:14.075577  6224 allocator_facade.cc:145] FLAGS_use_stream_safe_cuda_allocator is invalid for naive_best_fit strategy
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [simplify_with_basic_ops_pass]
--- Running IR pass [layer_norm_fuse_pass]
---    Fused 0 subgraphs into layer_norm op.
--- Running IR pass [attention_lstm_fuse_pass]
--- Running IR pass [seqconv_eltadd_relu_fuse_pass]
--- Running IR pass [seqpool_cvm_concat_fuse_pass]
--- Running IR pass [mul_lstm_fuse_pass]
--- Running IR pass [fc_gru_fuse_pass]
---    fused 0 pairs of fc gru patterns
--- Running IR pass [mul_gru_fuse_pass]
--- Running IR pass [seq_concat_fc_fuse_pass]
--- Running IR pass [squeeze2_matmul_fuse_pass]
--- Running IR pass [reshape2_matmul_fuse_pass]
W1202 07:44:14.165925  6224 op_compat_sensible_pass.cc:219]  Check the Attr(transpose_Y) of Op(matmul) in pass(reshape2_matmul_fuse_pass) failed!
W1202 07:44:14.165951  6224 map_matmul_to_mul_pass.cc:668] Reshape2MatmulFusePass in op compat failed.
--- Running IR pass [flatten2_matmul_fuse_pass]
--- Running IR pass [map_matmul_v2_to_mul_pass]
--- Running IR pass [map_matmul_v2_to_matmul_pass]
--- Running IR pass [map_matmul_to_mul_pass]
I1202 07:44:14.169189  6224 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [fc_fuse_pass]
I1202 07:44:14.170653  6224 fuse_pass_base.cc:57] ---  detected 1 subgraphs
--- Running IR pass [repeated_fc_relu_fuse_pass]
--- Running IR pass [squared_mat_sub_fuse_pass]
--- Running IR pass [conv_bn_fuse_pass]
I1202 07:44:14.219425  6224 fuse_pass_base.cc:57] ---  detected 53 subgraphs
--- Running IR pass [conv_eltwiseadd_bn_fuse_pass]
--- Running IR pass [conv_transpose_bn_fuse_pass]
--- Running IR pass [conv_transpose_eltwiseadd_bn_fuse_pass]
--- Running IR pass [is_test_pass]
--- Running IR pass [runtime_context_cache_pass]
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I1202 07:44:14.268868  6224 analysis_predictor.cc:717] ======= optimize end =======
I1202 07:44:14.272181  6224 naive_executor.cc:98] ---  skip [feed], feed -> inputs
I1202 07:44:14.273878  6224 naive_executor.cc:98] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
[1] "Output data size is: 1000"
[1] "Output data shape is: (1000,)"
```

## R 预测程序开发说明

使用 Paddle Inference 开发 R 预测程序仅需以下五个步骤：


(1) 在 R 中引入 Paddle Python 预测库

```r
library(reticulate) # 调用Paddle
use_python("/opt/python3.7/bin/python")

np <- import("numpy")
paddle_infer <- import("paddle.inference")
```

(2) 创建配置对象，并根据需求配置，详细可参考 [Python API 文档 - Config](../api_reference/python_api_doc/Config_index)

```r
# 创建 config，并设置预测模型路径
config <- paddle_infer$Config("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
```

(3) 根据Config创建预测对象，详细可参考 [Python API 文档 - Predictor](../api_reference/python_api_doc/Predictor)

```r
predictor <- paddle_infer$create_predictor(config)
```

(4) 设置模型输入 Tensor，详细可参考 [Python API 文档 - Tensor](../api_reference/python_api_doc/Tensor)

```r
# 获取输入的名称
input_names <- predictor$get_input_names()
input_handle <- predictor$get_input_handle(input_names[1])

# 设置输入
input_data <- np$random$randn(as.integer(1 * 3 * 318 * 318))
input_data <- np_array(input_data, dtype="float32")$reshape(as.integer(c(1, 3, 318, 318)))
input_handle$reshape(as.integer(c(1, 3, 318, 318)))
input_handle$copy_from_cpu(input_data)
```

(5) 执行预测，详细可参考 [Python API 文档 - Predictor](../api_reference/python_api_doc/Predictor)

```r
predictor$run()
```

(5) 获得预测结果，详细可参考 [Python API 文档 - Tensor](../api_reference/python_api_doc/Tensor)

```r
output_names <- predictor$get_output_names()
output_handle <- predictor$get_output_handle(output_names[1])
output_data <- output_handle$copy_to_cpu()
output_data <- np_array(output_data)$reshape(as.integer(-1)) # numpy.ndarray类型
```