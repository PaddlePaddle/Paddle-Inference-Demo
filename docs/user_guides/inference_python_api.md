# Python 预测 API介绍

Fluid提供了高度优化的[C++预测库](./native_infer.html)，为了方便使用，我们也提供了C++预测库对应的Python接口，下面是详细的使用说明。

和C++ API接口类似，使用Python预测API预测也包含以下几个主要步骤：

- 配置推理选项
- 创建Predictor
- 准备模型输入
- 模型推理
- 获取模型输出

我们同样先从一个简单程序入手，介绍这一流程：

``` python
def create_predictor():
    # 通过AnalysisConfig配置推理选项
    config = AnalysisConfig("./resnet50/model", "./resnet50/params")
    config.switch_use_feed_fetch_ops(False)
    config.enable_use_gpu(100, 0)
    config.enable_mkldnn()
    config.enable_memory_optim()
    predictor = create_paddle_predictor(config)
    return predictor

def run(predictor, data):
    # 准备模型输入
    input_names = predictor.get_input_names()
    for i,  name in enumerate(input_names):
        input_tensor = predictor.get_input_tensor(name)
        input_tensor.reshape(data[i].shape)
        input_tensor.copy_from_cpu(data[i].copy())

    # 执行模型推理
    predictor.zero_copy_run()

    results = []
    # 获取模型输出
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_tensor(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results
```

以上的程序中`create_predictor `函数对推理过程进行了配置以及创建了Predictor。 `run `函数进行了输入数据的准备、模型推理以及输出数据的获取过程。

在接下来的部分中，我们会依次对程序中出现的AnalysisConfig，Predictor，模型输入，模型输出进行详细的介绍。

## 一、推理配置管理器AnalysisConfig
AnalysisConfig管理AnalysisPredictor的推理配置，提供了模型路径设置、推理引擎运行设备选择以及多种优化推理流程的选项。配置中包括了必选配置以及可选配置。

### 1. 必选配置
#### a.设置模型和参数路径
* non-combined形式：模型文件夹`model_dir`下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`。 使用方式为：

``` python
config.set_model("./model_dir")
```
* combined形式：模型文件夹`model_dir`下只有一个模型文件`model`和一个参数文件`params`时，传入模型文件和参数文件路径。使用方式为：

``` python
config.set_model("./model_dir/model", "./model_dir/params")
```

* 内存加载模式：如果模型是从内存加载，可以使用:

``` python
config.set_model_buffer(model_buffer, model_size, params_buffer, params_size)
```	

关于`non-combined` 以及 `combined`模型介绍，请参照[这里]()。

#### b. 关闭feed与fetch OP
`config.switch_use_feed_fetch_ops(False)  # 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项`

我们用一个小的例子来说明我们为什么要关掉它们。  

假设我们有一个模型，模型运行的序列为:
`input -> FEED_OP -> feed_out -> CONV_OP -> conv_out -> FETCH_OP -> output`

序列中大写字母的`FEED_OP`, `CONV_OP`, `FETCH_OP` 为模型中的OP， 小写字母的`input`，`feed_out`，`output` 为模型中的变量。

ZeroCopy模式下：

- 通过`predictor.get_input_tensor(input_names[0])`获取模型输入为`FEED_OP`的输出， 即`feed_out`。
- 通过`predictor.get_output_tensor(output_names[0])`接口获取的模型的输出为`FETCH_OP`的输入，即`conv_out`。

ZeroCopy的方式避免了`input->FEED_OP` 以及 `FETCH_OP->output` 的copy，从而能加速推理性能，对小的模型效果加速明显。

### 2. 可选配置
 
#### a. 加速CPU推理
 
``` python
# 开启MKLDNN，可加速CPU推理，要求预测库带MKLDNN功能。
config.enable_mkldnn()	  	  		
# 可以设置CPU数学库线程数math_threads，可加速推理。
# 注意：math_threads * 外部线程数 需要小于总的CPU的核心数目，否则会影响预测性能。
config.set_cpu_math_library_num_threads(10) 

```

#### b. 使用GPU推理

``` python
# enable_use_gpu后，模型将运行在GPU上。
# 第一个参数表示预先分配显存数目，第二个参数表示设备的ID。
config.enable_use_gpu(100, 0) 
```

如果使用的预测lib带Paddle-TRT子图功能，可以打开TRT选项进行加速： 

``` python
# 开启TensorRT推理，可提升GPU推理性能，需要使用带TensorRT的推理库
config.enable_tensorrt_engine(1 << 30,    # workspace_size   
                        	 batch_size,    # max_batch_size     
                        	 3,    # min_subgraph_size 
                       		 AnalysisConfig.Precision.Float32,    # precision 
                        	 False,    # use_static 
                        	 False,    # use_calib_mode
                        	 )
```
通过计算图分析，Paddle可以自动将计算图中部分子图融合，并调用NVIDIA的 TensorRT 来进行加速。


#### c. 内存/显存优化

``` python
config.enable_memory_optim()  # 开启内存/显存复用
```
该配置设置后，在模型图分析阶段会对图中的变量进行依赖分类，两两互不依赖的变量会使用同一块内存/显存空间，缩减了运行时的内存/显存占用（模型较大或batch较大时效果显著）。


#### d. debug开关


``` python
# 该配置设置后，会关闭模型图分析阶段的任何图优化，预测期间运行同训练前向代码一致。
config.switch_ir_optim(False)
```

``` python
# 该配置设置后，会在模型图分析的每个阶段后保存图的拓扑信息到.dot文件中，该文件可用graphviz可视化。
config.switch_ir_debug(True)
```

## 二、预测器PaddlePredictor

PaddlePredictor 是在模型上执行推理的预测器，根据AnalysisConfig中的配置进行创建。

``` python
predictor = create_paddle_predictor(config)
```

create_paddle_predictor 期间首先对模型进行加载，并且将模型转换为由变量和运算节点组成的计算图。接下来将进行一系列的图优化，包括OP的横向纵向融合，删除无用节点，内存/显存优化，以及子图（Paddle-TRT）的分析，加速推理性能，提高吞吐。


## 三：输入/输出

### 1. 准备输入

#### a. 获取模型所有输入的Tensor名字

``` python
input_names = predictor.get_input_names()
```

#### b. 获取对应名字下的Tensor

``` python
# 获取第0个输入
input_tensor = predictor.get_input_tensor(input_names[0])
```

#### c. 将输入数据copy到Tensor中

``` python
# 在copy前需要设置Tensor的shape
input_tensor.reshape((batch_size, channels, height, width))
# Tensor会根据上述设置的shape从input_data中拷贝对应数目的数据。input_data为numpy数组。
input_tensor.copy_from_cpu(input_data)
```

### 2. 获取输出
#### a. 获取模型所有输出的Tensor名字

``` python
output_names = predictor.get_output_names()
```

#### b. 获取对应名字下的Tensor

``` python
# 获取第0个输出
output_tensor = predictor.get_output_tensor(ouput_names[0])
```

#### c. 将数据copy到Tensor中

``` python
# output_data为numpy数组
output_data = output_tensor.copy_to_cpu()
```


## 下一步

看到这里您是否已经对 Paddle Inference 的 Python API 使用有所了解了呢？请访问[这里]()进行样例测试。
