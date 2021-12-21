## 运行Java ResNet50 Demo

### 安装(Linux)

##### 1.下载C预测库

您可以选择直接下载[paddle_inference_c](https://github.com/PaddlePaddle/Paddle-Inference-Demo/blob/master/docs/user_guides/download_lib.md)预测库，或通过源码编译的方式安装，源码编译方式参考官网文档，注意这里cmake编译时打开`-DON_INFER=ON`,在编译目录下得到`paddle_inference_c_install_dir`。

##### 2.准备预测部署模型

下载 [resnet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle Combined 形式的模型。

```
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得 resnet50 目录结构如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams
└── inference.pdiparams.info
```

##### 3.准备预测执行目录

```
git clone github.com/paddlepaddle/paddle/paddle/fluid/inference/javaapi
```

##### 3. 编译动态链接库和jar包

```bash
在javaapi目录下执行

./build_gpu.sh {c预测库目录} {jni头文件目录} {jni系统头文件目录}

以笔者的目录结构为例
./build.sh /root/paddle_c/paddle_inference_c_2.2/paddle_inference_c /usr/lib/jvm/java-8-openjdk-amd64/include /usr/lib/jvm/java-8-openjdk-amd64/include/linux

执行完成后，会在当前目录下生成JavaInference.jar和libpaddle_inference.so
```

##### 5.运行单测，验证

```
在javaapi目录下执行

./test.sh {c预测库目录} {.pdmodel文件目录} {.pdiparams文件目录}

以笔者的目录结构为例
./test.sh "/root/paddle_c/paddle_inference_c_2.2/paddle_inference_c"  "/root/paddle_c/resnet50/inference.pdmodel" "/root/paddle_c/resnet50/inference.pdiparams"
```

### 在Java中使用Paddle预测

##### 1.导入相关类

```java
import com.baidu.paddle.inference.Predictor;
import com.baidu.paddle.inference.Config;
import com.baidu.paddle.inference.Tensor;
```

##### 2.加载生成的动态库

```java
    static {
        System.loadLibrary("paddle_inference");
    }
```

