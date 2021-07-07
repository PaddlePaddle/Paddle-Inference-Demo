## 运行GO ResNet50 Demo

### GO安装Paddle Inference

1. 确认使用Paddle的CommitId

您可以通过`git log -1`的方式，确认您使用的Paddle版本的CommitId

2. 使用`go get`获取golang paddle api

```
# 此处使用上一步记录的CommitId，假设为76e5724
COMMITID=76e5724
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@${COMMITID}
```

3. 下载C预测库

您可以选择直接下载[paddle_inference_c](../../docs/user_guides/download_lib.md)，或通过源码编译的方式安装，源码编译方式参考官网文档，注意这里cmake编译时打开`-DON_INFER=ON`,在编译目录下得到`paddle_inference_c_install_dir`。


4. 软链

`go get`默认会将代码下载到`GOMODCACHE`目录下，您可以通过`go env | grep GOMODCACHE`的方式，查看该路径，在官网发布的docker镜像中该路径一般默认为`/root/gopath/pkg/mod`，进入到golang api代码路径建立软连接，将c预测库命名为`paddle_inference_c`。

```bash
eval $(go env | grep GOMODCACHE)
# 按需修改最后的goapi版本号
cd ${GOMODCACHE}/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20210517084506-76e5724c16a5/
ln -s ${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c_install_dir paddle_inference_c
```

5. 运行单测，验证

```
bash test.sh
```


### 获取Resnet50模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

### **样例编译**
 
```
go mod init demo
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@76e5724
go build .
```

### **运行样例**

```shell
./demo -thread_num 2 -work_num 100 -cpu_math 4

./demo -thread_num 16 -work_num 1000 -use_gpu

./demo -thread_num 16 -work_num 1000 -use_gpu -use_trt
```
