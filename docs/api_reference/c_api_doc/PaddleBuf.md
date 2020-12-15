#  PaddleBuf 方法

PaddleBuf 用于设置 PaddleTensor 的数据信息，主要用于对输入 Tensor 中的数据进行赋值。

## 创建 PaddleBuf 对象


```c
// 创建 PaddleBuf 对象
// 参数：None
// 返回：PD_PaddleBuf* - PaddleBuf 对象指针
PD_PaddleBuf* PD_NewPaddleBuf();

// 删除 PaddleBuf 对象
// 参数：buf - PaddleBuf 对象指针
// 返回：None
void PD_DeletePaddleBuf(PD_PaddleBuf* buf);
```

代码示例:

```c
// 创建 PaddleBuf 对象
PD_PaddleBuf* input_buffer = PD_NewPaddleBuf();

// 删除 PaddleBuf 对象
PD_DeletePaddleBuf(input_buffer);
```

## 设置 PaddleBuf 对象

```c
// 设置 PaddleBuf 的大小
// 参数：buf - PaddleBuf 对象指针
//      length - 需要设置的大小
// 返回：None
void PD_PaddleBufResize(PD_PaddleBuf* buf, size_t length);

// 重置 PaddleBuf 包含的数据和大小
// 参数：buf - PaddleBuf 对象指针
//      data - 需要设置的数据
//      length - 需要设置的大小
// 返回：None
void PD_PaddleBufReset(PD_PaddleBuf* buf, void* data, size_t length);

// 判断 PaddleBuf 是否为空
// 参数：buf - PaddleBuf 对象指针
// 返回：bool - PaddleBuf 是否为空
bool PD_PaddleBufEmpty(PD_PaddleBuf* buf);

// 获取 PaddleBuf 中的数据
// 参数：buf - PaddleBuf 对象指针
// 返回：void* - PaddleBuf 中的数据指针
void* PD_PaddleBufData(PD_PaddleBuf* buf);

// 获取 PaddleBuf 中的数据大小
// 参数：buf - PaddleBuf 对象指针
// 返回：size_t - PaddleBuf 中的数据大小
size_t PD_PaddleBufLength(PD_PaddleBuf* buf);
```

代码示例:

```c
// 创建 PaddleBuf
PD_PaddleBuf* input_buffer = PD_NewPaddleBuf();
// 判断 PaddleBuf 是否为空 - True
printf("PaddleBuf empty: %s\n", PD_PaddleBufEmpty(input_buffer) ? "True" : "False");

int input_size = 10;
float* input_data  = malloc(sizeof(float) * input_size);
int i = 0;
for (i = 0; i < input_size ; i++){ 
  input_data[i] = 1.0f;
}
// 重置 PaddleBuf 包含的数据和大小
PD_PaddleBufReset(input_buffer, (void*)(input_data), sizeof(float) * input_size);

// 获取 PaddleBuf 的大小  - 4 * 10
printf("PaddleBuf size is: %ld\n", PD_PaddleBufLength(input_buffer));
```

