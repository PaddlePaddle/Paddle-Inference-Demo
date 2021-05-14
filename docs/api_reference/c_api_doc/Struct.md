# 动态数组结构体

动态数组结构体是 Paddle Inference 为了兼顾效率和安全，创建的一组 C 结构体类型，用来进行上层和底层的数据交互。它的来源分为两种，一种是用户自己创建，这种情况由用户自己负责进行内存的回收。另外一种是来自于 Paddle Inference C api的返回值，这种情况需要用户显式地调用相应的 Destroy 函数进行对象的销毁。

## OneDimArrayInt32

OneDimArrayInt32 是 int32_t 类型的一维数组，结构体与 API 定义如下：

```c
// OneDimArrayInt32 结构体定义
typedef struct PD_OneDimArrayInt32 {
  size_t size;   // 数组长度
  int32_t* data; // 数组元素指针
} PD_OneDimArrayInt32;

// 销毁由 paddle inferecen C api 返回的 OneDimArrayInt32 对象
// 参数：array - 需要销毁的 OneDimArrayInt32 对象指针
// 返回：None
void PD_OneDimArrayInt32Destroy(PD_OneDimArrayInt32* array);
```

## OneDimArraySize

OneDimArraySize 是 size_t 类型的一维数组，结构体与 API 定义如下：

```c
// OneDimArraySize 结构体定义
typedef struct PD_OneDimArraySize {
  size_t size;   // 数组长度
  size_t* data;  // 数组元素指针
} PD_OneDimArraySize;

// 销毁由 paddle inferecen C api 返回的 OneDimArraySize 对象
// 参数：array - 需要销毁的 OneDimArraySize 对象指针
// 返回：None
void PD_OneDimArraySizeDestroy(PD_OneDimArraySize* array);
```

## OneDimArrayCstr

OneDimArrayCstr 是 const char* 类型的一维数组，结构体与 API 定义如下：

```c
// OneDimArrayCstr 结构体定义
typedef struct PD_OneDimArrayCstr {
  size_t size;   // 数组长度
  char** data;   // 数组元素指针
} PD_OneDimArraySize;

// 销毁由 paddle inferecen C api 返回的 OneDimArrayCstr 对象
// 参数：array - 需要销毁的 OneDimArrayCstr 对象指针
// 返回：None
void PD_OneDimArrayCstrDestroy(PD_OneDimArrayCstr* array);
```

## TwoDimArraySize

TwoDimArraySize 是 size_t 类型的二维数组，也可以理解为是`OneDimArraySize指针`类型的一维数组，结构体与 API 定义如下：

```c
// TwoDimArraySize 结构体定义
typedef struct PD_TwoDimArraySize {
  size_t size;   // 数组长度
  PD_OneDimArraySize** data;  // 数组元素指针
} PD_TwoDimArraySize;

// 销毁由 paddle inferecen C api 返回的 TwoDimArraySize 对象
// 参数：array - 需要销毁的 TwoDimArraySize 对象指针
// 返回：None
void PD_TwoDimArraySizeDestroy(PD_TwoDimArraySize* array);
```