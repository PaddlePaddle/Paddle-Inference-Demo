# 枚举类型

## DataType

DataType 为模型中 Tensor 的数据精度, 默认值为 PD_DATA_FLOAT32。枚举变量定义如下：

```c
// DataType 枚举类型定义
PD_ENUM(PD_DataType){
    PD_DATA_UNK = -1,
    PD_DATA_FLOAT32,
    PD_DATA_INT32,
    PD_DATA_INT64,
    PD_DATA_UINT8,
};
```
## PrecisionType

PrecisionType 为模型的运行精度，枚举变量定义如下：

```c
// PrecisionType 枚举类型定义
PD_ENUM(PD_PrecisionType){
    PD_PRECISION_FLOAT32 = 0,
    PD_PRECISION_INT8,
    PD_PRECISION_HALF,
};
```
## PlaceType

PlaceType 为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。枚举变量定义如下：

```c
// PlaceType 枚举类型定义
PD_ENUM(PD_PlaceType){PD_PLACE_UNK = -1, PD_PLACE_CPU, PD_PLACE_GPU,
                      PD_PLACE_XPU};
```