# 使用 CPU 进行预测

**注意：**

1. 在 CPU 型号允许的情况下，进行预测库下载或编译试尽量使用带 AVX 和 MKL 的版本。
2. 可以尝试使用 Intel 的 MKLDNN 进行 CPU 预测加速，默认 CPU 不启用 MKLDNN。

3. 在 CPU 可用核心数足够时，可以通过设置 `SetCpuMathLibraryNumThreads` 将线程数调高一些，默认线程数为 1。



## CPU 设置

相关方法定义如下：

```java
// 设置 CPU Blas 库计算线程数
// 参数：mathThreadsNum: int, blas库计算线程数
// 返回：无
public void setCpuMathLibraryNumThreads(int mathThreadsNum);

// 获取 CPU Blas 库计算线程数
// 参数：无
// 返回：int - cpu blas 库计算线程数
public int getCpuMathLibraryNumThreads();
```

代码示例：

```java
import com.baidu.paddle.inference.Config;

public class Hello {

    static {
        System.load(".so文件路径");
    }

    public static void main(String[] args) {
        // 创建 Config 对象
        Config config = new Config();
        
        // 设置模型线程数
        config.setCpuMathLibraryNumThreads(10);
            
        // 输出模型线程数
        System.out.println("Threads Num dir is: " + config.getCpuMathLibraryNumThreads());
    }
}
```



## MKLDNN 设置

**注意：**

1. 启用 MKLDNN 的前提为已经使用 CPU 进行预测，否则启用 MKLDNN 无法生效。
2. 启用 MKLDNN BF16 要求 CPU 型号可以支持 AVX512，否则无法启用 MKLDNN BF16。



相关方法定义如下：

```java
// 启用 MKLDNN 进行预测加速
// 参数：无
// 返回：无
public void enableMKLDNN();

// 判断是否启用 MKLDNN
// 参数：无
// 返回：boolean - 是否启用 MKLDNN
public boolean mkldnnEnabled();

// 启用 MKLDNN BFLOAT16
// 参数：无
// 返回：无
public void enableMkldnnBfloat16();

// 判断是否启用 MKLDNN BFLOAT16
// 参数：无
// 返回：boolean - 是否启用 MKLDNN BFLOAT16
public boolean mkldnnBfloat16Enabled()
```



代码示例：

```java
import com.baidu.paddle.inference.Config;

public class Hello {

    static {
        System.load(".so文件路径");
    }

    public static void main(String[] args) {
        // 创建 Config 对象
        Config config = new Config();
        
        // 启用 MKLDNN 进行预测
        config.enableMKLDNN();
        
        // 通过 API 获取 MKLDNN 启用结果 - true
        System.out.println("Enable MKLDNN is: " + config.mkldnnEnabled());
        
        // 启用 MKLDNN BFLOAT16 进行预测
    	config.enableMkldnnBfloat16();
            
        // 通过 API 获取 MKLDNN BFLOAT16 启用结果
    	// 如果当前CPU支持AVX512，则返回 true, 否则返回 false
        System.out.println("Enable MKLDNN BF16 is: " + config.mkldnnBfloat16Enabled());
    }
}
```

