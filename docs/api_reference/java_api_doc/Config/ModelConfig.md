# 设置预测模型

## 从文件中加载预测模型 - 非Combined模型

相关方法定义如下：

```java
// 设置模型文件路径
// 参数：modelDir: String, 模型文件夹路径
// 返回：无
public void setCppModelDir(String modelDir);
// 获取非combine模型的文件夹路径
// 参数：无
// 返回：string - 模型文件夹路径
public String getCppModelDir();
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
        
        // 设置预测模型路径，这里为非 Combined 模型
        config.setCppModelDir("model路径");
        
        // 输出模型路径
        System.out.println("Non-combined model dir is: " + config.getCppModelDir());
    }
}
```



## 从文件中加载预测模型 - Combined模型

相关方法定义如下：

```java
// 设置模型文件路径
// 参数：model: String,Combined 模型文件所在路径
//      params: String,Combined 模型参数文件所在路径
// 返回：无
public void setCppModel(String modelFile, String paramsFile);

// 获取 Combined 模型的模型文件路径
// 参数：无
// 返回：string - 模型文件路径
public String getCppProgFile();

// 获取 Combined 模型的参数文件路径
// 参数：无
// 返回：string - 参数文件路径
public String getCppParamsFile();
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
        
        // 设置预测模型路径，这里为非 Combined 模型
        config.setCppModel("model路径", "params路径");
        
        // 输出模型路径
        System.out.println("Combined model path is: " + config.getCppProgFile());
        System.out.println("Combined param path is: " + config.getCppParamsFile());
    }
}
```

