import numpy as np
import paddle
import paddle.inference as paddle_infer
from gap import gap

if __name__ == '__main__':
    np_data = np.ones((32, 3, 7, 7)).astype("float32")
    model_name = "custom_gap_infer_model/custom_gap"
    config = paddle_infer.Config(model_name+".pdmodel", model_name+".pdiparams")
    config.enable_use_gpu(100, 0)
    config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=0,
            precision_mode=paddle.inference.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False,
        )
    config.set_trt_dynamic_shape_info(
        {"x": [32, 3, 7, 7]},
        {"x": [32, 3, 7, 7]},
        {"x": [32, 3, 7, 7]},
    )
    predictor = paddle_infer.create_predictor(config)
    input_tensor = predictor.get_input_handle(
        predictor.get_input_names()[0]
    )
    input_tensor.reshape(np_data.shape)
    input_tensor.copy_from_cpu(np_data.copy())
    predictor.run()
    output_tensor = predictor.get_output_handle(
        predictor.get_output_names()[0]
    )
    result = output_tensor.copy_to_cpu()
    print(np.array(result))