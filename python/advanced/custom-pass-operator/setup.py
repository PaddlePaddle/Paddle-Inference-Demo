import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]


setup(
    name="custom_relu_op_pass",
    ext_modules=CUDAExtension(
        sources=[
            "./custom_relu_op_pass/custom_relu_op.cc",
            "./custom_relu_op_pass/custom_relu_op.cu",
            "./custom_relu_op_pass/custom_relu_pass.cc",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
            ] + get_gencode_flags()
        },
    ),
)
