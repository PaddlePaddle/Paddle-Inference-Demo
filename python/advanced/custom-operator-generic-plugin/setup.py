import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

sources = ['gap_op.cc', 'gap.cu']
extension = CUDAExtension
flags = {"cxx": ["-DPADDLE_WITH_CUDA", "-DPADDLE_WITH_TENSORRT"]}

extension = extension(sources=sources, extra_compile_args=flags)
setup(name='gap', ext_modules=extension)
