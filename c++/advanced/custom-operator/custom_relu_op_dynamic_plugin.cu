#include "NvInferPlugin.h"
#include "custom_relu_op_dynamic_plugin.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <memory.h>
#include <sstream>

using namespace nvinfer1;

template <typename data_t>
__global__ void relu_cuda_forward_kernel(const data_t* x,
                                         data_t* y,
                                         const int num) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<data_t>(0.));
  }
}

int ReluCudaKernel(const void* input, void* output, const int num, const cudaStream_t& stream) {
  int block = 512;
  int grid = (num + block - 1) / block;
  relu_cuda_forward_kernel <<< grid, block, 0, stream >>>((float*)input, (float*)output, num);
  return 0;
}   

CustomReluDynamic::CustomReluDynamic(const void* buffer, size_t length) {
  const char *d = reinterpret_cast<const char *>(buffer), *a = d;
  assert(d == a + length);
}

void CustomReluDynamic::serialize(void* buffer) const noexcept{
  char *d = reinterpret_cast<char *>(buffer), *a = d;
  assert(d == a + getSerializationSize());
}

IPluginV2DynamicExt* CustomReluDynamic::clone() const noexcept {
  auto* plugin = new CustomReluDynamic();
  return plugin;
}

DimsExprs CustomReluDynamic::getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept{
  assert(nbInputs == 1);
  return inputs[0];
}

void CustomReluDynamic::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
        const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {
  assert(nbInputs == 1);
  assert(nbOutputs == 1);
}

int32_t CustomReluDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept{
  const void* inputData = inputs[0];
  void* outputData = outputs[0];
  size_t numel = 1;
  for(int i = 0; i < inputDesc->dims.nbDims; i++) {
    numel = numel*inputDesc->dims.d[i];
  }
  int status = ReluCudaKernel(inputData, outputData, numel, stream);
  return status;
}