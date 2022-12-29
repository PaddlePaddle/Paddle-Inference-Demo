#include "custom_relu_op_plugin.h"

#include <cassert>
#include <string>
#include <vector>
#include <memory.h>

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

CustomRelu::CustomRelu(const void* buffer, size_t length){
  const char *d = reinterpret_cast<const char *>(buffer), *a = d;
  input_dims_ = read<Dims>(d);
  assert(d == a + length);
}

Dims CustomRelu::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept{
  assert(nbInputDims == 1);
  Dims output = inputs[0];
  return output;
}

int CustomRelu::enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept{
  const void* inputData = inputs[0];
  void* outputData = outputs[0];
  int numel = batchSize * ProductOfDims(input_dims_);
  int status = ReluCudaKernel(inputData, outputData, numel, stream);
  return status;
}

void CustomRelu::serialize(void* buffer) const noexcept{
  char *d = reinterpret_cast<char *>(buffer), *a = d;
  write<Dims>(d, input_dims_);
  assert(d == a + getSerializationSize());
}

void CustomRelu::configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
    DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
    bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept {
  assert(nbInputs == 1);
  assert(nbOutputs == 1);
  input_dims_ = inputDims[0];
}

IPluginV2Ext* CustomRelu::clone() const noexcept {
    auto* plugin = new CustomRelu(input_dims_);
  return plugin;
}
