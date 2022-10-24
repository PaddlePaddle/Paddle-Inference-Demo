#pragma once

#include "NvInferPlugin.h"

#include <cassert>
#include <string>
#include <vector>

using namespace nvinfer1;

inline size_t ProductOfDims(Dims dims) {
  size_t result = 1;
  for(size_t i = 0; i < dims.nbDims; i++) {
    result *= dims.d[i];
  }
  return result;
}

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer)
{
    T val{};
    memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
    return val;
}

class CustomRelu : public IPluginV2Ext{
public:
  CustomRelu() { }

  CustomRelu(Dims input_dims) : input_dims_(input_dims) { }

  CustomRelu(const void* buffer, size_t length);

  ~CustomRelu() override = default;

  const char* getPluginType() const noexcept override { return "custom_relu_paddle_trt_plugin"; }

  const char* getPluginVersion() const noexcept override { return "1"; }

  int getNbOutputs() const noexcept override { 
    return 1; 
  }

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

  bool supportsFormat(DataType type, PluginFormat format) const noexcept override{ return true; }

  int initialize() noexcept override { 
    return 0; 
  }

  void terminate() noexcept override { }

  size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace,
      cudaStream_t stream) noexcept override;

  size_t getSerializationSize() const noexcept override { 
    return sizeof(Dims);
  }

  void serialize(void* buffer) const noexcept override;

  void destroy() noexcept override { 
    delete this; 
  }
  
  void setPluginNamespace(const char* libNamespace) noexcept override { namespace_ = libNamespace; }

  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  /*IPluginV2Ext method*/
  nvinfer1::DataType getOutputDataType(
      int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override {return inputTypes[index];}

  bool isOutputBroadcastAcrossBatch(
      int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept override {return false;}

  bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override {return false;}

  void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
      DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
      bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept override;

  IPluginV2Ext* clone() const noexcept override;

private:
  Dims input_dims_;
  std::string namespace_;
};

class CustomReluPluginCreator : public IPluginCreator{
public:
  CustomReluPluginCreator() { }

  ~CustomReluPluginCreator() override = default;

  const char* getPluginName() const noexcept override { return "custom_relu_paddle_trt_plugin"; }

  const char* getPluginVersion() const noexcept override { return "1"; }

  void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override { 
      plugin_namespace_ = pluginNamespace;
  }

  const char* getPluginNamespace() const noexcept override { 
    return plugin_namespace_.c_str();
  }

  const PluginFieldCollection* getFieldNames() noexcept override { return nullptr; }

  IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
    return new CustomRelu();
  }

  IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override { 
    return new CustomRelu(serialData, serialLength); 
  }

private:
  std::string plugin_namespace_;
};

REGISTER_TENSORRT_PLUGIN(CustomReluPluginCreator);