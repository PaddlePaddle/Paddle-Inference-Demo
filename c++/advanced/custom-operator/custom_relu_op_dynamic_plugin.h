#pragma once

#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
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

class CustomReluDynamic : public IPluginV2DynamicExt{
public:
  CustomReluDynamic() { }

  CustomReluDynamic(const void* buffer, size_t length);

  ~CustomReluDynamic() override = default;

  const char* getPluginType() const noexcept override { return "custom_relu_paddle_trt_dynamic_plugin"; }

  const char* getPluginVersion() const noexcept override { return "1"; }

  int getNbOutputs() const noexcept override { return 1; }

  int initialize() noexcept override { return 0; }

  void terminate() noexcept override { }

  size_t getSerializationSize() const noexcept override { 
    return 0;
  }

  void serialize(void* buffer) const noexcept override;

  void destroy() noexcept override { 
    delete this; 
  }
  
  void setPluginNamespace(const char* libNamespace) noexcept override { namespace_ = libNamespace; }

  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

  /*IPluginV2Ext method*/
  nvinfer1::DataType getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override {
    return inputTypes[index];
  }

  /*IPluginV2DynamicExt method*/
  IPluginV2DynamicExt* clone() const noexcept override;

  DimsExprs getOutputDimensions(
      int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

  bool supportsFormatCombination(
      int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override{ 
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }

  void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,
    const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

  size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept override{ return 0; }

  int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
  std::string namespace_;
};

class CustomReluDynamicCreator : public IPluginCreator{
public:
  CustomReluDynamicCreator() { }

  ~CustomReluDynamicCreator() override = default;

  const char* getPluginName() const noexcept override { return "custom_relu_paddle_trt_dynamic_plugin"; }

  const char* getPluginVersion() const noexcept override { return "1"; }

  void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override { 
    plugin_namespace_ = pluginNamespace;
  }

  const char* getPluginNamespace() const noexcept override { 
    return plugin_namespace_.c_str();
  }

  const PluginFieldCollection* getFieldNames() noexcept override { return nullptr; }

  IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
    return new CustomReluDynamic();
  }

  IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override { 
    return new CustomReluDynamic(serialData, serialLength); 
  }

private:
  std::string plugin_namespace_;
};

REGISTER_TENSORRT_PLUGIN(CustomReluDynamicCreator);