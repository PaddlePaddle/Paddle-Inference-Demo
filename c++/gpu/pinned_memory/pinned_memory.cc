#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using paddle_infer::DataLayout;
using paddle_infer::PlaceType;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(warmup, 100, "warmup.");
DEFINE_int32(repeats, 1000, "repeats.");
DEFINE_string(run_mode, "paddle_gpu", "run_mode which can be: trt_fp32, trt_fp16, trt_int8 and paddle_gpu");
DEFINE_bool(use_dynamic_shape, false, "use trt dynaminc shape.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(500, 0);
  
  if (FLAGS_run_mode == "trt_fp32") {
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 5,
                                PrecisionType::kFloat32, false, false);
  } else if (FLAGS_run_mode == "trt_fp16") {
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 5,
                                PrecisionType::kHalf, false, false);
  } else if (FLAGS_run_mode == "trt_int8") {
    config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 5,
                                PrecisionType::kInt8, false, true);
  }
  
  if(FLAGS_use_dynamic_shape){
    std::map<std::string, std::vector<int>> min_input_shape = {
        {"image", {FLAGS_batch_size, 3, 112, 112}}};
    std::map<std::string, std::vector<int>> max_input_shape = {
        {"image", {FLAGS_batch_size, 3, 448, 448}}};
    std::map<std::string, std::vector<int>> opt_input_shape = {
        {"image", {FLAGS_batch_size, 3, 224, 224}}};
    config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                  opt_input_shape);
  }

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

void run(Predictor *predictor, float* input,
         const std::vector<int> &input_shape, float* out_data, const std::vector<int> &out_shape) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  //input_t->CopyFromCpu(input);
  input_t->ShareExternalData<float>(input, input_shape, PlaceType::kGPU);
  
  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    
    output_t->ShareExternalData<float>(out_data, out_shape, PlaceType::kGPU);
    //output_t->CopyToCpu(out_data);
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();
  float* input_data;
  float* out_data;
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224};
  std::vector<int> out_shape = {FLAGS_batch_size, 1000};
  int input_data_size = FLAGS_batch_size * 3 * 224 * 224;
  int out_data_size = FLAGS_batch_size * 1000;

  cudaHostAlloc((void**)&input_data, sizeof(float) *  input_data_size, cudaHostAllocMapped);
  cudaHostAlloc((void**)&out_data, sizeof(float) * out_data_size, cudaHostAllocMapped);
  
  for (size_t i = 0; i < input_data_size; ++i)
    input_data[i] = i % 255 * 0.1; 
  
  run(predictor.get(), input_data, input_shape, out_data, out_shape);

  for (size_t i = 0; i < out_data_size; i += 100) {
    LOG(INFO) << i << " : " << out_data[i] << std::endl;
  }
  cudaFreeHost((void**)&input_data);
  cudaFreeHost((void**)&out_data);
  return 0;
}
