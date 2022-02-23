#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cuda_runtime.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::DataLayout;
using paddle_infer::PlaceType;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_bool(use_gpu, false, "use gpu.");

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
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(100, 0);
  } else {
    config.EnableMKLDNN();
  }

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

void run(Predictor *predictor, float* input,
         const std::vector<int> &input_shape, float *output) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  if (FLAGS_use_gpu){
    input_t->ShareExternalData<float>(input, input_shape, PlaceType::kGPU);
    output_t->ShareExternalData<float>(output, {FLAGS_batch_size, 1000}, PlaceType::kGPU);
  }else{
    input_t->ShareExternalData<float>(input, input_shape, PlaceType::kCPU);
    output_t->ShareExternalData<float>(output, {FLAGS_batch_size, 1000}, PlaceType::kCPU);
  }

  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224};
  std::vector<float> input_data(FLAGS_batch_size * 3 * 224 * 224);
  for (size_t i = 0; i < input_data.size(); ++i)
    input_data[i] = i % 255 * 0.1;
  std::vector<float> out_data;
  out_data.resize(FLAGS_batch_size * 1000);
  if (FLAGS_use_gpu) {
    float* input;
    cudaMalloc((void **) &input, FLAGS_batch_size * 3 * 224 * 224 * sizeof(float));
    cudaMemcpy(input, input_data.data(), FLAGS_batch_size * 3 * 224 * 224 * sizeof(float), cudaMemcpyHostToDevice);

    float* output;  
    cudaMalloc((void **) &output, FLAGS_batch_size * 1000 * sizeof(float));
    run(predictor.get(), input, input_shape, output);
    cudaMemcpy(out_data.data(), output, FLAGS_batch_size * 1000 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(input);
    cudaFree(output);
  } else {
    run(predictor.get(), input_data.data(), input_shape, out_data.data());
  }

  for (size_t i = 0; i < out_data.size(); i += 100) {
    LOG(INFO) << i << " : " << out_data[i] << std::endl;
  }
  return 0;
}
