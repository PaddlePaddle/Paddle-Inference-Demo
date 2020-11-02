#include "paddle/include/paddle_inference_api.h"
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <numeric>

DEFINE_string(model_dir, "./mobilenetv1", "Directory of the inference model.");
DEFINE_bool(use_gpu, false, "use_gpu");

namespace paddle_infer {

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void PrepareConfig(Config *config) {
  config->SetProgFile(FLAGS_model_dir + "/model");
  config->SetParamsFile(FLAGS_model_dir + "/params");
  if (FLAGS_use_gpu) {
    config->EnableUseGpu(100, 0);
  }
}

void Run(Predictor *predictor, int batch_size, int repeat) {
  int channels = 3;
  int height = 224;
  int width = 224;
  int input_num = channels * height * width * batch_size;

  // prepare inputs
  float *input = new float[input_num];
  for (int i = 0; i < input_num; ++i) {
    input[i] = i % 10 * 0.1;
  }

  std::string in_string;
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(input);

  // run
  auto time1 = time();
  for (size_t i = 0; i < repeat; i++) {
    CHECK(predictor->Run());
  }
  auto time2 = time();

  // get the output
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  LOG(INFO) << "batch: " << batch_size << " predict cost: "
            << time_diff(time1, time2) / static_cast<float>(repeat) << "ms"
            << std::endl;
}

void Demo(int repeat) {
  Config config;
  PrepareConfig(&config);
  auto predictor = CreatePredictor(config);
  auto pause = [](const std::string &hint) {
    std::string temp;
    LOG(INFO) << hint;
    std::getline(std::cin, temp);
  };
  pause("Pause, init predictor done, please enter any character to continue "
        "running.");

  Run(predictor.get(), 100, repeat);
  pause("Pause, batch_size=100 run done, please observe the GPU memory usage "
        "or CPU memory usage.");

  predictor->ShrinkMemory();
  pause("Pause, ShrinkMemory has been called, please observe the changes of "
        "GPU memory or CPU memory usage.");

  Run(predictor.get(), 2, repeat);
  pause("Pause, batch_size=2 run done, please observe the GPU memory usage or "
        "CPU memory usage.");
}
} // namespace paddle_infer

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle_infer::Demo(1);
  return 0;
}
