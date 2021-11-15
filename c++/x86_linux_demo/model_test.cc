#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(model_file, "", "Path of the inference model file.");
DEFINE_string(params_file, "", "Path of the inference params file.");
DEFINE_int32(threads, 1, "CPU threads.");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Init config
  paddle_infer::Config config;
  if (FLAGS_model_dir == "") {
    config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
  } else {
    config.SetModel(FLAGS_model_dir); // Load no-combined model
  }
  config.EnableMKLDNN();
  config.SetCpuMathLibraryNumThreads(FLAGS_threads);
  config.SwitchIrOptim();
  
  // Create predictor
  auto predictor = paddle_infer::CreatePredictor(config);

  // Set input
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  std::vector<int> input_shape = {1, 3, 224, 224};
  std::vector<float> input_data(1 * 3 * 224 * 224, 1);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input_data.data());

  // Run
  predictor->Run();

  // Get output
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  auto max_iter = std::max_element(out_data.begin(), out_data.end());
  LOG(INFO) << "Output max_arg_index:" << max_iter - out_data.begin()
    << ", max_value:" << *max_iter;
  return 0;
}
