#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/extension.h"
#include "paddle_inference_api.h"

#include "cuda_runtime.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using paddle_infer::Predictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(warmup, 100, "warmup.");
DEFINE_int32(repeats, 1000, "repeats.");
DEFINE_string(run_mode, "",
              "run_mode which can be: gpu_fp16, default gpu_fp32.");

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

  if (FLAGS_run_mode == "gpu_fp16") {
    config.EnableUseGpu(500, 0, PrecisionType::kHalf);
  }

  config.SwitchIrDebug();
  config.SetOptimCacheDir("./optim_cache");

  config.EnableNewExecutor(true);
  config.EnableNewIR(true);

  config.EnableCustomPasses({"relu_replace_pass"});

  return CreatePredictor(config);
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  auto predictor = InitPredictor();

  auto input_names = predictor->GetInputNames();
  auto input_shapes = predictor->GetInputTensorShape();

  for (const auto &input_name : input_names) {
    // update input shape's batch size
    input_shapes[input_name][0] = FLAGS_batch_size;
  }

  { // warmup
    std::vector<paddle::Tensor> inputs, outputs;
    for (const auto &input_name : input_names) {
      auto input_tensor =
          paddle::full(input_shapes[input_name], 0.5, paddle::DataType::FLOAT32,
                       paddle::GPUPlace{});
      input_tensor.set_name(input_name);
      inputs.emplace_back(std::move(input_tensor));
    }
    for (size_t i = 0; i < FLAGS_warmup; ++i) {
      CHECK(predictor->Run(inputs, &outputs));
    }
    cudaDeviceSynchronize();
  }

  { // repeats
    std::vector<paddle::Tensor> inputs, outputs;
    for (const auto &input_name : input_names) {
      auto input_tensor =
          paddle::full(input_shapes[input_name], 0.5, paddle::DataType::FLOAT32,
                       paddle::GPUPlace{});
      input_tensor.set_name(input_name);
      inputs.emplace_back(std::move(input_tensor));
    }

    auto st = time();
    for (size_t i = 0; i < FLAGS_repeats; ++i) {
      CHECK(predictor->Run(inputs, &outputs));
    }
    cudaDeviceSynchronize();
    LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
              << " ms";

    CHECK(outputs.size() == 1UL);
    for (auto &output : outputs) {
      CHECK(output.place() == paddle::GPUPlace{});
      output = output.copy_to(paddle::CPUPlace{}, true);
      LOG(INFO) << output.name() << "'s data :";
      for (int64_t i = 0; i < output.numel(); i++) {
        if (output.dtype() == paddle::DataType::FLOAT32) {
          LOG(INFO) << output.data<float>()[i];
        } else if (output.dtype() == paddle::DataType::INT32) {
          LOG(INFO) << output.data<int>()[i];
        }
      }
    }
  }
  return 0;
}
