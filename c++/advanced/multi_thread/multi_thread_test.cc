#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 100, "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_bool(use_ort, false, "use ort.");
DEFINE_int32(thread_num, 5, "thread num");

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
  if (FLAGS_use_ort) {
    // 使用onnxruntime推理
    config.EnableONNXRuntime();
    // 开启onnxruntime优化
    config.EnableORTOptimization();
  } else {
    config.EnableMKLDNN();
  }

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

void run(Predictor *predictor, int thread_id, const std::vector<float> &input_data,
         const std::vector<int> &input_shape) {
  std::vector<float> out_data;

  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input_data.data());

  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());
  }
  LOG(INFO) << "Thread " << thread_id << " run done.";

  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  auto main_predictor = InitPredictor();
  std::vector<decltype(main_predictor)> predictors;
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    predictors.emplace_back(std::move(main_predictor->Clone()));
  }
  std::vector<float> input_data(FLAGS_batch_size * 3 * 224 * 224);
  for (size_t i = 0; i < input_data.size(); ++i)
    input_data[i] = i % 255 * 0.1;
  
  std::vector<int> strides(FLAGS_thread_num + 1, 0);
  for (int i = 1; i < strides.size(); ++ i) {
      if (i == strides.size()) {
        strides[i] = FLAGS_batch_size / FLAGS_thread_num * (FLAGS_batch_size % FLAGS_thread_num + i);
      } else {
        strides[i] = FLAGS_batch_size / FLAGS_thread_num * i;
      }
  }
  
  std::vector<std::thread> threads;
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    std::vector<int> input_shape = {strides[i + 1] - strides[i], 3, 224, 224};
    std::vector<float> input_data_i(input_data.begin() + strides[i] * 3 * 224 * 224, input_data.begin() + strides[i + 1] * 3 * 224 * 224);
    threads.emplace_back(run, predictors[i].get(), i, input_data_i, input_shape);
  }

  for (int i = 0; i < FLAGS_thread_num; ++i) {
    threads[i].join();
  }
  LOG(INFO) << "Run done";
}
