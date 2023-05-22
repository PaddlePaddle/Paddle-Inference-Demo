#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cuda_runtime.h>

#include "paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_bool(use_ort, false, "use ort.");
DEFINE_int32(thread_num, 2, "thread num");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::shared_ptr<Predictor> InitPredictor(void *stream) {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(500, 0);

  // use external stream.
  config.SetExecStream(stream);

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

void run(Predictor *predictor, int thread_id) {
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224};
  std::vector<float> input_data(FLAGS_batch_size * 3 * 224 * 224);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = i % 255 * 0.1;
  }
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
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<cudaStream_t> streams(FLAGS_thread_num);
  for (size_t i = 0; i < FLAGS_thread_num; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  {
    // Please make sure the stream is valid.
    auto main_predictor = InitPredictor(streams[FLAGS_thread_num - 1]);
    std::vector<decltype(main_predictor)> predictors;
    for (int i = 0; i < FLAGS_thread_num - 1; ++i) {
      predictors.emplace_back(std::move(main_predictor->Clone(streams[i])));
      LOG(INFO) << "Predictor[" << i << "] stream is "
                << predictors[i]->GetExecStream();
    }
    predictors.emplace_back(std::move(main_predictor));
    LOG(INFO) << "Predictor[" << FLAGS_thread_num - 1 << "] stream is "
              << predictors.back()->GetExecStream();

    std::vector<std::thread> threads;
    for (int i = 0; i < FLAGS_thread_num; ++i) {
      threads.emplace_back(run, predictors[i].get(), i);
    }
    for (int i = 0; i < FLAGS_thread_num; ++i) {
      threads[i].join();
    }

    LOG(INFO) << "Run done, Destroy predictor";
  }

  for (size_t i = 0; i < FLAGS_thread_num; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  LOG(INFO) << "Destroy stream.";
}
