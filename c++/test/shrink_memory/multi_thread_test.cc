#include "paddle/include/paddle_inference_api.h"
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

DEFINE_string(model_dir, "./mobilenetv1", "model directory.");
DEFINE_int32(thread_num, 1, "thread num");
DEFINE_bool(use_gpu, false, "use gpu.");
DEFINE_bool(test_leaky, false,
            "run 1000 times, and observe whether leaky memory or not.");

namespace paddle_infer {

void PrepareConfig(Config *config) {
  config->SetModel(FLAGS_model_dir + "/model", FLAGS_model_dir + "/params");
  if (FLAGS_use_gpu) {
    config->EnableUseGpu(500, 0);
  }
}

void Run(std::shared_ptr<Predictor> predictor, int thread_id) {

  auto run_one_loop = [&](int batch_size) {
    int channels = 3;
    int height = 224;
    int width = 224;
    int input_num = channels * height * width * batch_size;
    std::vector<float> in_data(input_num, 0);
    for (int i = 0; i < input_num; ++i) {
      in_data[i] = i % 255 * 0.1;
    }
    auto in_names = predictor->GetInputNames();
    auto in_handle = predictor->GetInputHandle(in_names[0]);
    in_handle->Reshape({batch_size, channels, height, width});
    in_handle->CopyFromCpu(in_data.data());
    CHECK(predictor->Run());
    auto out_names = predictor->GetOutputNames();
    auto out_handle = predictor->GetOutputHandle(out_names[0]);
    std::vector<float> out_data;
    std::vector<int> temp_shape = out_handle->shape();
    int output_num = std::accumulate(temp_shape.begin(), temp_shape.end(), 1,
                                     std::multiplies<int>());
    out_data.resize(output_num);
    out_handle->CopyToCpu(out_data.data());
    float mean_val = 0;
    for (size_t j = 0; j < output_num; ++j) {
      mean_val += out_data[j];
    }
    LOG(INFO) << "thread_id: " << thread_id << " batch_size: " << batch_size
              << " mean val: " << mean_val / output_num;
  };

  auto pause = [](const std::string &hint) {
    if (FLAGS_test_leaky) {
      return;
    }
    std::string temp;
    LOG(INFO) << hint;
    std::getline(std::cin, temp);
  };

  int run_times = 1;
  if (FLAGS_test_leaky) {
    run_times = 100;
  }
  for (int i = 0; i < run_times; ++i) {
    run_one_loop(40);
    pause("Pause, you can view the GPU memory usage, please enter any "
          "character to continue running. thread_id is " +
          std::to_string(thread_id));

    uint64_t bytes = predictor->TryShrinkMemory();
    LOG(INFO) << "thread_id " << thread_id << " release " << bytes / 1024. / 1024. << " MB";
    pause("Pause, ShrinkMemory has been called, please observe the changes of "
          "GPU memory. thread_idis " +
          std::to_string(thread_id));

    run_one_loop(1);
    pause("Pause, you can view the GPU memory usage, please enter any "
          "character to continue running. thread_id is " +
          std::to_string(thread_id));
  }
}
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  paddle_infer::Config config;
  paddle_infer::PrepareConfig(&config);
  auto main_predictor = paddle_infer::CreatePredictor(config);
  std::vector<decltype(main_predictor)> predictors;
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    predictors.emplace_back(std::move(main_predictor->Clone()));
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    threads.emplace_back(paddle_infer::Run, predictors[i], i);
  }
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    threads[i].join();
  }
  LOG(INFO) << "Run done";
}
