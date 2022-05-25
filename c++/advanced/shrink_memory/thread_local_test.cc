#include "helper.h"
#include "paddle/include/paddle_inference_api.h"
#include <chrono>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_bool(use_gpu, false, "use gpu.");
DEFINE_bool(test_leaky, false,
            "run 1000 times, and observe whether leaky memory or not.");

const size_t thread_num = 2;
paddle::inference::Timer timer_sum;
paddle::inference::Barrier barrier_init(thread_num);
paddle::inference::Barrier barrier_warmup(thread_num);

namespace paddle_infer {

void PrepareConfig(Config *config) {
  if (FLAGS_model_dir != "") {
    config->SetModel(FLAGS_model_dir);
  } else {
    config->SetModel(FLAGS_model_file, FLAGS_params_file);
  }
  if (FLAGS_use_gpu) {
    config->EnableUseGpu(500, 0);
  }
  // switch to thread_local allocator.
  config->EnableGpuMultiStream();
}

void Run(int thread_id) {
  Config config;
  PrepareConfig(&config);

  // create predictor
  static std::mutex mutex;

  std::shared_ptr<Predictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = CreatePredictor(config);
  }

  auto run_one_loop = [&](int batch_size) {
    // prepare inputs.
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

  barrier_init.Wait();
  run_one_loop(1);
  barrier_warmup.Wait();

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
          "character to continue running.");

    // release memory pool.
    predictor->TryShrinkMemory();
    pause("Pause, ShrinkMemory has been called, please observe the changes of "
          "GPU memory.");

    run_one_loop(1);
    pause("Pause, you can view the GPU memory usage, please enter any "
          "character to continue running.");
  }
}
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < thread_num; ++i) {
    threads.emplace_back([&, i]() { paddle_infer::Run(i); });
  }
  for (size_t i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
  LOG(INFO) << "Run done";
}
