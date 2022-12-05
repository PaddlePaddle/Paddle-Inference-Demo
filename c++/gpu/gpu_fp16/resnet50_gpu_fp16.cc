#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/experimental/phi/common/float16.h"
#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::CreatePredictor;
using paddle_infer::Predictor;
using phi::dtype::float16;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_bool(use_gpu, false, "use gpu.");
DEFINE_bool(use_gpu_fp16, false, "use gpu fp16.");
DEFINE_bool(use_xpu, false, "use xpu");
DEFINE_bool(use_npu, false, "use npu");
DEFINE_bool(use_ipu, false, "use ipu");

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
  } else if (FLAGS_use_gpu_fp16) {
    config.EnableUseGpu(100, 0, PrecisionType::kHalf);
  } else if (FLAGS_use_xpu) {
    config.EnableXpu();
  } else if (FLAGS_use_npu) {
    config.EnableNpu();
  } else if (FLAGS_use_ipu) {
    config.EnableIpu();
  } else {
    config.EnableMKLDNN();
  }

  // Open the memory optim.
  config.EnableMemoryOptim();
  return CreatePredictor(config);
}

template <typename T>
void run(Predictor *predictor, T *const input_data,
         const std::vector<int> &input_shape, T *out_data) {
  int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int>());

  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input_data);

  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    output_t->CopyToCpu(out_data);
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();
  std::vector<int> input_shape = {FLAGS_batch_size, 3, 224, 224};
  const int input_volume = FLAGS_batch_size * 3 * 224 * 224;
  const int output_volume = FLAGS_batch_size * 1000;

  float input_data[input_volume];
  for (int i = 0; i < input_volume; i++) {
    input_data[i] = i % 255 * 0.1;
  }
  float out_data[output_volume];
  for (int i = 0; i < output_volume; i++) {
    out_data[i] = 0;
  }

  run<float>(predictor.get(), input_data, input_shape, out_data);
  for (size_t i = 0; i < output_volume; i += 100) {
    LOG(INFO) << i << " : " << out_data[i] << std::endl;
  }

  return 0;
}
