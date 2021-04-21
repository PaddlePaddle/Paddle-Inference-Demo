#include "paddle/include/paddle_inference_api.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count();
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  } else {
    config.SetModel(FLAGS_model_file, FLAGS_params_file);
  }
  config.EnableUseGpu(1000, 3);
  //config.EnableTensorRtEngine(1 << 30, FLAGS_batch_size, 10, PrecisionType::kFloat32, false, false);
  return CreatePredictor(config);
}

void run(Predictor *predictor, const std::vector<float> &input,
         const std::vector<int> &input_shape, const std::vector<int32_t> &input_im,
         const std::vector<int> &input_im_shape, std::vector<float> *out_data) {
  auto input_names = predictor->GetInputNames();
  auto im_shape_handle = predictor->GetInputHandle(input_names[0]);
  im_shape_handle->Reshape(input_shape);
  im_shape_handle->CopyFromCpu(input.data());

  auto image_handle = predictor->GetInputHandle(input_names[1]);
  image_handle->Reshape(input_im_shape);
  image_handle->CopyFromCpu(input_im.data());

  int warmup = 10;
  int repeat = 100;

  for (int i = 0; i < warmup; i++)
    predictor->Run();

  auto time1 = time();
  for (int i = 0; i < repeat; i++) {
    predictor->Run();
  }
  auto time2 = time();
  double latency = time_diff(time1, time2) / repeat / 1000;
  std::cout << "batch: " << FLAGS_batch_size << " predict cost: " << latency << "ms" << std::endl;

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  const int height = 608;
  const int width = 608;
  const int channels = 3;
  std::vector<int> input_shape = {FLAGS_batch_size, channels, height, width};
  std::vector<float> input_data(FLAGS_batch_size * channels * height * width);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = i % 255 * 0.13f;
  }
  std::vector<int> input_im_shape = {FLAGS_batch_size, 2};
  std::vector<int32_t> input_im_data(FLAGS_batch_size * 2, 608);

  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, input_im_data, input_im_shape,
      &out_data);
  return 0;
}
