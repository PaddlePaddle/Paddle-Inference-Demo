// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/include/paddle_inference_api.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

int WARMUP_COUNT = 1;
int REPEAT_COUNT = 5;

const std::vector<int> INPUT_SHAPE = {1, 3, 224, 224};
const std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(label_path, "", "Path of the label.");
DEFINE_string(image_path, "", "Path of the image.");
DEFINE_string(nnadapter_device_names, "", "Names of nnadapter device");
DEFINE_string(nnadapter_context_properties,
              "",
              "Properties of nnadapter context");
DEFINE_string(nnadapter_model_cache_dir, "", "Cache dir of nnadapter model");
DEFINE_string(nnadapter_subgraph_partition_config_path,
              "",
              "Path of nnadapter subgraph partition config");

struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

bool write_file(const std::string &filename,
                const std::vector<char> &contents,
                bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "wb" : "w");
  if (!fp) return false;
  size_t size = contents.size();
  size_t offset = 0;
  const char *ptr = reinterpret_cast<const char *>(&(contents.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, 1, size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

std::vector<std::string> load_labels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

void preprocess(const float *input_image,
                const std::vector<float> &input_mean,
                const std::vector<float> &input_std,
                int input_width,
                int input_height,
                float *input_data) {
  // NHWC->NCHW
  int image_size = input_height * input_width;
  float *input_data_c0 = input_data;
  float *input_data_c1 = input_data + image_size;
  float *input_data_c2 = input_data + image_size * 2;
  int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t vmean0 = vdupq_n_f32(input_mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(input_mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(input_mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.0f / input_std[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.0f / input_std[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.0f / input_std[2]);
  for (; i < image_size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(input_image);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(input_data_c0, vs0);
    vst1q_f32(input_data_c1, vs1);
    vst1q_f32(input_data_c2, vs2);
    input_image += 12;
    input_data_c0 += 4;
    input_data_c1 += 4;
    input_data_c2 += 4;
  }
#endif
  for (; i < image_size; i++) {
    *(input_data_c0++) = (*(input_image++) - input_mean[0]) / input_std[0];
    *(input_data_c1++) = (*(input_image++) - input_mean[1]) / input_std[1];
    *(input_data_c2++) = (*(input_image++) - input_mean[2]) / input_std[2];
  }
}

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

std::vector<RESULT> postprocess(const float *output_data,
                                int64_t output_size,
                                const std::vector<std::string> &word_labels) {
  const int TOPK = 3;
  std::vector<std::pair<float, int>> vec;
  for (int i = 0; i < output_size; i++) {
    vec.push_back(std::make_pair(output_data[i], i));
  }
  std::partial_sort(
      vec.begin(), vec.begin() + TOPK, vec.end(), topk_compare_func);
  std::vector<RESULT> results(TOPK);
  for (int i = 0; i < TOPK; i++) {
    results[i].score = vec[i].first;
    results[i].class_id = vec[i].second;
    results[i].class_name = "Unknown";
    if (results[i].class_id >= 0 && results[i].class_id < word_labels.size()) {
      results[i].class_name = word_labels[results[i].class_id];
    }
  }
  return results;
}

void process(const float *input_image,
             std::vector<std::string> &word_labels,
             std::shared_ptr<Predictor> &predictor) {
  // Preprocess image and fill the data of input tensor
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);
  input_tensor->Reshape(INPUT_SHAPE);
  int input_width = INPUT_SHAPE[3];
  int input_height = INPUT_SHAPE[2];
  int num = std::accumulate(
      INPUT_SHAPE.begin(), INPUT_SHAPE.end(), 1, std::multiplies<int>());
  std::vector<float> input_data(num);
  double preprocess_start_time = get_current_us();
  preprocess(input_image,
             INPUT_MEAN,
             INPUT_STD,
             input_width,
             input_height,
             input_data.data());
  input_tensor->CopyFromCpu(input_data.data());
  double preprocess_end_time = get_current_us();
  double preprocess_time =
      (preprocess_end_time - preprocess_start_time) / 1000.0f;

  double prediction_time;
  // Run predictor
  // warm up to skip the first inference and get more stable time, remove it in
  // actual products
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  // repeat to obtain the average time, set REPEAT_COUNT=1 in actual products
  double max_time_cost = 0.0f;
  double min_time_cost = std::numeric_limits<float>::max();
  double total_time_cost = 0.0f;
  for (int i = 0; i < REPEAT_COUNT; i++) {
    auto start = get_current_us();
    predictor->Run();
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    if (cur_time_cost > max_time_cost) {
      max_time_cost = cur_time_cost;
    }
    if (cur_time_cost < min_time_cost) {
      min_time_cost = cur_time_cost;
    }
    total_time_cost += cur_time_cost;
    prediction_time = total_time_cost / REPEAT_COUNT;
    printf("iter %d cost: %f ms\n", i, cur_time_cost);
  }
  printf("warmup: %d repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
         WARMUP_COUNT,
         REPEAT_COUNT,
         prediction_time,
         max_time_cost,
         min_time_cost);

  // Get the data of output tensor and postprocess to output detected objects
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);
  auto out_shape = output_tensor->shape();
  int64_t output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  std::vector<float> output_data(output_size);
  output_tensor->CopyToCpu(output_data.data());
  double postprocess_start_time = get_current_us();
  std::vector<RESULT> results =
      postprocess(output_data.data(), output_size, word_labels);
  double postprocess_end_time = get_current_us();
  double postprocess_time =
      (postprocess_end_time - postprocess_start_time) / 1000.0f;

  printf("results: %d\n", (int)results.size());
  for (int i = 0; i < results.size(); i++) {
    printf(
        "Top%d %s - %f\n", i, results[i].class_name.c_str(), results[i].score);
  }
  printf("Preprocess time: %f ms\n", preprocess_time);
  printf("Prediction time: %f ms\n", prediction_time);
  printf("Postprocess time: %f ms\n\n", postprocess_time);
}

int main(int argc, char **argv) {
  // Load raw image data from file
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::ifstream image_file(
      FLAGS_image_path,
      std::ios::in | std::ios::binary);  // Raw RGB image with float data type
  if (!image_file) {
    printf("Failed to load image file %s\n", FLAGS_image_path.c_str());
    return -1;
  }
  size_t image_size = std::accumulate(
      INPUT_SHAPE.begin(), INPUT_SHAPE.end(), 1, std::multiplies<int>());
  std::vector<float> image_data(image_size);
  image_file.read(reinterpret_cast<char *>(image_data.data()),
                  image_size * sizeof(float));
  image_file.close();

  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableLiteEngine(paddle_infer::PrecisionType::kFloat32, true);
  config.NPU()
      .Enable()
      .SetDeviceNames({FLAGS_nnadapter_device_names})
      .SetContextProperties(FLAGS_nnadapter_context_properties)
      .SetModelCacheDir(FLAGS_nnadapter_model_cache_dir);

  std::shared_ptr<Predictor> predictor = nullptr;
  predictor = CreatePredictor(config);
  std::vector<std::string> word_labels = load_labels(FLAGS_label_path);
  for (size_t i = 0; i < 10000; ++i)
    process(image_data.data(), word_labels, predictor);
  return 0;
}
