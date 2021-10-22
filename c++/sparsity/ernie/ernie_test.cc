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

#include <functional>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <utility>
#include <utility>
#include <vector>

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(max_batch_size, 1, "max batch size");
DEFINE_bool(use_gpu, true, "use gpu.");
DEFINE_bool(use_trt, true, "use trt.");
DEFINE_bool(serialize, false, "serialize");
DEFINE_bool(tuned_dynamic_shape, false, "use tuned dynamic shape");
DEFINE_bool(tune, false, "tune to get shape range.");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_int32(warmup, 10, "warmup.");
DEFINE_int32(repeats, 10, "repeats.");

using Predictor = paddle_infer::Predictor;
using Config = paddle_infer::Config;

const std::string shape_range_info = "shape_range_info.pbtxt";

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); }
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void PrepareConfig(Config *config) {
  if (FLAGS_model_dir != "") {
    config->SetModel(FLAGS_model_dir);
  } else {
    config->SetModel(FLAGS_model_file, FLAGS_params_file);
  }

  if (FLAGS_use_gpu) {
    config->EnableUseGpu(500, 0);
    if (FLAGS_use_trt) {
      config->EnableTensorRtEngine(1 << 30, FLAGS_max_batch_size, 3,
                                   paddle_infer::PrecisionType::kHalf,
                                   FLAGS_serialize, false);
      if (FLAGS_tuned_dynamic_shape) {
        // config->EnableTunedTensorRtDynamicShape(shape_range_info, false);
      }
      std::map<std::string, std::vector<int>> min_input_shape = {
          {"eval_placeholder_0", {FLAGS_batch_size, 128, 1}},
          {"eval_placeholder_1", {FLAGS_batch_size, 128, 1}},
          {"eval_placeholder_2", {FLAGS_batch_size, 128, 1}},
          {"eval_placeholder_3", {FLAGS_batch_size, 128, 1}}};
      std::map<std::string, std::vector<int>> max_input_shape = {
          {"eval_placeholder_0", {256, 128, 1}},
          {"eval_placeholder_1", {256, 128, 1}},
          {"eval_placeholder_2", {256, 128, 1}},
          {"eval_placeholder_3", {256, 128, 1}}};

      std::map<std::string, std::vector<int>> opt_input_shape = {
          {"eval_placeholder_0", {FLAGS_batch_size, 128, 1}},
          {"eval_placeholder_1", {FLAGS_batch_size, 128, 1}},
          {"eval_placeholder_2", {FLAGS_batch_size, 128, 1}},
          {"eval_placeholder_3", {FLAGS_batch_size, 128, 1}}};
      config->SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                     opt_input_shape);
      config->DisableGlogInfo();
    }
  }

  if (FLAGS_tune) {
    // config->CollectShapeRangeInfo(shape_range_info);
  }
  // config->SwitchIrDebug(true);
  // LOG(INFO) << config->Summary();
}

void SingleThreadRun(
    std::shared_ptr<Predictor> predictor,
    const std::unordered_map<std::string,
                             std::pair<std::vector<int>, std::vector<int64_t>>>
        &input_info,
    std::unordered_map<std::string, std::pair<std::vector<int>,
                                              std::vector<float>>> *output_info,
    int thread_id, int batch) {
  auto in_names = predictor->GetInputNames();
  int i = 0;
  for (auto &name : in_names) {
    if (i > 2) {
      break;
    }
    auto in_handle = predictor->GetInputHandle(name);
    in_handle->Reshape(input_info.at(name).first);
    in_handle->CopyFromCpu(input_info.at(name).second.data());
    i++;
  }

  auto in_handle3 = predictor->GetInputHandle(in_names[3]);
  std::vector<float> input3(batch * 128);
  in_handle3->Reshape(std::vector<int>{batch, 128, 1});
  in_handle3->CopyFromCpu(input3.data());

  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());

  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    output_info->clear();
    auto out_names = predictor->GetOutputNames();
    for (auto &name : out_names) {
      auto out_handle = predictor->GetOutputHandle(name);
      std::vector<int> shape = out_handle->shape();
      int num = std::accumulate(shape.begin(), shape.end(), 1,
                                std::multiplies<int>());
      std::vector<float> out_data;
      if (out_handle->type() == paddle_infer::DataType::FLOAT32) {
        std::vector<float> tmp_out_data(num);
        out_handle->CopyToCpu(tmp_out_data.data());
        out_data.insert(out_data.begin(), tmp_out_data.begin(),
                        tmp_out_data.end());
      } else if (out_handle->type() == paddle_infer::DataType::INT32) {
        std::vector<int32_t> tmp_out_data(num);
        out_handle->CopyToCpu(tmp_out_data.data());
        out_data.insert(out_data.begin(), tmp_out_data.begin(),
                        tmp_out_data.end());
      } else {
        LOG(FATAL) << "not supported type.";
      }
      output_info->insert(
          std::make_pair(name, std::make_pair(shape, out_data)));
    }
  }
  std::cout << "batch_size:" << batch << std::endl;
  std::cout << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms" << std::endl;
  std::cout << thread_id << " run done." << std::endl;
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  Config config;
  PrepareConfig(&config);

  auto predictor = paddle_infer::CreatePredictor(config);
  auto in_names = predictor->GetInputNames();
  auto out_name = predictor->GetOutputNames()[0];

  std::unordered_map<std::string,
                     std::pair<std::vector<int>, std::vector<int64_t>>>
      input_infos;
  std::unordered_map<std::string,
                     std::pair<std::vector<int>, std::vector<float>>>
      output_infos;

  std::vector<int> batchs{1, 2, 4, 8, 16, 32, 64, 128, 256};
  for (auto batch : batchs) {
    std::vector<int> shape{batch, 128, 1};
    int num =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<int64_t> in_data(num);
    for (int i = 0; i < num; ++i) {
      in_data[i] = i % 50006;
    }
    input_infos[in_names[0]] = std::make_pair(shape, in_data);

    std::vector<int64_t> token(num);
    for (int i = 0; i < num; ++i) {
      token[i] = i % 2;
    }
    input_infos[in_names[1]] = std::make_pair(shape, token);

    std::vector<int64_t> tmp3(num);
    for (int i = 0; i < num; ++i) {
      tmp3[i] = i % 2;
    }
    input_infos[in_names[2]] = std::make_pair(shape, tmp3);

    SingleThreadRun(predictor, input_infos, &output_infos, 0, batch);
  }

  LOG(INFO) << "Run done";
}
