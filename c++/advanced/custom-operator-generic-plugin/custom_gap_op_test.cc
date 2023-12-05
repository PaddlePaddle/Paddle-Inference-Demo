/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <numeric>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map>

#include "paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

void run(Predictor *predictor, const std::vector<float> &input,
         const std::vector<int> &input_shape, std::vector<float> *out_data) {
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape(input_shape);
  input_t->CopyFromCpu(input.data());

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());
}

int main() {
  paddle::AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel("custom_gap_infer_model/custom_gap.pdmodel",
                  "custom_gap_infer_model/custom_gap.pdiparams");
  config.EnableTensorRtEngine(1 << 30, 1, 1, paddle_infer::PrecisionType::kFloat32, true, false);
  std::map<std::string, std::vector<int>> min_input_shape = {{"x", {32, 3, 7, 7}}};
  std::map<std::string, std::vector<int>> max_input_shape = {{"x", {32, 3, 7, 7}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {{"x", {32, 3, 7, 7}}};
  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,opt_input_shape);

  auto predictor{paddle_infer::CreatePredictor(config)};
  std::vector<int> input_shape = {32, 3, 7, 7};
  std::vector<float> input_data(32 * 3 * 7 * 7, 0.5f);
  std::vector<float> out_data;
  run(predictor.get(), input_data, input_shape, &out_data);
  for (auto e : out_data) {
    LOG(INFO) << e << '\n';
  }
  return 0;
}
