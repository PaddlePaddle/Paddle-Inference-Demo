#include "paddle_inference_api.h"

#include <functional>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>
#include <utility>

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(max_batch_size, 1, "max batch size");
DEFINE_bool(use_gpu, true, "use gpu.");
DEFINE_bool(use_trt, true, "use trt.");
DEFINE_string(trt_precision, "trt_fp32", "trt_fp32, trt_fp16, etc.");
DEFINE_bool(serialize, false, "serialize");
DEFINE_bool(tuned_dynamic_shape, false, "use tuned dynamic shape");
DEFINE_bool(tune, false, "tune to get shape range.");
DEFINE_bool(allow_build_at_runtime, true, "allow rebuild trt engine at runtime");

using Predictor = paddle_infer::Predictor;
using Config = paddle_infer::Config;

const std::string shape_range_info = "shape_range_info.pbtxt";

paddle_infer::PrecisionType GetPrecisionType(const std::string& ptype) {
  if (ptype == "trt_fp32")
    return paddle_infer::PrecisionType::kFloat32;
  if (ptype == "trt_fp16")
    return paddle_infer::PrecisionType::kHalf;
  return paddle_infer::PrecisionType::kFloat32;
}

std::vector<int> GetInputShape(const std::string& s, const std::string delimiter=":") {
  std::vector<int> res;
  size_t start = 0;
  size_t end = s.find(delimiter);
  while (end != std::string::npos) {
    std::string val = s.substr(start, end - start);
    res.push_back(std::stoi(val));
    start = end + delimiter.length();
    end = s.find(delimiter, start);
  }
  if (!s.substr(start, end).empty())
    res.push_back(std::stoi(s.substr(start, end)));
  return res;
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
                                   GetPrecisionType(FLAGS_trt_precision), FLAGS_serialize, false);
      if (FLAGS_tuned_dynamic_shape) {
        config->EnableTunedTensorRtDynamicShape(shape_range_info, FLAGS_allow_build_at_runtime);
      }
    }
  }

  if (FLAGS_tune) {
    config->CollectShapeRangeInfo(shape_range_info);
  }

  LOG(INFO) << config->Summary();
}

void SingleThreadRun(std::shared_ptr<Predictor> predictor, const std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<int64_t>>>& input_info,
                     const std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>>& input_float_info,
                     std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>>* output_info, int thread_id) {
  auto in_names = predictor->GetInputNames();
  for (auto& name : in_names) {
    auto in_handle = predictor->GetInputHandle(name);
    if (input_float_info.count(name)) {
        in_handle->Reshape(input_float_info.at(name).first);
        in_handle->CopyFromCpu(input_float_info.at(name).second.data());
    } else {
        in_handle->Reshape(input_info.at(name).first);
        in_handle->CopyFromCpu(input_info.at(name).second.data());
    }
  }

  CHECK(predictor->Run());

  output_info->clear();
  auto out_names = predictor->GetOutputNames();
  for (auto& name : out_names) {
    auto out_handle = predictor->GetOutputHandle(name);
    std::vector<int> shape = out_handle->shape();
    int num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> out_data;
    if (out_handle->type() == paddle_infer::DataType::FLOAT32) {
        std::vector<float> tmp_out_data(num);
        out_handle->CopyToCpu(tmp_out_data.data());
        out_data.insert(out_data.begin(), tmp_out_data.begin(), tmp_out_data.end());
    } else if (out_handle->type() == paddle_infer::DataType::INT32) {
        std::vector<int32_t> tmp_out_data(num);
        out_handle->CopyToCpu(tmp_out_data.data());
        out_data.insert(out_data.begin(), tmp_out_data.begin(), tmp_out_data.end());
    } else {
        LOG(FATAL) << "not supported type.";
    }
    output_info->insert(std::make_pair(name, std::make_pair(shape, out_data)));
  }
  VLOG(1) << thread_id << " run done.";
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  Config config;
  PrepareConfig(&config);

  auto predictor = paddle_infer::CreatePredictor(config);
  auto in_names = predictor->GetInputNames();
  auto out_name = predictor->GetOutputNames()[0]; // "save_infer_model/scale_0.tmp_1"

  std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<int64_t>>> input_infos;
  std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>> input_float_infos;
  std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>> output_infos;

  std::vector<int32_t> seq_lens{32, 64, 128};
  for (size_t b = 1; b <= FLAGS_max_batch_size; b++) {
    for (auto s : seq_lens) {
      std::vector<int> shape{b, s, 1};
      int num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      std::vector<int64_t> in_data0(num);
      for (int i = 0; i < num; ++i) {
        in_data0[i] = i % 18000;
      }
      input_infos[in_names[0]] = std::make_pair(shape, in_data0);

      std::vector<int64_t> in_data1(num);
      for (int i = 0; i < num; ++i) {
        in_data1[i] = i % 2;
      }
      input_infos[in_names[1]] = std::make_pair(shape, in_data1);

      std::vector<int64_t> in_data2(num);
      for (int i = 0; i < num; ++i) {
        in_data2[i] = i % 2;
      }
      input_infos[in_names[1]] = std::make_pair(shape, in_data1);


      std::vector<int64_t> in_data3(num);
      for (int i = 0; i < num; ++i) {
          in_data3[i] = i % 513;
      }
      input_infos[in_names[2]] = std::make_pair(shape, in_data3);

      std::vector<float> in_data4(num);
      for (int i = 0; i < num; ++i) {
          in_data4[i] = 0.f;
      }
      input_float_infos[in_names[3]] = std::make_pair(shape, in_data4);

      SingleThreadRun(predictor, input_infos, input_float_infos, &output_infos, 0);
    }
  }

  LOG(INFO) << "Run done";
}
