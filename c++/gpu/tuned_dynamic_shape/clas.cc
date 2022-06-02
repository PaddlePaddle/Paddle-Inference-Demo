#include "paddle/include/paddle_inference_api.h"

#include <functional>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
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
DEFINE_string(hs, "224", "input heights, separeted by ':'");
DEFINE_string(ws, "224", "input widths, separeted by ':'");
DEFINE_string(no_seen_hs, "224", "no seen input heights, separeted by ':'");
DEFINE_string(no_seen_ws, "224", "no seen input widths, separeted by ':'");

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
	      // config->Exp_DisableTensorRtOPs({"elementwise_add"});
        config->EnableTunedTensorRtDynamicShape(shape_range_info, FLAGS_allow_build_at_runtime);
      }
    }
  }

  if (FLAGS_tune) {
    config->CollectShapeRangeInfo(shape_range_info);
  }

  LOG(INFO) << config->Summary();
}

void SingleThreadRun(std::shared_ptr<Predictor> predictor, const std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>>& input_info,
                     std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>>* output_info, int thread_id) {
  auto in_names = predictor->GetInputNames();
  for (auto& name : in_names) {
    auto in_handle = predictor->GetInputHandle(name);
    in_handle->Reshape(input_info.at(name).first);
    in_handle->CopyFromCpu(input_info.at(name).second.data());
  }

  CHECK(predictor->Run());

  output_info->clear();
  auto out_names = predictor->GetOutputNames();
  for (auto& name : out_names) {
    auto out_handle = predictor->GetOutputHandle(name);
    std::vector<int> shape = out_handle->shape();
    int num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> out_data(num);
    out_handle->CopyToCpu(out_data.data());
    output_info->insert(std::make_pair(name, std::make_pair(shape, out_data)));
  }
  VLOG(1) << thread_id << " run done.";
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<int> hs = GetInputShape(FLAGS_hs);
  std::vector<int> ws = GetInputShape(FLAGS_ws);
  CHECK_EQ(hs.size(), ws.size()) << "The input height size and width size should be same";
  std::vector<int> no_seen_hs = GetInputShape(FLAGS_no_seen_hs);
  std::vector<int> no_seen_ws = GetInputShape(FLAGS_no_seen_ws);
  CHECK_EQ(no_seen_hs.size(), no_seen_ws.size()) << "The input height size and width size should be same";

  Config config;
  PrepareConfig(&config);

  auto predictor = paddle_infer::CreatePredictor(config);
  auto in_name = predictor->GetInputNames()[0]; // "x"
  auto out_name = predictor->GetOutputNames()[0]; // "save_infer_model/scale_0.tmp_1"

  std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>> input_infos;
  std::unordered_map<std::string, std::pair<std::vector<int>, std::vector<float>>> output_infos;
  constexpr int channel = 3;

  for (size_t b = 1; b <= FLAGS_max_batch_size; b++) {
    for (size_t i = 0; i < hs.size(); ++i) {
      int h = hs[i];
      int w = ws[i];
      std::vector<int> shape{b, channel, h, w};
      int num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      std::vector<float> in_data(num);
      for (int i = 0; i < num; ++i) {
        in_data[i] = i % 255 * 0.13f;
      }
      input_infos[in_name] = std::make_pair(shape, in_data);
      SingleThreadRun(predictor, input_infos, &output_infos, 0);
      LOG(INFO) << "Run input shape{" << b << ", " << channel << ", " << h << ", " << w << "} done.";
    }
  }

  // if we support allow_build_at_runtime, test no seen shape and rebuild trt engine
  if (!FLAGS_tune && FLAGS_allow_build_at_runtime) {
    LOG(INFO) << "Test no seen shape and rebuild trt engine";
    int b = FLAGS_max_batch_size;
    for (size_t i = 0; i < no_seen_hs.size(); ++i) {
      int h = no_seen_hs[i];
      int w = no_seen_ws[i];
      std::vector<int> shape{b, channel, h, w};
      int num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
      std::vector<float> in_data(num);
      for (int i = 0; i < num; ++i) {
        in_data[i] = i % 255 * 0.13f;
      }
      input_infos[in_name] = std::make_pair(shape, in_data);
      SingleThreadRun(predictor, input_infos, &output_infos, 0);
      LOG(INFO) << "Run input shape{" << b << ", " << channel << ", " << h << ", " << w << "} done.";
    }
  }

  LOG(INFO) << "Run done";
}
