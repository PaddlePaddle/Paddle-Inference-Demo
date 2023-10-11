#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iterator>
#include <functional>
#include <unistd.h>
#include <float.h>
#include <thread>

#include <sys/stat.h>

#include "cuda_runtime.h"
#include "paddle_inference_api.h"
#include "paddle/extension.h"
#include "utils/table_printer.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(run_mode, "trt_fp32", "mode which can be: trt_fp32, trt_fp16, trt_int8");
DEFINE_string(shapes, "", "input_shapes");
DEFINE_string(collect_shape, "", "collect shape.");
DEFINE_string(shape_file, "", "collect shape file.");
DEFINE_bool(use_calib, false, "use trt int8 calibration.");
DEFINE_bool(use_cuda_graph, false, "use cuda_graph.");
DEFINE_bool(memory_optim, false, "use memory optimize.");
DEFINE_double(rtol, 1e-4, "relative tolerance");
DEFINE_double(atol, 1e-4, "absolute tolerance");
DEFINE_string(cache_dir, "./cache/", "cache_dir.");

// info used in hook funcs
namespace check_diff {
  std::vector<std::string> tensor_names;
  std::unordered_map<std::string, std::string> tensor2op;
  std::unordered_map<std::string, paddle::Tensor> baseline_tensor;
  std::unordered_map<std::string, std::vector<std::string>> tensor_match_info;
  std::vector<std::string> mark_tensor_names;
}

void save_baseline_hook (const std::string &op_type,
                        const std::string &tensor_name,
                        const paddle::Tensor &tensor) {
  LOG(INFO) << ">>>> save_baseline_hook: " << op_type << " " << tensor_name;
  auto cpu_tensor = tensor.copy_to(paddle::CPUPlace(), true);
  check_diff::tensor_names.emplace_back(tensor_name);
  check_diff::tensor2op[tensor_name] = op_type;
  check_diff::baseline_tensor[tensor_name] = cpu_tensor;
}

template <typename T>
int assert_close(const T& actual, const T& desired, const double& atol, const double& rtol, double& max_atol, double& max_rtol) {
  double cur_atol = static_cast<double>(std::abs(actual - desired));
  double cur_rtol = cur_atol / (std::abs(desired) + 1e-7);
  max_atol = std::max(cur_atol, max_atol);
  max_rtol = std::max(cur_rtol, max_rtol);
  return !std::isfinite(cur_atol) || !std::isfinite(cur_rtol) || (cur_atol > atol + rtol * std::abs(desired)) ? 1 : 0;
}

void assert_tensor_close_hook(const std::string &op_type,
                       const std::string &tensor_name,
                       const paddle::Tensor &tensor) {
  LOG(INFO) << ">>>> assert_tensor_close_hook: " << op_type << " " << tensor_name;
  auto actual = tensor.copy_to(paddle::CPUPlace(), true);
  if (check_diff::baseline_tensor.count(tensor_name)) {
    auto desired = check_diff::baseline_tensor[tensor_name];
    
    std::vector<std::string> match_status;
    // match_status.emplace_back(op_type); // present op type after ir optim
    match_status.emplace_back(check_diff::tensor2op[tensor_name]); // present op type before ir optim    
    match_status.emplace_back(tensor_name);
   
    // convert shape to string
    auto get_shape = [](const paddle::Tensor& tmp_tensor) {
      std::string shape_str;
      shape_str += "[" + std::to_string(tmp_tensor.shape()[0]);
      for (size_t i = 1; i < tmp_tensor.shape().size(); i++) {
        shape_str += "," + std::to_string(tmp_tensor.shape()[i]);
      }
      shape_str += "]";
      return shape_str;
    };

    std::string shapeMatch;
    int mis_match_num = 0;
    double max_atol = 0., max_rtol = 0.;
    double min_base = DBL_MAX, min_cur = DBL_MAX;
    double max_base = -DBL_MAX, max_cur = -DBL_MAX;
    auto updateMinMax = [&min_base, &min_cur, &max_base, &max_cur](double actual_val, double desired_val) {
      min_base = std::min(min_base, desired_val);
      min_cur = std::min(min_cur, actual_val);
      max_base = std::max(max_base, desired_val);
      max_cur =  std::max(max_cur, actual_val);
    };

    if (actual.shape() != desired.shape()) {
      shapeMatch = "expect " + get_shape(desired) +  ", but got " + get_shape(actual);
      match_status.emplace_back(shapeMatch);
    } else {
      assert(actual.numel() == desired.numel());
      shapeMatch = "match, " + get_shape(desired);
      match_status.emplace_back(shapeMatch);
      for (size_t i = 0; i < desired.numel(); i++) {
        if (tensor.dtype() == paddle::DataType::INT32) {
          mis_match_num += assert_close<int32_t>(actual.data<int32_t>()[i], desired.data<int32_t>()[i], FLAGS_atol, FLAGS_rtol, max_atol, max_rtol);
          updateMinMax(actual.data<int32_t>()[i], desired.data<int32_t>()[i]);
        } else if (tensor.dtype() == paddle::DataType::INT64) {
          mis_match_num += assert_close<int64_t>(actual.data<int64_t>()[i], desired.data<int64_t>()[i], FLAGS_atol, FLAGS_rtol, max_atol, max_rtol);
          updateMinMax(actual.data<int64_t>()[i], desired.data<int64_t>()[i]);
        } else if (tensor.dtype() == paddle::DataType::FLOAT32) {
          mis_match_num += assert_close<float>(actual.data<float>()[i], desired.data<float>()[i], FLAGS_atol, FLAGS_rtol, max_atol, max_rtol);
          updateMinMax(actual.data<float>()[i], desired.data<float>()[i]);
        } else if (tensor.dtype() == paddle::DataType::FLOAT64) {
          mis_match_num += assert_close<double>(actual.data<double>()[i], desired.data<double>()[i], FLAGS_atol, FLAGS_rtol, max_atol, max_rtol);
          updateMinMax(actual.data<double>()[i], desired.data<double>()[i]);
        } else if (tensor.dtype() == paddle::DataType::BOOL) {
          mis_match_num += assert_close<int>(static_cast<int>(actual.data<bool>()[i]), static_cast<int>(desired.data<bool>()[i]), FLAGS_atol, FLAGS_rtol, max_atol, max_rtol);
          updateMinMax(static_cast<int>(actual.data<bool>()[i]), static_cast<int>(desired.data<bool>()[i]));
        } else if (tensor.dtype() == paddle::DataType::FLOAT16) {
          mis_match_num += assert_close<float>(static_cast<float>(actual.data<phi::dtype::float16>()[i]), static_cast<float>(desired.data<phi::dtype::float16>()[i]), FLAGS_atol, FLAGS_rtol, max_atol, max_rtol);
          updateMinMax(static_cast<int>(actual.data<bool>()[i]), static_cast<int>(desired.data<bool>()[i]));
        } else {
          LOG(WARNING) << "Unsupported data type " << tensor.dtype();
          return;
        }
      }
      match_status.emplace_back(std::to_string(mis_match_num) + "/" + std::to_string(desired.numel()));
      match_status.emplace_back(std::to_string(max_atol));
      match_status.emplace_back(std::to_string(max_rtol));
      std::string min_info;
      min_info += min_cur == DBL_MAX ? "nan" : std::to_string(min_cur);
      min_info += min_base == DBL_MAX ? "(nan)" : "(" + std::to_string(min_base) + ")";
      match_status.emplace_back(min_info);
      std::string max_info;
      max_info += max_cur == -DBL_MAX ? "nan" : std::to_string(max_cur);
      max_info += max_base == -DBL_MAX ? "(nan)" : "(" + std::to_string(max_base) + ")";
      match_status.emplace_back(max_info);
    }
    check_diff::tensor_match_info[tensor_name] = match_status;
  } else {
    LOG(WARNING) << "Tensor " << tensor_name << " not found in paddle inference with ir optim off.";
  }
}

void run(Predictor *predictor, std::unordered_map<std::string, std::vector<int>>& input_shapes, std::unordered_map<std::string, std::vector<float>>& input_datas) {
  auto input_names = predictor->GetInputNames();
  std::unordered_set<std::string> input_name_set(input_names.begin(), input_names.end());
  for (auto it : input_shapes) {
    auto name = it.first;
    if (input_name_set.count(name)) {
      auto& shape = it.second;
      auto handle = predictor->GetInputHandle(name);
      std::vector<int> input_shape = handle->shape();
      handle->Reshape(shape);
      handle->CopyFromCpu(input_datas.at(name).data());
    }
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(predictor->GetExecStream());
  CHECK(predictor->Run());
  cudaStreamSynchronize(stream);
}

std::shared_ptr<Predictor> InitBaselinePredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(500, 0);
  config.SetOptimCacheDir(FLAGS_cache_dir);
  config.SwitchIrOptim(false);
  return CreatePredictor(config);
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(500, 0);
  config.SetOptimCacheDir(FLAGS_cache_dir);
  config.EnableTunedTensorRtDynamicShape(FLAGS_shape_file, false);

  if (FLAGS_run_mode == "trt_fp32") {
    config.EnableTensorRtEngine(1 << 30, 1, 1,
                                PrecisionType::kFloat32, true, false, FLAGS_use_cuda_graph);
  } else if (FLAGS_run_mode == "trt_fp16") {
    config.EnableTensorRtEngine(1 << 30, 1, 1,
                                PrecisionType::kHalf, true, false, FLAGS_use_cuda_graph);
  } else if (FLAGS_run_mode == "trt_int8") {
    config.EnableTensorRtEngine(1 << 30, 1, 1,
                                PrecisionType::kInt8, false, FLAGS_use_calib, FLAGS_use_cuda_graph);
  }

  // serialize engine info to json
  config.EnableTensorRtInspector(true);

  // mark tensorrt outputs for checking diff
  config.MarkTrtEngineOutputs(check_diff::mark_tensor_names);
  
  // Open the memory optim.
  if (FLAGS_memory_optim) {
    config.EnableMemoryOptim();
  }

  config.SwitchIrDebug(true);

  // config.Exp_DisableTensorRtOPs({});
  // config.pass_builder()->DeletePass("");

  LOG(INFO) << config.Summary();
  return CreatePredictor(config);
}

// 1x2x3x4 -> vector<int>{1,2,3,4}
std::vector<int> ProcessShapeString(const std::string& shape) {
  std::vector<int> data;
  int start = 0;
  int seg = shape.find("x", start);
  while (seg != std::string::npos) {
    int val = std::stoi(shape.substr(start, seg - start));
    data.push_back(val);
    start = seg + 1;
    seg = shape.find("x", start);
  }
  data.push_back(std::stoi(shape.substr(start)));
  return data;
}

// x:1x2,y:2x3 -> vector<string>{"x:1x2", "y:2x3"}
std::vector<std::string> ProcessMultiShape(const std::string& shape) {
  std::vector<std::string> res;
  int start = 0;
  int seg = shape.find(",", start);
  while(seg != std::string::npos) {
    res.push_back(shape.substr(start, seg - start));
    start = seg + 1;
    seg = shape.find(",", start);
  }
  res.push_back(shape.substr(start));
  return res;
}

std::unordered_map<std::string, std::vector<int>> GetInputShape(const std::string& shape) {
  std::unordered_map<std::string, std::vector<int>> res;
  auto multi_shapes = ProcessMultiShape(shape);
  for (auto& s : multi_shapes) {
    auto seg = s.find(":");
    CHECK_NE(seg, std::string::npos);
    auto name = s.substr(0, seg);
    auto shape_val = ProcessShapeString(s.substr(seg+1));
    res[name] = shape_val;
  }
  return res;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // make cache dir
  if (access(FLAGS_cache_dir.c_str(), F_OK) == -1) {
    int status = mkdir(FLAGS_cache_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (status != 0) {
      LOG(FATAL) << "mkdir failed: " << FLAGS_cache_dir;
    }
  }

  // init predictor to get input shapes
  auto predictor_baseline = InitBaselinePredictor();
  // input shape
  std::unordered_map<std::string, std::vector<int>> input_shapes;
  if (!FLAGS_shapes.empty()) {
    input_shapes = GetInputShape(FLAGS_shapes);
  } else {
    auto input_tensor_shape = predictor_baseline.get()->GetInputTensorShape();
    // if input shapes are not given, we will use the default shape and set -1 to 1
    for (auto& it: input_tensor_shape) {
      std::vector<int> tmp;
      std::for_each(it.second.begin(), it.second.end(), [&tmp](int64_t& x){
        tmp.emplace_back(static_cast<int>(x) < 0 ? 1 :static_cast<int>(x));});
      input_shapes[it.first] = tmp;
    }
  }
  // print shape
  for (auto& it: input_shapes) {
    std::string shape_tmp;
    shape_tmp += it.first + ": [";
    for (auto& tmp: it.second) {
      shape_tmp += std::to_string(tmp) + ",";
    }
    shape_tmp[shape_tmp.size() - 1] = ']';
    LOG(INFO) << shape_tmp;
  }
  // input data
  std::unordered_map<std::string, std::vector<float>> input_datas;
  for (auto& it : input_shapes) {
    int num = std::accumulate(it.second.begin(), it.second.end(), 1, std::multiplies<int>());
    std::vector<float> data;
    for (int i = 0; i < num; ++ i) data.emplace_back((float)(rand()) / (float)(RAND_MAX));
    input_datas[it.first] = data;
  }

  // collect shape
  if (FLAGS_collect_shape.empty()) {
    FLAGS_collect_shape = FLAGS_cache_dir + "shape_range_info.pbtxt";
  }
  if (FLAGS_shape_file.empty()) {
    FLAGS_shape_file = FLAGS_collect_shape;
    if(access(FLAGS_collect_shape.c_str(), F_OK ) != -1 ) {
      LOG(INFO) << "Load shape file " << FLAGS_collect_shape;
    } else {
      Config config;
      if (FLAGS_model_dir != "") {
        config.SetModel(FLAGS_model_dir);
      }
      config.SetModel(FLAGS_model_file, FLAGS_params_file);
      config.EnableUseGpu(500, 0);
      config.CollectShapeRangeInfo(FLAGS_collect_shape);
      LOG(INFO) << "========clollect shape========";
      auto predictor_collect = CreatePredictor(config);
      run(predictor_collect.get(), input_shapes, input_datas);
      LOG(INFO) << "The shape file has been saved in " << FLAGS_collect_shape;
    }
  }

  // run baseline predictor
  // auto predictor_baseline = InitBaselinePredictor();
  predictor_baseline->RegisterOutputHook(save_baseline_hook);
  run(predictor_baseline.get(), input_shapes, input_datas);

  // mark and check all tensors
  check_diff::mark_tensor_names = check_diff::tensor_names;

  // run tensorrt predictor
  auto predictor = InitPredictor();
  predictor->RegisterOutputHook(assert_tensor_close_hook);
  run(predictor.get(), input_shapes, input_datas);

  std::vector<std::string> header{"Operator Type", "Tensor Name", "Shape", 
          "Mismatched Elements", "Max Atol", "Max Rtol", "Min Val(base)", "Max Val(base)"};
  paddle::inference::TablePrinter table(header);
  for(auto& tensor_name: check_diff::tensor_names) {
    if (check_diff::tensor_match_info.count(tensor_name) > 0) {
      table.InsertRow(check_diff::tensor_match_info[tensor_name]);
    }
  }
  table.PrintTableCout();

  return 0;
}
