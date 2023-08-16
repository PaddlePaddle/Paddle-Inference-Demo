#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <numeric>
#include <iomanip>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iterator>
#include <functional>

#include "cuda_runtime.h"
#include "paddle/include/paddle_inference_api.h"
#include "paddle/include/paddle/extension.h"
#include "utils/table_printer.h"
#include "utils/string_processor.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(warmup, 0, "warmup.");
DEFINE_int32(repeats, 1, "repeats.");
DEFINE_string(run_mode, "trt_fp32", "run_mode which can be: trt_fp32, trt_fp16, trt_int8 and paddle_gpu");
DEFINE_string(baseline_mode, "", "baseline_mode which can be: trt_fp32, trt_fp16, trt_int8 and paddle_gpu");
DEFINE_string(shapes, "", "input_shapes");
DEFINE_string(load_input, "", "load_input: a:a.data,b:b.data");
DEFINE_string(collect_shape, "", "collect shape.");
DEFINE_string(shape_file, "", "collect shape file.");
DEFINE_string(out_file, "", "out file.");
DEFINE_bool(use_calib, false, "use trt int8 calibration.");
DEFINE_bool(use_cuda_graph, false, "use cuda_graph.");
DEFINE_bool(memory_optim, false, "use memory optimize.");
DEFINE_double(rtol, 1e-4, "relative tolerance");
DEFINE_double(atol, 1e-4, "absolute tolerance");
DEFINE_string(cache_dir, "./cache/", "cache_dir.");
DEFINE_bool(check, false, "check outpus.");
DEFINE_bool(check_all, false, "check all tensors, including intermediate tensor and output.");
DEFINE_string(check_tensor, "", "check specific tensors, example: tensor1,tensor2");


namespace hook_got_info {
  std::vector<std::string> tensor_names;
  std::map<std::string, std::string> tensor2op;
  std::map<std::string, paddle::Tensor> baseline_tensor;
  std::vector<std::vector<std::string>> mismatch_tensors;
  bool fluid_mode = true;
  size_t check_conut = 0;
  std::set<std::string> output_names;
  std::vector<std::vector<std::string>> output_match_status;
}

namespace input_info {
  std::unordered_map<std::string, std::vector<int>> input_shapes;
  std::unordered_map<std::string, std::vector<float>> input_datas;
  std::unordered_map<std::string, std::string> input_files;
  std::once_flag randomInitFlag;
  void randomInitInputData() {
    for (auto& it : input_shapes) {
      int num = std::accumulate(it.second.begin(), it.second.end(), 1, std::multiplies<int>());
      std::vector<float> data;
      for (int i = 0; i < num; ++ i) data.emplace_back((float)(rand()) / (float)(RAND_MAX));
      input_datas[it.first] = data;
    }
  }
}

void save_baseline_hook (const std::string &op_type,
                        const std::string &tensor_name,
                        const paddle::Tensor &tensor) {
  auto cpu_tensor = tensor.copy_to(paddle::CPUPlace(), true);
  if (hook_got_info::fluid_mode) {
    hook_got_info::tensor_names.emplace_back(tensor_name);
    hook_got_info::tensor2op[tensor_name] = op_type;
  }
  hook_got_info::baseline_tensor[tensor_name] = cpu_tensor;
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
  auto actual = tensor.copy_to(paddle::CPUPlace(), true);
  if (hook_got_info::baseline_tensor.count(tensor_name)) {
    // LOG(INFO) << "tensor_name: " << tensor_name;
    hook_got_info::check_conut ++;
    auto desired = hook_got_info::baseline_tensor[tensor_name];
    std::vector<std::string> match_status;
    match_status.emplace_back(hook_got_info::tensor2op[tensor_name]);
    match_status.emplace_back(tensor_name);
    // assert shape equal
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
    if (actual.shape() != desired.shape()) {
      shapeMatch = "expect " + get_shape(desired) +  ", but got " + get_shape(actual);
    } else {
      assert(actual.numel() == desired.numel());
      shapeMatch = "match, " + get_shape(desired);
      for (size_t i = 0; i < desired.numel(); i++) {
        if (tensor.dtype() == paddle::DataType::INT32) {
          mis_match_num += assert_close<int32_t>(actual.data<int32_t>()[i], desired.data<int32_t>()[i], FLAGS_atol, FLAGS_atol, max_atol, max_rtol);
        } else if (tensor.dtype() == paddle::DataType::INT64) {
          mis_match_num += assert_close<int64_t>(actual.data<int64_t>()[i], desired.data<int64_t>()[i], FLAGS_atol, FLAGS_atol, max_atol, max_rtol);
        } else if (tensor.dtype() == paddle::DataType::FLOAT32) {
          mis_match_num += assert_close<float>(actual.data<float>()[i], desired.data<float>()[i], FLAGS_atol, FLAGS_atol, max_atol, max_rtol);
        } else if (tensor.dtype() == paddle::DataType::FLOAT64) {
          mis_match_num += assert_close<double>(actual.data<double>()[i], desired.data<double>()[i], FLAGS_atol, FLAGS_atol, max_atol, max_rtol);
        } else if (tensor.dtype() == paddle::DataType::BOOL) {
          mis_match_num += assert_close<int>(static_cast<int>(actual.data<bool>()[i]), static_cast<int>(desired.data<bool>()[i]), FLAGS_atol, FLAGS_atol, max_atol, max_rtol);
        } else {
          LOG(WARNING) << "Unsupported data type " << tensor.dtype();
          return;
        }
      }
    }
    match_status.emplace_back(shapeMatch);
    match_status.emplace_back(std::to_string(mis_match_num) + "/" + std::to_string(desired.numel()));
    match_status.emplace_back(std::to_string(max_atol));
    match_status.emplace_back(std::to_string(max_rtol));
    // LOG(INFO) << "max atol: " << max_atol << ", max rtol: " << max_rtol;
    if (hook_got_info::output_names.count(tensor_name)) {
      hook_got_info::output_match_status.emplace_back(match_status);
    } else if (mis_match_num > 0) {
      hook_got_info::mismatch_tensors.emplace_back(match_status);
    }
  } else {
    LOG(WARNING) << "Tensor " << tensor_name << " not found in paddle fluid inference.";
  }
}

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void run(Predictor *predictor, bool test_mode=false) {
  // if input shapes are not given, we will use the default shape and set the -1 dim to 1
  if (input_info::input_shapes.size() == 0) {
    auto input_tensor_shape = predictor->GetInputTensorShape();
    for (auto& it: input_tensor_shape) {
      std::vector<int> tmp;
      std::for_each(it.second.begin(), it.second.end(), [&tmp](auto& x){
        tmp.emplace_back(static_cast<int>(x) < 0 ? 1 :static_cast<int>(x));});
      input_info::input_shapes[it.first] = tmp;
    }
  }
  // debug print shape
  for (auto& it: input_info::input_shapes) {
    std::string shape_tmp;
    shape_tmp += it.first + ": [";
    for (auto& tmp: it.second) {
      shape_tmp += std::to_string(tmp) + ",";
    }
    shape_tmp[shape_tmp.size() - 1] = ']';
    LOG(INFO) << shape_tmp;
  }
  // if input datas are not given, we will use random input datas
  if (input_info::input_datas.size() == 0) {
    std::call_once(input_info::randomInitFlag, input_info::randomInitInputData);
  }
  // for (auto& it: input_info::input_datas) {
  //   std::string data_tmp = it.first + ": " + std::to_string(it.second.size());
  //   LOG(INFO) << data_tmp; 
  // }
  auto input_names = predictor->GetInputNames();
  std::set<std::string> input_name_set(input_names.begin(), input_names.end());
  for (auto it : input_info::input_shapes) {
    auto name = it.first;
    if (input_name_set.count(name)) {
      auto& shape = it.second;
      auto handle = predictor->GetInputHandle(name);
      std::vector<int> input_shape = handle->shape();
      handle->Reshape(shape);
      handle->CopyFromCpu(input_info::input_datas.at(name).data());
    }
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(predictor->GetExecStream());

  if (test_mode) {
    CHECK(predictor->Run());
  } else {
    for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());
    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    std::vector<std::string> header{"Enqueue Time", "Run Avg Time"};
    paddle::inference::TablePrinter table(header);
    auto st = time();
    double enqueue_time = 0;
    for (size_t i = 0; i < FLAGS_repeats; ++i) {
      auto enqueue_s = time();
      CHECK(predictor->Run());
      enqueue_time += time_diff(enqueue_s, time());
      cudaStreamSynchronize(stream);
    }
    table.InsertRow({std::to_string(enqueue_time / FLAGS_repeats) + " ms", std::to_string(time_diff(st, time()) / FLAGS_repeats) + " ms"});
    LOG(INFO) << table.PrintTable();
  }

  if (!FLAGS_out_file.empty() && !test_mode) {
    std::fstream out_file;
    out_file.open(FLAGS_out_file, std::ios_base::out);

    auto output_names = predictor->GetOutputNames();
    for (int i = 0; i < output_names.size(); ++i) {
      auto name = output_names[i];
      auto handle = predictor->GetOutputHandle(name);
      std::vector<int> output_shape = handle->shape();
      int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());
      std::vector<float> out_data(out_num, 0);
      handle->CopyToCpu(out_data.data());
      for (size_t i = 0; i < out_data.size() - 1; ++i)
        out_file << out_data[i] << ",";
      out_file << out_data[out_data.size() - 1]  << "\n";
      //std::copy(out_data.begin(), out_data.end() - 1, output_iterator);
      //*output_iterator++ = *(out_data.end() - 1) + "\n";
    }
    for (auto& n : output_names) {
      out_file << n << "\n";
    }
    out_file.close();
  }
}

std::shared_ptr<Predictor> InitPredictorBaseline() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(500, 0);
  config.SwitchIrOptim(false);
  return CreatePredictor(config);
}

std::shared_ptr<Predictor> InitPredictorTRTDynamic(std::vector<std::string> mark_name = {}, bool baseline_mode=false) {
  std::string run_mode;
  if(baseline_mode) {
    run_mode = FLAGS_baseline_mode;
  } else {
    run_mode = FLAGS_run_mode;
  }
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(500, 0);

  if (run_mode == "trt_fp32") {
    config.EnableTensorRtEngine(1 << 30, 1, 1,
                                PrecisionType::kFloat32, true, false, FLAGS_use_cuda_graph);
  } else if (run_mode == "trt_fp16") {
    config.EnableTensorRtEngine(1 << 30, 1, 1,
                                PrecisionType::kHalf, true, false, FLAGS_use_cuda_graph);
  } else if (run_mode == "trt_int8") {
    config.EnableTensorRtEngine(1 << 30, 1, 1,
                                PrecisionType::kInt8, false, FLAGS_use_calib, FLAGS_use_cuda_graph);
  }
  config.SwitchIrDebug(true);
  config.SetOptimCacheDir(FLAGS_cache_dir);
  if (run_mode.find("trt") != std::string::npos) {
    auto runForShapeFile = [](const std::string& shape_file, Config config) {
      config.CollectShapeRangeInfo(shape_file);
      auto predictor_cs = CreatePredictor(config);
      run(predictor_cs.get(), true);
    };
    if (FLAGS_shape_file.empty()) {
      if(FLAGS_collect_shape.empty()) {
        FLAGS_collect_shape = FLAGS_cache_dir + GetModelName(FLAGS_model_file) + ".pbtxt";
      }
      if(access(FLAGS_collect_shape.c_str(), F_OK ) != -1 ) {
        LOG(INFO) << "Load shape file " << FLAGS_collect_shape;
      } else {
        runForShapeFile(FLAGS_collect_shape, config);
        LOG(INFO) << "The shape file has been saved in " << FLAGS_collect_shape;
      }
    }
    config.EnableTunedTensorRtDynamicShape(FLAGS_shape_file);
  }
  config.MarkTrtEngineOutputs(mark_name);
  // Open the memory optim.
  if (FLAGS_memory_optim) {
    config.EnableMemoryOptim();
  }
  //config.EnableTensorRtInspector();
  LOG(INFO) << config.Summary();
  return CreatePredictor(config);
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (!FLAGS_shapes.empty()) {
    input_info::input_shapes = GetInputShape(FLAGS_shapes);
  }

  if (!FLAGS_load_input.empty()) {
    input_info::input_files = GetInputFile(FLAGS_load_input);
    for (auto& it : input_info::input_files) {
      LOG(INFO) << it.first << ", " << it.second;
    }
  }

  for (auto it : input_info::input_shapes) {
      if (input_info::input_files.count(it.first)) {
        auto data = LoadInputFrom(input_info::input_files[it.first]);
        input_info::input_datas[it.first] = data;
      } else {
        int num = std::accumulate(it.second.begin(), it.second.end(), 1, std::multiplies<int>());
        std::vector<float> data(num, 0.1);
        input_info::input_datas[it.first] = data;
      }
  }

  if (FLAGS_check || FLAGS_check_all || !FLAGS_check_tensor.empty()) {
    // paddle fuild baseline
    auto predictor_baseline = InitPredictorBaseline();
    auto output_names = predictor_baseline->GetOutputNames();
    for (auto& output_name: output_names) {
      hook_got_info::output_names.insert(output_name);
    }
    predictor_baseline->RegisterOutputHook(save_baseline_hook);
    run(predictor_baseline.get(), true);

    // user set baseline
    if (!FLAGS_baseline_mode.empty()) {
      hook_got_info::fluid_mode = false;
      auto predictor_baseline2 = InitPredictorTRTDynamic({hook_got_info::tensor_names}, true);
      predictor_baseline2->RegisterOutputHook(save_baseline_hook);
      run(predictor_baseline2.get(), true);
    }
    
    // run
    std::shared_ptr<Predictor> predictor;
    if (FLAGS_check) {
      predictor = InitPredictorTRTDynamic();
    } else if (FLAGS_check_all) {
      predictor = InitPredictorTRTDynamic({hook_got_info::tensor_names});
    } else {
      std::vector<std::string> mark_names = ProcessMultiShape(FLAGS_check_tensor);
      predictor = InitPredictorTRTDynamic(mark_names);
    }
    predictor->RegisterOutputHook(assert_tensor_close_hook);
    run(predictor.get(), true);

    // print result
    LOG(INFO) << "Mismatched Tensor Num: " << hook_got_info::mismatch_tensors.size();
    std::vector<std::string> header{"Operator Type", "Tensor Name", "Shape", "Mismatched Elements", "Max Atol", "Max Rtol"};
    paddle::inference::TablePrinter table(header);
    if (hook_got_info::mismatch_tensors.size() > 0) {
      for (auto& match_status: hook_got_info::mismatch_tensors) {
        table.InsertRow(match_status);
      }
      table.InsetDivider();
      for (auto& match_status: hook_got_info::output_match_status) {
        table.InsertRow(match_status);
      }
      LOG(INFO) << "Check result are as follows: ";
      table.PrintTableCout();
    } else {
      std::string output_num = std::to_string(hook_got_info::check_conut);
      table.InsertRow({"All(" + output_num +") output are equal."});
      LOG(INFO) << table.PrintTable();
    }
  } else {
    auto predictor = InitPredictorTRTDynamic();
    run(predictor.get());
  }
  return 0;
}
