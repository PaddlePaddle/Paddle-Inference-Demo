#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "lac.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(model_file, "", "Path of the inference model file.");
DEFINE_string(params_file, "", "Path of the inference params file.");
DEFINE_string(infer_data, "", "Path of the inference params file.");
DEFINE_int32(threads, 1, "CPU threads.");
DEFINE_int32(batch_size, 1, "batch size.");
DEFINE_int32(iterations,100, "needed iterations");
DEFINE_bool(with_accuracy_layer, false, "with accuracy or not");

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor(
    const paddle::PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config =
      reinterpret_cast<const paddle::AnalysisConfig *>(config);
  if (use_analysis) {
    return paddle::CreatePaddlePredictor<paddle::AnalysisConfig>(
        *analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return paddle::CreatePaddlePredictor<paddle::NativeConfig>(native_config);
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::vector<paddle::PaddleTensor>> input_slots_all;
  std::vector<std::vector<paddle::PaddleTensor>> outputs;
  SetInput(&input_slots_all);
  // Init config
  paddle_infer::Config config;
  if (FLAGS_model_dir == "") {
    config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
  } else {
    config.SetModel(FLAGS_model_dir); // Load no-combined model
  }
  config.EnableMKLDNN();
  config.SetCpuMathLibraryNumThreads(FLAGS_threads);
  config.SwitchIrOptim();
  config.EnableMemoryOptim();
  std::cout<<"-------------------------Warning---------------------------"<<std::endl;
  // Create predictor
  // auto predictor = paddle_infer::CreatePredictor(config);
  auto predictor =
      CreatePredictor(reinterpret_cast<paddle::PaddlePredictor::Config *>(&config),
                      false);

  PredictionRun(predictor.get(), input_slots_all, &outputs, 1);
  for (int i = 0 ; i<100; i++){
    for (int j = 0 ; j < 3; j++){
      auto* temp = static_cast<int64_t *>(outputs[i][j].data.data());
      std::cout<<*temp<<" ";
    }
    std::cout<<std::endl;
  }

  return 0;
}
