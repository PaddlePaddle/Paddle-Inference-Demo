#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.EnableUseGpu(100, 0);

  // Open the memory optim.
  config.EnableMemoryOptim();

  int max_batch = 32;
  int max_single_seq_len = 128;
  int opt_single_seq_len = 64;
  int min_batch_seq_len = 1;
  int max_batch_seq_len = 512;
  int opt_batch_seq_len = 256;

  std::string input_name0 = "read_file_0.tmp_0";
  std::string input_name1 = "read_file_0.tmp_1";
  std::string input_name2 = "read_file_0.tmp_2";
  std::string input_name3 = "read_file_0.tmp_4";

  std::vector<int> min_shape = {min_batch_seq_len};
  std::vector<int> max_shape = {max_batch_seq_len};
  std::vector<int> opt_shape = {opt_batch_seq_len};
  // Set the input's min, max, opt shape
  std::map<std::string, std::vector<int>> min_input_shape = {
      {input_name0, min_shape},
      {input_name1, min_shape},
      {input_name2, {1}},
      {input_name3, {1, 1, 1}}};
  std::map<std::string, std::vector<int>> max_input_shape = {
      {input_name0, max_shape},
      {input_name1, max_shape},
      {input_name2, {max_batch + 1}},
      {input_name3, {1, max_single_seq_len, 1}}};
  std::map<std::string, std::vector<int>> opt_input_shape = {
      {input_name0, opt_shape},
      {input_name1, opt_shape},
      {input_name2, {max_batch + 1}},
      {input_name3, {1, opt_single_seq_len, 1}}};

  // only kHalf supported
  config.EnableTensorRtEngine(1 << 30, 1, 5, Config::Precision::kHalf, false,
                              false);
  // erinie varlen must be used with dynamic shape
  config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
                                opt_input_shape);
  // erinie varlen must be used with oss
  config.EnableTensorRtOSS();

  return CreatePredictor(config);
}

void run(Predictor *predictor, std::vector<float> *out_data) {
  const int run_batch = 2;
  const int run_seq_len = 71;
  const int max_seq_len = 128;

  int32_t i1[run_seq_len] = {
      // sentence 1
      1, 3558, 4, 75, 491, 89, 340, 313, 93, 4, 255, 10, 75, 321, 4095, 1902, 4,
      134, 49, 75, 311, 14, 44, 178, 543, 15, 12043, 2, 75, 201, 340, 9, 14, 44,
      486, 218, 1140, 279, 12043, 2,
      // sentence 2
      101, 2054, 2234, 2046, 2486, 2044, 1996, 2047, 4552, 2001, 9536, 1029,
      102, 2004, 1997, 2008, 2154, 1010, 1996, 2047, 4552, 9536, 2075, 1996,
      2117, 3072, 2234, 2046, 2486, 1012, 102,
  };
  int32_t i2[run_seq_len] = {
      // sentence 1
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      // sentence 2
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1};
  // shape info of this batch
  int32_t i3[3] = {0, 40, 71};
  // max_seq_len represents the max length of all the sentences, only length of
  // input i4 is useful, data means nothing.
  int32_t i4[max_seq_len] = {0};

  auto input_names = predictor->GetInputNames();
  // first input
  auto input_t1 = predictor->GetInputHandle(input_names[0]);
  input_t1->Reshape({run_seq_len});
  input_t1->CopyFromCpu(i1);

  // second input
  auto input_t2 = predictor->GetInputHandle(input_names[1]);
  input_t2->Reshape({run_seq_len});
  input_t2->CopyFromCpu(i2);

  // third input
  auto input_t3 = predictor->GetInputHandle(input_names[2]);
  input_t3->Reshape({run_batch + 1});
  input_t3->CopyFromCpu(i3);

  // fourth input
  auto input_t4 = predictor->GetInputHandle(input_names[3]);
  input_t4->Reshape({1, max_seq_len, 1});
  input_t4->CopyFromCpu(i4);

  CHECK(predictor->Run());

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
  out_data->resize(out_num);
  output_t->CopyToCpu(out_data->data());

  return;
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();
  std::vector<float> out_data;
  run(predictor.get(), &out_data);

  for (auto r : out_data) {
    LOG(INFO) << r;
  }
  return 0;
}
