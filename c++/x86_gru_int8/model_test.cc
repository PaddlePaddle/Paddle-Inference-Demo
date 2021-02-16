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

template <typename T>
constexpr paddle::PaddleDType GetPaddleDType();

template <>
constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
  return paddle::PaddleDType::INT64;
}

template <>
constexpr paddle::PaddleDType GetPaddleDType<float>() {
  return paddle::PaddleDType::FLOAT32;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset, std::string name)
      : file_(file), position_(beginning_offset), name_(name) {}

  paddle::PaddleTensor NextBatch(std::vector<int> shape, std::vector<size_t> lod) {
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    paddle::PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));
    if (lod.empty() == false) {
      tensor.lod.clear();
      tensor.lod.push_back(lod);
    }
    file_.seekg(position_);
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    file_.read(reinterpret_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position_ = file_.tellg();
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::string name_;
};

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

std::vector<size_t> ReadSentenceLod(std::ifstream &file, size_t offset,
                                    int64_t total_sentences_num) {
  std::vector<size_t> sentence_lod(total_sentences_num);

  file.clear();
  file.seekg(offset);
  file.read(reinterpret_cast<char *>(sentence_lod.data()),
            total_sentences_num * sizeof(size_t));

  if (file.eof()) LOG(ERROR) << "Reached end of stream";
  if (file.fail()) throw std::runtime_error("Failed reading file.");
  return sentence_lod;
}

void SetInput(std::vector<std::vector<paddle::PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    std::cout<< "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_sentences_num = 0L;
  int64_t total_words_num = 0L;
  file.seekg(0);
  file.read(reinterpret_cast<char *>(&total_sentences_num), sizeof(int64_t));
  LOG(INFO) << "Total sentences in file: " << total_sentences_num;
  file.read(reinterpret_cast<char *>(&total_words_num), sizeof(int64_t));
  LOG(INFO) << "Total words in file: " << total_words_num;
  size_t lods_beginning_offset = static_cast<size_t>(file.tellg());
  auto words_begining_offset =
      lods_beginning_offset + sizeof(size_t) * total_sentences_num;
  auto targets_beginning_offset =
      words_begining_offset + sizeof(int64_t) * total_words_num;

  std::vector<size_t> lod_full =
      ReadSentenceLod(file, lods_beginning_offset, total_sentences_num);

  size_t lods_sum = std::accumulate(lod_full.begin(), lod_full.end(), 0UL);
  // EXPECT_EQ(lods_sum, static_cast<size_t>(total_words_num));

  TensorReader<int64_t> words_reader(file, words_begining_offset, "words");

  TensorReader<int64_t> targets_reader(file, targets_beginning_offset,
                                       "targets");
  // If FLAGS_iterations is set to 0, run all batches
  auto iterations_max = total_sentences_num / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  for (auto i = 0; i < iterations; i++) {
    // Calculate the words num.  Shape=[words_num, 1]
    std::vector<size_t> batch_lod = {0};
    size_t num_words = 0L;
    std::transform(lod_full.begin() + i * FLAGS_batch_size,
                   lod_full.begin() + (i + 1) * FLAGS_batch_size,
                   std::back_inserter(batch_lod),
                   [&num_words](const size_t lodtemp) -> size_t {
                     num_words += lodtemp;
                     return num_words;
                   });
    auto words_tensor = words_reader.NextBatch(
        {static_cast<int>(batch_lod[FLAGS_batch_size]), 1}, batch_lod);
    if (FLAGS_with_accuracy_layer) {
      auto targets_tensor = targets_reader.NextBatch(
          {static_cast<int>(batch_lod[FLAGS_batch_size]), 1}, batch_lod);
      inputs->emplace_back(std::vector<paddle::PaddleTensor>{
          std::move(words_tensor), std::move(targets_tensor)});
    } else {
      inputs->emplace_back(std::vector<paddle::PaddleTensor>{std::move(words_tensor)});
    }
  }
}

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

void PredictionRun(paddle::PaddlePredictor *predictor,
                   const std::vector<std::vector<paddle::PaddleTensor>> &inputs,
                   std::vector<std::vector<paddle::PaddleTensor>> *outputs,
                   int num_threads,
                   float *sample_latency = nullptr) {
  int iterations = inputs.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(inputs.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  Timer run_timer;
  double elapsed_time = 0;
  int predicted_num = 0;

  for (int i = 0; i < iterations; i++) {
    run_timer.tic();
    predictor->Run(inputs[i], &(*outputs)[i], FLAGS_batch_size);
    elapsed_time += run_timer.toc();

    predicted_num += FLAGS_batch_size;
    if (predicted_num % 100 == 0) {
      LOG(INFO) << "Infer " << predicted_num << " samples";
    }
  }

  auto batch_latency = elapsed_time / iterations;
  // PrintTime(FLAGS_batch_size, num_threads, batch_latency, iterations);

  if (sample_latency != nullptr)
    *sample_latency = batch_latency / FLAGS_batch_size;
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
      std::cout<< static_cast<int64_t *>(outputs[i][j].data.data())<<" ";
    }
    std::cout<<std::endl;
  }

  // nums_infer, nums_label, nums_correct
  return 0;
}
