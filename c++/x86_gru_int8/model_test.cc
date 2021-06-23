#include <glog/logging.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "paddle_inference_api.h"

DEFINE_string(infer_model, "", "path to the model");
DEFINE_string(infer_data, "", "path to the input data");
DEFINE_int32(batch_size, 50, "inference batch size");
DEFINE_int32(iterations,
             0,
             "number of batches to process. 0 means testing whole dataset");
DEFINE_int32(num_threads, 1, "num of threads to run in parallel");
DEFINE_bool(with_accuracy_layer,
            true,
            "Set with_accuracy_layer to true if provided model has accuracy "
            "layer and requires label input");
DEFINE_bool(use_analysis,
            true,
            "If use_analysis is set to true, the model will be optimized");


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

void SetInput(std::vector<std::vector<paddle::PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    LOG(INFO) << "Couldn't open file: " << FLAGS_infer_data;
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
  auto iterations_max = total_sentences_num / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  for (auto i = 0; i < iterations; i++) {
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

std::vector<double> Lexical_Test(
    const std::vector<std::vector<paddle::PaddleTensor>> &input_slots_all,
    std::vector<std::vector<paddle::PaddleTensor>> *outputs, paddle::AnalysisConfig *config,
    const bool use_analysis=FLAGS_use_analysis) {
  auto predictor =
      CreatePredictor(reinterpret_cast<paddle::PaddlePredictor::Config *>(config),
                      use_analysis);
  int iterations = input_slots_all.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(input_slots_all.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  int predicted_num = 0;
  struct timeval start, end;
  for (int i = 0; i < iterations; i++) {
    predictor->Run(input_slots_all[i], &(*outputs)[i], FLAGS_batch_size);
    predicted_num += FLAGS_batch_size;
    if (predicted_num % 100 == 0) {
      std::cout << "Infer " << predicted_num << " samples" << std::endl;
    }
  }
  
  std::vector<double> acc_res(3);
  if (FLAGS_with_accuracy_layer) {
    std::vector<int64_t> acc_sum(3);
    for (size_t i = 0; i < outputs->size(); i++) {
      for (size_t j = 0; j < 3UL; j++) {
        acc_sum[j] =
            acc_sum[j] + *static_cast<int64_t *>((*outputs)[i][j].data.data());
      }
    }
    // nums_infer, nums_label, nums_correct
    auto precision =
        acc_sum[0]
            ? static_cast<double>(acc_sum[2]) / static_cast<double>(acc_sum[0])
            : 0;
    auto recall =
        acc_sum[1]
            ? static_cast<double>(acc_sum[2]) / static_cast<double>(acc_sum[1])
            : 0;
    auto f1_score =
        acc_sum[2]
            ? static_cast<float>(2 * precision * recall) / (precision + recall)
            : 0;

    std::cout << "Precision:  " << std::fixed << std::setw(6)
              << std::setprecision(5) << precision << std::endl;
    std::cout << "Recall:  " << std::fixed << std::setw(6)
              << std::setprecision(5) << recall << std::endl;
    std::cout << "F1 score: " << std::fixed << std::setw(6)
              << std::setprecision(5) << f1_score << std::endl;

    acc_res = {precision, recall, f1_score};
    return acc_res;
  } else {
    std::cout << "No accuracy result. To get accuracy result provide a model "
                 "with accuracy layers in it and use --with_accuracy_layer "
                 "option.";
  }
  return acc_res;
}

int main(int argc, char *argv[]){
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle::AnalysisConfig cfg;
  cfg.SetModel(FLAGS_infer_model);
  cfg.SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  if (FLAGS_use_analysis){
    cfg.SwitchIrOptim(true);
    cfg.EnableMKLDNN();
  }
  
  std::vector<std::vector<paddle::PaddleTensor>> outputs;
  std::vector<std::vector<paddle::PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  std::vector<double> acc_analysis(3);
  std::cout<<"********************************";
  acc_analysis = Lexical_Test(input_slots_all, &outputs, &cfg, true);
  return 0;
}
