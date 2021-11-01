#include "paddle/include/paddle_inference_api.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace flags {
  DEFINE_string(model, "", "model directory");
  DEFINE_string(data, "", "inference data");
  DEFINE_string(mode, "fp32", "inference mode");
}

DEFINE_uint32(batch_size, 1, "batch size");
DEFINE_uint64(seq_lens, 0, "max seq len");
DEFINE_int32(num_labels, 2, "number of labels");
DEFINE_int32(out_predict, 1, "whether to output predition");
DEFINE_int32(min_graph, 5, "min graph size in trt option");
DEFINE_int32(ignore_copy, 0, "whether to ignore the copy cost");

class Timer {
public:
    Timer() {
        reset();
    }

    void start() {
        start_t = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto end_t = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_t - start_t;
        total_time += diff.count();
    }

    void reset() {
        total_time = 0.;
    }

    double report() {
        return total_time;
    }
private:
    double total_time;
    std::chrono::high_resolution_clock::time_point start_t;
};

template<typename T>
class Analytics {
public:
    T max() {
        return *std::max_element(data_.begin(), data_.end());
    }

    T min() {
        return *std::min_element(data_.begin(), data_.end());
    }

    T mean() {
        return std::accumulate(data_.begin(), data_.end(), 0.0)/data_.size();
    }

    T len() {
        return data_.size();
    }

    Analytics(const std::vector<T>& data): data_(data) {}
    Analytics() {}

private:
    std::vector<T> data_;
};

class BatchSeq {
public:
    size_t batch_len_;
    size_t batch_size_;
    size_t max_seq_len_;
    
    int32_t* srcs_;
    int32_t* sents_;
    int32_t* cu_seqlens_;

    BatchSeq(const size_t batch_len, const size_t batch_size): batch_len_(batch_len), batch_size_(batch_size) {
        max_seq_len_ = 0;
        srcs_ = (int32_t*) malloc(batch_len_ * sizeof(int32_t));
        sents_ = (int32_t*) malloc(batch_len_ * sizeof(int32_t));
        cu_seqlens_ = (int32_t*) malloc((batch_size_ + 1) * sizeof(int32_t));
        memset(srcs_, 0, batch_len_ * sizeof(int32_t));
        memset(sents_, 0, batch_len_ * sizeof(int32_t));
        memset(cu_seqlens_, 0, (batch_size_ + 1) * sizeof(int32_t));
    }
};

template<typename T>
void read_line(const std::string& line, T* place_holder, const size_t n) {
    std::stringstream ss(line);
    for (size_t i = 0; i < n; ++ i) {
        ss >> place_holder[i];
    }
}

auto read_inputs(std::string input_file)
    -> std::tuple<std::vector<BatchSeq>, Analytics<size_t>> {
    std::ifstream fin(input_file);
    std::string line;
    std::vector<std::string> buffer;
    std::vector<BatchSeq> inputs;

    while (std::getline(fin, line)) {
        buffer.emplace_back(line);
    }

    constexpr size_t offset = 5;

    assert(buffer.size() % offset == 0);
    const size_t num_sents = buffer.size() / offset;
    std::vector<size_t> all_batch_len;

    for (size_t i = 0; i < num_sents; i += FLAGS_batch_size) {
        size_t batch_size = std::min((size_t)FLAGS_batch_size, num_sents - i);
        size_t batch_len = 0;

        for (size_t j = 0; j < batch_size; ++ j) {
            const size_t idx = i + j;
            const size_t seq_len = std::stoi(buffer[idx * offset], nullptr, 10);
            batch_len += seq_len;
        }
        all_batch_len.emplace_back(batch_len);

        BatchSeq bs(batch_len, batch_size);
        size_t max_seq_len = 0;
        batch_len = 0;
        for (size_t j = 0; j < batch_size; ++ j) {
            const size_t idx = i + j;
            const size_t seq_len = std::stoi(buffer[idx * offset], nullptr, 10);
            read_line(buffer[idx * offset + 1], bs.srcs_ + batch_len, seq_len);
            read_line(buffer[idx * offset + 2], bs.sents_ + batch_len, seq_len);
            bs.cu_seqlens_[j + 1] = bs.cu_seqlens_[j] + seq_len;
            max_seq_len = std::max(max_seq_len, seq_len);
            batch_len += seq_len;
        }
        bs.max_seq_len_ = max_seq_len;
        inputs.emplace_back(bs);
    }
    return {inputs, Analytics<size_t>(all_batch_len)};
}

template <typename T>
paddle::AnalysisConfig* configure(T* analytics) {
    auto config = new paddle::AnalysisConfig();
    config->SetModel(flags::FLAGS_model + "/model.pdmodel",
                    flags::FLAGS_model + "/model.pdiparams");
    std::string cache_dir(flags::FLAGS_model + "/bs." +
        std::to_string(FLAGS_batch_size) + ".engine");
    struct stat buffer;
    if (stat(cache_dir.c_str(), &buffer) == 0) {
      config->SetOptimCacheDir(cache_dir);
    }
    config->EnableUseGpu(100, 0);
    config->SwitchSpecifyInputNames(true);
    config->EnableCUDNN();
    config->SwitchIrOptim(true);
    config->EnableMemoryOptim();
    config->SwitchUseFeedFetchOps(false);

    const int min_len = 1;
    const int max_len = FLAGS_batch_size * 128;
    const int opt_len = FLAGS_batch_size * 128;

    std::cerr << "[len settings] min = " << min_len 
              << ", max = " << max_len 
              << ", opt = " << opt_len << std::endl;

    const std::vector<int> min_shape = {min_len};
    const std::vector<int> max_shape = {max_len};
    const std::vector<int> opt_shape = {opt_len};

    const std::map<std::string, std::vector<int>> min_input_shape = {
            {"eval_placeholder_0", min_shape}, 
            {"eval_placeholder_1", min_shape}, 
            {"eval_placeholder_2", {1}},
            {"eval_placeholder_3", {1, 1, 1}},
    };
    const std::map<std::string, std::vector<int>> max_input_shape = {
            {"eval_placeholder_0", max_shape}, 
            {"eval_placeholder_1", max_shape}, 
            {"eval_placeholder_2", {(int)FLAGS_batch_size + 1}}, 
            {"eval_placeholder_3", {1, 128, 1}},
    };
    const std::map<std::string, std::vector<int>> opt_input_shape = {
            {"eval_placeholder_0", opt_shape}, 
            {"eval_placeholder_1", opt_shape}, 
            {"eval_placeholder_2", {(int)FLAGS_batch_size + 1}}, 
            {"eval_placeholder_3", {1, 128, 1}},
    };

    if (flags::FLAGS_mode == "trt-fp16") {
        config->EnableTensorRtEngine(1 << 30, 1, FLAGS_min_graph,
            paddle::AnalysisConfig::Precision::kHalf, false, false);
        config->SetTRTDynamicShapeInfo(min_input_shape, max_input_shape, opt_input_shape);
    } else {
	assert(false && "ernie-varlen currently support fp16 only.");
    }

    config->EnableTensorRtOSS();

    return config;
}

// real = 1 for actual running
// real = 0 for perf comm time
template <typename T>
auto predict(paddle::PaddlePredictor *predictor, T first, T last, bool output, bool run)
    -> std::tuple<size_t, double> {

    auto input_names = predictor->GetInputNames();
    auto output_names = predictor->GetOutputNames();

    float* output_data = (float*) malloc(FLAGS_batch_size * FLAGS_num_labels * sizeof(float));
    size_t total_seqs = 0;
    Timer timer;

    std::vector<int> ans_buffer;

    for (auto it = first; it != last; ++ it) {
        timer.start();
        const BatchSeq& batch = *it;
        total_seqs += batch.batch_size_;

        std::cerr << "batch.batch_size_: " << batch.batch_size_ 
                  << ", batch.batch_len_: " << batch.batch_len_ 
                  << ", batch.max_seq_len_: " << batch.max_seq_len_ << std::endl;

        auto input0 = predictor->GetInputTensor(input_names[0]);
        input0->Reshape({(int)batch.batch_len_});
        input0->copy_from_cpu(batch.srcs_);

        auto input1 = predictor->GetInputTensor(input_names[1]);
        input1->Reshape({(int)batch.batch_len_});
        input1->copy_from_cpu(batch.sents_);

        auto input2 = predictor->GetInputTensor(input_names[2]);
        input2->Reshape({(int)batch.batch_size_ + 1});
        input2->copy_from_cpu(batch.cu_seqlens_);

        std::vector<int> dummy_input;
        dummy_input.resize(batch.max_seq_len_);

        auto input3 = predictor->GetInputTensor(input_names[3]);
        input3->Reshape({1, (int)batch.max_seq_len_, 1});
        input3->copy_from_cpu(dummy_input.data());

        if (run) predictor->ZeroCopyRun();

        auto output_tensor = predictor->GetOutputTensor(output_names[0]);
        output_tensor->copy_to_cpu(output_data);
        timer.stop();

        if (not output)
            continue;

        for (int b = 0; b < batch.batch_size_; ++ b) {
            double max_ = -1;
            int arg_max_ = -1;
            for (int x = 0; x < FLAGS_num_labels; ++ x) {
                if (output_data[FLAGS_num_labels * b + x] > max_) {
                    max_ = output_data[FLAGS_num_labels * b + x];
                    arg_max_ = x;
                }
            }
            ans_buffer.push_back(arg_max_);
        }
    }
    if (output) {
        for (int i = 0; i < ans_buffer.size(); ++ i) {
            std::cout << ans_buffer[i] << std::endl;
        }
    }
    free(output_data);
    return {total_seqs, timer.report()};
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(*argv);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    assert(not flags::FLAGS_model.empty());

    std::vector<BatchSeq> inputs;
    Analytics<size_t> analytics;

    std::tie(inputs, analytics) = read_inputs(flags::FLAGS_data);
    auto config = configure(&analytics);
    auto predictor = CreatePaddlePredictor(*config);

    predict(predictor.get(), inputs.cbegin(), inputs.cbegin() + 1, false, true); // warmup

    size_t num_seq;
    double total_time = 0., copy_time = 0.;

    std::tie(num_seq, total_time) = predict(predictor.get(), inputs.begin(), inputs.end(), FLAGS_out_predict, true);
    if (FLAGS_ignore_copy) {
      // run one more time without forwarding to get the copy time
      std::tie(num_seq, copy_time) = predict(predictor.get(), inputs.begin(), inputs.end(), false, false);
      total_time -= copy_time;
    }

    std::cout << "Sents/s " << num_seq / total_time << std::endl;
    std::cerr << num_seq << ", " << total_time << std::endl;
}
