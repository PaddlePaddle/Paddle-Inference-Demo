#include <assert.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <sys/time.h>    
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "lac.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_path, "", "Directory of the inference model.");
DEFINE_string(conf_path, "", "Path of the inference conf_path.");
DEFINE_string(input_path, "", "Path of the inference input_path.");
DEFINE_string(label_path, "", "Path of the inference label_path.");
DEFINE_int32(test_num, 0, "Path of the inference test_num.");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  int threads = 1;
  std::cout << "model_path:" << FLAGS_model_path << std::endl;
  std::cout << "conf_path:" << FLAGS_conf_path << std::endl;
  std::cout << "input_path:" << FLAGS_input_path << std::endl;
  std::cout << "label_path:" << FLAGS_label_path << std::endl;
  std::cout << "test_num:" << FLAGS_test_num << std::endl;

  LAC lac(FLAGS_model_path, FLAGS_conf_path);
  struct timeval start;
  struct timeval end;
  int64_t cnt = 0;
  int64_t char_cnt = 0;
  std::fstream input_fs(FLAGS_input_path);
  std::fstream label_fs(FLAGS_label_path);
  if (!input_fs.is_open() || !label_fs.is_open()) {
    std::cerr << "open input or label file error";
    return 1;
  }
  int i = 0, right_num = 0, error_num = 0;
  gettimeofday(&start, NULL);
  std::string query;
  std::string output_str;
  std::string label_str;
  auto count=0;
  while (!input_fs.eof() && !label_fs.eof() && count < FLAGS_test_num) {
    std::getline(input_fs, query);
    std::getline(label_fs, label_str);
    cnt++;
    char_cnt += query.length();
    auto result = lac.lexer(query);
    output_str = "";
    for (int i = 0; i < result.size(); i++) {
      if (result[i].tag.length() == 0) {
        output_str += (result[i].word + " ");
      } else {
        output_str += (result[i].word + "\001" + result[i].tag + " ");
      }
    }
    if (output_str == label_str) {
      right_num++;
    } else {
      error_num++;
    }
    count++;
  }
  gettimeofday(&end, NULL);
  double time =
      end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  std::cerr << "using time: " << time << " \t qps:" << cnt / time
            << "\tc/s:" << char_cnt / time << std::endl;
  std::cerr << "right_num :" << right_num << "\t error num:" << error_num
            << "\t ratio:"
            << static_cast<float>(right_num) / (right_num + error_num)
            << std::endl;

  return 0;
}
