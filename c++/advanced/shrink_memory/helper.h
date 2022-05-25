// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <glog/logging.h>
#include <fstream>
#if !defined(_WIN32)
#include <sys/time.h>
#endif
#include <algorithm>
#include <chrono>  // NOLINT
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace paddle {
namespace inference {

// Timer for timer
class Timer {
 public:
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

class Barrier {
 public:
  explicit Barrier(std::size_t count) : _count(count) {}
  void Wait() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (--_count) {
      _cv.wait(lock, [this] { return _count == 0; });
    } else {
      _cv.notify_all();
    }
  }

 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::size_t _count;
};

static int GetUniqueId() {
  static int id = 0;
  return id++;
}

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces,
                  bool ignore_null = true) {
  pieces->clear();
  if (str.empty()) {
    if (!ignore_null) {
      pieces->push_back(str);
    }
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}
static void split_to_float(const std::string &str, char sep,
                           std::vector<float> *fs) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*fs),
                 [](const std::string &v) { return std::stof(v); });
}
static void split_to_int64(const std::string &str, char sep,
                           std::vector<int64_t> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*is),
                 [](const std::string &v) { return std::stoi(v); });
}
static void split_to_int(const std::string &str, char sep,
                         std::vector<int> *is) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*is),
                 [](const std::string &v) { return std::stoi(v); });
}

static void PrintTime(int batch_size, int repeat, int num_threads, int tid,
                      double batch_latency, int epoch = 1) {
  double sample_latency = batch_latency / batch_size;
  LOG(INFO) << "====== threads: " << num_threads << ", thread id: " << tid
            << " ======";
  LOG(INFO) << "====== batch_size: " << batch_size << ", iterations: " << epoch
            << ", repetitions: " << repeat << " ======";
  LOG(INFO) << "====== batch latency: " << batch_latency
            << "ms, number of samples: " << batch_size * epoch
            << ", sample latency: " << sample_latency
            << "ms, fps: " << 1000.f / sample_latency << " ======";
}

}  // namespace inference
}  // namespace paddle
