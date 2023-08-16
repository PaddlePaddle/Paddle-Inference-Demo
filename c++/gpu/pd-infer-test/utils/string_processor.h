// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <unordered_map>    

inline std::string GetModelName(const std::string &model_file) {
  std::string model_name;
  int start = 0;
  int seg = model_file.find("/", start);
  while (seg != std::string::npos) {
    model_name = model_file.substr(start, seg - start);
    start = seg + 1;
    seg = model_file.find("/", start);
  }
  model_name = model_file.substr(start, model_file.size() - start);
  seg = model_name.find(".", start);
  if (seg != std::string::npos) {
    return model_name.substr(0, seg);
  } else {
    return model_name;
  }
}

// 1x2x3x4 -> vector<int>{1,2,3,4}
inline std::vector<int> ProcessShapeString(const std::string& shape) {
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
inline std::vector<std::string> ProcessMultiShape(const std::string& shape) {
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

inline std::unordered_map<std::string, std::vector<int>> GetInputShape(const std::string& shape) {
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

inline std::unordered_map<std::string, std::string> GetInputFile(const std::string& input) {
  std::unordered_map<std::string, std::string> res;
  auto multi_input = ProcessMultiShape(input);
  for (auto& s : multi_input) {
    auto seg = s.find(":");
    CHECK_NE(seg, std::string::npos);
    auto name = s.substr(0, seg);
    auto file = s.substr(seg+1);
    //auto shape_val = ProcessShapeString(s.substr(seg+1));
    res[name] = file;
  }
  return res;
}

inline std::vector<float> LoadInputFrom(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<float> data;
  if (file) {
    file.seekg(0, file.end);
    int size = file.tellg();
    file.seekg(0);
    data.resize(size/sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    //for (size_t i = 0; i < data.size(); i+=1000) {
    //  LOG(INFO) << "data[" << i << "]=" << data[i];
    //}
  }
  if (data.size() == 0) {
    LOG(FATAL) << "read " << path << " failed.";
  }
  return data;
}