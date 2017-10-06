/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/saver.pb.h"
#include "paddle/memory/memcpy.h"
#include "paddle/memory/memory.h"

#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <iterator>

#include <glog/logging.h>

namespace paddle {
namespace framework {

LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end) {
  LoD new_lod;
  new_lod.reserve(level_end - level_begin);
  for (size_t i = level_begin; i < level_end; i++) {
    new_lod.emplace_back(in.at(i));
  }
  return new_lod;
}

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  // slice the lod.
  LoD new_lod;
  new_lod.reserve(in.size() - level);
  auto start = in.at(level)[elem_begin];
  auto end = in.at(level)[elem_end];

  for (auto it = in.begin() + level; it != in.end(); it++) {
    auto it_begin = std::find(it->begin(), it->end(), start);
    auto it_end = std::find(it_begin, it->end(), end);
    PADDLE_ENFORCE(it_begin != it->end(), "error in parsing lod info");
    PADDLE_ENFORCE(it_end != it->end(), "error in parsing lod info");
    new_lod.emplace_back(it_begin, it_end + 1);
    // reset offset if tensor is copyed and sliced.
    std::transform(new_lod.back().begin(), new_lod.back().end(),
                   new_lod.back().begin(),
                   [start](int v) { return v - start; });
    PADDLE_ENFORCE_EQ(new_lod.back().front(), 0, "error in slice LoD");
  }
  PADDLE_ENFORCE_LE(new_lod.size(), in.size());
  return new_lod;
}

bool operator==(const LoD& a, const LoD& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto& a_level = a[i];
    const auto& b_level = b[i];
    if (a_level.size() != b_level.size()) {
      return false;
    }
    for (size_t j = 0; j < a_level.size(); j++) {
      if (a_level[j] != b_level[j]) {
        return false;
      }
    }
  }
  return true;
}

size_t LoDTensor::NumElements(size_t level, size_t idx) const {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(idx, NumElements(level));
  // the last level of LoD, just return number of records in Tensor
  if (level == NumLevels() - 1) {
    return lod_[level][idx + 1] - lod_[level][idx];
  }
  // high level of LoD, and there is another lower level, return number of
  // lower-level elements
  auto tmp = SliceInLevel(lod_, level, idx, idx + 1);
  PADDLE_ENFORCE_GE(tmp.size(), 2);
  // there is a 0 as a placeholder stored in LoD, so the number of elements
  // equals lod.size() - 1
  return tmp[1].size() - 1;
}

void LoDTensor::ShrinkLevels(size_t level_begin, size_t level_end) {
  auto new_lod = framework::SliceLevels(lod_, level_begin, level_end);
  lod_ = new_lod;
}

void LoDTensor::ShrinkInLevel(size_t level, size_t elem_begin,
                              size_t elem_end) {
  PADDLE_ENFORCE_LT(level, NumLevels());
  PADDLE_ENFORCE_LT(elem_begin, NumElements(level));
  PADDLE_ENFORCE_LT(elem_end, NumElements(level) + 1);

  auto new_lod = framework::SliceInLevel(lod_, level, elem_begin, elem_end);
  lod_ = new_lod;
}

std::string LoDTensor::SerializeToString() const {
  LoDTensorProto desc;

  // set data_type
  if (this->type() == typeid(bool)) desc.set_data_type(DataType::BOOL);
  if (this->type() == typeid(int16_t)) desc.set_data_type(DataType::INT16);
  if (this->type() == typeid(int32_t)) desc.set_data_type(DataType::INT32);
  if (this->type() == typeid(int64_t)) desc.set_data_type(DataType::INT64);
  // FIXME(dzh): there is no fp16 in standard c++
  if (this->type() == typeid(int_fast16_t)) desc.set_data_type(DataType::FP16);
  if (this->type() == typeid(float)) desc.set_data_type(DataType::FP16);
  if (this->type() == typeid(double)) desc.set_data_type(DataType::FP16);

  // set dims
  std::vector<int64_t> dims = vectorize(this->dims());
  for (auto& dim : dims) {
    desc.add_dims(dim);
  }

  // set lod information
  desc.set_lod_level(this->NumLevels());
  std::string desc_bytes = desc.SerializeAsString();

  // FIXME(dzh) : implement fix chunk size buffer.
  size_t DESC_SIZE = desc_bytes.size();
  size_t DATA_SIZE = holder_->size() - offset_;

  const size_t BUFFER_SIZE = DESC_SIZE + DATA_SIZE + 2 * sizeof(size_t);
  char* buffer =
      static_cast<char*>(memory::Alloc(platform::CPUPlace(), BUFFER_SIZE));

  // format: desc_size data_size, desc_bytes, data_bytes.
  platform::Place place = holder_->place();
  platform::Place src_place = platform::CPUPlace();
  platform::Place dst_place = platform::CPUPlace();

  memory::Copy(dst_place, buffer, src_place, &DESC_SIZE, sizeof(size_t));
  memory::Copy(dst_place, buffer + sizeof(size_t), src_place, &DATA_SIZE,
               sizeof(size_t));
  memory::Copy(dst_place, buffer + sizeof(size_t) * 2, src_place,
               desc_bytes.c_str(), desc_bytes.size());

  // copy gpu data to cpu implicitly.
  PADDLE_ENFORCE(this->numel() != 0, " Serialize a empty Tensor!");
  int element_width = holder->size() / this->numel();
  memory::Copy(
      dst_place, buffer + sizeof(size_t) * 2 + desc_bytes.size(), place,
      static_cast<char*>(holder_->ptr()) + offset_ / element_width, DATA_SIZE);

  return std::string(buffer, BUFFER_SIZE);
}

void LoDTensor::DeserializeFromString(const std::string& s) {
  size_t DESC_SIZE, DATA_SIZE;
  platform::Place src_place = platform::CPUPlace();
  platform::Place dst_place = holder_->place();
  memory::Copy(dst_place, &DESC_SIZE, src_place, s.c_str(), sizeof(size_t));
  memory::Copy(dst_place, &DATA_SIZE, src_place, s.c_str() + sizeof(size_t),
               sizeof(size_t));

  // parse LoDTensorDesc
  LoDTensorProto desc;
  desc.ParseFromArray(s.c_str() + sizeof(size_t) * 2, DESC_SIZE);

  std::vector<int64_t> dims;
  std::copy(desc.dims.begin(), desc.dims().end(), std::back_inserter(dims));
  this->Resize(dims);
  auto* ptr = this->mutable_data(dst_place);
  memory::Copy(dst_place, ptr, src_place,
               s.c_str() + sizeof(size_t) * 2 + DESC_SIZE, DATA_SIZE);
}

}  // namespace framework
}  // namespace paddle
