// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {
using FeedType = phi::DenseTensor;
using FetchType = paddle::variant<phi::DenseTensor,
                                  phi::TensorArray,
                                  phi::Vocab,
                                  phi::SparseCooTensor>;

template <>
struct PhiVectorType<FeedType> {
  const char* type_name = "PhiVectorFeedType";
};

class FetchVector : public phi::ExtendedTensor,
                    public phi::TypeInfoTraits<phi::TensorBase, FetchVector> {
 public:
  FetchVector() = default;

  explicit FetchVector(const std::vector<FetchType>& init_data)
      : data_(init_data) {}

  FetchVector(FetchVector&& other) = default;

  FetchVector(const FetchVector& other) = default;

  FetchVector& operator=(const FetchVector& other) = default;

  FetchVector& operator=(const std::vector<FetchType>& other) {
    data_ = other;
    return *this;
  }

  FetchVector& operator=(FetchVector&& other) = default;

  /// \brief Destroy the PhiVector and release exclusive resources.
  virtual ~FetchVector() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "FetchVector"; }

  size_t size() const { return data_.size(); }

  bool empty() const { return data_.empty(); }

  const FetchType& back() const { return data_.back(); }

  FetchType& back() { return data_.back(); }

  void resize(size_t size) { data_.resize(size); }

  void clear() { data_.clear(); }

  void emplace_back(const FetchType& feed_data) {
    data_.emplace_back(feed_data);
  }

  void emplace_back() { data_.emplace_back(); }

  void push_back(const FetchType& feed_data) { data_.push_back(feed_data); }

  void pop_back() { data_.pop_back(); }

  const FetchType& operator[](size_t index) const { return data_[index]; }

  FetchType& operator[](size_t index) { return data_[index]; }

  FetchType& at(size_t index) { return data_.at(index); }

  const FetchType& at(size_t index) const { return data_.at(index); }

  typename std::vector<FetchType>::iterator begin() { return data_.begin(); }

  typename std::vector<FetchType>::const_iterator begin() const {
    return data_.begin();
  }

  typename std::vector<FetchType>::iterator end() { return data_.end(); }

  typename std::vector<FetchType>::const_iterator end() const {
    return data_.end();
  }

 private:
  std::vector<FetchType> data_;
};

using FeedList = PhiVector<FeedType>;
using FetchList = FetchVector;
using FetchUnmergedList = std::vector<std::vector<FetchType>>;

}  // namespace phi

namespace paddle {
namespace framework {
using FeedType = phi::FeedType;
using FetchType = phi::FetchType;
using FeedList = phi::FeedList;
using FetchList = phi::FetchList;
using FetchUnmergedList = phi::FetchUnmergedList;
}  // namespace framework
}  // namespace paddle
