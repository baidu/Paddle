/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/common/layout.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/utils/any.h"
#include "paddle/utils/optional.h"
#include "paddle/utils/test_macros.h"

namespace phi {

/*
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
using LegacyLoD = std::vector<std::vector<size_t>>;
using LoD = LegacyLoD;

/// \brief The meta data of dense tensor. Take the structure type
/// and use all default operations.
///
struct TEST_API DenseTensorMeta {
  DenseTensorMeta();
  DenseTensorMeta(DataType dtype, const DDim& dims);
  DenseTensorMeta(DataType dtype, const DDim& dims, const DDim& stride);
  DenseTensorMeta(DataType dtype,
                  const DDim& dims,
                  DataLayout layout,
                  size_t offset = 0);
  DenseTensorMeta(DataType dtype,
                  const DDim& dims,
                  DataLayout layout,
<<<<<<< HEAD
                  LoD  lod,
=======
                  const LegacyLoD& legacy_lod,
>>>>>>> 337f0d225e59843df33c4ca189ae312b8c36a4f3
                  size_t offset = 0);

  DenseTensorMeta(const DenseTensorMeta& other);

  DenseTensorMeta& operator=(const DenseTensorMeta& other);
  DenseTensorMeta& operator=(DenseTensorMeta&& other);

  static DDim calc_strides(const DDim& dims);

  /// \brief Test whether the metadata is valid. Does not throw exceptions.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept;

  bool is_contiguous() const;

  bool is_scalar{false};
  /// \brief Determine whether using gpudnn speed-up library in the new dygraph.
  /// It maybe also support OneDNN library in the near future.
  bool use_gpudnn{true};
  DDim dims;
  DataType dtype{DataType::UNDEFINED};
  DataLayout layout{DataLayout::NCHW};
  LegacyLoD legacy_lod;
  size_t offset{0};
  DDim strides;
};

inline bool operator==(const DenseTensorMeta& lhs, const DenseTensorMeta& rhs) {
  return (lhs.is_scalar == rhs.is_scalar) && lhs.use_gpudnn == rhs.use_gpudnn &&
         (lhs.dims == rhs.dims) && (lhs.dtype == rhs.dtype) &&
         (lhs.layout == rhs.layout) && (lhs.legacy_lod == rhs.legacy_lod) &&
         (lhs.offset == rhs.offset) && (lhs.strides == rhs.strides);
}

struct StringTensorMeta {
  StringTensorMeta() = default;
  explicit StringTensorMeta(const DDim& dims);
  /// \brief Test whether the metadata is valid. Does not throw exceptions.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept;

  /// During the entire life cycle of a DenseTensor, the following attributes
  /// marked with `const` are expected to remain unchanged.
  bool is_scalar{false};
  DDim dims;
  size_t offset{0};
};

inline bool operator==(const StringTensorMeta& lhs,
                       const StringTensorMeta& rhs) {
  return (lhs.is_scalar == rhs.is_scalar) && (lhs.dims == rhs.dims) &&
         (lhs.offset == rhs.offset);
}

struct SparseTensorMeta {
  SparseTensorMeta() = default;
  explicit SparseTensorMeta(const DDim& dims);
  explicit SparseTensorMeta(const DDim& dims, const DataLayout& layout);
  /// \brief Test whether the metadata is valid. Does not throw exceptions.
  /// \return Whether the metadata is valid.
  bool valid() const noexcept;

  DDim dims;
  DataType dtype{DataType::UNDEFINED};
  DataLayout layout{DataLayout::NCHW};
};

inline bool operator==(const SparseTensorMeta& lhs,
                       const SparseTensorMeta& rhs) {
  return (lhs.dims == rhs.dims) && (lhs.layout == rhs.layout);
}

}  // namespace phi
