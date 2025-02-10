/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/lod_tensor.h"

#include <cstdint>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/version.h"

namespace paddle::framework {

std::string LegacyLoDToString(const LegacyLoD &lod) {
  std::ostringstream stream;
  stream << lod;
  return stream.str();
}

bool operator==(const LegacyLoD &a, const LegacyLoD &b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto &a_level = a[i];
    const auto &b_level = b[i];
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

bool CheckLegacyLoD(const LegacyLoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;
    // check: the first offset(the begin offset) of each level should be 0.
    if (level.front() != 0) return false;
    // check: all the offsets in a level should be non-descending
    if (!std::is_sorted(level.begin(), level.end())) {
      return false;
    }
  }
  // check: the lowest level's last offset should equals `tensor_height` if
  //        tensor_height>0.
  if (tensor_height > 0 &&
      static_cast<size_t>(tensor_height) != in.back().back())
    return false;

  // check: the higher level's last offset should equals the lower level's
  // size-1.
  // NOTE LegacyLoD store the levels from top to bottom, so the higher level
  // goes first.
  for (size_t level = 0; level < in.size() - 1; level++) {
    if (in[level].back() != in[level + 1].size() - 1) return false;
  }
  return true;
}

<<<<<<< HEAD
bool CheckAbsLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: all the offsets in a level should be ascending(no same items
    // allowed).
    if (!std::is_sorted(level.begin(), level.begin(), [](size_t a, size_t b) {
          if (a < b) return true;
          return false;
        })) {
      return false;
    }

    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;

    // check: the first offset of each level should be 0, and the last should be
    // the same(the height of underlying tensor).
    if (level.front() != 0) return false;
    if (tensor_height < 0) {
      tensor_height = static_cast<int>(level.back());
    } else if (static_cast<size_t>(tensor_height) != level.back()) {
      return false;
    }
  }
  return true;
}

using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;
LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD &lod,
                                        size_t start_idx,
                                        size_t end_idx,
                                        size_t start_level) {
  LoD sub_lod;

  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    PADDLE_ENFORCE_LE(start_idx,
                      end_idx,
                      platform::errors::InvalidArgument(
                          "The start index should be less than the end index, "
                          "but received start index is %d, end index is %d.",
                          start_idx,
                          end_idx));
    PADDLE_ENFORCE_LT(
        end_idx,
        lod[level_idx].size(),
        platform::errors::InvalidArgument(
            "The end index should be less than the LoD level size, but "
            "received end index is %d, LoD level size is %d.",
            end_idx,
            lod[level_idx].size()));
    std::vector<size_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }

  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

void SerializeToStream(std::ostream &os,
                       const phi::DenseTensor &tensor,
                       const platform::DeviceContext &dev_ctx) {
  {  // the 1st field, uint32_t version for DenseTensor
    os.write(
        reinterpret_cast<const char *>(&paddle::framework::kCurTensorVersion),
        sizeof(paddle::framework::kCurTensorVersion));
  }
  {
    // the 2st field, LoD information
    // uint64_t lod_level
    // uint64_t lod_level_1 size in byte.
    // int*     lod_level_1 data
    // ...
    const auto& lod = tensor.lod();
    uint64_t size = lod.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto &each : lod) {
      size = each.size() * sizeof(framework::LoD::value_type::value_type);
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }
  // the 3st field, Tensor
  paddle::framework::TensorToStream(
      os, static_cast<phi::DenseTensor>(tensor), dev_ctx);
}

void SerializeToStream(std::ostream &os, const phi::DenseTensor &tensor) {
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  const platform::DeviceContext *dev_ctx = nullptr;
  auto place = tensor.place();
  dev_ctx = pool.Get(place);
  SerializeToStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &os, phi::DenseTensor *tensor) {
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  const platform::DeviceContext *dev_ctx = nullptr;
  dev_ctx = pool.Get(phi::CPUPlace());
  DeserializeFromStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &is,
                           phi::DenseTensor *tensor,
                           const platform::DeviceContext &dev_ctx,
                           const size_t &seek,
                           const std::vector<int64_t> &shape) {
  {
    // the 1st field, unit32_t version for DenseTensor
    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
                      true,
                      phi::errors::InvalidArgument(
                          "Tensor version %u is not supported.", version));
    PADDLE_ENFORCE_EQ(
        version,
        0U,
        phi::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level = 0;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
  }
  // the 3st filed, Tensor
  paddle::framework::TensorFromStream(
      is, static_cast<phi::DenseTensor *>(tensor), dev_ctx, seek, shape);
}

void DeserializeFromStream(std::istream &is,
                           phi::DenseTensor *tensor,
                           const platform::DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for DenseTensor
    uint32_t version = 0;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(paddle::framework::IsTensorVersionSupported(version),
                      true,
                      phi::errors::InvalidArgument(
                          "Tensor version %u is not supported.", version));
    PADDLE_ENFORCE_EQ(
        version,
        0U,
        phi::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level = 0;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size = 0;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::vector<size_t> tmp(size / sizeof(size_t));
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(size));
      lod[i] = tmp;
    }
  }
  // the 3st filed, Tensor
  paddle::framework::TensorFromStream(
      is, static_cast<phi::DenseTensor *>(tensor), dev_ctx);
}

LoD ConvertToOffsetBasedLoD(const LoD &length_lod) {
  LoD offset_lod;
=======
LegacyLoD ConvertToOffsetBasedLegacyLoD(const LegacyLoD &length_lod) {
  LegacyLoD offset_lod;
>>>>>>> 4c9bc9e3cd7680200be9f244f9a5d374345a6741
  offset_lod.reserve(length_lod.size());
  for (const auto &item : length_lod) {
    std::vector<size_t> level;
    level.reserve(item.size() + 1);
    size_t tmp = 0;
    level.push_back(tmp);
    for (auto i : item) {
      tmp += i;
      level.push_back(tmp);
    }
    offset_lod.push_back(level);
  }
  return offset_lod;
}

}  // namespace paddle::framework
