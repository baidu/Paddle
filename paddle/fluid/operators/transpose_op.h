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

#pragma once

#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

enum { kTransposeMKLDNNFP32 = 1, kTransposeMKLDNNINT8 = 2 };

template <typename DeviceContext, typename T>
inline void TransCompute(const int dim,
                         const DeviceContext& dev_ctx,
                         const phi::DenseTensor& in,
                         phi::DenseTensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 0:
      phi::Copy<DeviceContext>(dev_ctx, in, dev_ctx.GetPlace(), false, out);
      break;
    case 1:
      phi::funcs::Transpose<DeviceContext, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      phi::funcs::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      phi::funcs::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      phi::funcs::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      phi::funcs::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      phi::funcs::Transpose<DeviceContext, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      phi::funcs::TransposeNormal<DeviceContext, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

enum PermuteType {
  kCopy = 1,
  kTranspose = 2,
  kVecPermute = 3,
  kNormalPermute = 4
};

constexpr int kBlockRows = 16;
constexpr int kTileSize = 32;

// Simplify the input dims and permute dims if possible.
template <typename T>
class DimsSimplifier {
 public:
  explicit DimsSimplifier(const int sm_count,
                          const size_t rank,
                          const int64_t numel,
                          const std::vector<int32_t>& perm,
                          const std::vector<int>& dims,
                          const T* src,
                          T* dst)
      : perm_(rank), src_dims(rank), count_(numel) {
    SimplifyPermAndDims(rank, dims, perm);
    if (rank_ > 1) {
      vec_size_ = GetPermVecSize(sm_count, src, dst);
    }
    perm_.resize(rank_);
    src_dims.resize(rank_);
    dst_dims.resize(rank_);

    for (auto i = 0; i < rank_; ++i) {
      dst_dims[i] = src_dims[perm_[i]];
    }
  }

  int GetRank() const { return rank_; }
  int GetVecSize() const { return vec_size_; }
  int64_t GetCount() const { return count_; }
  PermuteType GetPermType() const { return type_; }

  std::vector<int> GetPerm() const { return perm_; }
  std::vector<int> GetSrcDims() const { return src_dims; }
  std::vector<int> GetDstDims() const { return dst_dims; }

 private:
  int rank_{1};
  int64_t count_{0};
  int vec_size_{1};
  std::vector<int> perm_;
  std::vector<int> src_dims;
  std::vector<int> dst_dims;
  PermuteType type_{kCopy};

  void SimplifyPermAndDims(const size_t rank,
                           const std::vector<int>& in_dims,
                           const std::vector<int32_t>& perm) {
    int combined_dims[phi::DDim::kMaxRank];
    int valid_map[phi::DDim::kMaxRank];

    // Merge consecutive dims to the fist one dim and
    // leave original dim to be 1. Example below :
    // perm: [2, 3, 0, 1], origin_dims : [4, 8, 2, 5]
    // new_dims: [4, 8, 2, 5] -> [32, 1, 10, 1]
    int start_perm_idx = 0;
    while (start_perm_idx < rank) {
      const int start_dim_idx = perm[start_perm_idx];
      combined_dims[start_dim_idx] = in_dims[start_dim_idx];
      int end_perm_idx = start_perm_idx + 1;

      while (end_perm_idx < rank &&
             perm[end_perm_idx] == perm[end_perm_idx - 1] + 1) {
        const int end_dim_idx = perm[end_perm_idx];
        combined_dims[start_dim_idx] *= in_dims[end_dim_idx];
        combined_dims[end_dim_idx] = 1;
        end_perm_idx += 1;
      }
      start_perm_idx = end_perm_idx;
    }

    // Reorder combined dims and marked useless dim as -1.
    // for example, if combined dims is [32, 1, 10, 1],
    // valid_map is [0, -1, 1, -1] and generate simplified
    // dims as [32, 10]
    int valid_dim_idx = 0;
    bool sequential_flag = false;
    for (auto i = 0; i < rank; ++i) {
      const int src_dim = combined_dims[i];
      if (src_dim == 1) {
        valid_map[i] = -1;
      } else {
        sequential_flag = true;
        valid_map[i] = valid_dim_idx;
        src_dims[valid_dim_idx] = src_dim;
        valid_dim_idx += 1;
      }
    }

    if (valid_dim_idx == 0) {
      src_dims[0] = 1;
      perm_[0] = 0;
      return;
    } else if (valid_dim_idx == 1) {
      type_ = PermuteType::kCopy;
    }

    // Acquire simplified perm with help of combined dims
    // and original perm, finally simplified perm is [1, 0]
    int perm_idx = 0;
    for (auto i = 0; i < rank; ++i) {
      const int mapped = valid_map[perm[i]];
      if (mapped >= 0) {
        perm_[perm_idx] = mapped;
        perm_idx += 1;
      }
    }
    rank_ = valid_dim_idx;
  }

  int GetPermVecSize(const int sm_count, const T* src, T* dst) {
    // For gerneal_permute kernel, there is good chance for
    // vectorized write.
    type_ = PermuteType::kNormalPermute;
    int vec_size = phi::GetVectorizedSize<T>(dst);

    // While the last dim is fixed, there is good chance for
    // both vectorized read and write.
    if (perm_[rank_ - 1] == rank_ - 1) {
      int tmp_size = std::min(vec_size, phi::GetVectorizedSize<T>(src));
      tmp_size = GetDimVesSize(tmp_size, src_dims[rank_ - 1]);
      if (tmp_size > 1) {
        type_ = kVecPermute;
        vec_size = tmp_size;
      }
    }

    // Once only transpose at the last 2 dims, there is good
    // chance for vectorized read.
    if ((rank_ == 2 && perm_[1] == 0 && perm_[0] == 1) ||
        (rank_ == 3 && perm_[2] == 1 && perm_[1] == 2)) {
      type_ = PermuteType::kTranspose;
      int tmp_vec = std::min(vec_size, phi::GetVectorizedSize<T>(src));
      // With bytes limitation of shared_memory, the VecSize shall be
      // restricted for the type whose byte-size is less than 8 (double).
      vec_size =
          sizeof(T) > 8 ? 1 : GetDimVesSize(tmp_vec, src_dims[rank_ - 1]);
    }
    return vec_size;
  }

  // To find if highest common divisor and make it as vec_size.
  int GetDimVesSize(const int vec_size, const size_t target_dim) {
    int dim_vec_size = 1;
    for (auto size = vec_size; size > 0; size /= 2) {
      if (target_dim % size == 0) {
        dim_vec_size = size;
        break;
      }
    }
    return dim_vec_size;
  }
};

}  // namespace operators
}  // namespace paddle
