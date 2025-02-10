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

#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/dynload/flashattn.h"
#endif

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/arange_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {
namespace funcs {

template <typename T>
__global__ void SimpleScaleKernel(int64_t numel, float scale, T* inout) {
  CUDA_KERNEL_LOOP_TYPE(i, numel, int64_t) {
    inout[i] = static_cast<T>(scale * static_cast<float>(inout[i]));
  }
}

inline std::string MemoryDebugString(const phi::DenseTensor& t) {
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  int64_t allocated =
      phi::memory_utils::DeviceMemoryStatCurrentValue("Allocated", device_id);
  int64_t reserved =
      phi::memory_utils::DeviceMemoryStatCurrentValue("Reserved", device_id);

  std::stringstream ss;
  ss << "shape=[" << t.dims()
     << "], size=" << static_cast<float>(t.memory_size()) / (1 << 20)
     << " MB, ptr=" << t.data()
     << "; [MEMORY] allocated=" << static_cast<float>(allocated) / (1 << 20)
     << " MB"
     << ", reserved=" << static_cast<float>(reserved) / (1 << 20) << " MB";
  return ss.str();
}

template <typename T>
void AllocWithDebugInfo(const phi::GPUContext& dev_ctx,
                        const std::string& info,
                        phi::DenseTensor* t) {
  dev_ctx.Alloc<T>(t, t->numel() * sizeof(T));
  if (VLOG_IS_ON(4)) {
    VLOG(4) << info << ": " << MemoryDebugString(*t);
  }
}

inline std::string TensorDebugString(const phi::DenseTensor* t,
                                     const std::string& info) {
  std::stringstream ss;
  ss << info << ": ";
  if (t) {
    if (t->initialized()) {
      ss << "shape=[" << t->dims() << "], ptr=" << t->data();
    } else {
      ss << "not initialized";
    }
  } else {
    ss << "nullptr";
  }
  return ss.str();
}

inline void WaitWithDebugInfo(const phi::GPUContext& dev_ctx) {
  if (VLOG_IS_ON(5)) {
    dev_ctx.Wait();
    VLOG(5) << "[Flash attn Synchronize] ";
  }
}

template <typename T>
inline void TypeDebugInfo() {
  if (VLOG_IS_ON(4)) {
    if (std::is_same<T, phi::dtype::float16>::value) {
      VLOG(4) << "[Grad]: T is phi::dtype::float16.";
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
      VLOG(4) << "[Grad]: T is phi::dtype::bfloat16.";
    } else if (std::is_same<T, float>::value) {
      VLOG(4) << "[Grad]: T is float.";
    }
  }
}

template <typename T>
struct TernaryAddFunctor {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
struct GateAttentionConfig {
 public:
  const phi::GPUContext& dev_ctx;

  bool merge_qkv;
  bool has_gating;
  bool use_flash_attn;

  int64_t batch_size;
  int64_t seq_len_m;
  int64_t seq_len_r;
  int64_t q_dim;
  int64_t kv_dim;
  int64_t head_dim;
  int64_t m_size;
  int64_t num_heads;

  phi::DDim qkv_out_dims;
  phi::DDim qkv_transpose_out_dims;

  phi::DDim q_out_dims;
  phi::DDim kv_out_dims;
  phi::DDim q_transpose_out_dims;
  phi::DDim kv_transpose_out_dims;

  phi::DDim qk_out_dims;
  phi::DDim softmax_out_dims;
  phi::DDim qktv_out_dims;
  phi::DDim gate_out_dims;

  GateAttentionConfig(const phi::GPUContext& dev_ctx,
                      const phi::DenseTensor* query,
                      const phi::DenseTensor* key,
                      const phi::DenseTensor* query_weight,
                      const phi::DenseTensor* qkv_weight,
                      bool merge_qkv,
                      bool has_gating,
                      bool use_flash_attn)
      : dev_ctx(dev_ctx),
        merge_qkv(merge_qkv),
        has_gating(has_gating),
        use_flash_attn(use_flash_attn) {
    // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
    batch_size = query->dims()[0];
    seq_len_m = query->dims()[1];
    seq_len_r = query->dims()[2];
    q_dim = query->dims()[3];

    if (merge_qkv) {
      PADDLE_ENFORCE_NOT_NULL(
          qkv_weight,
          common::errors::NotFound("The input qkv_weight can not be nullptr "
                                   "when merge_qkv is true."));

      // When q_dim == kv_dim, QKV matmul can be computed merged.
      // qkv_weight: shape=[3, num_heads, head_dim, q_dim]
      num_heads = qkv_weight->dims()[1];
      head_dim = qkv_weight->dims()[2];
      m_size = seq_len_r;
      kv_dim = q_dim;

      qkv_out_dims = {batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim};
      qkv_transpose_out_dims = {
          3, batch_size, seq_len_m, num_heads, seq_len_r, head_dim};
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          key,
          common::errors::NotFound(
              "The input key can not be nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          query_weight,
          common::errors::NotFound("The input query_weight can not be "
                                   "nullptr when merge_qkv is false."));

      // When q_dim != kv_dim, QKV matmul must be computed saparately.
      // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
      // query_w: shape=[q_dim, num_heads, head_dim]
      num_heads = query_weight->dims()[1];
      head_dim = query_weight->dims()[2];
      m_size = key->dims()[2];
      kv_dim = key->dims()[3];

      q_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, head_dim};
      kv_out_dims = {batch_size, seq_len_m, m_size, num_heads, head_dim};
      q_transpose_out_dims = {
          batch_size, seq_len_m, num_heads, seq_len_r, head_dim};
      kv_transpose_out_dims = {
          batch_size, seq_len_m, num_heads, m_size, head_dim};
    }

    qk_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, m_size};
    softmax_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, m_size};
    qktv_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, head_dim};
    gate_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, head_dim};
  }

  bool CanUseFlashAttn() const {
#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
    if (!std::is_same<T, phi::dtype::bfloat16>::value &&
        !std::is_same<T, phi::dtype::float16>::value) {
      return false;
    }

    if (merge_qkv && batch_size == 1) {
      if (head_dim == 32 || head_dim == 64 || head_dim == 128) {
        return use_flash_attn;
      }
    }
#endif
    return false;
  }

  int64_t GetQuerySize() const {
    return batch_size * seq_len_m * seq_len_r * num_heads * head_dim;
  }

  phi::DenseTensor* GetQKVOut() {
    if (!qkv_out.IsInitialized()) {
      qkv_out.Resize(qkv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "qkv_out", &qkv_out);
    }
    return &qkv_out;
  }

  phi::DenseTensor* GetQueryOut() {
    if (!query_out.IsInitialized()) {
      query_out.Resize(q_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "query_out", &query_out);
    }
    return &query_out;
  }

  phi::DenseTensor* GetKeyOut() {
    if (!key_out.IsInitialized()) {
      key_out.Resize(kv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "key_out", &key_out);
    }
    return &key_out;
  }

  phi::DenseTensor* GetValueOut() {
    if (!value_out.IsInitialized()) {
      value_out.Resize(kv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "value_out", &value_out);
    }
    return &value_out;
  }

  phi::DenseTensor* GetQKOut(phi::DenseTensor* softmax_out) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = m_size;
    if (!softmax_out || phi::UseCudnnSoftmax<T>(dev_ctx, softmax_dim, true)) {
      // Not sure whether cudnn softmax can execute inplace.
      if (!qkv_out.IsInitialized()) {
        qk_out.Resize(qk_out_dims);
        AllocWithDebugInfo<T>(dev_ctx, "qk_out", &qk_out);
      }
      return &qk_out;
    } else {
      // Enable inplace softmax.
      return softmax_out;
    }
  }

  phi::DenseTensor* GetQKTVOut(phi::DenseTensor* gate_out) {
    if (has_gating && gate_out) {
      // Reuse gate_out.
      gate_out->Resize(qktv_out_dims);
      return gate_out;
    } else {
      if (!qktv_out.IsInitialized()) {
        qktv_out.Resize(qktv_out_dims);
        AllocWithDebugInfo<T>(dev_ctx, "qktv_out", &qktv_out);
      }
      return &qktv_out;
    }
  }

  void ClearQKVOut() {
    if (qkv_out.IsInitialized()) {
      qkv_out.clear();
    }
  }

  void ClearQKOut() {
    if (qk_out.IsInitialized()) {
      qk_out.clear();
    }
  }

  void ClearQKTVOut() {
    if (qktv_out.IsInitialized()) {
      qktv_out.clear();
    }
  }

 protected:
  phi::DenseTensor qkv_out;
  phi::DenseTensor query_out;
  phi::DenseTensor key_out;
  phi::DenseTensor value_out;
  // qk_out = BatchedGEMM(Q, K^T)
  // qk_out: shape=[batch_size, seq_len_m, num_heads, seq_len_r, m_size]
  // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
  // The shape of qk_out, softmax_out is the same, thus can be called inplace.
  phi::DenseTensor qk_out;
  // qktv_out may reuse gate_out.
  phi::DenseTensor qktv_out;
};

template <typename T>
struct GateAttentionGradConfig : public GateAttentionConfig<T> {
 public:
  GateAttentionGradConfig(const phi::GPUContext& dev_ctx,
                          const phi::DenseTensor* query,
                          const phi::DenseTensor* key,
                          const phi::DenseTensor* query_weight,
                          const phi::DenseTensor* qkv_weight,
                          bool merge_qkv,
                          bool has_gating,
                          bool use_flash_attn)
      : GateAttentionConfig<T>(dev_ctx,
                               query,
                               key,
                               query_weight,
                               qkv_weight,
                               merge_qkv,
                               has_gating,
                               use_flash_attn) {}

  phi::DenseTensor* GetQKVOutGrad() {
    if (!qkv_out_grad.IsInitialized()) {
      qkv_out_grad.Resize(this->qkv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "qkv_out_grad", &qkv_out_grad);
    }
    return &qkv_out_grad;
  }

  phi::DenseTensor* GetQueryOutGrad() {
    if (!query_out_grad.IsInitialized()) {
      query_out_grad.Resize(this->q_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "query_out_grad", &query_out_grad);
    }
    return &query_out_grad;
  }

  phi::DenseTensor* GetKeyOutGrad() {
    if (!key_out_grad.IsInitialized()) {
      key_out_grad.Resize(this->kv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "key_out_grad", &key_out_grad);
    }
    return &key_out_grad;
  }

  phi::DenseTensor* GetValueOutGrad() {
    if (!value_out_grad.IsInitialized()) {
      value_out_grad.Resize(this->kv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "value_out_grad", &value_out_grad);
    }
    return &value_out_grad;
  }

  phi::DenseTensor* GetQKOutGrad(phi::DenseTensor* softmax_out_grad) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = this->m_size;
    if (!softmax_out_grad ||
        phi::UseCudnnSoftmax<T>(this->dev_ctx, softmax_dim, true)) {
      if (!qk_out_grad.IsInitialized()) {
        qk_out_grad.Resize(this->qk_out_dims);
        AllocWithDebugInfo<T>(this->dev_ctx, "qk_out_grad", &qk_out_grad);
      }
      return &qk_out_grad;
    } else {
      return softmax_out_grad;
    }
  }

 protected:
  phi::DenseTensor qkv_out_grad;
  phi::DenseTensor query_out_grad;
  phi::DenseTensor key_out_grad;
  phi::DenseTensor value_out_grad;
  phi::DenseTensor qk_out_grad;
};

template <typename T>
class FMHAGateRef {
 public:
  FMHAGateRef(const phi::GPUContext& dev_ctx, bool merge_qkv)
      : dev_ctx_(dev_ctx), merge_qkv_(merge_qkv) {}

  void ComputeForward(const phi::DenseTensor* nonbatched_bias,
                      const phi::DenseTensor* src_mask,
                      phi::DenseTensor* q_transpose_out,
                      phi::DenseTensor* k_transpose_out,
                      phi::DenseTensor* v_transpose_out,
                      phi::DenseTensor* qkv_transpose_out,
                      phi::DenseTensor* softmax_out,
                      phi::DenseTensor* fmha_out,
                      phi::DenseTensor* gate_out,
                      GateAttentionConfig<T>* config) {
    T* q_ptr = nullptr;
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;

    if (merge_qkv_) {
      // qkv_transpose_out = transpose(qkv_out)
      PADDLE_ENFORCE_NOT_NULL(
          qkv_transpose_out,
          common::errors::NotFound("The input qkv_transpose_out can not be "
                                   "nullptr when merge_qkv is true."));

      phi::DenseTensor* qkv_out = config->GetQKVOut();
      ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);
      config->ClearQKVOut();

      // q_size == k_size
      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          q_transpose_out,
          common::errors::NotFound("The input q_transpose_out can not be "
                                   "nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          k_transpose_out,
          common::errors::NotFound("The input k_transpose_out can not be "
                                   "nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          v_transpose_out,
          common::errors::NotFound("The input v_transpose_out can not be "
                                   "nullptr when merge_qkv is false."));

      phi::DenseTensor* query_out = config->GetQueryOut();
      phi::DenseTensor* key_out = config->GetKeyOut();
      phi::DenseTensor* value_out = config->GetValueOut();
      ComputeQKVTransposeForward(*query_out,
                                 *key_out,
                                 *value_out,
                                 q_transpose_out,
                                 k_transpose_out,
                                 v_transpose_out);

      // q_size != k_size
      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();
    }
    // qk_out = BatchedGEMM(Q, K^T)
    // [batch_size, seq_len_m, num_heads, seq_len_r, head_dim] *
    //                [batch_size, seq_len_m, num_heads, m_size, head_dim]
    // -> [batch_size, seq_len_m, num_heads, seq_len_r, m_size]
    phi::DenseTensor* qk_out = config->GetQKOut(softmax_out);
    T* qk_out_ptr = qk_out->data<T>();

    int64_t gemm_batch_size =
        config->batch_size * config->seq_len_m * config->num_heads;
    int64_t gemm_m = config->seq_len_r;
    int64_t gemm_n = config->m_size;
    int64_t gemm_k = config->head_dim;
    T alpha = static_cast<T>(1.0 / sqrt(config->head_dim));
    // attn = matmul(q, k.transpose(-1, -2))
    ComputeBatchedGEMM(q_ptr,
                       k_ptr,
                       qk_out_ptr,
                       false,
                       true,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       gemm_batch_size,
                       alpha);

    // attn = softmax_dropout(attn, 0, self.training, mask=mask, bias=bias)
    // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
    ComputeBiasMaskSoftmaxForward(
        nonbatched_bias, src_mask, qk_out, softmax_out);
    config->ClearQKOut();

    // qktv_out = BatchedGEMM(softmax_out, V)
    // [batch_size, seq_len_m, num_heads, seq_len_r, m_size] *
    // [batch_size, seq_len_m, num_heads, m_size,    head_dim]
    // -> [batch_size, seq_len_m, num_heads, seq_len_r, head_dim]
    phi::DenseTensor* qktv_out = config->GetQKTVOut(gate_out);
    T* qktv_out_ptr = qktv_out->data<T>();

    gemm_m = config->seq_len_r;
    gemm_n = config->head_dim;
    gemm_k = config->m_size;

    // o = matmul(attn, v)
    T* softmax_out_ptr = softmax_out->data<T>();
    ComputeBatchedGEMM(softmax_out_ptr,
                       v_ptr,
                       qktv_out_ptr,
                       false,
                       false,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       gemm_batch_size);

    // fmha_out = transpose(qktv_out)
    // o = o.transpose(-2, -3).contiguous()
    ComputeQKTVTransposeForward(*qktv_out, fmha_out);

    config->ClearQKTVOut();
    if (config->has_gating) {
      gate_out->Resize(config->gate_out_dims);
    }
  }

  void ComputeBackward(const phi::DenseTensor* q_transpose_out,
                       const phi::DenseTensor* k_transpose_out,
                       const phi::DenseTensor* v_transpose_out,
                       const phi::DenseTensor* qkv_transpose_out,
                       const phi::DenseTensor* softmax_out,
                       const phi::DenseTensor* fmha_out_grad,
                       phi::DenseTensor* src_mask_grad,
                       phi::DenseTensor* nonbatched_bias_grad,
                       GateAttentionGradConfig<T>* config) {
    const T* q_ptr = nullptr;
    const T* k_ptr = nullptr;
    const T* v_ptr = nullptr;

    T* q_grad_ptr = nullptr;
    T* k_grad_ptr = nullptr;
    T* v_grad_ptr = nullptr;

    phi::DenseTensor q_transpose_out_grad;
    phi::DenseTensor k_transpose_out_grad;
    phi::DenseTensor v_transpose_out_grad;
    phi::DenseTensor qkv_transpose_out_grad;

    if (merge_qkv_) {
      PADDLE_ENFORCE_NOT_NULL(
          qkv_transpose_out,
          common::errors::NotFound("The input qkv_transpose_out can not be "
                                   "nullptr when merge_qkv is true."));

      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;

      qkv_transpose_out_grad.Resize(config->qkv_transpose_out_dims);
      AllocWithDebugInfo<T>(
          dev_ctx_, "qkv_transpose_out_grad", &qkv_transpose_out_grad);

      q_grad_ptr = qkv_transpose_out_grad.data<T>();
      k_grad_ptr = q_grad_ptr + q_size;
      v_grad_ptr = k_grad_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          q_transpose_out,
          common::errors::NotFound("The input q_transpose_out can not be "
                                   "nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          k_transpose_out,
          common::errors::NotFound("The input k_transpose_out can not be "
                                   "nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          v_transpose_out,
          common::errors::NotFound("The input v_transpose_out can not be "
                                   "nullptr when merge_qkv is false."));

      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();

      q_transpose_out_grad.Resize(config->q_transpose_out_dims);
      k_transpose_out_grad.Resize(config->kv_transpose_out_dims);
      v_transpose_out_grad.Resize(config->kv_transpose_out_dims);

      q_grad_ptr = dev_ctx_.Alloc<T>(&q_transpose_out_grad,
                                     q_transpose_out_grad.numel() * sizeof(T));
      k_grad_ptr = dev_ctx_.Alloc<T>(&k_transpose_out_grad,
                                     k_transpose_out_grad.numel() * sizeof(T));
      v_grad_ptr = dev_ctx_.Alloc<T>(&v_transpose_out_grad,
                                     v_transpose_out_grad.numel() * sizeof(T));
    }

    phi::DenseTensor softmax_out_grad;
    softmax_out_grad.Resize(config->softmax_out_dims);
    AllocWithDebugInfo<T>(dev_ctx_, "softmax_out_grad", &softmax_out_grad);

    int64_t gemm_batch_size =
        config->batch_size * config->seq_len_m * config->num_heads;
    {
      // Forward: fmha_out = transpose(qktv_out)
      phi::DenseTensor qktv_out_grad;
      qktv_out_grad.Resize(config->qktv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx_, "qktv_out_grad", &qktv_out_grad);
      ComputeQKTVTransposeBackward(*fmha_out_grad, &qktv_out_grad);

      // Forward: qktv_out = BatchedGEMM(softmax_out, V)
      // Backward:
      //  V_grad = BatchedGEMM(softmax_out^T, qktv_out_grad) (dy = x^T * dout)
      int64_t gemm_m = config->m_size;
      int64_t gemm_n = config->head_dim;
      int64_t gemm_k = config->seq_len_r;

      const T* softmax_out_ptr = softmax_out->data<T>();
      const T* qktv_out_grad_ptr = qktv_out_grad.data<T>();
      ComputeBatchedGEMM(softmax_out_ptr,
                         qktv_out_grad_ptr,
                         v_grad_ptr,
                         true,
                         false,
                         gemm_m,
                         gemm_n,
                         gemm_k,
                         gemm_batch_size);

      // Backward: softmax_out_grad = qktv_out_grad * V^T (dx = dout * y^T)
      gemm_m = config->seq_len_r;
      gemm_n = config->m_size;
      gemm_k = config->head_dim;

      T* softmax_out_grad_ptr = softmax_out_grad.data<T>();
      ComputeBatchedGEMM(qktv_out_grad_ptr,
                         v_ptr,
                         softmax_out_grad_ptr,
                         false,
                         true,
                         gemm_m,
                         gemm_n,
                         gemm_k,
                         gemm_batch_size);
    }

    phi::DenseTensor* qk_out_grad = config->GetQKOutGrad(&softmax_out_grad);
    ComputeBiasMaskSoftmaxBackward(&softmax_out_grad,
                                   softmax_out,
                                   src_mask_grad,
                                   qk_out_grad,
                                   nonbatched_bias_grad);

    // Forward: qk_out = BatchedGEMM(Q, K^T)
    // Backward: k_grad = BatchedGEMM(qk_out_grad^T, Q) (dy = dout^t * x)
    int64_t gemm_m = config->m_size;
    int64_t gemm_n = config->head_dim;
    int64_t gemm_k = config->seq_len_r;
    T alpha = static_cast<T>(1.0 / sqrt(config->head_dim));

    T* qk_out_grad_ptr = qk_out_grad->data<T>();
    ComputeBatchedGEMM(qk_out_grad_ptr,
                       q_ptr,
                       k_grad_ptr,
                       true,
                       false,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       gemm_batch_size,
                       alpha);

    // Backward: q_grad = BatchedGEMM(qk_out_grad, K) (dx = dout * y)
    gemm_m = config->seq_len_r;
    gemm_n = config->head_dim;
    gemm_k = config->m_size;
    ComputeBatchedGEMM(qk_out_grad_ptr,
                       k_ptr,
                       q_grad_ptr,
                       false,
                       false,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       gemm_batch_size,
                       alpha);

    if (merge_qkv_) {
      phi::DenseTensor* qkv_out_grad = config->GetQKVOutGrad();
      ComputeQKVTransposeBackward(qkv_transpose_out_grad, qkv_out_grad);
    } else {
      phi::DenseTensor* q_out_grad = config->GetQueryOutGrad();
      phi::DenseTensor* k_out_grad = config->GetKeyOutGrad();
      phi::DenseTensor* v_out_grad = config->GetValueOutGrad();
      ComputeQKVTransposeBackward(q_transpose_out_grad,
                                  k_transpose_out_grad,
                                  v_transpose_out_grad,
                                  q_out_grad,
                                  k_out_grad,
                                  v_out_grad);
    }
  }

  void ComputeQKVTransposeForward(const phi::DenseTensor& q_out,
                                  const phi::DenseTensor& k_out,
                                  const phi::DenseTensor& v_out,
                                  phi::DenseTensor* q_transpose_out,
                                  phi::DenseTensor* k_transpose_out,
                                  phi::DenseTensor* v_transpose_out) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, q_out, perm, q_transpose_out);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, k_out, perm, k_transpose_out);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, v_out, perm, v_transpose_out);
  }

  void ComputeQKVTransposeBackward(const phi::DenseTensor& q_transpose_out_grad,
                                   const phi::DenseTensor& k_transpose_out_grad,
                                   const phi::DenseTensor& v_transpose_out_grad,
                                   phi::DenseTensor* q_out_grad,
                                   phi::DenseTensor* k_out_grad,
                                   phi::DenseTensor* v_out_grad) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, q_transpose_out_grad, perm, q_out_grad);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, k_transpose_out_grad, perm, k_out_grad);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, v_transpose_out_grad, perm, v_out_grad);
  }

  // [batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim] ->
  //         [3, batch_size, seq_len_m, num_heads, seq_len_r, head_dim]
  void ComputeQKVTransposeForward(const phi::DenseTensor& qkv_out,
                                  phi::DenseTensor* qkv_transpose_out) {
    std::vector<int> perm = {3, 0, 1, 4, 2, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_out, perm, qkv_transpose_out);
  }

  void ComputeQKVTransposeBackward(
      const phi::DenseTensor& qkv_transpose_out_grad,
      phi::DenseTensor* qkv_out_grad) {
    std::vector<int> perm = {1, 2, 4, 0, 3, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_transpose_out_grad, perm, qkv_out_grad);
  }

  // [batch_size, seq_len_m, num_head, seq_len_r, c] ->
  //         [batch_size, seq_len_m, seq_len_r, num_head, c]
  void ComputeQKTVTransposeForward(const phi::DenseTensor& qktv_out,
                                   phi::DenseTensor* fmha_out) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(dev_ctx_, qktv_out, perm, fmha_out);
  }

  void ComputeQKTVTransposeBackward(const phi::DenseTensor& fmha_out_grad,
                                    phi::DenseTensor* qktv_out_grad) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, fmha_out_grad, perm, qktv_out_grad);
  }

  // qk_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxForward(const phi::DenseTensor* nonbatched_bias,
                                     const phi::DenseTensor* src_mask,
                                     phi::DenseTensor* qk_out,
                                     phi::DenseTensor* softmax_out) {
    if (nonbatched_bias) {
      std::vector<const phi::DenseTensor*> ins = {
          qk_out, src_mask, nonbatched_bias};
      std::vector<phi::DenseTensor*> outs = {qk_out};
      phi::funcs::BroadcastKernel<T>(
          dev_ctx_, ins, &outs, TernaryAddFunctor<T>());
    } else {
      std::vector<const phi::DenseTensor*> ins = {qk_out, src_mask};
      std::vector<phi::DenseTensor*> outs = {qk_out};
      phi::funcs::BroadcastKernel<T>(
          dev_ctx_, ins, &outs, phi::funcs::AddFunctor<T>());
    }
    phi::SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *qk_out, -1, softmax_out);
  }

  // src_mask_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxBackward(const phi::DenseTensor* softmax_out_grad,
                                      const phi::DenseTensor* softmax_out,
                                      phi::DenseTensor* src_mask_grad,
                                      phi::DenseTensor* qk_out_grad,
                                      phi::DenseTensor* nonbatched_bias_grad) {
    PADDLE_ENFORCE_NOT_NULL(
        qk_out_grad,
        common::errors::NotFound("The qk_out_grad can not be nullptr."));

    PADDLE_ENFORCE_EQ(qk_out_grad->dims(),
                      softmax_out->dims(),
                      common::errors::InvalidArgument(
                          "The shape of qk_out_grad and softmax_out is "
                          "expected to be the same. But received qk_out_grad's "
                          "shape = %s, softmax_out's shape = %s.",
                          qk_out_grad->dims(),
                          softmax_out->dims()));

    PADDLE_ENFORCE_EQ(src_mask_grad,
                      nullptr,
                      common::errors::InvalidArgument(
                          "src_mask_grad is expected to be nullptr."));

    phi::SoftmaxBackwardCUDAKernelDriver<T>(
        dev_ctx_, *softmax_out, *softmax_out_grad, -1, qk_out_grad);

    if (nonbatched_bias_grad) {
      // [batch_size, seq_len_m, num_heads, seq_len_r, m_size] ->
      //      [batch_size, 1, num_heads, seq_len_r, m_size]
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx_,
          *qk_out_grad,
          nonbatched_bias_grad,
          kps::IdentityFunctor<T>(),
          {1});
    }
  }

 private:
  void ComputeBatchedGEMM(const T* a_ptr,
                          const T* b_ptr,
                          T* c_ptr,
                          bool trans_a,
                          bool trans_b,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          int64_t batch_size,
                          T alpha = static_cast<T>(1.0),
                          T beta = static_cast<T>(0.0)) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;

    CBLAS_TRANSPOSE cblas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE cblas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    blas.BatchedGEMM(cblas_trans_a,
                     cblas_trans_b,
                     m,
                     n,
                     k,
                     alpha,
                     a_ptr,
                     b_ptr,
                     beta,
                     c_ptr,
                     batch_size,
                     stride_a,
                     stride_b);
  }

  const phi::GPUContext& dev_ctx_;
  bool merge_qkv_;
};

template <typename T>
class FlashAttnWithGating {
 public:
  FlashAttnWithGating(const phi::GPUContext& dev_ctx, bool merge_qkv)
      : dev_ctx_(dev_ctx), merge_qkv_(merge_qkv) {}

  void ComputeForward(const phi::DenseTensor* nonbatched_bias,
                      const phi::DenseTensor* src_mask,
                      phi::DenseTensor* qkv_transpose_out,
                      phi::DenseTensor* softmax_lse,
                      phi::DenseTensor* fmha_out,
                      GateAttentionConfig<T>* config) {
#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
    bool is_bf16 =
        qkv_transpose_out->dtype() == DataType::BFLOAT16 ? true : false;
    TypeDebugInfo<T>();

    PADDLE_ENFORCE_NOT_NULL(
        qkv_transpose_out,
        common::errors::NotFound("The input qkv_transpose_out can not be "
                                 "nullptr when merge_qkv is true."));

    // 1. Transpose qkv_out for flash_attn.
    phi::DenseTensor* qkv_out = config->GetQKVOut();
    ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);
    config->ClearQKVOut();

    // q_size == k_size
    int64_t q_size = config->GetQuerySize();
    T* q_ptr = qkv_transpose_out->data<T>();
    T* k_ptr = q_ptr + q_size;
    T* v_ptr = k_ptr + q_size;

    // 2. Scale Q: q_ptr = alpha * q_ptr
    ComputeScaleQ(q_size, config->head_dim, q_ptr);

    // 3. flash_attn parameter setting.
    phi::DenseTensor cu_seq_q;
    phi::DenseTensor cu_seq_k;
    InitArgumentsAndSeqTensors(config, &cu_seq_q, &cu_seq_k);

    std::vector<int64_t> temp_mask_dim = GetCompressedDim(src_mask);
    std::vector<int64_t> temp_bias_dim = GetCompressedDim(nonbatched_bias);

    softmax_lse->Resize({fa_batch_size_, fa_num_heads_, fa_softmax_lse_dim_});
    AllocWithDebugInfo<float>(dev_ctx_, "softmax_lse", softmax_lse);

    if (VLOG_IS_ON(6)) {
      VLOG(6) << "temp_mask_dim={" << common::make_ddim(temp_mask_dim) << "}";
      VLOG(6) << "temp_bias_dim={" << common::make_ddim(temp_bias_dim) << "}";
      VLOG(6) << TensorDebugString(&cu_seq_q, "cu_seq_q");
      VLOG(6) << TensorDebugString(&cu_seq_k, "cu_seq_k");
      VLOG(6) << TensorDebugString(nonbatched_bias, "nonbatched_bias");
      VLOG(6) << TensorDebugString(src_mask, "src_mask");
      VLOG(6) << TensorDebugString(qkv_transpose_out, "qkv_transpose_out");
      VLOG(6) << TensorDebugString(softmax_lse, "softmax_lse");
      VLOG(6) << TensorDebugString(fmha_out, "fmha_out");
    }

    // 4. Get workspace size and run the flash-attention kernel.
    uint64_t workspace_size = 0;
    phi::DenseTensor workspace;
    cudaStream_t stream = dev_ctx_.stream();
    for (bool need_calc : {false, true}) {
      // first calling, need_calc=false, set out_ptr to nullptr to calculate
      // workspace size second calling, need_calc=true, run flash-attention
      // kernel.
      void* out_ptr =
          need_calc ? static_cast<void*>(fmha_out->data()) : nullptr;
      void* workspace_ptr = nullptr;
      if (need_calc) {
        VLOG(6) << "Step 2: Call the flash-attention kernel";
        if (workspace_size > 0) {
          workspace = CreateWorkspace(workspace_size);
          workspace_ptr = static_cast<void*>(workspace.data());
        }
      } else {
        VLOG(6) << "Step 1: Calculate the workspace_size";
      }
      bool succ = phi::dynload::flash_attn_fwd_with_bias_and_mask(
          static_cast<const void*>(q_ptr),
          static_cast<const void*>(k_ptr),
          static_cast<const void*>(v_ptr),
          out_ptr,  // set out to nullptr to calculate workspace size
          cu_seq_q.data<int32_t>(),
          cu_seq_k.data<int32_t>(),
          fa_total_q_,
          fa_total_k_,
          fa_batch_size_,
          fa_num_heads_,
          fa_head_size_,
          fa_max_seqlen_q_,
          fa_max_seqlen_k_,
          fa_dropout_prob_,
          fa_softmax_scale_,
          fa_zero_tensors_,
          is_bf16,
          fa_num_splits_,
          softmax_lse->data(),
          workspace_ptr,
          &workspace_size,
          stream,
          fa_seed_,
          fa_offset_,
          src_mask ? src_mask->data() : nullptr,
          nonbatched_bias ? nonbatched_bias->data() : nullptr,
          src_mask ? temp_mask_dim.data() : nullptr,
          nonbatched_bias ? temp_bias_dim.data() : nullptr);
      PADDLE_ENFORCE_EQ(
          succ,
          true,
          common::errors::External(phi::dynload::flash_attn_error()));
      WaitWithDebugInfo(dev_ctx_);
    }
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "FlashAttention is unsupported, please set use_flash_attn to false."));
#endif
  }

  void ComputeBackward(const phi::DenseTensor* qkv_transpose_out,
                       const phi::DenseTensor* src_mask,
                       const phi::DenseTensor* nonbatched_bias,
                       const phi::DenseTensor* softmax_lse,
                       const phi::DenseTensor* fmha_out,
                       const phi::DenseTensor* fmha_out_grad,
                       phi::DenseTensor* src_mask_grad,
                       phi::DenseTensor* nonbatched_bias_grad,
                       GateAttentionGradConfig<T>* config) {
#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
    bool is_bf16 =
        qkv_transpose_out->dtype() == DataType::BFLOAT16 ? true : false;
    TypeDebugInfo<T>();

    PADDLE_ENFORCE_NOT_NULL(
        qkv_transpose_out,
        common::errors::NotFound("The input qkv_transpose_out can not be"
                                 "nullptr when merge_qkv is true."));

    int64_t q_size = config->GetQuerySize();
    const T* q_ptr = qkv_transpose_out->data<T>();
    const T* k_ptr = q_ptr + q_size;
    const T* v_ptr = k_ptr + q_size;

    phi::DenseTensor qkv_transpose_out_grad;
    qkv_transpose_out_grad.Resize(common::make_ddim({3,
                                                     config->batch_size,
                                                     config->seq_len_m,
                                                     config->seq_len_r,
                                                     config->num_heads,
                                                     config->head_dim}));
    AllocWithDebugInfo<T>(
        dev_ctx_, "qkv_transpose_out_grad", &qkv_transpose_out_grad);

    T* q_grad_ptr = qkv_transpose_out_grad.data<T>();
    T* k_grad_ptr = q_grad_ptr + q_size;
    T* v_grad_ptr = k_grad_ptr + q_size;
    WaitWithDebugInfo(dev_ctx_);

    // 1. flash_attn parameter setting.
    phi::DenseTensor cu_seq_q;
    phi::DenseTensor cu_seq_k;
    InitArgumentsAndSeqTensors(config, &cu_seq_q, &cu_seq_k);
    const int32_t* cu_seq_q_ptr = cu_seq_q.data<int32_t>();
    const int32_t* cu_seq_k_ptr = cu_seq_k.data<int32_t>();

    std::vector<int64_t> temp_mask_dim = GetCompressedDim(src_mask);
    std::vector<int64_t> temp_bias_dim = GetCompressedDim(nonbatched_bias);

    phi::DenseTensor softmax_d;
    softmax_d.Resize(softmax_lse->dims());
    AllocWithDebugInfo<float>(dev_ctx_, "d_softmax_lse", &softmax_d);

    phi::DenseTensor bias_d;
    if (nonbatched_bias) {
      bias_d.Resize(
          {fa_batch_size_, fa_num_heads_, fa_max_seqlen_q_, fa_max_seqlen_k_});
      AllocWithDebugInfo<T>(dev_ctx_, "d_bias", &bias_d);
    }

    if (VLOG_IS_ON(6)) {
      VLOG(6) << TensorDebugString(fmha_out, "fmha_out");
      VLOG(6) << TensorDebugString(fmha_out_grad, "fmha_out_grad");
      VLOG(6) << TensorDebugString(softmax_lse, "softmax_lse");
      VLOG(6) << TensorDebugString(&softmax_d, "softmax_d");
      VLOG(6) << TensorDebugString(nonbatched_bias, "nonbatched_bias");
      VLOG(6) << TensorDebugString(&bias_d, "bias_d");
    }

    // 2. Get workspace size and run the flash-attention kernel.
    uint64_t workspace_size = 0;
    phi::DenseTensor workspace;
    cudaStream_t stream = dev_ctx_.stream();
    for (bool need_calc : {false, true}) {
      // first calling, need_calc=false, set out_ptr to nullptr to calculate
      // workspace size second calling, need_calc=true, run flash-attention
      // kernel.
      const void* out_ptr =
          need_calc ? static_cast<const void*>(fmha_out->data()) : nullptr;
      void* workspace_ptr = nullptr;
      if (need_calc) {
        VLOG(6) << "Step 2: Call the flash-attention kernel";
        if (workspace_size > 0) {
          workspace = CreateWorkspace(workspace_size);
          workspace_ptr = static_cast<void*>(workspace.data());
        }
      } else {
        VLOG(6) << "Step 1: Calculate the workspace_size";
      }

      bool succ = phi::dynload::flash_attn_bwd_with_bias_and_mask(
          static_cast<const void*>(q_ptr),
          static_cast<const void*>(k_ptr),
          static_cast<const void*>(v_ptr),
          static_cast<void*>(q_grad_ptr),
          static_cast<void*>(k_grad_ptr),
          static_cast<void*>(v_grad_ptr),
          out_ptr,  // set out to nullptr to calculate workspace size
          static_cast<const void*>(fmha_out_grad->data()),
          cu_seq_q_ptr,
          cu_seq_k_ptr,
          fa_total_q_,
          fa_total_k_,
          fa_batch_size_,
          fa_num_heads_,
          fa_head_size_,
          fa_max_seqlen_q_,
          fa_max_seqlen_k_,
          fa_dropout_prob_,
          fa_softmax_scale_,
          fa_zero_tensors_,
          is_bf16,
          fa_num_splits_,
          softmax_lse->data(),
          softmax_d.data(),
          nonbatched_bias ? bias_d.data() : nullptr,
          workspace_ptr,
          &workspace_size,
          stream,
          fa_seed_,
          fa_offset_,
          src_mask ? src_mask->data() : nullptr,
          nonbatched_bias ? nonbatched_bias->data() : nullptr,
          src_mask ? temp_mask_dim.data() : nullptr,
          nonbatched_bias ? temp_bias_dim.data() : nullptr);
      PADDLE_ENFORCE_EQ(
          succ,
          true,
          common::errors::External(phi::dynload::flash_attn_error()));
      WaitWithDebugInfo(dev_ctx_);
    }

    if (nonbatched_bias) {
      // compare block reduce
      auto dbias_first_dim = bias_d.numel() / nonbatched_bias->numel();
      bias_d.Resize({dbias_first_dim,
                     temp_bias_dim[0],
                     temp_bias_dim[1],
                     temp_bias_dim[2],
                     temp_bias_dim[3]});
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx_,
          bias_d,
          nonbatched_bias_grad,
          kps::IdentityFunctor<T>(),
          {0});
    }

    // 3. Scale Q's grad: q_grad_ptr = alpha * q_grad_ptr
    ComputeScaleQ(q_size, config->head_dim, q_grad_ptr);

    // 4. Compute the grad of qkv_out.
    phi::DenseTensor* qkv_out_grad = config->GetQKVOutGrad();
    ComputeQKVTransposeBackward(qkv_transpose_out_grad, qkv_out_grad);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "FlashAttention is unsupported, please set use_flash_attn to false."));
#endif
  }

 private:
  std::vector<int64_t> GetCompressedDim(const phi::DenseTensor* tensor) {
    std::vector<int64_t> compressed_dims;
    if (tensor) {
      int64_t first_dim = 1;
      const auto& origin_dims = tensor->dims();
      auto rank = origin_dims.size();
      for (int i = 0; i < rank - 3; ++i) {
        first_dim *= origin_dims[i];
      }
      compressed_dims = {first_dim,
                         origin_dims[rank - 3],
                         origin_dims[rank - 2],
                         origin_dims[rank - 1]};
    }
    return compressed_dims;
  }

  phi::DenseTensor CreateWorkspace(uint64_t workspace_size) {
    phi::DenseTensor workspace;
    if (workspace_size > 0) {
      workspace = phi::Empty<float, phi::GPUContext>(
          dev_ctx_, {int64_t(workspace_size / sizeof(float))});
    }
    VLOG(5) << "Allocate workspace: workspace_size=" << workspace_size;
    return workspace;
  }

  void GenerateSeedAndOffset(int64_t batch_size, int64_t num_heads) {
    auto gen = dev_ctx_.GetGenerator();
    uint64_t inc = batch_size * num_heads * 32;
    auto seed_offset_pair = gen->IncrementOffset(inc);
    fa_seed_ = seed_offset_pair.first;
    fa_offset_ = seed_offset_pair.second;
  }

  void InitArgumentsAndSeqTensors(GateAttentionConfig<T>* config,
                                  phi::DenseTensor* cu_seq_q,
                                  phi::DenseTensor* cu_seq_k) {
    fa_batch_size_ = static_cast<int>(config->batch_size) *
                     static_cast<int>(config->seq_len_m);
    fa_num_heads_ = static_cast<int>(config->num_heads);  // qkv_dims[2];
    fa_head_size_ = static_cast<int>(config->head_dim);   // qkv_dims[3];
    fa_max_seqlen_q_ = config->seq_len_r;
    fa_max_seqlen_k_ = config->m_size;
    fa_total_q_ = fa_batch_size_ * fa_max_seqlen_q_;
    fa_total_k_ = fa_batch_size_ * fa_max_seqlen_k_;

    // 0 for an internal heuristic, which is optimal
    fa_num_splits_ = 0;
    fa_zero_tensors_ = false;

    fa_softmax_lse_dim_ = ((fa_max_seqlen_q_ + 16 - 1) / 16) * 16;
    fa_softmax_scale_ = 1.0f;
    fa_dropout_prob_ = 0.0f;
    GenerateSeedAndOffset(fa_batch_size_, fa_num_heads_);

    phi::ArangeNullaryKernel<int32_t, phi::GPUContext>(
        dev_ctx_,
        0,
        (fa_batch_size_ + 1) * fa_max_seqlen_q_,
        fa_max_seqlen_q_,
        cu_seq_q);
    phi::ArangeNullaryKernel<int32_t, phi::GPUContext>(
        dev_ctx_,
        0,
        (fa_batch_size_ + 1) * fa_max_seqlen_k_,
        fa_max_seqlen_k_,
        cu_seq_k);

    if (VLOG_IS_ON(6)) {
      VLOG(6) << "fa_batch_size       : " << fa_batch_size_;
      VLOG(6) << "fa_total_q          : " << fa_total_q_;
      VLOG(6) << "fa_total_k          : " << fa_total_k_;
      VLOG(6) << "fa_num_heads        : " << fa_num_heads_;
      VLOG(6) << "fa_head_size        : " << fa_head_size_;
      VLOG(6) << "fa_max_seqlen_q     : " << fa_max_seqlen_q_;
      VLOG(6) << "fa_max_seqlen_k     : " << fa_max_seqlen_k_;
      VLOG(6) << "fa_num_splits       : " << fa_num_splits_;
      VLOG(6) << "fa_softmax_lse_dim  : " << fa_softmax_lse_dim_;
      VLOG(6) << "fa_softmax_scale    : " << fa_softmax_scale_;
      VLOG(6) << "fa_dropout_prob     : " << fa_dropout_prob_;
    }
  }

  // [batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim] ->
  //         [3, batch_size, seq_len_m, seq_len_r, num_heads, head_dim]
  void ComputeQKVTransposeForward(const phi::DenseTensor& qkv_out,
                                  phi::DenseTensor* qkv_transpose_out) {
    std::vector<int> perm = {3, 0, 1, 2, 4, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_out, perm, qkv_transpose_out);
  }

  // [3, batch_size, seq_len_m, seq_len_r, num_heads, head_dim] ->
  //        [batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim]
  void ComputeQKVTransposeBackward(
      const phi::DenseTensor& qkv_transpose_out_grad,
      phi::DenseTensor* qkv_out_grad) {
    std::vector<int> perm = {1, 2, 3, 0, 4, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_transpose_out_grad, perm, qkv_out_grad);
  }

  void ComputeScaleQ(int64_t numel, int64_t head_dim, T* ptr) {
    float scale = static_cast<float>(1.0f / std::sqrt(head_dim));
    VLOG(6) << "[ComputeScaleQ] numel=" << numel << ", scale=" << scale;

    auto gpu_config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx_, numel, 1);
    SimpleScaleKernel<T><<<gpu_config.block_per_grid,
                           gpu_config.thread_per_block,
                           0,
                           dev_ctx_.stream()>>>(numel, scale, ptr);
  }

  const phi::GPUContext& dev_ctx_;
  bool merge_qkv_;

  int fa_batch_size_;
  int fa_total_q_;
  int fa_total_k_;
  int fa_num_heads_;
  int fa_head_size_;
  int fa_max_seqlen_q_;
  int fa_max_seqlen_k_;
  int fa_num_splits_;
  int fa_softmax_lse_dim_;
  float fa_softmax_scale_{1.0f};
  float fa_dropout_prob_{0.0f};
  uint64_t fa_seed_{0};
  uint64_t fa_offset_{0};
  bool fa_zero_tensors_{false};
};

}  // namespace funcs
}  // namespace phi
