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
#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/arg_min_max_op_base.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/fluid/operators/utils.h"

#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

#define CREATE_TENSOR(tensor, dtype, ...)             \
  LoDTensor tensor;                                   \
  tensor.Resize(framework::make_ddim({__VA_ARGS__})); \
  tensor.mutable_data<dtype>(ctx.GetPlace())

#define ELEMENT_BINARY_OP(lhs, rhs, output, functor_type, dtype)            \
  ElementwiseComputeEx<functor_type##Functor<dtype>, DeviceContext, dtype>( \
      ctx, &lhs, &rhs, -1, functor_type##Functor<dtype>(), &output)

#define ADD(lhs, rhs, output, dtype) \
  ELEMENT_BINARY_OP(lhs, rhs, output, Add, dtype)

#define SUB(lhs, rhs, output, dtype) \
  ELEMENT_BINARY_OP(lhs, rhs, output, Sub, dtype)

#define MUL(lhs, rhs, output, dtype) \
  ELEMENT_BINARY_OP(lhs, rhs, output, Mul, dtype)

template <typename T, typename IndType>
struct CPUArgmax {
  void operator()(const Tensor& input, Tensor* out_idx, Tensor* out, int axis) {
    framework::DDim input_dims = input.dims();
    int64_t pre = 1;
    int64_t post = 1;
    int64_t n = input_dims[axis];
    for (int i = 0; i < axis; i++) {
      pre *= input_dims[i];
    }

    for (int i = axis + 1; i < input_dims.size(); i++) {
      post *= input_dims[i];
    }
    int64_t height = pre * post;
    int64_t width = n;
    const T* in_data = input.data<T>();
    IndType* out_idx_data = out_idx->data<IndType>();
    T* out_data = out->data<T>();
// Reduce
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < height; ++i) {
      int64_t h = i / post;
      int64_t w = i % post;
      IndType max_idx = -1;
      T max_value = std::numeric_limits<T>::lowest();
      for (int64_t j = 0; j < width; ++j) {
        if (in_data[h * width * post + j * post + w] > max_value) {
          max_value = in_data[h * width * post + j * post + w];
          max_idx = j;
        }
      }
      out_data[i] = max_value;
      out_idx_data[i] = max_idx;
    }
  }
};

template <typename T, typename Functor>
void SameDimsBinaryOP(const Tensor& lhs, const Tensor& rhs, Tensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  T* out_ptr = out->data<T>();
  auto nums = out->numel();
  Functor functor;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < nums; ++i) {
    out_ptr[i] = functor(lhs_ptr[i], rhs_ptr[i]);
  }
}

// Need to gurantee that lhs, rhs have same dims.
#define SAME_DIMS_ELEMENT_BINARY_OP(lhs, rhs, output, functor_type, dtype) \
  SameDimsBinaryOP<dtype, functor_type##Functor<dtype>>(lhs, rhs, &output)

template <typename T>
void GetStrides(const std::vector<T>& dims, std::vector<T>* strides) {
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    (*strides)[i] = (*strides)[i + 1] * dims[i + 1];
  }
}

inline uint32_t GetOutputIndex(const uint32_t* x_dims_array, const int max_dim,
                               const uint32_t* index_array) {
  int index_ = 0;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] > 1) {
      index_ = index_ * x_dims_array[i] + index_array[i];
    }
  }
  return index_;
}

inline void UpdateOutputIndexArray(const uint32_t* out_dims_array,
                                   const int max_dim, uint32_t* index_array) {
  for (int i = max_dim - 1; i >= 0; --i) {
    ++index_array[i];
    if (index_array[i] >= out_dims_array[i]) {
      index_array[i] -= out_dims_array[i];
    } else {
      break;
    }
  }
}
// Need to gurantee that lhs, rhs have same dim size.
template <typename T, typename Functor>
void SimpleBroadcastBinaryOP(const Tensor& lhs, const Tensor& rhs,
                             Tensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  T* out_ptr = out->data<T>();
  uint32_t nums = static_cast<uint32_t>(out->numel());
  uint32_t out_dims_size = static_cast<uint32_t>(out->dims().size());
  std::vector<uint32_t> output_dims(out_dims_size);
  std::vector<uint32_t> lhs_dims(out_dims_size);
  std::vector<uint32_t> rhs_dims(out_dims_size);
  std::copy(lhs.dims().Get(), lhs.dims().Get() + out_dims_size,
            lhs_dims.data());
  std::copy(rhs.dims().Get(), rhs.dims().Get() + out_dims_size,
            rhs_dims.data());
  std::copy(out->dims().Get(), out->dims().Get() + out_dims_size,
            output_dims.data());

  std::vector<uint32_t> output_strides(out_dims_size, 1);
  std::vector<uint32_t> lhs_strides(out_dims_size, 1);
  std::vector<uint32_t> rhs_strides(out_dims_size, 1);
  std::vector<uint32_t> index_array(out_dims_size, 0);
  GetStrides(output_dims, &output_strides);
  GetStrides(lhs_dims, &lhs_strides);
  GetStrides(rhs_dims, &rhs_strides);

  Functor functor;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (uint32_t i = 0; i < nums; ++i) {
    uint32_t output_idx = i;
    uint32_t lhs_idx = 0;
    uint32_t rhs_idx = 0;
    for (uint32_t j = 0; j < out_dims_size; ++j) {
      uint32_t curr_idx = output_idx / output_strides[j];
      output_idx %= output_strides[j];
      lhs_idx += (lhs_dims[j] > 1) ? curr_idx * lhs_strides[j] : 0;
      rhs_idx += (rhs_dims[j] > 1) ? curr_idx * rhs_strides[j] : 0;
    }
    // uint32_t lhs_idx = GetOutputIndex(lhs_dims.data(), out_dims_size,
    // index_array.data());
    // uint32_t rhs_idx = GetOutputIndex(rhs_dims.data(), out_dims_size,
    // index_array.data());
    // UpdateOutputIndexArray(output_dims.data(), out_dims_size,
    // index_array.data());
    out_ptr[i] = functor(lhs_ptr[lhs_idx], rhs_ptr[rhs_idx]);
  }
}

template <typename T, typename IndexT = int>
void CPUGather(const platform::DeviceContext& ctx, const Tensor& src,
               const Tensor& index, Tensor* output) {
  int64_t index_size = index.dims()[0];
  auto src_dims = src.dims();
  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  // slice size
  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];
  // input size
  const size_t slice_bytes = slice_size * sizeof(T);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < index_size; ++i) {
    IndexT index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

template <typename T>
void ARange(T* in, int64_t num, const T& scale) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < num; ++i) {
    in[i] = i * scale;
  }
}

class TensorBuffer {
 public:
  explicit TensorBuffer(const LoDTensor& in) : buffer_(in), offset_(0) {
    buffer_.Resize({buffer_.numel()});
  }
  Tensor GetBufferBlock(std::initializer_list<int64_t> shape) {
    int64_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int64_t>());
    Tensor block = buffer_.Slice(offset_, offset_ + size);
    offset_ += size;
    block.Resize(shape);
    return block;
  }

 private:
  LoDTensor buffer_;  // need to resize 1-D Tensor
  int offset_;
};

template <typename DeviceContext, typename T>
class ViterbiDecodeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool with_start_stop_tag = ctx.Attr<bool>("with_start_stop_tag");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto curr_place = ctx.GetPlace();

    auto* input = ctx.Input<Tensor>("Input");
    auto batch_size = static_cast<int>(input->dims()[0]);
    auto seq_len = static_cast<int>(input->dims()[1]);
    auto n_labels = static_cast<int>(input->dims()[2]);

    // Create a large int data buffer
    int buffer_size = batch_size * seq_len + batch_size * n_labels * seq_len +
                      7 * batch_size + 2;
    CREATE_TENSOR(int_buffer, int64_t, buffer_size);
    TensorBuffer int_tensor_buffer(int_buffer);

    // Create a large float data buffer
    buffer_size = seq_len * batch_size * n_labels + 5 * batch_size * n_labels +
                  2 * n_labels * n_labels + batch_size * n_labels * n_labels +
                  3 * batch_size;
    CREATE_TENSOR(float_buffer, T, buffer_size);
    TensorBuffer float_tensor_buffer(float_buffer);

    auto* length = ctx.Input<Tensor>("Length");
    Tensor left_length = int_tensor_buffer.GetBufferBlock({batch_size, 1});
    framework::TensorCopy(*length, curr_place, dev_ctx, &left_length);

    int64_t max_seq_len =
        *std::max_element(left_length.data<int64_t>(),
                          left_length.data<int64_t>() + left_length.numel());

    auto* scores = ctx.Output<Tensor>("Scores");
    scores->mutable_data<T>(curr_place);

    auto* path = ctx.Output<Tensor>("Path");
    path->Resize({batch_size, max_seq_len});
    path->mutable_data<int64_t>(curr_place);

    Tensor temp_path =
        int_tensor_buffer.GetBufferBlock({max_seq_len, batch_size});
    auto batch_path = Unbind(temp_path);
    for (auto it = batch_path.begin(); it != batch_path.end(); ++it) {
      it->Resize({batch_size});
    }

    Tensor inputs_t_exp =
        float_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
    std::vector<int> axis{1, 0, 2};
    TransCompute<DeviceContext, T>(axis.size(), dev_ctx, *input, &inputs_t_exp,
                                   axis);

    auto* transition = ctx.Input<Tensor>("Transition");
    Tensor trans_exp = float_tensor_buffer.GetBufferBlock({n_labels, n_labels});
    framework::TensorCopy(*transition, curr_place, dev_ctx, &trans_exp);
    trans_exp.Resize({1, n_labels, n_labels});

    Tensor alpha = float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    math::SetConstant<DeviceContext, T> float_functor;
    math::SetConstant<DeviceContext, int64_t> int_functor;

    std::vector<Tensor> historys;
    Tensor zero = int_tensor_buffer.GetBufferBlock({1});
    int_functor(dev_ctx, &zero, 0);
    Tensor one = int_tensor_buffer.GetBufferBlock({1});
    int_functor(dev_ctx, &one, 1);
    Tensor float_one = float_tensor_buffer.GetBufferBlock({batch_size, 1});
    float_functor(dev_ctx, &float_one, static_cast<T>(1.0));
    Tensor alpha_trn_sum =
        float_tensor_buffer.GetBufferBlock({batch_size, n_labels, n_labels});
    Tensor alpha_max =
        float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    Tensor alpha_argmax =
        int_tensor_buffer.GetBufferBlock({seq_len, batch_size, n_labels});
    auto alpha_argmax_unbind = Unbind(alpha_argmax);
    Tensor alpha_nxt =
        float_tensor_buffer.GetBufferBlock({batch_size, n_labels});
    Tensor int_mask = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor float_mask = float_tensor_buffer.GetBufferBlock({batch_size, 1});
    Tensor stop_trans_exp =
        float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
    Tensor start_trans_exp =
        float_tensor_buffer.GetBufferBlock({1, 1, n_labels});
    Tensor rest_trans_exp =
        float_tensor_buffer.GetBufferBlock({1, n_labels - 2, n_labels});
    Tensor last_ids = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor batch_offset = int_tensor_buffer.GetBufferBlock({batch_size});
    Tensor gather_idx = int_tensor_buffer.GetBufferBlock({batch_size});

    std::vector<const Tensor*> shape_refer{&rest_trans_exp, &stop_trans_exp,
                                           &start_trans_exp};
    std::vector<Tensor*> outputs{&rest_trans_exp, &stop_trans_exp,
                                 &start_trans_exp};
    math::SplitFunctor<DeviceContext, T> split_functor;
    split_functor(dev_ctx, trans_exp, shape_refer, 1, &outputs);
    stop_trans_exp.Resize({1, n_labels});
    start_trans_exp.Resize({1, n_labels});
    auto logit0 = inputs_t_exp.Slice(0, 1);
    logit0.Resize({batch_size, n_labels});
    if (with_start_stop_tag) {
      ADD(logit0, start_trans_exp, alpha, T);
      ElementwiseComputeEx<EqualFunctor<T>, DeviceContext, int64_t, T>(
          ctx, &left_length, &one, -1, EqualFunctor<T>(), &float_mask);
      MUL(stop_trans_exp, float_mask, alpha_nxt, T);
      SAME_DIMS_ELEMENT_BINARY_OP(alpha, alpha_nxt, alpha, Add, T);
    } else {
      alpha = logit0;
    }
    SUB(left_length, one, left_length, int64_t);
    CPUArgmax<T, int64_t> argmax;
    for (int64_t i = 1; i < max_seq_len; ++i) {
      Tensor logit = inputs_t_exp.Slice(i, i + 1);
      logit.Resize({batch_size, n_labels});
      Tensor& alpha_exp = alpha.Resize({batch_size, n_labels, 1});
      // ADD(alpha_exp, trans_exp, alpha_trn_sum, T);
      SimpleBroadcastBinaryOP<T, AddFunctor<T>>(alpha_exp, trans_exp,
                                                &alpha_trn_sum);
      auto alpha_argmax_temp = alpha_argmax_unbind[i - 1];
      alpha_argmax_temp.Resize({batch_size, n_labels});

      argmax(alpha_trn_sum, &alpha_argmax_temp, &alpha_max, 1);
      historys.push_back(alpha_argmax_temp);

      SAME_DIMS_ELEMENT_BINARY_OP(alpha_max, logit, alpha_nxt, Add, T);

      alpha.Resize({batch_size, n_labels});

      // mask = paddle.cast((left_length > 0), dtype='float32')
      // alpha = mask * alpha_nxt + (1 - mask) * alpha
      ElementwiseComputeEx<GreaterThanFunctor<T>, DeviceContext, int64_t, T>(
          ctx, &left_length, &zero, -1, GreaterThanFunctor<T>(), &float_mask);
      // alpha_nxt = mask * alpha_nxt
      // MUL(alpha_nxt, float_mask, alpha_nxt, T);
      SimpleBroadcastBinaryOP<T, MulFunctor<T>>(alpha_nxt, float_mask,
                                                &alpha_nxt);
      // inv_mask = 1 - mask
      SAME_DIMS_ELEMENT_BINARY_OP(float_one, float_mask, float_mask, Sub, T);
      // alpha = (1 - mask) * alpha
      // MUL(alpha, float_mask, alpha, T);
      SimpleBroadcastBinaryOP<T, MulFunctor<T>>(alpha, float_mask, &alpha);
      // alpha += alpha_nxt
      SAME_DIMS_ELEMENT_BINARY_OP(alpha, alpha_nxt, alpha, Add, T);
      if (with_start_stop_tag) {  // cost 10% time
        ElementwiseComputeEx<EqualFunctor<T>, DeviceContext, int64_t, T>(
            ctx, &left_length, &one, -1, EqualFunctor<T>(), &float_mask);
        // trans_exp: [1, n, n]
        // alpha += mask * trans_exp[:, self.stop_idx]
        MUL(stop_trans_exp, float_mask, alpha_nxt, T);
        SimpleBroadcastBinaryOP<T, MulFunctor<T>>(stop_trans_exp, float_mask,
                                                  &alpha_nxt);
        SAME_DIMS_ELEMENT_BINARY_OP(alpha, alpha_nxt, alpha, Add, T);
      }
      // SUB(left_length, one, left_length, int64_t);
      SimpleBroadcastBinaryOP<int64_t, SubFunctor<int64_t>>(left_length, one,
                                                            &left_length);
    }

    // scores, last_ids = alpha.max(1), alpha.argmax(1)
    argmax(alpha, &last_ids, scores, 1);

    // tag_mask = paddle.cast((left_length >= 0), 'int64')
    left_length.Resize({batch_size});
    ElementwiseComputeEx<GreaterEqualFunctor<int64_t>, DeviceContext, int64_t>(
        ctx, &left_length, &zero, -1, GreaterEqualFunctor<int64_t>(),
        &int_mask);

    // last_ids_update = last_ids * tag_mask
    int last_ids_index = 1;
    int actual_len = std::min(seq_len, static_cast<int>(max_seq_len));

    SAME_DIMS_ELEMENT_BINARY_OP(last_ids, int_mask,
                                batch_path[actual_len - last_ids_index], Mul,
                                int64_t);
    ARange(batch_offset.data<int64_t>(), static_cast<int64_t>(batch_size),
           static_cast<int64_t>(n_labels));

    for (auto hist = historys.rbegin(); hist != historys.rend(); ++hist) {
      ++last_ids_index;
      // ADD(left_length, one, left_length, int64_t);
      SimpleBroadcastBinaryOP<int64_t, AddFunctor<int64_t>>(left_length, one,
                                                            &left_length);
      SAME_DIMS_ELEMENT_BINARY_OP(batch_offset, last_ids, gather_idx, Add,
                                  int64_t);
      // tag_mask = paddle.cast((left_length >= 0), 'int64')
      // last_ids_update = paddle.gather(hist.flatten(), gather_idx) * tag_mask
      Tensor& last_ids_update = batch_path[actual_len - last_ids_index];
      hist->Resize({batch_size * n_labels});
      CPUGather<int64_t, int64_t>(dev_ctx, *hist, gather_idx, &last_ids_update);
      ElementwiseComputeEx<GreaterEqualFunctor<int64_t>, DeviceContext,
                           int64_t>(ctx, &left_length, &zero, -1,
                                    GreaterEqualFunctor<int64_t>(), &int_mask);
      SAME_DIMS_ELEMENT_BINARY_OP(last_ids_update, int_mask, last_ids_update,
                                  Mul, int64_t);
      // tag_mask = 1 - tag_mask
      // SUB(one, int_mask, int_mask, int64_t);
      SimpleBroadcastBinaryOP<int64_t, SubFunctor<int64_t>>(one, int_mask,
                                                            &int_mask);
      // last_ids = last_ids_update + last_ids * (1 - tag_mask)
      SAME_DIMS_ELEMENT_BINARY_OP(last_ids, int_mask, last_ids, Mul, int64_t);
      SAME_DIMS_ELEMENT_BINARY_OP(last_ids_update, last_ids, last_ids, Add,
                                  int64_t);
    }
    // transpose batch_path
    axis = {1, 0};
    TransCompute<DeviceContext, int64_t>(axis.size(), dev_ctx, temp_path, path,
                                         axis);
  }
};

}  // namespace operators
}  // namespace paddle
