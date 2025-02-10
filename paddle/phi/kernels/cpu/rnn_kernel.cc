// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/rnn_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cpu/rnn_functor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/gru_compute.h"
#include "paddle/phi/kernels/funcs/lstm_compute.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
struct Cell {
  virtual ~Cell() = default;
  virtual void operator()(const CPUContext* dev_ctx UNUSED,
                          DenseTensor* input UNUSED,
                          const DenseTensor* weight_hh UNUSED,
                          const DenseTensor* init_h UNUSED,
                          const DenseTensor* init_c UNUSED,
                          DenseTensor* last_h UNUSED,
                          DenseTensor* last_c UNUSED,
                          DenseTensor* last_c_act UNUSED,
                          DenseTensor* output UNUSED,
                          const DenseTensor* bias_hh UNUSED,
                          DenseTensor* weight_hh_gru UNUSED) const {}
};

template <typename T,
          template <typename>
          class EigenActivationFunctor,
          funcs::detail::ActivationType act_type>
struct SimpleRNNCell : Cell<T> {
  void operator()(const CPUContext* dev_ctx,
                  DenseTensor* input,
                  const DenseTensor* weight_hh,
                  const DenseTensor* init_h,
                  const DenseTensor* init_c UNUSED,
                  DenseTensor* last_h UNUSED,
                  DenseTensor* last_c UNUSED,
                  DenseTensor* last_c_act UNUSED,
                  DenseTensor* output UNUSED,
                  const DenseTensor* bias_hh UNUSED,
                  DenseTensor* weight_hh_gru UNUSED) const override {
    auto blas = phi::funcs::GetBlas<CPUContext, T>(*dev_ctx);
    auto mat_dim_a =
        phi::funcs::CreateMatrixDescriptor(init_h->dims(), 0, false);
    auto mat_dim_b =
        phi::funcs::CreateMatrixDescriptor(weight_hh->dims(), 0, true);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    // convert the batch matmul to matmul, this operator could be speed faster
    blas.MatMul(*init_h,
                mat_dim_a,
                *weight_hh,
                mat_dim_b,
                static_cast<T>(1.0),
                input,
                static_cast<T>(1.0));
    auto z = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(input, "Input", "z", "Activation"));
    auto hidden = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(output, "Output", "hidden", "Activation"));

    auto* place = dev_ctx->eigen_device();
    EigenActivationFunctor<T> functor;
    functor(*place, z, hidden);
  }
};

template <typename T>
struct GRUCell : Cell<T> {
  void operator()(const CPUContext* dev_ctx,
                  DenseTensor* input,
                  const DenseTensor* weight_hh,
                  const DenseTensor* init_h,
                  const DenseTensor* init_c UNUSED,
                  DenseTensor* last_h UNUSED,
                  DenseTensor* last_c,
                  DenseTensor* last_c_act UNUSED,
                  DenseTensor* output,
                  const DenseTensor* bias_hh,
                  DenseTensor* weight_hh_gru) const override {
    auto blas = phi::funcs::GetBlas<CPUContext, T>(*dev_ctx);
    auto mat_dim_a =
        phi::funcs::CreateMatrixDescriptor(init_h->dims(), 0, false);
    auto mat_dim_b =
        phi::funcs::CreateMatrixDescriptor(weight_hh_gru->dims(), 0, true);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    // convert the batch matmul to matmul, this operator could be speed faster
    blas.MatMul(*init_h,
                mat_dim_a,
                *weight_hh_gru,
                mat_dim_b,
                static_cast<T>(1.0),
                input,
                static_cast<T>(1.0));
    size_t frame_size = init_h->dims()[2];
    size_t batch_size = init_h->dims()[1];

    phi::funcs::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = weight_hh->data<T>();
    gru_value.state_weight = weight_hh->data<T>() + 2 * frame_size * frame_size;
    gru_value.reset_bias = bias_hh->data<T>() + 2 * frame_size;

    gru_value.gate_value = input->data<T>();
    gru_value.reset_output_value = last_c->data<T>();
    gru_value.output_value = output->data<T>();
    gru_value.prev_out_value = init_h->data<T>();

    auto gate_act = phi::funcs::detail::GetActivationType("sigmoid_v2");
    auto cand_act = phi::funcs::detail::GetActivationType("tanh_v2");

    phi::funcs::GRUUnitFunctorV2<CPUContext, T>::compute(
        *dev_ctx, gru_value, frame_size, batch_size, cand_act, gate_act);
  }
};

template <typename T>
struct LSTMCell : Cell<T> {
  void operator()(const CPUContext* dev_ctx,
                  DenseTensor* input,
                  const DenseTensor* weight_hh,
                  const DenseTensor* init_h,
                  const DenseTensor* init_c,
                  DenseTensor* last_h UNUSED,
                  DenseTensor* last_c,
                  DenseTensor* last_c_act,
                  DenseTensor* output,
                  const DenseTensor* bias_hh UNUSED,
                  DenseTensor* weight_hh_gru UNUSED) const override {
    auto blas = phi::funcs::GetBlas<CPUContext, T>(*dev_ctx);
    auto mat_dim_a =
        phi::funcs::CreateMatrixDescriptor(init_h->dims(), 0, false);
    auto mat_dim_b =
        phi::funcs::CreateMatrixDescriptor(weight_hh->dims(), 0, true);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    // convert the batch matmul to matmul, this operator could be speed faster
    blas.MatMul(*init_h,
                mat_dim_a,
                *weight_hh,
                mat_dim_b,
                static_cast<T>(1.0),
                input,
                static_cast<T>(1.0));

    phi::funcs::LstmMetaValue<T> lstm_value;
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;

    auto gate_act = phi::funcs::detail::GetActivationType("sigmoid_v2");
    auto cell_act = phi::funcs::detail::GetActivationType("tanh_v2");
    auto cand_act = phi::funcs::detail::GetActivationType("tanh_v2");

    size_t frame_size = init_h->dims()[2];
    size_t batch_size = init_h->dims()[1];

    DenseTensor cell_pre_act;
    if (last_c_act == nullptr) { /* is test */
      cell_pre_act.Resize(init_h->dims());
      dev_ctx->Alloc<T>(&cell_pre_act);
      last_c_act = &cell_pre_act;
    }

    lstm_value.prev_state_value = init_c->data<T>();
    lstm_value.gate_value = input->data<T>();
    lstm_value.output_value = output->data<T>();
    lstm_value.state_value = last_c->data<T>();
    lstm_value.state_active_value = last_c_act->data<T>();
    T cell_clip = 0.0;
    phi::funcs::LstmUnitFunctor<CPUContext, T>::compute(*dev_ctx,
                                                        lstm_value,
                                                        frame_size,
                                                        batch_size,
                                                        cell_clip,
                                                        gate_act,
                                                        cell_act,
                                                        cand_act,
                                                        false);
  }
};

template <typename T, typename CellType>
struct Layer {
  explicit Layer(const CellType& cell) : cell_(cell) {}
  virtual ~Layer() = default;
  void preprocess(const CPUContext& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& weight,
                  const DenseTensor& bias_ih,
                  const DenseTensor& bias_hh,
                  const std::string& mode,
                  bool is_test,
                  DenseTensor* cache_input) {
    // create the temp input for the X * W_ih^T + Bias_ih
    const int& hidden_size = weight.dims()[0];  // NOLINT
    cache_input->Resize(
        common::make_ddim({input.dims()[0], input.dims()[1], hidden_size}));
    if (is_test) {
      dev_ctx.Alloc<T>(cache_input);
    }
    auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(input.dims(), 0, false);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(weight.dims(), 0, true);
    // convert the batch matmul to matmul, this operator could be speed faster
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    blas.MatMul(input,
                mat_dim_a,
                weight,
                mat_dim_b,
                static_cast<T>(1.0),
                cache_input,
                static_cast<T>(0));

    auto in =
        EigenMatrix<T>::Reshape(*cache_input, cache_input->dims().size() - 1);
    auto bias_ih_tmp = EigenMatrix<T>::From(
        bias_ih, common::make_ddim({1, bias_ih.dims()[0]}));
    const int row_num = static_cast<int>(common::product(cache_input->dims()) /
                                         cache_input->dims()[2]);
    in = in + bias_ih_tmp.broadcast(Eigen::DSizes<int, 2>(row_num, 1));
    if (is_gru(mode)) {
      // reset_gate update_gate cell_gate = [1, 1, 0]
      DenseTensor bias_hh_tmp = Empty<T>(dev_ctx, {bias_hh.numel()});
      Copy(dev_ctx, bias_hh, CPUPlace(), false, &bias_hh_tmp);
      bias_hh_tmp.Resize({3, bias_hh_tmp.numel() / 3});
      auto bias_hh_tmp_unbind = Unbind(bias_hh_tmp);
      phi::funcs::SetConstant<CPUContext, T> zero;
      zero(dev_ctx, &bias_hh_tmp_unbind[2], static_cast<T>(0.0));

      auto bias_hh_after_mask = EigenMatrix<T>::From(
          bias_hh_tmp, common::make_ddim({1, bias_hh.dims()[0]}));
      in = in + bias_hh_after_mask.broadcast(Eigen::DSizes<int, 2>(row_num, 1));
    } else {
      auto bias_hh_no_mask = EigenMatrix<T>::From(
          bias_hh, common::make_ddim({1, bias_hh.dims()[0]}));
      in = in + bias_hh_no_mask.broadcast(Eigen::DSizes<int, 2>(row_num, 1));
    }
  }

  void postprocess(const CPUContext& dev_ctx,
                   DenseTensor* output,
                   const DenseTensor* init_h,
                   const DenseTensor* init_c,
                   DenseTensor* last_h,
                   DenseTensor* last_c,
                   const DenseTensor& mask_tensor,
                   const std::string& mode) {
    // in the output, if mask flag is 0, we will return the zero data
    auto& place = *dev_ctx.eigen_device();
    auto out = EigenMatrix<T>::Reshape(*output, output->dims().size() - 1);
    auto mask = EigenMatrix<T>::From(
        mask_tensor, common::make_ddim({mask_tensor.dims()[1], 1}));
    auto pre_h = EigenMatrix<T>::Reshape(*init_h, init_h->dims().size() - 1);
    auto curr_h = EigenMatrix<T>::Reshape(*last_h, last_h->dims().size() - 1);
    auto mask_broadcast = mask.broadcast(
        Eigen::DSizes<int, 2>(1, static_cast<int>(output->dims()[2])));
    curr_h.device(place) = out * mask_broadcast + pre_h * (1 - mask_broadcast);
    out.device(place) = out * mask_broadcast;

    if (is_lstm(mode)) {
      auto pre_c = EigenMatrix<T>::Reshape(*init_c, init_c->dims().size() - 1);
      auto curr_c = EigenMatrix<T>::Reshape(*last_c, last_c->dims().size() - 1);
      curr_c.device(place) =
          curr_c * mask_broadcast + pre_c * (1 - mask_broadcast);
    }
  }

  virtual void operator()(const CPUContext& dev_ctx UNUSED,
                          const DenseTensor* input UNUSED,
                          const std::vector<DenseTensor>& vec UNUSED,
                          const std::vector<DenseTensor>& init_h UNUSED,
                          const std::vector<DenseTensor>& init_c UNUSED,
                          const DenseTensor* sequence_length UNUSED,
                          std::vector<DenseTensor> last_h UNUSED,
                          std::vector<DenseTensor> last_c UNUSED,
                          DenseTensor* output UNUSED,
                          const int& layer_idx UNUSED,
                          const int& gate_num UNUSED,
                          DenseTensor* gate_value UNUSED,
                          DenseTensor* cell_value UNUSED,
                          DenseTensor* cell_act_value UNUSED,
                          const std::string& mode UNUSED,
                          bool is_test UNUSED) {}

  void RunTestIter(const CPUContext& dev_ctx,
                   const DenseTensor* input,
                   const std::vector<DenseTensor>& vec,
                   const std::vector<DenseTensor>& init_h,
                   const std::vector<DenseTensor>& init_c,
                   const DenseTensor* sequence_length,
                   std::vector<DenseTensor>* last_h_ptr,
                   std::vector<DenseTensor>* last_c_ptr,
                   DenseTensor* output,
                   int layer_idx,
                   DenseTensor* gate_value,
                   DenseTensor* cell_value UNUSED,
                   DenseTensor* cell_act_value UNUSED,
                   bool is_bidirect,
                   int offset,
                   const std::string& mode) {
    bool is_reverse = false;
    if (is_bidirect) {
      layer_idx = 2 * layer_idx + offset;
      if (offset > 0) {
        is_reverse = true;
      }
    }
    const int time_step = static_cast<int>(input->dims()[0]);
    this->preprocess(dev_ctx,
                     *input,
                     vec[0 + offset * 4],
                     vec[2 + offset * 4],
                     vec[3 + offset * 4],
                     mode,
                     true,
                     gate_value);
    auto input_tensors = Unbind(*gate_value);
    auto output_tensors = Unbind(*output);
    if (is_reverse) {
      std::reverse(input_tensors.begin(), input_tensors.end());
      std::reverse(output_tensors.begin(), output_tensors.end());
    }
    std::vector<DenseTensor> mask_tensor_list;
    // construct the mask matrix for the mask
    bool has_sequence_length = false;
    if (sequence_length != nullptr) {
      has_sequence_length = true;
    }
    DenseTensor mask_matrix;
    int mask_min_length = time_step;
    if (has_sequence_length) {
      mask_matrix.Resize(common::make_ddim({time_step, input->dims()[1]}));

      CreateMaskMatrix<T>(
          dev_ctx, sequence_length, &mask_matrix, is_reverse, &mask_min_length);
      mask_tensor_list = Unbind(mask_matrix);
    }
    if (is_reverse) {
      mask_min_length = mask_min_length - time_step + 1;
    }
    bool has_allocate_mem_c = false;
    bool has_use_last_h_holder = false;
    const int& reverse_flag = is_reverse ? -1 : 1;

    // define the init_h holder for the swap
    DenseTensor init_h_temp;
    Copy(dev_ctx, *&init_h[layer_idx], dev_ctx.GetPlace(), false, &init_h_temp);

    DenseTensor* init_h_holder = &init_h_temp;
    DenseTensor* last_h_holder = nullptr;
    if (0 < mask_min_length) {
      last_h_holder = &(output_tensors[0]);
    } else {
      last_h_holder = &(*last_h_ptr)[layer_idx];
      has_use_last_h_holder = true;
    }

    DenseTensor* init_c_holder = nullptr;
    const DenseTensor* init_c_temp_holder = nullptr;
    DenseTensor init_c_temp;
    DenseTensor* last_c_holder = nullptr;
    DenseTensor last_c_temp;

    if (is_lstm(mode)) {
      last_c_holder = &(*last_c_ptr)[layer_idx];
      init_c_temp_holder = &init_c[layer_idx];
    } else if (is_gru(mode)) {
      // for reset output value
      last_c_temp.Resize(init_h[layer_idx].dims());
      dev_ctx.Alloc<T>(&last_c_temp);
      last_c_holder = &last_c_temp;
    }

    DenseTensor weight_hh_tmp;  // for gru
    if (is_gru(mode)) {
      weight_hh_tmp.Resize(vec[1 + offset * 4].dims());
      dev_ctx.Alloc<T>(&weight_hh_tmp);
      Copy(dev_ctx,
           vec[1 + offset * 4],
           dev_ctx.GetPlace(),
           false,
           &weight_hh_tmp);
      weight_hh_tmp.Resize({3, weight_hh_tmp.numel() / 3});
      auto weight_hh_tmp_unbind = Unbind(weight_hh_tmp);
      phi::funcs::SetConstant<CPUContext, T> zero;
      zero(dev_ctx, &weight_hh_tmp_unbind[2], static_cast<T>(0.0));
      weight_hh_tmp.Resize(vec[1 + offset * 4].dims());
    }

    for (int i = 0; i < time_step; i++) {
      bool in_mask = (reverse_flag * i) >= mask_min_length;
      if (i > 0) {
        if (!has_allocate_mem_c) {
          if (is_lstm(mode) || is_gru(mode)) {
            init_c_temp.Resize(init_h[layer_idx].dims());
            dev_ctx.Alloc<T>(&init_c_temp);
            init_c_holder = &init_c_temp;
          }
          has_allocate_mem_c = true;
        }
        SwapPointer(&init_c_holder, &last_c_holder);
        init_c_temp_holder = init_c_holder;
      }
      cell_(&dev_ctx,
            &input_tensors[i],
            &vec[1 + offset * 4],
            init_h_holder,
            init_c_temp_holder,
            last_h_holder,
            last_c_holder,
            nullptr,
            &output_tensors[i],
            &vec[3 + offset * 4] /* bias_hh */,
            &weight_hh_tmp);
      if (in_mask) {
        this->postprocess(dev_ctx,
                          &output_tensors[i],
                          init_h_holder,
                          init_c_temp_holder,
                          last_h_holder,
                          last_c_holder,
                          mask_tensor_list[i],
                          mode);
      }
      // prepare next step
      if (i + 1 < time_step) {
        bool next_step_mask = (reverse_flag * (i + 1)) >= mask_min_length;
        if (next_step_mask) {
          if (!has_use_last_h_holder) {
            init_h_holder = &(*last_h_ptr)[layer_idx];
          }
        } else {
          init_h_holder = &(output_tensors[i + 1]);
        }
        SwapPointer(&init_h_holder, &last_h_holder);
      }
    }
    if (has_sequence_length) {
      if (last_h_holder != &(*last_h_ptr)[layer_idx]) {
        Copy(dev_ctx,
             *last_h_holder,
             dev_ctx.GetPlace(),
             false,
             &(*last_h_ptr)[layer_idx]);
      }
    } else {
      Copy(dev_ctx,
           output_tensors[time_step - 1],
           dev_ctx.GetPlace(),
           false,
           &(*last_h_ptr)[layer_idx]);
    }

    if (time_step % 2 == 0) {
      if (is_lstm(mode)) {
        Copy(dev_ctx,
             *last_c_holder,
             dev_ctx.GetPlace(),
             false,
             &(*last_c_ptr)[layer_idx]);
      }
    }
  }

  void RunIter(const CPUContext& dev_ctx,
               const DenseTensor* input,
               const std::vector<DenseTensor>& vec,
               const std::vector<DenseTensor>& init_h,
               const std::vector<DenseTensor>& init_c,
               const DenseTensor* sequence_length,
               std::vector<DenseTensor>* last_h_ptr,
               std::vector<DenseTensor>* last_c_ptr,
               DenseTensor* output,
               int layer_idx,
               DenseTensor* gate_value,
               DenseTensor* cell_value,
               DenseTensor* cell_act_value,
               bool is_bidirect,
               int offset,
               const std::string& mode,
               bool is_test) {
    if (is_test) {
      RunTestIter(dev_ctx,
                  input,
                  vec,
                  init_h,
                  init_c,
                  sequence_length,
                  last_h_ptr,
                  last_c_ptr,
                  output,
                  layer_idx,
                  gate_value,
                  cell_value,
                  cell_act_value,
                  is_bidirect,
                  offset,
                  mode);
      return;
    }
    bool is_reverse = false;
    if (is_bidirect) {
      layer_idx = 2 * layer_idx + offset;
      if (offset > 0) {
        is_reverse = true;
      }
    }
    const int time_step = static_cast<int>(input->dims()[0]);
    this->preprocess(dev_ctx,
                     *input,
                     vec[0 + offset * 4],
                     vec[2 + offset * 4],
                     vec[3 + offset * 4],
                     mode,
                     is_test,
                     gate_value);
    auto input_tensors = Unbind(*gate_value);
    auto output_tensors = Unbind(*output);
    if (is_reverse) {
      std::reverse(input_tensors.begin(), input_tensors.end());
      std::reverse(output_tensors.begin(), output_tensors.end());
    }
    std::vector<DenseTensor> mask_tensor_list;
    // construct the mask matrix for the mask
    bool has_sequence_length = false;
    if (sequence_length != nullptr) {
      has_sequence_length = true;
    }
    DenseTensor mask_matrix;
    int mask_min_length = time_step;
    if (has_sequence_length) {
      mask_matrix.Resize(common::make_ddim({time_step, input->dims()[1]}));
      CreateMaskMatrix<T>(
          dev_ctx, sequence_length, &mask_matrix, is_reverse, &mask_min_length);
      mask_tensor_list = Unbind(mask_matrix);
    }
    if (is_reverse) {
      mask_min_length = mask_min_length - time_step + 1;
    }

    // define the init_h holder for the swap
    bool has_use_last_h_holder = false;
    const int& reverse_flag = is_reverse ? -1 : 1;

    std::vector<DenseTensor> cell_value_tensors;
    std::vector<DenseTensor> cell_act_value_tensors;

    DenseTensor init_h_temp;
    Copy(dev_ctx, *&init_h[layer_idx], dev_ctx.GetPlace(), false, &init_h_temp);
    DenseTensor* init_h_holder = &init_h_temp;
    DenseTensor* last_h_holder = nullptr;
    if (0 < mask_min_length) {
      last_h_holder = &(output_tensors[0]);
    } else {
      last_h_holder = &(*last_h_ptr)[layer_idx];
      has_use_last_h_holder = true;
    }

    const DenseTensor* init_c_holder = nullptr;
    DenseTensor* last_c_holder = nullptr;
    DenseTensor* last_c_act_holder = nullptr;
    if (is_lstm(mode) || is_gru(mode)) {
      cell_value->Resize({time_step, cell_value->numel() / time_step});
      cell_value_tensors = Unbind(*cell_value);
      if (is_lstm(mode)) {
        cell_act_value->Resize(
            {time_step, cell_act_value->numel() / time_step});
        cell_act_value_tensors = Unbind(*cell_act_value);
      }
    }
    DenseTensor weight_hh_tmp;  // for gru
    if (is_gru(mode)) {
      weight_hh_tmp.Resize(vec[1 + offset * 4].dims());
      dev_ctx.Alloc<T>(&weight_hh_tmp);
      Copy(dev_ctx,
           vec[1 + offset * 4],
           dev_ctx.GetPlace(),
           false,
           &weight_hh_tmp);
      weight_hh_tmp.Resize({3, weight_hh_tmp.numel() / 3});
      auto weight_hh_tmp_unbind = Unbind(weight_hh_tmp);
      phi::funcs::SetConstant<CPUContext, T> zero;
      zero(dev_ctx, &weight_hh_tmp_unbind[2], static_cast<T>(0.0));
      weight_hh_tmp.Resize(vec[1 + offset * 4].dims());
    }
    for (int i = 0; i < time_step; i++) {
      bool in_mask = (reverse_flag * i) >= mask_min_length;
      if (is_lstm(mode)) {
        if (i == 0) {
          init_c_holder = &init_c[layer_idx];
        } else {
          init_c_holder = &cell_value_tensors[i - 1];
        }
        cell_value_tensors[i].Resize(init_c[layer_idx].dims());
        cell_act_value_tensors[i].Resize(init_c[layer_idx].dims());
        last_c_holder = &cell_value_tensors[i];
        last_c_act_holder = &cell_act_value_tensors[i];
      } else if (is_gru(mode)) {
        cell_value_tensors[i].Resize(init_h[layer_idx].dims());
        last_c_holder = &cell_value_tensors[i];
      }

      cell_(&dev_ctx,
            &input_tensors[i],
            &vec[1 + offset * 4],
            init_h_holder,
            init_c_holder,
            last_h_holder,
            last_c_holder,
            last_c_act_holder,
            &output_tensors[i],
            &vec[3 + offset * 4] /* bias_hh */,
            &weight_hh_tmp);
      if (in_mask) {
        this->postprocess(dev_ctx,
                          &output_tensors[i],
                          init_h_holder,
                          init_c_holder,
                          last_h_holder,
                          last_c_holder,
                          mask_tensor_list[i],
                          mode);
      }
      // prepare next step
      if (i + 1 < time_step) {
        bool next_step_mask = (reverse_flag * (i + 1)) >= mask_min_length;
        if (next_step_mask) {
          if (!has_use_last_h_holder) {
            init_h_holder = &(*last_h_ptr)[layer_idx];
          }
        } else {
          init_h_holder = &(output_tensors[i + 1]);
        }
        SwapPointer(&init_h_holder, &last_h_holder);
      }
    }
    if (has_sequence_length) {
      if (last_h_holder != &(*last_h_ptr)[layer_idx]) {
        Copy(dev_ctx,
             *last_h_holder,
             dev_ctx.GetPlace(),
             false,
             &(*last_h_ptr)[layer_idx]);
      }
    } else {
      Copy(dev_ctx,
           output_tensors[time_step - 1],
           dev_ctx.GetPlace(),
           false,
           &(*last_h_ptr)[layer_idx]);
    }
    if (is_lstm(mode)) {
      Copy(dev_ctx,
           cell_value_tensors[time_step - 1],
           dev_ctx.GetPlace(),
           false,
           &(*last_c_ptr)[layer_idx]);
    }
  }
  // Cell for the rnn module
  CellType cell_;
};

template <typename T, typename CellType>
struct SingleLayer : public Layer<T, CellType> {
  explicit SingleLayer(const CellType& cell) : Layer<T, CellType>(cell) {}
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor* input,
                  const std::vector<DenseTensor>& vec,
                  const std::vector<DenseTensor>& init_h,
                  const std::vector<DenseTensor>& init_c,
                  const DenseTensor* sequence_length,
                  std::vector<DenseTensor> last_h,
                  std::vector<DenseTensor> last_c,
                  DenseTensor* output,
                  const int& layer_idx,
                  const int& gate_num UNUSED,
                  DenseTensor* gate_value,
                  DenseTensor* cell_value,
                  DenseTensor* cell_act_value,
                  const std::string& mode,
                  bool is_test) override {
    this->RunIter(dev_ctx,
                  input,
                  vec,
                  init_h,
                  init_c,
                  sequence_length,
                  &last_h,
                  &last_c,
                  output,
                  layer_idx,
                  gate_value,
                  cell_value,
                  cell_act_value,
                  false,
                  0,
                  mode,
                  is_test);
  }
};

template <typename T, typename CellType>
struct BidirLayer : public Layer<T, CellType> {
  explicit BidirLayer(const CellType& cell) : Layer<T, CellType>(cell) {}
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor* input,
                  const std::vector<DenseTensor>& vec,
                  const std::vector<DenseTensor>& init_h,
                  const std::vector<DenseTensor>& init_c,
                  const DenseTensor* sequence_length,
                  std::vector<DenseTensor> last_h,
                  std::vector<DenseTensor> last_c,
                  DenseTensor* output,
                  const int& layer_idx,
                  const int& gate_num UNUSED,
                  DenseTensor* gate_value,
                  DenseTensor* cell_value,
                  DenseTensor* cell_act_value,
                  const std::string& mode,
                  bool is_test) override {
    std::vector<DenseTensor> output_vec(2);
    DenseTensor forward_input_w, forward_cell_value, forward_cell_act_value;
    DenseTensor backward_input_w, backward_cell_value, backward_cell_act_value;
    int time_step = static_cast<int>(input->dims()[0]);
    int batch_size = static_cast<int>(input->dims()[1]);
    int hidden_size = static_cast<int>(output->dims()[2]);
    for (int i = 0; i < 2; ++i) {
      output_vec[i].Resize({time_step, batch_size, hidden_size / 2});
      dev_ctx.Alloc<T>(&output_vec[i]);
    }
    if (!is_test) {
      gate_value->Resize({2, gate_value->numel() / 2});
      forward_input_w = gate_value->Slice(0, 1);
      backward_input_w = gate_value->Slice(1, 2);

      if (is_lstm(mode) || is_gru(mode)) /* for lstm and gru */ {
        cell_value->Resize({2, cell_value->numel() / 2});
        cell_act_value->Resize({2, cell_act_value->numel() / 2});
        forward_cell_value = cell_value->Slice(0, 1);
        backward_cell_value = cell_value->Slice(1, 2);
        if (is_lstm(mode)) {
          forward_cell_act_value = cell_act_value->Slice(0, 1);
          backward_cell_act_value = cell_act_value->Slice(1, 2);
        }
      }
    }

    this->RunIter(dev_ctx,
                  input,
                  vec,
                  init_h,
                  init_c,
                  sequence_length,
                  &last_h,
                  &last_c,
                  &output_vec[0],
                  layer_idx,
                  &forward_input_w,
                  &forward_cell_value,
                  &forward_cell_act_value,
                  true,
                  0,
                  mode,
                  is_test);

    this->RunIter(dev_ctx,
                  input,
                  vec,
                  init_h,
                  init_c,
                  sequence_length,
                  &last_h,
                  &last_c,
                  &output_vec[1],
                  layer_idx,
                  &backward_input_w,
                  &backward_cell_value,
                  &backward_cell_act_value,
                  true,
                  1,
                  mode,
                  is_test);

    // concat the output result
    funcs::ConcatFunctor<CPUContext, T> concat_functor;
    concat_functor(dev_ctx, output_vec, static_cast<int>(2), output);
  }
};

template <typename T, typename Context>
void RnnKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<const DenseTensor*>& pre_state,
               const std::vector<const DenseTensor*>& weight_list,
               const paddle::optional<DenseTensor>& sequence_length,
               float dropout_prob,
               bool is_bidirec,
               int input_size,
               int hidden_size,
               int num_layers,
               const std::string& mode,
               int seed,
               bool is_test,
               DenseTensor* out,
               DenseTensor* dropout_state,
               std::vector<DenseTensor*> state,
               DenseTensor* reserve) {
  if (!is_test) {
    if (dropout_state->IsInitialized()) {
      if (dropout_state->numel() != out->numel()) dropout_state->clear();
    }
    const auto& out_dim = out->dims();
    Full<uint8_t>(dev_ctx, {out_dim.Get(), out_dim.size()}, 1, dropout_state);
  }

  // init the output and allocate the memory
  dev_ctx.template Alloc<T>(out);
  int gate_num = 4;
  dev_ctx.template Alloc<T>(state[0]);
  if (is_lstm(mode)) {
    dev_ctx.template Alloc<T>(state[1]);
    RnnFunc<LSTMCell<T>, Layer, SingleLayer, BidirLayer, T>(
        dev_ctx,
        &x,
        weight_list,
        pre_state[0],
        pre_state[1],
        sequence_length.get_ptr(),
        state[0],
        state[1],
        out,
        dropout_state,
        num_layers,
        gate_num,
        input_size,
        hidden_size,
        is_bidirec,
        mode,
        dropout_prob,
        is_test,
        seed,
        reserve);
  } else if (is_rnn_relu(mode)) {  // NOLINT
    gate_num = 1;
    RnnFunc<SimpleRNNCell<T,
                          funcs::ReluCPUFunctor,
                          phi::funcs::detail::ActivationType::kReLU>,
            Layer,
            SingleLayer,
            BidirLayer,
            T>(dev_ctx,
               &x,
               weight_list,
               pre_state[0],
               nullptr,
               sequence_length.get_ptr(),
               state[0],
               nullptr,
               out,
               dropout_state,
               num_layers,
               gate_num,
               input_size,
               hidden_size,
               is_bidirec,
               mode,
               dropout_prob,
               is_test,
               seed,
               reserve);
  } else if (is_rnn_tanh(mode)) {
    gate_num = 1;
    RnnFunc<SimpleRNNCell<T,
                          funcs::TanhFunctor,
                          phi::funcs::detail::ActivationType::kTanhV2>,
            Layer,
            SingleLayer,
            BidirLayer,
            T>(dev_ctx,
               &x,
               weight_list,
               pre_state[0],
               nullptr,
               sequence_length.get_ptr(),
               state[0],
               nullptr,
               out,
               dropout_state,
               num_layers,
               gate_num,
               input_size,
               hidden_size,
               is_bidirec,
               mode,
               dropout_prob,
               is_test,
               seed,
               reserve);
  } else if (is_gru(mode)) {
    gate_num = 3;
    RnnFunc<GRUCell<T>, Layer, SingleLayer, BidirLayer, T>(
        dev_ctx,
        &x,
        weight_list,
        pre_state[0],
        nullptr,
        sequence_length.get_ptr(),
        state[0],
        nullptr,
        out,
        dropout_state,
        num_layers,
        gate_num,
        input_size,
        hidden_size,
        is_bidirec,
        mode,
        dropout_prob,
        is_test,
        seed,
        reserve);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rnn, CPU, ALL_LAYOUT, phi::RnnKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
