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

#include "paddle/phi/kernels/rnn_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cpu/rnn_functor.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"
#include "paddle/phi/kernels/funcs/gru_compute.h"
#include "paddle/phi/kernels/funcs/lstm_compute.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
void BackupTensor(const CPUContext& dev_ctx,
                  DenseTensor* dst,
                  DenseTensor* src) {
  dst->Resize(src->dims());  // NOLINT
  dev_ctx.Alloc<T>(dst);
  Copy(dev_ctx, *src, dev_ctx.GetPlace(), false, dst);
}

template <typename T>
void CreateLstmValue(phi::funcs::LstmMetaValue<T>* lstm_value) {
  lstm_value->check_ig = nullptr;
  lstm_value->check_fg = nullptr;
  lstm_value->check_og = nullptr;
}

template <typename T>
void CreateLstmGrad(phi::funcs::LstmMetaGrad<T>* lstm_grad) {
  lstm_grad->check_ig_grad = nullptr;
  lstm_grad->check_fg_grad = nullptr;
  lstm_grad->check_og_grad = nullptr;
}

template <typename T>
struct GradCell {
  virtual ~GradCell() = default;
  virtual void operator()(const CPUContext& dev_ctx UNUSED,
                          DenseTensor* gate_tensor UNUSED,
                          DenseTensor* state_tensor UNUSED,
                          DenseTensor* act_state_tensor UNUSED,
                          DenseTensor* hidden_tensor UNUSED,
                          const DenseTensor* weight_hh UNUSED,
                          DenseTensor* pre_hidden UNUSED,
                          DenseTensor* pre_state UNUSED,
                          DenseTensor* grad_hidden UNUSED,
                          DenseTensor* grad_state UNUSED,
                          DenseTensor* grad_gate UNUSED,
                          DenseTensor* grad_weight_hh UNUSED,
                          DenseTensor* grad_pre_hidden UNUSED,
                          DenseTensor* grad_pre_state UNUSED,
                          DenseTensor* grad_bias_hh UNUSED,
                          const DenseTensor& mask_tensor UNUSED,
                          bool has_sequence_length UNUSED) const {}

  void postprocess_pre_hidden_grad(const CPUContext& dev_ctx,
                                   DenseTensor* grad_pre_hidden,
                                   DenseTensor* grad_pre_hidden_bak,
                                   DenseTensor* grad_pre_state,
                                   DenseTensor* grad_pre_state_bak,
                                   const DenseTensor& mask_tensor,
                                   bool has_sequence_length) const {
    if (has_sequence_length) {
      auto& place = *dev_ctx.eigen_device();
      auto mask = EigenMatrix<T>::From(
          mask_tensor, common::make_ddim({mask_tensor.dims()[1], 1}));
      auto mask_broadcast = mask.broadcast(Eigen::DSizes<int, 2>(
          1, static_cast<int>(grad_pre_hidden->dims()[2])));
      auto pre_hidden_grad = EigenMatrix<T>::Reshape(
          *grad_pre_hidden, grad_pre_hidden->dims().size() - 1);
      auto pre_hidden_bak_grad = EigenMatrix<T>::Reshape(
          *grad_pre_hidden_bak, grad_pre_hidden_bak->dims().size() - 1);
      pre_hidden_grad.device(place) =
          (1 - mask_broadcast) * pre_hidden_bak_grad +
          pre_hidden_grad * mask_broadcast;
      if (grad_pre_state) {
        auto pre_state_grad = EigenMatrix<T>::Reshape(
            *grad_pre_state, grad_pre_state->dims().size() - 1);
        auto pre_state_bak_grad = EigenMatrix<T>::Reshape(
            *grad_pre_state_bak, grad_pre_state_bak->dims().size() - 1);
        pre_state_grad.device(place) =
            (1 - mask_broadcast) * pre_state_bak_grad +
            pre_state_grad * mask_broadcast;
      }
    }
  }

  virtual void update_pre_hidden_grad(const CPUContext& dev_ctx,
                                      DenseTensor* grad_gate,
                                      const DenseTensor* weight_hh,
                                      DenseTensor* grad_pre_hidden,
                                      DenseTensor* grad_pre_hidden_bak,
                                      DenseTensor* grad_pre_state,
                                      DenseTensor* grad_pre_state_bak,
                                      const DenseTensor& mask_tensor,
                                      bool has_sequence_length) const {
    auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);
    DenseTensor* grad_gate_tmp = grad_gate;
    auto mat_dim_a =
        phi::funcs::CreateMatrixDescriptor(grad_gate_tmp->dims(), 0, false);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    auto mat_dim_b =
        phi::funcs::CreateMatrixDescriptor(weight_hh->dims(), 0, false);
    blas.MatMul(*grad_gate_tmp,
                mat_dim_a,
                *weight_hh,
                mat_dim_b,
                static_cast<T>(1.0),
                grad_pre_hidden,
                0);
    postprocess_pre_hidden_grad(dev_ctx,
                                grad_pre_hidden,
                                grad_pre_hidden_bak,
                                grad_pre_state,
                                grad_pre_state_bak,
                                mask_tensor,
                                has_sequence_length);
  }

  virtual void update_weight_hh_grad(const CPUContext& dev_ctx,
                                     DenseTensor* grad_gate,
                                     DenseTensor* pre_hidden,
                                     DenseTensor* grad_weight_hh) const {
    auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);
    auto mat_dim_c =
        phi::funcs::CreateMatrixDescriptor(grad_gate->dims(), 0, true);
    mat_dim_c.height_ *= mat_dim_c.batch_size_;
    mat_dim_c.batch_size_ = 0;
    auto mat_dim_d =
        phi::funcs::CreateMatrixDescriptor(pre_hidden->dims(), 0, false);
    mat_dim_d.height_ *= mat_dim_d.batch_size_;
    mat_dim_d.batch_size_ = 0;
    blas.MatMul(*grad_gate,
                mat_dim_c,
                *pre_hidden,
                mat_dim_d,
                static_cast<T>(1.0),
                grad_weight_hh,
                static_cast<T>(1.0));
  }
};

template <typename T, template <typename> class EigenActivationBackwardFunctor>
struct SimpleRNNGradCell : GradCell<T> {
  void operator()(const CPUContext& dev_ctx,
                  DenseTensor* gate_tensor,
                  DenseTensor* state_tensor UNUSED,
                  DenseTensor* act_state_tensor UNUSED,
                  DenseTensor* hidden_tensor,
                  const DenseTensor* weight_hh,
                  DenseTensor* pre_hidden,
                  DenseTensor* pre_state UNUSED,
                  DenseTensor* grad_hidden,
                  DenseTensor* grad_state UNUSED,
                  DenseTensor* grad_gate,
                  DenseTensor* grad_weight_hh,
                  DenseTensor* grad_pre_hidden,
                  DenseTensor* grad_pre_state UNUSED,
                  DenseTensor* grad_bias_hh UNUSED,
                  const DenseTensor& mask_tensor,
                  bool has_sequence_length) const override {
    DenseTensor grad_pre_hidden_bak;
    if (has_sequence_length) {
      BackupTensor<T>(dev_ctx, &grad_pre_hidden_bak, grad_pre_hidden);
    }
    // h = act(z)
    // update dz
    auto dz = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(grad_gate, "Output", "dz", "Grad"));
    auto dh = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(grad_hidden, "Input", "dh", "Grad"));
    auto h = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(hidden_tensor, "Input", "h", "Value"));
    // useless, but need this argument to execute functor
    auto z = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(gate_tensor, "Input", "z", "Value"));

    auto* place = dev_ctx.eigen_device();
    EigenActivationBackwardFunctor<T> functor;
    functor(*place, z, h, dh, dz);

    // update grad_weight_hh, grad_pre_hidden
    this->update_pre_hidden_grad(dev_ctx,
                                 grad_gate,
                                 weight_hh,
                                 grad_pre_hidden,
                                 &grad_pre_hidden_bak,
                                 nullptr,
                                 nullptr,
                                 mask_tensor,
                                 has_sequence_length);
    this->update_weight_hh_grad(dev_ctx, grad_gate, pre_hidden, grad_weight_hh);
  }
};

template <typename T>
struct GRUGradCell : GradCell<T> {
  void operator()(const CPUContext& dev_ctx,
                  DenseTensor* gate_tensor,
                  DenseTensor* state_tensor,
                  DenseTensor* act_state_tensor UNUSED,
                  DenseTensor* hidden_tensor UNUSED,
                  const DenseTensor* weight_hh,
                  DenseTensor* pre_hidden,
                  DenseTensor* pre_state UNUSED,
                  DenseTensor* grad_hidden,
                  DenseTensor* grad_state,
                  DenseTensor* grad_gate,
                  DenseTensor* grad_weight_hh,
                  DenseTensor* grad_pre_hidden,
                  DenseTensor* grad_pre_state UNUSED,
                  DenseTensor* grad_bias_hh,
                  const DenseTensor& mask_tensor,
                  bool has_sequence_length) const override {
    size_t frame_size = pre_hidden->dims()[2];
    size_t batch_size = pre_hidden->dims()[1];
    DenseTensor grad_pre_hidden_bak;
    if (has_sequence_length) {
      BackupTensor<T>(dev_ctx, &grad_pre_hidden_bak, grad_pre_hidden);
    }
    // zero pre_hidden
    phi::funcs::SetConstant<CPUContext, T> zero;
    zero(dev_ctx, grad_pre_hidden, static_cast<T>(0.0));
    phi::funcs::GRUMetaValue<T> gru_value;
    phi::funcs::GRUMetaGrad<T> gru_grad;
    gru_value.gate_value = gate_tensor->data<T>();
    gru_value.prev_out_value = pre_hidden->data<T>();
    gru_value.reset_output_value = state_tensor->data<T>();
    gru_value.state_weight = weight_hh->data<T>() + 2 * frame_size * frame_size;
    gru_value.gate_weight = weight_hh->data<T>();

    gru_grad.gate_grad = grad_gate->data<T>();
    gru_grad.reset_output_grad = grad_state->data<T>();  // NOLINT
    gru_grad.prev_out_grad = grad_pre_hidden->data<T>();
    gru_grad.output_grad = grad_hidden->data<T>();
    gru_grad.gate_weight_grad = grad_weight_hh->data<T>();
    gru_grad.state_weight_grad =
        grad_weight_hh->data<T>() + 2 * frame_size * frame_size;
    gru_grad.bias_hh_grad = grad_bias_hh->data<T>();

    auto act_gate = phi::funcs::detail::GetActivationType("sigmoid_v2");
    auto act_node = phi::funcs::detail::GetActivationType("tanh_v2");
    phi::funcs::GRUUnitGradFunctorV2<CPUContext, T>::compute(dev_ctx,
                                                             gru_value,
                                                             gru_grad,
                                                             frame_size,
                                                             batch_size,
                                                             act_node,
                                                             act_gate);

    this->postprocess_pre_hidden_grad(dev_ctx,
                                      grad_pre_hidden,
                                      &grad_pre_hidden_bak,
                                      nullptr,
                                      nullptr,
                                      mask_tensor,
                                      has_sequence_length);
  }
};

template <typename T>
struct LSTMGradCell : GradCell<T> {
  void operator()(const CPUContext& dev_ctx,
                  DenseTensor* gate_tensor,
                  DenseTensor* state_tensor,
                  DenseTensor* act_state_tensor,
                  DenseTensor* hidden_tensor UNUSED,
                  const DenseTensor* weight_hh,
                  DenseTensor* pre_hidden,
                  DenseTensor* pre_state,
                  DenseTensor* grad_hidden,
                  DenseTensor* grad_state,
                  DenseTensor* grad_gate,
                  DenseTensor* grad_weight_hh,
                  DenseTensor* grad_pre_hidden,
                  DenseTensor* grad_pre_state,
                  DenseTensor* grad_bias_hh UNUSED,
                  const DenseTensor& mask_tensor,
                  bool has_sequence_length) const override {
    size_t frame_size = state_tensor->dims()[2];
    size_t batch_size = state_tensor->dims()[1];

    DenseTensor grad_pre_hidden_bak;
    DenseTensor grad_pre_state_bak;
    if (has_sequence_length) {
      BackupTensor<T>(dev_ctx, &grad_pre_hidden_bak, grad_pre_hidden);
      BackupTensor<T>(dev_ctx, &grad_pre_state_bak, grad_pre_state);
    }

    phi::funcs::LstmMetaValue<T> lstm_value;
    phi::funcs::LstmMetaGrad<T> lstm_grad;
    CreateLstmValue(&lstm_value);
    CreateLstmGrad(&lstm_grad);
    lstm_value.gate_value = gate_tensor->data<T>();
    lstm_value.state_value = state_tensor->data<T>();
    lstm_value.state_active_value = act_state_tensor->data<T>();
    lstm_value.prev_state_value = pre_state->data<T>();  // NOLINT

    lstm_grad.state_grad = grad_state->data<T>();  // NOLINT
    lstm_grad.gate_grad = grad_gate->data<T>();
    lstm_grad.output_grad = grad_hidden->data<T>();
    lstm_grad.prev_state_grad = grad_pre_state->data<T>();

    lstm_value.output_value = nullptr;
    lstm_grad.state_active_grad = nullptr;

    auto gate_act = phi::funcs::detail::GetActivationType("sigmoid_v2");
    auto state_act = phi::funcs::detail::GetActivationType("tanh_v2");
    auto cand_act = phi::funcs::detail::GetActivationType("tanh_v2");

    T cell_clip = 0.0;
    phi::funcs::LstmUnitGradFunctor<CPUContext, T>::compute(dev_ctx,
                                                            lstm_value,
                                                            lstm_grad,
                                                            frame_size,
                                                            batch_size,
                                                            cell_clip,
                                                            gate_act,
                                                            state_act,
                                                            cand_act,
                                                            false);
    this->update_pre_hidden_grad(dev_ctx,
                                 grad_gate,
                                 weight_hh,
                                 grad_pre_hidden,
                                 &grad_pre_hidden_bak,
                                 grad_pre_state,
                                 &grad_pre_state_bak,
                                 mask_tensor,
                                 has_sequence_length);
    this->update_weight_hh_grad(dev_ctx, grad_gate, pre_hidden, grad_weight_hh);
  }
};

template <typename T, typename GradCellType>
struct GradLayer {
  explicit GradLayer(const GradCellType& cell) : cell_(cell) {}
  virtual ~GradLayer() = default;
  void run_rnn_grad_function(
      const CPUContext& dev_ctx,
      const DenseTensor* input,
      DenseTensor* input_grad,
      const DenseTensor* sequence_length,
      std::vector<DenseTensor>* init_h_unbind,
      std::vector<DenseTensor>* init_c_unbind,
      std::vector<DenseTensor>* init_h_grad_unbind,
      std::vector<DenseTensor>* init_c_grad_unbind,
      DenseTensor* layer_grad_gate_tensor,
      std::vector<DenseTensor>* layer_gate_tensor_unbind,
      std::vector<DenseTensor>* layer_grad_gate_tensor_unbind,
      std::vector<DenseTensor>* layer_state_tensor_unbind,
      std::vector<DenseTensor>* layer_act_state_tensor_unbind,
      std::vector<DenseTensor>* output_tensor_unbind,
      std::vector<DenseTensor>* output_grad_tensor_unbind,
      const std::vector<DenseTensor>& last_h_grad_unbind,
      const std::vector<DenseTensor>& last_c_grad_unbind,
      const std::vector<std::vector<DenseTensor>>& parameter_lists,
      std::vector<std::vector<DenseTensor>>* weight_list_grad,
      int layer_idx,
      int time_step,
      bool has_sequence_length,
      bool is_bidirec,
      bool is_reverse,
      const std::string& mode) {
    int direction_num = is_bidirec ? 2 : 1;
    int current_reverse_idx = is_reverse ? 1 : 0;
    int current_layer_idx = direction_num * layer_idx + current_reverse_idx;
    int begin_idx = 0;
    if (is_reverse) {
      begin_idx = time_step;
    }

    DenseTensor mask_matrix;
    std::vector<DenseTensor> mask_tensor_list;
    int mask_min_length = time_step;
    if (has_sequence_length) {
      mask_matrix.Resize(common::make_ddim({time_step, input->dims()[1]}));
      CreateMaskMatrix<T>(
          dev_ctx, sequence_length, &mask_matrix, is_reverse, &mask_min_length);
      mask_tensor_list = Unbind(mask_matrix);
    }
    // copy the last_h, last_c for swapping pointer
    DenseTensor a, b;
    DenseTensor* dynamic_grad_last_h = &a;
    DenseTensor* dynamic_grad_last_c = &b;
    dynamic_grad_last_h->Resize(last_h_grad_unbind[current_layer_idx].dims());
    dev_ctx.Alloc<T>(dynamic_grad_last_h);
    Copy(dev_ctx,
         last_h_grad_unbind[current_layer_idx],
         dev_ctx.GetPlace(),
         false,
         dynamic_grad_last_h);
    if (!last_c_grad_unbind.empty()) {
      dynamic_grad_last_c->Resize(last_c_grad_unbind[current_layer_idx].dims());
      dev_ctx.Alloc<T>(dynamic_grad_last_c);
      Copy(dev_ctx,
           last_c_grad_unbind[current_layer_idx],
           dev_ctx.GetPlace(),
           false,
           dynamic_grad_last_c);
    } else {
      dynamic_grad_last_c = nullptr;
    }

    DenseTensor c, d;
    DenseTensor* dynamic_grad_pre_h = &c;
    DenseTensor* dynamic_grad_pre_c = &d;
    phi::funcs::SetConstant<CPUContext, T> zero;
    if (!init_h_grad_unbind->empty()) {
      dynamic_grad_pre_h->ShareDataWith(
          (*init_h_grad_unbind)[current_layer_idx]);
    } else {
      dynamic_grad_pre_h->Resize(dynamic_grad_last_h->dims());
      dev_ctx.Alloc<T>(dynamic_grad_pre_h);
      zero(dev_ctx, dynamic_grad_pre_h, static_cast<T>(0.0));
    }
    if (!init_c_grad_unbind->empty()) {
      dynamic_grad_pre_c->ShareDataWith(
          (*init_c_grad_unbind)[current_layer_idx]);
    } else {
      if (is_lstm(mode) || is_gru(mode)) {
        dynamic_grad_pre_c->Resize(dynamic_grad_last_h->dims());
        dev_ctx.Alloc<T>(dynamic_grad_pre_c);
        if (is_gru(mode)) {
          dynamic_grad_last_c = dynamic_grad_pre_c;
        }
      } else {
        dynamic_grad_pre_c = nullptr;
      }
    }

    if (is_reverse) {
      // must be reverse the input, output, input_grad, output_grad
      // the gate and grad_gate must be reverse
      std::reverse(layer_gate_tensor_unbind->begin(),
                   layer_gate_tensor_unbind->end());
      std::reverse(layer_grad_gate_tensor_unbind->begin(),
                   layer_grad_gate_tensor_unbind->end());
      /*
      if (has_sequence_length) {
        std::reverse(mask_tensor_list.begin(), mask_tensor_list.end());
      }*/
      std::reverse(output_tensor_unbind->begin(), output_tensor_unbind->end());
      std::reverse(output_grad_tensor_unbind->begin(),
                   output_grad_tensor_unbind->end());
    }

    DenseTensor* weight_grad =
        &((*weight_list_grad)[layer_idx][current_reverse_idx * 4 + 1]);
    dev_ctx.Alloc<T>(weight_grad);
    zero(dev_ctx, weight_grad, static_cast<T>(0.0));

    DenseTensor* pre_hidden = nullptr;
    DenseTensor* pre_state = nullptr;
    DenseTensor* hidden = nullptr;
    if (is_gru(mode)) {
      zero(dev_ctx,
           &((*weight_list_grad)[layer_idx][current_reverse_idx * 4 + 3]),
           static_cast<T>(0.0));
    }
    for (int i = time_step - 1; i >= 0; --i) {
      if (has_sequence_length) {
        this->mask_preprocess(dev_ctx,
                              &(*output_grad_tensor_unbind)[i],
                              dynamic_grad_last_h,
                              dynamic_grad_last_c,
                              dynamic_grad_pre_h,
                              dynamic_grad_pre_c,
                              mask_tensor_list[i],
                              mode);
      } else {
        this->preprocess(
            dev_ctx, &(*output_grad_tensor_unbind)[i], dynamic_grad_last_h);
      }
      hidden = &(*output_tensor_unbind)[i];
      if (i == 0) {
        pre_hidden = &(*init_h_unbind)[current_layer_idx];
        if (!init_c_unbind->empty()) {
          pre_state = &(*init_c_unbind)[current_layer_idx];
        }
      } else {
        pre_hidden = &(*output_tensor_unbind)[i - 1];
        if (!layer_state_tensor_unbind->empty()) {
          pre_state = &(*layer_state_tensor_unbind)[begin_idx + i - 1];
        }
      }
      this->cell_(
          dev_ctx,
          &(*layer_gate_tensor_unbind)[i],
          &(*layer_state_tensor_unbind)[begin_idx + i],
          &(*layer_act_state_tensor_unbind)[begin_idx + i],
          hidden,
          &(parameter_lists[layer_idx][current_reverse_idx * 4 + 1]),
          pre_hidden,
          pre_state,
          dynamic_grad_last_h,
          dynamic_grad_last_c,
          &(*layer_grad_gate_tensor_unbind)[i],
          weight_grad,
          dynamic_grad_pre_h,
          dynamic_grad_pre_c,
          &((*weight_list_grad)[layer_idx][current_reverse_idx * 4 + 3]),
          mask_tensor_list[i],
          has_sequence_length);
      SwapPointer(&dynamic_grad_last_h, &dynamic_grad_pre_h);
      SwapPointer(&dynamic_grad_last_c, &dynamic_grad_pre_c);
    }
    // postproces for gradient for w_hi, X, bias_hi, bias_hh
    this->postprocess(dev_ctx,
                      *layer_grad_gate_tensor,
                      *input,
                      input_grad,
                      parameter_lists[layer_idx],
                      &((*weight_list_grad)[layer_idx]),
                      is_reverse,
                      mode);

    // copy the gradient to init_c init_h
    if (!(*init_h_grad_unbind).empty() && time_step % 2 == 0) {
      Copy(dev_ctx,
           *dynamic_grad_last_h,
           dev_ctx.GetPlace(),
           false,
           &((*init_h_grad_unbind)[current_layer_idx]));
    }
    if (!(*init_c_grad_unbind).empty() && time_step % 2 == 0) {
      Copy(dev_ctx,
           *dynamic_grad_last_c,
           dev_ctx.GetPlace(),
           false,
           &((*init_c_grad_unbind)[current_layer_idx]));
    }
  }

  virtual void operator()(
      const CPUContext& dev_ctx UNUSED,
      const DenseTensor* input UNUSED,
      const DenseTensor* output UNUSED,
      const std::vector<DenseTensor>& init_h_unbind UNUSED,
      const std::vector<DenseTensor>& init_c_unbind UNUSED,
      const std::vector<DenseTensor>& last_h_grad_unbind UNUSED,
      const std::vector<DenseTensor>& last_c_grad_unbind UNUSED,
      const std::vector<DenseTensor>& gate_tensor_unbind UNUSED,
      const std::vector<DenseTensor>& state_tensor_unbind UNUSED,
      const std::vector<DenseTensor>& act_state_tensor_unbind UNUSED,
      const DenseTensor* output_grad UNUSED,
      const std::vector<std::vector<DenseTensor>>& parameter_lists UNUSED,
      const DenseTensor* sequence_length UNUSED,
      DenseTensor* input_grad UNUSED,
      std::vector<DenseTensor>* init_h_grad_unbind UNUSED,
      std::vector<DenseTensor>* init_c_grad_unbind UNUSED,
      const std::vector<std::vector<DenseTensor>>& weight_list_grad UNUSED,
      int layer_idx UNUSED,
      bool is_bidirec UNUSED,
      int hidden_size UNUSED,
      const std::string& mode UNUSED,
      int gate_num UNUSED) {}

  void preprocess(const CPUContext& dev_ctx,
                  const DenseTensor* grad_output,
                  DenseTensor* grad_last_h) {
    auto& place = *dev_ctx.eigen_device();
    auto output_grad =
        EigenMatrix<T>::Reshape(*grad_output, grad_output->dims().size() - 1);
    auto last_h_grad =
        EigenMatrix<T>::Reshape(*grad_last_h, grad_last_h->dims().size() - 1);
    // the output gradient contribute the gradient to last_h
    last_h_grad.device(place) = last_h_grad + output_grad;
  }

  void mask_preprocess(const CPUContext& dev_ctx,
                       const DenseTensor* grad_output,
                       DenseTensor* grad_last_h,
                       DenseTensor* grad_last_c,
                       DenseTensor* grad_pre_h,
                       DenseTensor* grad_pre_c,
                       const DenseTensor& mask_tensor,
                       const std::string& mode) {
    auto& place = *dev_ctx.eigen_device();
    auto mask = EigenMatrix<T>::From(
        mask_tensor, common::make_ddim({mask_tensor.dims()[1], 1}));
    auto mask_broadcast = mask.broadcast(
        Eigen::DSizes<int, 2>(1, static_cast<int>(grad_output->dims()[2])));

    auto last_h_grad =
        EigenMatrix<T>::Reshape(*grad_last_h, grad_last_h->dims().size() - 1);
    auto pre_h_grad =
        EigenMatrix<T>::Reshape(*grad_pre_h, grad_pre_h->dims().size() - 1);
    auto output_grad =
        EigenMatrix<T>::Reshape(*grad_output, grad_output->dims().size() - 1);
    last_h_grad.device(place) = last_h_grad + output_grad * mask_broadcast;
    pre_h_grad.device(place) = (1 - mask_broadcast) * last_h_grad;
    last_h_grad.device(place) = mask_broadcast * last_h_grad;

    if (grad_last_c && grad_pre_c && is_lstm(mode)) {
      auto last_c_grad =
          EigenMatrix<T>::Reshape(*grad_last_c, grad_last_c->dims().size() - 1);
      auto pre_c_grad =
          EigenMatrix<T>::Reshape(*grad_pre_c, grad_pre_c->dims().size() - 1);
      pre_c_grad.device(place) = (1 - mask_broadcast) * last_c_grad;
      last_c_grad.device(place) = mask_broadcast * last_c_grad;
    }
  }

  void postprocess(const CPUContext& dev_ctx,
                   const DenseTensor& grad_gate,
                   const DenseTensor& input,
                   DenseTensor* input_grad,
                   const std::vector<DenseTensor>& parameters,
                   std::vector<DenseTensor>* grad_parameters,
                   int is_reverse,
                   const std::string& mode) {
    // we get the grad_gate step by step, and need to bradocast the grad to the
    // grad_w_hi, grad_bias_hi, grad_bias_hh
    int begin_idx = 0;
    if (is_reverse) {
      begin_idx = 4;
    }
    auto blas = phi::funcs::GetBlas<CPUContext, T>(dev_ctx);

    // calc the gradient for the w_hi
    auto mat_dim_out_grad =
        phi::funcs::CreateMatrixDescriptor(grad_gate.dims(), 0, true);
    auto mat_dim_input =
        phi::funcs::CreateMatrixDescriptor(input.dims(), 0, false);
    mat_dim_out_grad.width_ *= mat_dim_out_grad.batch_size_;
    mat_dim_out_grad.batch_size_ = 0;
    mat_dim_input.height_ *= mat_dim_input.batch_size_;
    mat_dim_input.batch_size_ = 0;
    blas.MatMul(grad_gate,
                mat_dim_out_grad,
                input,
                mat_dim_input,
                static_cast<T>(1.0),
                &((*grad_parameters)[begin_idx + 0]),
                T(0));

    // calc the gradient for the X
    auto mat_dim_out_grad_new =
        phi::funcs::CreateMatrixDescriptor(grad_gate.dims(), 0, false);
    mat_dim_out_grad_new.height_ *= mat_dim_out_grad_new.batch_size_;
    mat_dim_out_grad_new.batch_size_ = 0;
    auto mat_dim_parameter =
        phi::funcs::CreateMatrixDescriptor(parameters[0].dims(), 0, false);
    blas.MatMul(grad_gate,
                mat_dim_out_grad_new,
                parameters[begin_idx + 0],
                mat_dim_parameter,
                static_cast<T>(1.0),
                input_grad,
                T(1));

    // calc the gradient of Bias_hi, Bias_hh
    phi::funcs::ColwiseSum<CPUContext, T> col_sum;
    DenseTensor tmp_grad_gate;
    tmp_grad_gate.ShareDataWith(grad_gate);
    tmp_grad_gate.Resize(
        {grad_gate.dims()[0] * grad_gate.dims()[1], grad_gate.dims()[2]});
    col_sum(dev_ctx, tmp_grad_gate, &((*grad_parameters)[begin_idx + 2]));
    // Bias_hh
    if (!is_gru(mode)) {
      col_sum(dev_ctx, tmp_grad_gate, &((*grad_parameters)[begin_idx + 3]));
    }
  }
  GradCellType cell_;
};

template <typename T, typename GradCellType>
struct SingleGradLayer : GradLayer<T, GradCellType> {
  // explicit SingleGradLayer(GradCellType& cell) : cell_(cell) {}
  explicit SingleGradLayer(const GradCellType& cell)
      : GradLayer<T, GradCellType>(cell) {}
  ~SingleGradLayer() override = default;
  using GradLayer<T, GradCellType>::operator();
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor* input,
                  const DenseTensor* output,
                  std::vector<DenseTensor>* init_h_unbind,
                  std::vector<DenseTensor>* init_c_unbind,
                  const std::vector<DenseTensor>& last_h_grad_unbind,
                  const std::vector<DenseTensor>& last_c_grad_unbind,
                  const std::vector<DenseTensor>& gate_tensor_unbind,
                  const std::vector<DenseTensor>& state_tensor_unbind,
                  const std::vector<DenseTensor>& act_state_tensor_unbind,
                  const DenseTensor* output_grad,
                  const std::vector<std::vector<DenseTensor>>& parameter_lists,
                  const DenseTensor* sequence_length,
                  DenseTensor* input_grad,
                  std::vector<DenseTensor>* init_h_grad_unbind,
                  std::vector<DenseTensor>* init_c_grad_unbind,
                  std::vector<std::vector<DenseTensor>>* weight_list_grad,
                  int layer_idx,
                  bool is_bidirec,
                  int hidden_size,
                  const std::string& mode,
                  int gate_num) {
    phi::funcs::SetConstant<CPUContext, T> zero;
    zero(dev_ctx, input_grad, static_cast<T>(0.0));

    int time_step = static_cast<int>(input->dims()[0]);
    int batch_size = static_cast<int>(input->dims()[1]);
    int direction_num = is_bidirec ? 2 : 1;

    // in this section, create the gate_state_grad for the postprocess calculate
    // ubind the output, the output from [time_step, batch_size, hidden_size]
    auto output_tensor_unbind = Unbind(*output);
    auto output_grad_tensor_unbind = Unbind(*output_grad);
    auto layer_gate_tensor = gate_tensor_unbind[layer_idx];
    layer_gate_tensor.Resize(
        {time_step * direction_num, batch_size, hidden_size * gate_num});
    auto layer_gate_tensor_unbind = Unbind(layer_gate_tensor);
    // the gate_tensor and the grad_gate_tensor must be unbind
    DenseTensor layer_grad_gate_tensor;
    layer_grad_gate_tensor.Resize(layer_gate_tensor.dims());
    dev_ctx.Alloc<T>(&layer_grad_gate_tensor);
    auto layer_grad_gate_tensor_unbind = Unbind(layer_grad_gate_tensor);

    DenseTensor layer_state_tensor;
    std::vector<DenseTensor> layer_state_tensor_unbind;
    if (!state_tensor_unbind.empty()) {
      layer_state_tensor = state_tensor_unbind[layer_idx];
      layer_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_state_tensor_unbind = Unbind(layer_state_tensor);
    }

    DenseTensor layer_act_state_tensor;
    std::vector<DenseTensor> layer_act_state_tensor_unbind;
    if (!act_state_tensor_unbind.empty()) {
      layer_act_state_tensor = act_state_tensor_unbind[layer_idx];
      layer_act_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_act_state_tensor_unbind = Unbind(layer_act_state_tensor);
    }
    bool has_sequence_length = sequence_length == nullptr ? false : true;
    this->run_rnn_grad_function(dev_ctx,
                                input,
                                input_grad,
                                sequence_length,
                                init_h_unbind,
                                init_c_unbind,
                                init_h_grad_unbind,
                                init_c_grad_unbind,
                                &layer_grad_gate_tensor,
                                &layer_gate_tensor_unbind,
                                &layer_grad_gate_tensor_unbind,
                                &layer_state_tensor_unbind,
                                &layer_act_state_tensor_unbind,
                                &output_tensor_unbind,
                                &output_grad_tensor_unbind,
                                last_h_grad_unbind,
                                last_c_grad_unbind,
                                parameter_lists,
                                weight_list_grad,
                                layer_idx,
                                time_step,
                                has_sequence_length,
                                is_bidirec,
                                false,
                                mode);
  }
};

template <typename T>
void split_tensor_at_last_dim(const CPUContext& dev_ctx,
                              const DenseTensor* output,
                              std::vector<DenseTensor*>* output_vec,
                              int axis) {
  std::vector<const DenseTensor*> shape_refer;
  (*output_vec)[0]->Resize(
      {output->dims()[0], output->dims()[1], output->dims()[2] / 2});
  dev_ctx.Alloc<T>((*output_vec)[0]);
  (*output_vec)[1]->Resize(
      {output->dims()[0], output->dims()[1], output->dims()[2] / 2});
  dev_ctx.Alloc<T>((*output_vec)[1]);
  shape_refer.emplace_back((*output_vec)[0]);
  shape_refer.emplace_back((*output_vec)[1]);
  funcs::SplitFunctor<CPUContext, T> functor;
  functor(dev_ctx, *output, shape_refer, axis, output_vec);
}

template <typename T, typename GradCellType>
struct BidirGradLayer : GradLayer<T, GradCellType> {
  explicit BidirGradLayer(const GradCellType& cell)
      : GradLayer<T, GradCellType>(cell) {}
  ~BidirGradLayer() override = default;
  using GradLayer<T, GradCellType>::operator();
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor* input,
                  const DenseTensor* output,
                  std::vector<DenseTensor>* init_h_unbind,
                  std::vector<DenseTensor>* init_c_unbind,
                  const std::vector<DenseTensor>& last_h_grad_unbind,
                  const std::vector<DenseTensor>& last_c_grad_unbind,
                  const std::vector<DenseTensor>& gate_tensor_unbind,
                  const std::vector<DenseTensor>& state_tensor_unbind,
                  const std::vector<DenseTensor>& act_state_tensor_unbind,
                  const DenseTensor* output_grad,
                  const std::vector<std::vector<DenseTensor>>& parameter_lists,
                  const DenseTensor* sequence_length,
                  DenseTensor* input_grad,
                  std::vector<DenseTensor>* init_h_grad_unbind,
                  std::vector<DenseTensor>* init_c_grad_unbind,
                  std::vector<std::vector<DenseTensor>>* weight_list_grad,
                  int layer_idx,
                  bool is_bidirec,
                  int hidden_size,
                  const std::string& mode,
                  int gate_num) {
    int time_step = static_cast<int>(input->dims()[0]);
    int batch_size = static_cast<int>(input->dims()[1]);
    int direction_num = is_bidirec ? 2 : 1;
    // split the output two tensor to output_forward, output_backward
    phi::funcs::SetConstant<CPUContext, T> zero;
    zero(dev_ctx, input_grad, static_cast<T>(0.0));

    std::vector<DenseTensor*> output_vec;
    DenseTensor forward_output;
    DenseTensor backward_output;
    std::vector<DenseTensor> forward_output_tensor_unbind;
    std::vector<DenseTensor> backward_output_tensor_unbind;
    // in the last layer, we will use the output as the last hidden
    // the output just the concat the forward hidden, backward hidden, so just
    // split it
    // in other layer, we just split the hidden in the rows
    output_vec.emplace_back(&forward_output);
    output_vec.emplace_back(&backward_output);
    split_tensor_at_last_dim<T>(dev_ctx, output, &output_vec, 2);
    forward_output_tensor_unbind = Unbind(*(output_vec[0]));
    backward_output_tensor_unbind = Unbind(*(output_vec[1]));

    std::vector<DenseTensor*> output_grad_vec;
    DenseTensor grad_forward_output;
    DenseTensor grad_backward_output;
    output_grad_vec.emplace_back(&grad_forward_output);
    output_grad_vec.emplace_back(&grad_backward_output);
    split_tensor_at_last_dim<T>(dev_ctx, output_grad, &output_grad_vec, 2);
    auto forward_output_grad_tensor_unbind = Unbind(*(output_grad_vec[0]));
    auto backward_output_grad_tensor_unbind = Unbind(*(output_grad_vec[1]));

    // the gate_tensor and the grad_gate_tensor must be unbind
    auto layer_gate_tensor = gate_tensor_unbind[layer_idx];
    layer_gate_tensor.Resize(
        {time_step * 2, batch_size, hidden_size * gate_num});
    auto layer_forward_gate_tensor = layer_gate_tensor.Slice(0, time_step);
    auto layer_backward_gate_tensor =
        layer_gate_tensor.Slice(time_step, 2 * time_step);
    auto layer_forward_gate_tensor_unbind = Unbind(layer_forward_gate_tensor);
    auto layer_backward_gate_tensor_unbind = Unbind(layer_backward_gate_tensor);

    DenseTensor layer_grad_gate_tensor;
    layer_grad_gate_tensor.Resize(layer_gate_tensor.dims());
    dev_ctx.Alloc<T>(&layer_grad_gate_tensor);
    zero(dev_ctx, &layer_grad_gate_tensor, static_cast<T>(0.0));
    auto layer_forward_grad_gate_tensor =
        layer_grad_gate_tensor.Slice(0, time_step);
    auto layer_backward_grad_gate_tensor =
        layer_grad_gate_tensor.Slice(time_step, 2 * time_step);
    auto layer_forward_grad_gate_tensor_unbind =
        Unbind(layer_forward_grad_gate_tensor);
    auto layer_backward_grad_gate_tensor_unbind =
        Unbind(layer_backward_grad_gate_tensor);

    DenseTensor layer_state_tensor;
    std::vector<DenseTensor> layer_state_tensor_unbind;
    if (!state_tensor_unbind.empty()) {
      layer_state_tensor = state_tensor_unbind[layer_idx];
      layer_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_state_tensor_unbind = Unbind(layer_state_tensor);
    }

    DenseTensor layer_act_state_tensor;
    std::vector<DenseTensor> layer_act_state_tensor_unbind;
    if (!act_state_tensor_unbind.empty()) {
      layer_act_state_tensor = act_state_tensor_unbind[layer_idx];
      layer_act_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_act_state_tensor_unbind = Unbind(layer_act_state_tensor);
    }
    const bool& has_sequence_length = sequence_length == nullptr ? false : true;

    this->run_rnn_grad_function(dev_ctx,
                                input,
                                input_grad,
                                sequence_length,
                                init_h_unbind,
                                init_c_unbind,
                                init_h_grad_unbind,
                                init_c_grad_unbind,
                                &layer_forward_grad_gate_tensor,
                                &layer_forward_gate_tensor_unbind,
                                &layer_forward_grad_gate_tensor_unbind,
                                &layer_state_tensor_unbind,
                                &layer_act_state_tensor_unbind,
                                &forward_output_tensor_unbind,
                                &forward_output_grad_tensor_unbind,
                                last_h_grad_unbind,
                                last_c_grad_unbind,
                                parameter_lists,
                                weight_list_grad,
                                layer_idx,
                                time_step,
                                has_sequence_length,
                                is_bidirec,
                                false,
                                mode);

    this->run_rnn_grad_function(dev_ctx,
                                input,
                                input_grad,
                                sequence_length,
                                init_h_unbind,
                                init_c_unbind,
                                init_h_grad_unbind,
                                init_c_grad_unbind,
                                &layer_backward_grad_gate_tensor,
                                &layer_backward_gate_tensor_unbind,
                                &layer_backward_grad_gate_tensor_unbind,
                                &layer_state_tensor_unbind,
                                &layer_act_state_tensor_unbind,
                                &backward_output_tensor_unbind,
                                &backward_output_grad_tensor_unbind,
                                last_h_grad_unbind,
                                last_c_grad_unbind,
                                parameter_lists,
                                weight_list_grad,
                                layer_idx,
                                time_step,
                                has_sequence_length,
                                is_bidirec,
                                true,
                                mode);
  }
};

template <typename T>
void dropout_cpu_grad_function_inplace(const CPUContext& dev_ctx,
                                       DenseTensor* grad_x,
                                       const DenseTensor* mask,
                                       float dropout_prob) {
  DropoutHelper<T>(dev_ctx, grad_x, grad_x, mask, dropout_prob);
}

template <typename GradCellType,
          template <typename, typename>
          class SingleGradLayerT,
          template <typename, typename>
          class BidirGradLayerT,
          typename T>
void RnnGradFunc(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<const DenseTensor*>& pre_state,
                 const std::vector<const DenseTensor*>& weight_list,
                 const paddle::optional<DenseTensor>& sequence_length,
                 const DenseTensor& out,
                 const DenseTensor& dropout_state,
                 const DenseTensor& reserve,
                 const DenseTensor& out_grad,
                 const std::vector<const DenseTensor*>& state_grad,
                 float dropout_prob,
                 bool is_bidirec,
                 int input_size UNUSED,
                 int hidden_size,
                 int num_layers,
                 const std::string& mode,
                 int seed UNUSED,
                 bool is_test,
                 int gate_num,
                 DenseTensor* x_grad,
                 std::vector<DenseTensor*> pre_state_grad,
                 std::vector<DenseTensor*> weight_grad_list) {
  const DenseTensor* init_h = pre_state[0];
  const DenseTensor* init_c = nullptr;
  if (is_lstm(mode)) {
    init_c = pre_state[1];
  }
  const DenseTensor* last_h_grad = state_grad[0];
  const DenseTensor* last_c_grad = nullptr;
  if (is_lstm(mode)) {
    last_c_grad = state_grad[1];
  }

  DenseTensor* init_h_grad = nullptr;
  DenseTensor* init_c_grad = nullptr;
  if (!pre_state_grad.empty()) {  // has gradient
    init_h_grad = pre_state_grad[0];
    if (is_lstm(mode) && pre_state_grad.size() > 1) {
      init_c_grad = pre_state_grad[1];
    }
  }

  // get the input_size, batch_size, time_step
  const int time_step = static_cast<int>(x.dims()[0]);
  const int batch_size = static_cast<int>(x.dims()[1]);
  const int direction_num = is_bidirec ? 2 : 1;

  // allocate the memory and initization the x_grad
  DenseTensor x_grad_value;
  if (!x_grad) {
    x_grad = &x_grad_value;
  }
  x_grad->Resize(x.dims());
  dev_ctx.Alloc<T>(x_grad);

  if (init_h_grad) {
    init_h_grad->Resize(init_h->dims());
    dev_ctx.Alloc<T>(init_h_grad);
  }
  if (init_c_grad) {
    init_c_grad->Resize(init_c->dims());
    dev_ctx.Alloc<T>(init_c_grad);
  }

  // reset the parameter to sorted order and allocate the memory
  std::vector<std::vector<DenseTensor>> parameter_lists;
  parameter_lists.reserve(num_layers);
  ResetParameterVector(
      weight_list, num_layers, gate_num, is_bidirec, &parameter_lists);

  for (auto& weight_grad : weight_grad_list) {
    dev_ctx.Alloc<T>(weight_grad);
  }
  std::vector<std::vector<DenseTensor>> parameter_lists_grad;
  parameter_lists_grad.reserve(num_layers);
  ResetParameterVector(weight_grad_list,
                       num_layers,
                       gate_num,
                       is_bidirec,
                       &parameter_lists_grad);

  // resolve the state of reverse_state
  DenseTensor gate_tensor;
  DenseTensor state_tensor;
  DenseTensor act_state_tensor;
  DenseTensor hidden_tensor;
  SplitReserveData(dev_ctx,
                   direction_num,
                   time_step,
                   batch_size,
                   hidden_size,
                   gate_num,
                   num_layers,
                   mode,
                   &reserve,
                   &gate_tensor,
                   &state_tensor,
                   &act_state_tensor,
                   &hidden_tensor);
  int gate_num_tmp = gate_num;
  if (gate_num == 0) {
    gate_num_tmp = 1;
  }
  gate_tensor.Resize({num_layers,
                      time_step * direction_num,
                      batch_size,
                      hidden_size * gate_num_tmp});
  if (state_tensor.numel() > 0) {
    state_tensor.Resize(
        {num_layers, time_step * direction_num, batch_size, hidden_size});
  }
  if (act_state_tensor.numel() > 0) {
    act_state_tensor.Resize(
        {num_layers, time_step * direction_num, batch_size, hidden_size});
  }
  if (num_layers > 1) {
    hidden_tensor.Resize(
        {num_layers - 1, time_step, batch_size, hidden_size * direction_num});
  }

  // unbind
  auto last_h_grad_unbind = Unbind(*last_h_grad);
  auto gate_tensor_unbind = Unbind(gate_tensor);
  std::vector<DenseTensor> last_c_grad_unbind;
  if (last_c_grad) {
    last_c_grad_unbind = Unbind(*last_c_grad);
  }

  std::vector<DenseTensor> init_h_unbind, init_c_unbind;
  std::vector<DenseTensor> init_h_grad_unbind, init_c_grad_unbind;
  std::vector<DenseTensor> state_tensor_unbind, act_state_tensor_unbind;
  std::vector<DenseTensor> hidden_tensor_unbind;

  init_h_unbind = Unbind(*init_h);
  if (init_c) {
    init_c_unbind = Unbind(*init_c);
  }

  if (init_h_grad != nullptr) {
    init_h_grad_unbind = Unbind(*init_h_grad);
  }
  if (init_c_grad != nullptr) {
    init_c_grad_unbind = Unbind(*init_c_grad);
  }
  if (state_tensor.numel() > 0) {
    state_tensor_unbind = Unbind(state_tensor);
  }
  if (act_state_tensor.numel() > 0) {
    act_state_tensor_unbind = Unbind(act_state_tensor);
  }
  if (num_layers > 1) {
    hidden_tensor_unbind = Unbind(hidden_tensor);
  }
  // squeeze the hidden first dim
  for (auto& hidden_tensor : hidden_tensor_unbind) {
    hidden_tensor.Resize(common::slice_ddim(
        hidden_tensor.dims(), 1, hidden_tensor.dims().size()));
  }
  // add the output tensor to the hidden vector
  DenseTensor tmp;
  hidden_tensor_unbind.emplace_back(tmp);
  hidden_tensor_unbind[num_layers - 1].ShareDataWith(out);

  GradCellType cell;
  DenseTensor layer_input;
  DenseTensor layer_output;
  DenseTensor* layer_x_grad_holder = nullptr;
  DenseTensor tmp_out;
  tmp_out.ShareDataWith(out_grad);
  DenseTensor* layer_output_grad_holder = &tmp_out;
  DenseTensor x_grad_temp;
  DenseTensor output_grad_temp;

  bool has_allocate_mem = false;
  for (int i = num_layers - 1; i >= 0; --i) {
    // the layer input output had saved, just use the data
    if (i > 0) {
      if (layer_input.numel() == 0) {
        layer_input.Resize(hidden_tensor_unbind[i - 1].dims());
        dev_ctx.Alloc<T>(&layer_input);
      }
      DropoutHelper<T>(dev_ctx,
                       &hidden_tensor_unbind[i - 1],
                       &layer_input,
                       &dropout_state,
                       dropout_prob);
    } else {
      layer_input.ShareDataWith(x);
    }
    layer_output.ShareDataWith(hidden_tensor_unbind[i]);
    if (num_layers == 1) {
      layer_x_grad_holder = x_grad;
    } else {
      if (i == num_layers - 1) {
        x_grad_temp.Resize(layer_input.dims());
        dev_ctx.Alloc<T>(&x_grad_temp);
        layer_x_grad_holder = &x_grad_temp;
      }
    }
    if (is_bidirec) {
      BidirGradLayerT<T, GradCellType> layer(cell);
      layer(dev_ctx,
            &layer_input,
            &layer_output,
            &init_h_unbind,
            &init_c_unbind,
            last_h_grad_unbind,
            last_c_grad_unbind,
            gate_tensor_unbind,
            state_tensor_unbind,
            act_state_tensor_unbind,
            layer_output_grad_holder,
            parameter_lists,
            sequence_length.get_ptr(),
            layer_x_grad_holder,
            &init_h_grad_unbind,
            &init_c_grad_unbind,
            &parameter_lists_grad,
            i,
            is_bidirec,
            hidden_size,
            mode,
            gate_num_tmp);
    } else {
      SingleGradLayerT<T, GradCellType> layer(cell);
      layer(dev_ctx,
            &layer_input,
            &layer_output,
            &init_h_unbind,
            &init_c_unbind,
            last_h_grad_unbind,
            last_c_grad_unbind,
            gate_tensor_unbind,
            state_tensor_unbind,
            act_state_tensor_unbind,
            layer_output_grad_holder,
            parameter_lists,
            sequence_length.get_ptr(),
            layer_x_grad_holder,
            &init_h_grad_unbind,
            &init_c_grad_unbind,
            &parameter_lists_grad,
            i,
            is_bidirec,
            hidden_size,
            mode,
            gate_num_tmp);
    }

    // calculate the dropout gradient for the layer_x_grad_holder
    // dropout_state save in the forward process
    if (i > 0) {
      if ((!is_test) && (dropout_prob != 0)) {
        dropout_cpu_grad_function_inplace<T>(
            dev_ctx, layer_x_grad_holder, &dropout_state, dropout_prob);
      }
    }

    if (i - 1 == 0) {
      layer_output_grad_holder = x_grad;
    } else {
      if (!has_allocate_mem) {
        output_grad_temp.Resize(layer_x_grad_holder->dims());
        dev_ctx.Alloc<T>(&output_grad_temp);
        layer_output_grad_holder = &output_grad_temp;
        has_allocate_mem = true;
      }
    }
    SwapPointer(&layer_x_grad_holder, &layer_output_grad_holder);
  }
}

template <typename T, typename Context>
void RnnGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<const DenseTensor*>& pre_state,
                   const std::vector<const DenseTensor*>& weight_list,
                   const paddle::optional<DenseTensor>& sequence_length,
                   const DenseTensor& out,
                   const DenseTensor& dropout_state,
                   const DenseTensor& reserve,
                   const DenseTensor& out_grad,
                   const std::vector<const DenseTensor*>& state_grad,
                   float dropout_prob,
                   bool is_bidirec,
                   int input_size,
                   int hidden_size,
                   int num_layers,
                   const std::string& mode,
                   int seed,
                   bool is_test,
                   DenseTensor* x_grad,
                   std::vector<DenseTensor*> pre_state_grad,
                   std::vector<DenseTensor*> weight_grad_list) {
  int gate_num = 4;
  if (is_lstm(mode)) {
    RnnGradFunc<LSTMGradCell<T>, SingleGradLayer, BidirGradLayer, T>(
        dev_ctx,
        x,
        pre_state,
        weight_list,
        sequence_length,
        out,
        dropout_state,
        reserve,
        out_grad,
        state_grad,
        dropout_prob,
        is_bidirec,
        input_size,
        hidden_size,
        num_layers,
        mode,
        seed,
        is_test,
        gate_num,
        x_grad,
        pre_state_grad,
        weight_grad_list);
  } else if (is_gru(mode)) {
    gate_num = 3;
    RnnGradFunc<GRUGradCell<T>, SingleGradLayer, BidirGradLayer, T>(
        dev_ctx,
        x,
        pre_state,
        weight_list,
        sequence_length,
        out,
        dropout_state,
        reserve,
        out_grad,
        state_grad,
        dropout_prob,
        is_bidirec,
        input_size,
        hidden_size,
        num_layers,
        mode,
        seed,
        is_test,
        gate_num,
        x_grad,
        pre_state_grad,
        weight_grad_list);
    // run gru
  } else if (is_rnn_relu(mode)) {  // NOLINT
    gate_num = 1;
    RnnGradFunc<SimpleRNNGradCell<T, funcs::ReluGradFunctor>,
                SingleGradLayer,
                BidirGradLayer,
                T>(dev_ctx,
                   x,
                   pre_state,
                   weight_list,
                   sequence_length,
                   out,
                   dropout_state,
                   reserve,
                   out_grad,
                   state_grad,
                   dropout_prob,
                   is_bidirec,
                   input_size,
                   hidden_size,
                   num_layers,
                   mode,
                   seed,
                   is_test,
                   gate_num,
                   x_grad,
                   pre_state_grad,
                   weight_grad_list);
    // run rnn
  } else if (is_rnn_tanh(mode)) {
    gate_num = 1;
    RnnGradFunc<SimpleRNNGradCell<T, funcs::TanhGradFunctor>,
                SingleGradLayer,
                BidirGradLayer,
                T>(dev_ctx,
                   x,
                   pre_state,
                   weight_list,
                   sequence_length,
                   out,
                   dropout_state,
                   reserve,
                   out_grad,
                   state_grad,
                   dropout_prob,
                   is_bidirec,
                   input_size,
                   hidden_size,
                   num_layers,
                   mode,
                   seed,
                   is_test,
                   gate_num,
                   x_grad,
                   pre_state_grad,
                   weight_grad_list);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    rnn_grad, CPU, ALL_LAYOUT, phi::RnnGradKernel, float, double) {}
