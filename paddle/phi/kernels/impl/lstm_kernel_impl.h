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
#include <string>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"
#include "paddle/phi/kernels/funcs/lstm_compute.h"
#include "paddle/phi/kernels/funcs/lstm_utils.h"

namespace phi {

template <typename T, typename Context>
void LSTMKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const paddle::optional<DenseTensor>& h0,
                const paddle::optional<DenseTensor>& c0,
                const DenseTensor& weight,
                const DenseTensor& bias,
                bool use_peepholes,
                bool is_reverse,
                bool is_test,
                const std::string& gate_activation,
                const std::string& cell_activation,
                const std::string& candidate_activation,
                DenseTensor* hidden,
                DenseTensor* cell,
                DenseTensor* batch_gate,
                DenseTensor* batch_cell_pre_act) {
  auto* hidden_t0 = h0.get_ptr();
  auto* cell_t0 = c0.get_ptr();

  phi::DenseTensor* batch_gate_new = nullptr;
  phi::DenseTensor batch_gate_temp;
  if (is_test) {
    batch_gate_new = &batch_gate_temp;
    batch_gate_new->Resize(input.dims());
  } else {
    batch_gate_new = batch_gate;
  }

  dev_ctx.template Alloc<T>(batch_gate_new);
  dev_ctx.template Alloc<T>(hidden);
  dev_ctx.template Alloc<T>(cell);

  phi::funcs::DenseTensor2BatchFunctor<Context, T> to_batch;
  to_batch(dev_ctx, input, batch_gate_new, true, is_reverse);

  auto in_dims = input.dims();
  int frame_size = static_cast<int>(in_dims[1] / 4);
  phi::DDim dims({in_dims[0], frame_size});

  if (bias.initialized()) {
    phi::DenseTensor b = bias;
    b.Resize({bias.numel(), 1});
    phi::DenseTensor gate_bias = b.Slice(0, 4 * frame_size);
    phi::funcs::RowwiseAdd<Context, T> add_bias;
    add_bias(dev_ctx, *batch_gate_new, gate_bias, batch_gate_new);
  }

  phi::funcs::LstmMetaValue<T> lstm_value;
  if (bias.initialized() && use_peepholes) {
    T* bias_data = const_cast<T*>(bias.data<T>());
    // the code style in LstmMetaValue will be updated later.

    lstm_value.check_ig = bias_data + 4 * frame_size;
    lstm_value.check_fg = lstm_value.check_ig + frame_size;
    lstm_value.check_og = lstm_value.check_fg + frame_size;
  } else {
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;
  }
  lstm_value.prev_state_value = nullptr;
  phi::DenseTensor ordered_c0;

  phi::Vector<size_t> order(batch_gate_new->lod()[2]);

  if (cell_t0) {
    // Since the batch computing for LSTM reorders the input sequence
    // according to their length. The initialized cell state also needs
    // to reorder.
    ReorderInitState<Context, T>(dev_ctx, *cell_t0, order, &ordered_c0, true);
    lstm_value.prev_state_value = ordered_c0.data<T>();
  }

  // Use the local variable as here.
  phi::DenseTensor batch_hidden, batch_cell, batch_cell_pre_act_temp;
  phi::DenseTensor* batch_cell_pre_act_p;
  if (is_test) {
    batch_cell_pre_act_p = &batch_cell_pre_act_temp;
  } else {
    batch_cell_pre_act_p = batch_cell_pre_act;
  }
  batch_hidden.Resize(dims);
  batch_cell.Resize(dims);
  dev_ctx.template Alloc<T>(&batch_hidden);
  dev_ctx.template Alloc<T>(&batch_cell);
  batch_cell_pre_act_p->Resize(dims);
  dev_ctx.template Alloc<T>(batch_cell_pre_act_p);

  auto batch_starts = batch_gate_new->lod()[0];
  size_t num_batch = batch_starts.size() - 1;
  auto gate_act = phi::funcs::detail::GetActivationType(gate_activation);
  auto cell_act = phi::funcs::detail::GetActivationType(cell_activation);
  auto cand_act = phi::funcs::detail::GetActivationType(candidate_activation);

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  for (size_t n = 0; n < num_batch; n++) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);

    phi::DenseTensor gate_t = batch_gate_new->Slice(bstart, bend);
    phi::DenseTensor out_t = batch_hidden.Slice(bstart, bend);
    phi::DenseTensor cell_t = batch_cell.Slice(bstart, bend);
    phi::DenseTensor cell_pre_act_t = batch_cell_pre_act_p->Slice(bstart, bend);

    int cur_batch_size = bend - bstart;

    if (n > 0) {
      int pre_h_start = static_cast<int>(batch_starts[n - 1]);
      int pre_h_end = pre_h_start + cur_batch_size;
      auto pre_hidden_t = batch_hidden.Slice(pre_h_start, pre_h_end);
      blas.MatMul(pre_hidden_t,
                  false,
                  weight,
                  false,
                  static_cast<T>(1.0),
                  &gate_t,
                  static_cast<T>(1.0));
    } else if (hidden_t0 != nullptr) {
      // If n == 0 and there is no initialized hidden state, that is to say
      // the H0 is zeros, the calculation W_h * H0 will be skipped.
      // If n == 0 and there is initialized hidden state, calculate W_h * H0.

      // Since the batch computing for LSTM reorders the input sequence
      // according to their length. The initialized hidden state also needs
      // to reorder.
      phi::DenseTensor ordered_h0;
      ReorderInitState<Context, T>(
          dev_ctx, *hidden_t0, order, &ordered_h0, true);
      blas.MatMul(ordered_h0,
                  false,
                  weight,
                  false,
                  static_cast<T>(1.0),
                  &gate_t,
                  static_cast<T>(1.0));
    }

    lstm_value.gate_value = gate_t.data<T>();
    lstm_value.output_value = out_t.data<T>();
    lstm_value.state_value = cell_t.data<T>();
    lstm_value.state_active_value = cell_pre_act_t.data<T>();
    T cell_clip = 0.0;
    phi::funcs::LstmUnitFunctor<Context, T>::compute(dev_ctx,
                                                     lstm_value,
                                                     frame_size,
                                                     cur_batch_size,
                                                     cell_clip,
                                                     gate_act,
                                                     cell_act,
                                                     cand_act);
    lstm_value.prev_state_value = lstm_value.state_value;
  }

  phi::funcs::Batch2DenseTensorFunctor<Context, T> to_seq;
  batch_hidden.set_lod(batch_gate_new->lod());
  // restore the output hidden in phi::DenseTensor from the batch hidden
  to_seq(dev_ctx, batch_hidden, hidden);

  batch_cell.set_lod(batch_gate_new->lod());
  // restore the output cell state in phi::DenseTensor from the batch cell
  to_seq(dev_ctx, batch_cell, cell);
}

template <typename T, typename Context>
void LSTMGradKernel(const Context& dev_ctx,
                    const DenseTensor& input_in,
                    const paddle::optional<DenseTensor>& h0_in,
                    const paddle::optional<DenseTensor>& c0_in,
                    const DenseTensor& weight_in,
                    const DenseTensor& bias_in,
                    const DenseTensor& hidden_in,
                    const DenseTensor& cell_in,
                    const DenseTensor& batch_gate_in,
                    const DenseTensor& batch_cell_pre_act_in,
                    const DenseTensor& hidden_grad,
                    bool use_peepholes,
                    bool is_reverse,
                    bool is_test,
                    const std::string& gate_activation,
                    const std::string& cell_activation,
                    const std::string& candidate_activation,
                    DenseTensor* input_grad,
                    DenseTensor* h0_grad,
                    DenseTensor* c0_grad,
                    DenseTensor* weight_grad,
                    DenseTensor* bias_grad) {
  auto* input = &input_in;
  auto* weight = &weight_in;
  auto* bias = &bias_in;

  auto* hidden_out = &hidden_in;
  auto* cell_out = &cell_in;

  auto* batch_gate = &batch_gate_in;
  auto* batch_cell_pre_act = &batch_cell_pre_act_in;

  auto* hidden_g = &hidden_grad;

  auto* in_g = input_grad;
  auto* weight_g = weight_grad;
  auto* bias_g = bias_grad;

  auto* h0 = h0_in.get_ptr();
  auto* c0 = c0_in.get_ptr();

  auto* h0_g = h0_grad;
  auto* c0_g = c0_grad;

  phi::funcs::SetConstant<Context, T> zero;
  if (weight_g) {
    dev_ctx.template Alloc<T>(weight_g);
    zero(dev_ctx, weight_g, static_cast<T>(0.0));
  }

  // ordered_h0/c0 is the reordered hidden/cell initialization.
  // ordered_h0_g/c0_g is the reordered gradient of hidden/cell
  // initialization.
  phi::DenseTensor ordered_h0, ordered_c0, ordered_h0_g, ordered_c0_g;
  phi::Vector<size_t> order(batch_gate->lod()[2]);

  if (c0) {
    ReorderInitState<Context, T>(dev_ctx, *c0, order, &ordered_c0, true);
  }
  if (c0 && c0_g) {
    ordered_c0_g.Resize(c0_g->dims());
    dev_ctx.template Alloc<T>(&ordered_c0_g);
  }

  auto in_dims = input->dims();
  auto out_dims = hidden_g->dims();
  int frame_size = static_cast<int>(in_dims[1] / 4);
  PADDLE_ENFORCE_EQ(frame_size,
                    out_dims[1],
                    common::errors::InvalidArgument(
                        "The second dimension of Input(hidden_grad) should be "
                        "%d, but received %d in LSTM@Grad operator.",
                        frame_size,
                        out_dims[1]));

  phi::funcs::LstmMetaValue<T> lstm_value;
  if (bias && use_peepholes) {
    T* bias_data = const_cast<T*>(bias->data<T>());
    lstm_value.check_ig = bias_data + 4 * frame_size;
    lstm_value.check_fg = lstm_value.check_ig + frame_size;
    lstm_value.check_og = lstm_value.check_fg + frame_size;
  } else {
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;
  }

  phi::funcs::LstmMetaGrad<T> lstm_grad;

  if (bias && bias_g) {
    dev_ctx.template Alloc<T>(bias_g);
    zero(dev_ctx, bias_g, static_cast<T>(0.0));
  }
  if (bias && bias_g && use_peepholes) {
    T* bias_g_data = bias_g->data<T>();
    lstm_grad.check_ig_grad = bias_g_data + 4 * frame_size;
    lstm_grad.check_fg_grad = lstm_grad.check_ig_grad + frame_size;
    lstm_grad.check_og_grad = lstm_grad.check_fg_grad + frame_size;
  } else {
    lstm_grad.check_ig_grad = nullptr;
    lstm_grad.check_fg_grad = nullptr;
    lstm_grad.check_og_grad = nullptr;
  }

  phi::funcs::DenseTensor2BatchFunctor<Context, T> to_batch;

  auto ToBatch = [&batch_gate, &to_batch](const Context& ctx,
                                          const phi::DenseTensor& src,
                                          const phi::DDim& dims,
                                          phi::DenseTensor& dst) {
    dst.Resize(dims);
    ctx.template Alloc<T>(&dst);
    dst.set_lod(batch_gate->lod());
    to_batch(ctx, src, &dst, false);
  };

  phi::DenseTensor batch_hidden, batch_hidden_g, batch_cell;
  ToBatch(dev_ctx, *hidden_out, out_dims, batch_hidden);
  ToBatch(dev_ctx, *hidden_g, out_dims, batch_hidden_g);
  ToBatch(dev_ctx, *cell_out, out_dims, batch_cell);

  phi::DenseTensor batch_cell_g, batch_gate_g;
  batch_cell_g.Resize(out_dims);
  dev_ctx.template Alloc<T>(&batch_cell_g);
  // TODO(qingqing) support the case output cell has gradient.
  // to_batch(dev_ctx, *cell_g, batch_cell_g, false);
  zero(dev_ctx, &batch_cell_g, static_cast<T>(0.0));
  batch_gate_g.Resize(batch_gate->dims());
  dev_ctx.template Alloc<T>(&batch_gate_g);
  batch_gate_g.set_lod(batch_gate->lod());

  auto gate_act = phi::funcs::detail::GetActivationType(gate_activation);
  auto cell_act = phi::funcs::detail::GetActivationType(cell_activation);
  auto cand_act = phi::funcs::detail::GetActivationType(candidate_activation);

  auto batch_starts = batch_gate->lod()[0];
  size_t num_batch = batch_starts.size() - 1;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);

    phi::DenseTensor gate = batch_gate->Slice(bstart, bend);
    phi::DenseTensor cell = batch_cell.Slice(bstart, bend);
    phi::DenseTensor cell_pre_act = batch_cell_pre_act->Slice(bstart, bend);
    lstm_value.gate_value = gate.data<T>();
    lstm_value.state_value = cell.data<T>();
    lstm_value.state_active_value = cell_pre_act.data<T>();

    phi::DenseTensor out_g = batch_hidden_g.Slice(bstart, bend);
    phi::DenseTensor gate_g = batch_gate_g.Slice(bstart, bend);
    phi::DenseTensor cell_g = batch_cell_g.Slice(bstart, bend);
    lstm_grad.state_grad = cell_g.data<T>();
    lstm_grad.gate_grad = gate_g.data<T>();
    lstm_grad.output_grad = out_g.data<T>();

    if (n > 0) {
      int bstart_pre = static_cast<int>(batch_starts[n - 1]);
      phi::DenseTensor cell_pre = batch_cell.Slice(bstart_pre, bstart);
      phi::DenseTensor cell_pre_g = batch_cell_g.Slice(bstart_pre, bstart);
      lstm_value.prev_state_value = cell_pre.data<T>();
      lstm_grad.prev_state_grad = cell_pre_g.data<T>();
    } else {
      lstm_value.prev_state_value = c0 ? ordered_c0.data<T>() : nullptr;
      lstm_grad.prev_state_grad = c0_g ? ordered_c0_g.data<T>() : nullptr;
    }

    // lstm_value.output_value not used in bp, set to nullptr
    // lstm_grad.state_active_grad not used in bp, set to nullptr
    lstm_value.output_value = nullptr;
    lstm_grad.state_active_grad = nullptr;
    int cur_batch_size = bend - bstart;
    T cell_clip = 0.0;
    phi::funcs::LstmUnitGradFunctor<Context, T>::compute(dev_ctx,
                                                         lstm_value,
                                                         lstm_grad,
                                                         frame_size,
                                                         cur_batch_size,
                                                         cell_clip,
                                                         gate_act,
                                                         cell_act,
                                                         cand_act);

    if (n > 0) {
      int pre_h_start = static_cast<int>(batch_starts[n - 1]);
      int pre_h_end = pre_h_start + cur_batch_size;
      auto pre_hidden_g = batch_hidden_g.Slice(pre_h_start, pre_h_end);
      blas.MatMul(gate_g,
                  false,
                  *weight,
                  true,
                  static_cast<T>(1.0),
                  &pre_hidden_g,
                  static_cast<T>(1.0));
      if (weight_g) {
        /* backward weight */
        auto pre_hidden = batch_hidden.Slice(pre_h_start, pre_h_end);
        blas.MatMul(pre_hidden,
                    true,
                    gate_g,
                    false,
                    static_cast<T>(1.0),
                    weight_g,
                    static_cast<T>(1.0));
      }
    } else {
      if (h0 && weight_g) {
        ReorderInitState<Context, T>(dev_ctx, *h0, order, &ordered_h0, true);
        blas.MatMul(ordered_h0,
                    true,
                    gate_g,
                    false,
                    static_cast<T>(1.0),
                    weight_g,
                    static_cast<T>(1.0));
      }
      if (h0 && h0_g) {
        ordered_h0_g.Resize(h0_g->dims());
        dev_ctx.template Alloc<T>(&ordered_h0_g);
        blas.MatMul(gate_g,
                    false,
                    *weight,
                    true,
                    static_cast<T>(1.0),
                    &ordered_h0_g,
                    static_cast<T>(0.0));
      }
    }
  }

  phi::funcs::Batch2DenseTensorFunctor<Context, T> to_seq;
  if (in_g) {
    /* backward data */
    dev_ctx.template Alloc<T>(in_g);
    to_seq(dev_ctx, batch_gate_g, in_g);
  }
  if (bias && bias_g) {
    /* backward bias */
    phi::DenseTensor b_g = *bias_g;
    b_g.Resize({bias_g->numel(), 1});
    phi::DenseTensor gate_bias_g = b_g.Slice(0, 4 * frame_size);
    phi::funcs::ColwiseSum<Context, T> col_sum;
    col_sum(dev_ctx, batch_gate_g, &gate_bias_g);
  }

  if (h0 && h0_g) {
    ReorderInitState<Context, T>(dev_ctx, ordered_h0_g, order, h0_g, false);
  }
  if (c0 && c0_g) {
    ReorderInitState<Context, T>(dev_ctx, ordered_c0_g, order, c0_g, false);
  }
}
}  // namespace phi
