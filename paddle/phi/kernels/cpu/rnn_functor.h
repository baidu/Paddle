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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

#define DEFINE_MODE_DETECTOR(MODE_NAME, MODE_STR)       \
  inline bool is_##MODE_NAME(const std::string& mode) { \
    return mode == #MODE_STR;                           \
  }

DEFINE_MODE_DETECTOR(lstm, LSTM);
DEFINE_MODE_DETECTOR(gru, GRU);
DEFINE_MODE_DETECTOR(rnn_relu, RNN_RELU);
DEFINE_MODE_DETECTOR(rnn_tanh, RNN_TANH);

inline void SwapPointer(DenseTensor** a, DenseTensor** b) {
  DenseTensor* c = *a;
  *a = *b;
  *b = c;
}

template <typename T>
void CreateMaskMatrix(const CPUContext& dev_ctx,
                      const DenseTensor* sequence_length,
                      DenseTensor* mask_matrix,
                      const bool& is_reverse,
                      int* min_seq_len) {
  const auto& seq_len_vec = phi::GetVectorFromTensor<int>(sequence_length);
  const int table_width = mask_matrix->dims()[0];
  DenseTensor temp =
      Empty<T>(dev_ctx, {mask_matrix->dims()[1], mask_matrix->dims()[0]});
  T* data_temp = temp.data<T>();
  std::fill(data_temp, data_temp + mask_matrix->numel(), static_cast<T>(1.0));
  *min_seq_len = table_width;
  for (unsigned int i = 0; i < seq_len_vec.size(); i++) {
    // reset the mask matrix
    *min_seq_len = std::min(seq_len_vec[i], *min_seq_len);
    if (seq_len_vec[i] == table_width) {
      continue;
    }
    if (is_reverse) {
      std::fill(data_temp + i * table_width,
                data_temp + (i + 1) * table_width - seq_len_vec[i],
                static_cast<T>(0));
    } else {
      std::fill(data_temp + i * table_width + seq_len_vec[i],
                data_temp + (i + 1) * table_width,
                static_cast<T>(0));
    }
  }
  dev_ctx.Alloc<T>(mask_matrix);
  std::vector<int> trans_vec;
  trans_vec.emplace_back(1);
  trans_vec.emplace_back(0);
  funcs::TransCompute<CPUContext, T>(2, dev_ctx, temp, mask_matrix, trans_vec);
}

template <typename TensorType>
void ResetParameterVector(const std::vector<TensorType>& raw_params_vec,
                          int num_layers,
                          int gate_num UNUSED,
                          bool is_bidirec,
                          std::vector<std::vector<DenseTensor>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    std::vector<DenseTensor> tensor_list;
    tensor_list.reserve(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      DenseTensor tensor_holder;
      tensor_list.emplace_back(tensor_holder);
    }
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      tensor_list[j].ShareDataWith(*raw_params_vec[tensor_idx]);
    }
    params_vec->emplace_back(tensor_list);
  }
}

template <typename T>
void DropoutHelper(const CPUContext& dev_ctx,
                   DenseTensor* x,
                   DenseTensor* y,
                   const DenseTensor* mask,
                   float dropout_prob) {
  auto& place = *dev_ctx.eigen_device();
  auto dropout_mask = EigenVector<uint8_t>::Flatten(*mask);
  auto in = EigenVector<T>::Flatten(*x);
  auto out = EigenVector<T>::Flatten(*y);
  if (dropout_prob == 1.0f) {
    out.device(place) = static_cast<T>(0) * in;
  } else {
    out.device(place) =
        in * dropout_mask.cast<T>() / static_cast<T>(1.0f - dropout_prob);
  }
}

template <typename T>
void DropoutCpuFunctionInplace(const CPUContext& dev_ctx,
                               DenseTensor* x,
                               DenseTensor* y,
                               DenseTensor* mask,
                               const float& dropout_prob,
                               const int& seed_number,
                               bool is_test,
                               bool* is_has_reset) {
  if (is_test) {
    return;
  }
  size_t size = common::product(x->dims());
  auto* mask_data = mask->data<uint8_t>();
  if (!(*is_has_reset)) {
    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      std::fill(mask_data, mask_data + size, static_cast<uint8_t>(0));
    } else {
      std::shared_ptr<std::mt19937_64> engine;
      if (seed_number) {
        engine = std::make_shared<std::mt19937_64>();
        engine->seed(seed_number);
      } else {
        engine = dev_ctx.GetGenerator()->GetCPUEngine();
      }
      std::uniform_real_distribution<float> dist(0, 1);
      for (size_t i = 0; i < size; ++i) {
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
        } else {
          mask_data[i] = 1;
        }
      }
    }
    *is_has_reset = true;
  }
  DropoutHelper<T>(dev_ctx, x, y, mask, dropout_prob);
}

template <typename Context, typename TensorType>
void SplitReserveData(const Context& dev_ctx UNUSED,
                      int direction_num UNUSED,
                      int time_step UNUSED,
                      int batch_size UNUSED,
                      int hidden_size UNUSED,
                      int gate_num,
                      int num_layers,
                      const std::string& mode,
                      TensorType* reserve_data,
                      DenseTensor* gate_data,
                      DenseTensor* cell_data,
                      DenseTensor* cell_act_data,
                      DenseTensor* hidden_data) {
  int gate_data_idx = gate_num * num_layers;
  int cell_data_idx = (gate_num + 1) * num_layers;
  int cell_act_data_idx = (gate_num + 2) * num_layers;
  // simple rnn
  int hidden_data_start_idx = gate_data_idx;
  *gate_data = reserve_data->Slice(0, gate_data_idx);
  if (is_lstm(mode)) {
    *cell_data = reserve_data->Slice(gate_data_idx, cell_data_idx);
    *cell_act_data = reserve_data->Slice(cell_data_idx, cell_act_data_idx);
    hidden_data_start_idx = cell_act_data_idx;
  } else if (is_gru(mode)) {
    *cell_data = reserve_data->Slice(gate_data_idx, cell_data_idx);
    hidden_data_start_idx = cell_data_idx;
  }
  int hidden_data_idx = hidden_data_start_idx + (num_layers - 1);
  if (num_layers > 1) {
    *hidden_data = reserve_data->Slice(hidden_data_start_idx, hidden_data_idx);
  }
}

template <typename CellType, typename T, typename Context>
void AllocateReserveData(const Context& dev_ctx,
                         bool is_bidirec,
                         int num_layers,
                         int gate_num,
                         int hidden_size,
                         const std::string& mode,
                         DenseTensor* reserve_data,
                         DenseTensor* gate_data,
                         DenseTensor* cell_data,
                         DenseTensor* cell_act_data,
                         DenseTensor* hidden_data,
                         const DenseTensor* input) {
  int direction_num = is_bidirec ? 2 : 1;
  int time_step = input->dims()[0];
  int batch_size = input->dims()[1];
  int block_size = direction_num * time_step * batch_size * hidden_size;
  int hidden_data_idx = (num_layers - 1);
  if (is_lstm(mode)) {
    hidden_data_idx += (gate_num + 2) * num_layers;
  } else if (is_gru(mode)) {
    hidden_data_idx += (gate_num + 1) * num_layers;
  } else {
    hidden_data_idx += gate_num * num_layers;
  }

  reserve_data->Resize({hidden_data_idx, block_size});
  dev_ctx.template Alloc<T>(reserve_data);
  SplitReserveData(dev_ctx,
                   direction_num,
                   time_step,
                   batch_size,
                   hidden_size,
                   gate_num,
                   num_layers,
                   mode,
                   reserve_data,
                   gate_data,
                   cell_data,
                   cell_act_data,
                   hidden_data);
}

inline std::vector<DenseTensor> Unbind(const DenseTensor& in) {
  int64_t size = in.dims()[0];
  std::vector<DenseTensor> tensors;
  tensors.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    tensors.emplace_back(in.Slice(i, i + 1));
  }
  return tensors;
}

template <typename CellType,
          template <typename, typename>
          class LayerT,
          template <typename, typename>
          class SingleLayerT,
          template <typename, typename>
          class BidirLayerT,
          typename T,
          typename Context>
void RnnFunc(const Context& dev_ctx,
             const DenseTensor* input,
             const std::vector<const DenseTensor*>& weight_list,
             const DenseTensor* init_h,
             const DenseTensor* init_c,
             const DenseTensor* sequence_length,
             DenseTensor* last_h,
             DenseTensor* last_c,
             DenseTensor* output,
             DenseTensor* dropout_mask,
             int num_layers,
             int gate_num,
             int input_size UNUSED,
             int hidden_size,
             bool is_bidirec,
             const std::string& cell_type,
             float dropout_prob,
             bool is_test,
             int seed,
             DenseTensor* reserve_data) {
  int direction_num = is_bidirec ? 2 : 1;
  const auto& init_h_dims = init_h->dims();
  PADDLE_ENFORCE_EQ(init_h_dims[0],
                    num_layers * direction_num,
                    common::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of init hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers,
                        init_h_dims[0]));
  if (is_lstm(cell_type)) {
    const auto& init_c_dims = init_c->dims();  // NOLINT
    PADDLE_ENFORCE_EQ(init_c_dims[0],
                      num_layers * direction_num,
                      common::errors::InvalidArgument(
                          "The num_layers of in RNN layer must be the same as "
                          "first dim of cell state hidden, but received"
                          " num_layers:%d, dim:%d",
                          num_layers,
                          init_h_dims[0]));
  }
  CellType cell;

  std::vector<std::vector<DenseTensor>> parameter_lists;
  parameter_lists.reserve(num_layers);
  ResetParameterVector(
      weight_list, num_layers, gate_num, is_bidirec, &parameter_lists);

  DenseTensor gate_data, cell_data, cell_act_data, hidden_data;

  if (!is_test) {
    AllocateReserveData<CellType, T, Context>(dev_ctx,
                                              is_bidirec,
                                              num_layers,
                                              gate_num,
                                              hidden_size,
                                              cell_type,
                                              reserve_data,
                                              &gate_data,
                                              &cell_data,
                                              &cell_act_data,
                                              &hidden_data,
                                              input);
    gate_data.Resize({num_layers, gate_data.numel() / num_layers});
    cell_data.Resize({num_layers, cell_data.numel() / num_layers});
    cell_act_data.Resize({num_layers, cell_act_data.numel() / num_layers});

    if (num_layers > 1) {
      hidden_data.Resize(
          {num_layers - 1, hidden_data.numel() / (num_layers - 1)});
    }
  }

  DenseTensor* input_holder = nullptr;
  DenseTensor* output_holder = output;
  bool has_allocate_mem = false;

  auto init_h_unbind = Unbind(*init_h);
  auto last_h_unbind = Unbind(*last_h);
  std::vector<DenseTensor> init_c_unbind, last_c_unbind;
  if (is_lstm(cell_type)) {
    PADDLE_ENFORCE_NOT_NULL(
        init_c, common::errors::InvalidArgument("init_c contains no data."));
    PADDLE_ENFORCE_NOT_NULL(
        last_c, common::errors::InvalidArgument("last_c contains no data."));
    init_c_unbind = Unbind(*init_c);
    last_c_unbind = Unbind(*last_c);
  }

  DenseTensor curr_gate_data, curr_cell_data, curr_cell_act_data;
  DenseTensor curr_hidden_data, prev_hidden_data;
  DenseTensor temp;
  bool has_dropout_reset = false;
  for (int i = 0; i < num_layers; i++) {
    if (!is_test) {
      if (cell_data.numel() > 0) /** for lstm, gru **/ {
        curr_cell_data = cell_data.Slice(i, i + 1);
      }
      if (cell_act_data.numel() > 0) /*for lstm*/ {
        curr_cell_act_data = cell_act_data.Slice(i, i + 1);
      }
      curr_gate_data = gate_data.Slice(i, i + 1);
      output_holder = output;
      if (i < num_layers - 1 && num_layers > 1) {
        curr_hidden_data = hidden_data.Slice(i, i + 1);
        curr_hidden_data.Resize(output->dims());
        output_holder = &curr_hidden_data;
      }
    }
    if (i > 0) {
      if (!has_allocate_mem) {
        temp.Resize(output->dims());
        dev_ctx.template Alloc<T>(&temp);
        input_holder = &temp;
        has_allocate_mem = true;
      }
      if (!is_test) {
        prev_hidden_data = hidden_data.Slice(i - 1, i);
        input_holder->Resize(output->dims());
        if (dropout_prob != 0) {
          DropoutCpuFunctionInplace<T>(dev_ctx,
                                       &prev_hidden_data,
                                       input_holder,
                                       dropout_mask,
                                       dropout_prob,
                                       seed,
                                       is_test,
                                       &has_dropout_reset);
        } else {
          input_holder = &prev_hidden_data;
          input_holder->Resize(output->dims());
        }
      } else {
        SwapPointer(&output_holder, &input_holder);
      }
    }
    const DenseTensor* input_temp_holder = input;
    if (i > 0) {
      input_temp_holder = input_holder;
    }
    LayerT<T, CellType>* layer;
    SingleLayerT<T, CellType> slayer(cell);
    BidirLayerT<T, CellType> blayer(cell);
    if (is_bidirec) {
      layer = &blayer;
    } else {
      layer = &slayer;
    }
    (*layer)(dev_ctx,
             input_temp_holder,
             parameter_lists[i],
             init_h_unbind,
             init_c_unbind,
             sequence_length,
             last_h_unbind,
             last_c_unbind,
             output_holder,
             i,
             gate_num,
             &curr_gate_data,
             &curr_cell_data,
             &curr_cell_act_data,
             cell_type,
             is_test);
  }
  if (num_layers % 2 == 0) {
    Copy(dev_ctx, *output_holder, dev_ctx.GetPlace(), false, output);
  }
}

}  // namespace phi
