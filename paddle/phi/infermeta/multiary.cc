/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/multiary.h"

#include <vector>

#include "glog/logging.h"

#include "paddle/common/layout.h"
#include "paddle/phi/backends/device_memory_alignment.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {

std::vector<DDim> GetMetaTensorsDim(
    const std::vector<const MetaTensor*>& tensors) {
  std::vector<DDim> dims;
  dims.reserve(tensors.size());
  for (const MetaTensor* tensor : tensors) {
    dims.emplace_back(tensor->dims());
  }
  return dims;
}

void AdadeltaInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& avg_squared_grad,
                       const MetaTensor& avg_squared_update,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float rho,
                       float epsilon,
                       bool multi_precision,
                       MetaTensor* param_out,
                       MetaTensor* avg_squared_grad_out,
                       MetaTensor* avg_squared_update_out,
                       MetaTensor* master_param_out) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(
      common::product(lr_dims),
      1,
      common::errors::InvalidArgument("LearningRate should have one element"));
  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      grad.dims(),
      errors::InvalidArgument(
          "Param and grad input of AdadeltaOp should have same dimension."));
  PADDLE_ENFORCE_EQ(
      param_dims,
      avg_squared_grad.dims(),
      errors::InvalidArgument("Param and AvgSquaredGrad input of AdadeltaOp "
                              "should have same dimension"));
  PADDLE_ENFORCE_EQ(
      param_dims,
      avg_squared_update.dims(),
      errors::InvalidArgument("Param and AvgSquaredUpdate input of AdadeltaOp "
                              "should have same dimension"));
  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(
        param_dims,
        master_param.dims(),
        errors::InvalidArgument("Param and MasterParam input of AdadeltaOp "
                                "should have same dimension"));
  }

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  avg_squared_grad_out->set_dims(param_dims);
  avg_squared_grad_out->set_dtype(avg_squared_grad.dtype());

  avg_squared_update_out->set_dims(param_dims);
  avg_squared_update_out->set_dtype(avg_squared_update.dtype());

  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();
  if (multi_precision && master_param.initialized()) {
    master_param_out->set_dims(param_dims);
    master_param_out->set_dtype(MPType);
  }
}

void AdagradInferMeta(const MetaTensor& param,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& master_param,
                      float epsilon,
                      bool multi_precision,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* master_param_out) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(
      common::product(lr_dims),
      1,
      common::errors::InvalidArgument("LearningRate should have one element"));
  auto param_dims = param.dims();

  PADDLE_ENFORCE_EQ(
      param_dims,
      moment.dims(),
      common::errors::InvalidArgument("Param and Moment input of AdagradOp "
                                      "should have the same dimension."));
  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(
        param_dims,
        master_param.dims(),
        errors::InvalidArgument("Param and MasterParam input of AdadeltaOp "
                                "should have same dimension"));
  }

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());
  moment_out->set_dims(param_dims);
  moment_out->set_dtype(moment.dtype());
  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();
  if (multi_precision && master_param.initialized()) {
    master_param_out->set_dims(param_dims);
    master_param_out->set_dtype(MPType);
  }
}

void AdamInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& moment2_max,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   const Scalar& beta1,
                   const Scalar& beta2,
                   const Scalar& epsilon,
                   bool lazy_mode,
                   int64_t min_row_size_to_use_multithread,
                   bool multi_precision,
                   bool use_global_beta_pow,
                   bool amsgrad,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* moment2_max_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(
      common::product(lr_dims),
      1,
      errors::InvalidArgument(
          "The number of LearningRate shall be 1, but received %d. Maybe "
          "the Input variable LearningRate has not "
          "been initialized. You may need to confirm "
          "if you put exe.run(startup_program) "
          "after optimizer.minimize function.",
          common::product(lr_dims)));
  auto beta1_pow_dims = beta1_pow.dims();
  VLOG(3) << "dims of Beta1Pow : [" << beta1_pow_dims << "]";
  PADDLE_ENFORCE_GE(common::product(beta1_pow_dims),
                    1,
                    errors::InvalidArgument(
                        "The size of Beta1 power accumulator should be greater "
                        "than 0, but received %d.",
                        common::product(beta1_pow_dims)));
  auto beta2_pow_dims = beta2_pow.dims();
  VLOG(3) << "dims of Beta2Pow : [" << beta2_pow_dims << "]";
  PADDLE_ENFORCE_GE(common::product(beta2_pow_dims),
                    1,
                    errors::InvalidArgument(
                        "The size of Beta2 power accumulator should be greater "
                        "than 0, but received %d.",
                        common::product(beta2_pow_dims)));

  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment1.dims(),
      errors::InvalidArgument(
          "Param and Moment1 input of AdamOp should have same dimension. But "
          "received Param dims: [%s], Moment1 dims: [%s].",
          param_dims,
          moment1.dims()));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment2.dims(),
      errors::InvalidArgument(
          "Param and Moment2 input of AdamOp should have same dimension. But "
          "received Param dims: [%s], Moment2 dims: [%s].",
          param_dims,
          moment2.dims()));
  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(
        param_dims,
        master_param.dims(),
        errors::InvalidArgument(
            "Param and Moment1 input of AdamOp should have same dimension. But "
            "received Param dims: [%s], MasterParam dims: [%s].",
            param_dims,
            master_param.dims()));
  }

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  moment1_out->set_dims(param_dims);
  moment1_out->set_dtype(moment1.dtype());
  moment2_out->set_dims(param_dims);
  moment2_out->set_dtype(moment2.dtype());
  if (amsgrad) {
    moment2_max_out->set_dims(param_dims);
    moment2_max_out->set_dtype(moment2.dtype());
  }

  beta1_pow_out->set_dims(beta1_pow_dims);
  beta1_pow_out->set_dtype(beta1_pow.dtype());
  beta2_pow_out->set_dims(beta2_pow_dims);
  beta2_pow_out->set_dtype(beta2_pow.dtype());

  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();
  if (multi_precision && master_param.initialized()) {
    master_param_outs->set_dims(param_dims);
    master_param_outs->set_dtype(MPType);
  }
}

void AdamaxInferMeta(const MetaTensor& param,
                     const MetaTensor& grad,
                     const MetaTensor& learning_rate,
                     const MetaTensor& moment,
                     const MetaTensor& inf_norm,
                     const MetaTensor& beta1_pow,
                     const MetaTensor& master_param,
                     float beta1,
                     float beta2,
                     float epsilon,
                     bool multi_precision,
                     MetaTensor* param_out,
                     MetaTensor* moment_out,
                     MetaTensor* inf_norm_out,
                     MetaTensor* master_param_outs) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(
      product(lr_dims),
      0,
      errors::InvalidArgument("Maybe the Input variable LearningRate has not "
                              "been initialized. You may need to confirm "
                              "if you put exe.run(startup_program) "
                              "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(
      product(lr_dims),
      1,
      errors::InvalidArgument("Learning rate should have 1 dimension"));
  auto beta1_pow_dims = beta1_pow.dims();
  PADDLE_ENFORCE_EQ(product(beta1_pow_dims),
                    1,
                    errors::InvalidArgument(
                        "Beta1 power accumulator should have 1 dimension"));
  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      grad.dims(),
      errors::InvalidArgument(
          "Param and Grad input of AdamaxOp should have same dimension"));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment.dims(),
      errors::InvalidArgument(
          "Param and Moment input of AdamaxOp should have same dimension"));
  PADDLE_ENFORCE_EQ(
      param_dims,
      inf_norm.dims(),
      errors::InvalidArgument(
          "Param and InfNorm input of AdamaxOp should have same dimension"));
  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(
        param_dims,
        master_param.dims(),
        errors::InvalidArgument("Param and MasterParam input of AdamaxOp "
                                "should have same dimension"));
  }

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  moment_out->set_dims(param_dims);
  moment_out->set_dtype(moment.dtype());

  inf_norm_out->set_dims(param_dims);
  inf_norm_out->set_dtype(inf_norm.dtype());

  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();
  if (multi_precision && master_param.initialized()) {
    master_param_outs->set_dims(param_dims);
    master_param_outs->set_dtype(MPType);
  }
}

void AdamwInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& moment2_max,
                    const MetaTensor& beta1_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& master_param,
                    const MetaTensor& skip_update,
                    const Scalar& beta1,
                    const Scalar& beta2,
                    const Scalar& epsilon,
                    float lr_ratio,
                    float coeff,
                    bool with_decay,
                    bool lazy_mode,
                    int64_t min_row_size_to_use_multithread,
                    bool multi_precision,
                    bool use_global_beta_pow,
                    bool amsgrad,
                    MetaTensor* param_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* moment2_max_out,
                    MetaTensor* beta1_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* master_param_outs) {
  AdamInferMeta(param,
                grad,
                learning_rate,
                moment1,
                moment2,
                moment2_max,
                beta1_pow,
                beta2_pow,
                master_param,
                skip_update,
                beta1,
                beta2,
                epsilon,
                lazy_mode,
                min_row_size_to_use_multithread,
                multi_precision,
                use_global_beta_pow,
                amsgrad,
                param_out,
                moment1_out,
                moment2_out,
                moment2_max_out,
                beta1_pow_out,
                beta2_pow_out,
                master_param_outs);
}

void AddNInferMeta(const std::vector<const MetaTensor*>& x,
                   MetaTensor* out,
                   MetaConfig config) {
  auto N = x.size();
  PADDLE_ENFORCE_GT(
      N,
      0,
      common::errors::InvalidArgument(
          "The input tensor X's dimensions of AddNOp "
          "should be larger than 0. But received X's dimensions %d.",
          N));
  if (N == 1) {
    VLOG(3) << "Warning: AddNOp have only one input, may waste memory";
  }
  bool is_all_0d_tensor = true;
  phi::DDim in_dim({0});
  for (size_t i = 0; i < x.size(); ++i) {
    auto x_dim = x[i]->dims();
    // x_dim.size() == 1 means the real dim of selected rows is [0]
    if (x[i]->is_selected_rows() && x_dim.size() == 1) {
      continue;
    }
    // for 0D tensor
    if (x_dim.size() == 0) {
      continue;
    }
    is_all_0d_tensor = false;
    if (common::product(in_dim) == 0) {
      in_dim = x_dim;
    } else {
      if (config.is_runtime) {
        PADDLE_ENFORCE_EQ(in_dim,
                          x_dim,
                          common::errors::InvalidArgument(
                              "The input tensor X of AddNOp must"
                              " have same shape. But received X[0]'s shape = "
                              "[%s], X[%d]'s shape = [%s].",
                              in_dim,
                              i,
                              x_dim));
      } else {
        PADDLE_ENFORCE_EQ(
            in_dim.size(),
            x_dim.size(),
            common::errors::InvalidArgument(
                "The input tensor X of AddNOp must have same "
                "dimensions. But received X[0]'s dimensions = %d, X[0]'s "
                "shape = "
                "[%s], X[%d]'s dimensions = %d, X[%d]'s shape = [%s].",
                in_dim.size(),
                in_dim,
                i,
                x_dim.size(),
                i,
                x_dim));
        // if in_dim or x_dim has -1, not check equal
        for (int j = 0; j < x_dim.size(); ++j) {
          if (x_dim[j] == -1 || in_dim[j] == -1) {
            continue;
          }
          PADDLE_ENFORCE_EQ(
              in_dim[j],
              x_dim[j],
              common::errors::InvalidArgument(
                  "The input tensor X of AddNOp must have same shape "
                  "if not -1."
                  "But received X[0]'s shape = [%s], X[%d]'s shape = [%s].",
                  in_dim,
                  i,
                  x_dim));
        }
      }
    }
  }
  if (is_all_0d_tensor) {
    out->set_dims(common::make_ddim({}));
  } else {
    out->set_dims(in_dim);
  }
  out->share_lod(*x[0]);
  out->set_dtype(x[0]->dtype());
}

// TODO(YuanRisheng) This InferMeta is used in Fluid
//                   and will be deleted in the future.
void AddNTensorArrayInferMeta(const std::vector<const MetaTensor*>& x,
                              MetaTensor* out,
                              MetaConfig config) {
  bool has_tensor_array = false;
  for (auto input : x) {
    if (input->is_tensor_array()) {
      if (out->is_tensor_array()) {
        out->set_dtype(input->dtype());
        out->set_layout(input->layout());
      }
      has_tensor_array = true;
      break;
    }
  }

  if (!has_tensor_array) {
    AddNInferMeta(x, out, config);
  }
}

void ASGDInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& d,
                   const MetaTensor& y,
                   const MetaTensor& n,
                   const MetaTensor& master_param,
                   bool multi_precision,
                   MetaTensor* param_out,
                   MetaTensor* d_out,
                   MetaTensor* y_out,
                   MetaTensor* master_param_out) {
  PADDLE_ENFORCE_NOT_NULL(
      param_out,
      common::errors::InvalidArgument(
          "Output(ParamOut) of ASGDOp should not be null."));

  PADDLE_ENFORCE_NOT_NULL(d_out,
                          common::errors::InvalidArgument(
                              "Output(DOut) of ASGDOp should not be null."));

  PADDLE_ENFORCE_NOT_NULL(y_out,
                          common::errors::InvalidArgument(
                              "Output(YOut) of ASGDOp should not be null."));

  param_out->set_dims(param.dims());
  param_out->set_dtype(param.dtype());
  d_out->set_dims(d.dims());
  d_out->set_dtype(d.dtype());
  y_out->set_dims(y.dims());
  y_out->set_dtype(y.dtype());
  if (multi_precision) {
    master_param_out->set_dims(master_param.dims());
    if (DataType::FLOAT16 == master_param.dtype() ||
        DataType::BFLOAT16 == master_param.dtype()) {
      master_param_out->set_dtype(DataType::FLOAT32);
    } else {
      master_param_out->set_dtype(master_param.dtype());
    }
  }
}

void AttentionLstmInferMeta(const MetaTensor& x,
                            const MetaTensor& c0,
                            const MetaTensor& h0,
                            const MetaTensor& attention_weight,
                            const MetaTensor& attention_bias,
                            const MetaTensor& attention_scalar,
                            const MetaTensor& attention_scalar_bias,
                            const MetaTensor& lstm_weight,
                            const MetaTensor& lstm_bias,
                            const std::string& gate_activation,
                            const std::string& cell_activation,
                            const std::string& candidate_activation,
                            MetaTensor* hidden,
                            MetaTensor* cell,
                            MetaTensor* attentioned_x,
                            MetaTensor* attention_fc_out,
                            MetaTensor* lstm_x,
                            MetaTensor* lstm_out,
                            MetaConfig config) {
  const auto& x_dims = x.dims();
  const int M = static_cast<int>(x_dims[1]);
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Expected input(X)'s dimension is 2. But received %d.",
                        x_dims.size()));

  const auto& w_dims = lstm_weight.dims();
  const int D = static_cast<int>(w_dims[1] / 4);
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      common::errors::InvalidArgument(
          "Expected input(LSTMWeight)'s dimension is 2.But received %d.",
          w_dims.size()));
  PADDLE_ENFORCE_EQ(
      w_dims[0],
      D + M,
      common::errors::InvalidArgument(
          "LSTMWeight dims should be (%d + %d) * %d.", D, M, 4 * D));

  const auto& b_dims = lstm_bias.dims();
  PADDLE_ENFORCE_EQ(
      b_dims.size(),
      2,
      common::errors::InvalidArgument("Input(LSTMBias)'s rank must be 2."));
  PADDLE_ENFORCE_EQ(b_dims[0],
                    1,
                    common::errors::InvalidArgument(
                        "LSTMBias dims should be 1 x %d.", 4 * D));
  PADDLE_ENFORCE_EQ(b_dims[1],
                    4 * D,
                    common::errors::InvalidArgument(
                        "LSTMBias dims should be 1 x %d.", 4 * D));

  const auto& c_dims = c0.dims();
  PADDLE_ENFORCE_EQ(
      c_dims.size(),
      2,
      common::errors::InvalidArgument("Input(C0)'s rank must be 2."));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        c_dims[1],
        D,
        common::errors::InvalidArgument("C0 dims should be N x %d.", D));
  }

  if (h0.initialized()) {
    const auto& h_dims = h0.dims();
    PADDLE_ENFORCE_EQ(
        h_dims.size(),
        2UL,
        common::errors::InvalidArgument(
            "Expected input(H0)'s dimension is 2. But received %d.",
            h_dims.size()));
    if (config.is_runtime ||
        (common::product(c_dims) > 0 && common::product(h_dims) > 0)) {
      PADDLE_ENFORCE_EQ(h_dims,
                        c_dims,
                        common::errors::InvalidArgument(
                            "The dimension of Input(H0) and Input(C0) "
                            "should be the same."));
    }
  }

  const auto& atten_w_dims = attention_weight.dims();
  PADDLE_ENFORCE_EQ(atten_w_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Input(AttentionWeight)'s rank must be 2."));
  PADDLE_ENFORCE_EQ(atten_w_dims[0],
                    M + D,
                    common::errors::InvalidArgument(
                        "Expected `AttentionWeight` shape is [(%d + %d), 1]. "
                        "But received shape = [%d, 1], shape[0] is not %d.",
                        M,
                        D,
                        atten_w_dims[0],
                        M + D));
  PADDLE_ENFORCE_EQ(atten_w_dims[1],
                    1,
                    common::errors::InvalidArgument(
                        "AttentionWeight shapes must be (%d + %d) * 1.", M, D));

  if (attention_bias.initialized()) {
    const auto& atten_b_dims = attention_bias.dims();
    PADDLE_ENFORCE_EQ(atten_b_dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Input(AttentionBias)'s rank must be 2."));
    PADDLE_ENFORCE_EQ(
        atten_b_dims[0],
        1,
        common::errors::InvalidArgument("AttentionBias shapes must be 1 * 1."));
    PADDLE_ENFORCE_EQ(
        atten_b_dims[1],
        1,
        common::errors::InvalidArgument("AttentionBias shapes must be 1 * 1."));
  }

  if (attention_scalar.initialized()) {
    const auto& dims = attention_scalar.dims();
    PADDLE_ENFORCE_EQ(dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Input(AttentionScalar)'s rank must be 2."));
    PADDLE_ENFORCE_EQ(dims[0],
                      1,
                      common::errors::InvalidArgument(
                          "AttentionScalar shapes must be 1 * 1."));
    PADDLE_ENFORCE_EQ(dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "AttentionScalar shapes must be 1 * 1."));
  }

  if (attention_scalar_bias.initialized()) {
    const auto& dims = attention_scalar_bias.dims();
    PADDLE_ENFORCE_EQ(dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Input(AttentionScalarBias)'s rank must be 2."));
    PADDLE_ENFORCE_EQ(dims[0],
                      1,
                      common::errors::InvalidArgument(
                          "AttentionScalarBias shapes must be 1 * 1."));
    PADDLE_ENFORCE_EQ(dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "AttentionScalarBias shapes must be 1 * 1."));
  }

  phi::DDim out_dims({x_dims[0], D});
  hidden->set_dims(out_dims);
  cell->set_dims(out_dims);
  attentioned_x->set_dims({x_dims[0], 1});
  lstm_x->set_dims({1, M});
  lstm_out->set_dims({1, 4 * D});
  // AttentionFCOut should be reshape as (maxseqlen,1) in runtime
  hidden->share_lod(x);
  cell->share_lod(x);
  hidden->set_dtype(x.dtype());
  cell->set_dtype(x.dtype());
}

void AucInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& stat_pos,
                  const MetaTensor& stat_neg,
                  const MetaTensor& ins_tag_weight,
                  const std::string& curve,
                  int num_thresholds,
                  int slide_steps,
                  MetaTensor* auc,
                  MetaTensor* stat_pos_out,
                  MetaTensor* stat_neg_out,
                  MetaConfig config) {
  auto predict_dims = input.dims();
  auto label_dims = label.dims();
  PADDLE_ENFORCE_GE(
      predict_dims.size(),
      2,
      common::errors::InvalidArgument(
          "The Input(Predict) has not been initialized properly. The "
          "shape of Input(Predict) = [%s], the shape size must be "
          "greater_equal 2.",
          predict_dims));
  auto predict_width = predict_dims[1];
  PADDLE_ENFORCE_NE(
      common::product(predict_dims),
      0,
      common::errors::InvalidArgument(
          "The Input(Predict) has not been initialized properly. The "
          "shape of Input(Predict) = [%s], the shape can not involes 0.",
          predict_dims));
  PADDLE_ENFORCE_NE(
      common::product(label_dims),
      0,
      common::errors::InvalidArgument(
          "The Input(Label) has not been initialized properly. The "
          "shape of Input(Label) = [%s], the shape can not involes 0.",
          label_dims));

  if (config.is_runtime) {
    PADDLE_ENFORCE_LE(
        predict_width,
        2,
        common::errors::InvalidArgument("Only support binary classification,"
                                        "prediction dims[1] should be 1 or 2"));
  }
  auto predict_height = input.dims()[0];
  auto label_height = label.dims()[0];

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(predict_height,
                      label_height,
                      common::errors::InvalidArgument(
                          "Out and Label should have same height."));
  }

  int num_pred_buckets = num_thresholds + 1;

  PADDLE_ENFORCE_GE(
      num_pred_buckets,
      1,
      common::errors::InvalidArgument("num_thresholds must larger than 1"));
  PADDLE_ENFORCE_GE(
      slide_steps,
      0,
      common::errors::InvalidArgument("slide_steps must be natural number"));

  auc->set_dims(common::make_ddim({}));
  auc->set_dtype(DataType::INT64);

  if (slide_steps) {
    stat_pos_out->set_dims({(1 + slide_steps) * num_pred_buckets + 1});
    stat_pos_out->set_dtype(DataType::INT64);
    stat_neg_out->set_dims({(1 + slide_steps) * num_pred_buckets + 1});
    stat_neg_out->set_dtype(DataType::INT64);
  } else {
    stat_pos_out->set_dims({1, num_pred_buckets});
    stat_pos_out->set_dtype(DataType::INT64);
    stat_neg_out->set_dims({1, num_pred_buckets});
    stat_neg_out->set_dtype(DataType::INT64);
  }
}

void AverageAccumulatesInferMeta(const MetaTensor& param,
                                 const MetaTensor& in_sum_1,
                                 const MetaTensor& in_sum_2,
                                 const MetaTensor& in_sum_3,
                                 const MetaTensor& in_num_accumulates,
                                 const MetaTensor& in_old_num_accumulates,
                                 const MetaTensor& in_num_updates,
                                 float average_window,
                                 int64_t max_average_window,
                                 int64_t min_average_window,
                                 MetaTensor* out_sum_1,
                                 MetaTensor* out_sum_2,
                                 MetaTensor* out_sum_3,
                                 MetaTensor* out_num_accumulates,
                                 MetaTensor* out_old_num_accumulates,
                                 MetaTensor* out_num_updates) {
  // auto in_dim = param.dims;
  PADDLE_ENFORCE_NE(
      out_sum_1,
      nullptr,
      errors::NotFound(
          "Output(out_sum_1) of AverageAccumulates should not be null."));
  PADDLE_ENFORCE_NE(
      out_sum_2,
      nullptr,
      errors::NotFound(
          "Output(out_sum_2) of AverageAccumulates should not be null."));
  PADDLE_ENFORCE_NE(
      out_sum_3,
      nullptr,
      errors::NotFound(
          "Output(out_sum_3) of AverageAccumulates should not be null."));
  PADDLE_ENFORCE_NE(out_num_accumulates,
                    nullptr,
                    errors::NotFound("Output(out_num_accumulates) of "
                                     "AverageAccumulates should not be null."));

  PADDLE_ENFORCE_NE(out_old_num_accumulates,
                    nullptr,
                    errors::NotFound("Output(out_old_num_accumulates) of "
                                     "AverageAccumulates should not be null."));

  PADDLE_ENFORCE_NE(
      out_num_updates,
      nullptr,
      errors::NotFound(
          "Output(out_num_updates) of AverageAccumulates should not be null."));

  out_sum_1->set_dims(in_sum_1.dims());
  out_sum_1->set_dtype(in_sum_1.dtype());
  out_sum_2->set_dims(in_sum_2.dims());
  out_sum_2->set_dtype(in_sum_2.dtype());
  out_sum_3->set_dims(in_sum_3.dims());
  out_sum_3->set_dtype(in_sum_3.dtype());
  out_num_accumulates->set_dims({1});
  out_num_accumulates->set_dtype(in_num_accumulates.dtype());
  out_old_num_accumulates->set_dims({1});
  out_old_num_accumulates->set_dtype(in_old_num_accumulates.dtype());
  out_num_updates->set_dims({1});
  out_num_updates->set_dtype(in_num_updates.dtype());
}

void BatchNormInferMeta(const MetaTensor& x,
                        const MetaTensor& mean,
                        const MetaTensor& variance,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        bool is_test,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout_str,
                        bool use_global_stats,
                        bool trainable_statistics,
                        MetaTensor* y,
                        MetaTensor* mean_out,
                        MetaTensor* variance_out,
                        MetaTensor* saved_mean,
                        MetaTensor* saved_variance,
                        MetaTensor* reserve_space,
                        MetaConfig config) {
  const auto x_dims = x.dims();
  for (int i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_EQ(
        (x_dims[i] == -1) || (x_dims[i] > 0),
        true,
        common::errors::InvalidArgument(
            "Each dimension of input tensor is expected to be -1 or a "
            "positive number, but received %d. Input's shape is [%s].",
            x_dims[i],
            x_dims));
  }

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of input "
          "X must greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      common::errors::InvalidArgument(
          "ShapeError: the dimension of input X "
          "must smaller than or equal to 5. But received: the shape of input X "
          "= [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));

  const int64_t C = ((config.is_run_mkldnn_kernel == true) ||
                             (data_layout == DataLayout::kNCHW)
                         ? x_dims[1]
                         : x_dims[x_dims.size() - 1]);
  if (scale) {
    PADDLE_ENFORCE_EQ(
        scale.dims().size(),
        1UL,
        common::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale.dims().size(),
            scale.dims().size()));
  }

  if (bias) {
    PADDLE_ENFORCE_EQ(
        bias.dims().size(),
        1UL,
        common::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias.dims(),
            bias.dims().size()));
  }

  bool check = true;

  if (!scale || !bias ||
      ((!config.is_runtime) && (contain_unknown_dim(scale.dims()) ||
                                contain_unknown_dim(bias.dims()) || C == -1))) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(scale.dims()[0],
                      C,
                      common::errors::InvalidArgument(
                          "ShapeError: the shape of scale must equal to [%d]"
                          "But received: the shape of scale is [%d]",
                          C,
                          scale.dims()[0]));
    PADDLE_ENFORCE_EQ(bias.dims()[0],
                      C,
                      common::errors::InvalidArgument(
                          "ShapeError: the shape of bias must equal to [%d]"
                          "But received: the shape of bias is [%d]",
                          C,
                          bias.dims()[0]));
  }
  auto dtype = x.dtype();
  if (dtype == phi::DataType::FLOAT16 || dtype == phi::DataType::BFLOAT16 ||
      dtype == phi::DataType::UINT16) {
    dtype = phi::DataType::FLOAT32;
  }

  y->set_dims(x_dims);
  mean_out->set_dims({C});
  mean_out->set_dtype(mean.dtype());
  variance_out->set_dims({C});
  variance_out->set_dtype(variance.dtype());
  if (saved_mean) {
    saved_mean->set_dims({C});
    saved_mean->set_dtype(dtype);
  }
  if (saved_variance) {
    saved_variance->set_dims({C});
    saved_variance->set_dtype(dtype);
  }
  if (reserve_space) {
    reserve_space->set_dims({-1});
    reserve_space->set_dtype(DataType::UINT8);
  }
  y->share_lod(x);
  y->set_dtype(x.dtype());
  y->set_layout(x.layout());
}

void BatchNormInferInferMeta(const MetaTensor& x,
                             const MetaTensor& mean,
                             const MetaTensor& variance,
                             const MetaTensor& scale,
                             const MetaTensor& bias,
                             float momentum,
                             float epsilon,
                             const std::string& data_layout,
                             MetaTensor* y,
                             MetaTensor* mean_out,
                             MetaTensor* variance_out,
                             MetaConfig config) {
  BatchNormInferMeta(x,
                     mean,
                     variance,
                     scale,
                     bias,
                     /*is_test=*/true,
                     momentum,
                     epsilon,
                     data_layout,
                     /*use_global_stats=*/false,
                     /*trainable_statistics=*/false,
                     y,
                     mean_out,
                     variance_out,
                     /*saved_mean=*/nullptr,
                     /*saved_variance=*/nullptr,
                     /*reserve_space=*/nullptr,
                     config);
}

void BilinearInferMeta(const MetaTensor& x,
                       const MetaTensor& y,
                       const MetaTensor& weight,
                       const MetaTensor& bias,
                       MetaTensor* out,
                       MetaConfig config) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto weight_dims = weight.dims();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2UL,
      errors::InvalidArgument("The input(X) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      2UL,
      errors::InvalidArgument("The input(Y) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      weight_dims.size(),
      3UL,
      errors::InvalidArgument(
          "Expected the input(Weight) is a 3D tensor. But received %dD tensor.",
          weight_dims.size()));
  if (config.is_runtime || (x_dims[0] > 0 && y_dims[0] > 0)) {
    PADDLE_ENFORCE_EQ(x_dims[0],
                      y_dims[0],
                      errors::InvalidArgument(
                          "The first dimension(batch_size) of input(X) must be "
                          "equal to the first dimension of the input(Y)."));
  }
  PADDLE_ENFORCE_EQ(x_dims[1],
                    weight_dims[1],
                    errors::InvalidArgument(
                        "The second dimension of input(X) must be equal to "
                        "the second dimension of the input(Weight)."));
  PADDLE_ENFORCE_EQ(y_dims[1],
                    weight_dims[2],
                    errors::InvalidArgument(
                        "The second dimension of input(Y) must be equal to "
                        "the third dimension of the input(Weight)."));

  if (bias) {
    auto bias_dims = bias.dims();
    PADDLE_ENFORCE_EQ(bias_dims.size(),
                      2UL,
                      errors::InvalidArgument(
                          "The Input(Bias) must be a 2-D tensor with "
                          "the 2nd dimension fixed to 1 (a row vector)."));
    PADDLE_ENFORCE_EQ(bias_dims[0],
                      1UL,
                      errors::InvalidArgument(
                          "The Input(Bias) must be a 2-D tensor with "
                          "the 2nd dimension fixed to 1 (a row vector)."));
    PADDLE_ENFORCE_EQ(bias_dims[1],
                      weight_dims[0],
                      errors::InvalidArgument(
                          "The second dimension of input(Bias) must be equal "
                          "to the first dimension of the input(Weight)."));
  }

  out->set_dims({x_dims[0], weight_dims[0]});
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void BeamSearchInferMeta(const MetaTensor& pre_ids,
                         const MetaTensor& pre_scores,
                         const MetaTensor& ids,
                         const MetaTensor& scores,
                         int level,
                         int beam_size,
                         int end_id,
                         bool is_accumulated,
                         MetaTensor* selected_ids,
                         MetaTensor* selected_scores,
                         MetaTensor* parent_idx) {
  const auto& id_dims = pre_ids.dims();
  selected_scores->set_dims(pre_scores.dims());
  selected_ids->set_dims(id_dims);
  parent_idx->set_dims({id_dims[0]});
  selected_scores->set_dtype(pre_scores.dtype());
  selected_ids->set_dtype(pre_ids.dtype());
  parent_idx->set_dtype(pre_ids.dtype());
}

void BroadcastTensorsInferMeta(const std::vector<const MetaTensor*>& x,
                               std::vector<MetaTensor*> out) {
  int target_rank = 0;
  const auto& input_dims = GetMetaTensorsDim(x);

  // 1. Find Output rank = max(Inputs rank)
  for (const auto& input_ddim : input_dims) {
    target_rank = std::max(target_rank, input_ddim.size());
  }

  std::vector<int64_t> target_dims(target_rank, 0);
  // 2. Output dim(axis=x) = max(Inputs dim(axis=x))
  for (int index = 0; index < target_rank; index++) {
    // Loop axes in reverse order,
    // For each axis, take the maximum as target size
    // Fill size = 1 if shape vector exhausts
    int target_dim_size = 1;
    for (const auto& input_ddim : input_dims) {
      // Reversed order
      int axis = static_cast<int>(input_ddim.size()) - index - 1;
      int dim_size = 1;
      if (axis >= 0) {
        dim_size = static_cast<int>(input_ddim[axis]);
      }

      if (target_dim_size != 1 && dim_size != 1 &&
          target_dim_size != dim_size) {
        PADDLE_THROW(errors::InvalidArgument(
            "BroadcastTensorsOp inputs does not satisfy bcast semantics, "
            "please check axis = %d in reverse order",
            index));
      }

      // We performed bcast semantics check at python level
      // So input tensors should all have legal shape
      target_dim_size = dim_size == 1 ? target_dim_size : dim_size;
    }
    target_dims[target_rank - index - 1] = target_dim_size;
  }

  // 3. Set Output Dim
  for (size_t i = 0; i < out.size(); i++) {
    out[i]->set_dims(common::make_ddim(target_dims));
    out[i]->share_lod(*(x[i]));
    out[i]->set_dtype(x[i]->dtype());
  }
}

void CheckFiniteAndUnscaleInferMeta(const std::vector<const MetaTensor*>& xs,
                                    const MetaTensor& scale,
                                    std::vector<MetaTensor*> outs,
                                    MetaTensor* found_infinite) {
  PADDLE_ENFORCE_EQ(
      xs.size(),
      outs.size(),
      common::errors::InvalidArgument(
          "The input(X) and output(Out) should have same size in "
          "Operator(check_finite_and_unscale), size of input(X) is %d "
          "and size of output(Out) is %d.",
          xs.size(),
          outs.size()));
  for (size_t i = 0; i < xs.size(); ++i) {
    outs[i]->set_dims(xs[i]->dims());
    outs[i]->set_dtype(xs[i]->dtype());
  }
  found_infinite->set_dims({1});
  found_infinite->set_dtype(DataType::BOOL);
}

void CoalesceTensorInferMeta(const std::vector<const MetaTensor*>& input,
                             DataType dtype,
                             bool copy_data,
                             bool set_constant,
                             bool persist_output,
                             float constant,
                             bool use_align,
                             int align_size,
                             int size_of_dtype,
                             const std::vector<int64_t>& concated_shapes,
                             const std::vector<int64_t>& concated_ranks,
                             std::vector<MetaTensor*> output,
                             MetaTensor* fused_output,
                             MetaConfig config) {
  if (size_of_dtype == -1) {
    size_of_dtype = static_cast<int>(phi::SizeOf(dtype));
  }
  PADDLE_ENFORCE_EQ(
      input.size(),
      output.size(),
      common::errors::InvalidArgument(
          "The size of output meta vector should be equal to input"));
  for (size_t idx = 0; idx < input.size(); ++idx) {
    output[idx]->set_dims(input[idx]->dims());
    output[idx]->set_dtype(input[idx]->dtype());
    output[idx]->set_layout(input[idx]->layout());
  }
  if (config.is_runtime) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    int64_t numel = 0;
    for (auto item : input) {
      const auto& dim = item->dims();
      auto size = common::product(dim);
      auto len = use_align
                     ? phi::Alignment(static_cast<size_t>(size) * size_of_dtype,
                                      phi::GPUPlace(),
                                      align_size) /
                           size_of_dtype
                     : static_cast<size_t>(size);
      numel += len;
    }
    if (fused_output) {
      fused_output->set_dims(common::make_ddim({numel}));
      fused_output->set_dtype(dtype);
      VLOG(4) << "fused_output size:" << common::make_ddim({numel});
    }
#else
    return;
#endif
  } else {
    auto alignment = [](size_t size, size_t align_size) {
      size_t remaining = size % align_size;
      auto aligned_size =
          remaining == 0 ? size : size + (align_size - remaining);
      VLOG(4) << remaining << " " << size << " " << align_size << " "
              << aligned_size;
      return aligned_size;
    };
    VLOG(4) << "align_size: " << align_size;
    if (use_align && align_size > 0) {
      int64_t numel = 0;

      for (auto item : input) {
        const auto& dim = item->dims();
        auto size = common::product(dim);
        auto len = use_align
                       ? alignment(static_cast<size_t>(size) * size_of_dtype,
                                   align_size) /
                             size_of_dtype
                       : static_cast<size_t>(size);
        numel += static_cast<int64_t>(len);
      }
      if (fused_output) {
        fused_output->set_dims(common::make_ddim({numel}));
        fused_output->set_dtype(dtype);
        VLOG(4) << "fused_output size:" << common::make_ddim({numel});
      }
    }
  }
}

void CheckMemoryContinueInferMeta(const std::vector<const MetaTensor*>& input,
                                  MetaTensor* output,
                                  std::vector<MetaTensor*> xout,
                                  MetaConfig config) {
  if (config.is_runtime) {
    return;
  }
  int64_t numel = 0;
  for (auto item : input) {
    const auto& dim = item->dims();
    auto size = common::product(dim);
    auto len = size * phi::SizeOf(item->dtype());
    numel += static_cast<int64_t>(len);
  }
  output->set_dims(common::make_ddim({numel}));
  output->set_dtype(phi::DataType::INT8);
}

void ConcatInferMeta(const std::vector<const MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_GE(x.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The size of input meta vector should be greater"
                        "than 0."));
  if (axis_scalar.FromTensor() && !config.is_runtime) {
    auto out_dims =
        common::make_ddim(std::vector<int>(x.at(0)->dims().size(), -1));
    out->set_dims(out_dims);
    out->set_dtype(x.at(0)->dtype());
    out->set_layout(x.at(0)->layout());
    out->share_lod(*x.at(0));
    return;
  }

  int axis = axis_scalar.to<int>();
  // 1. calculate axis
  int rank = x.at(0)->dims().size();
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      common::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }

  // 2. calculate out dims
  std::vector<phi::DDim> x_dims;
  x_dims.reserve(x.size());
  for (const auto* x_t : x) {
    x_dims.emplace_back(x_t->dims());
  }
  phi::DDim out_dim =
      phi::funcs::ComputeAndCheckShape(config.is_runtime, x_dims, axis);

  out->set_dims(out_dim);
  out->set_dtype(x.at(0)->dtype());
  out->set_layout(x.at(0)->layout());
  out->share_lod(*x.at(0));
}

void ChunkEvalInferMeta(const MetaTensor& inference,
                        const MetaTensor& label,
                        const MetaTensor& seq_length,
                        int num_chunk_types,
                        const std::string& chunk_scheme,
                        const std::vector<int>& excluded_chunk_types,
                        MetaTensor* precision,
                        MetaTensor* recall,
                        MetaTensor* f1_score,
                        MetaTensor* num_infer_chunks,
                        MetaTensor* num_label_chunks,
                        MetaTensor* num_correct_chunks) {
  const auto& inference_dim = inference.dims();
  const auto& label_dim = label.dims();

  PADDLE_ENFORCE_EQ(
      inference_dim,
      label_dim,
      common::errors::InvalidArgument(
          "Input(Inference)'s shape must be the same as Input(Label)'s "
          "shape, but received [%s] (Inference) vs [%s] (Label).",
          inference_dim,
          label_dim));

  bool use_padding = seq_length.initialized();
  if (use_padding) {
    PADDLE_ENFORCE_EQ((inference_dim.size() == 3 && inference_dim[2] == 1) ||
                          inference_dim.size() == 2,
                      true,
                      common::errors::InvalidArgument(
                          "when Input(SeqLength) is provided, Input(Inference) "
                          "should be of dim 3 (batch_size, bucket, 1) or dim 2 "
                          "(batch_size, bucket), but received [%s].",
                          inference_dim));
    auto seq_length_dim = seq_length.dims();
    PADDLE_ENFORCE_LE(seq_length_dim.size(),
                      2,
                      common::errors::InvalidArgument(
                          "Input(SeqLength)'s rank should not be greater "
                          "than 2, but received %d.",
                          seq_length_dim.size()));
  }

  precision->set_dims({1});
  recall->set_dims({1});
  f1_score->set_dims({1});
  num_infer_chunks->set_dims({1});
  num_label_chunks->set_dims({1});
  num_correct_chunks->set_dims({1});

  precision->set_dtype(phi::DataType::FLOAT32);
  recall->set_dtype(phi::DataType::FLOAT32);
  f1_score->set_dtype(phi::DataType::FLOAT32);
  num_infer_chunks->set_dtype(phi::DataType::INT64);
  num_label_chunks->set_dtype(phi::DataType::INT64);
  num_correct_chunks->set_dtype(phi::DataType::INT64);
}

void CrfDecodingInferMeta(const MetaTensor& emission,
                          const MetaTensor& transition,
                          const MetaTensor& label,
                          const MetaTensor& length,
                          MetaTensor* viterbi_path,
                          MetaConfig config) {
  auto emission_dims = emission.dims();
  bool has_length = length.initialized();

  if (has_length) {
    PADDLE_ENFORCE_EQ(emission_dims.size(),
                      3,
                      common::errors::InvalidArgument(
                          "The Input(Emission) should be a 3-D tensor. But "
                          "received: input rank %u, input shape [%s]. ",
                          emission_dims.size(),
                          emission_dims));
  } else {
    PADDLE_ENFORCE_EQ(emission_dims.size(),
                      2,
                      common::errors::InvalidArgument(
                          "The Input(Emission) should be a 2-D tensor. But "
                          "received: input rank %u, input shape [%s].",
                          emission_dims.size(),
                          emission_dims));
  }

  auto transition_dims = transition.dims();
  PADDLE_ENFORCE_EQ(transition_dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "The Input(Transition) should be a 2-D tensor. But "
                        "received: input rank %u, input shape [%s].",
                        transition_dims.size(),
                        transition_dims));
  PADDLE_ENFORCE_EQ(
      transition_dims[0] - 2,
      transition_dims[1],
      common::errors::InvalidArgument(
          "An invalid dimension for the Input(Transition), which should "
          "be a 2-D tensor with shape [(D + 2) x D]. But received: input "
          "rank %u, "
          "input shape [%s].",
          transition_dims.size(),
          transition_dims));
  if (config.is_runtime || (emission_dims[emission_dims.size() - 1] > 0 &&
                            transition_dims[transition_dims.size() - 1] > 0)) {
    PADDLE_ENFORCE_EQ(emission_dims[emission_dims.size() - 1],
                      transition_dims[transition_dims.size() - 1],
                      common::errors::InvalidArgument(
                          "The last dimension of the Input(Emission) and the "
                          "Input(Transition) "
                          "should be equal to the tag number. But received "
                          "Input(Emission): rank "
                          "%u, shape [%s]; received Input(Transition): rank "
                          "%u, shape [%s].",
                          emission_dims.size(),
                          emission_dims,
                          transition_dims.size(),
                          transition_dims));
  }
  if (label.initialized()) {
    auto label_dims = label.dims();
    if (length.initialized()) {
      PADDLE_ENFORCE_EQ(
          (label_dims.size() == 3UL && label_dims[2] == 1) ||
              label_dims.size() == 2UL,
          true,
          common::errors::InvalidArgument(
              "The Input(Label) should be a 3-D tensor with last dimension "
              "fixed to 1 or a 2-D tensor in padding mode. But received: "
              "input "
              "rank %u, input shape [%s].",
              label_dims.size(),
              label_dims));
    } else {
      PADDLE_ENFORCE_EQ(
          (label_dims.size() == 2UL && label_dims[1] == 1) ||
              label_dims.size() == 1UL,
          true,
          common::errors::InvalidArgument(
              "The Input(Label) should be a 2-D tensor with last "
              "dimension fixed to 1 or a 1-D tensor. But received: "
              "input rank %u, input shape [%s].",
              label_dims.size(),
              label_dims));
    }
    if (config.is_runtime || (emission_dims[0] > 0 && label_dims[0] > 0)) {
      PADDLE_ENFORCE_EQ(
          emission_dims[0],
          label_dims[0],
          common::errors::InvalidArgument(
              "The first dimension of Input(Emission) and Input(Label) "
              "should be the same. But received Input(Emission): rank %u, "
              "shape [%s]; received Input(Label): rank %u, shape [%s].",
              emission_dims.size(),
              emission_dims,
              label_dims.size(),
              label_dims));
    }
  }

  viterbi_path->share_lod(emission);
  if (has_length) {
    viterbi_path->set_dims({emission_dims[0], emission_dims[1]});
  } else {
    viterbi_path->set_dims({emission_dims[0], 1});
  }
  viterbi_path->set_dtype(emission.dtype());
}

void CudnnLSTMInferMeta(
    const MetaTensor& x,
    const MetaTensor& init_h,
    const MetaTensor& init_c,
    const MetaTensor& w,
    const paddle::optional<std::vector<const MetaTensor*>>& weight_list,
    const MetaTensor& sequence_length,
    float dropout_prob,
    bool is_bidirec,
    int hidden_size,
    int num_layers,
    bool is_test,
    int seed,
    MetaTensor* out,
    MetaTensor* last_h,
    MetaTensor* last_c,
    MetaTensor* reserve,
    MetaTensor* state_out) {
  auto in_dims = x.dims();
  auto init_h_dims = init_h.dims();

  auto init_c_dims = init_c.dims();

  PADDLE_ENFORCE_EQ(in_dims.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of Input in CudnnLSTM  must be 3. But "
                        "received Input's rank is %d.",
                        in_dims.size()));
  PADDLE_ENFORCE_EQ(init_h_dims.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of InitH in CudnnLSTM  must be 3. But "
                        "received InitH's rank is %d.",
                        init_h_dims.size()));

  if (sequence_length) {
    auto seq_dims = sequence_length.dims();
    PADDLE_ENFORCE_EQ(
        in_dims[1],
        seq_dims[0],
        common::errors::InvalidArgument(
            "The size of SequenceLength has to equal the batch_size. But "
            "received batch_size is %d and the size of SequenceLength is %d.",
            in_dims[1],
            seq_dims[0]));
  }

  PADDLE_ENFORCE_EQ(in_dims[1],
                    init_h_dims[1],
                    common::errors::InvalidArgument(
                        "The in_dims[1] (Input dims) and init_h_dims[1] (InitH "
                        "dims) should be equal. But "
                        "received in_dims[1] is %d and init_h_dims[1] is %d.",
                        in_dims[1],
                        init_h_dims[1]));

  PADDLE_ENFORCE_EQ(init_c_dims,
                    init_h_dims,
                    common::errors::InvalidArgument(
                        "The InitC dims and InitH "
                        "dims should be equal. But "
                        "received init_c_dims is %d and init_h_dims is %d.",
                        init_c_dims,
                        init_h_dims));

  auto out_dims = in_dims;
  out_dims[2] = is_bidirec ? hidden_size * 2 : hidden_size;
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  last_h->set_dims(init_c_dims);
  last_h->set_dtype(x.dtype());
  last_c->set_dims(init_h_dims);
  last_c->set_dtype(x.dtype());

  reserve->set_dtype(phi::DataType::UINT8);
  state_out->set_dtype(phi::DataType::UINT8);
}

void LSTMInferMeta(const MetaTensor& input,
                   const MetaTensor& h0,
                   const MetaTensor& c0,
                   const MetaTensor& weight,
                   const MetaTensor& bias,
                   bool use_peepholes,
                   bool is_reverse,
                   bool is_test,
                   const std::string& gate_activation,
                   const std::string& cell_activation,
                   const std::string& candidate_activation,
                   MetaTensor* hidden,
                   MetaTensor* cell,
                   MetaTensor* batch_gate,
                   MetaTensor* batch_cell_pre_act,
                   MetaConfig config) {
  const auto& in_dims = input.dims();
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      2,
      common::errors::InvalidArgument(
          "Input(X)'s rank must be 2, but received %d.", in_dims.size()));

  if (h0) {
    PADDLE_ENFORCE_EQ(
        c0.initialized(),
        true,
        common::errors::NotFound("Input(Cell) and Input(Hidden) of LSTM "
                                 "should not be null at the same time."));
    const auto& h_dims = h0.dims();
    const auto& c_dims = c0.dims();
    PADDLE_ENFORCE_EQ(h_dims,
                      c_dims,
                      common::errors::InvalidArgument(
                          "The dimension of Input(H0) and Input(C0) should "
                          "be the same, but received [%s] (H0) vs [%s] (C0).",
                          h_dims,
                          c_dims));
  }

  int frame_size = static_cast<int>(in_dims[1] / 4);
  const auto& w_dims = weight.dims();
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      common::errors::InvalidArgument(
          "The rank of Input(Weight) should be 2, but received %d.",
          w_dims.size()));
  PADDLE_ENFORCE_EQ(w_dims[0],
                    frame_size,
                    common::errors::InvalidArgument(
                        "The first dimension of Input(Weight) should be %d, "
                        "but received %d.",
                        frame_size,
                        w_dims[0]));
  PADDLE_ENFORCE_EQ(w_dims[1],
                    4 * frame_size,
                    common::errors::InvalidArgument(
                        "The second dimension of Input(Weight) should be 4 * "
                        "%d, but received %d.",
                        frame_size,
                        w_dims[1]));

  const auto& b_dims = bias.dims();
  PADDLE_ENFORCE_EQ(b_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input(Bias) should be 2, but received %d.",
                        b_dims.size()));
  PADDLE_ENFORCE_EQ(
      b_dims[0],
      1,
      common::errors::InvalidArgument(
          "The first dimension of Input(Bias) should be 1, but received %d.",
          b_dims[0]));

  if (use_peepholes) {
    PADDLE_ENFORCE_EQ(
        b_dims[1],
        7 * frame_size,
        common::errors::InvalidArgument(
            "The second dimension of Input(Bias) should be 7 * %d if enable "
            "peepholes connection, but received %d.",
            frame_size,
            b_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        b_dims[1],
        4 * frame_size,
        common::errors::InvalidArgument(
            "The second dimension of Input(Bias) should be 4 * %d if disable "
            "peepholes connection, but received %d.",
            frame_size,
            b_dims[1]));
  }

  phi::DDim out_dims({in_dims[0], frame_size});
  hidden->set_dims(out_dims);
  cell->set_dims(out_dims);
  if (!is_test) {
    batch_gate->set_dims(in_dims);
    batch_cell_pre_act->set_dims(out_dims);
  }
  hidden->share_lod(input);
  cell->share_lod(input);
  hidden->set_dtype(input.dtype());
  cell->set_dtype(input.dtype());
}

void DecayedAdagradInferMeta(const MetaTensor& param,
                             const MetaTensor& grad,
                             const MetaTensor& moment,
                             const MetaTensor& learning_rate,
                             float decay,
                             float epsilon,
                             MetaTensor* param_out,
                             MetaTensor* moment_out) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(common::product(lr_dims),
                    0,
                    common::errors::InvalidArgument(
                        "Maybe the Input variable LearningRate has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(
      common::product(lr_dims),
      1,
      common::errors::InvalidArgument("LearningRate should have one element"));
  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(param_dims,
                    grad.dims(),
                    common::errors::InvalidArgument(
                        "Param and Grad input of DecayedAdagradOp should have "
                        "the same dimension."));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment.dims(),
      common::errors::InvalidArgument(
          "Param and Moment input of DecayedAdagradOp should have "
          "the same dimension."));

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());
  moment_out->set_dims(param_dims);
  moment_out->set_dtype(param.dtype());
}

inline int ConvOutputSize(
    int input_size, int filter_size, int dilation, int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size,
      0,
      common::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But received: output's size is %d. The output's size is computed by "
          "((input_size + 2 * padding - (dilation * (filter_size - 1) + 1)) / "
          "stride + 1), where input_size is %d, padding is %d, "
          "filter_size is %d, dilation is %d, stride is %d.",
          output_size,
          input_size,
          padding,
          filter_size,
          dilation,
          stride));

  return output_size;
}

void DeformableConvInferMeta(const MetaTensor& x,
                             const MetaTensor& offset,
                             const MetaTensor& filter,
                             const MetaTensor& mask,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& dilations,
                             int deformable_groups,
                             int groups,
                             int im2col_step,
                             MetaTensor* out,
                             MetaConfig config) {
  auto in_dims = x.dims();
  auto offset_dims = offset.dims();
  auto filter_dims = filter.dims();

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      common::errors::InvalidArgument("Conv input should be 4-D tensor, get %u",
                                      in_dims.size()));
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    filter_dims.size(),
                    common::errors::InvalidArgument(
                        "Conv input dimension and filter dimension should be "
                        "the same. The difference is [%d]: [%d]",
                        in_dims.size(),
                        filter_dims.size()));
  PADDLE_ENFORCE_EQ(in_dims.size() - strides.size(),
                    2U,
                    common::errors::InvalidArgument(
                        "Conv input dimension and strides "
                        "dimension should be consistent. But received input "
                        "dimension:[%d], strides dimension:[%d]",
                        in_dims.size(),
                        strides.size()));
  PADDLE_ENFORCE_EQ(paddings.size(),
                    strides.size(),
                    common::errors::InvalidArgument(
                        "Conv paddings dimension and Conv strides dimension "
                        "should be the same. The difference is [%d]: [%d]",
                        paddings.size(),
                        strides.size()));

  PADDLE_ENFORCE_EQ(
      in_dims[1],
      filter_dims[1] * groups,
      common::errors::InvalidArgument(
          "The number of input channels should be equal to filter "
          "channels * groups. The difference is [%d]: [%d]",
          in_dims[1],
          filter_dims[1] * groups));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      common::errors::InvalidArgument(
          "The number of output channels should be divided by groups. But "
          "received output channels:[%d], groups:[%d]",
          filter_dims[0],
          groups));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % deformable_groups,
      0,
      common::errors::InvalidArgument(
          "The number of output channels should be "
          "divided by deformable groups. The difference is [%d]: [%d]",
          filter_dims[0] % groups,
          0));

  if (in_dims[0] > im2col_step) {
    PADDLE_ENFORCE_EQ(
        in_dims[0] % im2col_step,
        0U,
        common::errors::InvalidArgument(
            "Input batchsize must be smaller than or divide im2col_step. But "
            "received Input batchsize:[%d], im2col_step:[%d]",
            in_dims[0],
            im2col_step));
  }

  for (size_t i = 0; i < strides.size(); ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0U,
        common::errors::InvalidArgument("stride %d size incorrect", i));
  }
  for (size_t i = 0; i < dilations.size(); ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0U,
        common::errors::InvalidArgument("dilation %d size incorrect", i));
  }

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (int i = 0; i < static_cast<int>(strides.size()); ++i) {
    if (!config.is_runtime &&
        (in_dims[i + 2] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(
          ConvOutputSize(static_cast<int>(in_dims[i + 2]),
                         static_cast<int>(filter_dims[i + 2]),
                         dilations[i],
                         paddings[i],
                         strides[i]));
    }
  }

  PADDLE_ENFORCE_EQ(
      output_shape[1] % deformable_groups,
      0U,
      common::errors::InvalidArgument(
          "output num_filter must divide deformable group size. But received "
          "output num_filter:[%d], deformable group size:[%d]",
          output_shape[1],
          deformable_groups));

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(output_shape[2],
                      offset_dims[2],
                      common::errors::InvalidArgument(
                          "output height must equal to offset map height. "
                          "The difference is [%d]: [%d]",
                          output_shape[2],
                          offset_dims[2]));
    PADDLE_ENFORCE_EQ(output_shape[3],
                      offset_dims[3],
                      common::errors::InvalidArgument(
                          "output width must equal to offset map width. The "
                          "difference is [%d]: [%d]",
                          output_shape[3],
                          offset_dims[3]));

    PADDLE_ENFORCE_EQ(offset_dims[1] % (filter_dims[2] * filter_dims[3]),
                      0U,
                      common::errors::InvalidArgument(
                          "offset filter must divide deformable group size. "
                          "But received [%d]: [%d]",
                          offset_dims[1],
                          filter_dims[2] * filter_dims[3]));
    PADDLE_ENFORCE_EQ(
        offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
        deformable_groups,
        common::errors::InvalidArgument(
            "offset filter must divide deformable group size. But received "
            "[%d]: [%d]",
            offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
            deformable_groups));

    if (mask) {
      auto mask_dims = mask.dims();
      PADDLE_ENFORCE_EQ(output_shape[2],
                        mask_dims[2],
                        common::errors::InvalidArgument(
                            "output height must equal to mask map height. The "
                            "difference is [%d] vs [%d]",
                            output_shape[2],
                            mask_dims[2]));
      PADDLE_ENFORCE_EQ(output_shape[3],
                        mask_dims[3],
                        common::errors::InvalidArgument(
                            "output width must equal to mask map width. The "
                            "difference is [%d] vs [%d]",
                            output_shape[3],
                            mask_dims[3]));

      PADDLE_ENFORCE_EQ(mask_dims[1] % (filter_dims[2] * filter_dims[3]),
                        0U,
                        common::errors::InvalidArgument(
                            "mask filter must divide deformable group size. "
                            "But received [%d]: [%d]",
                            mask_dims[1],
                            filter_dims[2] * filter_dims[3]));
      PADDLE_ENFORCE_EQ(mask_dims[1] / (filter_dims[2] * filter_dims[3]),
                        deformable_groups,
                        common::errors::InvalidArgument(
                            "mask filter must divide deformable group size. "
                            "But received [%d]: [%d]",
                            mask_dims[1] / (filter_dims[2] * filter_dims[3]),
                            deformable_groups));
    }
  }

  out->set_dims(common::make_ddim(output_shape));
  out->set_dtype(x.dtype());
}

void DetectionMapInferMeta(const MetaTensor& detect_res,
                           const MetaTensor& label,
                           const MetaTensor& has_state,
                           const MetaTensor& pos_count,
                           const MetaTensor& true_pos,
                           const MetaTensor& false_pos,
                           int class_num,
                           int background_label,
                           float overlap_threshold,
                           bool evaluate_difficult,
                           const std::string& ap_type,
                           MetaTensor* accum_pos_count,
                           MetaTensor* accum_true_pos,
                           MetaTensor* accum_false_pos,
                           MetaTensor* m_ap,
                           MetaConfig config) {
  auto det_dims = detect_res.dims();
  PADDLE_ENFORCE_EQ(det_dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "Input(DetectRes) ndim must be 2, the shape is [N, 6],"
                        "but received the ndim is %d",
                        det_dims.size()));
  PADDLE_ENFORCE_EQ(det_dims[1],
                    6UL,
                    common::errors::InvalidArgument(
                        "The shape is of Input(DetectRes) [N, 6], but received"
                        " shape is [N, %d]",
                        det_dims[1]));
  auto label_dims = label.dims();
  PADDLE_ENFORCE_EQ(label_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The ndim of Input(Label) must be 2, but received %d",
                        label_dims.size()));
  if (config.is_runtime || label_dims[1] > 0) {
    PADDLE_ENFORCE_EQ(
        (label_dims[1] == 6 || label_dims[1] == 5),
        true,
        common::errors::InvalidArgument(
            "The shape of Input(Label) is [N, 6] or [N, 5], but received "
            "[N, %d]",
            label_dims[1]));
  }

  if (pos_count.initialized()) {
    PADDLE_ENFORCE_EQ(
        true_pos.initialized(),
        true,
        common::errors::InvalidArgument(
            "Input(TruePos) of DetectionMAPOp should not be null when "
            "Input(PosCount) is not null."));
    PADDLE_ENFORCE_EQ(
        false_pos.initialized(),
        true,
        common::errors::InvalidArgument(
            "Input(FalsePos) of DetectionMAPOp should not be null when "
            "Input(PosCount) is not null."));
  }

  m_ap->set_dims(common::make_ddim({1}));
}

void DgcInferMeta(const MetaTensor& u,
                  const MetaTensor& v,
                  const MetaTensor& grad,
                  const MetaTensor& param,
                  const MetaTensor& current_step_tensor,
                  const MetaTensor& nranks_tensor,
                  MetaTensor* u_out,
                  MetaTensor* v_out,
                  MetaTensor* encode_grad_out,
                  MetaTensor* grad_out,
                  MetaTensor* k_out,
                  MetaTensor* gather_buff) {}

void DGCMomentumInferMeta(const MetaTensor& param,
                          const MetaTensor& grad,
                          const MetaTensor& velocity,
                          const MetaTensor& learning_rate,
                          const MetaTensor& master_param,
                          const MetaTensor& current_step_tensor,
                          const MetaTensor& nranks_tensor,
                          float mu,
                          bool use_nesterov,
                          const std::string& regularization_method,
                          float regularization_coeff,
                          bool multi_precision,
                          float rescale_grad,
                          float rampup_begin_step,
                          MetaTensor* param_out,
                          MetaTensor* velocity_out,
                          MetaTensor* master_param_out,
                          MetaTensor* grad_out) {
  auto lr_dims = learning_rate.dims();

  PADDLE_ENFORCE_NE(common::product(lr_dims),
                    0,
                    common::errors::InvalidArgument(
                        "Maybe the Input variable LearningRate has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(common::product(lr_dims),
                    1,
                    common::errors::InvalidArgument(
                        "Learning_rate should be a scalar. But Received "
                        "LearningRate's dim [%s]",
                        common::product(lr_dims)));

  auto param_dims = param.dims();
  auto grad_dims = grad.dims();
  auto velocity_dims = velocity.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      grad_dims,
      common::errors::InvalidArgument(
          "Param and Grad input of MomentumOp should have the same "
          "dimension. But received Param's dim [%s] and Grad's dim [%s].",
          param_dims,
          grad_dims));
  PADDLE_ENFORCE_EQ(
      param_dims,
      velocity_dims,
      common::errors::InvalidArgument(
          "Param and Velocity of MomentumOp should have the same "
          "dimension. But received Param's dim [%s] and Velocity [%s].",
          param_dims,
          velocity_dims));

  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(
        param_dims,
        master_param.dims(),
        common::errors::InvalidArgument(
            "Param and MasterParam of MomentumOp should have the same "
            "dimension. But received Param's dim [%s] and MasterParam [%s].",
            param_dims,
            master_param.dims()));
  }

  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();

  param_out->set_dims(param_dims);
  velocity_out->set_dims(param_dims);
  velocity_out->set_dtype(MPType);
  if (multi_precision && master_param.initialized()) {
    master_param_out->set_dims(param_dims);
    master_param_out->set_dtype(MPType);
  }
}

void EditDistanceInferMeta(const MetaTensor& hyps,
                           const MetaTensor& refs,
                           const MetaTensor& hypslength,
                           const MetaTensor& refslength,
                           bool normalized,
                           MetaTensor* sequencenum,
                           MetaTensor* out) {
  auto hyp_dims = hyps.dims();
  auto ref_dims = refs.dims();

  if (hypslength && refslength) {
    auto hyp_length_dims = hypslength.dims();
    auto ref_length_dims = refslength.dims();

    PADDLE_ENFORCE_EQ(
        hyp_dims.size() == 2 && ref_dims.size() == 2 &&
            hyp_dims[0] == ref_dims[0],
        true,
        errors::InvalidArgument(
            "Input(hyps) and Input(refs) must be 2-D Tensors with "
            "identical first dimension. But received Input(Hyps): "
            "input rank %u, input shape [%s]; received Input(Refs): "
            "input rank %u, input shape [%s]",
            hyp_dims.size(),
            hyp_dims,
            ref_dims.size(),
            ref_dims));
    PADDLE_ENFORCE_EQ(
        hyp_length_dims[0] == ref_length_dims[0] &&
            hyp_length_dims[0] == hyp_dims[0],
        true,
        errors::InvalidArgument(
            "Input(hypslength), Input(refslength) and Input(hyps) "
            "should have identical first dimension. But received "
            "Input(hypslength): input rank %u, input shape [%s]; "
            "received Input(refslength): input rank %u, input shape "
            "[%s]; received Input(hyps): input rank %u, input shape "
            "[%s].",
            hyp_length_dims.size(),
            hyp_length_dims,
            ref_length_dims.size(),
            ref_length_dims,
            hyp_dims.size(),
            hyp_dims));
  } else {
    PADDLE_ENFORCE_EQ(
        hyp_dims.size() == 2 && hyp_dims[1] == 1,
        true,
        errors::InvalidArgument(
            "Input(Hyps) must be a 2-D DenseTensor with the 2nd dimension "
            "equal to 1. But received: input rank %u, input shape [%s].",
            hyp_dims.size(),
            hyp_dims));
    PADDLE_ENFORCE_EQ(
        ref_dims.size() == 2 && ref_dims[1] == 1,
        true,
        errors::InvalidArgument(
            "Input(Refs) must be a 2-D DenseTensor with the 2nd dimension "
            "equal to 1. But received: input rank %u, input shape [%s].",
            ref_dims.size(),
            ref_dims));
  }

  out->set_dims(refs.dims());
  out->set_dtype(DataType::FLOAT32);
  sequencenum->set_dims(common::make_ddim({1}));
  sequencenum->set_dtype(DataType::FLOAT32);
}

void FakeChannelWiseDequantizeMaxAbsInferMeta(
    const MetaTensor& x,
    const std::vector<const MetaTensor*>& scales,
    const std::vector<int>& quant_bits,
    int quant_axis,
    int x_num_col_dims,
    MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                      "the received is %d",
                                      quant_axis));
  PADDLE_ENFORCE_EQ(x_num_col_dims == 0,
                    false,
                    common::errors::InvalidArgument(
                        "'x_num_col_dims' should be larger than 0, but "
                        "the received is %d",
                        x_num_col_dims));
  out->set_dtype(x.dtype());
  out->share_dims(x);
  out->share_lod(x);
}

void FakeQuantOrWithDequantMovingAverageAbsMaxInferMeta(
    const MetaTensor& x,
    const MetaTensor& in_scale,
    const MetaTensor& in_accum,
    const MetaTensor& in_state,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    MetaTensor* out,
    MetaTensor* out_scale,
    MetaTensor* out_state,
    MetaTensor* out_accum) {
  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));
  if (out_state) {
    out_state->set_dims({1});
  }
  if (out_accum) {
    out_accum->set_dims({1});
  }
  out->set_dims(x.dims());
  out_scale->set_dims({1});
  out->share_lod(x);
}

void FtrlInferMeta(const MetaTensor& param,
                   const MetaTensor& squared_accumulator,
                   const MetaTensor& linear_accumulator,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   float l1,
                   float l2,
                   float lr_power,
                   MetaTensor* param_out,
                   MetaTensor* squared_accum_out,
                   MetaTensor* linear_accum_out) {
  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(param_dim,
                    grad.dims(),
                    common::errors::InvalidArgument(
                        "Two input of FTRL Op's dimension must be same, but "
                        "param_dim is %d, Grad is %d",
                        param_dim,
                        grad.dims()));

  auto lr_dim = learning_rate.dims();
  PADDLE_ENFORCE_NE(common::product(lr_dim),
                    0,
                    common::errors::InvalidArgument(
                        "Maybe the Input variable LearningRate has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(common::product(lr_dim),
                    1,
                    common::errors::InvalidArgument(
                        "Learning Rate should be a scalar, but got %d",
                        common::product(lr_dim)));

  param_out->set_dims(param_dim);
  param_out->set_dtype(param.dtype());
  squared_accum_out->set_dims(param_dim);
  squared_accum_out->set_dtype(param.dtype());
  linear_accum_out->set_dims(param_dim);
  linear_accum_out->set_dtype(param.dtype());
}

void FusedBatchNormActInferMeta(const MetaTensor& x,
                                const MetaTensor& scale,
                                const MetaTensor& bias,
                                const MetaTensor& mean,
                                const MetaTensor& variance,
                                MetaTensor* y,
                                MetaTensor* mean_out,
                                MetaTensor* variance_out,
                                MetaTensor* saved_mean,
                                MetaTensor* saved_variance,
                                MetaTensor* reserve_space) {
  BatchNormInferMeta(x,
                     mean,
                     variance,
                     scale,
                     bias,
                     false,
                     0.0,
                     0.0,
                     "NHWC",
                     false,
                     false,
                     y,
                     mean_out,
                     variance_out,
                     saved_mean,
                     saved_variance,
                     reserve_space);
}

void FusedBiasActInferMeta(const MetaTensor& x,
                           const MetaTensor& bias,
                           const MetaTensor& dequant_scales,
                           const MetaTensor& shift,
                           const MetaTensor& smooth,
                           const std::string& act_method,
                           const std::string& compute_dtype,
                           float quant_scale,
                           int quant_round_type,
                           float quant_max_bound,
                           float quant_min_bound,
                           MetaTensor* out,
                           MetaConfig config) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      common::errors::InvalidArgument(
          "The size of Input(x) must greater than 1: %s", x_dims));
  int x_last_dim = x_dims.size() - 1;
  auto dim = x_dims[x_last_dim];

  std::vector<int64_t> x_shapes;
  for (int i = 0; i < x_dims.size(); i++) {
    x_shapes.push_back(x_dims[i]);
  }

  if (config.is_runtime) {
    PADDLE_ENFORCE_GT(
        x.numel() / dim,
        0,
        common::errors::InvalidArgument("The size of Attr(rows) must > 0"));

    PADDLE_ENFORCE_GT(
        dim,
        0,
        common::errors::InvalidArgument("The size of Attr(cols) must > 0"));
  }

  if (act_method == "geglu" || act_method == "swiglu") {
    PADDLE_ENFORCE_EQ(
        dim % 2,
        0,
        common::errors::InvalidArgument(
            "The second dimension of x must be even, but receive %d", dim));
    x_shapes[x_last_dim] /= 2;
    out->set_dims(common::make_ddim(x_shapes));
  } else if (act_method == "gelu" || act_method == "relu") {
    out->set_dims(common::make_ddim(x_shapes));
  } else {
    PADDLE_THROW(
        errors::InvalidArgument("act_method must be geglu, swiglu or gelu, "
                                "but get act_method (%s)",
                                act_method));
  }

  auto FBADtypeCheck = [](const MetaTensor& check_tensor,
                          const std::string& tensor_name,
                          const std::string& compute_dtype) {
    if (compute_dtype == "bf16") {
      PADDLE_ENFORCE_EQ(
          check_tensor.dtype(),
          phi::DataType::BFLOAT16,
          common::errors::InvalidArgument(
              "Input(%s) dtype must be the same with Attr(compute_dtype)",
              tensor_name));
    } else if (compute_dtype == "fp16") {
      PADDLE_ENFORCE_EQ(
          check_tensor.dtype(),
          phi::DataType::FLOAT16,
          common::errors::InvalidArgument(
              "Input(%s) dtype must be the same with Attr(compute_dtype)",
              tensor_name));
    } else if (compute_dtype == "fp32") {
      PADDLE_ENFORCE_EQ(
          check_tensor.dtype(),
          phi::DataType::FLOAT32,
          common::errors::InvalidArgument(
              "Input(%s) dtype must be the same with Attr(compute_dtype)",
              tensor_name));
    }
  };

  // In the case of quantization enabled, the dtype for computation is
  // determined based on compute_dtype.
  if (x.dtype() == phi::DataType::INT32) {
    PADDLE_ENFORCE_NE(
        compute_dtype,
        "default",
        common::errors::InvalidArgument(
            "If Input(x) dtype is INT32, Attr(compute_dtype) must be set."));

    if (bias) {
      FBADtypeCheck(bias, "bias", compute_dtype);
    }

    if (quant_scale > 0) {
      out->set_dtype(phi::DataType::INT8);
    } else {
      if (compute_dtype == "bf16") {
        out->set_dtype(phi::DataType::BFLOAT16);
      } else if (compute_dtype == "fp16") {
        out->set_dtype(phi::DataType::FLOAT16);
      } else if (compute_dtype == "fp32") {
        out->set_dtype(phi::DataType::FLOAT32);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "In the case of quantization enabled with Input(x) INT32, "
            "Attr(compute_dtype) must be set in (bf16, fp16, fp32), "
            "but get compute_dtype (%s)",
            compute_dtype));
      }
    }
  } else {
    // x.dtype() != phi::DataType::INT32
    if (bias) {
      if (compute_dtype != "default") {
        FBADtypeCheck(bias, "bias", compute_dtype);
        FBADtypeCheck(x, "x", compute_dtype);
      } else {
        PADDLE_ENFORCE_EQ(x.dtype(),
                          bias.dtype(),
                          common::errors::InvalidArgument(
                              "Input(x) and Input(bias) must be the "
                              "same dtype in this situation"));
      }
    } else {
      // bias not exist
      if (compute_dtype != "default") {
        FBADtypeCheck(x, "x", compute_dtype);
      }
    }
    if (quant_scale > 0) {
      if (fabs(quant_max_bound - 127.0f) < 0.000001) {
        out->set_dtype(phi::DataType::INT8);
      } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
        out->set_dtype(phi::DataType::FLOAT8_E4M3FN);
      }
    } else {
      out->set_dtype(x.dtype());
    }
  }
  out->set_layout(x.layout());
}

void FusedLayerNormInferMeta(const MetaTensor& x,
                             const MetaTensor& bias,
                             const MetaTensor& residual,
                             const MetaTensor& norm_weight,
                             const MetaTensor& norm_bias,
                             const float epsilon,
                             const float residual_alpha,
                             const int begin_norm_axis,
                             const float quant_scale,
                             const int quant_round_type,
                             const float quant_max_bound,
                             const float quant_min_bound,
                             MetaTensor* out,
                             MetaTensor* residual_out,
                             MetaTensor* mean,
                             MetaTensor* variance,
                             MetaConfig config) {
  std::vector<int64_t> x_dims_vec = common::vectorize(x.dims());
  auto x_dims_size = x_dims_vec.size();

  size_t normalized_dims = 1;
  for (size_t i = begin_norm_axis; i < x_dims_size; ++i) {
    normalized_dims *= x_dims_vec[i];
  }

  int32_t rows = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= static_cast<int32_t>(x.dims()[i]);
  }
  if (config.is_runtime) {
    if (norm_weight) {
      PADDLE_ENFORCE_EQ(normalized_dims,
                        norm_weight.dims()[0],
                        common::errors::InvalidArgument(
                            "The normalized size of Input(X) must equal to be"
                            "the size of Weight, but received"
                            "normalized size of Input(X) is [%d], received size"
                            "of Weight is [%d]",
                            normalized_dims,
                            norm_weight.dims()[0]));
    }
  }

  auto out_dims = common::make_ddim(x_dims_vec);

  out->set_dims(out_dims);
  if (residual_out && !norm_weight && !norm_bias) {
    out->set_dtype(x.dtype());
  } else {
    if (quant_scale > 0) {
      if (fabs(quant_max_bound - 127.0f) < 0.000001) {
        out->set_dtype(phi::DataType::INT8);
      } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
        out->set_dtype(phi::DataType::FLOAT8_E4M3FN);
      }
    } else {
      out->set_dtype(x.dtype());
    }
  }
  out->set_layout(x.layout());

  residual_out->set_dims(out_dims);
  residual_out->set_dtype(x.dtype());
  residual_out->set_layout(x.layout());

  mean->set_dims(common::make_ddim({rows}));
  mean->set_dtype(DataType::FLOAT32);
  mean->set_layout(x.layout());

  variance->set_dims(common::make_ddim({rows}));
  variance->set_dtype(DataType::FLOAT32);
  variance->set_layout(x.layout());
}

void FusedLinearParamGradAddInferMeta(const MetaTensor& x,
                                      const MetaTensor& dout,
                                      const MetaTensor& dweight,
                                      const MetaTensor& dbias,
                                      bool multi_precision,
                                      bool has_bias,
                                      MetaTensor* dweight_out,
                                      MetaTensor* dbias_out) {
  const auto dtype = dout.dtype();
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      dtype,
      common::errors::InvalidArgument(
          "The data type of Input(x) and Input(dout) must be the same."));

  const auto& x_dims = x.dims();
  const auto& dout_dims = dout.dims();
  int rank = dout_dims.size();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      rank,
      common::errors::InvalidArgument(
          "The shape of Input(x) and Input(dout) do not match: %s vs %s.",
          x_dims,
          dout_dims));
  for (int i = 0; i + 1 < x_dims.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        x_dims[i],
        dout_dims[i],
        common::errors::InvalidArgument(
            "The shape of Input(x) and Input(dout) do not match: %s vs %s.",
            x_dims,
            dout_dims));
  }

  const phi::DDim& weight_dims = {x_dims[rank - 1], dout_dims[rank - 1]};
  if (dweight) {
    PADDLE_ENFORCE_EQ(
        weight_dims,
        dweight.dims(),
        common::errors::InvalidArgument(
            "The shape of input(dweight) does not match the other inputs."));
  }

  const auto mp_dtype =
      (dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16)
          ? DataType::FLOAT32
          : dtype;

  if (has_bias && dbias_out) {
    dbias_out->set_dims({weight_dims[1]});
    dbias_out->set_dtype(multi_precision ? mp_dtype : dtype);
  }

  if (dweight_out) {
    dweight_out->set_dims(weight_dims);
    dweight_out->set_dtype(multi_precision ? mp_dtype : dtype);
  }
}

void FusionGroupInferMeta(const std::vector<const MetaTensor*>& ins,
                          const std::vector<int>& outs_dtype,
                          const std::vector<int>& inputs_dtype,
                          const std::string& func_name,
                          int type,
                          std::vector<MetaTensor*> outs) {
  const size_t num_ins = ins.size();
  const size_t num_outs = outs.size();

  PADDLE_ENFORCE_GE(
      num_ins,
      1UL,
      common::errors::InvalidArgument(
          "Expected the number of inputs >= 1. Received %d.", num_ins));
  PADDLE_ENFORCE_GE(
      num_outs,
      1UL,
      common::errors::InvalidArgument(
          "Expected the number of outputs >= 1. Received %d.", num_outs));

  PADDLE_ENFORCE_EQ(type,
                    0UL,
                    common::errors::InvalidArgument(
                        "Only support fusion of elementwise operations."));

  std::vector<phi::DDim> x_dims;
  for (size_t i = 0; i < num_ins; ++i) {
    x_dims.push_back(ins[i]->dims());
  }

  if (type == 0) {
    for (size_t i = 1; i < num_ins; ++i) {
      PADDLE_ENFORCE_EQ(x_dims[0],
                        x_dims[i],
                        common::errors::InvalidArgument(
                            "All the inputs' dims is expected to be the same. "
                            "But received [%s] (name: %s) vs [%s] (name: %s).",
                            x_dims[0],
                            ins[0],
                            x_dims[i],
                            ins[i]));
    }
    for (size_t j = 0; j < num_outs; ++j) {
      outs[j]->set_dims(x_dims[0]);
    }
  }

  // Only lod of Inputs[0] would be shared with Outs.
  for (size_t j = 0; j < num_outs; ++j) {
    outs[j]->share_lod(*ins[0]);
  }

  for (size_t j = 0; j < num_outs; ++j) {
    if (outs_dtype[j] == phi::TransToProtoVarType(phi::DataType::FLOAT16)) {
      outs[j]->set_dtype(phi::DataType::FLOAT16);
    } else if (outs_dtype[j] ==
               phi::TransToProtoVarType(phi::DataType::FLOAT32)) {
      outs[j]->set_dtype(phi::DataType::FLOAT32);
    } else if (outs_dtype[j] ==
               phi::TransToProtoVarType(phi::DataType::FLOAT64)) {
      outs[j]->set_dtype(phi::DataType::FLOAT64);
    }
  }
}

void GenerateProposalsV2InferMeta(const MetaTensor& scores,
                                  const MetaTensor& bbox_deltas,
                                  const MetaTensor& im_shape,
                                  const MetaTensor& anchors,
                                  const MetaTensor& variances,
                                  int pre_nms_top_n,
                                  int post_nms_top_n,
                                  float nms_thresh,
                                  float min_size,
                                  float eta,
                                  bool pixel_offset,
                                  MetaTensor* rpn_rois,
                                  MetaTensor* rpn_roi_probs,
                                  MetaTensor* rpn_rois_num) {
  rpn_rois->set_dims(common::make_ddim({-1, 4}));
  rpn_roi_probs->set_dims(common::make_ddim({-1, 1}));
  if (rpn_rois_num) {
    rpn_rois_num->set_dims(common::make_ddim({scores.dims()[0]}));
  }
}

void LegacyGenerateProposalsInferMeta(const MetaTensor& scores,
                                      const MetaTensor& bbox_deltas,
                                      const MetaTensor& im_info,
                                      const MetaTensor& anchors,
                                      const MetaTensor& variances,
                                      int pre_nms_top_n,
                                      int post_nms_top_n,
                                      float nms_thresh,
                                      float min_size,
                                      float eta,
                                      MetaTensor* rpn_rois,
                                      MetaTensor* rpn_roi_probs,
                                      MetaTensor* rpn_rois_num) {
  GenerateProposalsV2InferMeta(scores,
                               bbox_deltas,
                               im_info,
                               anchors,
                               variances,
                               pre_nms_top_n,
                               post_nms_top_n,
                               nms_thresh,
                               min_size,
                               eta,
                               true,
                               rpn_rois,
                               rpn_roi_probs,
                               rpn_rois_num);
}

void GraphKhopSamplerInferMeta(const MetaTensor& row,
                               const MetaTensor& col_ptr,
                               const MetaTensor& x,
                               const MetaTensor& eids,
                               const std::vector<int>& sample_sizes,
                               bool return_eids,
                               MetaTensor* out_src,
                               MetaTensor* out_dst,
                               MetaTensor* sample_index,
                               MetaTensor* reindex_x,
                               MetaTensor* out_eids) {
  // GKS: GraphKhopSampler
  auto GKSShapeCheck = [](const phi::DDim& dims, std::string tensor_name) {
    if (dims.size() == 2) {
      PADDLE_ENFORCE_EQ(dims[1],
                        1,
                        common::errors::InvalidArgument(
                            "The last dim of %s should be 1 when it "
                            "is 2D, but we get %d",
                            tensor_name,
                            dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dims.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              dims.size()));
    }
  };

  GKSShapeCheck(row.dims(), "row");
  GKSShapeCheck(col_ptr.dims(), "col_ptr");
  GKSShapeCheck(x.dims(), "x");
  PADDLE_ENFORCE_EQ(
      !sample_sizes.empty(),
      true,
      common::errors::InvalidArgument(
          "The parameter 'sample_sizes' in GraphSampleOp must be set. "
          "But received 'sample_sizes' is empty."));

  if (return_eids) {
    GKSShapeCheck(eids.dims(), "eids");
    out_eids->set_dims({-1});
    out_eids->set_dtype(row.dtype());
  }

  out_src->set_dims({-1, 1});
  out_src->set_dtype(row.dtype());
  out_dst->set_dims({-1, 1});
  out_dst->set_dtype(row.dtype());
  sample_index->set_dims({-1});
  sample_index->set_dtype(DataType::INT32);
  reindex_x->set_dims(x.dims());
  reindex_x->set_dtype(x.dtype());
}

void GraphReindexInferMeta(const MetaTensor& x,
                           const MetaTensor& neighbors,
                           const MetaTensor& count,
                           const MetaTensor& hashtable_value,
                           const MetaTensor& hashtable_index,
                           MetaTensor* reindex_src,
                           MetaTensor* reindex_dst,
                           MetaTensor* out_nodes) {
  bool flag_buffer_hashtable =
      hashtable_value.initialized() && hashtable_index.initialized();
  auto GraphReindexShapeCheck = [](const phi::DDim& dims,
                                   std::string tensor_name) {
    if (dims.size() == 2) {
      PADDLE_ENFORCE_EQ(dims[1],
                        1,
                        common::errors::InvalidArgument(
                            "The last dim of %s should be 1 when it "
                            "is 2D, but we get %d",
                            tensor_name,
                            dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dims.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              dims.size()));
    }
  };

  GraphReindexShapeCheck(x.dims(), "X");
  GraphReindexShapeCheck(neighbors.dims(), "Neighbors");
  GraphReindexShapeCheck(count.dims(), "Count");
  if (flag_buffer_hashtable) {
    GraphReindexShapeCheck(hashtable_value.dims(), "HashTable_Value");
    GraphReindexShapeCheck(hashtable_index.dims(), "HashTable_Index");
  }

  reindex_src->set_dims({-1});
  reindex_src->set_dtype(neighbors.dtype());
  reindex_dst->set_dims({-1});
  reindex_dst->set_dtype(neighbors.dtype());
  out_nodes->set_dims({-1});
  out_nodes->set_dtype(x.dtype());
}

void GruInferMeta(const MetaTensor& input,
                  const MetaTensor& h0,
                  const MetaTensor& weight,
                  const MetaTensor& bias,
                  const std::string& activation,
                  const std::string& gate_activation,
                  bool is_reverse,
                  bool origin_mode,
                  bool is_test,
                  MetaTensor* batch_gate,
                  MetaTensor* batch_reset_hidden_prev,
                  MetaTensor* batch_hidden,
                  MetaTensor* hidden,
                  MetaConfig config) {
  const auto& input_dims = input.dims();
  const auto& weight_dims = weight.dims();
  int input_size = static_cast<int>(input_dims[1]);
  int frame_size = static_cast<int>(weight_dims[0]);
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(input_size,
                      frame_size * 3,
                      common::errors::InvalidArgument(
                          "The second dimension of Input(Input) must be 3 "
                          "times of frame_size in GRUOp, but received %d "
                          "(Input) vs %d (frame_size).",
                          input_size,
                          frame_size));
  }
  PADDLE_ENFORCE_EQ(
      weight_dims[1],
      frame_size * 3,
      common::errors::InvalidArgument(
          "The shape of Input(Weight) matrix must be [frame_size, frame_size "
          "* 3], but received [%d, %d] (Weight) vs [%d, %d] (frame_size).",
          weight_dims[0],
          weight_dims[1],
          frame_size,
          frame_size * 3));
  if (h0.initialized()) {
    const auto& h0_dims = h0.dims();
    PADDLE_ENFORCE_EQ(
        h0_dims[1],
        frame_size,
        common::errors::InvalidArgument(
            "The width of Input(H0) must be equal to frame_size, but "
            "received %d (width of H0) vs %d (frame_size).",
            h0_dims[1],
            frame_size));
  }
  if (bias.initialized()) {
    const auto& bias_dims = bias.dims();
    int bias_height = static_cast<int>(bias_dims[0]);
    int bias_width = static_cast<int>(bias_dims[1]);
    PADDLE_ENFORCE_EQ(
        bias_height,
        1,
        common::errors::InvalidArgument(
            "The shape of Bias must be [1, frame_size * 3], but received "
            "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
            bias_height,
            bias_width,
            frame_size * 3));
    PADDLE_ENFORCE_EQ(
        bias_width,
        frame_size * 3,
        common::errors::InvalidArgument(
            "The shape of Bias must be [1, frame_size * 3], but received "
            "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
            bias_height,
            bias_width,
            frame_size * 3));
  }
  if (!is_test) {
    batch_gate->set_dims(input_dims);
    batch_gate->set_dtype(input.dtype());
    batch_reset_hidden_prev->set_dims({input_dims[0], frame_size});
    batch_reset_hidden_prev->set_dtype(input.dtype());
    batch_hidden->set_dims({input_dims[0], frame_size});
    batch_hidden->set_dtype(input.dtype());
  }
  hidden->set_dims({input_dims[0], frame_size});
  hidden->set_dtype(input.dtype());
  hidden->share_lod(input);
}

void GruUnitInferMeta(const MetaTensor& input,
                      const MetaTensor& hidden_prev,
                      const MetaTensor& weight,
                      const MetaTensor& bias,
                      int activation,
                      int gate_activation,
                      bool origin_mode,
                      MetaTensor* gate,
                      MetaTensor* reset_hidden_prev,
                      MetaTensor* hidden,
                      MetaConfig config) {
  const auto& input_dims = input.dims();
  const auto& hidden_prev_dims = hidden_prev.dims();
  const auto& weight_dims = weight.dims();
  int batch_size = static_cast<int>(input_dims[0]);
  int input_size = static_cast<int>(input_dims[1]);
  int frame_size = static_cast<int>(hidden_prev_dims[1]);
  int weight_height = static_cast<int>(weight_dims[0]);
  int weight_width = static_cast<int>(weight_dims[1]);
  if (config.is_runtime || input_size >= 0) {
    PADDLE_ENFORCE_EQ(input_size,
                      frame_size * 3,
                      common::errors::InvalidArgument(
                          "The second dimension of Input(Input) must be 3 "
                          "times of frame_size in GRUUnitOp, but received %d "
                          "(Input) vs %d (frame_size).",
                          input_size,
                          frame_size));
  }
  PADDLE_ENFORCE_EQ(
      weight_height,
      frame_size,
      common::errors::InvalidArgument(
          "The shape of Input(Weight) matrix must be [frame_size, frame_size "
          "* 3] in GRUUnitOp, but received [%d, %d] (Weight) vs [%d, %d] "
          "(frame_size).",
          weight_height,
          weight_width,
          frame_size,
          frame_size * 3));
  PADDLE_ENFORCE_EQ(
      weight_width,
      frame_size * 3,
      common::errors::InvalidArgument(
          "The shape of Input(Weight) matrix must be [frame_size, frame_size "
          "* 3] in GRUUnitOp, but received [%d, %d] (Weight) vs [%d, %d] "
          "(frame_size).",
          weight_height,
          weight_width,
          frame_size,
          frame_size * 3));

  if (bias.initialized()) {
    const auto& bias_dims = bias.dims();
    int bias_height = static_cast<int>(bias_dims[0]);
    int bias_width = static_cast<int>(bias_dims[1]);
    PADDLE_ENFORCE_EQ(
        bias_height,
        1,
        common::errors::InvalidArgument(
            "The shape of Bias must be [1, frame_size * 3], but received "
            "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
            bias_height,
            bias_width,
            frame_size * 3));
    PADDLE_ENFORCE_EQ(
        bias_width,
        frame_size * 3,
        common::errors::InvalidArgument(
            "The shape of Bias must be [1, frame_size * 3], but received "
            "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
            bias_height,
            bias_width,
            frame_size * 3));
  }
  gate->set_dims({batch_size, frame_size * 3});
  reset_hidden_prev->set_dims({batch_size, frame_size});
  hidden->set_dims({batch_size, frame_size});

  gate->set_dtype(input.dtype());
  reset_hidden_prev->set_dtype(input.dtype());
  hidden->set_dtype(input.dtype());
}

void GraphSampleNeighborsInferMeta(const MetaTensor& row,
                                   const MetaTensor& col_ptr,
                                   const MetaTensor& x,
                                   const MetaTensor& eids,
                                   const MetaTensor& perm_buffer,
                                   int sample_size,
                                   bool return_eids,
                                   bool flag_perm_buffer,
                                   MetaTensor* out,
                                   MetaTensor* out_count,
                                   MetaTensor* out_eids) {
  // GSN: GraphSampleNeighbors
  auto GSNShapeCheck = [](const phi::DDim& dims, std::string tensor_name) {
    if (dims.size() == 2) {
      PADDLE_ENFORCE_EQ(dims[1],
                        1,
                        common::errors::InvalidArgument(
                            "The last dim of %s should be 1 when it "
                            "is 2D, but we get %d",
                            tensor_name,
                            dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dims.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              dims.size()));
    }
  };

  GSNShapeCheck(row.dims(), "Row");
  GSNShapeCheck(col_ptr.dims(), "Col_Ptr");
  GSNShapeCheck(x.dims(), "X");
  if (return_eids) {
    GSNShapeCheck(eids.dims(), "Eids");
    out_eids->set_dims({-1});
    out_eids->set_dtype(row.dtype());
  }
  if (flag_perm_buffer) {
    GSNShapeCheck(perm_buffer.dims(), "Perm_Buffer");
  }

  out->set_dims({-1});
  out->set_dtype(row.dtype());
  out_count->set_dims({-1});
  out_count->set_dtype(DataType::INT32);
}

void HSigmoidLossInferMeta(const MetaTensor& x,
                           const MetaTensor& label,
                           const MetaTensor& w,
                           const MetaTensor& bias,
                           const MetaTensor& path,
                           const MetaTensor& code,
                           int num_classes,
                           bool is_sparse,
                           MetaTensor* out,
                           MetaTensor* pre_out,
                           MetaTensor* w_out) {
  const int64_t input_dims = x.dims()[0];
  const int64_t label_dims = label.dims()[0];
  PADDLE_ENFORCE_EQ(input_dims,
                    label_dims,
                    common::errors::InvalidArgument(
                        "The first dimension of "
                        "input and label is expected to be the same. "
                        "But received input's first dimension is %d; "
                        "label's first dimension is %d.",
                        input_dims,
                        label_dims));

  std::vector<int64_t> output_shape({input_dims, 1});
  out->set_dims(common::make_ddim(output_shape));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

static void Interpolate1DInferShapeCheck(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();

  PADDLE_ENFORCE_EQ("linear",
                    interp_method,
                    common::errors::InvalidArgument(
                        "Interpolation method can only be \"linear\" when"
                        "Input(X) dimension is 3, but got method = %s .",
                        interp_method));
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  for (int i = 0; i < dim_x.size(); ++i) {
    PADDLE_ENFORCE_NE(dim_x[i],
                      0,
                      common::errors::InvalidArgument(
                          "The shape of input(x) should be larger "
                          "than 0, but received shape[%d] is %d ",
                          i,
                          dim_x[i]));
  }
  if (size_tensor && !size_tensor->empty()) {
    // top priority size
    auto inputs_name = size_tensor.get();
    PADDLE_ENFORCE_EQ(
        inputs_name.size(),
        1,
        common::errors::InvalidArgument(
            "Input(SizeTensor)'size of Op(interpolate) must be 1. "
            "Attr(out_shape)'s length must be 1 for 3-D input tensor, but got "
            "size = %d .",
            inputs_name.size()));
    phi::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_w};
    } else {
      dim_out = {dim_x[0], out_w, dim_x[2]};
    }
    output->set_dims(dim_out);
    output->set_dtype(x.dtype());

    return;
  }

  int out_w_tmp = 0;
  if (scale_tensor) {
    auto scale_tensor_dim = scale_tensor.dims();
    PADDLE_ENFORCE_EQ(
        scale_tensor_dim.size() == 1 || scale_tensor_dim.size() == 0,
        true,
        common::errors::InvalidArgument(
            "Scale's dimension size must be 1 or 0, but got dimension = %d .",
            scale_tensor_dim.size()));
    if (scale_tensor_dim.size() == 1) {
      PADDLE_ENFORCE_EQ(scale_tensor_dim[0],
                        1,
                        common::errors::InvalidArgument(
                            "Scale's shape must be 1, but got shape = %d .",
                            scale_tensor_dim[0]));
    }
    out_w_tmp = -1;
  } else {
    if (!scale.empty()) {
      float scale_w = -1;
      scale_w = scale[0];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          common::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      if (scale_w > 0.) {
        // round down
        out_w_tmp =
            static_cast<int>(data_layout == DataLayout::kNCHW
                                 ? static_cast<float>(dim_x[2]) * scale_w
                                 : static_cast<float>(dim_x[1]) * scale_w);
        // protect when input shape is -1
        out_w_tmp = out_w_tmp > 0 ? out_w_tmp : -1;
      }
    } else {
      out_w_tmp = out_w;
    }
  }

  if (out_size && config.is_runtime) {
    auto out_size_dim = out_size.dims();
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(),
        1,
        common::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got dimension = %d .",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(
        out_size_dim[0],
        1,
        common::errors::InvalidArgument(
            "OutSize's 0-th dimension's value must be 1, but got value = %d .",
            out_size_dim[0]));

    // dims will be seted in kernel
    output->set_dtype(x.dtype());
    output->share_lod(x);
    return;
  }

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_w_tmp};
  } else {
    dim_out = {dim_x[0], out_w_tmp, dim_x[2]};
  }
  output->set_dims(dim_out);
  output->set_dtype(x.dtype());
}

static void Interpolate2DInferShapeCheck(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();

  PADDLE_ENFORCE_EQ(
      ("bilinear" == interp_method || "nearest" == interp_method ||
       "bicubic" == interp_method),
      true,
      common::errors::InvalidArgument(
          "Interpolation method can only be \"bilinear\" or \"nearest\" when "
          "Input(X) dimension is 4, but got method = %s.",
          interp_method));
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  for (int i = 0; i < dim_x.size(); ++i) {
    PADDLE_ENFORCE_NE(dim_x[i],
                      0,
                      common::errors::InvalidArgument(
                          "The shape of input(x) should be larger "
                          "than 0, but received shape[%d] is %d ",
                          i,
                          dim_x[i]));
  }

  if (size_tensor && !size_tensor->empty()) {
    // top priority size
    auto inputs_name = size_tensor.get();
    PADDLE_ENFORCE_EQ(
        inputs_name.size(),
        2,
        common::errors::InvalidArgument(
            "Input(SizeTensor)'size of Op(interpolate) must be 2. "
            "Attr(out_shape)'s length must be 2 for 4-D input "
            "tensor, but got size = %d .",
            inputs_name.size()));
    phi::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_h, out_w, dim_x[3]};
    }
    output->set_dims(dim_out);
    output->set_dtype(x.dtype());

    return;
  }

  int out_h_tmp = 0, out_w_tmp = 0;

  if (scale_tensor) {
    auto scale_tensor_dim = scale_tensor.dims();
    PADDLE_ENFORCE_EQ(
        scale_tensor_dim.size() == 1 || scale_tensor_dim.size() == 0,
        true,
        common::errors::InvalidArgument(
            "Scale's dimension size must be 1 or 0, but got dimension = %d .",
            scale_tensor_dim.size()));

    if (scale_tensor_dim.size() == 1) {
      PADDLE_ENFORCE_EQ(
          scale_tensor_dim[0] == 2 || scale_tensor_dim[0] == 1,
          true,
          common::errors::InvalidArgument(
              "Scale's shape must be 2 or 1, but got shape = %d .",
              scale_tensor_dim[0]));
    }

    out_h_tmp = -1;
    out_w_tmp = -1;
  } else {
    if (!scale.empty()) {
      float scale_h = -1;
      float scale_w = -1;
      scale_h = scale[0];
      scale_w = scale[1];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          common::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          common::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      if (scale_h > 0. && scale_w > 0.) {
        // round down
        out_h_tmp =
            static_cast<int>(data_layout == DataLayout::kNCHW
                                 ? static_cast<float>(dim_x[2]) * scale_h
                                 : static_cast<float>(dim_x[1]) * scale_h);
        out_w_tmp =
            static_cast<int>(data_layout == DataLayout::kNCHW
                                 ? static_cast<float>(dim_x[3]) * scale_w
                                 : static_cast<float>(dim_x[2]) * scale_w);
        // protect when input shape is -1
        out_h_tmp = out_h_tmp > 0 ? out_h_tmp : -1;
        out_w_tmp = out_w_tmp > 0 ? out_w_tmp : -1;
      }
    } else {
      out_h_tmp = out_h;
      out_w_tmp = out_w;
    }
  }

  if (out_size && config.is_runtime) {
    auto out_size_dim = out_size.dims();
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(),
        1,
        common::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got dimension = %d .",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(
        out_size_dim[0],
        2,
        common::errors::InvalidArgument(
            "OutSize's dim[0] must be 2, but got dimension = %d .",
            out_size_dim[0]));
    // dims will be seted in kernel
    output->set_dtype(x.dtype());
    output->share_lod(x);
    return;
  }

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_h_tmp, out_w_tmp};
  } else {
    dim_out = {dim_x[0], out_h_tmp, out_w_tmp, dim_x[3]};
  }

  output->set_dims(dim_out);
  output->set_dtype(x.dtype());
}

static void Interpolate3DInferShapeCheck(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();

  PADDLE_ENFORCE_EQ(
      ("nearest" == interp_method || "trilinear" == interp_method),
      true,
      common::errors::InvalidArgument(
          "Interpolation method can only be \"trilinear\" or "
          "\"nearest\" when Input(X) "
          "dimension is 5, but got method = %s .",
          interp_method));
  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);

  for (int i = 0; i < dim_x.size(); ++i) {
    PADDLE_ENFORCE_NE(dim_x[i],
                      0,
                      common::errors::InvalidArgument(
                          "The shape of input(x) should be larger "
                          "than 0, but received shape[%d] is %d ",
                          i,
                          dim_x[i]));
  }

  if (size_tensor && !size_tensor->empty()) {
    // top priority size
    auto inputs_name = size_tensor.get();
    PADDLE_ENFORCE_EQ(
        inputs_name.size(),
        3,
        common::errors::InvalidArgument(
            "Input(SizeTensor)'s size of Op(interpolate) must be 3. "
            "Attr(out_shape)'s length must be 3 for 5-D input "
            "tensor, but got size = %d .",
            inputs_name.size()));
    phi::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_d, out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_d, out_h, out_w, dim_x[4]};
    }
    output->set_dims(dim_out);
    output->set_dtype(x.dtype());
    return;
  }

  int out_d_tmp = 0, out_h_tmp = 0, out_w_tmp = 0;
  if (scale_tensor) {
    auto scale_tensor_dim = scale_tensor.dims();
    PADDLE_ENFORCE_EQ(
        scale_tensor_dim.size() == 1 || scale_tensor_dim.size() == 0,
        true,
        common::errors::InvalidArgument(
            "Scale's dimension size must be 1 or 0, but got size = %d .",
            scale_tensor_dim.size()));
    PADDLE_ENFORCE_EQ(scale_tensor_dim[0] == 3 || scale_tensor_dim[0] == 1,
                      true,
                      common::errors::InvalidArgument(
                          "Scale's shape must be 3 or 1, but got shape = %d .",
                          scale_tensor_dim[0]));
    out_d_tmp = -1;
    out_h_tmp = -1;
    out_w_tmp = -1;
  } else {
    if (!scale.empty()) {
      float scale_d = -1;
      float scale_h = -1;
      float scale_w = -1;
      scale_d = scale[0];
      scale_h = scale[1];
      scale_w = scale[2];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          common::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          common::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      PADDLE_ENFORCE_EQ(
          scale_d > 0,
          true,
          common::errors::InvalidArgument(
              "The scale_d in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_d));
      if (scale_d > 0. && scale_h > 0. && scale_w > 0.) {
        // round down
        out_d_tmp =
            static_cast<int>(data_layout == DataLayout::kNCHW
                                 ? static_cast<float>(dim_x[2]) * scale_d
                                 : static_cast<float>(dim_x[1]) * scale_d);
        out_h_tmp =
            static_cast<int>(data_layout == DataLayout::kNCHW
                                 ? static_cast<float>(dim_x[3]) * scale_h
                                 : static_cast<float>(dim_x[2]) * scale_h);
        out_w_tmp =
            static_cast<int>(data_layout == DataLayout::kNCHW
                                 ? static_cast<float>(dim_x[4]) * scale_w
                                 : static_cast<float>(dim_x[3]) * scale_w);
        // protect when input shape is -1
        out_d_tmp = out_d_tmp > 0 ? out_d_tmp : -1;
        out_h_tmp = out_h_tmp > 0 ? out_h_tmp : -1;
        out_w_tmp = out_w_tmp > 0 ? out_w_tmp : -1;
      }
    } else {
      out_d_tmp = out_d;
      out_h_tmp = out_h;
      out_w_tmp = out_w;
    }
  }

  if (out_size && config.is_runtime) {
    auto out_size_dim = out_size.dims();
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(),
        1,
        common::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got size is %d.",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(out_size_dim[0],
                      3,
                      common::errors::InvalidArgument(
                          "OutSize's dim[0] must be 3, but got size is %d.",
                          out_size_dim[0]));
    // dims will be seted in kernel
    output->set_dtype(x.dtype());
    output->share_lod(x);
    return;
  }

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_d_tmp, out_h_tmp, out_w_tmp};
  } else {
    dim_out = {dim_x[0], out_d_tmp, out_h_tmp, out_w_tmp, dim_x[4]};
  }
  output->set_dims(dim_out);
  output->set_dtype(x.dtype());
}

void InterpolateInferMeta(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();  // NCHW format
  PADDLE_ENFORCE_EQ(
      (dim_x.size() == 3 || dim_x.size() == 4 || dim_x.size() == 5),
      true,
      common::errors::Unimplemented(
          "Input(X) dimension must be 3, 4 or 5, but got dimension = %d .",
          dim_x.size()));
  if (dim_x.size() == 3) {
    // shape check for 1D interpolate for input tensor shape NCHW
    Interpolate1DInferShapeCheck(x,
                                 out_size,
                                 size_tensor,
                                 scale_tensor,
                                 data_layout_str,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale,
                                 interp_method,
                                 align_corners,
                                 align_mode,
                                 output,
                                 config);
  } else if (dim_x.size() == 4) {
    // shape check for 2D interpolate for input tensor shape NCHW
    Interpolate2DInferShapeCheck(x,
                                 out_size,
                                 size_tensor,
                                 scale_tensor,
                                 data_layout_str,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale,
                                 interp_method,
                                 align_corners,
                                 align_mode,
                                 output,
                                 config);
  } else {  // dim_x.size() == 5
    // shape check for 3D interpolate for input tensor shape NCDHW
    Interpolate3DInferShapeCheck(x,
                                 out_size,
                                 size_tensor,
                                 scale_tensor,
                                 data_layout_str,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale,
                                 interp_method,
                                 align_corners,
                                 align_mode,
                                 output,
                                 config);
  }
}

void LegacyInterpolateInferMeta(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  const auto& dim_x = x.dims();
  std::vector<float> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateInferMeta(x,
                       out_size,
                       size_tensor,
                       scale_tensor,
                       data_layout,
                       out_d,
                       out_h,
                       out_w,
                       scale_vec,
                       interp_method,
                       align_corners,
                       align_mode,
                       output,
                       config);
}

void IndexPutInferMeta(const MetaTensor& x,
                       const std::vector<const MetaTensor*>& indices,
                       const MetaTensor& value,
                       bool accumulate,
                       MetaTensor* out) {
  auto in_dims = x.dims();
  PADDLE_ENFORCE_LT(
      in_dims.size(),
      7,
      common::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.",
          in_dims.size()));
  out->share_meta(x);
}

void LambInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   float weight_decay,
                   float beta1,
                   float beta2,
                   float epsilon,
                   bool always_adapt,
                   bool multi_precision,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(
      common::product(lr_dims),
      0,
      common::errors::InvalidArgument(
          "The number of LearningRate shall not be 0, but received %d. Maybe "
          "the Input variable LearningRate has not "
          "been initialized. You may need to confirm "
          "if you put exe.run(startup_program) "
          "after optimizer.minimize function.",
          common::product(lr_dims)));
  PADDLE_ENFORCE_EQ(
      common::product(lr_dims),
      1,
      common::errors::InvalidArgument(
          "Learning rate should have 1 dimension, but received %d.",
          common::product(lr_dims)));
  auto beta1_pow_dims = beta1_pow.dims();
  PADDLE_ENFORCE_GE(common::product(beta1_pow_dims),
                    1,
                    common::errors::InvalidArgument(
                        "The size of Beta1 power accumulator should be "
                        "greater than 0, but received %d.",
                        common::product(beta1_pow_dims)));
  auto beta2_pow_dims = beta2_pow.dims();
  PADDLE_ENFORCE_GE(common::product(beta2_pow_dims),
                    1,
                    common::errors::InvalidArgument(
                        "The size of Beta2 power accumulator should be "
                        "greater than 0, but received %d.",
                        common::product(beta2_pow_dims)));

  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment1.dims(),
      common::errors::InvalidArgument(
          "Param and Moment1 input of LambOp should have same dimension. But "
          "received Param dims: [%s], Moment1 dims: [%s].",
          param_dims,
          moment1.dims()));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment2.dims(),
      errors::InvalidArgument(
          "Param and Moment2 input of AdamOp should have same dimension. But "
          "received Param dims: [%s], Moment2 dims: [%s].",
          param_dims,
          moment2.dims()));
  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(param_dims,
                      master_param.dims(),
                      errors::InvalidArgument(
                          "Param and MasterParam input of AdamOp should have "
                          "same dimension. But "
                          "received Param dims: [%s], MasterParam dims: [%s].",
                          param_dims,
                          master_param.dims()));
  }

  PADDLE_ENFORCE_NOT_NULL(
      param_out, errors::NotFound("The output param_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      moment1_out,
      errors::NotFound("The output moment1_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      moment2_out,
      errors::NotFound("The output moment2_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      beta1_pow_out,
      errors::NotFound("The output beta1_pow_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      beta2_pow_out,
      errors::NotFound("The output beta2_pow_out can not be nullptr"));
  param_out->set_dims(param_dims);

  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();

  moment1_out->set_dims(param_dims);
  moment1_out->set_dtype(moment1.dtype());
  moment2_out->set_dims(param_dims);
  moment2_out->set_dtype(moment2.dtype());

  beta1_pow_out->set_dims(beta1_pow_dims);
  beta1_pow_out->set_dtype(beta1_pow.dtype());
  beta2_pow_out->set_dims(beta2_pow_dims);
  beta2_pow_out->set_dtype(beta2_pow.dtype());

  if (multi_precision && master_param.initialized()) {
    master_param_outs->set_dims(param_dims);
    master_param_outs->set_dtype(MPType);
  }
}

void LarsMomentumInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& velocity,
    const std::vector<const MetaTensor*>& learning_rate,
    const std::vector<const MetaTensor*>& grad,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    const std::vector<float>& lars_weight_decay,
    float mu,
    float lars_coeff,
    float epsilon,
    bool multi_precision,
    float rescale_grad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> velocity_out,
    std::vector<MetaTensor*> master_param_out) {
  std::vector<DDim> lr_dims = GetMetaTensorsDim(learning_rate);
  std::vector<DDim> grad_dim = GetMetaTensorsDim(grad);
  std::vector<DDim> param_dim = GetMetaTensorsDim(param);
  std::vector<DDim> velocity_dim = GetMetaTensorsDim(velocity);

  PADDLE_ENFORCE_EQ(
      param_dim.size(),
      grad_dim.size(),
      common::errors::InvalidArgument(
          "Input(Param) and Input(Grad) of LarsMomentumOp should have "
          "same quantity. But number of Param is [%d] and Grad is [%d].",
          param_dim.size(),
          grad_dim.size()));
  PADDLE_ENFORCE_EQ(
      param_dim.size(),
      velocity_dim.size(),
      common::errors::InvalidArgument(
          "Input(Param) and Input(Velocity) of LarsMomentumOp should "
          "have same quantity. But number of Param is [%d] and Velocity "
          "is [%d].",
          param_dim.size(),
          velocity_dim.size()));
  PADDLE_ENFORCE_EQ(
      lars_weight_decay.size(),
      grad_dim.size(),
      common::errors::InvalidArgument(
          "Attr(Lars_weight_decay) and "
          "Input(Grad) of LarsMomentumOp should have same quantity. "
          "But number of Lars_weight_decay is [%d] and Grad is [%d].",
          lars_weight_decay.size(),
          grad_dim.size()));

  for (auto& lr_dim : lr_dims) {
    PADDLE_ENFORCE_EQ(common::product(lr_dim),
                      1,
                      common::errors::InvalidArgument(
                          "Learning_rate should be a scalar. But Received "
                          "LearningRate's dim [%s]",
                          common::product(lr_dim)));
  }

  for (size_t i = 0; i < param_dim.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        param_dim[i],
        grad_dim[i],
        common::errors::InvalidArgument(
            "Input(Param) and Input(Grad) input of LarsMomentumOp shall "
            "have same dimension. But Param`s dim is [%s] and Grad's dim "
            "is [%s].",
            param_dim[i],
            grad_dim[i]));
    PADDLE_ENFORCE_EQ(
        param_dim[i],
        velocity_dim[i],
        common::errors::InvalidArgument(
            "Input(Param) and Input(Velocity) of LarsMomentumOp shall have "
            "same dimension. But Param dim [%s] differs with Velocity dim "
            "[%s].",
            param_dim[i],
            velocity_dim[i]));
  }

  for (size_t i = 0; i < param_out.size(); i++) {
    auto MPType = (param[i]->dtype() == phi::DataType::FLOAT16 ||
                   param[i]->dtype() == phi::DataType::BFLOAT16)
                      ? phi::DataType::FLOAT32
                      : param[i]->dtype();
    param_out[i]->set_dims(param_dim[i]);
    param_out[i]->set_dtype(param[i]->dtype());
    velocity_out[i]->set_dims(param_dim[i]);
    velocity_out[i]->set_dtype(MPType);
    if (master_param != nullptr) {
      master_param_out[i]->set_dims(param_dim[i]);
      master_param_out[i]->set_dtype(MPType);
    }
  }
}

void LLMInt8LinearInferMeta(const MetaTensor& x,
                            const MetaTensor& weight,
                            const MetaTensor& bias,
                            const MetaTensor& weight_scale,
                            const float threshold,
                            MetaTensor* out) {
  auto x_dims = x.dims();
  auto w_dims = weight.dims();
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2UL,
      errors::InvalidArgument("The input(weight) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      x_dims[x_dims.size() - 1],
      w_dims[1],
      errors::InvalidArgument(
          "Input(X) dim[-1] and Input(Weight) dim[1] should be equal."
          "But received Input(X) dim[-1](%s) != Input(Weight) dim[1](%s)",
          x_dims[x_dims.size() - 1],
          w_dims[1]));
  PADDLE_ENFORCE_EQ(
      w_dims[0] % 16,
      0,
      common::errors::InvalidArgument(
          "The first dimension of input must be divisible by 16, but got[%d]",
          w_dims[0]));
  PADDLE_ENFORCE_EQ(
      w_dims[1] % 16,
      0,
      common::errors::InvalidArgument(
          "The second dimension of input must be divisible by 16, but got[%d]",
          w_dims[1]));
  PADDLE_ENFORCE_EQ(
      weight_scale.dims()[0],
      w_dims[0],
      errors::InvalidArgument(
          "Input(weight_scale) dim[0] and Input(Weight) dim[0] should be equal."
          "But received Input(weight_scale) dim[0](%s) != Input(Weight) "
          "dim[0](%s)",
          weight_scale.dims()[0],
          w_dims[0]));
  auto out_dims = x_dims;
  out_dims[out_dims.size() - 1] = w_dims[0];
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void LogspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       const MetaTensor& base,
                       DataType dtype,
                       MetaTensor* out) {
  auto s_dims = start.dims();
  PADDLE_ENFORCE_EQ(
      common::product(s_dims),
      1,
      common::errors::InvalidArgument("The size of Input(Start) must be 1,"
                                      "but received input size is %s.",
                                      common::product(s_dims)));
  auto e_dims = stop.dims();
  PADDLE_ENFORCE_EQ(
      common::product(e_dims),
      true,
      common::errors::InvalidArgument("The size of Input(Stop) must be 1,"
                                      "but received input size is %s.",
                                      common::product(e_dims)));
  auto num_dims = number.dims();
  PADDLE_ENFORCE_EQ(
      common::product(num_dims),
      true,
      common::errors::InvalidArgument("The size of Input(Num) must be 1,"
                                      "but received input size is %s.",
                                      common::product(num_dims)));
  auto b_dims = base.dims();
  PADDLE_ENFORCE_EQ(common::product(b_dims),
                    true,
                    common::errors::InvalidArgument(
                        "The size of Input(Base) must be 1,"
                        "but received input size is common::product(b_dims).",
                        common::product(b_dims)));
  out->set_dims(common::make_ddim({-1}));
  out->set_dtype(dtype);
}

void MergedAdamInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& learning_rate,
    const std::vector<const MetaTensor*>& moment1,
    const std::vector<const MetaTensor*>& moment2,
    const paddle::optional<std::vector<const MetaTensor*>>& moment2_max,
    const std::vector<const MetaTensor*>& beta1_pow,
    const std::vector<const MetaTensor*>& beta2_pow,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> moment1_out,
    std::vector<MetaTensor*> moment2_out,
    std::vector<MetaTensor*> moment2_max_out,
    std::vector<MetaTensor*> beta1_pow_out,
    std::vector<MetaTensor*> beta2_pow_out,
    std::vector<MetaTensor*> master_param_out) {}

void MergedMomentumInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& velocity,
    const std::vector<const MetaTensor*>& learning_rate,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> velocity_out,
    std::vector<MetaTensor*> master_param_out) {}

void MemoryEfficientAttentionInferMeta(const MetaTensor& query,
                                       const MetaTensor& key,
                                       const MetaTensor& value,
                                       const MetaTensor& bias,
                                       const MetaTensor& cu_seqlens_q,
                                       const MetaTensor& cu_seqlens_k,
                                       const MetaTensor& causal_diagonal,
                                       const MetaTensor& seqlen_k,
                                       const Scalar& max_seqlen_q,
                                       const Scalar& max_seqlen_k,
                                       const bool causal,
                                       const double dropout_p,
                                       const float scale,
                                       const bool is_test,
                                       MetaTensor* output,
                                       MetaTensor* logsumexp,
                                       MetaTensor* seed_and_offset) {
  PADDLE_ENFORCE_EQ(
      query.dims().size(),
      4,
      common::errors::InvalidArgument("Query should be a 4-D tensor"
                                      "But received Query dimension(%s)",
                                      query.dims().size()));
  PADDLE_ENFORCE_EQ(
      key.dims().size(),
      4,
      common::errors::InvalidArgument("Key should be a 4-D tensor"
                                      "But received Key dimension(%s)",
                                      key.dims().size()));
  PADDLE_ENFORCE_EQ(
      value.dims().size(),
      4,
      common::errors::InvalidArgument("Value should be a 4-D tensor"
                                      "But received Value dimension(%s)",
                                      value.dims().size()));

  const int64_t query_batch_size = query.dims()[0];
  const int64_t query_seq_length = query.dims()[1];
  const int64_t query_num_head = query.dims()[2];
  const int64_t query_head_size = query.dims()[3];

  const int64_t key_batch_size = key.dims()[0];
  const int64_t key_seq_length = key.dims()[1];
  const int64_t key_num_head = key.dims()[2];
  const int64_t key_head_size = key.dims()[3];

  const int64_t value_batch_size = value.dims()[0];
  const int64_t value_seq_length = value.dims()[1];
  const int64_t value_num_head = value.dims()[2];
  const int64_t value_head_size = value.dims()[3];

  PADDLE_ENFORCE_EQ(((query_batch_size == key_batch_size) &&
                     (key_batch_size == value_batch_size)),
                    true,
                    common::errors::InvalidArgument(
                        "The batchsize of Query, Key, Value should be equal."));

  PADDLE_ENFORCE_EQ(
      ((query_num_head == key_num_head) && (key_num_head == value_num_head)),
      true,
      common::errors::InvalidArgument(
          "The head number of Query, Key, Value should be equal."));

  PADDLE_ENFORCE_EQ(query_head_size == key_head_size,
                    true,
                    common::errors::InvalidArgument(
                        "The head size of Query, Key should be equal."));

  PADDLE_ENFORCE_EQ(key_seq_length == value_seq_length,
                    true,
                    common::errors::InvalidArgument(
                        "The seq length of Key, Value should be equal."));
  std::vector<int64_t> out_dims(
      {query_batch_size, query_seq_length, query_num_head, value_head_size});
  std::vector<int64_t> logsumexp_dims({query_num_head, query_batch_size});
  std::vector<int64_t> seed_and_offset_dims({2});

  output->set_dims(common::make_ddim(out_dims));
  output->share_lod(query);
  output->set_dtype(query.dtype());
  output->set_layout(query.layout());

  logsumexp->set_dims(common::make_ddim(logsumexp_dims));
  logsumexp->set_dtype(phi::DataType::FLOAT32);

  seed_and_offset->set_dims(common::make_ddim(seed_and_offset_dims));
  seed_and_offset->set_dtype(phi::DataType::INT64);
}

void MeshgridInferMeta(const std::vector<const MetaTensor*>& inputs,
                       std::vector<MetaTensor*> outputs) {
  const size_t inputs_num = inputs.size();

  std::vector<int> out_shape = std::vector<int>(inputs_num);

  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i]->dims().size() == 0) {
      out_shape[i] = 1;
    } else {
      out_shape[i] = static_cast<int>(inputs[i]->dims()[0]);
    }
  }
  auto out_dims = common::make_ddim(std::vector<int>(out_shape));
  for (auto& output : outputs) {
    output->set_dims(out_dims);
    output->set_dtype(inputs[0]->dtype());
  }
}

void MomentumInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& velocity,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float mu,
                       bool use_nesterov,
                       const std::string& regularization_method,
                       float regularization_coeff,
                       bool multi_precision,
                       float rescale_grad,
                       MetaTensor* param_out,
                       MetaTensor* velocity_out,
                       MetaTensor* master_param_out) {
  PADDLE_ENFORCE_NE(
      param_out,
      nullptr,
      errors::NotFound("Output(ParamOut) of Momentum should not be null."));
  PADDLE_ENFORCE_NE(
      velocity_out,
      nullptr,
      errors::NotFound("Output(VelocityOut) of Momentum should not be null."));

  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(
      common::product(lr_dims),
      0,
      errors::InvalidArgument("Maybe the Input variable LearningRate has not "
                              "been initialized. You may need to confirm "
                              "if you put exe.run(startup_program) "
                              "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(
      common::product(lr_dims),
      1,
      errors::InvalidArgument("Learning_rate should be a scalar. But Received "
                              "LearningRate's dim [%s]",
                              common::product(lr_dims)));

  auto param_dim = param.dims();
  param_out->set_dims(param_dim);
  auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                 param.dtype() == phi::DataType::BFLOAT16)
                    ? phi::DataType::FLOAT32
                    : param.dtype();
  velocity_out->set_dims(param_dim);
  velocity_out->set_dtype(MPType);
  if (master_param_out) {
    master_param_out->set_dims(param_dim);
    master_param_out->set_dtype(MPType);
  }
}

void MultiDotInferMeta(const std::vector<const MetaTensor*>& x,
                       MetaTensor* out) {
  auto inputs_dims = GetMetaTensorsDim(x);

  const size_t inputs_num = inputs_dims.size();
  PADDLE_ENFORCE_GT(
      inputs_num,
      static_cast<size_t>(1),
      common::errors::InvalidArgument(
          "The number of input tensors in multi_dot op should > 1."));

  const size_t n = inputs_dims.size();
  auto first_dim = inputs_dims[0];

  bool is_vector = false;
  phi::DDim out_dim;

  PADDLE_ENFORCE_LT(
      first_dim.size(),
      static_cast<size_t>(3),
      common::errors::InvalidArgument(
          "multi_dot: the first input tensor must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the first tensor is 1D of size n view it as a row vector (1, n)
  if (first_dim.size() == 1) {
    first_dim = common::make_ddim({1, static_cast<int>(first_dim[0])});
    is_vector = true;
  }

  auto last_dim = inputs_dims[n - 1];
  PADDLE_ENFORCE_LT(
      last_dim.size(),
      static_cast<size_t>(3),
      common::errors::InvalidArgument(
          "the last input tensor of multi_dot must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (last_dim.size() == 1) {
    last_dim = common::make_ddim({static_cast<int>(last_dim[0]), 1});
    out_dim =
        is_vector ? common::make_ddim({}) : common::make_ddim({first_dim[0]});
  } else {
    out_dim = is_vector ? common::make_ddim({last_dim[1]})
                        : common::make_ddim({first_dim[0], last_dim[1]});
  }

  auto width = first_dim.at(1);
  for (size_t i = 1; i < n - 1; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(),
                      static_cast<size_t>(2),
                      common::errors::InvalidArgument(
                          "the input tensor of multi_dot op must be 2D."));

    const auto& tmp_dim = inputs_dims[i];
    PADDLE_ENFORCE_EQ(
        tmp_dim[0],
        width,
        common::errors::InvalidArgument(
            "the input matrix does not meet the multiplication requirements."));
    width = tmp_dim[1];
  }

  PADDLE_ENFORCE_EQ(
      last_dim[0],
      width,
      common::errors::InvalidArgument(
          "the input matrix does not meet the multiplication requirements."));

  out->set_dims(out_dim);
  out->set_dtype(x.at(0)->dtype());
  out->share_lod(*x.at(0));
}

void MultiplexInferMeta(const std::vector<const MetaTensor*>& ins,
                        const MetaTensor& ids,
                        MetaTensor* out) {
  PADDLE_ENFORCE_NE(
      ins.empty(),
      true,
      common::errors::InvalidArgument("MultiInput(X) shouldn't be empty."));
  auto ids_dim = ids.dims();
  PADDLE_ENFORCE_EQ(ids_dim.size(),
                    2,
                    common::errors::PreconditionNotMet(
                        "The index tensor must be a vector with 2 dimensions"));
  PADDLE_ENFORCE_EQ(
      ids_dim[1],
      1,
      common::errors::PreconditionNotMet(
          "The index tensor must be a vector with batchSize x 1."));

  auto ins_dims = GetMetaTensorsDim(ins);
  auto num_ins = ins_dims.size();
  PADDLE_ENFORCE_GT(num_ins,
                    1,
                    common::errors::InvalidArgument(
                        "multiplex operator should have more than "
                        "one candidate input tensors."));

  auto in_dim = ins_dims[0];
  PADDLE_ENFORCE_GE(
      in_dim.size(),
      2,
      common::errors::InvalidArgument(
          "The rank of candidate tensors must be not less than 2."));
  for (size_t i = 1; i < num_ins; i++) {
    auto dim = ins_dims[i];
    PADDLE_ENFORCE_EQ(
        in_dim,
        dim,
        common::errors::PreconditionNotMet(
            "All the candidate tensors must have the same size."));
  }

  PADDLE_ENFORCE_GE(
      in_dim[0],
      ids_dim[0],
      common::errors::InvalidArgument("The 2nd-dim of input cannot be smaller "
                                      "than batchSize of the index tensor."));

  in_dim[0] = ids_dim[0];
  out->set_dims(in_dim);
  out->set_dtype(ins[0]->dtype());
}

void NAdamInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& momentum_decay_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& mu_product,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& master_param,
                    float beta1,
                    float beta2,
                    float epsilon,
                    float momentum_decay,
                    bool multi_precision,
                    MetaTensor* param_out,
                    MetaTensor* momentum_decay_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* mu_product_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* master_param_out) {
  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(param_dim,
                    moment1.dims(),
                    common::errors::InvalidArgument(
                        "Param and Momentum input of NAdamOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and Moment1 [%s]",
                        param_dim,
                        moment1.dims()));
  PADDLE_ENFORCE_EQ(param_dim,
                    moment2.dims(),
                    common::errors::InvalidArgument(
                        "Param and Momentum input of NAdamOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and Moment2 [%s]",
                        param_dim,
                        moment2.dims()));

  auto lr_dim = learning_rate.dims();
  PADDLE_ENFORCE_EQ(common::product(lr_dim),
                    1,
                    common::errors::InvalidArgument(
                        "Learning Rate of NAdamOp should be a scalar. But "
                        "received LearningRate's dim [%s]",
                        common::product(lr_dim)));

  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(param_dim,
                      master_param.dims(),
                      errors::InvalidArgument(
                          "Param and MasterParam input of NAdamOp should "
                          "have same dimension. But "
                          "received Param dims: [%s], MasterParam dims: [%s].",
                          param_dim,
                          master_param.dims()));
  }

  param_out->set_dims(param_dim);
  param_out->set_dtype(param.dtype());

  momentum_decay_pow_out->set_dims(momentum_decay_pow.dims());
  momentum_decay_pow_out->set_dtype(momentum_decay_pow.dtype());
  beta2_pow_out->set_dims(beta2_pow.dims());
  beta2_pow_out->set_dtype(beta2_pow.dtype());
  mu_product_out->set_dims(mu_product.dims());
  mu_product_out->set_dtype(mu_product.dtype());

  moment1_out->set_dims(param_dim);
  moment1_out->set_dtype(moment1.dtype());
  moment2_out->set_dims(param_dim);
  moment2_out->set_dtype(moment2.dtype());

  if (multi_precision && master_param.initialized()) {
    auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                   param.dtype() == phi::DataType::BFLOAT16)
                      ? phi::DataType::FLOAT32
                      : param.dtype();
    master_param_out->set_dims(param_dim);
    master_param_out->set_dtype(MPType);
  }
}

void NceInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& weight,
                  const MetaTensor& bias,
                  const MetaTensor& sample_weight,
                  const MetaTensor& custom_dist_probs,
                  const MetaTensor& custom_dist_alias,
                  const MetaTensor& custom_dist_alias_probs,
                  int num_total_classes,
                  const std::vector<int>& custom_neg_classes,
                  int num_neg_samples,
                  int sampler,
                  int seed,
                  bool is_sparse,
                  bool remote_prefetch,
                  bool is_test,
                  MetaTensor* cost,
                  MetaTensor* sample_logits,
                  MetaTensor* sample_labels,
                  MetaConfig config) {
  auto x_dims = input.dims();
  auto label_dims = label.dims();
  if (config.is_runtime || (x_dims[0] > 0 && label_dims[0] > 0)) {
    PADDLE_ENFORCE_EQ(
        x_dims[0],
        label_dims[0],
        common::errors::InvalidArgument(
            "The first dimension of Input(Input) and Input(Label) should be "
            "equal in runtime. But received: Input(Input)'s shape = [%s] "
            "with 1st dim =  %d, Input(Label)'s shape = [%s] with 1st dim = "
            "%d.",
            x_dims,
            x_dims[0],
            label_dims,
            label_dims[0]));
  }
  int num_true_classes =
      static_cast<int>(label_dims.size() == 2 ? label_dims[1] : 1);
  if (bias) {
    PADDLE_ENFORCE_EQ(
        weight.dims()[0],
        bias.dims()[0],
        common::errors::InvalidArgument(
            "The first dimension of Input(Weight) and Input(Bias) "
            "should be equal. But received: Input(Weight)'s shape = [%s] "
            "with 1st dim = %d, and Input(Bias)'s shape = [%s] with 1st dim "
            "= %d.",
            weight.dims(),
            weight.dims()[0],
            bias.dims(),
            bias.dims()[0]));
  }

  PADDLE_ENFORCE_EQ(
      num_total_classes,
      weight.dims()[0],
      common::errors::InvalidArgument(
          "The number of total classes should be equal to the first "
          "dimension of Input(Weight). But received: Attr(num_total_classes) "
          "= %d, Input(Weight)'s shape = [%s] with 1st dim = %d.",
          num_total_classes,
          weight.dims(),
          weight.dims()[0]));
  if (custom_neg_classes.size() > 0) {
    PADDLE_ENFORCE_EQ(
        custom_neg_classes.size(),
        static_cast<size_t>(num_neg_samples),
        common::errors::InvalidArgument(
            "The size of Attr(custom_neg_classes) should be equal "
            "to the number of negative samples. But received: "
            "custom_neg_classes.size() = %d, num_neg_samples = %d.",
            custom_neg_classes.size(),
            num_neg_samples));
  }
  // set dims of output(Out)
  std::vector<int64_t> out_dims;
  out_dims.push_back(x_dims[0]);
  out_dims.push_back(1);
  cost->set_dims(common::make_ddim(out_dims));
  cost->set_dtype(DataType::FLOAT32);

  if (!is_test) {
    // set dims of output(SampleOut)
    std::vector<int64_t> sample_out_dims;
    sample_out_dims.push_back(x_dims[0]);
    sample_out_dims.push_back(
        (num_true_classes == -1) ? -1 : (num_neg_samples + num_true_classes));
    sample_logits->set_dims(common::make_ddim(sample_out_dims));
    sample_labels->set_dims(common::make_ddim(sample_out_dims));
  }
}

void PsroiPoolInferMeta(const MetaTensor& x,
                        const MetaTensor& rois,
                        const MetaTensor& rois_num,
                        int pooled_height,
                        int pooled_width,
                        int output_channels,
                        float spatial_scale,
                        MetaTensor* out) {
  auto input_dims = x.dims();
  auto rois_dims = rois.dims();

  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      errors::InvalidArgument("The format of input tensor is NCHW"));
  PADDLE_ENFORCE_EQ(
      rois_dims.size(),
      2,
      errors::InvalidArgument(
          "ROIs should be a 2-D DenseTensor of shape (num_rois, 4) "
          "given as [(x1, y1, x2, y2), ...]"));
  PADDLE_ENFORCE_EQ(
      rois_dims[1],
      4,
      errors::InvalidArgument(
          "ROIs should be a 2-D DenseTensor of shape (num_rois, 4) "
          "given as [(x1, y1, x2, y2), ...]"));
  if (rois_num) {
    auto rois_num_dims = rois_num.dims();
    PADDLE_ENFORCE_EQ(
        rois_num_dims.size(),
        1,
        errors::InvalidArgument("The second dimension of RoisNum should "
                                "be 1, but received dimension is %d",
                                rois_num_dims.size()));
  }

  PADDLE_ENFORCE_EQ(
      input_dims[1],
      output_channels * pooled_height * pooled_width,
      errors::InvalidArgument(
          "the channel of X(%d) "
          "should be equal to the product of "
          "output_channels(%d), pooled_height(%d) and pooled_width(%d)",
          input_dims[1],
          output_channels,
          pooled_height,
          pooled_width));

  PADDLE_ENFORCE_GT(pooled_height,
                    0,
                    errors::InvalidArgument(
                        "The pooled output height must be greater than 0"));
  PADDLE_ENFORCE_GT(pooled_width,
                    0,
                    errors::InvalidArgument(
                        "The pooled output width must be greater than 0"));
  PADDLE_ENFORCE_GT(output_channels,
                    1,
                    errors::InvalidArgument(
                        "The pooled output channels must greater than 1"));
  PADDLE_ENFORCE_GT(
      spatial_scale,
      0.0f,
      errors::InvalidArgument("The spatial scale must greater than 0."));

  auto out_dims = input_dims;
  out_dims[0] = rois_dims[0];
  out_dims[1] =
      output_channels;  // input_dims[1] / (pooled_height * pooled_width);
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void PyramidHashInferMeta(const MetaTensor& x,
                          const MetaTensor& w,
                          const MetaTensor& white_list,
                          const MetaTensor& black_list,
                          int num_emb,
                          int space_len,
                          int pyramid_layer,
                          int rand_len,
                          float drop_out_percent,
                          int is_training,
                          bool use_filter,
                          int white_list_len,
                          int black_list_len,
                          int seed,
                          float lr,
                          const std::string& distribute_update_vars,
                          MetaTensor* out,
                          MetaTensor* drop_pos,
                          MetaTensor* x_temp_out,
                          MetaConfig config) {
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input(X) of PyramidHashOP is invalid. "
                        "It should be 2, but got %d",
                        x_dims.size()));

  const auto& w_dims = w.dims();
  PADDLE_ENFORCE_EQ(w_dims.size(),
                    2,
                    common::errors::InvalidArgument(
                        "The rank of Input(W) of PyramidHashOP is invalid. "
                        "It should be 2, but got %d",
                        w_dims.size()));

  PADDLE_ENFORCE_EQ(
      w_dims[0],
      space_len + rand_len,
      common::errors::InvalidArgument(
          "The first dimension of Input(W) of PyramidHashOP is invalid. "
          "It should be space_len + rand_len, but now %d != %d + %d",
          w_dims[0],
          space_len,
          rand_len));
  PADDLE_ENFORCE_EQ(
      w_dims[1],
      1,
      common::errors::InvalidArgument(
          "The second dimension of Input(W) of PyramidHashOP is invalid."
          " It should be 1, but got %d",
          w_dims[1]));

  PADDLE_ENFORCE_EQ(
      num_emb % rand_len,
      0,
      common::errors::InvalidArgument(
          "The PyramidHashOP's Attr(num_emb) should mod Attr(rand_len), "
          "but num_emb is %d, rand_len is %d",
          num_emb,
          rand_len));

  if (white_list_len > 0) {
    PADDLE_ENFORCE_EQ(
        white_list.initialized(),
        true,
        common::errors::NotFound("Input(WhiteList) of PyramidHashOP is not "
                                 "found but white_list_len > 0."));
    const auto& wl_dims = white_list.dims();
    PADDLE_ENFORCE_EQ(
        wl_dims.size(),
        2,
        common::errors::InvalidArgument(
            "The rank of Input(WhiteList) of PyramidHashOP is invalid."
            " It should be 2, but got %d",
            wl_dims.size()));
    PADDLE_ENFORCE_EQ(wl_dims[0],
                      white_list_len,
                      common::errors::InvalidArgument(
                          "The first dimension of Input(WhiteList) of "
                          "PyramidHashOP is invalid."
                          " It should be equal to Attr(white_list_len) "
                          ", but first dimension is %d, white_list_len is %d",
                          wl_dims[0],
                          white_list_len));
    PADDLE_ENFORCE_EQ(wl_dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "The second dimension of Input(WhiteList) of "
                          "PyramidHashOP is invalid."
                          " It should be 1, but got %d",
                          wl_dims[1]));
  }

  if (black_list_len > 0) {
    const auto& bl_dims = black_list.dims();
    PADDLE_ENFORCE_EQ(
        bl_dims.size(),
        2,
        common::errors::InvalidArgument(
            "The rank of Input(BlackList) of PyramidHashOP is invalid."
            " It should be 2, but got %d",
            bl_dims.size()));
    PADDLE_ENFORCE_EQ(bl_dims[0],
                      black_list_len,
                      common::errors::InvalidArgument(
                          "The first dimension of Input(BlackList) of "
                          "PyramidHashOP is invalid."
                          " It should be equal to Attr(black_list_len)"
                          ", but first dimension is %d, black_list_len is %d",
                          bl_dims[0],
                          black_list_len));
    PADDLE_ENFORCE_EQ(bl_dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "The second dimension of Input(BlackList) of "
                          "PyramidHashOP is invalid."
                          " It should be 1, but got %d",
                          bl_dims[1]));
  }

  if (config.is_runtime) {
    // something to do in runtime.
  } else {
    // compile time
    out->set_dims(common::make_ddim({-1, num_emb}));
    x_temp_out->set_dims(x_dims);
    out->share_lod(x);
  }
}

void QuantizeLinearInferMeta(const MetaTensor& x,
                             const MetaTensor& scale,
                             const MetaTensor& zero_point,
                             const MetaTensor& in_accum,
                             const MetaTensor& in_state,
                             int quant_axis,
                             int bit_length,
                             int round_type,
                             bool is_test,
                             bool only_observer,
                             MetaTensor* y,
                             MetaTensor* out_state,
                             MetaTensor* out_accum,
                             MetaTensor* out_scale) {
  PADDLE_ENFORCE_EQ(
      quant_axis == 0 || quant_axis == 1 || quant_axis == -1,
      true,
      common::errors::InvalidArgument("'quant_axis' should be 0, 1 or -1, but "
                                      "the received is %d",
                                      quant_axis));
  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    common::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));
  PADDLE_ENFORCE_EQ(round_type == 0 || round_type == 1,
                    true,
                    common::errors::InvalidArgument(
                        "'round_type' should be 0 or 1, 0 rounding to "
                        "nearest ties to even and 1 is rounding to nearest "
                        "ties away from zero.but the received is %d",
                        round_type));
  y->set_dims(x.dims());
  y->share_lod(x);
  if (out_scale) {
    if (quant_axis < 0) {
      out_scale->set_dims(scale.dims());
    } else {
      out_scale->set_dims({x.dims()[quant_axis]});
    }
  }
  if (out_accum) {
    out_accum->set_dims(in_accum.dims());
  }
  if (out_state) {
    out_state->set_dims(in_state.dims());
  }
}

void RAdamInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& beta1_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& rho,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& master_param,
                    float beta1,
                    float beta2,
                    float epsilon,
                    bool multi_precision,
                    MetaTensor* param_out,
                    MetaTensor* beta1_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* rho_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* master_param_out) {
  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(param_dim,
                    moment1.dims(),
                    common::errors::InvalidArgument(
                        "Param and Momentum input of RAdamOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and Moment1 [%s]",
                        param_dim,
                        moment1.dims()));
  PADDLE_ENFORCE_EQ(param_dim,
                    moment2.dims(),
                    common::errors::InvalidArgument(
                        "Param and Momentum input of RAdamOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and Moment2 [%s]",
                        param_dim,
                        moment2.dims()));

  auto lr_dim = learning_rate.dims();
  PADDLE_ENFORCE_EQ(common::product(lr_dim),
                    1,
                    common::errors::InvalidArgument(
                        "Learning Rate of RAdamOp should be a scalar. But "
                        "received LearningRate's dim [%s]",
                        common::product(lr_dim)));

  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(param_dim,
                      master_param.dims(),
                      errors::InvalidArgument(
                          "Param and MasterParam input of RAdamOp should "
                          "have same dimension. But "
                          "received Param dims: [%s], MasterParam dims: [%s].",
                          param_dim,
                          master_param.dims()));
  }

  param_out->set_dims(param_dim);
  param_out->set_dtype(param.dtype());

  beta1_pow_out->set_dims(beta1_pow.dims());
  beta1_pow_out->set_dtype(beta1_pow.dtype());
  beta2_pow_out->set_dims(beta2_pow.dims());
  beta2_pow_out->set_dtype(beta2_pow.dtype());
  rho_out->set_dims(rho.dims());
  rho_out->set_dtype(rho.dtype());

  moment1_out->set_dims(param_dim);
  moment1_out->set_dtype(moment1.dtype());
  moment2_out->set_dims(param_dim);
  moment2_out->set_dtype(moment2.dtype());

  if (multi_precision && master_param.initialized()) {
    auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                   param.dtype() == phi::DataType::BFLOAT16)
                      ? phi::DataType::FLOAT32
                      : param.dtype();
    master_param_out->set_dims(param_dim);
    master_param_out->set_dtype(MPType);
  }
}

void RmsNormInferMeta(const MetaTensor& x,
                      const MetaTensor& bias,
                      const MetaTensor& residual,
                      const MetaTensor& norm_weight,
                      const MetaTensor& norm_bias,
                      const float epsilon,
                      const int begin_norm_axis,
                      const float quant_scale,
                      const int quant_round_type,
                      const float quant_max_bound,
                      const float quant_min_bound,
                      MetaTensor* out,
                      MetaTensor* residual_out,
                      MetaTensor* inv_var) {
  size_t x_dims_size = x.dims().size();

  size_t normalized_dims = 1;
  for (size_t i = begin_norm_axis; i < x_dims_size; ++i) {
    normalized_dims *= x.dims().at(i);
  }

  PADDLE_ENFORCE_EQ(normalized_dims,
                    norm_weight.dims()[0],
                    common::errors::InvalidArgument(
                        "The normalized size of Input(X) must equal to be "
                        "the size of Weight, but received "
                        "normalized size of Input(X) is [%d], received size "
                        "of Weight is [%d]",
                        normalized_dims,
                        norm_weight.dims()[0]));

  out->set_dims(x.dims());

  if (quant_scale > 0) {
    if (fabs(quant_max_bound - 127.0f) < 0.000001) {
      out->set_dtype(phi::DataType::INT8);
    } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
      out->set_dtype(phi::DataType::FLOAT8_E4M3FN);
    }
  } else {
    out->set_dtype(x.dtype());
  }
  out->set_layout(x.layout());
  out->share_lod(x);

  if (inv_var != nullptr) {
    inv_var->set_dtype(phi::DataType::FLOAT32);
    std::vector<int64_t> inv_var_dims;
    for (size_t i = size_t(0); i < static_cast<size_t>(begin_norm_axis); i++) {
      inv_var_dims.push_back(x.dims().at(i));
    }
    inv_var->set_dims(common::make_ddim(inv_var_dims));
    inv_var->set_layout(x.layout());
  }

  if (residual != nullptr) {
    residual_out->set_dims(x.dims());
    residual_out->set_dtype(x.dtype());
    residual_out->set_layout(x.layout());
    residual_out->share_lod(x);
  }
}

void RmspropInferMeta(const MetaTensor& param,
                      const MetaTensor& mean_square,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& mean_grad,
                      const MetaTensor& master_param,
                      float epsilon,
                      float decay,
                      float momentum,
                      bool centered,
                      bool multi_precision,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* mean_square_out,
                      MetaTensor* mean_grad_out,
                      MetaTensor* master_param_outs) {
  if (centered) {
    PADDLE_ENFORCE_NOT_NULL(
        mean_grad_out,
        common::errors::InvalidArgument(
            "Output(MeanGradOut) of RmspropOp should not be null."));
  }

  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(param_dim,
                    moment.dims(),
                    common::errors::InvalidArgument(
                        "Param and Momentum input of RmspropOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and Moment [%s]",
                        param_dim,
                        moment.dims()));
  PADDLE_ENFORCE_EQ(param_dim,
                    mean_square.dims(),
                    common::errors::InvalidArgument(
                        "Param and Momentum input of RmspropOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and MeanSquare [%s]",
                        param_dim,
                        mean_square.dims()));

  auto lr_dim = learning_rate.dims();
  PADDLE_ENFORCE_EQ(common::product(lr_dim),
                    1,
                    common::errors::InvalidArgument(
                        "Learning Rate of RmspropOp should be a scalar. But "
                        "received LearningRate's dim [%s]",
                        common::product(lr_dim)));

  if (master_param.initialized()) {
    PADDLE_ENFORCE_EQ(param_dim,
                      master_param.dims(),
                      errors::InvalidArgument(
                          "Param and MasterParam input of RmspropOp should "
                          "have same dimension. But "
                          "received Param dims: [%s], MasterParam dims: [%s].",
                          param_dim,
                          master_param.dims()));
  }

  param_out->set_dims(param_dim);
  param_out->set_dtype(param.dtype());
  moment_out->set_dims(param_dim);
  moment_out->set_dtype(moment.dtype());
  mean_square_out->set_dims(param_dim);
  mean_square_out->set_dtype(mean_square.dtype());
  if (centered) {
    mean_grad_out->set_dims(param_dim);
    mean_grad_out->set_dtype(mean_grad.dtype());
  }
  if (multi_precision && master_param.initialized()) {
    auto MPType = (param.dtype() == phi::DataType::FLOAT16 ||
                   param.dtype() == phi::DataType::BFLOAT16)
                      ? phi::DataType::FLOAT32
                      : param.dtype();
    master_param_outs->set_dims(param_dim);
    master_param_outs->set_dtype(MPType);
  }
}

void RnnInferMeta(const MetaTensor& x,
                  const std::vector<const MetaTensor*>& pre_state,
                  const std::vector<const MetaTensor*>& weight_list,
                  const MetaTensor& sequence_length,
                  float dropout_prob,
                  bool is_bidirec,
                  int input_size,
                  int hidden_size,
                  int num_layers,
                  const std::string& mode,
                  int seed,
                  bool is_test,
                  MetaTensor* out,
                  MetaTensor* dropout_state,
                  std::vector<MetaTensor*> state,
                  MetaTensor* reserve) {
  auto in_dims = x.dims();

  PADDLE_ENFORCE_EQ(in_dims.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of Input in RNN  must be 3. But "
                        "received Input's rank is %d.",
                        in_dims.size()));

  if (sequence_length) {
    auto seq_dims = sequence_length.dims();
    PADDLE_ENFORCE_EQ(
        in_dims[1],
        seq_dims[0],
        common::errors::InvalidArgument(
            "The size of SequenceLength has to equal the batch_size. But "
            "received batch_size is %d and the size of SequenceLength is %d.",
            in_dims[1],
            seq_dims[0]));
  }

  PADDLE_ENFORCE_EQ(pre_state[0]->dims().size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of PreState in RNN  must be 3. But "
                        "the received rank is %d.",
                        pre_state[0]->dims().size()));
  size_t i = 0;
  for (; i < pre_state.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        in_dims[1],
        pre_state[i]->dims()[1],
        common::errors::InvalidArgument(
            "The second dimension size (representing for batch size) of "
            "Input and PreState should be equal. But received %d and %d.",
            in_dims[1],
            pre_state[i]->dims()[1]));
    PADDLE_ENFORCE_EQ(
        pre_state[0]->dims(),
        pre_state[i]->dims(),
        common::errors::InvalidArgument(
            "The dims of all tensors in PreState should be same. But "
            "received PreState[0] is %s and PreState[%d] is %s.",
            pre_state[0]->dims(),
            i,
            pre_state[i]->dims()));
  }
  size_t num_state = mode == "LSTM" ? 2 : 1;
  PADDLE_ENFORCE_EQ(i,
                    num_state,
                    common::errors::InvalidArgument(
                        "The number of tensors in PreState of %s should be %d, "
                        "but received %d.",
                        mode,
                        2,
                        i));

  auto out_dims = in_dims;
  out_dims[2] = is_bidirec ? hidden_size * 2 : hidden_size;
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());

  int state_num = static_cast<int>(pre_state.size());
  for (int i = 0; i < state_num; ++i) {
    state[i]->set_dims(pre_state[i]->dims());
    state[i]->set_dtype(x.dtype());
  }
}

void RpropInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& prev,
                    const MetaTensor& learning_rate,
                    const MetaTensor& master_param,
                    const MetaTensor& learning_rate_range,
                    const MetaTensor& etas,
                    bool multi_precision,
                    MetaTensor* param_out,
                    MetaTensor* prev_out,
                    MetaTensor* learning_rate_out,
                    MetaTensor* master_param_out) {
  PADDLE_ENFORCE_NOT_NULL(
      param_out,
      common::errors::InvalidArgument(
          "Output(ParamOut) of RpropOp should not be null."));

  PADDLE_ENFORCE_NOT_NULL(
      prev_out,
      common::errors::InvalidArgument(
          "Output(PrevOut) of RpropOp should not be null."));

  PADDLE_ENFORCE_NOT_NULL(
      learning_rate_out,
      common::errors::InvalidArgument(
          "Output(LearningRateOut) of RpropOp should not be null."));

  param_out->set_dims(param.dims());
  param_out->set_dtype(param.dtype());
  prev_out->set_dims(prev.dims());
  prev_out->set_dtype(prev.dtype());
  learning_rate_out->set_dims(learning_rate.dims());
  learning_rate_out->set_dtype(learning_rate.dtype());
  if (multi_precision) {
    master_param_out->set_dims(master_param.dims());
    if (DataType::FLOAT16 == master_param.dtype() ||
        DataType::BFLOAT16 == master_param.dtype()) {
      master_param_out->set_dtype(DataType::FLOAT32);
    } else {
      master_param_out->set_dtype(master_param.dtype());
    }
  }
}

void SgdInferMeta(const MetaTensor& param,
                  const MetaTensor& learning_rate,
                  const MetaTensor& grad,
                  const MetaTensor& master_param,
                  bool multi_precision,
                  MetaTensor* param_out,
                  MetaTensor* master_param_out) {
  PADDLE_ENFORCE_NOT_NULL(param_out,
                          common::errors::InvalidArgument(
                              "Output(ParamOut) of SGDOp should not be null."));

  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(common::product(lr_dims),
                    1,
                    common::errors::InvalidArgument(
                        "Learning rate should have 1 element. But received "
                        "LearningRate dims [%s]",
                        common::product(lr_dims)));

  param_out->set_dims(param.dims());
  param_out->set_dtype(param.dtype());
  if (multi_precision) {
    master_param_out->set_dims(master_param.dims());
    if (DataType::FLOAT16 == master_param.dtype() ||
        DataType::BFLOAT16 == master_param.dtype()) {
      master_param_out->set_dtype(DataType::FLOAT32);
    } else {
      master_param_out->set_dtype(master_param.dtype());
    }
  }
}

void SigmoidCrossEntropyWithLogitsInferMeta(const MetaTensor& x,
                                            const MetaTensor& label,
                                            const MetaTensor& pos_weight,
                                            bool normalize,
                                            int ignore_index,
                                            MetaTensor* out,
                                            MetaConfig config) {
  auto x_dims = x.dims();
  auto labels_dims = label.dims();
  int rank = x_dims.size();
  PADDLE_ENFORCE_EQ(rank,
                    labels_dims.size(),
                    common::errors::InvalidArgument(
                        "Input(X) and Input(Label) shall have the same rank."
                        "But received: the rank of Input(X) is [%d], "
                        "the rank of Input(Label) is [%d].",
                        rank,
                        labels_dims.size()));

  bool check = true;
  if ((!config.is_runtime) &&
      (contain_unknown_dim(x_dims) || contain_unknown_dim(labels_dims))) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(
        common::slice_ddim(x_dims, 0, rank),
        common::slice_ddim(labels_dims, 0, rank),
        common::errors::InvalidArgument(
            "Input(X) and Input(Label) shall have the same shape "
            "except the last dimension. But received: the shape of "
            "Input(X) is [%s], the shape of Input(Label) is [%s].",
            x_dims,
            labels_dims));

    if (pos_weight) {
      auto weight_dims = pos_weight.dims();
      PADDLE_ENFORCE_EQ(
          common::slice_ddim(weight_dims, 0, rank),
          common::slice_ddim(labels_dims, 0, rank),
          common::errors::InvalidArgument(
              "Input(pos_weight) and Input(Label) shall have the same shape "
              "But received: the shape of Input(PosWeight) is [%s], "
              "the shape of Input(Label) is [%s].",
              weight_dims,
              labels_dims));
    }
  }

  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void SendUERecvInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& src_index,
                         const MetaTensor& dst_index,
                         const std::string& message_op,
                         const std::string& reduce_op,
                         const IntArray& out_size,
                         MetaTensor* out,
                         MetaTensor* dst_count) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        common::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        common::errors::InvalidArgument("The Dst_index should be 1D, "
                                        "when it is not 2D, but we get %d",
                                        dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    common::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      y_dims[0],
      src_index_dims[0],
      common::errors::InvalidArgument(
          "Expect Input Y to have size %d as Src_index on the first dimension, "
          "but we get %d",
          src_index_dims[0],
          y_dims[0]));

  auto x_dims = x.dims();
  if (reduce_op == "MEAN") {
    dst_count->set_dims({-1});
    dst_count->set_dtype(DataType::INT32);
  }

  // Infer out's shape according to x and e(need broadcasting condition)
  out->set_dtype(x.dtype());
  auto x_dims1 = common::vectorize<int>(x_dims);
  auto y_dims1 = common::vectorize<int>(y_dims);
  std::vector<int> x_dims2(x_dims1.begin() + 1, x_dims1.end());
  std::vector<int> y_dims2(y_dims1.begin() + 1, y_dims1.end());

  int max_dim = static_cast<int>(std::max(x_dims2.size(), y_dims2.size()));
  int axis = std::abs(static_cast<int>(x_dims2.size() - y_dims2.size()));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  // Only need to broadcast dimensions other than the 0th dimension.
  phi::funcs::GetBroadcastDimsArrays(common::make_ddim(x_dims2),
                                     common::make_ddim(y_dims2),
                                     x_dims_array.data(),
                                     y_dims_array.data(),
                                     out_dims_array.data(),
                                     max_dim,
                                     axis);
  out_dims_array.insert(out_dims_array.begin(), -1);
  out->set_dims(common::make_ddim(out_dims_array));
}

void SendUVInferMeta(const MetaTensor& x,
                     const MetaTensor& y,
                     const MetaTensor& src_index,
                     const MetaTensor& dst_index,
                     const std::string& message_op,
                     MetaTensor* out) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        common::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      common::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        common::errors::InvalidArgument("The Dst_index should be 1D, "
                                        "when it is not 2D, but we get %d",
                                        dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    common::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  // Infer out's shape according to x and y(need broadcasting condition)
  out->set_dtype(x.dtype());
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto x_dims1 = common::vectorize<int>(x_dims);
  auto y_dims1 = common::vectorize<int>(y_dims);
  std::vector<int> x_dims2(x_dims1.begin() + 1, x_dims1.end());
  std::vector<int> y_dims2(y_dims1.begin() + 1, y_dims1.end());
  int max_dim = static_cast<int>(std::max(x_dims2.size(), y_dims2.size()));
  int axis = std::abs(static_cast<int>(x_dims2.size() - y_dims2.size()));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  // Only need to broadcast dimensions other than the 0th dimension.
  phi::funcs::GetBroadcastDimsArrays(common::make_ddim(x_dims2),
                                     common::make_ddim(y_dims2),
                                     x_dims_array.data(),
                                     y_dims_array.data(),
                                     out_dims_array.data(),
                                     max_dim,
                                     axis);
  out_dims_array.insert(out_dims_array.begin(), src_index_dims[0]);  // NOLINT
  out->set_dims(common::make_ddim(out_dims_array));
}

void SparseAttentionInferMeta(const MetaTensor& q,
                              const MetaTensor& k,
                              const MetaTensor& v,
                              const MetaTensor& offset,
                              const MetaTensor& columns,
                              const MetaTensor& key_padding_mask,
                              const MetaTensor& attn_mask,
                              MetaTensor* out,
                              MetaTensor* sparse_dot_sdd,
                              MetaTensor* softmax) {
  const auto& dims_q = q.dims();
  const auto& dims_k = k.dims();
  const auto& dims_v = v.dims();
  const auto& dims_columns = columns.dims();

  PADDLE_ENFORCE_EQ(dims_q.size(),
                    static_cast<size_t>(4),
                    common::errors::InvalidArgument(
                        "Dimension in query' shapes should be 4."));
  PADDLE_ENFORCE_EQ(
      dims_k.size(),
      static_cast<size_t>(4),
      common::errors::InvalidArgument("Dimension in key' shapes should be 4."));
  PADDLE_ENFORCE_EQ(dims_v.size(),
                    static_cast<size_t>(4),
                    common::errors::InvalidArgument(
                        "Dimension in value' shapes should be 4."));

  auto batch_size = dims_q[0];
  auto num_heads = dims_q[1];
  auto M = dims_q[2];
  auto N = dims_q[3];
  auto sparse_nnz = dims_columns[2];
  out->set_dims({batch_size, num_heads, M, N});
  sparse_dot_sdd->set_dims({batch_size, num_heads, sparse_nnz});
  softmax->set_dims({batch_size, num_heads, sparse_nnz});
  out->share_lod(q);
  out->set_dtype(q.dtype());
}

void SparseMomentumInferMeta(const MetaTensor& param,
                             const MetaTensor& grad,
                             const MetaTensor& velocity,
                             const MetaTensor& index,
                             const MetaTensor& learning_rate,
                             MetaTensor* param_out,
                             MetaTensor* velocity_out,
                             MetaTensor* master_param_out) {
  auto lr_dims = common::product(learning_rate.dims());
  PADDLE_ENFORCE_EQ(lr_dims == 1,
                    true,
                    common::errors::InvalidArgument(
                        "Learning_rate should be a scalar. But Received "
                        "LearningRate's dim [%s]",
                        lr_dims));
  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dim,
      velocity.dims(),
      common::errors::InvalidArgument(
          "Param and Velocity of SparseMomentumOp should have the same "
          "dimension. But received Param's dim [%s] and Velocity [%s].",
          param_dim,
          velocity.dims()));
  param_out->set_dims(param_dim);
  velocity_out->set_dims(param_dim);
  if (master_param_out != nullptr) {
    master_param_out->set_dims(param_dim);
  }
}

void StackInferMeta(const std::vector<const MetaTensor*>& x,
                    int axis,
                    MetaTensor* out,
                    MetaConfig config) {
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "Number of Inputs(x) must be larger than 0, but"
                        " received value is:%d.",
                        x.size()));
  const auto& input_dims = GetMetaTensorsDim(x);
  // we reuse concat logic to compute out_dim. we set concat_axis==-1 to check
  // every axis in input_tensors.
  auto out_dim =
      phi::funcs::ComputeAndCheckShape(config.is_runtime, input_dims, -1);
  int rank = input_dims[0].size();
  PADDLE_ENFORCE_GE(
      axis,
      -(rank + 1),
      common::errors::InvalidArgument(
          "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d, "
          "but received axis is:%d.",
          rank,
          axis));
  PADDLE_ENFORCE_LT(
      axis,
      rank + 1,
      common::errors::InvalidArgument(
          "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d, "
          "but received axis is:%d",
          rank,
          axis));
  if (axis < 0) axis += (rank + 1);
  auto vec = common::vectorize<int64_t>(out_dim);
  vec.insert(vec.begin() + axis, input_dims.size());  // NOLINT
  out->set_dims(common::make_ddim(vec));
  out->set_dtype(x.at(0)->dtype());
  out->share_lod(*x.at(0));
}

void UnchangedMultiInferMeta(const std::vector<const MetaTensor*>& x,
                             std::vector<MetaTensor*> out) {
  PADDLE_ENFORCE_EQ(
      x.size(),
      out.size(),
      common::errors::InvalidArgument(
          "Input's size should be equal to the output's size"
          "but received input size: (%d) does not equals output_size: (%d)",
          x.size(),
          out.size()));
  for (size_t i = 0; i < x.size(); ++i) {
    if (out[i]) {
      out[i]->share_meta(*x[i]);
    }
  }
}

void ShareBufferInferMeta(const std::vector<const MetaTensor*>& xs,
                          const std::vector<bool>& share_dims_and_dtype,
                          std::vector<MetaTensor*> outs,
                          std::vector<MetaTensor*> xouts) {
  if (share_dims_and_dtype.empty()) {
    return;
  }
  PADDLE_ENFORCE_EQ(xs.size(),
                    share_dims_and_dtype.size(),
                    common::errors::PermissionDenied(
                        "The input(X) and attribute share_dims_and_dtype "
                        "should have the same size, but got size of input(X) "
                        "is %d and size of share_dims_and_dtype is %d.",
                        xs.size(),
                        share_dims_and_dtype.size()));

  for (size_t i = 0; i < xs.size(); ++i) {
    if (share_dims_and_dtype[i]) {
      outs[i]->set_dims(xs[i]->dims());
      outs[i]->set_dtype(xs[i]->dtype());
    }
  }
}

void UpdateLossScalingInferMeta(const std::vector<const MetaTensor*>& xs,
                                const MetaTensor& found_infinite,
                                const MetaTensor& prev_loss_scaling,
                                const MetaTensor& in_good_steps,
                                const MetaTensor& in_bad_steps,
                                std::vector<MetaTensor*> outs,
                                MetaTensor* loss_scaling,
                                MetaTensor* out_good_steps,
                                MetaTensor* out_bad_steps) {
  PADDLE_ENFORCE_EQ(xs.size(),
                    outs.size(),
                    common::errors::InvalidArgument(
                        "The input(X) and output(Out) should have same size in "
                        "Operator(update_loss_scaling), size of input(X) is %d "
                        "and size of output(Out) is %d.",
                        xs.size(),
                        outs.size()));
  for (size_t i = 0; i < xs.size(); ++i) {
    if (xs[i] != nullptr && outs[i] != nullptr) {
      outs[i]->set_dims(xs[i]->dims());
      outs[i]->set_dtype(xs[i]->dtype());
    }
  }
  loss_scaling->set_dims({1});
  out_good_steps->set_dims({1});
  out_good_steps->set_dtype(DataType::INT32);
  out_bad_steps->set_dims({1});
  out_bad_steps->set_dtype(DataType::INT32);
}

void WarpctcInferMeta(const MetaTensor& logits,
                      const MetaTensor& label,
                      const MetaTensor& logits_length,
                      const MetaTensor& labels_length,
                      int blank,
                      bool norm_by_times,
                      MetaTensor* loss,
                      MetaTensor* warpctcgrad) {
  auto logits_dims = logits.dims();
  int num_sequences, sequence_width, max_sequence_length;

  if (logits_length && labels_length) {
    max_sequence_length = static_cast<int>(logits_dims[0]);
    num_sequences = static_cast<int>(logits_dims[1]);
    sequence_width = static_cast<int>(logits_dims[2]);
  } else {
    max_sequence_length = -1;
    num_sequences = -1;
    sequence_width =
        static_cast<int>(common::product(logits_dims) / logits_dims[0]);
  }

  PADDLE_ENFORCE_GE(
      blank,
      0,
      errors::InvalidArgument(
          "The value of Attr(blank) should be in interval [0, %d), "
          "but received %d",
          blank));
  PADDLE_ENFORCE_LT(
      blank,
      sequence_width,
      errors::InvalidArgument(
          "The value of Attr(blank) should be in interval [0, %d), "
          "but received %d",
          blank));

  loss->set_dims({num_sequences, 1});
  loss->set_dtype(logits.dtype());
  warpctcgrad->set_dims({max_sequence_length, num_sequences, sequence_width});
  warpctcgrad->set_dtype(logits.dtype());
}

void WarprnntInferMeta(const MetaTensor& input,
                       const MetaTensor& label,
                       const MetaTensor& input_lengths,
                       const MetaTensor& label_lengths,
                       int blank,
                       float fastemit_lambda,
                       MetaTensor* loss,
                       MetaTensor* warpctcgrad) {
  auto input_dims = input.dims();
  int D = static_cast<int>(input_dims[3]);

  PADDLE_ENFORCE_GE(
      blank,
      0,
      errors::InvalidArgument(
          "The value of Attr(blank) should be in interval [0, %d), "
          "but received %d",
          blank));
  PADDLE_ENFORCE_LT(
      blank,
      D,
      errors::InvalidArgument(
          "The value of Attr(blank) should be in interval [0, %d), "
          "but received %d",
          blank));

  loss->set_dims({input_dims[0]});
  loss->set_dtype(input.dtype());
  warpctcgrad->set_dims(input_dims);
  warpctcgrad->set_dtype(input.dtype());
}

void WeightOnlyLinearInferMeta(const MetaTensor& x,
                               const MetaTensor& weight,
                               const MetaTensor& bias,
                               const MetaTensor& weight_scale,
                               const std::string& weight_dtype,
                               const int32_t arch,
                               const int32_t group_size,
                               MetaTensor* out) {
  PADDLE_ENFORCE((group_size == -1 || group_size == 64 || group_size == 128),
                 errors::InvalidArgument("group_size must be -1, 64 or 128."));

  auto weight_scale_dims = weight_scale.dims();

  auto x_dims = x.dims();
  auto w_dims = weight.dims();
  auto n = group_size == -1 ? weight_scale_dims[0] : weight_scale_dims[1];
  PADDLE_ENFORCE(
      weight_dtype == "int8" || weight_dtype == "int4",
      errors::InvalidArgument("quant_method must be 'int8' or 'int4'."));
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2UL,
      errors::InvalidArgument("The input(weight) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      w_dims[0] % 16,
      0,
      common::errors::InvalidArgument(
          "The first dimension of input must be divisible by 16, but got[%d]",
          w_dims[0]));
  PADDLE_ENFORCE_EQ(
      w_dims[1] % 16,
      0,
      common::errors::InvalidArgument(
          "The second dimension of input must be divisible by 16, but got[%d]",
          w_dims[1]));
  PADDLE_ENFORCE_EQ(
      x_dims[x_dims.size() - 1],
      w_dims[1],
      errors::InvalidArgument(
          "Input(X) dim[-1] and Input(Weight) dim[1] should be equal."
          "But received Input(X) dim[-1](%s) != Input(Weight) dim[1](%s)",
          x_dims[x_dims.size() - 1],
          w_dims[1]));
  if (bias.initialized()) {
    auto bias_dims = bias.dims();
    PADDLE_ENFORCE_EQ(
        bias_dims.size(),
        1UL,
        errors::InvalidArgument(
            "The size of Input(Bias)'s dimension should equal to 1UL.",
            bias_dims.size()));
  }

  // per-channel dequantization
  if (group_size == -1) {
    PADDLE_ENFORCE_EQ(
        weight_scale_dims.size(),
        1UL,
        errors::InvalidArgument("The input(weight_scale) must be a 1D Tensor."
                                "in per-channel mode."));
  } else /* groupwise dequantization */ {
    PADDLE_ENFORCE_EQ(
        weight_scale_dims.size(),
        2UL,
        errors::InvalidArgument("The input(weight_scale) must be a 2D Tensor"
                                " in groupwise mode."));
    PADDLE_ENFORCE_EQ(
        weight_scale_dims[0],
        (w_dims[1] + (group_size - 1)) / group_size,
        errors::InvalidArgument("The input(weight_scale) dim[0] must be equal "
                                "to Input(weight) dim[1] / group_size"
                                "But receive %d and %d",
                                weight_scale_dims[0],
                                (w_dims[1] + (group_size - 1)) / group_size));
  }

  auto out_dims = x_dims;
  out_dims[out_dims.size() - 1] = n;
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out) {
  auto cond_dims = condition.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      cond_dims.size(),
      x_dims.size(),
      common::errors::InvalidArgument(
          "The dims of Inputs(Condition) and Inputs(X) should be same. "
          "But received Condition's rank is [%d], X's rank is [%d]",
          cond_dims.size(),
          x_dims.size()));
  size_t cond_dims_size = static_cast<size_t>(cond_dims.size());
  for (size_t i = 0; i < cond_dims_size; ++i) {
    if (cond_dims[i] == -1 || x_dims[i] == -1) {
      continue;
    }
    PADDLE_ENFORCE_EQ(
        cond_dims[i],
        x_dims[i],
        common::errors::InvalidArgument(
            "The [%d] th of Inputs(Condition) and Inputs(X) should be same. "
            "But received Condition's shape is [%d], X's shape is [%d]",
            i,
            cond_dims[i],
            x_dims[i]));
  }

  PADDLE_ENFORCE_EQ(x_dims.size(),
                    y_dims.size(),
                    common::errors::InvalidArgument(
                        "The dims of Inputs(X) and Inputs(Y) should be same. "
                        "But received X's shape is [%d], Y's shape is [%d]",
                        x_dims.size(),
                        y_dims.size()));
  size_t x_dims_size = static_cast<size_t>(x_dims.size());
  for (size_t i = 0; i < x_dims_size; ++i) {
    if (x_dims[i] == -1 || y_dims[i] == -1) {
      continue;
    }
    PADDLE_ENFORCE_EQ(
        x_dims[i],
        y_dims[i],
        common::errors::InvalidArgument(
            "The [%d] th of Inputs(X) and Inputs(Y) should be same. "
            "But received X's shape is [%s], Y's shape is [%s]",
            i,
            x_dims[i],
            y_dims[i]));
  }

  out->share_meta(x);
}

void YoloBoxPostInferMeta(const MetaTensor& boxes0,
                          const MetaTensor& boxes1,
                          const MetaTensor& boxes2,
                          const MetaTensor& image_shape,
                          const MetaTensor& image_scale,
                          const std::vector<int>& anchors0,
                          const std::vector<int>& anchors1,
                          const std::vector<int>& anchors2,
                          int class_num,
                          float conf_thresh,
                          int downsample_ratio0,
                          int downsample_ratio1,
                          int downsample_ratio2,
                          bool clip_bbox,
                          float scale_x_y,
                          float nms_threshold,
                          MetaTensor* out,
                          MetaTensor* nms_rois_num,
                          MetaConfig config) {
  int batch = image_shape.dims()[0];
  out->set_dims(common::make_ddim({1, 6}));
  nms_rois_num->set_dims(common::make_ddim({batch}));
  out->set_dtype(DataType::FLOAT32);
  nms_rois_num->set_dtype(DataType::INT32);
}

void YoloLossInferMeta(const MetaTensor& x,
                       const MetaTensor& gt_box,
                       const MetaTensor& gt_label,
                       const MetaTensor& gt_score,
                       const std::vector<int>& anchors,
                       const std::vector<int>& anchor_mask,
                       int class_num,
                       float ignore_thresh,
                       int downsample_ratio,
                       bool use_label_smooth,
                       float scale_x_y,
                       MetaTensor* loss,
                       MetaTensor* objectness_mask,
                       MetaTensor* gt_match_mask) {
  auto dim_x = x.dims();
  auto dim_gtbox = gt_box.dims();
  auto dim_gtlabel = gt_label.dims();
  int anchor_num = static_cast<int>(anchors.size() / 2);
  int mask_num = static_cast<int>(anchor_mask.size());

  PADDLE_ENFORCE_EQ(dim_x.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input(X) should be a 4-D tensor. But received "
                        "X dimension size(%s)",
                        dim_x.size()));
  PADDLE_ENFORCE_EQ(dim_x[2],
                    dim_x[3],
                    common::errors::InvalidArgument(
                        "Input(X) dim[3] and dim[4] should be equal."
                        "But received dim[3](%s) != dim[4](%s)",
                        dim_x[2],
                        dim_x[3]));
  PADDLE_ENFORCE_EQ(
      dim_x[1],
      mask_num * (5 + class_num),
      common::errors::InvalidArgument(
          "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
          "+ class_num))."
          "But received dim[1](%s) != (anchor_mask_number * "
          "(5+class_num)(%s).",
          dim_x[1],
          mask_num * (5 + class_num)));
  PADDLE_ENFORCE_EQ(dim_gtbox.size(),
                    3,
                    common::errors::InvalidArgument(
                        "Input(GTBox) should be a 3-D tensor, but "
                        "received gtbox dimension size(%s)",
                        dim_gtbox.size()));
  PADDLE_ENFORCE_EQ(
      dim_gtbox[2],
      4,
      common::errors::InvalidArgument("Input(GTBox) dim[2] should be 4",
                                      "But receive dim[2](%s) != 5. ",
                                      dim_gtbox[2]));
  PADDLE_ENFORCE_EQ(dim_gtlabel.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Input(GTLabel) should be a 2-D tensor,"
                        "But received Input(GTLabel) dimension size(%s) != 2.",
                        dim_gtlabel.size()));
  PADDLE_ENFORCE_EQ(
      dim_gtlabel[0],
      dim_gtbox[0],
      common::errors::InvalidArgument(
          "Input(GTBox) dim[0] and Input(GTLabel) dim[0] should be same,"
          "But received Input(GTLabel) dim[0](%s) != "
          "Input(GTBox) dim[0](%s)",
          dim_gtlabel[0],
          dim_gtbox[0]));
  PADDLE_ENFORCE_EQ(
      dim_gtlabel[1],
      dim_gtbox[1],
      common::errors::InvalidArgument(
          "Input(GTBox) and Input(GTLabel) dim[1] should be same,"
          "But received Input(GTBox) dim[1](%s) != Input(GTLabel) "
          "dim[1](%s)",
          dim_gtbox[1],
          dim_gtlabel[1]));
  PADDLE_ENFORCE_GT(anchors.size(),
                    0,
                    common::errors::InvalidArgument(
                        "Attr(anchors) length should be greater then 0."
                        "But received anchors length(%s)",
                        anchors.size()));
  PADDLE_ENFORCE_EQ(anchors.size() % 2,
                    0,
                    common::errors::InvalidArgument(
                        "Attr(anchors) length should be even integer."
                        "But received anchors length(%s)",
                        anchors.size()));
  for (auto& item : anchor_mask) {
    PADDLE_ENFORCE_LT(
        item,
        anchor_num,
        common::errors::InvalidArgument(
            "Attr(anchor_mask) should not crossover Attr(anchors)."
            "But received anchor_mask[i](%s) > anchor_num(%s)",
            item,
            anchor_num));
  }
  PADDLE_ENFORCE_GT(class_num,
                    0,
                    common::errors::InvalidArgument(
                        "Attr(class_num) should be an integer greater then 0."
                        "But received class_num(%s) < 0",
                        class_num));

  if (gt_score) {
    auto dim_gtscore = gt_score.dims();
    PADDLE_ENFORCE_EQ(
        dim_gtscore.size(),
        2,
        common::errors::InvalidArgument("Input(GTScore) should be a 2-D tensor"
                                        "But received GTScore dimension(%s)",
                                        dim_gtbox.size()));
    PADDLE_ENFORCE_EQ(
        dim_gtscore[0],
        dim_gtbox[0],
        common::errors::InvalidArgument(
            "Input(GTBox) and Input(GTScore) dim[0] should be same"
            "But received GTBox dim[0](%s) != GTScore dim[0](%s)",
            dim_gtbox[0],
            dim_gtscore[0]));
    PADDLE_ENFORCE_EQ(
        dim_gtscore[1],
        dim_gtbox[1],
        common::errors::InvalidArgument(
            "Input(GTBox) and Input(GTScore) dim[1] should be same"
            "But received GTBox dim[1](%s) != GTScore dim[1](%s)",
            dim_gtscore[1],
            dim_gtbox[1]));
  }

  std::vector<int64_t> dim_out({dim_x[0]});
  loss->set_dims(common::make_ddim(dim_out));
  loss->set_dtype(x.dtype());

  std::vector<int64_t> dim_obj_mask({dim_x[0], mask_num, dim_x[2], dim_x[3]});
  objectness_mask->set_dims(common::make_ddim(dim_obj_mask));
  objectness_mask->set_dtype(x.dtype());

  std::vector<int64_t> dim_gt_match_mask({dim_gtbox[0], dim_gtbox[1]});
  gt_match_mask->set_dims(common::make_ddim(dim_gt_match_mask));
  gt_match_mask->set_dtype(x.dtype());
}

void FusedAdamInferMeta(
    const std::vector<const MetaTensor*>& params,
    const std::vector<const MetaTensor*>& grads,
    const MetaTensor& learning_rate,
    const std::vector<const MetaTensor*>& moments1,
    const std::vector<const MetaTensor*>& moments2,
    const paddle::optional<std::vector<const MetaTensor*>>& moments2_max,
    const std::vector<const MetaTensor*>& beta1_pows,
    const std::vector<const MetaTensor*>& beta2_pows,
    const paddle::optional<std::vector<const MetaTensor*>>& master_params,
    const MetaTensor& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    int chunk_size,
    float weight_decay,
    bool use_adamw,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,
    std::vector<MetaTensor*> params_out,
    std::vector<MetaTensor*> moments1_out,
    std::vector<MetaTensor*> moments2_out,
    std::vector<MetaTensor*> moments2_max_out,
    std::vector<MetaTensor*> beta1_pows_out,
    std::vector<MetaTensor*> beta2_pows_out,
    std::vector<MetaTensor*> master_params_out) {
  size_t in_size = params.size();
  for (size_t i = 0; i < in_size; i++) {
    params_out[i]->set_dims(params[i]->dims());
    params_out[i]->set_dtype(params[i]->dtype());
    moments1_out[i]->set_dims(moments1[i]->dims());
    moments1_out[i]->set_dtype(moments1[i]->dtype());
    moments2_out[i]->set_dims(moments2[i]->dims());
    moments2_out[i]->set_dtype(moments2[i]->dtype());
    if (amsgrad) {
      moments2_max_out[i]->set_dims(moments2_max.get()[i]->dims());
      moments2_max_out[i]->set_dtype(moments2_max.get()[i]->dtype());
    }
    beta1_pows_out[i]->set_dims(beta1_pows[i]->dims());
    beta1_pows_out[i]->set_dtype(beta1_pows[i]->dtype());
    beta2_pows_out[i]->set_dims(beta2_pows[i]->dims());
    beta2_pows_out[i]->set_dtype(beta2_pows[i]->dtype());
    if (master_params && !master_params_out.empty()) {
      master_params_out[i]->set_dims(master_params.get()[i]->dims());
      master_params_out[i]->set_dtype(master_params.get()[i]->dtype());
    }
  }
}

void FusedConvInferMeta(const MetaTensor& input,
                        const MetaTensor& filter,
                        const MetaTensor& bias,
                        const MetaTensor& residual_param,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& padding_algorithm,
                        const std::vector<int>& dilations,
                        int groups,
                        const std::string& data_format,
                        const std::string& mkldnn_data_type,
                        const std::string& fuse_activation,
                        bool fuse_residual_conn,
                        bool force_fp32_output,
                        MetaTensor* out,
                        MetaConfig config) {
  ConvInferMeta(input,
                filter,
                strides,
                paddings,
                padding_algorithm,
                dilations,
                groups,
                data_format,
                out,
                config);
}

void FusedRopeInferMeta(const MetaTensor& q,
                        const MetaTensor& k,
                        const MetaTensor& v,
                        const MetaTensor& sin,
                        const MetaTensor& cos,
                        const MetaTensor& position_ids,
                        bool use_neox_rotary_style,
                        bool time_major,
                        float rotary_emb_base,
                        MetaTensor* out_q,
                        MetaTensor* out_k,
                        MetaTensor* out_v) {
  auto input_dims = q.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    common::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        input_dims.size()));
  out_q->set_dims(q.dims());
  out_q->set_dtype(q.dtype());
  if (k) {
    out_k->set_dims(k.dims());
    out_k->set_dtype(k.dtype());
  } else {
    if (out_k) {
      out_k->set_dtype(q.dtype());
    }
  }
  if (v) {
    out_v->set_dims(v.dims());
    out_v->set_dtype(v.dtype());
  } else {
    if (out_v) {
      out_v->set_dtype(q.dtype());
    }
  }
}

void FusedMoeInferMeta(const MetaTensor& X,
                       const MetaTensor& gate_weight,
                       const MetaTensor& ffn1_weight,
                       const MetaTensor& ffn1_scale,
                       const MetaTensor& ffn1_bias,
                       const MetaTensor& ffn2_weight,
                       const MetaTensor& ffn2_scale,
                       const MetaTensor& ffn2_bias,
                       const std::string& quant_method,
                       const int moe_topk,
                       const bool group_moe,
                       const bool norm_topk_prob,
                       MetaTensor* out) {
  out->set_dims(X.dims());
  out->share_lod(X);
  out->set_dtype(X.dtype());
  out->set_layout(X.layout());
}

void WeightedSampleNeighborsInferMeta(const MetaTensor& row,
                                      const MetaTensor& col_ptr,
                                      const MetaTensor& edge_weight,
                                      const MetaTensor& x,
                                      const MetaTensor& eids,
                                      int sample_size,
                                      bool return_eids,
                                      MetaTensor* out,
                                      MetaTensor* out_count,
                                      MetaTensor* out_eids) {
  // GSN: GraphSampleNeighbors
  auto GSNShapeCheck = [](const phi::DDim& dims, std::string tensor_name) {
    if (dims.size() == 2) {
      PADDLE_ENFORCE_EQ(dims[1],
                        1,
                        common::errors::InvalidArgument(
                            "The last dim of %s should be 1 when it "
                            "is 2D, but we get %d",
                            tensor_name,
                            dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dims.size(),
          1,
          common::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              dims.size()));
    }
  };

  GSNShapeCheck(row.dims(), "row");
  GSNShapeCheck(col_ptr.dims(), "colptr");
  GSNShapeCheck(edge_weight.dims(), "edge_weight");
  GSNShapeCheck(x.dims(), "input_nodes");
  if (return_eids) {
    GSNShapeCheck(eids.dims(), "eids");
    out_eids->set_dims({-1});
    out_eids->set_dtype(row.dtype());
  }

  out->set_dims({-1});
  out->set_dtype(row.dtype());
  out_count->set_dims({-1});
  out_count->set_dtype(DataType::INT32);
}

void MultiheadMatmulInferMeta(const MetaTensor& input,
                              const MetaTensor& w,
                              const MetaTensor& bias,
                              const MetaTensor& bias_qk,
                              const bool transpose_q,
                              const bool transpose_k,
                              const bool transpose_v,
                              const float alpha,
                              const int head_number,
                              MetaTensor* out) {
  auto w_dims = w.dims();
  PADDLE_ENFORCE_GT(
      w_dims.size(),
      2,
      errors::InvalidArgument(
          "MultiheadMatmul's w is expected at least a 3-D tensor, but "
          "it's %d-D tensor now.",
          w_dims.size()));

  auto bias_dims = bias.dims();
  PADDLE_ENFORCE_GT(
      bias_dims.size(),
      1,
      errors::InvalidArgument(
          "MultiheadMatmul's bias should be at least 2-D tensor, but it's "
          "%d-D tensor now.",
          bias_dims.size()));

  out->set_dims(input.dims());
  out->set_dtype(input.dtype());
  out->share_lod(input);
}

void MoeDispatchInferMeta(const MetaTensor& X,
                          const MetaTensor& gating_output,
                          const int moe_topk,
                          const bool group_moe,
                          const bool topk_only_mode,
                          MetaTensor* permute_input,
                          MetaTensor* token_nums_per_expert,
                          MetaTensor* permute_indices_per_token,
                          MetaTensor* expert_scales_float,
                          MetaTensor* top_k_indices) {
  int token_rows = 0;
  auto input_dims = X.dims();
  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = X.dims()[input_dims.size() - 1];

  permute_input->set_dims({moe_topk * num_rows, hidden_size});
  permute_input->set_dtype(X.dtype());
  permute_input->set_layout(X.layout());

  permute_indices_per_token->set_dims({moe_topk, num_rows});
  permute_indices_per_token->set_dtype(DataType::INT32);
  permute_indices_per_token->set_layout(X.layout());

  expert_scales_float->set_dims({num_rows, moe_topk});
  expert_scales_float->set_dtype(DataType::FLOAT32);
  expert_scales_float->set_layout(X.layout());

  top_k_indices->set_dims({num_rows, moe_topk});
  top_k_indices->set_dtype(DataType::INT32);
  top_k_indices->set_layout(X.layout());
}

void MoeFFNInferMeta(const MetaTensor& permute_input,
                     const MetaTensor& token_nums_per_expert,
                     const MetaTensor& ffn1_weight,
                     const MetaTensor& ffn2_weight,
                     const MetaTensor& ffn1_bias,
                     const MetaTensor& ffn1_scale,
                     const MetaTensor& ffn2_scale,
                     const std::string& quant_method,
                     MetaTensor* ffn_out) {
  ffn_out->set_dims(permute_input.dims());
  ffn_out->share_lod(permute_input);
  ffn_out->set_dtype(permute_input.dtype());
  ffn_out->set_layout(permute_input.layout());
}

void MoeReduceInferMeta(const MetaTensor& ffn_out,
                        const MetaTensor& expert_scales_float,
                        const MetaTensor& permute_indices_per_token,
                        const MetaTensor& top_k_indices,
                        const MetaTensor& ffn2_bias,
                        const bool norm_topk_prob,
                        const float routed_scaling_factor,
                        MetaTensor* output) {
  auto ffn_out_dims = ffn_out.dims();
  const int top_k = top_k_indices.dims()[1];
  const int num_rows = ffn_out_dims[0] / top_k;
  const int hidden_size = ffn_out_dims[1];
  output->set_dims({num_rows, hidden_size});
  output->set_dtype(ffn_out.dtype());
  output->set_layout(ffn_out.layout());
}

void MaskedMultiheadAttentionInferMeta(const MetaTensor& x,
                                       const MetaTensor& cache_kv,
                                       const MetaTensor& bias,
                                       const MetaTensor& src_mask,
                                       const MetaTensor& cum_offsets,
                                       const MetaTensor& sequence_lengths,
                                       const MetaTensor& rotary_tensor,
                                       const MetaTensor& beam_cache_offset,
                                       const MetaTensor& qkv_out_scale,
                                       const MetaTensor& out_shift,
                                       const MetaTensor& out_smooth,
                                       int seq_len,
                                       int rotary_emb_dims,
                                       const bool use_neox_rotary_style,
                                       const std::string& compute_dtype,
                                       const float out_scale,
                                       const int quant_round_type,
                                       const float quant_max_bound,
                                       const float quant_min_bound,
                                       MetaTensor* out,
                                       MetaTensor* cache_kv_out,
                                       MetaTensor* beam_cache_offset_out) {
  int bsz = static_cast<int>(x.dims()[0]);
  auto cache_kv_dims = cache_kv.dims();
  int k_num_head = static_cast<int>(cache_kv.dims()[2]);
  int v_num_head = k_num_head;
  int dim_head = static_cast<int>(cache_kv.dims()[4]);
  // below's num_head is q's head actually.
  int num_head = x.dims()[x.dims().size() - 1] / dim_head - k_num_head -
                 v_num_head;  // NOLINT

  PADDLE_ENFORCE_EQ(
      num_head % k_num_head,
      0,
      errors::InvalidArgument(
          "The num_head of query must be divisible by the num_head of key, but "
          "received num_head of query is %d, and the num_head of key is %d",
          num_head,
          k_num_head));
  PADDLE_ENFORCE_EQ(
      cache_kv_dims.size(),
      5,
      errors::InvalidArgument("The cache_kv must be 5 dims, but got %d",
                              cache_kv_dims.size()));

  PADDLE_ENFORCE_EQ(
      cache_kv_dims[0],
      2,
      errors::InvalidArgument("The first dim of cache_kv must be 2, but got %d",
                              cache_kv_dims[0]));

  if (rotary_tensor) {
    PADDLE_ENFORCE_EQ(
        rotary_tensor.dtype(),
        DataType::FLOAT32,
        errors::InvalidArgument(
            "The dtype of rotary_tensor must be float32, but got %d",
            rotary_tensor.dtype()));
  }

  out->set_dims({bsz, num_head * dim_head});

  auto FBADtypeCheck = [](const MetaTensor& check_tensor,
                          const std::string& tensor_name,
                          const std::string& compute_dtype) {
    if (compute_dtype == "bf16") {
      PADDLE_ENFORCE_EQ(
          check_tensor.dtype(),
          phi::DataType::BFLOAT16,
          common::errors::InvalidArgument(
              "Input(%s) dtype must be the same with Attr(compute_dtype)",
              tensor_name));
    } else if (compute_dtype == "fp16") {
      PADDLE_ENFORCE_EQ(
          check_tensor.dtype(),
          phi::DataType::FLOAT16,
          common::errors::InvalidArgument(
              "Input(%s) dtype must be the same with Attr(compute_dtype)",
              tensor_name));
    } else if (compute_dtype == "fp32") {
      PADDLE_ENFORCE_EQ(
          check_tensor.dtype(),
          phi::DataType::FLOAT32,
          common::errors::InvalidArgument(
              "Input(%s) dtype must be the same with Attr(compute_dtype)",
              tensor_name));
    }
  };

  // In the case of quantization enabled, the dtype for computation is
  // determined based on compute_dtype.
  if (x.dtype() == phi::DataType::INT32) {
    PADDLE_ENFORCE_NE(
        compute_dtype,
        "default",
        common::errors::InvalidArgument(
            "If Input(x) dtype is INT32, Attr(compute_dtype) must be set."));

    if (bias) {
      FBADtypeCheck(bias, "bias", compute_dtype);
    }

    if (out_scale > 0) {
      out->set_dtype(phi::DataType::INT8);
    } else {
      if (compute_dtype == "bf16") {
        out->set_dtype(phi::DataType::BFLOAT16);
      } else if (compute_dtype == "fp16") {
        out->set_dtype(phi::DataType::FLOAT16);
      } else if (compute_dtype == "fp32") {
        out->set_dtype(phi::DataType::FLOAT32);
      } else {
        PADDLE_THROW(common::errors::InvalidArgument(
            "In the case of quantization enabled with Input(x) INT32, "
            "Attr(compute_dtype) must be set in (bf16, fp16, fp32), "
            "but get compute_dtype (%s)",
            compute_dtype));
      }
    }
  } else {
    if (bias) {
      if (compute_dtype != "default") {
        FBADtypeCheck(bias, "bias", compute_dtype);
        FBADtypeCheck(x, "x", compute_dtype);
      } else {
        PADDLE_ENFORCE_EQ(x.dtype(),
                          bias.dtype(),
                          common::errors::InvalidArgument(
                              "Input(x) and Input(bias) must be the "
                              "same dtype in this situation"));
      }
    } else {
      // bias not exist
      if (compute_dtype != "default") {
        FBADtypeCheck(x, "x", compute_dtype);
      }
    }
    if (out_scale > 0) {
      out->set_dtype(phi::DataType::INT8);
    } else {
      out->set_dtype(x.dtype());
    }
  }

  cache_kv_out->set_dims(cache_kv_dims);
  cache_kv_out->set_dtype(cache_kv.dtype());

  if (beam_cache_offset) {
    beam_cache_offset_out->set_dims(beam_cache_offset.dims());
    beam_cache_offset_out->set_dtype(beam_cache_offset.dtype());
  }
}

void FullWithTensorInferMeta(const IntArray& shape,
                             DataType dtype,
                             MetaTensor* out) {
  out->set_dims(common::make_ddim(shape.GetData()));
  out->set_dtype(dtype);
}

void TopPSamplingInferMeta(const MetaTensor& x,
                           const MetaTensor& ps,
                           const MetaTensor& threshold,
                           const MetaTensor& topp_seed,
                           int seed,
                           int k,
                           const std::string& mode,
                           MetaTensor* out,
                           MetaTensor* ids,
                           MetaTensor* topk_scores,
                           MetaTensor* topk_ids) {
  auto x_dims = x.dims();
  int bsz = x_dims[0];

  PADDLE_ENFORCE(
      mode == "truncated" || mode == "non-truncated",
      errors::InvalidArgument("mode must be 'truncated' or 'non-truncated'."));

  ids->set_dims(phi::make_ddim({bsz, 1}));
  ids->set_dtype(DataType::INT64);
  out->set_dims(phi::make_ddim({bsz, 1}));
  out->set_dtype(x.dtype());
  if (k > 0) {
    topk_ids->set_dims(phi::make_ddim({bsz, k}));
    topk_ids->set_dtype(DataType::INT64);
    topk_scores->set_dims(phi::make_ddim({bsz, k}));
    topk_scores->set_dtype(x.dtype());
  }
}

}  // namespace phi
PD_REGISTER_INFER_META_FN(batch_norm_infer, phi::BatchNormInferInferMeta);
