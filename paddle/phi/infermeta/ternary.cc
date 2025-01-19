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

#include "paddle/phi/infermeta/ternary.h"

#include "glog/logging.h"

#include "paddle/common/ddim.h"
#include "paddle/common/errors.h"
#include "paddle/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/impl/box_coder.h"

namespace phi {

void AccuracyInferMeta(const MetaTensor& out,
                       const MetaTensor& indice,
                       const MetaTensor& label,
                       MetaTensor* accuracy,
                       MetaTensor* correct,
                       MetaTensor* total,
                       MetaConfig config) {
  auto inference_dim = out.dims();
  auto label_dim = label.dims();
  // Assume indices has same shape as inference, because
  // it's the output of topk.
  PADDLE_ENFORCE_EQ(
      label_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: label's dimensions of AccuracyOp must be 2. "
          "But received label's dimensions = %d, label's shape = [%s]",
          label_dim.size(),
          label_dim));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(label_dim[1],
                      1,
                      phi::errors::InvalidArgument(
                          "ShapeError: label's second dimension of "
                          "AccuracyOp must be 1. But received label's "
                          "second dimension is = %d, label's shape = [%s]",
                          label_dim[1],
                          label_dim));
    PADDLE_ENFORCE_EQ(
        inference_dim[0],
        label_dim[0],
        phi::errors::InvalidArgument(
            "ShapeError: the output's num_rows of AccuracyOp must be"
            " the same as label's num_rows. But received output's "
            "shape = [%s], label's shape = [%s], output's num_rows = %d, "
            "label's "
            "num_rows = %d",
            inference_dim,
            label_dim,
            inference_dim[0],
            label_dim[0]));
  }

  accuracy->set_dims(common::make_ddim({}));
  correct->set_dims(common::make_ddim({}));
  total->set_dims(common::make_ddim({}));
  accuracy->set_dtype(out.dtype());
  correct->set_dtype(out.dtype());
  total->set_dtype(out.dtype());
  accuracy->share_lod(out);
}

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float beta,
                    float alpha,
                    MetaTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto ndim_input = input_dims.size();
  auto ndim_x = x_dims.size();
  auto ndim_y = y_dims.size();

  VLOG(3) << "addmm operator input.shape=" << input_dims
          << " x.shape=" << x_dims << " y.shape=" << y_dims << " beta=" << beta
          << " alpha=" << alpha << " ndim_input=" << ndim_input
          << " ndim_x=" << ndim_x << " ndim_y=" << ndim_y;

  PADDLE_ENFORCE_NE(
      product(input_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'input' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(x_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'x' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(y_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'y' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));
  // dim check
  PADDLE_ENFORCE_EQ(ndim_input == 2 || ndim_input == 1,
                    true,
                    errors::InvalidArgument(
                        "The input tensor input's dimension must be 2 or 1. "
                        "But received input's dimension = [%d].",
                        ndim_input));
  PADDLE_ENFORCE_EQ(
      ndim_x,
      2,
      errors::InvalidArgument("The input tensor x's dimension must be 2. "
                              "But received x's dimension = [%d].",
                              ndim_x));
  PADDLE_ENFORCE_EQ(
      ndim_y,
      2,
      errors::InvalidArgument("The input tensor y's dimension must be 2. "
                              "But received y's dimension = [%d].",
                              ndim_y));

  std::vector<int64_t> output_dims;
  output_dims.push_back(x_dims[0]);
  output_dims.push_back(y_dims[1]);

  out->set_dims(common::make_ddim(output_dims));
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

void AssignPosInferMeta(const MetaTensor& x,
                        const MetaTensor& cum_count,
                        const MetaTensor& eff_num_len,
                        MetaTensor* out) {
  phi::DataType X_dtype = x.dtype();
  phi::DataType cum_count_dtype = cum_count.dtype();

  PADDLE_ENFORCE_EQ(cum_count_dtype,
                    X_dtype,
                    phi::errors::InvalidArgument(
                        "The dtype of the cum_count and X should be same"));
  PADDLE_ENFORCE_EQ(cum_count_dtype,
                    phi::DataType::INT64,
                    phi::errors::InvalidArgument(
                        "The dtype of the cum_count_dtype, eff_num_len and "
                        "X should be same as int64"));
  out->set_dtype(X_dtype);
}

void BatchFCInferMeta(const MetaTensor& input,
                      const MetaTensor& w,
                      const MetaTensor& bias,
                      MetaTensor* out) {
  auto input_dims = input.dims();
  auto w_dims = w.dims();

  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      3,
      phi::errors::InvalidArgument("Input of BatchFCOp should have 3D."));
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      3,
      phi::errors::InvalidArgument("W of BatchFCOp should have 3D."));
  PADDLE_ENFORCE_EQ(
      input_dims[0],
      w_dims[0],
      phi::errors::InvalidArgument(
          "Input.dim[0] and W.dim[0] of BatchFCOp should be same."));
  PADDLE_ENFORCE_EQ(
      input_dims[2],
      w_dims[1],
      phi::errors::InvalidArgument(
          "Input.dim[2] and W.dim[1] of BatchFCOp should be same."));

  auto bias_dims = bias.dims();
  PADDLE_ENFORCE_EQ(bias_dims[0],
                    input_dims[0],
                    phi::errors::InvalidArgument(
                        "Bias.dim[0] should be same as input.dim[0]."));
  PADDLE_ENFORCE_EQ(bias_dims[1],
                    w_dims[2],
                    phi::errors::InvalidArgument(
                        "Bias.dim[1] should be same as input.dim[2]."));

  out->set_dims({input_dims[0], input_dims[1], w_dims[2]});
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

void BoxCoderInferMeta(const MetaTensor& prior_box,
                       const MetaTensor& prior_box_var,
                       const MetaTensor& target_box,
                       const std::string& code_type,
                       bool box_normalized,
                       int axis,
                       const std::vector<float>& variance,
                       MetaTensor* output_box,
                       MetaConfig config) {
  auto prior_box_dims = prior_box.dims();
  auto target_box_dims = target_box.dims();

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(prior_box_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The rank of Input PriorBox in BoxCoder operator "
                          "must be 2. But received rank = %d",
                          prior_box_dims.size()));
    PADDLE_ENFORCE_EQ(prior_box_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension of PriorBox in BoxCoder "
                          "operator must be 4. But received dimension = %d",
                          prior_box_dims[1]));
    if (prior_box_var) {
      auto prior_box_var_dims = prior_box_var.dims();
      PADDLE_ENFORCE_EQ(
          prior_box_var_dims.size(),
          2,
          phi::errors::InvalidArgument(
              "The rank of Input(PriorBoxVar) in BoxCoder operator"
              " should be 2. But received rank = %d",
              prior_box_var_dims.size()));
      PADDLE_ENFORCE_EQ(
          prior_box_dims,
          prior_box_var_dims,
          phi::errors::InvalidArgument(
              "The dimension of Input(PriorBoxVar) should be equal to"
              "the dimension of Input(PriorBox) in BoxCoder operator "
              "when the rank is 2."));
    }
  }

  auto box_code_type = phi::funcs::GetBoxCodeType(code_type);
  if (box_code_type == phi::funcs::BoxCodeType::kEncodeCenterSize) {
    PADDLE_ENFORCE_EQ(target_box_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The rank of Input TargetBox in BoxCoder operator "
                          "must be 2. But received rank is %d",
                          target_box_dims.size()));
    PADDLE_ENFORCE_EQ(target_box_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension of TargetBox in BoxCoder "
                          "operator is 4. But received dimension is %d",
                          target_box_dims[1]));
    output_box->set_dims({target_box_dims[0], prior_box_dims[0], 4});
  } else if (box_code_type == phi::funcs::BoxCodeType::kDecodeCenterSize) {
    PADDLE_ENFORCE_EQ(target_box_dims.size(),
                      3,
                      phi::errors::InvalidArgument(
                          "The rank of Input TargetBox in BoxCoder "
                          "operator must be 3. But received rank is %d",
                          target_box_dims.size()));
    PADDLE_ENFORCE_EQ(
        axis == 0 || axis == 1,
        true,
        phi::errors::InvalidArgument("axis in BoxCoder operator must be 0 or 1."
                                     "But received axis = %d",
                                     axis));
    if (config.is_runtime) {
      if (axis == 0) {
        PADDLE_ENFORCE_EQ(
            target_box_dims[1],
            prior_box_dims[0],
            phi::errors::InvalidArgument(
                "When axis is 0, The second "
                "dimension of TargetBox in BoxCoder "
                "should be equal to the first dimension of PriorBox."));
      } else if (axis == 1) {
        PADDLE_ENFORCE_EQ(
            target_box_dims[0],
            prior_box_dims[0],
            phi::errors::InvalidArgument(
                "When axis is 1, The first "
                "dimension of TargetBox in BoxCoder "
                "should be equal to the first dimension of PriorBox."));
      }
      PADDLE_ENFORCE_EQ(
          target_box_dims[2],
          prior_box_dims[1],
          phi::errors::InvalidArgument("The third dimension of TargetBox"
                                       " in BoxCoder should be equal to the "
                                       "second dimension of PriorBox."));
    }
    output_box->share_dims(target_box);
  }

  if (box_code_type == phi::funcs::BoxCodeType::kDecodeCenterSize &&
      axis == 1) {
    output_box->share_lod(prior_box);
  } else {
    output_box->share_lod(target_box);
  }
  output_box->set_dtype(target_box.dtype());
}

void CSoftmaxWithMultiLabelCrossEntropyInferMeta(
    const MetaTensor& logits,
    const MetaTensor& label,
    const MetaTensor& smooth_weight,
    int64_t ignore_index,
    bool sum_multi_label_loss,
    int ring_id,
    int rank,
    int nranks,
    MetaTensor* softmax,
    MetaTensor* loss,
    MetaConfig config) {
  auto logits_dims = logits.dims();
  auto labels_dims = label.dims();
  auto smooth_weight_dims = smooth_weight.dims();

  auto logits_rank = logits_dims.size();
  auto labels_rank = labels_dims.size();
  auto axis = logits_rank - 1;
  for (int i = 0; i < logits_rank; i++) {
    if (i != axis) {
      if (config.is_runtime || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
        PADDLE_ENFORCE_EQ(logits_dims[i],
                          labels_dims[i],
                          common::errors::InvalidArgument(
                              "Input(Logits) and Input(Label) should in "
                              "same shape in dimensions except axis."));
      }
    }
  }

  PADDLE_ENFORCE_GE(
      labels_dims[logits_rank - 1],
      1UL,
      common::errors::InvalidArgument(
          "the last dimension of Input(Label) should be greater than or equal "
          "to 1."
          "But received: the last dimension of Input(Label) is [%d],"
          "the last dimension is [%d]",
          labels_dims[logits_rank - 1],
          logits_rank - 1));

  for (int i = 0; i < labels_rank; ++i) {
    if (config.is_runtime ||
        (labels_dims[i] > 0 && smooth_weight_dims[i] > 0)) {
      PADDLE_ENFORCE_EQ(labels_dims[i],
                        smooth_weight_dims[i],
                        common::errors::InvalidArgument(
                            "Input(Label) and Input(SmoothWeight) should in "
                            "same shape in dimensions"));
    }
  }

  softmax->set_dims(logits_dims);
  if (sum_multi_label_loss) {
    labels_dims[axis] = 1;
  }
  loss->set_dims(labels_dims);
  softmax->share_lod(logits);
  loss->share_lod(logits);
}

void DistributedPushSparseInferMeta(
    const std::vector<const MetaTensor*>& ids,
    const std::vector<const MetaTensor*>& shows,
    const std::vector<const MetaTensor*>& clicks,
    int table_id,
    int size,
    bool is_distributed,
    const std::string& push_sparse_version,
    int64_t padding_idx,
    DataType dtype,
    bool is_test,
    bool use_cvm_op,
    std::vector<MetaTensor*> output) {
  auto ids_size = ids.size();
  std::vector<DDim> ids_dims;
  ids_dims.reserve(ids.size());
  for (size_t i = 1; i < ids_size; ++i) {
    PADDLE_ENFORCE_EQ(ids_dims[i].size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The dimension of the 'Ids' tensor must be 2."));
  }

  for (auto& out : output) {
    if (out == nullptr) {
      continue;
    }
    out->set_dtype(ids[0]->dtype());
  }
}

void DpsgdInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    float clip,
                    float batch_size,
                    float sigma,
                    int size,
                    MetaTensor* param_out) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(common::product(lr_dims),
                    1,
                    phi::errors::InvalidArgument(
                        "Learning rate should have 1 dimension. But Received "
                        "LearningRate's dims [%s].",
                        common::product(lr_dims)));
  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      grad.dims(),
      phi::errors::InvalidArgument(
          "Param and Grad input of DpsgdOp should have same dimension. But "
          "received Para's dim [%s] and Grad's dim [%s].",
          param_dims,
          grad.dims()));
  param_out->set_dims(param_dims);
}

void FakeQuantizeRangeAbsMaxInferMeta(const MetaTensor& x,
                                      const MetaTensor& in_scale,
                                      const MetaTensor& iter,
                                      int window_size,
                                      int bit_length,
                                      bool is_test,
                                      int round_type,
                                      MetaTensor* out,
                                      MetaTensor* out_scale,
                                      MetaTensor* out_scales) {
  PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                    true,
                    phi::errors::InvalidArgument(
                        "'bit_length' should be between 1 and 16, but "
                        "the received is %d",
                        bit_length));
  if (out_scales) {
    out_scales->set_dims({window_size});
  }
  out->set_dims(x.dims());
  out_scale->set_dims({1});
  out->share_lod(x);
}

void FlashAttnInferMeta(const MetaTensor& q,
                        const MetaTensor& k,
                        const MetaTensor& v,
                        MetaTensor* out,
                        MetaTensor* softmax,
                        MetaTensor* softmax_lse,
                        MetaTensor* seed_offset) {
  auto out_dims = q.dims();
  if (out_dims.size() == 4) {
    out_dims[3] = v.dims()[3];
  }
  out->set_dims(out_dims);
  out->set_dtype(q.dtype());
  out->set_layout(q.layout());
  softmax->set_dtype(q.dtype());
  softmax_lse->set_dtype(q.dtype());
  if (out_dims.size() == 4) {
    auto round_multiple = [](int x) { return (x + 127) / 128 * 128; };
    int batch_size = q.dims()[0];
    int num_heads = q.dims()[2];
    int seqlen_q_rounded = round_multiple(q.dims()[1]);
    int seqlen_k_rounded = round_multiple(k.dims()[1]);
    if (softmax) {
      softmax->set_dims(
          {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded});
    }
    if (softmax_lse) {
      softmax_lse->set_dims({batch_size, num_heads, seqlen_q_rounded});
    }
  }
  if (seed_offset) {
    seed_offset->set_dtype(phi::DataType::INT64);
    seed_offset->set_dims({2});
  }
}
void FlashAttnQKVPackedInferMeta(const MetaTensor& qkv,
                                 MetaTensor* out,
                                 MetaTensor* softmax,
                                 MetaTensor* softmax_lse,
                                 MetaTensor* seed_offset) {
  const auto& qkvdims = qkv.dims();
  PADDLE_ENFORCE(qkvdims.size() == 4 || qkvdims.size() == 5,
                 phi::errors::InvalidArgument(
                     "qkv dims must be 4(unpadded) or 5(padded batch)"));
  // qkv [total_*,nheads/nheads_k+2,nheads_k,headdim]
  auto out_dims = DDim({qkvdims[0], (qkvdims[1] - 2) * qkvdims[2], qkvdims[3]});
  if (qkvdims.size() == 5) {
    // qkv [batchsize,seqlen,nheads/nheads_k+2,nheads_k,headdim]
    out_dims =
        DDim{qkvdims[0], qkvdims[1], (qkvdims[2] - 2) * qkvdims[3], qkvdims[4]};
  }
  out->set_dims(out_dims);
  out->set_dtype(qkv.dtype());
  out->set_layout(qkv.layout());
  softmax->set_dtype(qkv.dtype());
  softmax_lse->set_dtype(qkv.dtype());
  if (seed_offset) {
    seed_offset->set_dtype(phi::DataType::INT64);
  }
}

void ArangeTensorInferMeta(const MetaTensor& start,
                           const MetaTensor& end,
                           const MetaTensor& step,
                           MetaTensor* out) {
  PADDLE_ENFORCE_EQ(common::product(start.dims()),
                    1,
                    phi::errors::InvalidArgument(
                        "The numel of Input(start) should be 1, but got %d",
                        common::product(start.dims())));

  PADDLE_ENFORCE_EQ(common::product(end.dims()),
                    1,
                    phi::errors::InvalidArgument(
                        "The numel of Input(end) should be 1, but got %d",
                        common::product(end.dims())));

  PADDLE_ENFORCE_EQ(common::product(step.dims()),
                    1,
                    phi::errors::InvalidArgument(
                        "The numel of Input(step) should be 1, but got %d",
                        common::product(step.dims())));

  out->set_dims({-1});
  out->set_dtype(start.dtype());
}

void InstanceNormInferMeta(const MetaTensor& x,
                           const MetaTensor& scale,
                           const MetaTensor& bias,
                           float epsilon,
                           MetaTensor* y,
                           MetaTensor* saved_mean,
                           MetaTensor* saved_variance,
                           MetaConfig config) {
  PADDLE_ENFORCE_NE(y,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The y in InstanceNormInferMeta can't be nullptr."));
  const auto x_dims = x.dims();
  PADDLE_ENFORCE_NE(common::product(x_dims),
                    0,
                    phi::errors::PreconditionNotMet(
                        "The Input variable X has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function."));
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "smaller than or equal to 5, But received: the shape of input "
          "X = [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));
  auto N = x_dims[0];
  auto C = x_dims[1];
  auto NxC = N * C;
  if (scale) {
    auto scale_dim = scale.dims();
    PADDLE_ENFORCE_EQ(
        scale_dim.size(),
        1UL,
        phi::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale_dim,
            scale_dim.size()));
    bool check = !((!config.is_runtime) && (common::product(scale_dim) <= 0));
    if (check) {
      PADDLE_ENFORCE_EQ(scale_dim[0],
                        C,
                        phi::errors::InvalidArgument(
                            "ShapeError: the shape of scale must equal to [%d]"
                            "But received: the shape of scale is [%d]",
                            C,
                            scale_dim[0]));
    }
  }
  if (bias) {
    auto bias_dim = bias.dims();
    PADDLE_ENFORCE_EQ(
        bias_dim.size(),
        1UL,
        phi::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias_dim,
            bias_dim.size()));
    bool check = !((!config.is_runtime) && (common::product(bias_dim) <= 0));
    if (check) {
      PADDLE_ENFORCE_EQ(bias_dim[0],
                        C,
                        phi::errors::InvalidArgument(
                            "ShapeError: the shape of bias must equal to [%d]"
                            "But received: the shape of bias is [%d]",
                            C,
                            bias_dim[0]));
    }
  }
  y->set_dims(x_dims);
  y->share_lod(x);
  y->set_dtype(x.dtype());
  y->set_layout(x.layout());
  phi::DataType x_dtype = x.dtype();
  phi::DataType param_type =
      (x_dtype == phi::DataType::BFLOAT16 || x_dtype == phi::DataType::FLOAT16)
          ? phi::DataType::FLOAT32
          : x_dtype;
  if (saved_mean) {
    saved_mean->set_dims({NxC});
    saved_mean->set_dtype(param_type);
  }
  if (saved_variance) {
    saved_variance->set_dims({NxC});
    saved_variance->set_dtype(param_type);
  }
}

void GlobalGatherInferMeta(const MetaTensor& x,
                           const MetaTensor& local_count,
                           const MetaTensor& global_count,
                           int ring_id,
                           bool use_calc_stream,
                           MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      phi::errors::InvalidArgument(
          "The ring_id (%d) for global gather op must be non-negative.",
          ring_id));
  auto input_dims = x.dims();
  auto ndim_input = input_dims.size();
  // dim check
  PADDLE_ENFORCE_EQ(
      ndim_input,
      2,
      phi::errors::InvalidArgument("The input tensor's dimension must be 2. "
                                   "But received input's dimension = %d.",
                                   ndim_input));
  phi::DDim out_dims = common::make_ddim({-1, -1});
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void GlobalScatterInferMeta(const MetaTensor& x,
                            const MetaTensor& local_count,
                            const MetaTensor& global_count,
                            int ring_id,
                            bool use_calc_stream,
                            MetaTensor* out) {
  PADDLE_ENFORCE_GE(
      ring_id,
      0,
      phi::errors::InvalidArgument(
          "The ring_id (%d) for global scatter op must be non-negative.",
          ring_id));
  auto input_dims = x.dims();
  auto ndim_input = input_dims.size();
  // dim check
  PADDLE_ENFORCE_EQ(
      ndim_input,
      2,
      phi::errors::InvalidArgument("The input tensor's dimension must be 2. "
                                   "But received input's dimension = %d.",
                                   ndim_input));

  phi::DDim out_dims = common::make_ddim({-1, -1});
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void AddGroupNormSiluInferMeta(const MetaTensor& x,
                               const MetaTensor& residual,
                               const MetaTensor& scale,
                               const MetaTensor& bias,
                               float epsilon,
                               int groups,
                               const std::string& data_layout_str,
                               const std::string& activation,
                               MetaTensor* y,
                               MetaTensor* residual_out,
                               MetaTensor* mean,
                               MetaTensor* variance) {
  PADDLE_ENFORCE_NE(y,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The y in GroupNormInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(mean,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The mean in GroupNormInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(
      variance,
      nullptr,
      phi::errors::InvalidArgument(
          "The variance in GroupNormInferMeta can't be nullptr."));

  auto x_dim = x.dims();
  PADDLE_ENFORCE_GE(
      x_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input(X)'s dimension of Op(group_norm) must be "
          "greater than 1. But received: %u-D Tensor, which shape is [%s].",
          x_dim.size(),
          x_dim));

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  const int64_t channel_num =
      (data_layout == DataLayout::kNCHW ? x_dim[1] : x_dim[x_dim.size() - 1]);
  auto batch_size = x_dim[0];
  PADDLE_ENFORCE_LE(
      groups,
      channel_num,
      phi::errors::InvalidArgument(
          "The Attr(groups) of Op(group_norm) must be less than or "
          "equal to the number of channels. But received: groups "
          "is [%s], channels is [%s], the Attr(data_layout) "
          "is [%s]. The error may come from wrong data_layout setting.",
          groups,
          channel_num,
          data_layout_str));
  PADDLE_ENFORCE_GE(
      groups,
      1,
      phi::errors::InvalidArgument(
          "The Attr(groups) of Op(group_norm) must be "
          "greater than or equal to 1. But received: groups is [%s].",
          groups));
  PADDLE_ENFORCE_EQ(
      channel_num % groups,
      0,
      phi::errors::InvalidArgument(
          "Expected number of channels in input to be divisible by "
          "num_groups, but got input channel is %d and num_groups is %d",
          channel_num,
          groups));

  if (scale) {
    PADDLE_ENFORCE_EQ(
        scale.dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The Input(Scale) of Op(group_norm) should be 1-D Tensor. "
            "But received: %u-D Tensor, the shape of Input(Scale) is [%s].",
            scale.dims().size(),
            scale.dims()));
    PADDLE_ENFORCE_EQ(
        scale.dims()[0],
        channel_num,
        phi::errors::InvalidArgument(
            "The Input(Scale)'s first dimension size of Op(group_norm) must "
            "be equal to the number of channels. But received: the "
            "Input(Scale)'s first dimension size is [%s], the channels is "
            "[%s], the Attr(data_layout) is [%s]. The error may come "
            "from wrong data_layout setting.",
            scale.dims()[0],
            channel_num,
            data_layout_str));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(
        bias.dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The Input(Bias) of Op(group_norm) should be 1-D Tensor. "
            "But received: %u-D Tensor, the shape of Input(Bias) is [%s].",
            bias.dims().size(),
            bias.dims()));
    PADDLE_ENFORCE_EQ(
        bias.dims()[0],
        channel_num,
        phi::errors::InvalidArgument(
            "The Input(Bias)'s first dimension size of "
            "Op(group_norm) must be equal to the number of channels. "
            "But received: the Input(Bias)'s first dimension size is [%s], "
            "the channels is [%s], the Attr(data_layout) is [%s]. The "
            "error may come from wrong data_layout setting.",
            bias.dims()[0],
            channel_num,
            data_layout_str));
  }
  y->set_dims(x_dim);
  y->set_dtype(x.dtype());
  y->share_lod(x);

  phi::DataType x_dtype = x.dtype();
  phi::DataType param_type =
      (x_dtype == phi::DataType::BFLOAT16 || x_dtype == phi::DataType::FLOAT16)
          ? phi::DataType::FLOAT32
          : x_dtype;
  if (mean) {
    mean->set_dims({batch_size, groups});
    mean->set_dtype(param_type);
  }
  if (variance) {
    variance->set_dims({batch_size, groups});
    variance->set_dtype(param_type);
  }
  if (residual_out) {
    residual_out->set_dims(x_dim);
    residual_out->set_dtype(x.dtype());
    residual_out->share_lod(x);
  }
}

void GroupNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        float epsilon,
                        int groups,
                        const std::string& data_layout_str,
                        MetaTensor* y,
                        MetaTensor* mean,
                        MetaTensor* variance) {
  PADDLE_ENFORCE_NE(y,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The y in GroupNormInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(mean,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The mean in GroupNormInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(
      variance,
      nullptr,
      phi::errors::InvalidArgument(
          "The variance in GroupNormInferMeta can't be nullptr."));

  auto x_dim = x.dims();
  PADDLE_ENFORCE_GE(
      x_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input(X)'s dimension of Op(group_norm) must be "
          "greater than 1. But received: %u-D Tensor, which shape is [%s].",
          x_dim.size(),
          x_dim));

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  const int64_t channel_num =
      (data_layout == DataLayout::kNCHW ? x_dim[1] : x_dim[x_dim.size() - 1]);
  auto batch_size = x_dim[0];
  PADDLE_ENFORCE_LE(
      groups,
      channel_num,
      phi::errors::InvalidArgument(
          "The Attr(groups) of Op(group_norm) must be less than or "
          "equal to the number of channels. But received: groups "
          "is [%s], channels is [%s], the Attr(data_layout) "
          "is [%s]. The error may come from wrong data_layout setting.",
          groups,
          channel_num,
          data_layout_str));
  PADDLE_ENFORCE_GE(
      groups,
      1,
      phi::errors::InvalidArgument(
          "The Attr(groups) of Op(group_norm) must be "
          "greater than or equal to 1. But received: groups is [%s].",
          groups));
  PADDLE_ENFORCE_EQ(
      channel_num % groups,
      0,
      phi::errors::InvalidArgument(
          "Expected number of channels in input to be divisible by "
          "num_groups, but got input channel is %d and num_groups is %d",
          channel_num,
          groups));

  if (scale) {
    PADDLE_ENFORCE_EQ(
        scale.dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The Input(Scale) of Op(group_norm) should be 1-D Tensor. "
            "But received: %u-D Tensor, the shape of Input(Scale) is [%s].",
            scale.dims().size(),
            scale.dims()));
    PADDLE_ENFORCE_EQ(
        scale.dims()[0],
        channel_num,
        phi::errors::InvalidArgument(
            "The Input(Scale)'s first dimension size of Op(group_norm) must "
            "be equal to the number of channels. But received: the "
            "Input(Scale)'s first dimension size is [%s], the channels is "
            "[%s], the Attr(data_layout) is [%s]. The error may come "
            "from wrong data_layout setting.",
            scale.dims()[0],
            channel_num,
            data_layout_str));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(
        bias.dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The Input(Bias) of Op(group_norm) should be 1-D Tensor. "
            "But received: %u-D Tensor, the shape of Input(Bias) is [%s].",
            bias.dims().size(),
            bias.dims()));
    PADDLE_ENFORCE_EQ(
        bias.dims()[0],
        channel_num,
        phi::errors::InvalidArgument(
            "The Input(Bias)'s first dimension size of "
            "Op(group_norm) must be equal to the number of channels. "
            "But received: the Input(Bias)'s first dimension size is [%s], "
            "the channels is [%s], the Attr(data_layout) is [%s]. The "
            "error may come from wrong data_layout setting.",
            bias.dims()[0],
            channel_num,
            data_layout_str));
  }
  y->set_dims(x_dim);
  y->set_dtype(x.dtype());
  y->share_lod(x);

  phi::DataType x_dtype = x.dtype();
  phi::DataType param_type =
      (x_dtype == phi::DataType::BFLOAT16 || x_dtype == phi::DataType::FLOAT16)
          ? phi::DataType::FLOAT32
          : x_dtype;
  if (mean) {
    mean->set_dims({batch_size, groups});
    mean->set_dtype(param_type);
  }
  if (variance) {
    variance->set_dims({batch_size, groups});
    variance->set_dtype(param_type);
  }
}

void LayerNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        float epsilon,
                        int begin_norm_axis,
                        MetaTensor* out,
                        MetaTensor* mean,
                        MetaTensor* variance,
                        MetaConfig config) {
  auto x_dim = x.dims();
  PADDLE_ENFORCE_GT(begin_norm_axis,
                    0,
                    phi::errors::InvalidArgument(
                        "'begin_norm_axis' in Op(LayerNorm) should be"
                        "greater than zero. But received [%d].",
                        begin_norm_axis));
  PADDLE_ENFORCE_LT(
      begin_norm_axis,
      x_dim.size(),
      phi::errors::InvalidArgument(
          "'begin_norm_axis' must be less than the dimensions of X,"
          "But received 'begin_norm_axis' is [%d],"
          "received the dimensions of X is [%d].",
          begin_norm_axis,
          x_dim.size()));

  auto matrix_dim = common::flatten_to_2d(x_dim, begin_norm_axis);

  // keep the axis size before normalization for shape of variance and mean
  auto before_norm_dims = slice_ddim(x_dim, 0, begin_norm_axis);
  // int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  if (scale) {
    PADDLE_ENFORCE_EQ(scale.dims().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The dimensions of Input(Scale) must be 1, but "
                          "received dimensions of"
                          "Input(Scale) is [%d]",
                          scale.dims().size()));
  }

  if (config.is_runtime && scale) {
    PADDLE_ENFORCE_EQ(
        scale.dims()[0],
        right,
        phi::errors::InvalidArgument(
            "The first dimension value of Input(Scale) must equal to be the"
            "second dimension value of the flattened 2D matrix of Input(X),"
            "But received the first dimension value of Input(Scale) is"
            "[%d], the second dimension value of the flattened 2D matrix of"
            " Input(Scale) is [%d].",
            scale.dims()[0],
            right));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(bias.dims().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The dimensions of Input(Bias) must be 1, but "
                          "received dimensions of"
                          "Input(Bias) is [%d]",
                          bias.dims().size()));
  }
  if (config.is_runtime && bias) {
    PADDLE_ENFORCE_EQ(
        bias.dims()[0],
        right,
        phi::errors::InvalidArgument(
            "The first dimension value of Input(Bias) must equal to be the"
            "second dimension value of the flattened 2D matrix of Input(X),"
            "But received the first dimension value of Input(Bias) is"
            "[%d], the second dimension value of the flattened 2D matrix of"
            " Input(Bias) is [%d].",
            bias.dims()[0],
            right));
  }

  PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                    true,
                    phi::errors::InvalidArgument(
                        "'epsilon' in Op(LayerNorm) should be between"
                        "0.0 and 0.001, But received [%s].",
                        epsilon));

  phi::DataType x_dtype = x.dtype();
  out->set_dims(x_dim);
  out->set_dtype(x_dtype);
  out->share_lod(x);

  phi::DataType param_type =
      (x_dtype == phi::DataType::BFLOAT16 || x_dtype == phi::DataType::FLOAT16)
          ? phi::DataType::FLOAT32
          : x_dtype;
  if (mean) {
    mean->set_dims({before_norm_dims});
    mean->set_dtype(param_type);
  }
  if (variance) {
    variance->set_dims({before_norm_dims});
    variance->set_dtype(param_type);
  }
}

void LayerNormGradInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            const MetaTensor& z,
                            MetaTensor* dx,
                            MetaTensor* dy,
                            MetaTensor* dz) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy && y) {
    dy->share_meta(y);
  }
  if (dz && z) {
    dz->share_meta(z);
  }
}

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto w_dims = weight.dims();
  DDim out_dims;
  out_dims = funcs::GetOutputDims(x_dims, y_dims);
  out_dims = funcs::GetOutputDims(out_dims, w_dims);
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void LinspaceRawInferMeta(const MetaTensor& start,
                          const MetaTensor& stop,
                          const MetaTensor& number,
                          MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      common::product(start.dims()),
      1,
      phi::errors::InvalidArgument("The size of Input(start) should be 1,"
                                   "but got %d.",
                                   common::product(start.dims())));

  PADDLE_ENFORCE_EQ(
      common::product(stop.dims()),
      1,
      phi::errors::InvalidArgument("The size of Input(stop) should be 1,"
                                   "but got %d.",
                                   common::product(stop.dims())));

  PADDLE_ENFORCE_EQ(
      common::product(number.dims()),
      1,
      phi::errors::InvalidArgument("The size of Input(number) should be 1,"
                                   "but got %d.",
                                   common::product(number.dims())));

  out->set_dims(common::make_ddim({-1}));
  out->set_dtype(start.dtype());
}

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       DataType dtype,
                       MetaTensor* out) {
  LinspaceRawInferMeta(start, stop, number, out);
}

void MatchMatrixTensorInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                const MetaTensor& w,
                                int dim_t,
                                MetaTensor* out,
                                MetaTensor* tmp,
                                MetaConfig config) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(X) should be equal to 2, "
                        "but received %d.",
                        x_dims.size()));

  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(y_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(Y) should be equal to 2, "
                        "but received %d.",
                        y_dims.size()));

  auto w_dims = w.dims();
  PADDLE_ENFORCE_EQ(w_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(W) should be equal to 3, "
                        "but received %d.",
                        w_dims.size()));

  PADDLE_ENFORCE_EQ(
      w_dims[0],
      x_dims[1],
      phi::errors::InvalidArgument(
          "The first dimension of Input(W) should be equal to the second "
          "dimension of Input(X). But received the first dimension of Input(W) "
          "is %d, the second dimension of Input(X) is %d.",
          w_dims[0],
          x_dims[1]));
  PADDLE_ENFORCE_EQ(
      w_dims[1],
      dim_t,
      phi::errors::InvalidArgument(
          "The second dimension of Input(W) should be equal to 'dim_t', but "
          "received the second dimension of Input(W) is %d, 'dim_t' is %d.",
          w_dims[1],
          dim_t));
  PADDLE_ENFORCE_EQ(
      w_dims[2],
      y_dims[1],
      phi::errors::InvalidArgument(
          "The last dimension of Input(W) should be equal to "
          "the second dimension of Input(Y). But received the last dimension "
          "of Input(W) is %d, the second dimension of Input(Y) is %d.",
          w_dims[2],
          y_dims[1]));

  int64_t out_dim_0 = -1;
  int64_t tmp_dim_0 = -1;
  if (!config.is_runtime) {
    out->share_lod(x);
    std::vector<int64_t> out_dims_vec{out_dim_0};
    out_dims_vec.push_back(1);
    std::vector<int64_t> tmp_dims_vec{tmp_dim_0};
    tmp_dims_vec.push_back(1);
    out->set_dims(common::make_ddim(out_dims_vec));
    out->set_dtype(x.dtype());
    tmp->set_dims(common::make_ddim(tmp_dims_vec));
    tmp->set_dtype(x.dtype());
  }
}

void MultiClassNMSInferMeta(const MetaTensor& bboxes,
                            const MetaTensor& scores,
                            const MetaTensor& rois_num,
                            float score_threshold,
                            int nms_top_k,
                            int keep_top_k,
                            float nms_threshold,
                            bool normalized,
                            float nms_eta,
                            int background_label,
                            MetaTensor* out,
                            MetaTensor* index,
                            MetaTensor* nms_rois_num,
                            MetaConfig config) {
  auto box_dims = bboxes.dims();
  auto score_dims = scores.dims();
  auto score_size = score_dims.size();

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        score_size == 2 || score_size == 3,
        true,
        errors::InvalidArgument("The rank of Input(Scores) must be 2 or 3"
                                ". But received rank = %d",
                                score_size));
    PADDLE_ENFORCE_EQ(
        box_dims.size(),
        3,
        errors::InvalidArgument("The rank of Input(BBoxes) must be 3"
                                ". But received rank = %d",
                                box_dims.size()));
    if (score_size == 3) {
      PADDLE_ENFORCE_EQ(box_dims[2] == 4 || box_dims[2] == 8 ||
                            box_dims[2] == 16 || box_dims[2] == 24 ||
                            box_dims[2] == 32,
                        true,
                        errors::InvalidArgument(
                            "The last dimension of Input"
                            "(BBoxes) must be 4 or 8, "
                            "represents the layout of coordinate "
                            "[xmin, ymin, xmax, ymax] or "
                            "4 points: [x1, y1, x2, y2, x3, y3, x4, y4] or "
                            "8 points: [xi, yi] i= 1,2,...,8 or "
                            "12 points: [xi, yi] i= 1,2,...,12 or "
                            "16 points: [xi, yi] i= 1,2,...,16"));
      PADDLE_ENFORCE_EQ(
          box_dims[1],
          score_dims[2],
          errors::InvalidArgument(
              "The 2nd dimension of Input(BBoxes) must be equal to "
              "last dimension of Input(Scores), which represents the "
              "predicted bboxes."
              "But received box_dims[1](%s) != socre_dims[2](%s)",
              box_dims[1],
              score_dims[2]));
    } else {
      PADDLE_ENFORCE_EQ(box_dims[2],
                        4,
                        errors::InvalidArgument(
                            "The last dimension of Input"
                            "(BBoxes) must be 4. But received dimension = %d",
                            box_dims[2]));
      PADDLE_ENFORCE_EQ(
          box_dims[1],
          score_dims[1],
          errors::InvalidArgument(
              "The 2nd dimension of Input"
              "(BBoxes) must be equal to the 2nd dimension of Input(Scores). "
              "But received box dimension = %d, score dimension = %d",
              box_dims[1],
              score_dims[1]));
    }
  }
  PADDLE_ENFORCE_NE(out,
                    nullptr,
                    errors::InvalidArgument(
                        "The out in MultiClassNMSInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(
      index,
      nullptr,
      errors::InvalidArgument(
          "The index in MultiClassNMSInferMeta can't be nullptr."));
  // Here the box_dims[0] is not the real dimension of output.
  // It will be rewritten in the computing kernel.

  out->set_dims(common::make_ddim({-1, box_dims[2] + 2}));
  out->set_dtype(bboxes.dtype());
  index->set_dims(common::make_ddim({-1, 1}));
  index->set_dtype(DataType::INT32);
  nms_rois_num->set_dims(common::make_ddim({-1}));
  nms_rois_num->set_dtype(DataType::INT32);
}

void MovingAverageAbsMaxScaleInferMeta(const MetaTensor& x,
                                       const MetaTensor& in_accum,
                                       const MetaTensor& in_state,
                                       MetaTensor* out,
                                       MetaTensor* out_scale,
                                       MetaTensor* out_state,
                                       MetaTensor* out_accum) {
  if (out) {
    out->set_dims(x.dims());
    out->share_lod(x);
    out_scale->set_dims({1});
  }
  if (out_state) {
    out_state->set_dims(in_state.dims());
  }
  if (out_accum) {
    out_accum->set_dims(in_accum.dims());
  }
}

void NllLossRawInferMeta(const MetaTensor& input,
                         const MetaTensor& label,
                         const MetaTensor& weight,
                         int64_t ignore_index,
                         const std::string& reduction,
                         MetaTensor* out,
                         MetaTensor* total_weight,
                         MetaConfig config) {
  auto x_dims = input.dims();
  auto label_dims = label.dims();
  PADDLE_ENFORCE_EQ(x_dims.size() == 2 || x_dims.size() == 4,
                    true,
                    phi::errors::InvalidArgument(
                        "The tensor rank of Input(X) must be 2 or 4."));
  bool contain_unknown_dim = common::contain_unknown_dim(x_dims) ||
                             common::contain_unknown_dim(label_dims);
  bool check = config.is_runtime || !contain_unknown_dim;
  if (check) {
    PADDLE_ENFORCE_EQ(
        x_dims[0],
        label_dims[0],
        phi::errors::InvalidArgument(
            "ShapeError: Expected input batch_size to match label batch_size,"
            "But received: the Input(x) batch_size is [%s], the Input(label) "
            " batch_size is [%s].",
            x_dims[0],
            label_dims[0]));
    if (weight) {
      auto w_dims = weight.dims();
      PADDLE_ENFORCE_EQ(
          w_dims.size(),
          1,
          phi::errors::InvalidArgument("Input(Weight) should be a 1D tensor."));
      PADDLE_ENFORCE_EQ(
          x_dims[1],
          w_dims[0],
          phi::errors::InvalidArgument(
              "Expected input tensor Weight's size should equal "
              "to the first dimension of the input tensor X. But received "
              "Weight's "
              "size is %d, the first dimension of input X is %d",
              w_dims[0],
              x_dims[1]));
    }
  }
  if (x_dims.size() == 2) {
    if (reduction == "none") {
      out->set_dims({x_dims[0]});
    } else {
      out->set_dims(common::make_ddim({}));
    }
  } else if (x_dims.size() == 4) {
    PADDLE_ENFORCE_EQ(label_dims.size(),
                      3,
                      phi::errors::InvalidArgument(
                          "Expected Input(Label) dimensions=3, received %d.",
                          label_dims.size()));
    auto input0 = x_dims[0];
    auto input2 = x_dims[2];
    auto input3 = x_dims[3];
    auto label0 = label_dims[0];
    auto label1 = label_dims[1];
    auto label2 = label_dims[2];
    PADDLE_ENFORCE_EQ(
        input0 == label0 && input2 == label1 && input3 == label2,
        true,
        phi::errors::InvalidArgument("Input(X) tensor shape should "
                                     "match to Input(Label) tensor "
                                     "shape."));
    if (reduction == "none") {
      out->set_dims({x_dims[0], x_dims[2], x_dims[3]});
    } else {
      out->set_dims(common::make_ddim({}));
    }
  }
  total_weight->set_dims(common::make_ddim({}));
  out->set_dtype(input.dtype());
  total_weight->set_dtype(input.dtype());
}

void PutAlongAxisInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& value,
                           int axis,
                           const std::string& reduce,
                           MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void PushGpupsSparseInferMeta(const std::vector<const MetaTensor*>& ids,
                              const std::vector<const MetaTensor*>& out,
                              const std::vector<int>& size,
                              bool is_sparse,
                              bool is_distributed,
                              std::vector<MetaTensor*> out_grad) {}

void RandomRoutingInferMeta(const MetaTensor& prob,
                            const MetaTensor& topk_value,
                            const MetaTensor& topk_idx,
                            MetaTensor* out) {
  // check dims
  auto topk_val_dims = topk_value.dims();
  auto prob_dims = prob.dims();
  auto topk_idx_dims = topk_idx.dims();

  PADDLE_ENFORCE_EQ(prob_dims[0],
                    topk_val_dims[0],
                    phi::errors::InvalidArgument(
                        "Output(Out) of ScatterNdAddOp should not be null."));

  PADDLE_ENFORCE_EQ(topk_idx_dims[1],
                    topk_val_dims[1],
                    phi::errors::InvalidArgument(
                        "Output(Out) of ScatterNdAddOp should not be null."));

  PADDLE_ENFORCE_EQ(topk_idx_dims[0],
                    topk_val_dims[0],
                    phi::errors::InvalidArgument(
                        "Output(Out) of ScatterNdAddOp should not be null."));

  out->set_dims(topk_idx_dims);
  out->set_dtype(topk_idx.dtype());
  out->share_lod(topk_idx);
}

void RankAttentionInferMeta(const MetaTensor& x,
                            const MetaTensor& rank_offset,
                            const MetaTensor& rank_param,
                            int max_rank,
                            int max_size,
                            MetaTensor* input_help,
                            MetaTensor* out,
                            MetaTensor* ins_rank) {
  auto x_dims = x.dims();
  auto ins_num = x_dims[0];
  auto param_dims = rank_param.dims();
  auto para_col = param_dims[1];
  auto rank_offset_dims = rank_offset.dims();
  auto x_fea_dim = x_dims[1];
  auto block_matrix_row = max_rank * x_fea_dim;

  PADDLE_ENFORCE_EQ(
      (rank_offset_dims[1] - 1) / 2,
      max_rank,
      phi::errors::InvalidArgument("Input(RankOffset) has wrong columns, "
                                   "except columns to be %d, but got %d",
                                   max_rank,
                                   (rank_offset_dims[1] - 1) / 2));

  std::vector<int64_t> out_dims({ins_num, para_col});
  out->set_dims(common::make_ddim(out_dims));
  out->set_dtype(x.dtype());

  std::vector<int64_t> input_help_dims({ins_num, block_matrix_row});
  input_help->set_dims(common::make_ddim(input_help_dims));
  input_help->set_dtype(x.dtype());

  std::vector<int64_t> ins_rank_dims({ins_num, 1});
  ins_rank->set_dims(common::make_ddim(ins_rank_dims));
  ins_rank->set_dtype(x.dtype());

  out->share_lod(x);
}

void RoiAlignInferMeta(const MetaTensor& x,
                       const MetaTensor& boxes,
                       const MetaTensor& boxes_num,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int sampling_ratio,
                       bool aligned,
                       MetaTensor* out,
                       MetaConfig config) {
  auto input_dims = x.dims();
  auto boxes_dims = boxes.dims();

  if (boxes_num) {
    auto boxes_num_dims = boxes_num.dims();
    PADDLE_ENFORCE_EQ(
        boxes_num_dims.size(),
        1,
        phi::errors::InvalidArgument("The size of boxes_num should be 1"
                                     ", but received size = %d",
                                     boxes_num_dims.size()));
  }
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The format of Input(x) in"
                        "RoiAlignOp is NCHW. And the rank of input must be 4. "
                        "But received rank = %d",
                        input_dims.size()));
  PADDLE_ENFORCE_EQ(boxes_dims.size(),
                    2,
                    phi::errors::InvalidArgument("The rank of Input(boxes) "
                                                 "in RoiAlignOp should be 2. "
                                                 "But the rank of boxes is %d",
                                                 boxes_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(boxes_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension "
                          "of Input(boxes) should be 4. But received the "
                          "dimension = %d",
                          boxes_dims[1]));
  }

  PADDLE_ENFORCE_GT(pooled_height,
                    0,
                    phi::errors::InvalidArgument(
                        "The 'pooled_height' attribute in RoiAlignOp is "
                        "invalid. The height must be greater than 0. But "
                        "received 'pooled_height' = %d",
                        pooled_height));
  PADDLE_ENFORCE_GT(pooled_width,
                    0,
                    phi::errors::InvalidArgument(
                        "The 'pooled_width' attribute in RoiAlignOp is "
                        "invalid. The width must be greater than 0. But "
                        "received 'pooled_width' = %d",
                        pooled_width));
  PADDLE_ENFORCE_GT(spatial_scale,
                    0.0f,
                    phi::errors::InvalidArgument(
                        "The 'spatial_scale' attribute in RoiAlignOp is "
                        "invalid. The scale must be greater than 0. But "
                        "received 'spatial_scale' = %f",
                        spatial_scale));

  auto out_dims = input_dims;
  out_dims[0] = boxes_dims[0];
  out_dims[1] = input_dims[1];
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void RoiPoolInferMeta(const MetaTensor& x,
                      const MetaTensor& boxes,
                      const MetaTensor& boxes_num,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      MetaTensor* out,
                      MetaTensor* arg_max) {
  auto input_dims = x.dims();
  auto boxes_dims = boxes.dims();

  if (boxes_num) {
    auto boxes_num_dims = boxes_num.dims();
    PADDLE_ENFORCE_EQ(
        boxes_num_dims.size(),
        1,
        phi::errors::InvalidArgument("The second dimension of boxes_num should "
                                     "be 1, but received dimension is %d",
                                     boxes_num_dims.size()));
  }
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The input data should be a four-dimensional "
                        "tensor with [N,C,H,W], but received input data with "
                        " %d dimension",
                        input_dims.size()));
  PADDLE_ENFORCE_EQ(
      boxes_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "boxes should be a 2-D LoDTensor with shape (num_boxes, 4)"
          "given as [[x1, y1, x2, y2], ...], but received boxes is "
          "%d-dimensional LoDTensor",
          boxes_dims.size()));
  PADDLE_ENFORCE_EQ(
      boxes_dims[1],
      4,
      phi::errors::InvalidArgument(
          "boxes should be a 2-D LoDTensor with shape (num_boxes, 4)"
          "given as [[x1, y1, x2, y2], ...]. But the second dimension of  "
          "the received data is %d",
          boxes_dims[1]));

  PADDLE_ENFORCE_GT(
      pooled_height,
      0,
      phi::errors::OutOfRange("The pooled output height must be greater than 0"
                              "but received height is %d",
                              pooled_height));
  PADDLE_ENFORCE_GT(
      pooled_width,
      0,
      phi::errors::OutOfRange("The pooled output width must be greater than 0"
                              "but received width is %d",
                              pooled_width));
  PADDLE_ENFORCE_GT(
      spatial_scale,
      0.0f,
      phi::errors::OutOfRange("The spatial scale must be greater than 0, "
                              "but received spatial scale is %f",
                              spatial_scale));

  auto out_dims = input_dims;
  out_dims[0] = boxes_dims[0];
  out_dims[1] = input_dims[1];
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  arg_max->set_dims(out_dims);
  arg_max->set_dtype(DataType::INT64);
}

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out) {
  const auto& updates_dims = updates.dims();
  const auto& ref_dims = x.dims();
  const auto& index_dims = index.dims();

  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of the index should be 1 when the "
                          "index is a 2D tensor, but we get %d.",
                          index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(index_dims.size() == 1 || index_dims.size() == 0,
                      true,
                      phi::errors::InvalidArgument(
                          "The index should be a 0D or 1D tensor when the "
                          "index is not a 2D tensor, but we get %d.",
                          index_dims.size()));
  }
  if (index_dims.size() != 0) {
    PADDLE_ENFORCE_EQ(
        (ref_dims.size() == updates_dims.size()),
        true,
        phi::errors::InvalidArgument(
            "When the Input(Index) is not a 0D tensor, the "
            "Input(X) and Input(Updates) should have the same shape size, "
            "but received the size of Input(x)'s shape is %d, the size of "
            "Input(Updates)'s shape is %d.",
            ref_dims.size(),
            updates_dims.size()));
    PADDLE_ENFORCE_EQ(
        updates_dims[0],
        index_dims[0],
        phi::errors::InvalidArgument(
            "Input(Updates) and Input(Ids) should have same batch-size, but"
            " received Input(Updates)'s batch-size is %d, Input(Ids)'s "
            "batch-size is %d.",
            updates_dims[0],
            index_dims[0]));
  } else {
    PADDLE_ENFORCE_EQ(
        (ref_dims.size() - 1 == updates_dims.size()),
        true,
        phi::errors::InvalidArgument(
            "When the Input(Index) is a 0D tensor, the "
            "Input(Updates) should have the shape size as Input(X)'s "
            "shape size - 1. But received the size of Input(x)'s shape is %d, "
            " the size of Input(Updates)'s shape is %d.",
            ref_dims.size(),
            updates_dims.size()));
  }
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out) {
  const auto& ref_dims = x.dims();
  auto ref_dims_size = ref_dims.size();
  const auto& index_dims = index.dims();
  int index_dims_size = static_cast<int>(index_dims.size());
  const auto& updates_dims = updates.dims();
  auto updates_dims_size = updates_dims.size();

  if (updates_dims_size == 0) {
    // check for 0d updates
    PADDLE_ENFORCE_EQ(
        index_dims_size,
        1,
        phi::errors::InvalidArgument("When the updates is a 0d tensor, the "
                                     "index should be a 1d tensor."));
    PADDLE_ENFORCE_EQ(
        index_dims[index_dims_size - 1],
        ref_dims_size,
        phi::errors::InvalidArgument(
            "When the update is a 0d tensor, The last dimension of "
            "Input(Index)'s shape should be equal with the rank of Input(X)."));
  } else {
    PADDLE_ENFORCE_LE(
        index_dims[index_dims_size - 1],
        ref_dims_size,
        phi::errors::InvalidArgument(
            "The last dimension of Input(Index)'s shape should be no greater "
            "than the rank of Input(X), but received the last dimension of "
            "Input(Index)'s shape is %d, the rank of Input(X) is %d.",
            index_dims[index_dims_size - 1],
            ref_dims_size));
    PADDLE_ENFORCE_GE(index_dims_size,
                      1UL,
                      phi::errors::InvalidArgument(
                          "The rank of Input(Index) should be greater than 1, "
                          "but received the rank of Input(Index) is %d.",
                          index_dims_size));

    // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
    std::vector<int64_t> r_updates_dims;
    bool without_dynamic_shape = true;
    for (int i = 0; i < index_dims_size - 1; ++i) {
      if (index_dims[i] == -1) {
        without_dynamic_shape = false;
      }
      r_updates_dims.emplace_back(index_dims[i]);
    }
    for (int i = static_cast<int>(index_dims[index_dims_size - 1]);
         i < ref_dims_size;
         ++i) {
      if (ref_dims[i] == -1) {
        without_dynamic_shape = false;
      }
      r_updates_dims.emplace_back(ref_dims[i]);
    }
    // check for non-0d updates
    PADDLE_ENFORCE_EQ(
        r_updates_dims.size(),
        updates_dims_size,
        phi::errors::InvalidArgument(
            "Updates has wrong shape. The shape of Updates and "
            "Input(Updates) "
            "should be same, but received the shape of Updates is %d, "
            "the shape of Input(Updates) is %d.",
            r_updates_dims.size(),
            updates_dims_size));
    if (without_dynamic_shape) {
      for (int64_t i = 0; i < updates_dims_size; ++i) {
        PADDLE_ENFORCE_EQ(
            r_updates_dims[i],
            updates_dims[i],
            phi::errors::InvalidArgument(
                "Updates has wrong shape. The dimensions of Updates and "
                "Input(Updates) should match, but received Updates's"
                "%d-th dimension is %d, Input(Updates)'s %d-th "
                "dimension is %d.",
                i,
                r_updates_dims[i],
                i,
                updates_dims[i]));
      }
    }
  }
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void SendURecvInferMeta(const MetaTensor& x,
                        const MetaTensor& src_index,
                        const MetaTensor& dst_index,
                        const std::string& reduce_op,
                        const IntArray& out_size,
                        MetaTensor* out,
                        MetaTensor* dst_count) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        phi::errors::InvalidArgument("The Dst_index should be 1D, "
                                     "when it is not 2D, but we get %d",
                                     dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    phi::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  auto dims = x.dims();
  std::vector<int64_t> dims_ = common::vectorize(dims);
  dims_[0] = -1;
  out->set_dims(common::make_ddim(dims_));
  out->set_dtype(x.dtype());

  if (reduce_op == "MEAN") {
    dst_count->set_dims({-1});
    dst_count->set_dtype(DataType::INT32);
  }
}

void SequenceConvInferMeta(const MetaTensor& x,
                           const MetaTensor& padding_data,
                           const MetaTensor& filter,
                           int context_length,
                           bool padding_trainable,
                           int context_start,
                           int context_stride,
                           MetaTensor* out) {
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  PADDLE_ENFORCE_EQ(
      context_stride,
      1,
      phi::errors::InvalidArgument(
          "Currently, SequenceConvOp only supports contextStride=1. But "
          "received contextStride = %u.",
          context_stride));
  PADDLE_ENFORCE_EQ(
      in_dims.size() == 2 && filter_dims.size() == 2,
      true,
      phi::errors::InvalidArgument(
          "Input(X, Filter) should be 2-D tensor. But received Input(X): "
          "input rank %u, input shape [%s]; received Input(Filter): "
          "input rank %u, input shape [%s].",
          in_dims.size(),
          in_dims,
          filter_dims.size(),
          filter_dims));
  PADDLE_ENFORCE_EQ(
      filter_dims[0],
      context_length * in_dims[1],
      phi::errors::InvalidArgument(
          "Filter's height should be context_length * "
          "input_hidden_size. But received: filter's height = %d, "
          "context_length * input_hidden_size = %d.",
          filter_dims[0],
          context_length * in_dims[1]));

  if (padding_trainable) {
    const phi::DDim& padding_dim = padding_data.dims();
    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    int total_pad = up_pad + down_pad;
    int input_width = static_cast<int>(in_dims[1]);
    bool start_equals_zero = context_start == 0;
    bool length_equals_one = context_length == 1;
    bool start_length = start_equals_zero && length_equals_one;

    PADDLE_ENFORCE_EQ(
        start_length,
        false,
        phi::errors::InvalidArgument(
            "If context_start is 0 and context_length is 1, paddingTrainable "
            "should be false."));
    PADDLE_ENFORCE_EQ(
        padding_dim.size(),
        2,
        phi::errors::InvalidArgument(
            "Input(PaddingData) should be 2-D tensor. But received: "
            "input rank %u, input shape [%s].",
            padding_dim.size(),
            padding_dim));
    PADDLE_ENFORCE_EQ(
        padding_dim[0] == total_pad && padding_dim[1] == input_width,
        true,
        phi::errors::InvalidArgument("Input(PaddingData)'s shape is not "
                                     "consistent with 'context_start' "
                                     "and 'context_length'. Received "
                                     "Input(PaddingData): input rank "
                                     "%u, "
                                     "input shape [%s].",
                                     padding_dim.size(),
                                     padding_dim));
  }

  in_dims[1] = filter_dims[1];
  out->set_dims(in_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void SparseMomentumInferMeta(const MetaTensor& param,
                             const MetaTensor& learning_rate,
                             const MetaTensor& velocity,
                             MetaTensor* param_out,
                             MetaTensor* velocity_out,
                             MetaTensor* master_param_out) {
  auto lr_dims = common::product(learning_rate.dims());
  PADDLE_ENFORCE_EQ(lr_dims == 1,
                    true,
                    phi::errors::InvalidArgument(
                        "Learning_rate should be a scalar. But Received "
                        "LearningRate's dim [%s]",
                        lr_dims));
  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dim,
      velocity.dims(),
      phi::errors::InvalidArgument(
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

void SpectralNormInferMeta(const MetaTensor& weight,
                           const MetaTensor& u,
                           const MetaTensor& v,
                           int dim,
                           int power_iters,
                           float eps,
                           MetaTensor* out,
                           MetaConfig config) {
  auto dim_weight = weight.dims();
  auto rank_weight = dim_weight.size();
  PADDLE_ENFORCE_GE(rank_weight,
                    2,
                    errors::InvalidArgument(
                        "The rank of Input(Weights) should be greater equal "
                        "than 2, but received Weight rank(%d)",
                        rank_weight));
  PADDLE_ENFORCE_LE(
      rank_weight,
      5,
      errors::InvalidArgument("The rank of Input(Weights) should be less equal "
                              "than 5, but received Weight rank(%d)",
                              rank_weight));

  auto dim_valid = dim == 0 || dim == 1;
  PADDLE_ENFORCE_EQ(dim_valid,
                    true,
                    errors::InvalidArgument(
                        "Attr(dim) can only be 0 or 1, but received %d", dim));
  PADDLE_ENFORCE_GE(
      power_iters,
      0,
      errors::InvalidArgument(
          "Attr(power_iters) should be greater equal then 0, but received %d",
          power_iters));

  int h = static_cast<int>(dim_weight[dim]);
  int w = 1;
  for (int i = 0; i < rank_weight; i++) {
    if (i != dim) {
      w *= static_cast<int>(dim_weight[i]);
    }
  }
  auto dim_u = u.dims();
  auto dim_v = v.dims();

  if (config.is_runtime || (dim_u[0] > 0 && h > 0)) {
    PADDLE_ENFORCE_EQ(dim_u[0],
                      h,
                      errors::InvalidArgument(
                          "Input(U) dimension[0] should be equal to "
                          "Input(Weight) dimension[Attr(dim)], but received "
                          "U dimension[0](%d) != Weight dimension[%d](%d)",
                          dim_u[0],
                          dim,
                          h));
  }

  if (config.is_runtime || (dim_v[0] > 0 && w > 0)) {
    PADDLE_ENFORCE_EQ(
        dim_v[0],
        w,
        errors::InvalidArgument(
            "Input(V) dimension[0] should be equal to the product of "
            "Input(Weight) dimension except dimension[Attr(dim)], but "
            "received V dimension[0](%d) != product of Input(Weight) "
            "dimension(%d)",
            dim_v[0],
            w));
  }

  if (out) {
    out->set_dims(dim_weight);
    out->set_dtype(weight.dtype());
    out->share_lod(weight);
  }
}

void ViterbiDecodeInferMeta(const MetaTensor& input,
                            const MetaTensor& transition,
                            const MetaTensor& length,
                            bool include_bos_eos_tag,
                            MetaTensor* scores,
                            MetaTensor* path,
                            MetaConfig config) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The rank of Input in ViterbiDecode  must be 3. But "
                        "received Input's rank is %d.",
                        in_dims.size()));
  auto length_dims = length.dims();
  PADDLE_ENFORCE_EQ(length_dims.size(),
                    1,
                    phi::errors::InvalidArgument(
                        "The rank of Length in ViterbiDecode must be 1. But "
                        "received Length's rank is %d.",
                        length_dims.size()));
  auto transition_dims = transition.dims();
  PADDLE_ENFORCE_EQ(
      transition_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The rank of Transition in ViterbiDecode must be 2. But "
          "received Transition's rank is %d.",
          transition_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        in_dims[0],
        length_dims[0],
        phi::errors::InvalidArgument(
            "The batch size of Input and Length should be equal."));
    PADDLE_ENFORCE_EQ(in_dims[2],
                      transition_dims[0],
                      phi::errors::InvalidArgument(
                          "The number of tags of Input (%d) and Transition "
                          "(%d) should be equal.",
                          transition_dims[0],
                          in_dims[2]));
  }
  scores->set_dims(length_dims);
  scores->set_dtype(length.dtype());
}

void QuantLinearInferMeta(const MetaTensor& x,
                          const MetaTensor& w,
                          const MetaTensor& bias,
                          int in_num_col_dims,
                          const std::string& activation_type,
                          bool padding_weights,
                          float scale_in,
                          const std::vector<float>& scale_weights,
                          int quant_round_type,
                          float quant_max_bound,
                          float quant_min_bound,
                          MetaTensor* y) {
  auto w_dims = w.dims();
  PADDLE_ENFORCE_EQ(
      w_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Weight of quant_linear is expected to be a 2-D tensor. "
          "But received the number of Weight's dimensions is %d, "
          "Weight's shape is %s.",
          w_dims.size(),
          w_dims));
  if (bias) {
    auto bias_dims = bias.dims();
    auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];

    PADDLE_ENFORCE_LE(bias_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The input Bias of quant_linear is expected to be a "
                          "1-D or 2-D tensor. But "
                          "received the number of Bias's dimensions is %d, "
                          "Bias's shape is %s.",
                          bias_dims.size(),
                          bias_dims));

    PADDLE_ENFORCE_EQ(
        bias_dims[bias_dims.size() - 1],
        w_dims1,
        phi::errors::InvalidArgument(
            "The last dimension of input Bias is expected be equal "
            "to the actual width of input Weight. But received the last "
            "dimension of Bias is %d, Bias's shape is %s; "
            "the actual width of Weight is %d, Weight's shape is %s.",
            bias_dims[bias_dims.size() - 1],
            bias_dims,
            w_dims1,
            w_dims));

    if (bias_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          bias_dims[0],
          1,
          phi::errors::InvalidArgument(
              "The first dimension of input Bias is expected to be 1, "
              "but received %d, Bias's shape is %s.",
              bias_dims[0],
              bias_dims));
    }
  }

  auto in_dims = x.dims();
  PADDLE_ENFORCE_LT(
      in_num_col_dims,
      in_dims.size(),
      phi::errors::InvalidArgument(
          "The attribute in_num_col_dims used to flatten Input to "
          "a 2-D tensor, is expected to be less than the number of "
          "Input's dimensions. But received in_num_col_dims is %d, "
          "the number of Input's dimensions is %d, Input's shape is %s.",
          in_num_col_dims,
          in_dims.size(),
          in_dims));

  if (!activation_type.empty()) {
    PADDLE_ENFORCE_EQ(
        activation_type,
        "relu",
        phi::errors::InvalidArgument(
            "The attribute activation_type of quant_linear is expected "
            "to be \"relu\", but received %s.",
            activation_type.c_str()));
  }

  std::vector<int64_t> output_dims;

  auto in_mat_dims = common::flatten_to_2d(in_dims, in_num_col_dims);
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
  PADDLE_ENFORCE_EQ(
      in_mat_dims[1],
      w_dims0,
      phi::errors::InvalidArgument(
          "The input's second dimension and weight's first dimension is "
          "expected to be the same. But received input's second dimension is "
          "%d, input's shape is %s; weight's first dimension is %d, weight's "
          "shape is %s.",
          in_mat_dims[1],
          in_mat_dims,
          w_dims0,
          common::make_ddim({w_dims0, w_dims1})));
  output_dims.reserve(static_cast<size_t>(in_num_col_dims) +
                      static_cast<size_t>(1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    output_dims.push_back(in_dims[i]);
  }
  output_dims.push_back(w_dims1);

  y->set_dims(common::make_ddim(output_dims));
  y->share_lod(x);
  y->set_dtype(x.dtype());
}
void TdmSamplerInferMeta(const MetaTensor& x,
                         const MetaTensor& travel,
                         const MetaTensor& layer,
                         bool output_positive,
                         const std::vector<int>& neg_samples_num_list,
                         const std::vector<int>& layer_offset_lod,
                         int seed,
                         int dtype,
                         MetaTensor* out,
                         MetaTensor* labels,
                         MetaTensor* mask,
                         MetaConfig config) {
  auto neg_samples_num_vec = neg_samples_num_list;
  auto output_positive_flag = output_positive;

  int64_t sample_res_length = 0;
  for (auto sample_nums : neg_samples_num_vec) {
    sample_res_length += sample_nums + (int64_t)output_positive_flag;
  }
  auto ddim = phi::make_ddim({-1, sample_res_length});
  auto input_dims = x.dims();
  if (config.is_runtime) {
    auto output_dims = phi::vectorize(input_dims);
    auto batch_size = output_dims[0];
    out->set_dims(phi::make_ddim({batch_size, sample_res_length}));
    mask->set_dims(phi::make_ddim({batch_size, sample_res_length}));
    if (labels) {
      labels->set_dims(phi::make_ddim({batch_size, sample_res_length}));
    }
  } else {
    out->set_dims(ddim);
    mask->set_dims(ddim);
    if (labels) {
      labels->set_dims(ddim);
    }
  }
  out->set_dtype(x.dtype());
  mask->set_dtype(x.dtype());
  if (labels) {
    labels->set_dtype(x.dtype());
  }
}
}  // namespace phi
