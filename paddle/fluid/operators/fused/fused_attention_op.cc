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

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::operators {

class FusedAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("QKVW"), "Input", "QKVW", "FusedAttentionOp");
    OP_INOUT_CHECK(
        ctx->HasInput("OutLinearW"), "Input", "OutLinearW", "FusedAttentionOp");

    if (ctx->Attrs().Get<bool>("pre_layer_norm") == true) {
      OP_INOUT_CHECK(
          ctx->HasOutput("LnMean"), "Output", "LnMean", "FusedAttentionOp");
      OP_INOUT_CHECK(ctx->HasOutput("LnVariance"),
                     "Output",
                     "LnVariance",
                     "FusedAttentionOp");
      OP_INOUT_CHECK(
          ctx->HasOutput("LnOut"), "Output", "LnOut", "FusedAttentionOp");
    } else {
      OP_INOUT_CHECK(
          ctx->HasOutput("Ln2Mean"), "Output", "Ln2Mean", "FusedAttentionOp");
      OP_INOUT_CHECK(ctx->HasOutput("Ln2Variance"),
                     "Output",
                     "Ln2Variance",
                     "FusedAttentionOp");
      OP_INOUT_CHECK(ctx->HasOutput("BiasDropoutResidualOut"),
                     "Output",
                     "BiasDropoutResidualOut",
                     "FusedAttentionOp");
    }

    // qkv_out: [batch_size, seq_len, 3, num_head, dim_head]
    OP_INOUT_CHECK(
        ctx->HasOutput("QKVOut"), "Output", "QKVOut", "FusedAttentionOp");
    if (ctx->HasInput("QKVBias")) {
      OP_INOUT_CHECK(ctx->HasOutput("QKVBiasOut"),
                     "Output",
                     "QKVBiasOut",
                     "FusedAttentionOp");
    }
    OP_INOUT_CHECK(ctx->HasOutput("TransposeOut2"),
                   "Output",
                   "TransposeOut2",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("QKOut"), "Output", "QKOut", "FusedAttentionOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("QKTVOut"), "Output", "QKTVOut", "FusedAttentionOp");

    if (ctx->HasInput("CacheKV")) {
      OP_INOUT_CHECK(ctx->HasOutput("CacheKVOut"),
                     "Output",
                     "CacheKVOut",
                     "FusedAttentionOp");
    }
    if (ctx->HasInput("SrcMask")) {
      OP_INOUT_CHECK(ctx->HasOutput("SrcMaskOut"),
                     "Output",
                     "SrcMaskOut",
                     "FusedAttentionOp");
    }
    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxOut"),
                   "Output",
                   "SoftmaxOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("AttnDropoutMaskOut"),
                   "Output",
                   "AttnDropoutMaskOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("AttnDropoutOut"),
                   "Output",
                   "AttnDropoutOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("FMHAOut"), "Output", "FMHAOut", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("OutLinearOut"),
                   "Output",
                   "OutLinearOut",
                   "FusedAttentionOp");

    OP_INOUT_CHECK(ctx->HasOutput("DropoutMaskOut"),
                   "Output",
                   "DropoutMaskOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "FusedAttentionOp");

    int num_heads = ctx->Attrs().Get<int>("num_heads");
    bool transpose_qkv_wb = ctx->Attrs().Get<bool>("transpose_qkv_wb");

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // if transpose_qkv_wb is False
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    // if transpose_qkv_wb is True
    // y: qkv's weight: [dim_embed, 3 * dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("QKVW");
    int dim_head;
    int hidden_size;
    int nranks = 1;
    if (transpose_qkv_wb) {
      PADDLE_ENFORCE_EQ(y_dim.size(),
                        2,
                        common::errors::InvalidArgument(
                            "The dimensions of qkv_weight must be 2 if enable"
                            "transpose_qkv_wb: (dim_embed, 3 * dim_embed),"
                            "but received dimensions of"
                            "Input is [%d]",
                            y_dim.size()));
      PADDLE_ENFORCE_GT(num_heads,
                        0,
                        common::errors::InvalidArgument(
                            "The num_heads must be provided and greater than 0 "
                            "if enable transpose_qkv_wb, but we got %d.",
                            num_heads));
      PADDLE_ENFORCE_EQ(y_dim[0] % num_heads,
                        0,
                        common::errors::InvalidArgument(
                            "First dim of qkv_w must be divisible by num heads "
                            "if enable transpose_qkv_wb, but receive first "
                            "dim of qkv_w is %d and num_heads is %d.",
                            y_dim[0],
                            num_heads));
      if (ctx->Attrs().Get<int>("ring_id") == -1) {
        PADDLE_ENFORCE_EQ(y_dim[0] * 3,
                          y_dim[1],
                          common::errors::InvalidArgument(
                              "The dimensions of qkv_weight must be 2"
                              "(dim_embed, 3 * dim_embed)."));
      } else {
        // compute the mp nranks
        nranks = (y_dim[0] * 3) / y_dim[1];
      }
      dim_head = y_dim[0] / (num_heads * nranks);
      hidden_size = y_dim[0];
    } else {
      PADDLE_ENFORCE_EQ(y_dim.size(),
                        4,
                        common::errors::InvalidArgument(
                            "The dimensions of qkv_weight must be 4 if not"
                            "enable transpose_qkv_wb: (3, num_head, dim_head, "
                            "dim_embed), but received [%d]",
                            y_dim.size()));
      PADDLE_ENFORCE_EQ(y_dim[0],
                        3,
                        common::errors::InvalidArgument(
                            "First dim of qkv_w must be 3 if disable "
                            "transpose_qkv_wb, but we got %d.",
                            y_dim[0]));
      if (ctx->Attrs().Get<int>("ring_id") == -1) {
        PADDLE_ENFORCE_EQ(y_dim[1] * y_dim[2],
                          y_dim[3],
                          common::errors::InvalidArgument(
                              "The dimensions of qkv_weight must be 4"
                              "(3, num_head, dim_head, dim_embed),"
                              "and must satisfy the limitations: "
                              "(num_head * dim_head == dim_embed)"));
      }
      num_heads = y_dim[1];
      dim_head = y_dim[2];
      hidden_size = y_dim[3];
    }
    PADDLE_ENFORCE_EQ(
        x_dim.size(),
        3,
        common::errors::InvalidArgument("The dimensions of x must be 3"
                                        "(batch_size, seq_len, dim_embed),"
                                        "but received dimensions of"
                                        "Input is [%d]",
                                        x_dim.size()));

    PADDLE_ENFORCE_EQ(x_dim[2],
                      hidden_size,
                      common::errors::InvalidArgument(
                          "ShapeError: the dimension of x_dim[2] and y_dim[3] "
                          "(y_dim[1] if enable transpose_qkv_w) "
                          "must be equal. But received: the shape "
                          "of input x = [%s], and the shape of "
                          "input qkv_weight = [%s]",
                          x_dim,
                          y_dim));

    if (ctx->Attrs().Get<bool>("pre_layer_norm") == true) {
      ctx->SetOutputDim("LnMean", {x_dim[0] * x_dim[1]});
      ctx->SetOutputDim("LnVariance", {x_dim[0] * x_dim[1]});
      ctx->SetOutputDim("LnOut", ctx->GetInputDim("X"));
    } else {
      ctx->SetOutputDim("Ln2Mean", {x_dim[0] * x_dim[1]});
      ctx->SetOutputDim("Ln2Variance", {x_dim[0] * x_dim[1]});
      ctx->SetOutputDim("BiasDropoutResidualOut", ctx->GetInputDim("X"));
    }

    if (transpose_qkv_wb) {
      // [batch_size, seq_len, 3 * num_heads * dim_head]
      ctx->SetOutputDim("QKVOut",
                        {x_dim[0], x_dim[1], 3 * num_heads * dim_head});

      if (ctx->HasInput("QKVBias")) {
        ctx->SetOutputDim("QKVBiasOut",
                          {x_dim[0], x_dim[1], 3 * num_heads * dim_head});
      }
    } else {
      // [batch_size, seq_len, 3, num_head, head_size]
      ctx->SetOutputDim("QKVOut", {x_dim[0], x_dim[1], 3, num_heads, dim_head});

      if (ctx->HasInput("QKVBias")) {
        ctx->SetOutputDim("QKVBiasOut",
                          {x_dim[0], x_dim[1], 3, num_heads, dim_head});
      }
    }

    // [3, batch_size, num_head, seq_len, head_size]
    ctx->SetOutputDim("TransposeOut2",
                      {3, x_dim[0], num_heads, x_dim[1], dim_head});

    // cache_seq_len + seq_len if cache else seq_len
    auto out_seq_len = x_dim[1];
    if (ctx->HasInput("CacheKV")) {
      // [2, batch_size, num_head, cache_seq_len, head_size]
      auto c_dim = ctx->GetInputDim("CacheKV");

      PADDLE_ENFORCE_EQ(
          c_dim.size(),
          5,
          common::errors::InvalidArgument(
              "The CacheKV must be 5 dims, but got %d", c_dim.size()));
      PADDLE_ENFORCE_EQ(c_dim[0],
                        2,
                        common::errors::InvalidArgument(
                            "The first dim of CacheKV must be 2, but got %d",
                            c_dim[0]));  // 2
      PADDLE_ENFORCE_EQ(c_dim[1],
                        x_dim[0],
                        common::errors::InvalidArgument(
                            "The second dim of CacheKV must be equal with "
                            "batch size %d, but got %d",
                            x_dim[0],
                            c_dim[1]));  // batch_size
      PADDLE_ENFORCE_EQ(c_dim[2],
                        num_heads,
                        common::errors::InvalidArgument(
                            "The third dim of CacheKV must be equal with num "
                            "head %d, but got %d",
                            num_heads,
                            c_dim[2]));  // num_head
      // In compile stage, input seq_len can be -1, in that case
      // c_dim[3] may < 0 in while
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_GE(
            c_dim[3],
            0,
            common::errors::InvalidArgument(
                "The forth dim of CacheKV must be greater than 0, but got %d",
                c_dim[3]));  // cache_seq_len
      }

      PADDLE_ENFORCE_EQ(c_dim[4],
                        dim_head,
                        common::errors::InvalidArgument(
                            "The fifth dim of CacheKV must be equal with head "
                            "size %d, but got %d",
                            dim_head,
                            c_dim[4]));  // head_size

      out_seq_len += c_dim[3];
      // [3, batch_size, num_head, cache_seq_len + seq_len, head_size]
      ctx->SetOutputDim("CacheKVOut",
                        {c_dim[0], c_dim[1], c_dim[2], out_seq_len, c_dim[4]});
    }

    // [batch, num_head, seq_len, out_seq_len]
    ctx->SetOutputDim("QKOut", {x_dim[0], num_heads, x_dim[1], out_seq_len});

    if (ctx->HasInput("SrcMask")) {
      ctx->SetOutputDim("SrcMaskOut",
                        {x_dim[0], num_heads, x_dim[1], out_seq_len});
    }
    // the same as QKOut's shape.
    ctx->SetOutputDim("AttnDropoutOut",
                      {x_dim[0], num_heads, x_dim[1], out_seq_len});
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("AttnDropoutMaskOut",
                        {x_dim[0], num_heads, x_dim[1], out_seq_len});
    }
    ctx->SetOutputDim("SoftmaxOut",
                      {x_dim[0], num_heads, x_dim[1], out_seq_len});
    // [batch_size, num_heads, seq_len, head_dim]
    ctx->SetOutputDim("QKTVOut", {x_dim[0], num_heads, x_dim[1], dim_head});
    // [batch_size, seq_len, number of heads*head size]
    ctx->SetOutputDim("FMHAOut", {x_dim[0], x_dim[1], num_heads, dim_head});

    ctx->SetOutputDim("OutLinearOut", ctx->GetInputDim("X"));

    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("DropoutMaskOut", ctx->GetInputDim("X"));
    }

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class FusedAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("LnScale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable();
    AddInput("LnBias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable();
    AddInput("QKVW", "The qkv weight tensor.");
    AddInput("QKVBias", "The qkv bias tensor.").AsDispensable();
    AddInput("CacheKV", "(optional) The cached KV for generation inference.")
        .AsDispensable();
    AddInput("SrcMask", "(optional) The attention mask tensor in fmha.")
        .AsDispensable();
    AddInput("OutLinearW", "The out_linear weight tensor.");
    AddInput("OutLinearBias", "The out_linear bias tensor.").AsDispensable();
    AddInput("Ln2Scale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable();
    AddInput("Ln2Bias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable();
    AddOutput("LnMean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("LnVariance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("LnOut", "The output of pre layer_norm.").AsIntermediate();
    AddOutput("QKVOut", "Result after qkv.").AsIntermediate();
    AddOutput("QKVBiasOut", "Result after qkv and bias op.").AsIntermediate();
    AddOutput("TransposeOut2", "Result in fmha.").AsIntermediate();
    AddOutput("QKOut", "Result in fmha.").AsIntermediate();
    AddOutput("QKTVOut", "Result in fmha.").AsIntermediate();
    AddOutput("SoftmaxOut", "Result in fmha.").AsIntermediate();
    AddOutput("AttnDropoutMaskOut", "Result in fmha.").AsIntermediate();
    AddOutput("AttnDropoutOut", "Result in fmha.").AsIntermediate();
    AddOutput("SrcMaskOut", "Result in fmha.").AsIntermediate();
    AddOutput("FMHAOut", "Result after fmha.").AsIntermediate();
    AddOutput("OutLinearOut", "Result after out_linear.").AsIntermediate();
    AddOutput("DropoutMaskOut", "The random sampled dropout mask.")
        .AsIntermediate();
    AddOutput("Ln2Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Ln2Variance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("BiasDropoutResidualOut",
              "Result of residual + dropout(src + bias).")
        .AsIntermediate();
    AddOutput("CacheKVOut", "The updated cache KV.").AsDispensable();
    AddOutput("Y", "Result after attention.");

    AddAttr<int>("num_heads", "The number head for multi_head_attention.")
        .SetDefault(-1);
    AddAttr<bool>("transpose_qkv_wb",
                  "The qkv_w shape is (h, 3h), do transpose to it.")
        .SetDefault(false);
    AddAttr<bool>("pre_layer_norm",
                  "if true, the attention op uses pre_layer_norm architecture, "
                  "else, uses post_layer_norm architecture. "
                  "[default false].")
        .SetDefault(false);
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f,
                            true,
                            common::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });

    // for dropout in fmha.
    AddAttr<float>("attn_dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f,
              true,
              common::errors::InvalidArgument(
                  "'attn_dropout_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("attn_dropout_fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(true);
    AddAttr<int>("attn_dropout_seed", "Dropout random seed.").SetDefault(0);
    AddAttr<std::string>(
        "attn_dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_rate)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_rate)"
        "2. upscale_in_train, upscale the outcome at training time, do nothing "
        "in inference"
        "   train: out = input * mask / ( 1.0 - dropout_rate )"
        "   inference: out = input"
        "   dropout op can be removed from the program. the program will be "
        "efficient")
        .SetDefault("upscale_in_train")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train",
              true,
              common::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });

    AddAttr<float>("dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f,
                            true,
                            common::errors::InvalidArgument(
                                "'dropout_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("dropout_fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(true);
    AddAttr<int>("dropout_seed", "Dropout random seed.").SetDefault(0);
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "The meaning is the same as 'attn_dropout_implementation'.")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train",
              true,
              common::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<float>("ln_epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &ln_epsilon) {
          PADDLE_ENFORCE_EQ(ln_epsilon >= 0.0f && ln_epsilon <= 0.001f,
                            true,
                            common::errors::InvalidArgument(
                                "'epsilon' of the second LayerNorm in Fused "
                                "attention op should be between"
                                "0.0 and 0.001, But received [%s].",
                                ln_epsilon));
        });
    AddAttr<bool>("add_residual", "Whether to add residual.").SetDefault(true);
    AddAttr<int>(
        "ring_id",
        "ring id for tensor model parallel. distributed training and inference")
        .SetDefault(-1);

    AddComment(R"DOC(
  The fused_attention operator is the same as following pseudo codes:

  // @input: [batch_size, seq_len, embed_dim]
  // @final_out: [batch_size, seq_len, num_heads, head_dim]
  residual = input
  if (pre_layernorm)
    query = layer_norm(input);
  out = compute_qkv(query) + qkv_bias;
  // fmha module
  {
    out = transpose(out, perm=[2, 0, 3, 1, 4]);
    out = q * k^t;
    out = attn_mask + out;
    out = softmax(out);
    out = dropout(out);
    out = out * v;
    out = transpose(out, perm=[0, 2, 1, 3]);

  }
  // out linear
  out = linear(out);
  if add_residual:
    out = residual + dropout(out);
  else:
    out = dropout(out);
  if (!pre_layernorm)
    out = layer_norm(out);
    )DOC");
  }
};

class FusedAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"),
                      false,
                      common::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));

    if (ctx->Attrs().Get<bool>("pre_layer_norm") == false) {
      OP_INOUT_CHECK(
          ctx->HasInput("Ln2Mean"), "Input", "Ln2Mean", "FusedAttentionGrad");
      OP_INOUT_CHECK(ctx->HasInput("Ln2Variance"),
                     "Input",
                     "Ln2Variance",
                     "FusedAttentionGrad");
      if (ctx->HasOutput(framework::GradVarName("Ln2Scale"))) {
        ctx->SetOutputDim(framework::GradVarName("Ln2Scale"),
                          ctx->GetInputDim("Ln2Scale"));
      }
      if (ctx->HasOutput(framework::GradVarName("Ln2Bias"))) {
        ctx->SetOutputDim(framework::GradVarName("Ln2Bias"),
                          ctx->GetInputDim("Ln2Bias"));
      }
    } else {
      OP_INOUT_CHECK(
          ctx->HasInput("LnMean"), "Input", "LnMean", "FusedAttentionGrad");
      OP_INOUT_CHECK(ctx->HasInput("LnVariance"),
                     "Input",
                     "LnVariance",
                     "FusedAttentionGrad");
      OP_INOUT_CHECK(
          ctx->HasInput("LnOut"), "Input", "LnOut", "FusedAttentionGrad");
    }

    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedAttentionGrad");
    OP_INOUT_CHECK(
        ctx->HasInput("QKVW"), "Input", "QKVW", "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearW"),
                   "Input",
                   "OutLinearW",
                   "FusedAttentionGrad");

    if (ctx->Attrs().Get<bool>("pre_layer_norm") == true) {
      if (ctx->HasOutput(framework::GradVarName("LnScale"))) {
        ctx->SetOutputDim(framework::GradVarName("LnScale"),
                          ctx->GetInputDim("LnScale"));
      }
      if (ctx->HasOutput(framework::GradVarName("LnBias"))) {
        ctx->SetOutputDim(framework::GradVarName("LnBias"),
                          ctx->GetInputDim("LnBias"));
      }
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("OutLinearBias"))) {
      ctx->SetOutputDim(framework::GradVarName("OutLinearBias"),
                        ctx->GetInputDim("OutLinearBias"));
    }
    if (ctx->HasOutput(framework::GradVarName("OutLinearW"))) {
      ctx->SetOutputDim(framework::GradVarName("OutLinearW"),
                        ctx->GetInputDim("OutLinearW"));
    }
    if (ctx->HasOutput(framework::GradVarName("QKVW"))) {
      ctx->SetOutputDim(framework::GradVarName("QKVW"),
                        ctx->GetInputDim("QKVW"));
    }
    if (ctx->HasOutput(framework::GradVarName("QKVBias"))) {
      ctx->SetOutputDim(framework::GradVarName("QKVBias"),
                        ctx->GetInputDim("QKVBias"));
    }

    if (ctx->Attrs().Get<bool>("pre_layer_norm") == true) {
      if (ctx->HasOutput(framework::GradVarName("LnOut"))) {
        ctx->SetOutputDim(framework::GradVarName("LnOut"),
                          ctx->GetInputDim("LnOut"));
      }
    } else {
      if (ctx->HasOutput(framework::GradVarName("BiasDropoutResidualOut"))) {
        ctx->SetOutputDim(framework::GradVarName("BiasDropoutResidualOut"),
                          ctx->GetInputDim("BiasDropoutResidualOut"));
      }
    }
    if (ctx->HasOutput(framework::GradVarName("FMHAOut"))) {
      ctx->SetOutputDim(framework::GradVarName("FMHAOut"),
                        ctx->GetInputDim("FMHAOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("QKTVOut"))) {
      ctx->SetOutputDim(framework::GradVarName("QKTVOut"),
                        ctx->GetInputDim("QKTVOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("TransposeOut2"))) {
      ctx->SetOutputDim(framework::GradVarName("TransposeOut2"),
                        ctx->GetInputDim("TransposeOut2"));
    }
    if (ctx->HasOutput(framework::GradVarName("QKOut"))) {
      ctx->SetOutputDim(framework::GradVarName("QKOut"),
                        ctx->GetInputDim("QKOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("SoftmaxOut"))) {
      ctx->SetOutputDim(framework::GradVarName("SoftmaxOut"),
                        ctx->GetInputDim("SoftmaxOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("AttnDropoutOut"))) {
      ctx->SetOutputDim(framework::GradVarName("AttnDropoutOut"),
                        ctx->GetInputDim("AttnDropoutOut"));
    }

    if (ctx->HasOutput(framework::GradVarName("SrcMaskOut"))) {
      ctx->SetOutputDim(framework::GradVarName("SrcMaskOut"),
                        ctx->GetInputDim("SrcMaskOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("QKVOut"))) {
      ctx->SetOutputDim(framework::GradVarName("QKVOut"),
                        ctx->GetInputDim("QKVOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("QKVBiasOut"))) {
      ctx->SetOutputDim(framework::GradVarName("QKVBiasOut"),
                        ctx->GetInputDim("QKVBiasOut"));
    }
    if (ctx->HasOutput(framework::GradVarName("OutLinearOut"))) {
      ctx->SetOutputDim(framework::GradVarName("OutLinearOut"),
                        ctx->GetInputDim("OutLinearOut"));
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_attention_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    // inputs x, parameters and their grad.
    op->SetInput("X", this->Input("X"));
    op->SetInput("QKVW", this->Input("QKVW"));

    if (this->HasInput("QKVBias")) {
      op->SetInput("QKVBias", this->Input("QKVBias"));
      op->SetOutput(framework::GradVarName("QKVBias"),
                    this->InputGrad("QKVBias"));
      op->SetInput("QKVBiasOut", this->Output("QKVBiasOut"));
      op->SetOutput(framework::GradVarName("QKVBiasOut"),
                    this->OutputGrad("QKVBiasOut"));
    }

    if (this->HasInput("SrcMask")) {
      op->SetInput("SrcMask", this->Input("SrcMask"));
      op->SetInput("SrcMaskOut", this->Output("SrcMaskOut"));
      op->SetOutput(framework::GradVarName("SrcMaskOut"),
                    this->OutputGrad("SrcMaskOut"));
    }

    op->SetInput("OutLinearW", this->Input("OutLinearW"));
    if (this->HasInput("OutLinearBias")) {
      op->SetInput("OutLinearBias", this->Input("OutLinearBias"));
      op->SetOutput(framework::GradVarName("OutLinearBias"),
                    this->InputGrad("OutLinearBias"));
    }

    op->SetAttrMap(this->Attrs());
    bool is_pre_layer_norm =
        PADDLE_GET_CONST(bool, op->GetAttr("pre_layer_norm"));
    if (is_pre_layer_norm) {
      if (this->HasInput("LnScale")) {
        op->SetInput("LnScale", this->Input("LnScale"));
        op->SetOutput(framework::GradVarName("LnScale"),
                      this->InputGrad("LnScale"));
      }
      if (this->HasInput("LnBias")) {
        op->SetInput("LnBias", this->Input("LnBias"));
        op->SetOutput(framework::GradVarName("LnBias"),
                      this->InputGrad("LnBias"));
      }
    } else {
      if (this->HasInput("Ln2Scale")) {
        op->SetInput("Ln2Scale", this->Input("Ln2Scale"));
        op->SetOutput(framework::GradVarName("Ln2Scale"),
                      this->InputGrad("Ln2Scale"));
      }
      if (this->HasInput("Ln2Bias")) {
        op->SetInput("Ln2Bias", this->Input("Ln2Bias"));
        op->SetOutput(framework::GradVarName("Ln2Bias"),
                      this->InputGrad("Ln2Bias"));
      }
    }

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("QKVW"), this->InputGrad("QKVW"));

    op->SetOutput(framework::GradVarName("OutLinearW"),
                  this->InputGrad("OutLinearW"));

    // use forward outputs as backward inputs.
    if (is_pre_layer_norm) {
      if (this->HasOutput("LnOut")) {
        op->SetInput("LnOut", this->Output("LnOut"));
      }
      if (this->HasOutput("LnMean")) {
        op->SetInput("LnMean", this->Output("LnMean"));
      }
      if (this->HasOutput("LnVariance")) {
        op->SetInput("LnVariance", this->Output("LnVariance"));
      }
    } else {
      op->SetInput("Ln2Mean", this->Output("Ln2Mean"));
      op->SetInput("Ln2Variance", this->Output("Ln2Variance"));
      op->SetInput("BiasDropoutResidualOut",
                   this->Output("BiasDropoutResidualOut"));
    }
    op->SetInput("QKVOut", this->Output("QKVOut"));

    op->SetInput("TransposeOut2", this->Output("TransposeOut2"));
    op->SetInput("QKOut", this->Output("QKOut"));
    op->SetInput("QKTVOut", this->Output("QKTVOut"));
    op->SetInput("SoftmaxOut", this->Output("SoftmaxOut"));
    op->SetInput("AttnDropoutMaskOut", this->Output("AttnDropoutMaskOut"));
    op->SetInput("AttnDropoutOut", this->Output("AttnDropoutOut"));

    op->SetInput("FMHAOut", this->Output("FMHAOut"));
    op->SetInput("OutLinearOut", this->Output("OutLinearOut"));
    op->SetInput("DropoutMaskOut", this->Output("DropoutMaskOut"));
    op->SetInput("QKVOut", this->Output("QKVOut"));

    // backward outputs: dinput
    if (is_pre_layer_norm) {
      if (this->HasOutput("LnOut")) {
        op->SetOutput(framework::GradVarName("LnOut"),
                      this->OutputGrad("LnOut"));
      }
    } else {
      op->SetOutput(framework::GradVarName("BiasDropoutResidualOut"),
                    this->OutputGrad("BiasDropoutResidualOut"));
    }

    op->SetOutput(framework::GradVarName("QKVOut"), this->OutputGrad("QKVOut"));

    op->SetOutput(framework::GradVarName("QKTVOut"),
                  this->OutputGrad("QKTVOut"));
    op->SetOutput(framework::GradVarName("TransposeOut2"),
                  this->OutputGrad("TransposeOut2"));
    op->SetOutput(framework::GradVarName("QKOut"), this->OutputGrad("QKOut"));
    op->SetOutput(framework::GradVarName("SoftmaxOut"),
                  this->OutputGrad("SoftmaxOut"));
    op->SetOutput(framework::GradVarName("AttnDropoutOut"),
                  this->OutputGrad("AttnDropoutOut"));

    op->SetOutput(framework::GradVarName("FMHAOut"),
                  this->OutputGrad("FMHAOut"));
    op->SetOutput(framework::GradVarName("OutLinearOut"),
                  this->OutputGrad("OutLinearOut"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(FusedAttentionGradNoNeedBufferInferer,
                                    "QKVBiasOut",
                                    "QKVOut",
                                    "QKOut",
                                    "QKTVOut",
                                    "OutLinearOut",
                                    "SrcMask");

}  // namespace paddle::operators

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_attention,
                  ops::FusedAttentionOp,
                  ops::FusedAttentionOpMaker,
                  ops::FusedAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedAttentionGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_attention_grad,
                  ops::FusedAttentionGradOp,
                  ops::FusedAttentionGradNoNeedBufferInferer);

REGISTER_OP_VERSION(fused_attention)
    .AddCheckpoint(
        R"ROC(
              Add a new attribute [add_residual] )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "add_residual",
            "A flag to indicate whether to add residual.",
            true));
