// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/backends/onednn/matmul_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace {
using dnnl::memory;
using paddle::framework::ExecutionContext;
using paddle::framework::GradVarName;
using phi::OneDNNContext;
using phi::vectorize;
using phi::funcs::OneDNNGetDataType;

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
phi::DenseTensor FoldOuterDims(const phi::DenseTensor &input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename T>
phi::DenseTensor FoldFirstAndLastDims(const OneDNNContext &dev_ctx,
                                      const phi::DenseTensor *input) {
  auto input_dims = vectorize(input->dims());
  if (input_dims.size() != 3) {
    return *input;
  }

  phi::DenseTensor output;
  output.Resize({input_dims[1], input_dims[0], input_dims[2]});

  auto output_dims = vectorize(output.dims());

  memory::data_type input_type = phi::funcs::ToOneDNNDataType(input->dtype());
  phi::funcs::ReorderOneDNNHandler reorder_handler(
      output_dims, input->dtype(), input_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      memory::format_tag::abc, phi::funcs::to_void_cast(input->data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      &output, memory::format_tag::bac, dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                  reorder_dst_memory_p);

  auto &astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  output.Resize({input_dims[1], input_dims[0] * input_dims[2]});
  return output;
}

phi::DDim GetDimForInput(const ExecutionContext &ctx, std::string input_name) {
  auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
  auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
  auto input_dims = ctx.Input<phi::DenseTensor>(input_name)->dims();
  if (!shape.empty() && !axis.empty()) {
    return input_dims.reshape(shape).transpose(axis);
  }
  return input_dims;
}

template <typename XT, typename YT, typename OT>
class MatMulV1OneDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  MatMulV1OneDNNHandler(const ExecutionContext &ctx,
                        const dnnl::engine engine,
                        phi::Place cpu_place,
                        const std::vector<int64_t> &x_org_dims,
                        bool trans_x,
                        const std::vector<int64_t> &y_org_dims,
                        bool trans_y,
                        bool is_output_fused,
                        const std::vector<int64_t> &x_strides_override,
                        const std::vector<int64_t> &y_strides_override)
      : phi::funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul>(engine,
                                                              cpu_place) {
    // M X K * K X N
    std::vector<int64_t> x_dims(x_org_dims);
    std::vector<int64_t> y_dims(y_org_dims);

    const int MB_idx = x_dims.size() - 3;
    const int H_idx = x_dims.size() - 2;
    const int W_idx = x_dims.size() - 1;

    if (trans_x) std::swap(x_dims[H_idx], x_dims[W_idx]);
    if (trans_y) std::swap(y_dims[H_idx], y_dims[W_idx]);

    const memory::dim M = x_dims[H_idx];
    const memory::dim K = x_dims[W_idx];
    const memory::dim N = y_dims[W_idx];

    std::vector<int64_t> x_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> y_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_ddims(x_dims.size() - 3, 1);

    x_strides.reserve(x_dims.size());
    y_strides.reserve(x_dims.size());
    out_strides.reserve(x_dims.size());

    if (!x_strides_override.empty()) {
      x_strides = x_strides_override;
    } else {
      if (!trans_x) {
        x_strides.insert(x_strides.end(), {M * K, K, 1});
      } else {
        x_strides.insert(x_strides.end(), {M * K, 1, M});
      }
    }

    if (!y_strides_override.empty()) {
      y_strides = y_strides_override;
    } else {
      if (!trans_y) {
        y_strides.insert(y_strides.end(), {N * K, N, 1});
      } else {
        y_strides.insert(y_strides.end(), {N * K, 1, K});
      }
    }

    out_strides.insert(out_strides.end(), {M * N, N, 1});
    out_ddims.insert(out_ddims.end(),
                     {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

    for (int i = x_dims.size() - 4; i >= 0; --i) {
      out_ddims[i] = std::max(x_dims[i], y_dims[i]);
      if (x_strides_override.empty()) {
        x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
      }
      if (y_strides_override.empty()) {
        y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
      }
      out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
    }

    // TODO(jczaja): Why not for int8??
    if (!phi::funcs::is_int8<OT>() && is_output_fused) {
      std::vector<int> transpose_axis = {0, 2, 1, 3};
      out_strides = phi::funcs::FakeTransposeStrides(out_ddims, transpose_axis);
    }

    auto x_md =
        memory::desc(x_dims, phi::funcs::OneDNNGetDataType<XT>(), x_strides);
    auto y_md =
        memory::desc(y_dims, phi::funcs::OneDNNGetDataType<YT>(), y_strides);
    auto out_md = memory::desc(
        out_ddims, phi::funcs::OneDNNGetDataType<OT>(), out_strides);

    const dnnl::primitive_attr matmul_attrs = CreateMatmulAttrs(ctx);

    this->AcquireForwardPrimitiveDescriptor(matmul_attrs, x_md, y_md, out_md);
  }

  float ComputeOutputScale(const ExecutionContext &ctx) {
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;
    if (ctx.HasAttr("Scale_x") && ctx.HasAttr("Scale_y") &&
        ctx.HasAttr("Scale_out")) {
      float scale_x = ctx.Attr<float>("Scale_x");
      float scale_y = ctx.Attr<float>("Scale_y");
      bool force_fp32_out = ctx.HasAttr("force_fp32_output")
                                ? ctx.Attr<bool>("force_fp32_output")
                                : false;
      float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
      alpha *= scale_out / (scale_x * scale_y);
    }
    return alpha;
  }

  dnnl::primitive_attr CreateMatmulAttrs(const ExecutionContext &ctx) {
    dnnl::primitive_attr matmul_attrs;
    dnnl::post_ops post_operations;

    float scale_out = ComputeOutputScale(ctx);
    if (scale_out != 1.0f) {
      matmul_attrs.set_output_scales(0, {scale_out});
    }

    if (ctx.HasInput("ResidualData")) {
      auto *residual_data = ctx.Input<phi::DenseTensor>("ResidualData");
      auto residual_data_tz = phi::vectorize(residual_data->dims());
      auto residual_data_md = memory::desc(residual_data_tz,
                                           phi::funcs::OneDNNGetDataType<OT>(),
                                           dnnl::memory::format_tag::any);
      post_operations.append_binary(dnnl::algorithm::binary_add,
                                    residual_data_md);
      if (ctx.HasAttr("Scale_in_eltwise")) {
        float sum_scale = scale_out / ctx.Attr<float>("Scale_in_eltwise");
        post_operations.append_sum(sum_scale);
      }
    }

    if (ctx.HasAttr("fused_output_scale")) {
      float scale_alpha = ctx.Attr<float>("fused_output_scale");
      post_operations.append_eltwise(
          1.0, dnnl::algorithm::eltwise_linear, scale_alpha, 0.0f);
    }

    matmul_attrs.set_post_ops(post_operations);
    return matmul_attrs;
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const phi::DenseTensor *input) {
    const YT *input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(),
        phi::funcs::to_void_cast<YT>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(phi::DenseTensor *output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence phi::DenseTensor size is bigger that
    // dst memory primitive size. So would we request less memory that is there
    // and it triggers an assertion.  So as there is no 'any' format here we can
    // leave default size of phi::DenseTensor as computed in ComputeInferShape
    OT *ptr = output->mutable_data<OT>(this->place_);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

template <typename XT, typename YT, typename OT>
class MatMulOneDNNHandler
    : public phi::funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  MatMulOneDNNHandler(const dnnl::engine engine,
                      phi::Place cpu_place,
                      phi::DenseTensor *x,
                      bool trans_x,
                      phi::DenseTensor *y,
                      bool trans_y,
                      phi::DenseTensor *out,
                      float scale)
      : phi::funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul>(engine,
                                                              cpu_place) {
    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x->dims(), 0, trans_x);
    auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y->dims(), 0, trans_y);

    memory::dim x_bs = mat_dim_x.batch_size_;
    memory::dim y_bs = mat_dim_y.batch_size_;

    memory::dim out_bs = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    const memory::dim M = mat_dim_x.height_;
    const memory::dim N = mat_dim_y.width_;
    const memory::dim K = mat_dim_x.width_;

    memory::dims x_dims = {x_bs > 0 ? x_bs : 1, M, K};
    memory::dims y_dims = {y_bs > 0 ? y_bs : 1, K, N};
    memory::dims out_dims = {out_bs, M, N};

    memory::dims x_strides =
        !trans_x ? memory::dims{M * K, K, 1} : memory::dims{M * K, 1, M};

    memory::dims y_strides =
        !trans_y ? memory::dims{N * K, N, 1} : memory::dims{N * K, 1, K};
    memory::dims out_strides = memory::dims{M * N, N, 1};

    auto x_md = memory::desc(x_dims, OneDNNGetDataType<XT>(), x_strides);
    auto y_md = memory::desc(y_dims, OneDNNGetDataType<YT>(), y_strides);
    auto out_md = memory::desc(out_dims, OneDNNGetDataType<OT>(), out_strides);

    dnnl::primitive_attr attrs;
    if (scale != 1.0f) attrs.set_output_scales(0, {scale});

    this->AcquireForwardPrimitiveDescriptor(attrs, x_md, y_md, out_md);
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const phi::DenseTensor *input) {
    const YT *input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(),
        phi::funcs::to_void_cast<YT>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(phi::DenseTensor *output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence phi::DenseTensor size is bigger that
    // dst memory primitive size. So would we request less memory that is there
    // and it triggers an assertion.  So as there is no 'any' format here we can
    // leave default size of phi::DenseTensor as computed in ComputeInferShape
    OT *ptr = output->mutable_data<OT>(this->place_);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
void ReshapeTensorToMatrixSequence(
    phi::DenseTensor *x, const phi::funcs::MatDescriptor &descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

/**
 * Reshape the x,y,out tensor to 3-D or 2-D tensor by matrix descriptor
 * Out = matmul(x, y)
 *
 * This method will first calculate X,Y matrix sequence, and then calculate
 * the out shape.
 *
 * Assume X = [BatchSize, H1, W1], Y = [BatchSize, H2, W2]
 * The out = [BatchSize, H1, W2]
 *
 * If there is no batch size in `X` and `Y`, the out will be [H1, W2]
 * If any of `X` and `Y` has batch size BatchSize, the out will have the
 * BatchSize.
 */
void ReshapeXYOutToMatrixSequence(phi::DenseTensor *x,
                                  phi::DenseTensor *y,
                                  phi::DenseTensor *out,
                                  bool trans_x,
                                  bool trans_y) {
  auto x_dim = phi::funcs::RowMatrixDimsFromVector(x->dims());
  auto y_dim = phi::funcs::ColumnMatrixDimsFromVector(y->dims());
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_,
                 mat_dim_y.width_});
  }

  ReshapeTensorToMatrixSequence(x, mat_dim_x);
  ReshapeTensorToMatrixSequence(y, mat_dim_y);
}

bool IsOutputFused(const ExecutionContext &ctx) {
  auto &fused_reshape_Out = ctx.Attr<std::vector<int>>("fused_reshape_Out");
  auto &fused_transpose_Out = ctx.Attr<std::vector<int>>("fused_transpose_Out");
  return !fused_reshape_Out.empty() && !fused_transpose_Out.empty();
}

template <typename T, typename T_out>
void ExecuteMatMulV1(const ExecutionContext &ctx,
                     const dnnl::engine onednn_engine,
                     const phi::DenseTensor *x,
                     const std::vector<int64_t> &x_dims,
                     bool trans_x,
                     const phi::DenseTensor *y,
                     const std::vector<int64_t> &y_dims,
                     bool trans_y,
                     phi::DenseTensor *out) {
  std::vector<int64_t> x_strides_override = phi::funcs::GetInputStrides(
      "X",
      x->dims(),
      trans_x,
      ctx.Attr<std::vector<int>>("fused_reshape_X"),
      ctx.Attr<std::vector<int>>("fused_transpose_X"));
  std::vector<int64_t> y_strides_override = phi::funcs::GetInputStrides(
      "Y",
      y->dims(),
      trans_y,
      ctx.Attr<std::vector<int>>("fused_reshape_Y"),
      ctx.Attr<std::vector<int>>("fused_transpose_Y"));

  MatMulV1OneDNNHandler<T, T, T_out> handler(ctx,
                                             onednn_engine,
                                             ctx.GetPlace(),
                                             x_dims,
                                             trans_x,
                                             y_dims,
                                             trans_y,
                                             IsOutputFused(ctx),
                                             x_strides_override,
                                             y_strides_override);
  const auto src_memory_p = handler.AcquireSrcMemory(x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(y);
  const auto dst_memory_p = handler.AcquireDstMemory(out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  if (ctx.HasInput("ResidualData")) {
    auto *residual_data = ctx.Input<phi::DenseTensor>("ResidualData");
    const auto residual_data_memory_p = handler.AcquireSrcMemory(residual_data);
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                        *residual_data_memory_p});
  }

  auto &astream = OneDNNContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  // TODO(jczaja): Explain why int8 format of dst is ABCD and do not need
  // permute
  if (IsOutputFused(ctx) && !phi::funcs::is_int8<T_out>()) {
    auto axis = ctx.Attr<std::vector<int>>("fused_transpose_Out");
    auto permuted_md = dst_memory_p->get_desc().permute_axes(axis);
    out->set_mem_desc(permuted_md.reshape(vectorize<int64_t>(out->dims())));
  } else {
    out->set_mem_desc(
        dst_memory_p->get_desc().reshape(vectorize<int64_t>(out->dims())));
  }
}

template <typename T>
class MatMulV1OneDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    if (ctx.HasAttr("head_number")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<int>("head_number"),
          1,
          phi::errors::Unimplemented(
              "oneDNN matmul doesn't support multiple heads. Expected "
              "head_number=1. But received `head_number` is %d",
              ctx.Attr<int>("head_number")));
    }
    constexpr bool is_int8 = phi::funcs::is_int8<T>();
    constexpr bool is_bfloat16 = phi::funcs::is_bfloat16<T>();
    const bool force_fp32_output = ctx.HasAttr("force_fp32_output")
                                       ? ctx.Attr<bool>("force_fp32_output")
                                       : false;

    const auto &dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto &onednn_engine = dev_ctx.GetEngine();

    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *y = ctx.Input<phi::DenseTensor>("Y");
    auto *out = ctx.Output<phi::DenseTensor>("Out");
    bool trans_x = ctx.HasAttr("trans_x") ? ctx.Attr<bool>("trans_x")
                                          : ctx.Attr<bool>("transpose_X");
    bool trans_y = ctx.HasAttr("trans_y") ? ctx.Attr<bool>("trans_y")
                                          : ctx.Attr<bool>("transpose_Y");

    auto x_dims = vectorize(GetDimForInput(ctx, "X"));
    auto y_dims = vectorize(GetDimForInput(ctx, "Y"));

    int ndims = std::max(x_dims.size(), y_dims.size());
    ndims = std::max(ndims, 3);

    std::vector<int64_t> x_bd_dims(ndims, 1);
    std::vector<int64_t> y_bd_dims(ndims, 1);

    CalculateMatrixDims(ctx, x_dims, y_dims, &x_bd_dims, &y_bd_dims, out);

    if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
      ExecuteMatMulV1<T, float>(ctx,
                                onednn_engine,
                                x,
                                x_bd_dims,
                                trans_x,
                                y,
                                y_bd_dims,
                                trans_y,
                                out);
    } else if (is_bfloat16) {
      ExecuteMatMulV1<T, paddle::platform::bfloat16>(ctx,
                                                     onednn_engine,
                                                     x,
                                                     x_bd_dims,
                                                     trans_x,
                                                     y,
                                                     y_bd_dims,
                                                     trans_y,
                                                     out);
    } else {
      ExecuteMatMulV1<T, int8_t>(ctx,
                                 onednn_engine,
                                 x,
                                 x_bd_dims,
                                 trans_x,
                                 y,
                                 y_bd_dims,
                                 trans_y,
                                 out);
    }
  }

 private:
  void CalculateMatrixDims(const ExecutionContext &ctx,
                           const std::vector<int64_t> &x_dims,
                           const std::vector<int64_t> &y_dims,
                           std::vector<int64_t> *x_bd_dims,
                           std::vector<int64_t> *y_bd_dims,
                           phi::DenseTensor *out) const {
    if (x_dims.size() == 1) {
      (*x_bd_dims)[(*x_bd_dims).size() - 1] = x_dims[0];
    } else if (x_dims.size() == 2) {
      (*x_bd_dims)[(*x_bd_dims).size() - 1] = x_dims[1];
      (*x_bd_dims)[(*x_bd_dims).size() - 2] = x_dims[0];
    } else {
      for (size_t i = 0; i < x_dims.size(); ++i) {
        (*x_bd_dims)[(*x_bd_dims).size() - x_dims.size() + i] = x_dims[i];
      }
    }
    if (y_dims.size() == 1) {
      (*y_bd_dims)[(*x_bd_dims).size() - 2] = y_dims[0];
    } else if (y_dims.size() == 2) {
      (*y_bd_dims)[(*y_bd_dims).size() - 1] = y_dims[1];
      (*y_bd_dims)[(*y_bd_dims).size() - 2] = y_dims[0];
    } else {
      for (size_t i = 0; i < y_dims.size(); ++i) {
        (*y_bd_dims)[(*y_bd_dims).size() - y_dims.size() + i] = y_dims[i];
      }
    }

    if (!IsOutputFused(ctx) && x_dims.size() > 2 && y_dims.size() > 2) {
      auto out_dims = vectorize(out->dims());
      for (size_t i = 0; i < (*x_bd_dims).size() - 2; ++i) {
        PADDLE_ENFORCE_EQ(
            (*x_bd_dims)[i] == (*y_bd_dims)[i] || (*x_bd_dims)[i] == 1 ||
                (*y_bd_dims)[i] == 1,
            true,
            phi::errors::InvalidArgument(
                "phi::DenseTensor dimensions are incorrect for broadcasting."
                "Dimensions in X and Y must be same or equal to 1, but "
                "received x_dim[%d]=%d and y_dims[%d]= %d",
                i,
                (*x_bd_dims)[i],
                i,
                (*y_bd_dims)[i]));
        (out_dims)[i] = std::max((*x_bd_dims)[i], (*y_bd_dims)[i]);
      }
      out->Resize(phi::make_ddim((out_dims)));
    }
  }
};

template <typename T>
class MatMulV1GradOneDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    if (ctx.HasAttr("head_number")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<int>("head_number"),
          1,
          phi::errors::Unimplemented(
              "oneDNN matmul doesn't support multiple heads. Expected "
              "head_number=1. But received `head_number` is %d",
              ctx.Attr<int>("head_number")));
    }

    const auto &dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto &onednn_engine = dev_ctx.GetEngine();

    auto x = *ctx.Input<phi::DenseTensor>("X");
    auto y = *ctx.Input<phi::DenseTensor>("Y");
    auto dout =
        *ctx.Input<phi::DenseTensor>(paddle::framework::GradVarName("Out"));
    auto *dx =
        ctx.Output<phi::DenseTensor>(paddle::framework::GradVarName("X"));
    auto *dy =
        ctx.Output<phi::DenseTensor>(paddle::framework::GradVarName("Y"));

    bool transpose_x = ctx.HasAttr("transpose_X")
                           ? ctx.Attr<bool>("transpose_X")
                           : ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.HasAttr("transpose_Y")
                           ? ctx.Attr<bool>("transpose_Y")
                           : ctx.Attr<bool>("trans_y");

    ReshapeXYOutToMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);

    phi::DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x.dims()) {
        dx->Resize(x.dims());
      }
    }

    phi::DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y.dims()) {
        dy->Resize(y.dims());
      }
    }

    if (transpose_x && transpose_y) {
      this->ExecuteMatMulGrad(
          ctx, dev_ctx, onednn_engine, &y, true, true, &dout, true, false, dx);
      this->ExecuteMatMulGrad(
          ctx, dev_ctx, onednn_engine, &dout, true, true, &x, true, false, dy);
    } else if (transpose_x) {
      this->ExecuteMatMulGrad(ctx,
                              dev_ctx,
                              onednn_engine,
                              &y,
                              false,
                              false,
                              &dout,
                              true,
                              false,
                              dx);
      this->ExecuteMatMulGrad(ctx,
                              dev_ctx,
                              onednn_engine,
                              &x,
                              false,
                              false,
                              &dout,
                              false,
                              true,
                              dy);
    } else if (transpose_y) {
      this->ExecuteMatMulGrad(ctx,
                              dev_ctx,
                              onednn_engine,
                              &dout,
                              false,
                              false,
                              &y,
                              false,
                              true,
                              dx);
      this->ExecuteMatMulGrad(
          ctx, dev_ctx, onednn_engine, &dout, true, true, &x, false, true, dy);
    } else {
      this->ExecuteMatMulGrad(ctx,
                              dev_ctx,
                              onednn_engine,
                              &dout,
                              false,
                              false,
                              &y,
                              true,
                              false,
                              dx);
      this->ExecuteMatMulGrad(
          ctx, dev_ctx, onednn_engine, &x, true, true, &dout, false, true, dy);
    }

    if (dx) {
      if (dx_dims != x.dims()) {
        dx->Resize(dx_dims);
        dx->set_mem_desc(x.mem_desc());
      }
    }
    if (dy) {
      if (dy_dims != y.dims()) {
        dy->Resize(dy_dims);
        dy->set_mem_desc(y.mem_desc());
      }
    }
  }

 private:
  void ExecuteMatMulGrad(const ExecutionContext &ctx,
                         const OneDNNContext &dev_ctx,
                         const dnnl::engine &engine,
                         phi::DenseTensor *x,
                         bool trans_x,
                         bool is_fold_init_dims_x,
                         phi::DenseTensor *y,
                         bool trans_y,
                         bool is_fold_init_dims_y,
                         phi::DenseTensor *out) const {
    // gradient is calculated in a different way when broadcasting is used
    bool need_combine = (x->dims().size() == 3 || y->dims().size() == 3) &&
                        out->dims().size() == 2;

    phi::DenseTensor x_combined, y_combined;
    if (!need_combine) {
      x_combined = *x;
      y_combined = *y;
    } else {
      x_combined = is_fold_init_dims_x ? FoldOuterDims(*x)
                                       : FoldFirstAndLastDims<T>(dev_ctx, x);
      y_combined = is_fold_init_dims_y ? FoldOuterDims(*y)
                                       : FoldFirstAndLastDims<T>(dev_ctx, y);
    }

    MatMulOneDNNHandler<T, T, T> handler(engine,
                                         ctx.GetPlace(),
                                         &x_combined,
                                         trans_x,
                                         &y_combined,
                                         trans_y,
                                         out,
                                         ctx.Attr<float>("alpha"));

    const auto src_memory_p = handler.AcquireSrcMemory(&x_combined);
    const auto weights_memory_p = handler.AcquireWeightsMemory(&y_combined);
    const auto dst_memory_p = handler.AcquireDstMemory(out);

    auto matmul_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto &astream = OneDNNContext::tls().get_stream();
    matmul_p->execute(astream, matmul_args);
    astream.wait();

    out->set_mem_desc(
        dst_memory_p->get_desc().reshape(vectorize<int64_t>(out->dims())));
  }
};

}  // anonymous namespace

REGISTER_OP_KERNEL(matmul,
                   MKLDNN,
                   ::phi::CPUPlace,
                   MatMulV1OneDNNKernel<float>,
                   MatMulV1OneDNNKernel<phi::dtype::bfloat16>,
                   MatMulV1OneDNNKernel<int8_t>,
                   MatMulV1OneDNNKernel<uint8_t>);

REGISTER_OP_KERNEL(matmul_grad,
                   MKLDNN,
                   ::phi::CPUPlace,
                   MatMulV1GradOneDNNKernel<float>,
                   MatMulV1GradOneDNNKernel<phi::dtype::bfloat16>);
