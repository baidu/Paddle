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

#include "paddle/phi/kernels/conv_grad_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/phi/kernels/gpudnn/conv_miopen_helper.h"
#else
#include "paddle/phi/kernels/gpudnn/conv_cudnn_v7.h"
#endif

#include "paddle/phi/backends/gpu/cuda/cudnn_workspace_helper.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/impl/conv_cudnn_impl.h"

#ifdef PADDLE_WITH_CUDNN_FRONTEND
// clang-format off
#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"
// clang-format on
#endif

namespace phi {

template <typename T, typename Context>
void ConvCudnnGradKernelImplV7(
    const DenseTensor* transformed_input,
    const DenseTensor* transformed_filter_channel,
    const DenseTensor* transformed_output_grad_channel,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    const Context& ctx,
    const std::vector<int>& strides,
    const std::vector<int>& padding_common,
    const std::vector<int>& dilations,
    phi::backends::gpu::DataLayout compute_format,
    phi::backends::gpu::DataLayout layout,
    bool use_addto,
    bool exhaustive_search,
    bool deterministic,
    int groups,
    DenseTensor* transformed_input_grad,
    DenseTensor* transformed_filter_grad_channel) {
  const T* input_data = transformed_input->data<T>();
  const T* output_grad_data = transformed_output_grad_channel->data<T>();
  const T* filter_data = transformed_filter_channel->data<T>();
  T* filter_grad_data = nullptr;
  T* input_grad_data = nullptr;
  T* transformed_input_grad_data = nullptr;

  auto handle = ctx.cudnn_handle();
  auto workspace_handle = ctx.cudnn_workspace_handle();

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;
  auto layout_tensor = phi::backends::gpu::GetCudnnTensorFormat(layout);

  ConvArgs args1{handle,
                 transformed_input_grad,
                 transformed_filter_channel,
                 transformed_output_grad_channel,
                 strides,
                 padding_common,
                 dilations,
                 dtype,
                 groups,
                 layout};
  ConvArgs args2{handle,
                 transformed_input,
                 transformed_filter_grad_channel,
                 transformed_output_grad_channel,
                 strides,
                 padding_common,
                 dilations,
                 dtype,
                 groups,
                 layout};

  int i_n, i_c, i_d, i_h, i_w;
  int o_n, o_c, o_d, o_h, o_w;
  if (compute_format == phi::backends::gpu::DataLayout::kNHWC) {
    GetNCDHW(transformed_input->dims(),
             phi::backends::gpu::DataLayout::kNHWC,
             &i_n,
             &i_c,
             &i_d,
             &i_h,
             &i_w);
    GetNCDHW(transformed_output_grad_channel->dims(),
             phi::backends::gpu::DataLayout::kNHWC,
             &o_n,
             &o_c,
             &o_d,
             &o_h,
             &o_w);
  } else {
    GetNCDHW(transformed_input->dims(),
             phi::backends::gpu::DataLayout::kNCHW,
             &i_n,
             &i_c,
             &i_d,
             &i_h,
             &i_w);
    GetNCDHW(transformed_output_grad_channel->dims(),
             phi::backends::gpu::DataLayout::kNCHW,
             &o_n,
             &o_c,
             &o_d,
             &o_h,
             &o_w);
  }

  int group_offset_in = i_c / groups * i_h * i_w * i_d;
  int group_offset_out = o_c / groups * o_h * o_w * o_d;
  int group_offset_filter = transformed_filter_channel->numel() / groups;

// ------------------- cudnn backward algorithm ---------------------
#ifdef PADDLE_WITH_HIP
  SearchResult<miopenConvBwdDataAlgorithm_t> bwd_result;
  SearchResult<miopenConvBwdWeightsAlgorithm_t> filter_result;
#else
  SearchResult<cudnnConvolutionBwdDataAlgo_t> bwd_result;
  SearchResult<cudnnConvolutionBwdFilterAlgo_t> filter_result;
#endif
  size_t workspace_size = 0;
  int iwo_groups = groups;
  int c_groups = 1;

#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_groups = 1;
  c_groups = groups;
  groups = 1;
#endif

  if (input_grad) {
    // ------------------- cudnn descriptors ---------------------
    input_grad_data = input_grad->data<T>();
    transformed_input_grad_data = transformed_input_grad->data<T>();

    args1.idesc.set(*transformed_input_grad, layout_tensor);
    args1.wdesc.set(*transformed_filter_channel, layout_tensor, iwo_groups);
    args1.odesc.set(*transformed_output_grad_channel, layout_tensor);
    args1.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    phi::AllowTF32Cudnn(),
                    c_groups);

#ifdef PADDLE_WITH_HIP
    using search1 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size = std::max(workspace_size, search1::GetWorkspaceSize(args1));
    bwd_result.algo = search1::Find<T>(
        args1, exhaustive_search, deterministic, workspace_size, ctx);
#else
    using search1 = SearchAlgorithm<ConvKind::kBackwardData>;
    bwd_result = search1::Find<T>(ctx, args1, exhaustive_search, deterministic);
    workspace_size = std::max(workspace_size, bwd_result.workspace_size);
#endif
  }

  if (filter_grad) {
    // ------------------- cudnn descriptors ---------------------
    filter_grad_data = transformed_filter_grad_channel->data<T>();

    args2.idesc.set(*transformed_input, layout_tensor);
    args2.wdesc.set(
        *transformed_filter_grad_channel, layout_tensor, iwo_groups);
    args2.odesc.set(*transformed_output_grad_channel, layout_tensor);
    args2.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    phi::AllowTF32Cudnn(),
                    c_groups);
#ifdef PADDLE_WITH_HIP
    using search2 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
    workspace_size = std::max(workspace_size, search2::GetWorkspaceSize(args2));
    filter_result.algo = search2::Find<T>(
        args2, exhaustive_search, deterministic, workspace_size, ctx);
#else
    using search2 = SearchAlgorithm<ConvKind::kBackwardFilter>;
    filter_result =
        search2::Find<T>(ctx, args2, exhaustive_search, deterministic);
    VLOG(3) << "filter algo: " << filter_result.algo << ", time "
            << filter_result.time;
    workspace_size = std::max(workspace_size, filter_result.workspace_size);
#endif
  }

  // ------------------- cudnn conv backward data ---------------------
  ScalingParamType<T> alpha = 1.0f;
#ifdef PADDLE_WITH_HIP
  // MIOPEN ONLY support beta to be 0.0f
  ScalingParamType<T> beta = 0.0f;
#else
  ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

#endif
  VLOG(4) << "Conv_grad: use_addto = " << use_addto;

  if (input_grad) {
// When beta is 0, it is unnecessary to reset input_grad.
// When beta is 1, the output cannot be reset since addt strategy used.
#ifdef PADDLE_WITH_HIP
    if (use_addto) {
      DenseTensor temp_tensor(transformed_input_grad->type());
      temp_tensor.Resize(transformed_input_grad->dims());
      T* temp_tensor_data = ctx.template Alloc<T>(&temp_tensor);
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::miopenConvolutionBackwardData(handle,
                                                            &alpha,
                                                            args1.odesc.desc(),
                                                            output_grad_data,
                                                            args1.wdesc.desc(),
                                                            filter_data,
                                                            args1.cdesc.desc(),
                                                            bwd_result.algo,
                                                            &beta,
                                                            args1.idesc.desc(),
                                                            temp_tensor_data,
                                                            cudnn_workspace_ptr,
                                                            workspace_size));
          },
          workspace_size);
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::miopenOpTensor(handle,
                                       miopenTensorOpAdd,
                                       &alpha,
                                       args1.idesc.desc(),
                                       transformed_input_grad_data,
                                       &alpha,
                                       args1.idesc.desc(),
                                       temp_tensor_data,
                                       &beta,
                                       args1.idesc.desc(),
                                       transformed_input_grad_data));
    } else {
      workspace_handle.RunFunc(
          [&](void* cudnn_workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::miopenConvolutionBackwardData(
                    handle,
                    &alpha,
                    args1.odesc.desc(),
                    output_grad_data,
                    args1.wdesc.desc(),
                    filter_data,
                    args1.cdesc.desc(),
                    bwd_result.algo,
                    &beta,
                    args1.idesc.desc(),
                    transformed_input_grad_data,
                    cudnn_workspace_ptr,
                    workspace_size));
          },
          workspace_size);
    }
#else
    ConvRunner<T, ConvKind::kBackwardData>::Apply(ctx,
                                                  args1,
                                                  bwd_result,
                                                  output_grad_data,
                                                  filter_data,
                                                  transformed_input_grad_data,
                                                  groups,
                                                  group_offset_in,
                                                  group_offset_filter,
                                                  group_offset_out,
                                                  workspace_size,
                                                  &workspace_handle,
                                                  use_addto);
#endif
  }

  // ------------------- cudnn conv backward filter ---------------------
  if (filter_grad) {
// Because beta is zero, it is unnecessary to reset filter_grad.
#ifdef PADDLE_WITH_HIP
    workspace_handle.RunFunc(
        [&](void* cudnn_workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::miopenConvolutionBackwardWeights(
                  handle,
                  &alpha,
                  args2.odesc.desc(),
                  output_grad_data,
                  args2.idesc.desc(),
                  input_data,
                  args2.cdesc.desc(),
                  filter_result.algo,
                  &beta,
                  args2.wdesc.desc(),
                  filter_grad_data,
                  cudnn_workspace_ptr,
                  workspace_size));
        },
        workspace_size);
#else
    ConvRunner<T, ConvKind::kBackwardFilter>::Apply(ctx,
                                                    args2,
                                                    filter_result,
                                                    output_grad_data,
                                                    input_data,
                                                    filter_grad_data,
                                                    groups,
                                                    group_offset_in,
                                                    group_offset_filter,
                                                    group_offset_out,
                                                    workspace_size,
                                                    &workspace_handle,
                                                    false);
#endif
  }
}

#ifdef PADDLE_WITH_CUDNN_FRONTEND
template <typename T, typename Context>
void ConvCudnnGradKernelImplV8(
    const DenseTensor* transformed_input,
    const DenseTensor* transformed_filter_channel,
    const DenseTensor* transformed_output_grad_channel,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    const Context& ctx,
    const std::vector<int>& strides,
    const std::vector<int>& padding_common,
    const std::vector<int>& dilations,
    phi::backends::gpu::DataLayout layout,
    bool use_addto,
    bool exhaustive_search,
    bool deterministic,
    int groups,
    DenseTensor* transformed_input_grad,
    DenseTensor* transformed_filter_grad_channel) {
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      common::errors::Unimplemented(
          "Group concolution using CUDNNv8 API is unsupported for now"));

  cudnnHandle_t handle = const_cast<cudnnHandle_t>(ctx.cudnn_handle());
  auto workspace_handle = ctx.cudnn_workspace_handle();

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;
  auto layout_format = phi::backends::gpu::GetCudnnTensorFormat(layout);

  if (input_grad) {
    CudnnConvBwdDataV8<T>(transformed_output_grad_channel,
                          transformed_filter_channel,
                          handle,
                          &workspace_handle,
                          strides,
                          padding_common,
                          dilations,
                          dtype,
                          layout_format,
                          use_addto,
                          exhaustive_search,
                          deterministic,
                          transformed_input_grad);
  }

  if (filter_grad) {
    CudnnConvBwdFilterV8<T>(transformed_input,
                            transformed_output_grad_channel,
                            handle,
                            &workspace_handle,
                            strides,
                            padding_common,
                            dilations,
                            dtype,
                            layout_format,
                            use_addto,
                            exhaustive_search,
                            deterministic,
                            transformed_filter_grad_channel);
  }
}
#endif

template <typename T, typename Context>
void ConvCudnnGradKernel(const Context& ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const DenseTensor& output_grad,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         const std::vector<int>& dilations_t,
                         int groups,
                         const std::string& data_format,
                         DenseTensor* input_grad,
                         DenseTensor* filter_grad) {
  if (input_grad) {
    ctx.template Alloc<T>(input_grad);
  }
  if (filter_grad) {
    ctx.template Alloc<T>(filter_grad);
  }

  bool has_use_addto = ctx.HasDnnAttr("use_addto");
  VLOG(4) << "GPUContext contains `use_addto`: " << has_use_addto;
  bool use_addto = has_use_addto
                       ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("use_addto"))
                       : false;

  std::vector<int> dilations = dilations_t;
  std::vector<int> strides = strides_t;
  std::vector<int> paddings = paddings_t;

  bool has_exhaustive_search = ctx.HasDnnAttr("exhaustive_search");
  VLOG(4) << "GPUContext contains `exhaustive_search`: "
          << has_exhaustive_search;
  bool exhaustive_search_attr =
      has_exhaustive_search
          ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("exhaustive_search"))
          : false;
  bool exhaustive_search =
      FLAGS_cudnn_exhaustive_search || exhaustive_search_attr;
  bool deterministic = FLAGS_cudnn_deterministic;
  auto exhaustive_deterministic = exhaustive_search && deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "Can't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
  // HIP MIOPEN ONLY SUPPORT NCHW format
  auto compute_format = phi::backends::gpu::DataLayout::kNCHW;
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
  const bool compute_in_nhwc =
      (dtype == CUDNN_DATA_HALF || dtype == CUDNN_DATA_BFLOAT16) &&
      IsVoltaOrLater(ctx);
#else
  const bool compute_in_nhwc = dtype == CUDNN_DATA_HALF && IsVoltaOrLater(ctx);
#endif
  auto compute_format = compute_in_nhwc && channel_last
                            ? phi::backends::gpu::DataLayout::kNHWC
                            : phi::backends::gpu::DataLayout::kNCHW;
#endif
  VLOG(3) << "Compute ConvGradOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == phi::backends::gpu::DataLayout::kNHWC ? "NHWC"
                                                                      : "NCHW");

  // transform Tensor
  DenseTensor transformed_input_channel(input.type());
  DenseTensor transformed_output_grad_channel(output_grad.type());
  DenseTensor transformed_input_grad_channel(input.type());
  DenseTensor transformed_filter_channel(filter.type());
  DenseTensor transformed_filter_grad_channel(filter.type());

  if (channel_last && compute_format == phi::backends::gpu::DataLayout::kNCHW) {
    VLOG(3) << "Transform input, output_grad, input_grad and tensor from "
               "NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);
    TransToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);

    ResizeToChannelFirst<Context, T>(
        ctx, &output_grad, &transformed_output_grad_channel);
    TransToChannelFirst<Context, T>(
        ctx, &output_grad, &transformed_output_grad_channel);

    if (input_grad) {
      ResizeToChannelFirst<Context, T>(
          ctx, input_grad, &transformed_input_grad_channel);
      // NOTE(zhiqiu): If inplace_addto strategy is enabled, we need to copy
      // the data of input_grad to transformed_input_grad_channel.
      if (use_addto) {
        TransToChannelFirst<Context, T>(
            ctx, input_grad, &transformed_input_grad_channel);
      }
    }
  } else {
    transformed_input_channel.ShareDataWith(input);
    transformed_output_grad_channel.ShareDataWith(output_grad);
    if (input_grad) {
      transformed_input_grad_channel.ShareDataWith(*input_grad);
    }
  }

  if (compute_format == phi::backends::gpu::DataLayout::kNHWC) {
    VLOG(3) << "Transform filter and filter_grad tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);

    if (filter_grad) {
      ResizeToChannelLast<Context, T>(
          ctx, filter_grad, &transformed_filter_grad_channel);
    }
  } else {
    transformed_filter_channel.ShareDataWith(filter);
    if (filter_grad) {
      transformed_filter_grad_channel.ShareDataWith(*filter_grad);
    }
  }

  //  update paddings
  auto in_dims = transformed_input_channel.dims();
  auto filter_dims = transformed_filter_channel.dims();
  DDim in_data_dims;
  DDim filter_data_dims;
  if (compute_format == phi::backends::gpu::DataLayout::kNCHW) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
    filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    filter_data_dims = slice_ddim(filter_dims, 1, filter_dims.size() - 1);
  }
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  // cuDNN only supports padding the same amount on every dimension.
  // So we create a new padded input tensor.
  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);
  Tensor transformed_input(input.type());
  Tensor transformed_input_grad(input.type());
  std::vector<int> padding_common(data_dim, 0);
  std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);

  if (!is_sys_pad) {
    // get pad
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];
    if (compute_format == phi::backends::gpu::DataLayout::kNCHW) {
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];
    } else {
      new_input_shape_vec[data_dim + 1] =
          transformed_input_channel.dims()[data_dim + 1];
    }

    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
      padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
      if (compute_format == phi::backends::gpu::DataLayout::kNCHW) {
        new_input_shape_vec[i + 2] =
            transformed_input_channel.dims()[i + 2] + padding_diff[i];
      } else {
        new_input_shape_vec[i + 1] =
            transformed_input_channel.dims()[i + 1] + padding_diff[i];
      }
      if (compute_format == phi::backends::gpu::DataLayout::kNCHW) {
        input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
      } else {
        input_pad[2 * i + 2] = paddings[2 * i] - padding_common[i];
        input_pad[2 * i + 2 + 1] = paddings[2 * i + 1] - padding_common[i];
      }
    }
    DDim new_input_shape(common::make_ddim(new_input_shape_vec));
    transformed_input.Resize(new_input_shape);
    ctx.template Alloc<T>(&transformed_input);

    transformed_input_grad.Resize(new_input_shape);

    if (input_grad) {
      ctx.template Alloc<T>(&transformed_input_grad);
    }
    // pad for input
    const int rank = transformed_input_channel.dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(ctx,
                                          input_pad,
                                          transformed_input_channel,
                                          pad_value,
                                          &transformed_input);
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(ctx,
                                          input_pad,
                                          transformed_input_channel,
                                          pad_value,
                                          &transformed_input);
      } break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
    }
  } else {
    transformed_input.ShareDataWith(transformed_input_channel);
    if (input_grad) {
      transformed_input_grad.ShareDataWith(transformed_input_grad_channel);
    }
    if (paddings.size() == data_dim) {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[i];
      }
    } else {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[2 * i];
      }
    }
  }
  phi::backends::gpu::DataLayout layout =
      compute_format == phi::backends::gpu::DataLayout::kNHWC
          ? phi::backends::gpu::DataLayout::kNHWC
          : phi::backends::gpu::DataLayout::kNCHW;
  if (transformed_input.dims().size() == 5) {
    layout = compute_format == phi::backends::gpu::DataLayout::kNHWC
                 ? phi::backends::gpu::DataLayout::kNDHWC
                 : phi::backends::gpu::DataLayout::kNCDHW;
  }

#ifdef PADDLE_WITH_CUDNN_FRONTEND
  if (dynload::IsCudnnFrontendEnabled() && (groups == 1))
    ConvCudnnGradKernelImplV8<T>(&transformed_input,
                                 &transformed_filter_channel,
                                 &transformed_output_grad_channel,
                                 input_grad,
                                 filter_grad,
                                 ctx,
                                 strides,
                                 padding_common,
                                 dilations,
                                 layout,
                                 use_addto,
                                 exhaustive_search,
                                 deterministic,
                                 groups,
                                 &transformed_input_grad,
                                 &transformed_filter_grad_channel);
  else
    ConvCudnnGradKernelImplV7<T>(&transformed_input,
                                 &transformed_filter_channel,
                                 &transformed_output_grad_channel,
                                 input_grad,
                                 filter_grad,
                                 ctx,
                                 strides,
                                 padding_common,
                                 dilations,
                                 compute_format,
                                 layout,
                                 use_addto,
                                 exhaustive_search,
                                 deterministic,
                                 groups,
                                 &transformed_input_grad,
                                 &transformed_filter_grad_channel);
#else
  ConvCudnnGradKernelImplV7<T>(&transformed_input,
                               &transformed_filter_channel,
                               &transformed_output_grad_channel,
                               input_grad,
                               filter_grad,
                               ctx,
                               strides,
                               padding_common,
                               dilations,
                               compute_format,
                               layout,
                               use_addto,
                               exhaustive_search,
                               deterministic,
                               groups,
                               &transformed_input_grad,
                               &transformed_filter_grad_channel);
#endif

  if (input_grad) {
    if (!is_sys_pad) {
      std::vector<int> starts(transformed_input_channel.dims().size(), 0);
      std::vector<int> axes(transformed_input_channel.dims().size(), 0);

      for (size_t i = 0; i < transformed_input_channel.dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }

      ctx.template Alloc<T>(&transformed_input_grad_channel);
      if (transformed_input_channel.dims().size() == 4) {
        RemovePaddingSlice<Context, T, 4>(ctx,
                                          &transformed_input_grad,
                                          &transformed_input_grad_channel,
                                          starts,
                                          axes);
      } else {
        RemovePaddingSlice<Context, T, 5>(ctx,
                                          &transformed_input_grad,
                                          &transformed_input_grad_channel,
                                          starts,
                                          axes);
      }
    }

    if (channel_last &&
        compute_format == phi::backends::gpu::DataLayout::kNCHW) {
      TransToChannelLast<Context, T>(
          ctx, &transformed_input_grad_channel, input_grad);
    }
  }

  if (filter_grad) {
    if (compute_format == phi::backends::gpu::DataLayout::kNHWC) {
      TransToChannelFirst<Context, T>(
          ctx, &transformed_filter_grad_channel, filter_grad);
    }
  }
}

template <typename T, typename Context>
void Conv3DCudnnGradKernel(const Context& dev_ctx,
                           const DenseTensor& input,
                           const DenseTensor& filter,
                           const DenseTensor& out_grad,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* input_grad,
                           DenseTensor* filter_grad) {
  ConvCudnnGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings,
                         padding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         input_grad,
                         filter_grad);
}

template <typename T, typename Context>
void DepthwiseConvCudnnGradKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const DenseTensor& filter,
                                  const DenseTensor& out_grad,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations,
                                  const std::string& data_format,
                                  bool use_addto,
                                  int workspace_size_MB,
                                  bool exhaustive_search,
                                  bool fuse_relu,
                                  DenseTensor* input_grad,
                                  DenseTensor* filter_grad) {
  ConvCudnnGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings,
                         padding_algorithm,
                         dilations,
                         groups,
                         data_format,
                         input_grad,
                         filter_grad);
}

template <typename T, typename Context>
void ConvCudnnGradGradKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    const std::vector<int>& dilations_t,
    int groups,
    const std::string& data_format,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    DenseTensor* out_grad_grad) {
  auto X = &input;
  auto W = &filter;
  auto dO = &out_grad;
  auto ddX = input_grad_grad.get_ptr();
  auto ddW = filter_grad_grad.get_ptr();

  auto ddO = out_grad_grad;
  auto dW = filter_grad;
  auto dX = input_grad;
  if (ddO) {
    ctx.template Alloc<T>(ddO);
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(ctx, ddO, static_cast<T>(0));
  }
  if (dW) {
    ctx.template Alloc<T>(dW);
  }
  if (dX) {
    ctx.template Alloc<T>(dX);
  }

  // const T* x = X->data<T>();
  const T* dy = dO->data<T>();
  const T* w = W->data<T>();

  const T* ddx = nullptr;
  const T* ddw = nullptr;
  T *dw, *dx, *ddy;
  dw = dx = ddy = nullptr;
  T* transformed_dx = nullptr;
  std::vector<int> dilations = dilations_t;

  bool has_exhaustive_search = ctx.HasDnnAttr("exhaustive_search");
  VLOG(4) << "GPUContext contains `exhaustive_search`: "
          << has_exhaustive_search;
  bool exhaustive_search_attr =
      has_exhaustive_search
          ? PADDLE_GET_CONST(bool, ctx.GetDnnAttr("exhaustive_search"))
          : false;
  bool exhaustive_search =
      FLAGS_cudnn_exhaustive_search || exhaustive_search_attr;
  bool deterministic = FLAGS_cudnn_deterministic;
  auto exhaustive_deterministic = exhaustive_search && deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "Can't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  std::vector<int> paddings = paddings_t;

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  // transform Tensors to channel first-----------
  DenseTensor transformed_X_channel(X->type());
  DenseTensor transformed_dO_channel(dO->type());
  DenseTensor transformed_ddX_channel(X->type());

  DenseTensor transformed_ddO_channel(dO->type());
  DenseTensor transformed_dX_channel(X->type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(ctx, X, &transformed_X_channel);
    TransToChannelFirst<Context, T>(ctx, X, &transformed_X_channel);

    ResizeToChannelFirst<Context, T>(ctx, dO, &transformed_dO_channel);
    TransToChannelFirst<Context, T>(ctx, dO, &transformed_dO_channel);

    if (ddX) {
      ResizeToChannelFirst<Context, T>(ctx, ddX, &transformed_ddX_channel);
      TransToChannelFirst<Context, T>(ctx, ddX, &transformed_ddX_channel);
    }

    if (ddO) {
      ResizeToChannelFirst<Context, T>(ctx, ddO, &transformed_ddO_channel);
    }
    if (dX) {
      ResizeToChannelFirst<Context, T>(ctx, dX, &transformed_dX_channel);
      ctx.template Alloc<T>(&transformed_dX_channel);
    }

  } else {
    transformed_X_channel = *X;
    transformed_dO_channel = *dO;
    if (ddX) {
      transformed_ddX_channel = *ddX;
    }
    if (ddO) {
      transformed_ddO_channel.ShareDataWith(*ddO);
    }
    if (dX) {
      transformed_dX_channel.ShareDataWith(*dX);
    }
  }

  auto in_dims = transformed_X_channel.dims();
  auto filter_dims = W->dims();
  DDim in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);
  DenseTensor transformed_X(X->type());
  DenseTensor transformed_ddX(X->type());

  DenseTensor transformed_dX(X->type());

  std::vector<int> padding_common(data_dim, 0);
  std::vector<int> input_pad(X->dims().size() * 2, 0);

  if (!is_sys_pad) {
    // get pad
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_X_channel.dims()[0];
    new_input_shape_vec[1] = transformed_X_channel.dims()[1];

    for (size_t i = 0; i < data_dim; ++i) {
      padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
      padding_common[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
      new_input_shape_vec[i + 2] =
          transformed_X_channel.dims()[i + 2] + padding_diff[i];
      input_pad[2 * i + 4] = paddings[2 * i] - padding_common[i];
      input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - padding_common[i];
    }
    DDim new_input_shape(common::make_ddim(new_input_shape_vec));
    transformed_X.Resize(new_input_shape);
    transformed_ddX.Resize(new_input_shape);
    transformed_dX.Resize(new_input_shape);

    ctx.template Alloc<T>(&transformed_X);

    if (ddX) {
      ctx.template Alloc<T>(&transformed_ddX);
    }
    if (dX) {
      ctx.template Alloc<T>(&transformed_dX);
    }

    // pad for input
    const int rank = X->dims().size();
    T pad_value(0.0);
    switch (rank) {
      case 4: {
        funcs::PadFunction<Context, T, 4>(
            ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
        if (ddX) {
          funcs::PadFunction<Context, T, 4>(ctx,
                                            input_pad,
                                            transformed_ddX_channel,
                                            pad_value,
                                            &transformed_ddX);
        }
      } break;
      case 5: {
        funcs::PadFunction<Context, T, 5>(
            ctx, input_pad, transformed_X_channel, pad_value, &transformed_X);
        if (ddX) {
          funcs::PadFunction<Context, T, 5>(ctx,
                                            input_pad,
                                            transformed_ddX_channel,
                                            pad_value,
                                            &transformed_ddX);
        }
      } break;
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "ConvOp only support tensors with 4 or 5 dimensions."));
    }

  } else {
    transformed_X.ShareDataWith(transformed_X_channel);
    if (ddX) {
      transformed_ddX.ShareDataWith(transformed_ddX_channel);
    }
    if (dX) {
      transformed_dX.ShareDataWith(transformed_dX_channel);
    }

    if (paddings.size() == data_dim) {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[i];
      }
    } else {
      for (size_t i = 0; i < data_dim; ++i) {
        padding_common[i] = paddings[2 * i];
      }
    }
  }

  const T* x = transformed_X.data<T>();

  int iwo_group = groups;
  int c_group = 1;
#if defined(PADDLE_WITH_HIP) || CUDNN_VERSION_MIN(7, 0, 1)
  iwo_group = 1;
  c_group = groups;
  groups = 1;
#endif
  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

  auto handle = ctx.cudnn_handle();
  auto layout = phi::backends::gpu::GetCudnnTensorFormat(
      phi::backends::gpu::DataLayout::kNCHW);

  ConvArgs args1{handle,
                 &transformed_ddX,
                 W,
                 &transformed_ddO_channel,
                 strides,
                 padding_common,
                 dilations,
                 dtype,
                 groups,
                 phi::backends::gpu::DataLayout::kNCHW};
  ConvArgs args2{handle,
                 &transformed_X,
                 ddW,
                 &transformed_ddO_channel,
                 strides,
                 padding_common,
                 dilations,
                 dtype,
                 groups,
                 phi::backends::gpu::DataLayout::kNCHW};
  ConvArgs args3{handle,
                 &transformed_ddX,
                 dW,
                 &transformed_dO_channel,
                 strides,
                 padding_common,
                 dilations,
                 dtype,
                 groups,
                 phi::backends::gpu::DataLayout::kNCHW};
  ConvArgs args4{handle,
                 &transformed_dX,
                 ddW,
                 &transformed_dO_channel,
                 strides,
                 padding_common,
                 dilations,
                 dtype,
                 groups,
                 phi::backends::gpu::DataLayout::kNCHW};

#ifdef PADDLE_WITH_HIP
  SearchResult<miopenConvFwdAlgorithm_t> fwd_result1;
  SearchResult<miopenConvFwdAlgorithm_t> fwd_result2;
  SearchResult<miopenConvBwdDataAlgorithm_t> data_result;
  SearchResult<miopenConvBwdWeightsAlgorithm_t> filter_result;
#else
  SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result1;
  SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result2;
  SearchResult<cudnnConvolutionBwdDataAlgo_t> data_result;
  SearchResult<cudnnConvolutionBwdFilterAlgo_t> filter_result;
#endif

  // ddo = conv(ddI, W) + conv(I, ddW)
  size_t workspace_size = 0;

  T* transformed_ddy_channel = nullptr;
  if (ddO) {
    ddy = ddO->data<T>();
    transformed_ddy_channel = transformed_ddO_channel.data<T>();
    if (ddX) {
      args1.idesc.set(transformed_ddX, iwo_group);
      args1.wdesc.set(*W, layout, iwo_group);
      args1.odesc.set(transformed_ddO_channel, iwo_group);
      args1.cdesc.set(dtype,
                      padding_common,
                      strides,
                      dilations,
                      phi::AllowTF32Cudnn(),
                      c_group);

#ifdef PADDLE_WITH_HIP
      using search1 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
      workspace_size = search1::GetWorkspaceSize(args1);
      fwd_result1.algo = search1::Find<T>(
          args1, exhaustive_search, false, workspace_size, ctx);
#else
      using search1 = SearchAlgorithm<ConvKind::kForward>;
      fwd_result1 = search1::Find<T>(ctx, args1, exhaustive_search, false);
      workspace_size = search1::GetWorkspaceSize(args1, fwd_result1.algo);
#endif
    }

    if (ddW) {
      ddw = ddW->data<T>();
      args2.idesc.set(transformed_X, iwo_group);
      args2.wdesc.set(*ddW, layout, iwo_group);
      args2.odesc.set(transformed_ddO_channel, iwo_group);
      args2.cdesc.set(dtype,
                      padding_common,
                      strides,
                      dilations,
                      phi::AllowTF32Cudnn(),
                      c_group);

#ifdef PADDLE_WITH_HIP
      using search2 = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
      workspace_size =
          std::max(workspace_size, search2::GetWorkspaceSize(args2));
      fwd_result2.algo = search2::Find<T>(
          args2, exhaustive_search, false, workspace_size, ctx);
#else
      using search2 = SearchAlgorithm<ConvKind::kForward>;
      fwd_result2 = search2::Find<T>(ctx, args2, exhaustive_search, false);
      workspace_size = std::max(
          workspace_size, search2::GetWorkspaceSize(args2, fwd_result2.algo));
#endif
    }
  }

  if (dW && ddX) {
    dw = dW->data<T>();
    args3.idesc.set(transformed_ddX, iwo_group);
    args3.wdesc.set(*dW, layout, iwo_group);
    args3.odesc.set(transformed_dO_channel, iwo_group);
    args3.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    phi::AllowTF32Cudnn(),
                    c_group);

#ifdef PADDLE_WITH_HIP
    using search3 = SearchAlgorithm<miopenConvBwdWeightsAlgorithm_t>;
    workspace_size = std::max(workspace_size, search3::GetWorkspaceSize(args3));
    filter_result.algo = search3::Find<T>(
        args3, exhaustive_search, deterministic, workspace_size, ctx);
#else
    using search3 = SearchAlgorithm<ConvKind::kBackwardFilter>;
    filter_result =
        search3::Find<T>(ctx, args3, exhaustive_search, deterministic);
    workspace_size = std::max(
        workspace_size, search3::GetWorkspaceSize(args3, filter_result.algo));
#endif
  }

  if (ddW && dX) {
    transformed_dx = transformed_dX.data<T>();

    args4.idesc.set(transformed_dX, iwo_group);
    args4.wdesc.set(*ddW, layout, iwo_group);
    args4.odesc.set(transformed_dO_channel, iwo_group);
    args4.cdesc.set(dtype,
                    padding_common,
                    strides,
                    dilations,
                    phi::AllowTF32Cudnn(),
                    c_group);

#ifdef PADDLE_WITH_HIP
    using search4 = SearchAlgorithm<miopenConvBwdDataAlgorithm_t>;
    workspace_size = std::max(workspace_size, search4::GetWorkspaceSize(args4));
    data_result.algo = search4::Find<T>(
        args4, exhaustive_search, deterministic, workspace_size, ctx);
#else
    using search4 = SearchAlgorithm<ConvKind::kBackwardData>;
    data_result =
        search4::Find<T>(ctx, args4, exhaustive_search, deterministic);
    workspace_size = std::max(
        workspace_size, search4::GetWorkspaceSize(args4, data_result.algo));
#endif
  }

  int i_n, i_c, i_d, i_h, i_w;
  GetNCDHW(
      transformed_X.dims(), DataLayout::kNCHW, &i_n, &i_c, &i_d, &i_h, &i_w);

  int o_n, o_c, o_d, o_h, o_w;
  GetNCDHW(transformed_dO_channel.dims(),
           DataLayout::kNCHW,
           &o_n,
           &o_c,
           &o_d,
           &o_h,
           &o_w);

  int group_offset_in = i_c / groups * i_h * i_w * i_d;
  int group_offset_out = o_c / groups * o_h * o_w * o_d;
  int group_offset_filter = W->numel() / groups;

  ScalingParamType<T> alpha = 1.0f;
  ScalingParamType<T> beta = 0.0f;

  // NOTE(zhiqiu): inplace addto is not supported in double grad yet.
  // ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f :
  // 0.0f;
  // VLOG(4) << "Conv_grad_grad: use_addto = " << ctx.Attr<bool>("use_addto");
  auto workspace_handle = ctx.cudnn_workspace_handle();

  if (ddO) {
    if (ddX) {
      ddx = transformed_ddX.data<T>();
#ifdef PADDLE_WITH_HIP
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::miopenConvolutionForward(handle,
                                                       &alpha,
                                                       args1.idesc.desc(),
                                                       ddx,
                                                       args1.wdesc.desc(),
                                                       w,
                                                       args1.cdesc.desc(),
                                                       fwd_result1.algo,
                                                       &beta,
                                                       args1.odesc.desc(),
                                                       transformed_ddy_channel,
                                                       workspace_ptr,
                                                       workspace_size));
          },
          workspace_size);
#else
      ConvRunner<T, ConvKind::kForward>::Apply(ctx,
                                               args1,
                                               fwd_result1,
                                               ddx,
                                               w,
                                               transformed_ddy_channel,
                                               groups,
                                               group_offset_in,
                                               group_offset_filter,
                                               group_offset_out,
                                               workspace_size,
                                               &workspace_handle,
                                               false);
#endif
    }
    if (ddW) {
#ifdef PADDLE_WITH_HIP
      // MIOPEN ONLY support beta to be 0.0f
      workspace_handle.RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::miopenConvolutionForward(handle,
                                                       &alpha,
                                                       args2.idesc.desc(),
                                                       x,
                                                       args2.wdesc.desc(),
                                                       ddw,
                                                       args2.cdesc.desc(),
                                                       fwd_result2.algo,
                                                       &beta,
                                                       args2.odesc.desc(),
                                                       transformed_ddy_channel,
                                                       workspace_ptr,
                                                       workspace_size));
          },
          workspace_size);
#else
      ConvRunner<T, ConvKind::kForward>::Apply(ctx,
                                               args2,
                                               fwd_result2,
                                               x,
                                               ddw,
                                               transformed_ddy_channel,
                                               groups,
                                               group_offset_in,
                                               group_offset_filter,
                                               group_offset_out,
                                               workspace_size,
                                               &workspace_handle,
                                               true);
#endif
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(ctx, &transformed_ddO_channel, ddO);
    }
  }
  T* transformed_dy_channel = transformed_dO_channel.data<T>();
  if (dW && ddX) {
    ddx = transformed_ddX.data<T>();
#ifdef PADDLE_WITH_HIP
    workspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::miopenConvolutionBackwardWeights(
                  handle,
                  &alpha,
                  args3.odesc.desc(),
                  transformed_dy_channel,
                  args3.idesc.desc(),
                  ddx,
                  args3.cdesc.desc(),
                  filter_result.algo,
                  &beta,
                  args3.wdesc.desc(),
                  dw,
                  workspace_ptr,
                  workspace_size));
        },
        workspace_size);
#else
    ConvRunner<T, ConvKind::kBackwardFilter>::Apply(ctx,
                                                    args3,
                                                    filter_result,
                                                    transformed_dy_channel,
                                                    ddx,
                                                    dw,
                                                    groups,
                                                    group_offset_in,
                                                    group_offset_filter,
                                                    group_offset_out,
                                                    workspace_size,
                                                    &workspace_handle,
                                                    false);
#endif
  }

  if (dX && ddW) {
    ddw = ddW->data<T>();
#ifdef PADDLE_WITH_HIP
    workspace_handle.RunFunc(
        [&](void* workspace_ptr) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              phi::dynload::miopenConvolutionBackwardData(
                  handle,
                  &alpha,
                  args4.odesc.desc(),
                  transformed_dy_channel,
                  args4.wdesc.desc(),
                  ddw,
                  args4.cdesc.desc(),
                  data_result.algo,
                  &beta,
                  args4.idesc.desc(),
                  transformed_dx,
                  workspace_ptr,
                  workspace_size));
        },
        workspace_size);
#else
    ConvRunner<T, ConvKind::kBackwardData>::Apply(ctx,
                                                  args4,
                                                  data_result,
                                                  transformed_dy_channel,
                                                  ddw,
                                                  transformed_dx,
                                                  groups,
                                                  group_offset_in,
                                                  group_offset_filter,
                                                  group_offset_out,
                                                  workspace_size,
                                                  &workspace_handle,
                                                  false);
#endif

    if (!is_sys_pad) {
      // reverse padded input
      std::vector<int> starts(X->dims().size(), 0);
      std::vector<int> axes(X->dims().size(), 0);

      for (size_t i = 0; i < X->dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }
      if (X->dims().size() == 4) {
        RemovePaddingSlice<Context, T, 4>(
            ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
      } else {
        RemovePaddingSlice<Context, T, 5>(
            ctx, &transformed_dX, &transformed_dX_channel, starts, axes);
      }
    }
    if (channel_last) {
      TransToChannelLast<Context, T>(ctx, &transformed_dX_channel, dX);
    }
  }
}

template <typename T, typename Context>
void DepthwiseConvDoubleGradGPUDNNKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations_t,
    const std::string& data_format,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    DenseTensor* out_grad_grad) {
  ConvCudnnGradGradKernel<T>(ctx,
                             input,
                             filter,
                             out_grad,
                             input_grad_grad,
                             filter_grad_grad,
                             strides,
                             paddings_t,
                             padding_algorithm,
                             dilations_t,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad,
                             out_grad_grad);
}

template <typename T, typename Context>
void Conv3DCudnnDoubleGradKernel(
    const Context& ctx,
    const DenseTensor& input,
    const DenseTensor& filter,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& input_grad_grad,
    const paddle::optional<DenseTensor>& filter_grad_grad,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& padding_algorithm,
    int groups,
    const std::vector<int>& dilations_t,
    const std::string& data_format,
    DenseTensor* input_grad,
    DenseTensor* filter_grad,
    DenseTensor* out_grad_grad) {
  ConvCudnnGradGradKernel<T>(ctx,
                             input,
                             filter,
                             out_grad,
                             input_grad_grad,
                             filter_grad_grad,
                             strides,
                             paddings_t,
                             padding_algorithm,
                             dilations_t,
                             groups,
                             data_format,
                             input_grad,
                             filter_grad,
                             out_grad_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(conv2d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnDoubleGradKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvDoubleGradGPUDNNKernel,
                   float,
                   phi::dtype::float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(conv2d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvDoubleGradGPUDNNKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(conv2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv2d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnGradGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConvDoubleGradGPUDNNKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif

#endif
