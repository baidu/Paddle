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

#include "paddle/phi/kernels/conv_kernel.h"

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
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"
// clang-format on
#endif

#include "paddle/common/flags.h"

COMMON_DECLARE_bool(manually_trans_conv_filter);

namespace phi {

template <typename T, typename Context>
void ConvCudnnKernelImplV7(const DenseTensor* transformed_input,
                           const DenseTensor* transformed_filter_channel,
                           const Context& ctx,
                           const std::vector<int>& strides,
                           const std::vector<int>& padding_common,
                           const std::vector<int>& dilations,
                           phi::backends::gpu::DataLayout compute_format,
                           phi::backends::gpu::DataLayout layout,
                           bool exhaustive_search,
                           bool deterministic,
                           int groups,
                           DenseTensor* transformed_output) {
  const T* input_data = transformed_input->data<T>();
  const T* filter_data = transformed_filter_channel->data<T>();
  T* output_data = transformed_output->data<T>();

  auto handle = ctx.cudnn_handle();
  auto workspace_handle = ctx.cudnn_workspace_handle();

  auto layout_format = phi::backends::gpu::GetCudnnTensorFormat(layout);
  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

  // ------------------- cudnn descriptors ---------------------
  ConvArgs args{handle,
                transformed_input,
                transformed_filter_channel,
                transformed_output,
                strides,
                padding_common,
                dilations,
                dtype,
                groups,
                compute_format};

#ifdef PADDLE_WITH_HIP
  // MIOPEN need to set groups in cdesc in miopen_desc.h
  args.cdesc.set(
      dtype, padding_common, strides, dilations, phi::AllowTF32Cudnn(), groups);
#else
  args.cdesc.set(
      dtype, padding_common, strides, dilations, phi::AllowTF32Cudnn());
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION_MIN(7, 0, 1)
  // cudnn 7 can support groups, no need to do it manually
  // FIXME(typhoonzero): find a better way to disable groups
  // rather than setting it to 1.
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnSetConvolutionGroupCount(args.cdesc.desc(), groups));
  groups = 1;
#endif
#ifdef PADDLE_WITH_HIP
  // MIOPEN do not set groups in wdesc after set groups in cdesc
  groups = 1;
#endif
  args.idesc.set(*transformed_input, layout_format);
  args.wdesc.set(*transformed_filter_channel, layout_format, groups);
  args.odesc.set(*transformed_output, layout_format);
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
    GetNCDHW(transformed_output->dims(),
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
    GetNCDHW(transformed_output->dims(),
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
  // ------------------- cudnn conv workspace ---------------------
  size_t workspace_size = 0;  // final workspace to allocate.
// ------------------- cudnn conv algorithm ---------------------
#ifdef PADDLE_WITH_HIP
  SearchResult<miopenConvFwdAlgorithm_t> fwd_result;
  using search = SearchAlgorithm<miopenConvFwdAlgorithm_t>;
  workspace_size = search::GetWorkspaceSize(args);
  fwd_result.algo = search::Find<T>(
      args, exhaustive_search, deterministic, workspace_size, ctx);
#else
  SearchResult<cudnnConvolutionFwdAlgo_t> fwd_result;
  using search = SearchAlgorithm<ConvKind::kForward>;
  fwd_result = search::Find<T>(ctx, args, exhaustive_search, deterministic);
  workspace_size = fwd_result.workspace_size;
#endif

#if defined(PADDLE_WITH_CUDA) && CUDNN_VERSION_MIN(7, 0, 1)
  // when groups > 1, SearchAlgorithm find algo is CUDNN_CONVOLUTION_\
    // FWD_ALGO_WINOGRAD_NONFUSED, but this kind of algorithm is unstable
  // in forward computation, so change the algorithm to CUDNN_CONVOLUTION_\
    // FWD_ALGO_IMPLICIT_GEMM manually.
  if (groups > 1) {
    fwd_result.algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
  }
#endif

  // ------------------- cudnn conv forward ---------------------
  ScalingParamType<T> alpha = 1.0f;
  ScalingParamType<T> beta = 0.0f;

  // NOTE(zhiqiu): inplace addto is not supported in double grad yet.
  // ScalingParamType<T> beta = ctx.Attr<bool>("use_addto") ? 1.0f : 0.0f;
  // VLOG(4) << "Conv: use_addto = " << ctx.Attr<bool>("use_addto");

#ifdef PADDLE_WITH_HIP
  workspace_handle.RunFunc(
      [&](void* workspace_ptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::miopenConvolutionForward(handle,
                                                   &alpha,
                                                   args.idesc.desc(),
                                                   input_data,
                                                   args.wdesc.desc(),
                                                   filter_data,
                                                   args.cdesc.desc(),
                                                   fwd_result.algo,
                                                   &beta,
                                                   args.odesc.desc(),
                                                   output_data,
                                                   workspace_ptr,
                                                   workspace_size));
      },
      workspace_size);
#else
  ConvRunner<T, ConvKind::kForward>::Apply(ctx,
                                           args,
                                           fwd_result,
                                           input_data,
                                           filter_data,
                                           output_data,
                                           groups,
                                           group_offset_in,
                                           group_offset_filter,
                                           group_offset_out,
                                           workspace_size,
                                           &workspace_handle,
                                           false);
#endif
}

#ifdef PADDLE_WITH_CUDNN_FRONTEND
template <typename T, typename Context>
void ConvCudnnKernelImplV8(const DenseTensor* input_tensor,
                           const DenseTensor* filter_channel_tensor,
                           const Context& ctx,
                           const std::vector<int>& strides,
                           const std::vector<int>& padding_common,
                           const std::vector<int>& dilations,
                           phi::backends::gpu::DataLayout layout,
                           bool exhaustive_search,
                           bool deterministic,
                           int groups,
                           DenseTensor* output_tensor) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kConvForwardV8);

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      common::errors::Unimplemented(
          "Group concolution using CUDNNv8 API unsupported for now"));

  T* input_data = const_cast<T*>(input_tensor->data<T>());
  T* filter_data = const_cast<T*>(filter_channel_tensor->data<T>());
  T* output_data = output_tensor->data<T>();
  cudnnHandle_t handle = const_cast<cudnnHandle_t>(ctx.cudnn_handle());
  auto workspace_handle = ctx.cudnn_workspace_handle();

  auto layout_format = phi::backends::gpu::GetCudnnTensorFormat(layout);
  auto dtype = phi::backends::gpu::CudnnDataType<T>::type;

  float alpha = 1.0f;
  float beta = 0.0f;

  using helper = CudnnFrontendConvHelper;
  auto op_graph = helper::BuildConvOperationGraph<
      CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR>(
      input_tensor,
      output_tensor,
      filter_channel_tensor,
      layout_format,
      strides,
      padding_common,
      dilations,
      dtype,
      handle,
      alpha,
      beta);

  if (plan_cache.FindPlan(op_graph, handle)) {
    const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
    int64_t workspace_size = 0;
    plan_cache.GetPlanAndWorkspaceSize(
        op_graph, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        input_data,
                        output_data,
                        filter_data,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    return;
  }

  auto plans = helper::FindExecutionPlans(&op_graph,
                                          exhaustive_search,
                                          deterministic,
                                          input_data,
                                          output_data,
                                          filter_data,
                                          handle,
                                          &workspace_handle);

  helper::ExecutePlansAndCache(handle,
                               &workspace_handle,
                               input_data,
                               output_data,
                               filter_data,
                               &plans,
                               exhaustive_search,
                               op_graph,
                               &plan_cache);
}
#endif

template <typename T, typename Context>
void ConvCudnnKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& filter,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings_t,
                     const std::string& padding_algorithm,
                     const std::vector<int>& dilations_t,
                     int groups,
                     const std::string& data_format,
                     DenseTensor* output) {
  ctx.template Alloc<T>(output);
  std::vector<int> paddings = paddings_t;
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
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
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
  // Tensor Core introduced from Volta GPUs supports more faster conv op
  // with FP16 or BF16 in NHWC data format.
  const bool compute_in_nhwc =
      (dtype == CUDNN_DATA_HALF || dtype == CUDNN_DATA_BFLOAT16) &&
      IsVoltaOrLater(ctx);
#else
  // Tensor Core introduced from Volta GPUs supports more faster conv op
  // with FP16 in NHWC data format. (BF16 require cudnn >= 8.1.0)
  const bool compute_in_nhwc = dtype == CUDNN_DATA_HALF && IsVoltaOrLater(ctx);
#endif
  // We will only do data format conversion from NHWC to NCHW.
  // cudnn will convert NCHW to NHWC automatically on Tensor Core.
  auto compute_format = compute_in_nhwc && channel_last
                            ? phi::backends::gpu::DataLayout::kNHWC
                            : phi::backends::gpu::DataLayout::kNCHW;
#endif
  VLOG(3) << "Compute ConvOp with cuDNN:"
          << " data_format=" << data_format << " compute_format="
          << (compute_format == phi::backends::gpu::DataLayout::kNHWC ? "NHWC"
                                                                      : "NCHW");

  // ------------ transformed tensor -----------
  DenseTensor transformed_input_channel(input.type());
  DenseTensor transformed_output(output->type());
  DenseTensor transformed_filter_channel(filter.type());

  if (channel_last && compute_format == phi::backends::gpu::DataLayout::kNCHW) {
    VLOG(3) << "Transform input tensor from NHWC to NCHW.";
    ResizeToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);
    TransToChannelFirst<Context, T>(ctx, &input, &transformed_input_channel);

    ResizeToChannelFirst<Context, T>(ctx, output, &transformed_output);

  } else {
    transformed_input_channel.ShareDataWith(input);
    transformed_output.ShareDataWith(*output);
  }
  if (compute_format == phi::backends::gpu::DataLayout::kNHWC &&
      !FLAGS_manually_trans_conv_filter) {
    VLOG(3) << "Transform filter tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, &filter, &transformed_filter_channel);
  } else {
    transformed_filter_channel.ShareDataWith(filter);
  }

  // update padding and dilation
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

  int data_dim = strides.size();  // 2d or 3d
  bool is_sys_pad = funcs::IsSymmetricPadding(paddings, data_dim);

  DenseTensor transformed_input;
  std::vector<int> padding_common(data_dim, 0);
  if (!is_sys_pad) {
    std::vector<int> padding_diff(data_dim);
    std::vector<int> new_input_shape_vec(data_dim + 2);
    new_input_shape_vec[0] = transformed_input_channel.dims()[0];

    if (compute_format == phi::backends::gpu::DataLayout::kNCHW) {
      new_input_shape_vec[1] = transformed_input_channel.dims()[1];
    } else {
      new_input_shape_vec[data_dim + 1] =
          transformed_input_channel.dims()[data_dim + 1];
    }

    std::vector<int> input_pad(transformed_input_channel.dims().size() * 2, 0);
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
    ConvCudnnKernelImplV8<T>(&transformed_input,
                             &transformed_filter_channel,
                             ctx,
                             strides,
                             padding_common,
                             dilations,
                             layout,
                             exhaustive_search,
                             deterministic,
                             groups,
                             &transformed_output);
  else
    ConvCudnnKernelImplV7<T>(&transformed_input,
                             &transformed_filter_channel,
                             ctx,
                             strides,
                             padding_common,
                             dilations,
                             compute_format,
                             layout,
                             exhaustive_search,
                             deterministic,
                             groups,
                             &transformed_output);
#else
  ConvCudnnKernelImplV7<T>(&transformed_input,
                           &transformed_filter_channel,
                           ctx,
                           strides,
                           padding_common,
                           dilations,
                           compute_format,
                           layout,
                           exhaustive_search,
                           deterministic,
                           groups,
                           &transformed_output);
#endif

  if (channel_last && compute_format == phi::backends::gpu::DataLayout::kNCHW) {
    TransToChannelLast<Context, T>(ctx, &transformed_output, output);
  }
}

template <typename T, typename Context>
void Conv3DCudnnKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& filter,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       int groups,
                       const std::vector<int>& dilations,
                       const std::string& data_format,
                       DenseTensor* out) {
  ConvCudnnKernel<T>(dev_ctx,
                     input,
                     filter,
                     strides,
                     paddings,
                     padding_algorithm,
                     dilations,
                     groups,
                     data_format,
                     out);
}

template <typename T, typename Context>
void DepthwiseConvCudnnKernel(const Context& dev_ctx,
                              const DenseTensor& input,
                              const DenseTensor& filter,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations,
                              const std::string& data_format,
                              DenseTensor* out) {
  ConvCudnnKernel<T>(dev_ctx,
                     input,
                     filter,
                     strides,
                     paddings,
                     padding_algorithm,
                     dilations,
                     groups,
                     data_format,
                     out);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(depthwise_conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::DepthwiseConvCudnnKernel,
                   float,
                   phi::dtype::float16) {}

#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(conv3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#elif CUDNN_VERSION_MIN(8, 6, 0) && CUDA_VERSION >= 11800 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   double,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(conv2d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::ConvCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Conv3DCudnnKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif

#endif

// todo register bfloat16
