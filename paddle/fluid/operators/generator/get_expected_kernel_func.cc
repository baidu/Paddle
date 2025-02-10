/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/operators/generator/get_expected_kernel_func.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
namespace paddle::operators {

// oneDNN's reduction kernel is optimized only for reducing throughout the
// most outer dims, so in case of another type of reduction, it would be
// better to fallback to native implementation
static bool ReduceOpHasOptimizedOneDNNKernel(
    const framework::ExecutionContext& ctx) {
  // native reduce kernels don't support bf16
  // so oneDNN kernel is enforced in that case
  if (ctx.Input<phi::DenseTensor>("X")->dtype() == phi::DataType::BFLOAT16)
    return true;

  if (!ctx.HasAttr("dim") || !ctx.HasAttr("reduce_all")) {
    return false;
  }

  auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
  const bool reduce_all = ctx.Attr<bool>("reduce_all");
  int ndims = ctx.Input<phi::DenseTensor>("X")->dims().size();

  if (reduce_all) {
    return true;
  }

  for (auto& reduce_dim : reduce_dims) {
    if (reduce_dim < 0) reduce_dim = ndims + reduce_dim;
  }
  sort(reduce_dims.begin(), reduce_dims.end());
  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    if (reduce_dims[reduce_dims.size() - i - 1] !=
        static_cast<int>(ndims - i - 1)) {
      return false;
    }
  }

  return true;
}

// only poolop
bool CanMKLDNNSupportPool(const framework::ExecutionContext& ctx) {
  if (ctx.Attr<bool>("adaptive") == false) return true;
  // oneDNN is supporting only unchangeable in size pool window
  auto src_tz = common::vectorize(ctx.Input<phi::DenseTensor>("X")->dims());
  if (!ctx.HasAttr("ksize")) {
    return false;
  }
  std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
  // Fast but not exhaustive check
  return ((src_tz[src_tz.size() - 1] % ksize[1] == 0) &&
          (src_tz[src_tz.size() - 2] % ksize[0] == 0));
}

phi::KernelKey GetCheckFiniteAndUnscaleExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto dtype = framework::proto::VarType::FP32;
  if (!ctx.MultiInputVar("X").empty()) {
    dtype = op_ptr->IndicateVarDataType(ctx, "X");
  }
  return phi::KernelKey(dtype, ctx.GetPlace());
}

phi::KernelKey GetConcatExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  (void)op_ptr;
  auto inputs = ctx.MultiInput<phi::DenseTensor>("X");
  auto input_data_type = framework::proto::VarType::Type(0);
  bool flag = false;
  for (auto* input : inputs) {
    if (input->IsInitialized()) {
      input_data_type = framework::TransToProtoVarType(input->dtype());
      flag = true;
      break;
    }
  }
  int batch_size = !inputs[0]->lod().empty() ? inputs[0]->lod()[0].size() - 1
                                             : inputs[0]->dims()[0];
  if (inputs.size() > 64 && batch_size < 1000) {
    op_ptr->SetDnnFallback(true);
  }
  if (flag == 0) {
    PADDLE_THROW(
        common::errors::InvalidArgument("All Inputs of Concat OP are Empty!"));
  }
  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetReduceExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  // choose cudnn kernel if the runtime supported.
  auto input_data_type = op_ptr->IndicateVarDataType(ctx, "X");

  if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5 ||
      !ReduceOpHasOptimizedOneDNNKernel(ctx)) {
    op_ptr->SetDnnFallback(true);
  }

  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(
        ctx.GetPlace().GetType() == phi::AllocationType::GPU ||
            ctx.GetPlace().GetType() == phi::AllocationType::XPU ||
            ctx.GetPlace().GetType() == phi::AllocationType::CUSTOM,
        true,
        common::errors::InvalidArgument(
            "float16 can only be used on GPU or NPU or XPU place"));
  }
  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetReduceGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  int out_dtype = ctx.Attr<int>("out_dtype");
  auto input_data_type =
      (out_dtype >= 0)
          ? static_cast<framework::proto::VarType::Type>(out_dtype)
          : op_ptr->IndicateVarDataType(ctx, framework::GradVarName("Out"));
  if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5) {
    op_ptr->SetDnnFallback(true);
  }

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetReduceOpUseInputPlaceExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  phi::KernelKey kt = op_ptr->OperatorWithKernel::GetExpectedKernelType(ctx);
  kt.set_backend(
      phi::TransToPhiBackend(ctx.Input<phi::DenseTensor>("X")->place()));
  return kt;
}

phi::KernelKey GetAssignExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  const framework::Variable* var = ctx.InputVar("X");
  if (var->IsType<phi::TensorArray>()) {
    auto t_arr = var->Get<phi::TensorArray>();
    // NOTE(liym27): Support an empty tensor array as Input.
    // And set the kernel type is float.
    if (t_arr.empty()) {
      return phi::KernelKey(framework::proto::VarType::FP32,
                            ctx.device_context().GetPlace());
    }
  }
  return phi::KernelKey(
      op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X"),
      ctx.device_context().GetPlace());
}

phi::KernelKey GetPoolExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto data_type = op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
  op_ptr->SetDnnFallback(!CanMKLDNNSupportPool(ctx));
  // NOTE(jiahongyu) END: Above codes originally enclosed by PADDLE_WITH_DNNL

  return phi::KernelKey(data_type, ctx.GetPlace());
}

phi::KernelKey GetPoolDoubleGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto data_type =
      op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "grad_x@GRAD");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
  op_ptr->SetDnnFallback(!CanMKLDNNSupportPool(ctx));
  // NOTE(jiahongyu) END: Above codes originally enclosed by PADDLE_WITH_DNNL

  return phi::KernelKey(data_type, ctx.GetPlace());
}

phi::KernelKey GetSgdExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto data_type = op_ptr->IndicateVarDataType(ctx, "Param");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
  const auto* param_var = ctx.InputVar("Param");
  const auto* grad_var = ctx.InputVar("Grad");

  // supported cases
  bool dense_param_sparse_grad = param_var->IsType<phi::DenseTensor>() &&
                                 grad_var->IsType<phi::SelectedRows>();
  bool dense_param_and_grad = param_var->IsType<phi::DenseTensor>() &&
                              grad_var->IsType<phi::DenseTensor>();
  if (!(dense_param_sparse_grad || dense_param_and_grad)) {
    op_ptr->SetDnnFallback(true);
  }
  // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

  return phi::KernelKey(data_type, ctx.GetPlace());
}

phi::KernelKey GetSoftmaxExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  // choose cudnn kernel if the runtime supported.
  std::string data_format = ctx.Attr<std::string>("data_format");
  phi::DataLayout layout_ = common::StringToDataLayout(data_format);
  auto input_data_type = op_ptr->IndicateVarDataType(ctx, "X");
  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(
        ctx.GetPlace().GetType() == phi::AllocationType::GPU ||
            ctx.GetPlace().GetType() == phi::AllocationType::XPU ||
            ctx.GetPlace().GetType() == phi::AllocationType::CUSTOM,
        true,
        common::errors::InvalidArgument(
            "float16 can only be used on GPU/XPU and custom place"));
  }
  return phi::KernelKey(
      ctx.GetPlace(), layout_, phi::TransToPhiDataType(input_data_type));
}

phi::KernelKey GetSoftmaxGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  // choose cudnn kernel if the runtime supported.
  std::string data_format = ctx.Attr<std::string>("data_format");
  phi::DataLayout layout_ = common::StringToDataLayout(data_format);
  auto input_data_type =
      op_ptr->IndicateVarDataType(ctx, framework::GradVarName("Out"));
  if (input_data_type == framework::proto::VarType::FP16) {
    if (!(ctx.GetPlace().GetType() == phi::AllocationType::GPU ||
          ctx.GetPlace().GetType() == phi::AllocationType::XPU ||
          ctx.GetPlace().GetType() == phi::AllocationType::CUSTOM))
      PADDLE_THROW(common::errors::InvalidArgument(
          "float16 can only be used on GPU/XPU and custom place"));
  }
  return phi::KernelKey(
      ctx.GetPlace(), layout_, phi::TransToPhiDataType(input_data_type));
}

phi::KernelKey GetStridedSliceExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto* in_var = ctx.InputVar("Input");
  auto is_in_var_array = in_var->IsType<phi::TensorArray>();
  if (is_in_var_array) {
    auto& tensor_array = in_var->Get<phi::TensorArray>();
    for (auto& tensor : tensor_array) {
      if (!(tensor.place().GetType() == phi::AllocationType::GPUPINNED)) {
        PADDLE_ENFORCE_EQ(
            phi::is_same_place(tensor.place(), ctx.device_context().GetPlace()),
            true,
            common::errors::InvalidArgument(
                "Place of context is %s. Place of input tensor is %s. They "
                "are should be same, but received different place.",
                string::to_string(ctx.device_context().GetPlace()),
                string::to_string(tensor.place())));
      }
    }
    return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "Input"),
                          ctx.GetPlace());
  }
  // NOTE: cuda pinned tensor need to copy its data to target place
  auto in_tensor = ctx.Input<phi::DenseTensor>("Input");
  if (in_tensor->place().GetType() == phi::AllocationType::GPUPINNED) {
    return phi::KernelKey(framework::TransToProtoVarType(in_tensor->dtype()),
                          ctx.GetPlace());
  }
  return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "Input"),
                        in_tensor->place());
}

phi::KernelKey GetStridedSliceGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  return phi::KernelKey(
      op_ptr->IndicateVarDataType(ctx, framework::GradVarName("Out")),
      ctx.GetPlace());
}

phi::KernelKey GetUpdateLossScalingExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto dtype = framework::proto::VarType::FP32;
  if (!ctx.MultiInputVar("X").empty()) {
    dtype = op_ptr->IndicateVarDataType(ctx, "X");
  }
  return phi::KernelKey(dtype, ctx.GetPlace());
}

phi::KernelKey GetMatrixNmsExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "Scores"),
                        phi::CPUPlace());
}

phi::KernelKey GetPad3dExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto input_data_type = op_ptr->IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_DNNL
  // only constant mode and non-blocked layouts are supported for oneDNN
  if (op_ptr->CanMKLDNNBeUsed(ctx, input_data_type) &&
      ctx.Attr<std::string>("mode") == "constant" &&
      ctx.Input<phi::DenseTensor>("X")->mem_desc().get_inner_nblks() == 0) {
    return phi::KernelKey(phi::Backend::ONEDNN,
                          phi::DataLayout::ONEDNN,
                          phi::TransToPhiDataType(input_data_type));
  }
#endif
  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetYoloLossExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "X"), phi::CPUPlace());
}

phi::KernelKey GetUniqueExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  (void)ctx;
  // Return CPUPlace when Attr("is_sorted") is false. Because it means
  // that fluid.layers.unique is called, but there is no cuda kernel.
  if (!ctx.Attr<bool>("is_sorted")) {
    return phi::KernelKey(
        op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        phi::CPUPlace());
  } else {
    // new version paddle.unique is called.
    return phi::KernelKey(
        op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.GetPlace());
  }
}

phi::KernelKey GetInstanceNormExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto input_data_type =
      op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto in_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    in_param_type = framework::proto::VarType::FP64;
  }
  if (ctx.HasInput("Scale")) {
    PADDLE_ENFORCE_EQ(
        in_param_type,
        framework::TransToProtoVarType(
            ctx.Input<phi::DenseTensor>("Scale")->dtype()),
        common::errors::InvalidArgument("Scale input should be of float type"));
  }
  if (ctx.HasInput("Bias")) {
    PADDLE_ENFORCE_EQ(
        in_param_type,
        framework::TransToProtoVarType(
            ctx.Input<phi::DenseTensor>("Bias")->dtype()),
        common::errors::InvalidArgument("Bias input should be of float type"));
  }

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetLayerNormExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto input_data_type =
      op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_DNNL
  int begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
  if (begin_norm_axis != ctx.Input<phi::DenseTensor>("X")->dims().size() - 1) {
    op_ptr->SetDnnFallback(true);
  }
  // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_DNNL

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetConvExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto input_data_type = op_ptr->IndicateVarDataType(ctx, "Input");
  // todo enable data layout when it's ready
  // (https://github.com/PaddlePaddle/Paddle/pull/20042)

  if (input_data_type != framework::proto::VarType::INT8 &&
      input_data_type != framework::proto::VarType::UINT8 &&
      input_data_type != framework::proto::VarType::BF16) {
    auto filter_data_type = framework::TransToProtoVarType(
        ctx.Input<phi::DenseTensor>("Filter")->dtype());
    PADDLE_ENFORCE_EQ(
        input_data_type,
        filter_data_type,
        common::errors::InvalidArgument(
            "input and filter data type should be consistent, "
            "but received input data type is %s and filter type "
            "is %s",
            paddle::framework::DataTypeToString(input_data_type),
            paddle::framework::DataTypeToString(filter_data_type)));
  }

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetBincountExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto data_type = ctx.HasInput("Weights")
                       ? op_ptr->IndicateVarDataType(ctx, "Weights")
                       : op_ptr->IndicateVarDataType(ctx, "X");
  return phi::KernelKey(data_type, ctx.device_context().GetPlace());
}

phi::KernelKey GetMulticlassNmsExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "Scores"),
                        phi::CPUPlace());
}

}  // namespace paddle::operators
