// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/pe/nn.h"

#include <functional>

#include "paddle/cinn/adt/op_equation_context.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/broadcast.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/layout.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace op {
using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

std::shared_ptr<OpStrategy> StrategyForRelu(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute relu_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of relu compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "At least one input tensor for relu compute! Please check."));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The pack_args's dimensions should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();
    auto out = pe::Relu(A.as_tensor_ref(), 0.0, tensor_name);
    *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of relu op is empty! Please check."));
  strategy->AddImpl(relu_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.relu.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForRelu6Symbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute relu6_compute([](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of relu6 compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "At least one input tensor for relu6 compute! Please check."));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The pack_args's dimensions should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();
    auto out = pe::Relu6(A.as_tensor_ref(), 0.0, tensor_name);
    *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of relu6 op is empty! Please check."));
  strategy->AddImpl(relu6_compute, lang::PackedFunc(), "strategy.relu6.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForReluSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute relu_compute([=](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of relu compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "At least one input tensor for relu compute! Please check."));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The pack_args's dimensions should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();
    auto out = pe::Relu(A.as_tensor_ref(), 0.0, tensor_name);
    *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of relu op is empty! Please check."));
  strategy->AddImpl(relu_compute, lang::PackedFunc(), "strategy.relu.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForRelu6(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute relu6_compute([](lang::Args args,
                                          lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of relu6 compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "At least one input tensor for relu6 compute! Please check."));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "The pack_args's dimensions should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();
    auto out = pe::Relu6(A.as_tensor_ref(), 0.0, tensor_name);
    *ret = CINNValuePack{{CINNValue(Expr(out.get()))}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of relu6 op is empty! Please check."));
  strategy->AddImpl(relu6_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.relu6.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForConv2d(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::vector<int> padding({0, 0});
  std::vector<int> stride({1, 1});
  std::vector<int> dilation({1, 1});
  std::string data_format = "NCHW";
  int groups = 1;
  std::string key = "";
  std::string conv_type = "";
  bool use_onednn = false;
  if (attrs.attr_store.find("padding") != attrs.attr_store.end()) {
    padding = absl::get<std::vector<int>>(attrs.attr_store.at("padding"));
  }
  if (attrs.attr_store.find("stride") != attrs.attr_store.end()) {
    stride = absl::get<std::vector<int>>(attrs.attr_store.at("stride"));
  }
  if (attrs.attr_store.find("dilation") != attrs.attr_store.end()) {
    dilation = absl::get<std::vector<int>>(attrs.attr_store.at("dilation"));
  }
  if (attrs.attr_store.find("data_format") != attrs.attr_store.end()) {
    data_format = absl::get<std::string>(attrs.attr_store.at("data_format"));
  }
  if (attrs.attr_store.find("groups") != attrs.attr_store.end()) {
    groups = absl::get<int>(attrs.attr_store.at("groups"));
  }
  if (attrs.attr_store.find("use_onednn") != attrs.attr_store.end()) {
    use_onednn = absl::get<bool>(attrs.attr_store.at("use_onednn"));
  }
  if (attrs.attr_store.find("key") != attrs.attr_store.end()) {
    key = absl::get<std::string>(attrs.attr_store.at("key"));
  }
  // get conv type
  if (attrs.attr_store.find("conv_type") != attrs.attr_store.end()) {
    conv_type = absl::get<std::string>(attrs.attr_store.at("conv_type"));
  } else {
    conv_type = "forward";
  }

#ifndef CINN_WITH_CUDNN
  PADDLE_ENFORCE_EQ(conv_type,
                    "forward",
                    ::common::errors::InvalidArgument(
                        "conv type should be 'forward', but got %s."
                        "cudnn is not found, backward_data/backward_filter is "
                        "not supported!",
                        conv_type));
#endif

  framework::CINNCompute conv2d_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    std::vector<CINNValue> res;
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of conv2d compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      2U,
                      ::common::errors::InvalidArgument(
                          "The size of pack_args in conv2d is incorrect. "
                          "Expected size should be greater than or equal "
                          "to 2, but receive %d. ",
                          pack_args.size()));
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE(B.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! B is not a tensor."));
    PADDLE_ENFORCE_EQ(
        padding.size(),
        2,
        ::common::errors::InvalidArgument(
            "The size of padding in conv2d op should be 2, but got %d.",
            padding.size()));
    PADDLE_ENFORCE_EQ(
        stride.size(),
        2,
        ::common::errors::InvalidArgument(
            "The size of stride in conv2d op should be 2, but got %d.",
            stride.size()));
    PADDLE_ENFORCE_EQ(
        dilation.size(),
        2,
        ::common::errors::InvalidArgument(
            "The size of dilation in conv2d op should be 2, but got %d.",
            dilation.size()));
    std::vector<ir::Tensor> out;
    VLOG(3) << "input shape: " << utils::Join(A.as_tensor_ref()->shape, ", ");
    VLOG(3) << "weight shape: " << utils::Join(B.as_tensor_ref()->shape, ", ");
    PADDLE_ENFORCE_GE(pack_args.size(),
                      3,
                      ::common::errors::InvalidArgument(
                          "The size of pack_args in conv2d op should be "
                          "greater than or equal to 3, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[2].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[2] is not string."));
    std::string tensor_name = pack_args[2].operator std::string();
    if (data_format == "NCHW") {
      // A is input: [N, C, H, W], B is filter: [C_out, C_in/group,
      // filter_h, filter_w]
      target.arch.Match(
          [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
          [&](common::X86Arch) {
            if (groups == 1 && !use_onednn) {
              out = pe::Conv2d_NCHW_5D(A.as_tensor_ref(),
                                       B.as_tensor_ref(),
                                       padding[0],
                                       padding[1],
                                       stride[0],
                                       stride[1],
                                       dilation[0],
                                       dilation[1],
                                       key,
                                       tensor_name,
                                       target);
            } else {
#ifdef CINN_WITH_DNNL
              out = pe::Conv2d_NCHW_ONEDNN(A.as_tensor_ref(),
                                           B.as_tensor_ref(),
                                           padding[0],
                                           padding[1],
                                           stride[0],
                                           stride[1],
                                           dilation[0],
                                           dilation[1],
                                           tensor_name);
#else
              out = pe::Conv2d_NCHW_5D(A.as_tensor_ref(),
                                       B.as_tensor_ref(),
                                       padding[0],
                                       padding[1],
                                       stride[0],
                                       stride[1],
                                       dilation[0],
                                       dilation[1],
                                       key,
                                       tensor_name);
#endif
            }
          },
          [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
          [&](common::NVGPUArch) {
            if (conv_type == "forward") {
              out = pe::Conv2d_NCHW(A.as_tensor_ref(),
                                    B.as_tensor_ref(),
                                    padding[0],
                                    padding[1],
                                    stride[0],
                                    stride[1],
                                    dilation[0],
                                    dilation[1],
                                    tensor_name);
              out.push_back(B.as_tensor_ref());
            } else {
#ifdef CINN_WITH_CUDNN
              // as backward_data and backward_filter is not
              // support now, we built a fake op to instead.
              // as the runtime use cudnn to compute the
              // conv2d, so this fake op is not been called.
              // When cinn support
              // backward_filter/backward_data code gen, this
              // code is to be removed.
              out = pe::Identity(A.as_tensor_ref());
              out.push_back(A.as_tensor_ref());
              out.push_back(B.as_tensor_ref());
#endif
            }
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            PADDLE_THROW(
                ::common::errors::Unimplemented("CINN old obsolete code!"));
          });
    } else if (data_format == "NHWC") {
      // A is input: [N, H, W, C], B is filter: [C_out, C_in/group,
      // filter_h, filter_w]
      out = pe::Conv2d_NHWC(A.as_tensor_ref(),
                            B.as_tensor_ref(),
                            padding[0],
                            padding[1],
                            stride[0],
                            stride[1],
                            dilation[0],
                            dilation[1],
                            tensor_name);
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support NCHW and NHWC data layout\n"));
    }

    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    PADDLE_ENFORCE(out.size() == 3U || out.size() == 2U || out.size() == 5U ||
                       out.size() == 12U,
                   ::common::errors::InvalidArgument(
                       "The output tensor sizes of conv2d op in conv2d op"
                       "should be 2 or 3 or 5."));

    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule conv2d_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of conv2d schedule is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound(
            "The vec_ast of conv2d schedule is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    target.arch.Match(
        [&](common::UnknownArch) {
          PADDLE_THROW(::common::errors::InvalidArgument(
              "This target [%s] is not supported yet.", target));
        },
        [&](common::X86Arch) {
          PADDLE_THROW(::common::errors::InvalidArgument(
              "This target [%s] is not supported yet.", target));
        },
        [&](common::ARMArch) {
          PADDLE_THROW(::common::errors::InvalidArgument(
              "This target [%s] is not supported yet.", target));
        },
        [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDNN
          // If conv_type is backward_filter or backward_data, we built a fake
          // op. As runtime use cudnn to compute conv2d, this fake op is not to
          // be called. When cinn support backward_filter/backward_data code
          // gen, this code is to be removed.
          if (conv_type != "forward") {
            PADDLE_ENFORCE_EQ(
                vec_ast.size(),
                1,
                ::common::errors::InvalidArgument(
                    "The size of vec_ast should be 1, but got %d.",
                    vec_ast.size()));
            pe::IRGpuScheduleInjective(ir_sch, output_shapes.front(), target);
            std::vector<CINNValue> res{
                CINNValue(ir_sch.GetModule().GetExprs().at(0))};
            *ret = CINNValuePack{res};
            return;
          }
#endif
          int expr_size = vec_ast.size();
          if (expr_size == 2) {
            pe::IRCudaScheduleConv(ir_sch, target);
            VLOG(3) << "After IRCudaScheduleConv, arg_pack[0] is : "
                    << ir_sch.GetModule().GetExprs().at(0);
            std::vector<CINNValue> res{
                CINNValue(ir_sch.GetModule().GetExprs().at(0))};
            *ret = CINNValuePack{res};
            return;
          } else {
            CINN_NOT_IMPLEMENTED
          }
        },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          PADDLE_THROW(
              ::common::errors::Unimplemented("CINN old obsolete code!"));
        });
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of conv2d op is empty! Please check."));
  strategy->AddImpl(conv2d_compute, conv2d_schedule, "strategy.conv2d.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForDepthwiseConv2d(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  std::vector<int> padding = {0, 0};
  std::vector<int> stride = {1, 1};
  std::vector<int> dilation = {1, 1};
  std::string data_format = "NCHW";
  std::string key;
  if (attrs.attr_store.find("padding") != attrs.attr_store.end()) {
    padding = absl::get<std::vector<int>>(attrs.attr_store.at("padding"));
  }
  if (attrs.attr_store.find("stride") != attrs.attr_store.end()) {
    stride = absl::get<std::vector<int>>(attrs.attr_store.at("stride"));
  }
  if (attrs.attr_store.find("data_format") != attrs.attr_store.end()) {
    data_format = absl::get<std::string>(attrs.attr_store.at("data_format"));
  }
  if (attrs.attr_store.find("dilation") != attrs.attr_store.end()) {
    dilation = absl::get<std::vector<int>>(attrs.attr_store.at("dilation"));
  }
  if (attrs.attr_store.find("key") != attrs.attr_store.end()) {
    key = absl::get<std::string>(attrs.attr_store.at("key"));
  }

  framework::CINNCompute depthwise_conv2d_compute([=](lang::Args args,
                                                      lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound("The input argument of depthwise_conv "
                                   "compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(
        pack_args.size(),
        2U,
        ::common::errors::InvalidArgument(
            "The size of pack_args in depthwise_conv is incorrect. "
            "Expected size should be greater than or equal "
            "to 2, but receive %d. ",
            pack_args.size()));
    Expr A = pack_args[0];
    Expr B = pack_args[1];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE(B.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! B is not a tensor."));
    PADDLE_ENFORCE_EQ(
        padding.size(),
        2,
        ::common::errors::InvalidArgument(
            "The size of padding in depthwise_conv op should be 2, but got %d.",
            padding.size()));
    PADDLE_ENFORCE_EQ(
        stride.size(),
        2,
        ::common::errors::InvalidArgument(
            "The size of stride in depthwise_conv op should be 2, but got %d.",
            stride.size()));
    PADDLE_ENFORCE(data_format == "NCHW" || data_format == "NHWC",
                   ::common::errors::InvalidArgument(
                       "only support NCHW/NHWC data_format."));
    std::vector<ir::Tensor> out;
    PADDLE_ENFORCE_GE(pack_args.size(),
                      3,
                      ::common::errors::InvalidArgument(
                          "The size of pack_args in depthwise_conv op should "
                          "be greater than or equal to 3, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[2].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[2] is not string."));
    std::string tensor_name = pack_args[2].operator std::string();
    if (data_format == "NCHW") {
      target.arch.Match(
          [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
          [&](common::X86Arch) {
            out = pe::Conv2d_NCHW_5D(A.as_tensor_ref(),
                                     B.as_tensor_ref(),
                                     padding[0],
                                     padding[1],
                                     stride[0],
                                     stride[1],
                                     dilation[0],
                                     dilation[1],
                                     key,
                                     tensor_name,
                                     target);
          },
          [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
          [&](common::NVGPUArch) {
            out = pe::Depthwise_Conv2d_NCHW(A.as_tensor_ref(),
                                            B.as_tensor_ref(),
                                            padding[0],
                                            padding[1],
                                            stride[0],
                                            stride[1],
                                            tensor_name);
          },
          [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
            PADDLE_THROW(
                ::common::errors::Unimplemented("CINN old obsolete code!"));
          });
    } else if (data_format == "NHWC") {
      out = pe::Depthwise_Conv2d_NHWC(A.as_tensor_ref(),
                                      B.as_tensor_ref(),
                                      padding[0],
                                      padding[1],
                                      stride[0],
                                      stride[1],
                                      tensor_name);
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support NCHW and NHWC data layout\n"));
    }

    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    PADDLE_ENFORCE(out.size() == 2U || out.size() == 1U || out.size() == 5U,
                   ::common::errors::InvalidArgument(
                       "The output tensor sizes of depthwise_conv op in "
                       "depthwise_conv op should be 1 or 2 or 5."));

    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule depthwise_conv2d_schedule([=](lang::Args args,
                                                        lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of InjectiveSchedule is empty! Please check."));
    cinn::common::CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound("The vec_ast is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    target.arch.Match(
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) {
          pe::IRCudaScheduleDepthwiseConv(ir_sch, vec_tensor);
        },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          PADDLE_THROW(
              ::common::errors::Unimplemented("CINN old obsolete code!"));
        });
    std::vector<cinn::common::CINNValue> res{
        cinn::common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = cinn::common::CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of depthwise_conv op is empty! Please check."));
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(depthwise_conv2d_compute,
                      depthwise_conv2d_schedule,
                      "strategy.depthwise_conv.x86",
                      1);
  } else {
    VLOG(3) << "depthwise_conv op with dtype != float32 is not implemented "
               "yet!";
  }
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForBatchNorm(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  float epsilon = 0.00001f;
  std::vector<std::string> input_layouts;
  if (attrs.attr_store.find("epsilon") != attrs.attr_store.end()) {
    epsilon = absl::get<float>(attrs.attr_store.at("epsilon"));
  }
  if (attrs.attr_store.find("input_layouts") != attrs.attr_store.end()) {
    input_layouts = absl::get<std::vector<std::string>>(
        attrs.attr_store.at("input_layouts"));
  }
  framework::CINNCompute batchnorm_compute([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of batchnorm compute is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    PADDLE_ENFORCE_GE(arg_pack.size(),
                      5U,
                      ::common::errors::InvalidArgument(
                          "The size of arg_pack in batchnorm compute should be "
                          "greater than or equal to 5, but got %d.",
                          arg_pack.size()));
    Expr A = arg_pack[0];
    Expr Scale = arg_pack[1];
    Expr Bias = arg_pack[2];
    Expr Mean = arg_pack[3];
    Expr Variance = arg_pack[4];
    PADDLE_ENFORCE_EQ(
        arg_pack.size(),
        6U,
        ::common::errors::InvalidArgument("The size of arg_pack in batchnorm "
                                          "compute should be 6, but got %d.",
                                          arg_pack.size()));
    PADDLE_ENFORCE(arg_pack[5].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! arg_pack[5] is not string."));
    std::string out_name = arg_pack[5];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    PADDLE_ENFORCE(Scale.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! Scale is not a tensor."));
    PADDLE_ENFORCE(Bias.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! Bias is not a tensor."));
    PADDLE_ENFORCE(Mean.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! Mean is not a tensor."));
    PADDLE_ENFORCE(Variance.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! Variance is not a tensor."));
    ir::Tensor out;
    auto tensor_input = A.as_tensor_ref();
    if (tensor_input->shape.size() != 4 &&
        std::holds_alternative<common::X86Arch>(target.arch)) {
      PADDLE_ENFORCE_EQ(
          input_layouts.size(),
          5U,
          ::common::errors::InvalidArgument(
              "batch_norm_NCHWc's input layouts should be 5, but got %d.",
              input_layouts.size()));
      std::string input_layout = input_layouts[0];
      PADDLE_ENFORCE_GE(
          input_layout.size(),
          5U,
          ::common::errors::InvalidArgument(
              "batch_norm_NCHWc's input layout should be 5, but got %d.",
              input_layout.size()));
      PADDLE_ENFORCE_EQ(input_layout.substr(0, 4),
                        "NCHW",
                        ::common::errors::InvalidArgument(
                            "batch_norm_NCHWc's input layout substr "
                            "should be 'NCHW', but got %s.",
                            input_layout.substr(0, 4)));
      PADDLE_ENFORCE_EQ(
          tensor_input->shape.size(),
          5U,
          ::common::errors::InvalidArgument(
              "batch_norm_NCHWc's input shape's size should be 5, but got %d.",
              tensor_input->shape.size()));
      out = pe::BatchNorm_NCHWc(tensor_input,
                                Scale.as_tensor_ref(),
                                Bias.as_tensor_ref(),
                                Mean.as_tensor_ref(),
                                Variance.as_tensor_ref(),
                                epsilon,
                                out_name);
    } else {
      out = pe::BatchNorm_NCHW(tensor_input,
                               Scale.as_tensor_ref(),
                               Bias.as_tensor_ref(),
                               Mean.as_tensor_ref(),
                               Variance.as_tensor_ref(),
                               epsilon,
                               out_name);
    }
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE(out_type.size(),
                 ::common::errors::NotFound(
                     "Out_type of batchnorm op is empty! Please check."));
  if (out_type[0] == Float(32)) {
    strategy->AddImpl(batchnorm_compute,
                      GetInjectiveScheduleFunc(output_shapes, target),
                      "strategy.batchnorm.x86",
                      1);
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "BatchNorm op with dtype != float32 is not implemented yet!"));
  }
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForPool1d(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute pool1d_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool1d compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "The input tensor of pool1d compute is empty! Please check."));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    auto attr_store = attrs.attr_store;
    std::vector<int> kernel_size;   // [kernel_w]
    std::vector<int> stride_size;   // [stride_w]
    std::vector<int> padding_size;  // [padding_left, padding_right]
    std::string pool_type = "max";
    bool ceil_mode = false;
    bool exclusive = true;
    std::string data_format = "NCW";
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "kernel_size") {
        kernel_size = absl::get<std::vector<int>>(iter.second);
      } else if (iter.first == "stride_size") {
        stride_size = absl::get<std::vector<int>>(iter.second);
      } else if (iter.first == "padding_size") {
        padding_size = absl::get<std::vector<int>>(iter.second);
      } else if (iter.first == "pool_type") {
        pool_type = absl::get<std::string>(iter.second);
      } else if (iter.first == "ceil_mode") {
        ceil_mode = absl::get<bool>(iter.second);
      } else if (iter.first == "exclusive") {
        exclusive = absl::get<bool>(iter.second);
      } else if (iter.first == "data_format") {
        data_format = absl::get<std::string>(iter.second);
      } else {
        LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
      }
    }
    PADDLE_ENFORCE_EQ(kernel_size.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Kernel_size for pool1d is empty! Please check."));
    PADDLE_ENFORCE_EQ(stride_size.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Stride_size for pool1d is empty! Please check."));
    PADDLE_ENFORCE_EQ(padding_size.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Padding_size for pool1d is empty! Please check."));
    PADDLE_ENFORCE(pool_type == "max" || pool_type == "avg",
                   ::common::errors::InvalidArgument(
                       "pool_type for pool1d should be max or avg."));

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();

    auto out = pe::Pool1d(A.as_tensor_ref(),
                          kernel_size,
                          stride_size,
                          padding_size,
                          pool_type,
                          ceil_mode,
                          exclusive,
                          data_format,
                          tensor_name);

    PADDLE_ENFORCE(out.size() == 1U || out.size() == 2U,
                   ::common::errors::InvalidArgument(
                       "The size of pe::Pool1d's output should be 1 or 2."));
    PADDLE_ENFORCE(out_type.size(),
                   ::common::errors::NotFound(
                       "Output type of Pool1d is empty! Please check."));
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(Expr(t.get())));
    }
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule pool1d_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool1d schedule is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound(
            "The input vec_ast of pool1d schedule is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    if (arg_pack.size() == 3UL) {
      PADDLE_ENFORCE_EQ(
          vec_tensor.size(),
          2,
          ::common::errors::InvalidArgument(
              "the size of vector tensor should be 2, but got %d.",
              vec_tensor.size()));
      Expr input_pad = vec_tensor[1];
      PADDLE_ENFORCE(input_pad.as_tensor(),
                     ::common::errors::InvalidArgument(
                         "Datatype error! input_pad is not a tensor."));
      auto block_input_pad = ir_sch.GetBlock(input_pad.as_tensor()->name);
      ir_sch.ComputeInline(block_input_pad);
    }
    auto schedule_nv_hygon = [&] {
      PADDLE_ENFORCE_EQ(
          vec_tensor.empty(),
          false,
          ::common::errors::NotFound("The vec_tensor is empty! Please check."));
      Expr Out = vec_tensor[0];
      PADDLE_ENFORCE(Out.as_tensor(),
                     ::common::errors::InvalidArgument(
                         "Datatype error! output is not a tensor."));
      auto loops = ir_sch.GetLoops(Out.as_tensor()->name);
      ir_sch.Split(loops[1], {-1, 2});
      loops = ir_sch.GetLoops(Out.as_tensor()->name);
      ir_sch.Bind(loops[0], "blockIdx.x");
      ir_sch.Bind(loops[1], "threadIdx.x");
    };
    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) {
          // Do nothing.
        },
        [&](common::ARMArch) {
          // Do nothing.
        },
        [&](common::NVGPUArch) { schedule_nv_hygon(); },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          schedule_nv_hygon();
        });
    std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pool1d_compute, pool1d_schedule, "strategy.pool1d.x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForPool2d(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  auto attr_store = attrs.attr_store;
  std::vector<int> kernel_size;   // [kernel_h, kernel_w]
  std::vector<int> stride_size;   // [stride_h, stride_w]
  std::vector<int> padding_size;  // [padding_top, padding_left,
                                  // padding_bottom, padding_right]
  std::string pool_type = "max";
  bool ceil_mode = false;
  bool exclusive = true;
  bool global_pooling = false;
  bool adaptive = false;
  std::string data_format = "NCHW";
  for (auto &iter : attrs.attr_store) {
    if (iter.first == "kernel_size") {
      kernel_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "stride_size") {
      stride_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "padding_size") {
      padding_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "pool_type") {
      pool_type = absl::get<std::string>(iter.second);
    } else if (iter.first == "ceil_mode") {
      ceil_mode = absl::get<bool>(iter.second);
    } else if (iter.first == "exclusive") {
      exclusive = absl::get<bool>(iter.second);
    } else if (iter.first == "data_format") {
      data_format = absl::get<std::string>(iter.second);
    } else if (iter.first == "global_pooling") {
      global_pooling = absl::get<bool>(iter.second);
    } else if (iter.first == "adaptive") {
      adaptive = absl::get<bool>(iter.second);
    }
  }
  // It can be removed after fixing the global_pool2d problem
  if (attr_store.count("origin_kernel_size")) {
    kernel_size =
        absl::get<std::vector<int>>(attr_store.at("origin_kernel_size"));
  }
  if (attr_store.count("origin_padding_size")) {
    padding_size =
        absl::get<std::vector<int>>(attr_store.at("origin_padding_size"));
  }
  if (attr_store.count("origin_global_pooling")) {
    global_pooling = absl::get<bool>(attr_store.at("origin_global_pooling"));
  }
  if (attr_store.count("origin_adaptive")) {
    adaptive = absl::get<bool>(attr_store.at("origin_adaptive"));
  }

  PADDLE_ENFORCE_EQ(kernel_size.empty(),
                    false,
                    ::common::errors::NotFound(
                        "Kernel_size for pool2d is empty! Please check."));
  PADDLE_ENFORCE_EQ(stride_size.empty(),
                    false,
                    ::common::errors::NotFound(
                        "Stride_size for pool2d is empty! Please check."));
  PADDLE_ENFORCE_EQ(padding_size.empty(),
                    false,
                    ::common::errors::NotFound(
                        "Padding_size for pool2d is empty! Please check."));
  PADDLE_ENFORCE(pool_type == "max" || pool_type == "avg",
                 ::common::errors::InvalidArgument(
                     "pool_type for pool2d should be max or avg."));

  PADDLE_ENFORCE_EQ(
      inputs.empty(),
      false,
      ::common::errors::NotFound(
          "The input tensor of pool2d compute is empty! Please check."));
  const ir::Tensor &A_tensor = inputs[0];
  PADDLE_ENFORCE(A_tensor->shape.size() == 4U || A_tensor->shape.size() == 5U,
                 ::common::errors::InvalidArgument(
                     "pool2d requires tensor's shape_size to be 4 or 5"));

  if (global_pooling) {
    int height_index = -1;
    int width_index = -1;
    if (data_format == "NCHW") {
      height_index = 2;
      width_index = 3;
    } else if (data_format == "NHWC") {
      height_index = 1;
      width_index = 2;
    } else if (data_format == "AnyLayout") {
      height_index = 2;
      width_index = 3;
      data_format = "NCHW";
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support 'NCHW' or 'NHWC' or 'AnyLayout' data_format.\n"));
    }
    kernel_size = {A_tensor->shape[height_index].as_int32(),
                   A_tensor->shape[width_index].as_int32()};
    padding_size = {0, 0, 0, 0};
  }
  if (kernel_size.size() == padding_size.size()) {
    padding_size.insert(
        padding_size.end(), padding_size.begin(), padding_size.end());
  }

  framework::CINNCompute global_pool2d_compute([=](lang::Args args,
                                                   lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool2d compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    ir::Tensor A_tensor = A.as_tensor_ref();

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();

    auto out = pe::GlobalPool2d(A_tensor, pool_type, tensor_name);
    PADDLE_ENFORCE(out.size() == 2U,
                   ::common::errors::InvalidArgument(
                       "The size of pe::GlobalPool2d's output should be 2."));
    *ret = CINNValuePack{{CINNValue(out[0]), CINNValue(out[1])}};
  });

  framework::CINNSchedule global_pool2d_schedule([=](lang::Args args,
                                                     lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool2d schedule is empty! Please check."));
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool1d schedule is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound("The vec_ast is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    target.arch.Match(
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) { pe::IRGlobalPoolScheduleGPU(ir_sch, target); },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          pe::IRGlobalPoolScheduleGPU(ir_sch, target);
        });
    std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = CINNValuePack{res};
  });

  framework::CINNCompute pool2d_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool2d compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    ir::Tensor A_tensor = A.as_tensor_ref();

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();

    auto out = pe::Pool2d(A_tensor,
                          kernel_size,
                          stride_size,
                          padding_size,
                          pool_type,
                          ceil_mode,
                          exclusive,
                          data_format,
                          adaptive,
                          tensor_name);

    PADDLE_ENFORCE(out.size() == 1U || out.size() == 2U,
                   ::common::errors::InvalidArgument(
                       "The size of pe::Pool2d's output should be 1 or 2."));
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    PADDLE_ENFORCE_EQ(out_type.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Output type of Pool2d is empty! Please check."));
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule pool2d_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool2d schedule is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound("The vec_ast is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    int arg_pack_size = arg_pack.size();
    // arg_pack_size == 3 case: input, input_pad, output
    // arg_pack_size == 4 case: input, input_pad, output, stage
    if (arg_pack_size == 3UL || arg_pack_size == 4UL) {
      PADDLE_ENFORCE_EQ(
          vec_tensor.size(),
          2,
          ::common::errors::InvalidArgument(
              "the size of vector tensor should be 2, but got %d.",
              vec_tensor.size()));
      Expr input_pad = vec_tensor[1];
      PADDLE_ENFORCE(input_pad.as_tensor(),
                     ::common::errors::InvalidArgument(
                         "Datatype error! input_pad is not a tensor."));
      const std::string &input_pad_name = input_pad.as_tensor()->name;
      VLOG(6) << "ComputeInline on " << input_pad_name;
      auto block_input_pad = ir_sch.GetBlock(input_pad_name);
      ir_sch.ComputeInline(block_input_pad);
    }
    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) {},
        [&](common::ARMArch) {},
        [&](common::NVGPUArch) {
          pe::IRPoolScheduleGPU(ir_sch, target, arg_pack_size);
        },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          pe::IRPoolScheduleGPU(ir_sch, target, arg_pack_size);
        });
    std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();

  bool use_warp_reduce = false;
  target.arch.Match(
      [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::X86Arch) { use_warp_reduce = false; },
      [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
      [&](common::NVGPUArch) {
        if (global_pooling && data_format == "NCHW") {
          // TODO(hp03): 32 may not be the exact number, try
          // also 16 or 8 or other number
          //      we choose 32 to make sure all the threads in
          //      a warp has work to do,
          if ((A_tensor->shape[2].as_int32() * A_tensor->shape[3].as_int32()) >=
              32) {
            use_warp_reduce = true;
          }
        }
      },
      [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
        CINN_NOT_IMPLEMENTED
      });
  strategy->AddImpl(pool2d_compute, pool2d_schedule, "strategy.pool2d.x86", 1);
  if (use_warp_reduce) {
    strategy->AddImpl(global_pool2d_compute,
                      global_pool2d_schedule,
                      "strategy.pool2d.gpu.global",
                      2);
  }

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForPool3d(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute pool3d_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool3d compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "The input tensor of pool3d compute is empty! Please check."));
    Expr A = pack_args[0];
    PADDLE_ENFORCE(A.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A is not a tensor."));
    auto attr_store = attrs.attr_store;
    std::vector<int> kernel_size;  // [kernel_d, kernel_h, kernel_w]
    std::vector<int> stride_size;  // [stride_d, stride_h, stride_w]
    std::vector<int>
        padding_size;  // [padding_front, padding_top, padding_left,
                       // padding_back, padding_bottom, padding_right]
    std::string pool_type = "max";
    bool ceil_mode = false;
    bool exclusive = true;
    std::string data_format = "NCDHW";
    for (auto &iter : attrs.attr_store) {
      if (iter.first == "kernel_size") {
        kernel_size = absl::get<std::vector<int>>(iter.second);
      } else if (iter.first == "stride_size") {
        stride_size = absl::get<std::vector<int>>(iter.second);
      } else if (iter.first == "padding_size") {
        padding_size = absl::get<std::vector<int>>(iter.second);
      } else if (iter.first == "pool_type") {
        pool_type = absl::get<std::string>(iter.second);
      } else if (iter.first == "ceil_mode") {
        ceil_mode = absl::get<bool>(iter.second);
      } else if (iter.first == "exclusive") {
        exclusive = absl::get<bool>(iter.second);
      } else if (iter.first == "data_format") {
        data_format = absl::get<std::string>(iter.second);
      } else {
        LOG(ERROR) << "Unsupported attr: " << iter.first << std::endl;
      }
    }
    PADDLE_ENFORCE_EQ(kernel_size.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Kernel_size for pool3d is empty! Please check."));
    PADDLE_ENFORCE_EQ(stride_size.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Stride_size for pool3d is empty! Please check."));
    PADDLE_ENFORCE_EQ(padding_size.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Padding_size for pool3d is empty! Please check."));
    PADDLE_ENFORCE(pool_type == "max" || pool_type == "avg",
                   ::common::errors::InvalidArgument(
                       "pool_type for pool3d should be max or avg."));

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();

    auto out = pe::Pool3d(A.as_tensor_ref(),
                          kernel_size,
                          stride_size,
                          padding_size,
                          pool_type,
                          ceil_mode,
                          exclusive,
                          data_format,
                          tensor_name);

    PADDLE_ENFORCE(out.size() == 1U || out.size() == 2U,
                   ::common::errors::InvalidArgument(
                       "The size of pe::Pool3d's output should be 1 or 2."));
    PADDLE_ENFORCE_EQ(out_type.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Output type of Pool3d is empty! Please check."));

    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(Expr(t.get())));
    }
    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule pool3d_schedule([=](lang::Args args,
                                              lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of pool3d schedule is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    std::vector<Expr> vec_tensor;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      } else if (arg_pack[i].is_tensor()) {
        Expr temp = arg_pack[i];
        vec_tensor.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound("The vec_ast is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    if (arg_pack.size() == 3UL) {
      PADDLE_ENFORCE_EQ(
          vec_tensor.size(),
          2,
          ::common::errors::InvalidArgument(
              "the size of vector tensor should be 2, but got %d.",
              vec_tensor.size()));
      Expr input_pad = vec_tensor[1];
      PADDLE_ENFORCE(input_pad.as_tensor(),
                     ::common::errors::InvalidArgument(
                         "Datatype error! input_pad is not a tensor."));
      auto block_input_pad = ir_sch.GetBlock(input_pad.as_tensor()->name);
      ir_sch.ComputeInline(block_input_pad);
    }
    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) { /*nothing*/ },
        [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) {
          PADDLE_ENFORCE_EQ(vec_tensor.empty(),
                            false,
                            ::common::errors::NotFound(
                                "The vec_tensor is empty! Please check."));
          Expr Out = vec_tensor[0];
          PADDLE_ENFORCE(Out.as_tensor(),
                         ::common::errors::InvalidArgument(
                             "Datatype error! Output is not a tensor."));
          auto loops = ir_sch.GetLoops(Out.as_tensor()->name);
          ir_sch.Split(loops[1], {-1, 2});
          loops = ir_sch.GetLoops(Out.as_tensor()->name);
          ir_sch.Bind(loops[0], "blockIdx.x");
          ir_sch.Bind(loops[1], "threadIdx.x");
        },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          PADDLE_ENFORCE_EQ(vec_tensor.empty(),
                            false,
                            ::common::errors::NotFound(
                                "The vec_tensor is empty! Please check."));
          Expr Out = vec_tensor[0];
          PADDLE_ENFORCE(Out.as_tensor(),
                         ::common::errors::InvalidArgument(
                             "Datatype error! Output is not a tensor."));
          auto loops = ir_sch.GetLoops(Out.as_tensor()->name);
          ir_sch.Split(loops[1], {-1, 2});
          loops = ir_sch.GetLoops(Out.as_tensor()->name);
          ir_sch.Bind(loops[0], "blockIdx.x");
          ir_sch.Bind(loops[1], "threadIdx.x");
        });
    std::vector<CINNValue> res{CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pool3d_compute, pool3d_schedule, "strategy.pool3d.x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForSoftmax(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  int axis = -1;
  bool use_onednn = false;
  if (attrs.attr_store.count("axis")) {
    axis = absl::get<int>(attrs.attr_store.at("axis"));
  }
  if (attrs.attr_store.count("use_onednn")) {
    use_onednn = absl::get<bool>(attrs.attr_store.at("use_onednn"));
  }
  framework::CINNCompute softmax_compute([=](lang::Args args,
                                             lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of softmax compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound(
            "The input tensors of softmax compute is empty! Please check."));
    Expr A_expr = pack_args[0];
    PADDLE_ENFORCE(A_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A_expr is not a tensor."));
    ir::Tensor A = A_expr.as_tensor_ref();
    int new_axis = axis;
    if (axis == -1) {
      new_axis = A->shape.size() - 1;
    }
    std::vector<ir::Tensor> out;

    PADDLE_ENFORCE_GE(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be greater than or "
                          "equal to 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(
        pack_args[pack_args.size() - 1].is_string(),
        ::common::errors::InvalidArgument(
            "Datatype error! pack_args[pack_args.size() - 1] is not string."));
    std::string tensor_name =
        pack_args[pack_args.size() - 1].operator std::string();

#ifdef CINN_WITH_DNNL
    if (use_onednn) {
      out = pe::SoftmaxONEDNN(A, new_axis, tensor_name);
    } else {
      out = pe::Softmax(A, new_axis, tensor_name);
    }
#else
    out = pe::Softmax(A, new_axis, tensor_name);
#endif
    std::vector<CINNValue> res;
    for (auto &t : out) {
      res.push_back(CINNValue(t));
    }
    PADDLE_ENFORCE_EQ(
        out.size(),
        2U,
        ::common::errors::InvalidArgument(
            "The size of pe::Softmax's output should be 2, but got %d.",
            out.size()));
    PADDLE_ENFORCE_EQ(out_type.empty(),
                      false,
                      ::common::errors::NotFound(
                          "Output type of Softmax is empty! Please check."));

    *ret = CINNValuePack{res};
  });

  framework::CINNSchedule softmax_schedule([=](lang::Args args,
                                               lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of softmax schedule is empty! Please check."));
    CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    PADDLE_ENFORCE_EQ(
        vec_ast.empty(),
        false,
        ::common::errors::NotFound("The vec_ast is empty! Please check."));
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    auto schedule_nv_hygon = [&] {
      if (output_shapes[0].size() > 1) {
        auto all_blocks = ir_sch.GetAllBlocks();
        PADDLE_ENFORCE_EQ(all_blocks.size(),
                          3,
                          ::common::errors::InvalidArgument(
                              "the size of all blocks should be 3, but got %d.",
                              all_blocks.size()));
        auto loops = ir_sch.GetLoops(all_blocks[2]);
        ir_sch.ComputeAt(all_blocks[1], loops.back());

        if (output_shapes[0][0] != 1) {
          ir_sch.SimpleComputeAt(all_blocks[0], loops[0]);
        }

        loops = ir_sch.GetLoops(all_blocks[2]);
        int loop_index = 1;
        if (output_shapes[0][0] == 1) loop_index--;
        PADDLE_ENFORCE_GE(loops.size(),
                          loop_index + 1,
                          ::common::errors::InvalidArgument(
                              "the size of loops should be greater "
                              "than or equal to %d, but got %d.",
                              loop_index + 1,
                              loops.size()));
        auto splited_loops = ir_sch.Split(loops[loop_index], {-1, 5});

        all_blocks = ir_sch.GetAllBlocks();
        loops = ir_sch.GetLoops(all_blocks[2]);
        ir_sch.Bind(loops[0], "blockIdx.x");
        ir_sch.Bind(loops[1], "threadIdx.x");
      }
      std::vector<CINNValue> res{
          CINNValue(ir_sch.GetModule().GetExprs().at(0))};
      *ret = CINNValuePack{res};
    };
    target.arch.Match(
        [&](common::UnknownArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::X86Arch) {
          pe::IRSoftmaxScheduleCPU(ir_sch, axis);
          std::vector<CINNValue> res{
              CINNValue(ir_sch.GetModule().GetExprs().at(0))};
          *ret = CINNValuePack{res};
        },
        [&](common::ARMArch) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) { schedule_nv_hygon(); },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          schedule_nv_hygon();
        });
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(
      softmax_compute, softmax_schedule, "strategy.softmax.x86", 1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForDropoutInfer(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  float dropout_prob = 0;
  std::string dropout_implementation = "downgrade_in_infer";
  if (attrs.attr_store.find("dropout_prob") != attrs.attr_store.end()) {
    dropout_prob = absl::get<float>(attrs.attr_store.at("dropout_prob"));
  }
  if (attrs.attr_store.find("dropout_implementation") !=
      attrs.attr_store.end()) {
    dropout_implementation =
        absl::get<std::string>(attrs.attr_store.at("dropout_implementation"));
  }

  framework::CINNCompute dropout_infer_compute([=](lang::Args args,
                                                   lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound("The input argument of dropout_infer "
                                   "compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.empty(),
        false,
        ::common::errors::NotFound("The input tensors of dropout_infer compute "
                                   "is empty! Please check."));
    Expr A_expr = pack_args[0];
    PADDLE_ENFORCE(A_expr.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! A_expr is not a tensor."));
    ir::Tensor A = A_expr.as_tensor_ref();

    PADDLE_ENFORCE_EQ(pack_args.size(),
                      2,
                      ::common::errors::InvalidArgument(
                          "the size of pack_args should be 2, but got %d.",
                          pack_args.size()));
    PADDLE_ENFORCE(pack_args[1].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[1] is not string."));
    std::string tensor_name = pack_args[1].operator std::string();

    auto out =
        pe::DropoutInfer(A, dropout_prob, dropout_implementation, tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(dropout_infer_compute,
                    GetInjectiveScheduleFunc(output_shapes, target),
                    "strategy.dropout_infer.x86",
                    1);

  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForSelect(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute select_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.empty(),
        false,
        ::common::errors::NotFound(
            "The input argument of select compute is empty! Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(
        pack_args.size(),
        3U,
        ::common::errors::InvalidArgument("the size of pack_args for select "
                                          "compute should be greater than or "
                                          "equal to 3, but got %d.",
                                          pack_args.size()));
    Expr condition = pack_args[0];
    Expr true_value = pack_args[1];
    Expr false_value = pack_args[2];
    PADDLE_ENFORCE(condition.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! condition is not a tensor."));
    PADDLE_ENFORCE(true_value.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! true_value is not a tensor."));
    PADDLE_ENFORCE(false_value.as_tensor(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! false_value is not a tensor."));

    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        4U,
        ::common::errors::InvalidArgument(
            "the size of pack_args for select compute should be 4, but got %d.",
            pack_args.size()));
    PADDLE_ENFORCE(pack_args[3].is_string(),
                   ::common::errors::InvalidArgument(
                       "Datatype error! pack_args[3] is not string."));
    std::string tensor_name = pack_args[3].operator std::string();

    auto out = pe::Select(condition.as_tensor_ref(),
                          true_value.as_tensor_ref(),
                          false_value.as_tensor_ref(),
                          tensor_name);

    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE_EQ(
      out_type.empty(),
      false,
      ::common::errors::NotFound("Out_type of select is empty! Please check."));
  strategy->AddImpl(select_compute,
                    GetInjectiveScheduleFunc(output_shapes, target, false),
                    "strategy.select.x86",
                    1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForSelectSymbolic(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<ir::Dim>> &output_shapes,
    const Target &target) {
  framework::CINNCompute select_compute([=](lang::Args args,
                                            lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(!args.empty(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The input argument of select compute is empty! "
                          "Please check."));
    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_GE(pack_args.size(),
                      3U,
                      ::common::errors::InvalidArgument(
                          "at least three input tensor for select compute."));
    Expr condition = pack_args[0];
    Expr true_value = pack_args[1];
    Expr false_value = pack_args[2];
    PADDLE_ENFORCE_NE(condition.as_tensor(),
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "The condation arg's type should be Tensor."));
    PADDLE_ENFORCE_NE(true_value.as_tensor(),
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "The true_value arg's type should be Tensor."));
    PADDLE_ENFORCE_NE(false_value.as_tensor(),
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "The false_value arg's type should be Tensor."));
    PADDLE_ENFORCE_EQ(pack_args.size(),
                      4U,
                      ::common::errors::InvalidArgument(
                          "The size of inputs must be equal to 4."));
    PADDLE_ENFORCE_EQ(pack_args[3].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The name arg's type should be string."));
    std::string tensor_name = pack_args[3].operator std::string();

    auto out = pe::Select(condition.as_tensor_ref(),
                          true_value.as_tensor_ref(),
                          false_value.as_tensor_ref(),
                          tensor_name);
    *ret = CINNValuePack{{CINNValue(out)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  PADDLE_ENFORCE_NE(out_type.size(),
                    0U,
                    ::common::errors::InvalidArgument(
                        "Out_type of select op is empty! Please check."));
  strategy->AddImpl(
      select_compute, lang::PackedFunc(), "strategy.select.x86", 1);
  return strategy;
}

std::shared_ptr<OpStrategy> StrategyForGradOp(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  PADDLE_THROW(::common::errors::Fatal(
      "Gradient operator will be decomposed into several primitive "
      "operators. Please Use Decomposer Program Pass."));
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(nn_ops) {
  CINN_REGISTER_OP(select)
      .describe("This operator implements the meta op 'Select'.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForSelect)
      .set_attr<cinn::hlir::framework::StrategyFunctionSymbolic>(
          "CINNStrategySymbolic", cinn::hlir::op::StrategyForSelectSymbolic)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kElementWise)
      .set_support_level(4);

  return true;
}
