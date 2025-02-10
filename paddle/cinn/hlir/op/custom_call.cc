// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/codegen_device_util.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/hlir/dialect/operator/ir/symbol_bindings.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/hlir/pe/elementwise.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/nn.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/hlir/pe/transform.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

#ifdef CINN_WITH_CUDNN
#include <cudnn.h>
#endif

namespace cinn {
namespace hlir {
namespace op {

using cinn::common::_CINNValuePack_;
using cinn::common::CINNValue;
using cinn::common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

using ArgsFunc =
    std::function<std::vector<ir::Expr>(const framework::NodeAttr &,
                                        const std::vector<ir::Tensor> &,
                                        const std::vector<std::vector<int>> &)>;

class CustomCallArgsFuncRegistry {
 public:
  static CustomCallArgsFuncRegistry &Global() {
    static CustomCallArgsFuncRegistry instance;
    return instance;
  }

  void Register(const std::string &custom_call,
                const cinn::common::Target &target,
                ArgsFunc args_func) {
    auto id = custom_call + "_" + target.arch_str();
    func_map_[id] = args_func;
  }

  ArgsFunc Lookup(const std::string &custom_call,
                  const cinn::common::Target &target) {
    auto id = custom_call + "_" + target.arch_str();
    PADDLE_ENFORCE_EQ(
        func_map_.count(id),
        true,
        ::common::errors::NotFound(
            "Can't find %s for target %s", custom_call, target.arch_str()));
    return func_map_[id];
  }

 private:
  CustomCallArgsFuncRegistry() {}
  std::unordered_map<std::string, ArgsFunc> func_map_;
};

std::shared_ptr<OpStrategy> StrategyForCustomCall(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute compute([=](lang::Args args, lang::RetValue *ret) {
    PADDLE_ENFORCE_EQ(
        args.size(),
        1UL,
        ::common::errors::InvalidArgument(
            "The size of 'args' should be 1, but received size %d.",
            args.size()));

    CINNValuePack pack_args = args[0];
    PADDLE_ENFORCE_EQ(
        pack_args.size(),
        2UL,
        ::common::errors::InvalidArgument(
            "The size of 'pack_args' should be 2, but received size %d.",
            pack_args.size()));
    PADDLE_ENFORCE_EQ(pack_args[0].is_string() && pack_args[1].is_string(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The pack_arg[0] and pack_arg[1] should be string."));
    std::string func_name = pack_args[0].operator std::string();
    std::string custom_call_api = pack_args[1].operator std::string();

    auto args_func =
        CustomCallArgsFuncRegistry::Global().Lookup(custom_call_api, target);
    // create call function.
    ir::Var kernel_args(KERNEL_ARGS, type_of<void *>());
    ir::Var kernel_args_num(KERNEL_ARGS_NUM, type_of<int>());

    auto args_list = args_func(attrs, inputs, output_shapes);
    std::vector<ir::Expr> host_args = {kernel_args, kernel_args_num};
    host_args.insert(host_args.end(), args_list.begin(), args_list.end());
    std::vector<ir::Argument> arguments = {
        ir::Argument(kernel_args, ir::Argument::IO::kOutput),
        ir::Argument(kernel_args_num, ir::Argument::IO::kInput)};
    // if target is nvgpu, add stream.
    target.arch.Match(
        [&](common::NVGPUArch) {
          ir::Var kernel_stream(KERNEL_STREAM, type_of<void *>());
          host_args.push_back(kernel_stream);
          arguments.emplace_back(kernel_stream, ir::Argument::IO::kOutput);
        },
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) {},
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          ir::Var kernel_stream(KERNEL_STREAM, type_of<void *>());
          host_args.push_back(kernel_stream);
          arguments.emplace_back(kernel_stream, ir::Argument::IO::kOutput);
        });
    auto call_extern_api = ir::Call::Make(Void(),
                                          custom_call_api,
                                          host_args,
                                          {},
                                          ir::CallType::Extern,
                                          ir::FunctionRef(),
                                          0);
    auto func =
        ir::_LoweredFunc_::Make(func_name, arguments, call_extern_api, {});

    VLOG(3) << func;
    *ret = CINNValuePack{{CINNValue(func)}};
  });

  framework::CINNSchedule schedule(
      [=](lang::Args args, lang::RetValue *ret) {});

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(compute, schedule, "strategy.custom_call.x86", 1);
  return strategy;
}

#ifdef CINN_WITH_CUDA
std::vector<ir::Expr> CustomCallArgsForCublas(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      2,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 2, but received size %d.",
          inputs.size()));
  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));
  PADDLE_ENFORCE_LE(inputs[0]->shape.size(),
                    4,
                    ::common::errors::InvalidArgument(
                        "The shape size of the first input should be less than "
                        "or equal to 4, but received size %d.",
                        inputs[0]->shape.size()));
  PADDLE_ENFORCE_LE(inputs[1]->shape.size(),
                    4,
                    ::common::errors::InvalidArgument(
                        "The shape size of the second input should be less "
                        "than or equal to 4, but received size %d.",
                        inputs[1]->shape.size()));

  const auto &attr_store = attrs.attr_store;
  bool trans_a = attr_store.count("trans_a")
                     ? absl::get<bool>(attr_store.at("trans_a"))
                     : false;
  bool trans_b = attr_store.count("trans_b")
                     ? absl::get<bool>(attr_store.at("trans_b"))
                     : false;
  bool trans_out = attr_store.count("trans_out")
                       ? absl::get<bool>(attr_store.at("trans_out"))
                       : false;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  int x_num_col_dims = attr_store.count("x_num_col_dims")
                           ? absl::get<int>(attr_store.at("x_num_col_dims"))
                           : 0;
  int y_num_col_dims = attr_store.count("y_num_col_dims")
                           ? absl::get<int>(attr_store.at("y_num_col_dims"))
                           : 0;
  bool is_infer = attr_store.count("is_infer")
                      ? absl::get<bool>(attr_store.at("is_infer"))
                      : false;
  PADDLE_ENFORCE_EQ(
      (x_num_col_dims == 0 && y_num_col_dims == 0) ||
          (x_num_col_dims > 0 && y_num_col_dims > 0),
      true,
      ::common::errors::InvalidArgument(
          "x_num_col_dims and y_num_cole_dims should both be 0 or positive"
          "now x_num_col_dims is %d and y_num_col_dims is %d",
          x_num_col_dims,
          y_num_col_dims));

  std::vector<ir::Expr> a_shape, b_shape;
  if (x_num_col_dims == 0 && y_num_col_dims == 0) {
    int a_rank = inputs[0]->shape.size();
    int b_rank = inputs[1]->shape.size();

    if (a_rank == 1) {
      a_shape.resize(4, ir::Expr(1));

      if (trans_a) {
        a_shape[2] = inputs[0]->shape[0];
      } else {
        a_shape[3] = inputs[0]->shape[0];
      }
    } else {
      a_shape = inputs[0]->shape;
      int insert_1_to_a = 4 - a_shape.size();
      for (int idx = 0; idx < insert_1_to_a; ++idx) {
        a_shape.insert(a_shape.begin(), ir::Expr(1));
      }
    }

    if (b_rank == 1) {
      b_shape.resize(4, ir::Expr(1));

      if (trans_b) {
        b_shape[3] = inputs[1]->shape[0];
      } else {
        b_shape[2] = inputs[1]->shape[0];
      }
    } else {
      b_shape = inputs[1]->shape;
      int insert_1_to_b = 4 - b_shape.size();
      for (int idx = 0; idx < insert_1_to_b; ++idx) {
        b_shape.insert(b_shape.begin(), ir::Expr(1));
      }
    }
  } else if (x_num_col_dims > 0 && y_num_col_dims > 0) {
    // input a shape.
    a_shape = {Expr(1), Expr(1)};
    int a_height = 1;
    int a_width = 1;
    for (int idx = 0; idx < x_num_col_dims; ++idx) {
      a_height *= inputs[0]->shape[idx].as_int32();
    }
    for (int idx = x_num_col_dims; idx < inputs[0]->shape.size(); ++idx) {
      a_width *= inputs[0]->shape[idx].as_int32();
    }
    a_shape.emplace_back(a_height);
    a_shape.emplace_back(a_width);

    // input b shape.
    b_shape = {Expr(1), Expr(1)};
    int b_height = 1;
    int b_width = 1;
    for (int idx = 0; idx < y_num_col_dims; ++idx) {
      b_height *= inputs[1]->shape[idx].as_int32();
    }
    for (int idx = y_num_col_dims; idx < inputs[1]->shape.size(); ++idx) {
      b_width *= inputs[1]->shape[idx].as_int32();
    }
    b_shape.emplace_back(b_height);
    b_shape.emplace_back(b_width);

    if (is_infer) {
      PADDLE_ENFORCE_EQ(a_width,
                        b_width,
                        ::common::errors::InvalidArgument(
                            "The K dimension of mul should be equal! Received: "
                            "a_width = %d, b_width = %d.",
                            a_width,
                            b_width));

      trans_b = true;
    } else {
      PADDLE_ENFORCE_EQ(a_width,
                        b_height,
                        ::common::errors::InvalidArgument(
                            "The K dimension of mul should be equal! Received: "
                            "a_width = %d, b_height = %d.",
                            a_width,
                            b_height));
    }
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("Unknown Matmul Setting!"));
  }

  PADDLE_ENFORCE_EQ(
      a_shape.size(),
      4,
      ::common::errors::InvalidArgument(
          "The size of 'a_shape' should be 4, but received size %d.",
          a_shape.size()));

  PADDLE_ENFORCE_EQ(
      b_shape.size(),
      4,
      ::common::errors::InvalidArgument(
          "The size of 'b_shape' should be 4, but received size %d.",
          b_shape.size()));
  // func args
  std::vector<ir::Expr> args = {ir::Expr(trans_a),
                                ir::Expr(trans_b),
                                ir::Expr(trans_out),
                                ir::Expr(alpha),
                                ir::Expr(beta)};
  args.insert(args.end(), a_shape.begin(), a_shape.end());
  args.insert(args.end(), b_shape.begin(), b_shape.end());
  return args;
}

std::vector<ir::Expr> CustomCallArgsForBatchedCublas(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_GT(
      inputs.size(),
      2,
      ::common::errors::InvalidArgument("The size of 'inputs' should be "
                                        "greater than 2, but received size %d.",
                                        inputs.size()));

  PADDLE_ENFORCE_GT(output_shapes.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The size of 'output_shapes' should "
                        "be greater than 1, but received size %d.",
                        output_shapes.size()));

  PADDLE_ENFORCE_EQ(inputs.size() - 1,
                    output_shapes.size(),
                    ::common::errors::InvalidArgument(
                        "The size of 'inputs' minus 1 should "
                        "be equal to the size of 'output_shapes'. "
                        "Received: inputs.size() - 1 = %d, "
                        "output_shapes.size() = %d.",
                        inputs.size() - 1,
                        output_shapes.size()));

  const auto &attr_store = attrs.attr_store;
  bool trans_a = attr_store.count("trans_a")
                     ? absl::get<bool>(attr_store.at("trans_a"))
                     : false;
  bool trans_b = attr_store.count("trans_b")
                     ? absl::get<bool>(attr_store.at("trans_b"))
                     : false;
  bool trans_out = attr_store.count("trans_out")
                       ? absl::get<bool>(attr_store.at("trans_out"))
                       : false;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  int x_num_col_dims = attr_store.count("x_num_col_dims")
                           ? absl::get<int>(attr_store.at("x_num_col_dims"))
                           : 0;
  int y_num_col_dims = attr_store.count("y_num_col_dims")
                           ? absl::get<int>(attr_store.at("y_num_col_dims"))
                           : 0;
  bool is_infer = attr_store.count("is_infer")
                      ? absl::get<bool>(attr_store.at("is_infer"))
                      : false;
  PADDLE_ENFORCE_EQ((x_num_col_dims == 0 && y_num_col_dims == 0) ||
                        (x_num_col_dims > 0 && y_num_col_dims > 0),
                    true,
                    ::common::errors::InvalidArgument(
                        "The values of 'x_num_col_dims' and "
                        "'y_num_col_dims' must either both be 0 "
                        "or both be greater than 0. Received: "
                        "x_num_col_dims = %d, y_num_col_dims = %d.",
                        x_num_col_dims,
                        y_num_col_dims));

  ir::Tensor left, right;
  PADDLE_ENFORCE_EQ((x_num_col_dims == 0 && y_num_col_dims == 0) ||
                        (x_num_col_dims > 0 && y_num_col_dims > 0),
                    true,
                    ::common::errors::InvalidArgument(
                        "The values of 'x_num_col_dims' and "
                        "'y_num_col_dims' must either both be 0 "
                        "or both be greater than 0. Received: "
                        "x_num_col_dims = %d, y_num_col_dims = %d.",
                        x_num_col_dims,
                        y_num_col_dims));
  if (absl::get<std::string>(attr_store.at("side")) == "left") {
    left = inputs[0];
    right = inputs[1];
  } else {
    left = inputs[1];
    right = inputs[0];
  }

  std::vector<ir::Expr> a_shape, b_shape;
  if (x_num_col_dims == 0 && y_num_col_dims == 0) {
    int a_rank = left->shape.size();
    int b_rank = right->shape.size();

    if (a_rank == 1) {
      a_shape.resize(4, ir::Expr(1));

      if (trans_a) {
        a_shape[2] = left->shape[0];
      } else {
        a_shape[3] = left->shape[0];
      }
    } else {
      a_shape = left->shape;
      int insert_1_to_a = 4 - a_shape.size();
      for (int idx = 0; idx < insert_1_to_a; ++idx) {
        a_shape.insert(a_shape.begin(), ir::Expr(1));
      }
    }

    if (b_rank == 1) {
      b_shape.resize(4, ir::Expr(1));

      if (trans_b) {
        b_shape[3] = right->shape[0];
      } else {
        b_shape[2] = right->shape[0];
      }
    } else {
      b_shape = right->shape;
      int insert_1_to_b = 4 - b_shape.size();
      for (int idx = 0; idx < insert_1_to_b; ++idx) {
        b_shape.insert(b_shape.begin(), ir::Expr(1));
      }
    }
  } else if (x_num_col_dims > 0 && y_num_col_dims > 0) {
    // input a shape.
    a_shape = {Expr(1), Expr(1)};
    int a_height = 1;
    int a_width = 1;
    for (int idx = 0; idx < x_num_col_dims; ++idx) {
      a_height *= left->shape[idx].as_int32();
    }
    for (int idx = x_num_col_dims; idx < left->shape.size(); ++idx) {
      a_width *= left->shape[idx].as_int32();
    }
    a_shape.emplace_back(a_height);
    a_shape.emplace_back(a_width);

    // input b shape.
    b_shape = {Expr(1), Expr(1)};
    int b_height = 1;
    int b_width = 1;
    for (int idx = 0; idx < y_num_col_dims; ++idx) {
      b_height *= right->shape[idx].as_int32();
    }
    for (int idx = y_num_col_dims; idx < right->shape.size(); ++idx) {
      b_width *= right->shape[idx].as_int32();
    }
    b_shape.emplace_back(b_height);
    b_shape.emplace_back(b_width);

    if (is_infer) {
      PADDLE_ENFORCE_EQ(a_width,
                        b_width,
                        ::common::errors::InvalidArgument(
                            "The K dimension of mul should be equal! "
                            "Received: a_width = %d, b_width = %d.",
                            a_width,
                            b_width));
      trans_b = true;
    } else {
      PADDLE_ENFORCE_EQ(a_width,
                        b_height,
                        ::common::errors::InvalidArgument(
                            "The K dimension of mul should be equal! "
                            "Received: a_width = %d, b_height = %d.",
                            a_width,
                            b_height));
    }
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument("Unknown Matmul Setting!"));
  }

  PADDLE_ENFORCE_EQ(
      a_shape.size(),
      4,
      ::common::errors::InvalidArgument(
          "The size of 'a_shape' should be 4, but received size %d.",
          a_shape.size()));

  PADDLE_ENFORCE_EQ(
      b_shape.size(),
      4,
      ::common::errors::InvalidArgument(
          "The size of 'b_shape' should be 4, but received size %d.",
          b_shape.size()));

  // func args
  std::vector<ir::Expr> args = {
      absl::get<std::string>(attr_store.at("side")) == "left" ? ir::Expr(0)
                                                              : ir::Expr(1),
      ir::Expr(trans_a),
      ir::Expr(trans_b),
      ir::Expr(trans_out),
      ir::Expr(alpha),
      ir::Expr(beta)};
  args.insert(args.end(), a_shape.begin(), a_shape.end());
  args.insert(args.end(), b_shape.begin(), b_shape.end());
  return args;
}

#endif

#ifdef CINN_WITH_CUDNN
std::vector<ir::Expr> CustomCallArgsForCudnnConvForward(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      2UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 2, but received size %d.",
          inputs.size()));
  /* PADDLE_ENFORCE_EQ(
       output_shapes.size(), 1UL,
       ::common::errors::InvalidArgument(
           "The size of 'output_shapes' should be 1, but received size %d.",
           output_shapes.size())); */
  const auto &attr_store = attrs.attr_store;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  PADDLE_ENFORCE_EQ(
      attr_store.count("padding"),
      true,
      ::common::errors::NotFound(
          "The CudnnConvForward custom_call must has attribute \"padding\""));
  auto padding = absl::get<std::vector<int>>(attr_store.at("padding"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("stride"),
      true,
      ::common::errors::NotFound(
          "The CudnnConvForward custom_call must has attribute \"stride\""));
  auto stride = absl::get<std::vector<int>>(attr_store.at("stride"));
  auto dilation = attr_store.count("dilation")
                      ? absl::get<std::vector<int>>(attr_store.at("dilation"))
                      : std::vector<int>({1, 1});
  std::string data_format =
      attr_store.count("data_format")
          ? absl::get<std::string>(attr_store.at("data_format"))
          : "NCHW";
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  int groups =
      attr_store.count("groups") ? absl::get<int>(attr_store.at("groups")) : 1;
  cudnnTensorFormat_t format =
      data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  std::vector<Expr> input = inputs[0]->shape;
  std::vector<Expr> filter = inputs[1]->shape;
  std::vector<Expr> output = {};
  std::transform(output_shapes[0].begin(),
                 output_shapes[0].end(),
                 std::back_inserter(output),
                 [](const int dim) { return ir::Expr(dim); });
  // if format is nhwc
  if (format == CUDNN_TENSOR_NHWC) {
    input = {input[0], input[3], input[1], input[2]};
    filter = {filter[0], filter[3], filter[1], filter[2]};
    output = {output[0], output[3], output[1], output[2]};
  }

  std::vector<ir::Expr> args = {
      ir::Expr(static_cast<int>(format)), ir::Expr(alpha), ir::Expr(beta)};
  args.insert(args.end(), input.begin(), input.end());
  args.insert(args.end(), filter.begin(), filter.end());
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.push_back(ir::Expr(dilation[0]));
  args.push_back(ir::Expr(dilation[1]));
  args.push_back(ir::Expr(groups));
  args.insert(args.end(), output.begin(), output.end());

  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnConvBackwardData(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      2UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 2, but received size %d.",
          inputs.size()));

  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  PADDLE_ENFORCE_EQ(
      attr_store.count("padding"),
      true,
      ::common::errors::NotFound("The CudnnConvBackwardData custom_call"
                                 "must has attribute \"padding\""));
  auto padding = absl::get<std::vector<int>>(attr_store.at("padding"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("stride"),
      true,
      ::common::errors::NotFound("The CudnnConvBackwardData custom_call"
                                 "must has attribute \"stride\""));
  auto stride = absl::get<std::vector<int>>(attr_store.at("stride"));
  auto dilation = attr_store.count("dilation")
                      ? absl::get<std::vector<int>>(attr_store.at("dilation"))
                      : std::vector<int>({1, 1});
  std::string data_format =
      attr_store.count("data_format")
          ? absl::get<std::string>(attr_store.at("data_format"))
          : "NCHW";
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  int groups =
      attr_store.count("groups") ? absl::get<int>(attr_store.at("groups")) : 1;
  cudnnTensorFormat_t format =
      data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  std::vector<Expr> input = {};
  std::transform(output_shapes[0].begin(),
                 output_shapes[0].end(),
                 std::back_inserter(input),
                 [](const int dim) { return ir::Expr(dim); });
  std::vector<Expr> filter = inputs[0]->shape;
  std::vector<Expr> output = inputs[1]->shape;
  // if format is nhwc
  if (format == CUDNN_TENSOR_NHWC) {
    input = {input[0], input[3], input[1], input[2]};
    filter = {filter[0], filter[3], filter[1], filter[2]};
    output = {output[0], output[3], output[1], output[2]};
  }

  std::vector<ir::Expr> args = {
      ir::Expr(static_cast<int>(format)), ir::Expr(alpha), ir::Expr(beta)};
  args.insert(args.end(), input.begin(), input.end());
  args.insert(args.end(), filter.begin(), filter.end());
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.push_back(ir::Expr(dilation[0]));
  args.push_back(ir::Expr(dilation[1]));
  args.push_back(ir::Expr(groups));
  args.insert(args.end(), output.begin(), output.end());
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnConvBackwardFilter(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      2UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 2, but received size %d.",
          inputs.size()));

  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  PADDLE_ENFORCE_EQ(
      attr_store.count("padding"),
      true,
      ::common::errors::NotFound("The CudnnConvBackwardFilter custom_call"
                                 "must has attribute \"padding\""));
  auto padding = absl::get<std::vector<int>>(attr_store.at("padding"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("stride"),
      true,
      ::common::errors::NotFound("The CudnnConvBackwardFilter custom_call"
                                 "must has attribute \"stride\""));
  auto stride = absl::get<std::vector<int>>(attr_store.at("stride"));
  auto dilation = attr_store.count("dilation")
                      ? absl::get<std::vector<int>>(attr_store.at("dilation"))
                      : std::vector<int>({1, 1});
  std::string data_format =
      attr_store.count("data_format")
          ? absl::get<std::string>(attr_store.at("data_format"))
          : "NCHW";
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  int groups =
      attr_store.count("groups") ? absl::get<int>(attr_store.at("groups")) : 1;

  cudnnTensorFormat_t format =
      data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  std::vector<Expr> input = inputs[0]->shape;
  std::vector<Expr> filter = {};
  std::transform(output_shapes[0].begin(),
                 output_shapes[0].end(),
                 std::back_inserter(filter),
                 [](const int dim) { return ir::Expr(dim); });
  std::vector<Expr> output = inputs[1]->shape;
  // if format is nhwc
  if (format == CUDNN_TENSOR_NHWC) {
    input = {input[0], input[3], input[1], input[2]};
    filter = {filter[0], filter[3], filter[1], filter[2]};
    output = {output[0], output[3], output[1], output[2]};
  }

  std::vector<ir::Expr> args = {
      ir::Expr(static_cast<int>(format)), ir::Expr(alpha), ir::Expr(beta)};
  args.insert(args.end(), input.begin(), input.end());
  args.insert(args.end(), filter.begin(), filter.end());
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.push_back(ir::Expr(dilation[0]));
  args.push_back(ir::Expr(dilation[1]));
  args.push_back(ir::Expr(groups));
  args.insert(args.end(), output.begin(), output.end());
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnPoolForward(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 1, but received size %d.",
          inputs.size()));

  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  PADDLE_ENFORCE_EQ(
      attr_store.count("kernel_size"),
      true,
      ::common::errors::NotFound("The CudnnPoolForward custom_call"
                                 "must has attribute \"kernel_size\""));
  auto kernel = absl::get<std::vector<int>>(attr_store.at("kernel_size"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("padding_size"),
      true,
      ::common::errors::NotFound("The CudnnPoolForward custom_call"
                                 "must has attribute \"padding_size\""));
  auto padding = absl::get<std::vector<int>>(attr_store.at("padding_size"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("stride_size"),
      true,
      ::common::errors::NotFound("The CudnnPoolForward custom_call"
                                 "must has attribute \"stride_size\""));
  auto stride = absl::get<std::vector<int>>(attr_store.at("stride_size"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("pool_type"),
      true,
      ::common::errors::NotFound("The CudnnPoolForward custom_call"
                                 "must has attribute \"pool_type\""));
  auto pool_type = absl::get<std::string>(attr_store.at("pool_type"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("data_format"),
      true,
      ::common::errors::NotFound("The CudnnPoolForward custom_call"
                                 "must has attribute \"data_format\""));
  std::string data_format =
      absl::get<std::string>(attr_store.at("data_format"));

  bool exclusive = attr_store.count("exclusive")
                       ? absl::get<bool>(attrs.attr_store.at("exclusive"))
                       : true;
  cudnnPoolingMode_t mode =
      pool_type == "max"
          ? CUDNN_POOLING_MAX
          : (exclusive ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                       : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
  cudnnTensorFormat_t format =
      data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  std::vector<Expr> input = inputs[0]->shape;
  std::vector<Expr> output;
  std::transform(output_shapes[0].begin(),
                 output_shapes[0].end(),
                 std::back_inserter(output),
                 [](const int dim) { return ir::Expr(dim); });
  // if format is nhwc
  if (format == CUDNN_TENSOR_NHWC) {
    input = {input[0], input[3], input[1], input[2]};
    output = {output[0], output[3], output[1], output[2]};
  }

  std::vector<ir::Expr> args = {ir::Expr(static_cast<int>(mode)),
                                ir::Expr(static_cast<int>(format)),
                                ir::Expr(alpha),
                                ir::Expr(beta)};
  args.insert(args.end(), input.begin(), input.end());
  args.push_back(ir::Expr(kernel[0]));
  args.push_back(ir::Expr(kernel[1]));
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.insert(args.end(), output.begin(), output.end());
  return args;
}

std::vector<ir::Expr> CustomCallArgsForCudnnPoolBackward(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      3UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 3, but received size %d.",
          inputs.size()));

  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  PADDLE_ENFORCE_EQ(
      attr_store.count("kernel_size"),
      true,
      ::common::errors::NotFound("The CudnnPoolBackward custom_call"
                                 "must has attribute \"kernel_size\""));
  auto kernel = absl::get<std::vector<int>>(attr_store.at("kernel_size"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("padding_size"),
      true,
      ::common::errors::NotFound("The CudnnPoolBackward custom_call"
                                 "must has attribute \"padding_size\""));
  auto padding = absl::get<std::vector<int>>(attr_store.at("padding_size"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("stride_size"),
      true,
      ::common::errors::NotFound("The CudnnPoolBackward custom_call"
                                 "must has attribute \"stride_size\""));
  auto stride = absl::get<std::vector<int>>(attr_store.at("stride_size"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("pool_type"),
      true,
      ::common::errors::NotFound("The CudnnPoolBackward custom_call"
                                 "must has attribute \"pool_type\""));
  auto pool_type = absl::get<std::string>(attr_store.at("pool_type"));
  PADDLE_ENFORCE_EQ(
      attr_store.count("data_format"),
      true,
      ::common::errors::NotFound("The CudnnPoolBackward custom_call"
                                 "must has attribute \"data_format\""));
  std::string data_format =
      absl::get<std::string>(attrs.attr_store.at("data_format"));

  bool exclusive = attr_store.count("exclusive")
                       ? absl::get<bool>(attrs.attr_store.at("exclusive"))
                       : true;
  cudnnPoolingMode_t mode =
      pool_type == "max"
          ? CUDNN_POOLING_MAX
          : (exclusive ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                       : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
  cudnnTensorFormat_t format =
      data_format == "NCHW" ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  std::vector<Expr> input = inputs[0]->shape;   // 'x'
  std::vector<Expr> output = inputs[1]->shape;  // 'y'
  // if format is nhwc
  if (format == CUDNN_TENSOR_NHWC) {
    input = {input[0], input[3], input[1], input[2]};
    output = {output[0], output[3], output[1], output[2]};
  }

  std::vector<ir::Expr> args = {ir::Expr(static_cast<int>(mode)),
                                ir::Expr(static_cast<int>(format)),
                                ir::Expr(alpha),
                                ir::Expr(beta)};
  args.insert(args.end(), input.begin(), input.end());
  args.push_back(ir::Expr(kernel[0]));
  args.push_back(ir::Expr(kernel[1]));
  args.push_back(ir::Expr(padding[0]));
  args.push_back(ir::Expr(padding[1]));
  args.push_back(ir::Expr(stride[0]));
  args.push_back(ir::Expr(stride[1]));
  args.insert(args.end(), output.begin(), output.end());

  return args;
}
#endif

std::vector<ir::Expr> CustomCallArgsForAssertTrue(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 1, but received size %d.",
          inputs.size()));

  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;
  PADDLE_ENFORCE_EQ(
      attr_store.count("msg"),
      true,
      ::common::errors::NotFound(
          "The assert_true custom_call must has attribute \"msg\""));
  // TODO(thisjiang): change type from 'int' to 'std::string' when custom call
  // support 'std::string' type
  int msg = absl::get<int>(attr_store.at("msg"));
  bool only_warning = attr_store.count("only_warning")
                          ? absl::get<bool>(attrs.attr_store.at("only_warning"))
                          : false;

  std::vector<ir::Expr> args = {ir::Expr(msg), ir::Expr(only_warning)};

  return args;
}

std::vector<ir::Expr> CustomCallArgsForGaussianRandom(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;

  float mean = attr_store.count("mean")
                   ? absl::get<float>(attrs.attr_store.at("mean"))
                   : 0.0f;
  float std = attr_store.count("std")
                  ? absl::get<float>(attrs.attr_store.at("std"))
                  : 1.0f;
  int seed = attr_store.count("seed")
                 ? absl::get<int>(attrs.attr_store.at("seed"))
                 : 0;

  std::vector<ir::Expr> args = {ir::Expr(mean), ir::Expr(std), ir::Expr(seed)};

  return args;
}

std::vector<ir::Expr> CustomCallArgsForUniformRandom(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;

  float min = attr_store.count("min")
                  ? absl::get<float>(attrs.attr_store.at("min"))
                  : -1.0f;
  float max = attr_store.count("max")
                  ? absl::get<float>(attrs.attr_store.at("max"))
                  : 1.0f;
  int seed = attr_store.count("seed")
                 ? absl::get<int>(attrs.attr_store.at("seed"))
                 : 0;

  PADDLE_ENFORCE_GE(max,
                    min,
                    ::common::errors::InvalidArgument(
                        "Arg 'max' must be greater than or equal to "
                        "'min'. Received: max = %d, min = %d.",
                        max,
                        min));

  std::vector<ir::Expr> args = {ir::Expr(min), ir::Expr(max), ir::Expr(seed)};

  return args;
}

std::vector<ir::Expr> CustomCallArgsForRandInt(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'output_shapes' should be 1, but received size %d.",
          output_shapes.size()));

  const auto &attr_store = attrs.attr_store;

  int seed = attr_store.count("seed")
                 ? absl::get<int>(attrs.attr_store.at("seed"))
                 : 0;

  std::vector<ir::Expr> args = {ir::Expr(seed)};

  return args;
}

std::vector<ir::Expr> CustomCallArgsForCholesky(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      1UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 1, but received size %d.",
          inputs.size()));
  const auto &attr_store = attrs.attr_store;
  PADDLE_ENFORCE_EQ(
      attr_store.count("upper"),
      true,
      ::common::errors::NotFound(
          "The cholesky custom_call must has attribute \"upper\""));

  ir::Tensor x = inputs.front();
  int ndim = static_cast<int>(x->shape.size());
  int batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= x->shape[i].as_int32();
  }
  int m = x->shape[ndim - 1].as_int32();

  auto upper = absl::get<bool>(attrs.attr_store.at("upper"));

  std::vector<ir::Expr> args = {
      ir::Expr(batch_size), ir::Expr(m), ir::Expr(upper)};

  return args;
}

std::vector<ir::Expr> CustomCallArgsForTriangularSolve(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(
      inputs.size(),
      2UL,
      ::common::errors::InvalidArgument(
          "The size of 'inputs' should be 2, but received size %d.",
          inputs.size()));
  const auto &attr_store = attrs.attr_store;
  PADDLE_ENFORCE_EQ(
      attr_store.count("left_side"),
      true,
      ::common::errors::NotFound("The TriangularSolve custom_call"
                                 "must has attribute \"left_side\""));
  PADDLE_ENFORCE_EQ(attr_store.count("upper"),
                    true,
                    ::common::errors::NotFound("The TriangularSolve custom_call"
                                               "must has attribute \"upper\""));
  PADDLE_ENFORCE_EQ(
      attr_store.count("transpose_a"),
      true,
      ::common::errors::NotFound("The TriangularSolve custom_call"
                                 "must has attribute \"transpose_a\""));
  PADDLE_ENFORCE_EQ(
      attr_store.count("unit_diagonal"),
      true,
      ::common::errors::NotFound("The TriangularSolve custom_call"
                                 "must has attribute \"unit_diagonal\""));

  ir::Tensor a = inputs[0];
  ir::Tensor b = inputs[1];
  int a_ndim = static_cast<int>(a->shape.size());
  int b_ndim = static_cast<int>(b->shape.size());
  int batch_size = 1;
  for (int i = 0; i < a_ndim - 2; i++) {
    batch_size *= a->shape[i].as_int32();
  }

  auto left_side = absl::get<bool>(attrs.attr_store.at("left_side"));
  auto upper = absl::get<bool>(attrs.attr_store.at("upper"));
  auto transpose_a = absl::get<bool>(attrs.attr_store.at("transpose_a"));
  auto unit_diagonal = absl::get<bool>(attrs.attr_store.at("unit_diagonal"));

  int m = a->shape[a_ndim - 1].as_int32();
  int k = left_side ? b->shape[b_ndim - 1].as_int32()
                    : b->shape[b_ndim - 2].as_int32();

  std::vector<ir::Expr> args = {ir::Expr(batch_size),
                                ir::Expr(m),
                                ir::Expr(k),
                                ir::Expr(left_side),
                                ir::Expr(upper),
                                ir::Expr(transpose_a),
                                ir::Expr(unit_diagonal)};

  return args;
}

std::vector<ir::Expr> CustomCallArgsForMemset(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  const auto &attr_store = attrs.attr_store;
  PADDLE_ENFORCE_EQ(attr_store.count("value"),
                    true,
                    ::common::errors::NotFound(
                        "The memset custom_call must has attribute \"value\""));
  PADDLE_ENFORCE_EQ(inputs.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The memset custom_call should not has any input"));
  PADDLE_ENFORCE_EQ(output_shapes.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The memset custom_call should only have "
                        "one output, but received size %d.",
                        output_shapes.size()));

  struct Visitor {
    int *scalar_;
    explicit Visitor(int *scalar) : scalar_(scalar) {}
    void operator()(float v) { *scalar_ = *reinterpret_cast<int *>(&v); }
    void operator()(double v) {
      auto tmp = static_cast<float>(v);
      *scalar_ = *reinterpret_cast<int *>(&tmp);
    }
    void operator()(int32_t v) { *scalar_ = v; }
    void operator()(int64_t v) { *scalar_ = static_cast<int>(v); }
    void operator()(bool v) { *scalar_ = v ? 0xFFFFFFFF : 0; }

#define EXPAND_MEMSET_TYPE_UNSUPPORTED(TYPE)                          \
  void operator()(const TYPE &) {                                     \
    std::stringstream ss;                                             \
    ss << "The type of \"value\" of memset custom_call not support: " \
       << #TYPE;                                                      \
    PADDLE_THROW(::common::errors::InvalidArgument(ss.str()));        \
  }

    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::string)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<int>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<int64_t>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<float>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<double>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<bool>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<std::string>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<symbol::DimExpr>)
    EXPAND_MEMSET_TYPE_UNSUPPORTED(std::vector<cinn::dialect::SymbolBinding>)
#undef EXPAND_MEMSET_TYPE_UNSUPPORTED
  };

  int value = 0;
  const auto &value_attr = attr_store.at("value");
  absl::visit(Visitor(&value), value_attr);
  // can support memset non-0 ?
  PADDLE_ENFORCE_EQ(
      value,
      0,
      ::common::errors::InvalidArgument(
          "Now memset only supports value 0, but received value %d.", value));

  size_t count = 1;
  for (auto dim : output_shapes[0]) {
    count *= dim;
  }

  const auto &dtype =
      cinn::common::Str2Type(absl::get<std::string>(attr_store.at("dtype")));
  count *= dtype.bytes();
  VLOG(4) << "call memset custom_call with value="
          << utils::Attribute2String(value_attr) << " (" << value
          << "), count=" << count;

  return {Expr(value), Expr(count)};
}

std::vector<ir::Expr> CustomCallArgsForMemcpy(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  PADDLE_ENFORCE_EQ(inputs.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The memcpy custom_call should only have "
                        "one input, but received size %d.",
                        inputs.size()));

  PADDLE_ENFORCE_EQ(output_shapes.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The memcpy custom_call should only have "
                        "one output, but received size %d.",
                        output_shapes.size()));

  const auto &input_shape = ToPodVector<int>(inputs[0]->shape);

  size_t count = 1;
  for (auto dim : input_shape) {
    count *= dim;
  }

  const auto &dtype = inputs[0]->type();
  count *= dtype.bytes();

  return {Expr(count)};
}

bool RegisterCustomCallArgsFunc() {
#ifdef CINN_WITH_CUDA
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cublas",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCublas);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_gaussian_random",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForGaussianRandom);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_uniform_random",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForUniformRandom);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_randint",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForRandInt);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cholesky_nvgpu",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCholesky);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_batched_cublas",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForBatchedCublas);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_triangular_solve_nvgpu",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForTriangularSolve);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cuda_memset",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForMemset);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cuda_memcpy",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForMemcpy);
#endif

#ifdef CINN_WITH_CUDNN
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cudnn_conv2d_forward",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCudnnConvForward);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cudnn_conv2d_backward_data",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCudnnConvBackwardData);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cudnn_conv2d_backward_filter",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCudnnConvBackwardFilter);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cudnn_pool2d_forward",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCudnnPoolForward);
  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cudnn_pool2d_backward",
      cinn::common::DefaultNVGPUTarget(),
      CustomCallArgsForCudnnPoolBackward);
#endif

#ifdef CINN_WITH_DNNL

#endif

#ifdef CINN_WITH_MKL_CBLAS

  CustomCallArgsFuncRegistry::Global().Register(
      "cinn_call_cholesky_host",
      cinn::common::DefaultHostTarget(),
      CustomCallArgsForCholesky);

#endif

  return true;
}

static bool registry_custom_call_list_func = RegisterCustomCallArgsFunc();
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(custom_call_op) {
  CINN_REGISTER_OP(custom_call)
      .describe("This operator implements the call of extern api!")
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForCustomCall)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible);

  return true;
}
