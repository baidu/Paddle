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

#include "paddle/cinn/backends/codegen_device_util.h"

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace backends {

std::tuple<ir::Module, ir::Module> SplitDeviceAndHostModule(ir::Module module) {
  detail::CollectBucketStrategyHostFunctionVisitor visitor(module->name);
  return visitor(module);
}

ir::Module CreateSwitchWithBroadcastConditionModule(
    const std::vector<ir::Expr> &broadcast_conditions,
    const std::vector<std::string> &case_func_names,
    const std::string &wrapper_func_name,
    const std::unordered_map<int, ir::Var> &symbolic_shape_var_index) {
  ir::Var kernel_args(KERNEL_ARGS, type_of<void *>());
  ir::Var kernel_args_num(KERNEL_ARGS_NUM, type_of<int>());
  ir::Var kernel_stream(KERNEL_STREAM, type_of<void *>());
  ir::Var tensor_shape_args(TENSOR_SHAPE_ARGS, type_of<int64_t **>());
  std::vector<ir::Argument> host_func_arguments = {
      ir::Argument(kernel_args, ir::Argument::IO::kOutput),
      ir::Argument(kernel_args_num, ir::Argument::IO::kInput),
      ir::Argument(kernel_stream, ir::Argument::IO::kOutput)};
  std::vector<ir::Argument> infer_shape_func_arguments = {
      ir::Argument(kernel_args, ir::Argument::IO::kOutput),
      ir::Argument(kernel_args_num, ir::Argument::IO::kInput),
      ir::Argument(tensor_shape_args, ir::Argument::IO::kOutput)};

  const auto &symbolic_arg_define = [&]() -> std::vector<ir::Expr> {
    std::vector<ir::Expr> arg_defs;
    for (const auto &item : symbolic_shape_var_index) {
      ir::Expr call_get_value_in_kernel_args =
          ir::Call::Make(Int(64),
                         runtime::intrinsic::get_value_in_cuda_kernel_args,
                         {kernel_args, ir::Expr(item.first)},
                         {},
                         ir::CallType::Extern,
                         ir::FunctionRef(),
                         0);
      ir::Expr let_symbol = ir::Expr(item.second);
      let_symbol->set_type(type_of<int64_t>());
      ir::Expr stmt = ir::Let::Make(let_symbol, call_get_value_in_kernel_args);
      arg_defs.push_back(stmt);
    }
    return arg_defs;
  }();

  const auto &CreateSwitchFunction =
      [&](std::vector<ir::Argument> func_arguments,
          const std::vector<ir::Expr> &read_args,
          std::string name_extend) -> ir::LoweredFunc {
    std::vector<ir::Expr> body_stmts(symbolic_arg_define);
    for (int i = 0; i < broadcast_conditions.size(); ++i) {
      ir::Expr callee = ir::Call::Make(Void(),
                                       case_func_names[i] + name_extend,
                                       read_args,
                                       {},
                                       ir::CallType::Extern,
                                       ir::FunctionRef(),
                                       0);
      if (i == 0) {
        body_stmts.emplace_back(
            ir::IfThenElse::Make(broadcast_conditions[i], callee));
      } else {
        auto false_expr = body_stmts.back();
        body_stmts.pop_back();
        body_stmts.emplace_back(
            ir::IfThenElse::Make(broadcast_conditions[i], callee, false_expr));
      }
    }
    ir::LoweredFunc caller =
        ir::_LoweredFunc_::Make(wrapper_func_name + name_extend,
                                func_arguments,
                                ir::Block::Make(body_stmts),
                                {});
    return caller;
  };

  ir::Module::Builder module_builder(wrapper_func_name + "_switch",
                                     cinn::common::DefaultHostTarget());
  ir::LoweredFunc host_func_caller = CreateSwitchFunction(
      host_func_arguments, {kernel_args, kernel_args_num, kernel_stream}, "");
  ir::LoweredFunc infer_shape_func_caller =
      CreateSwitchFunction(infer_shape_func_arguments,
                           {kernel_args, kernel_args_num, tensor_shape_args},
                           "_infer_shape");
  module_builder.AddFunctionWithoutOptim(host_func_caller);
  module_builder.AddFunctionWithoutOptim(infer_shape_func_caller);
  // no need cx86 func
  ir::LoweredFunc cx86_func_caller =
      ir::_LoweredFunc_::Make(wrapper_func_name + "_CX86",
                              host_func_arguments,
                              ir::Block::Make({}),
                              {});
  module_builder.AddFunctionWithoutOptim(cx86_func_caller);
  return module_builder.Build();
}

struct PredicatePrinter : public ir::IrPrinter {
  explicit PredicatePrinter(std::ostream &os) : ir::IrPrinter(os) {}

 private:
  void Visit(const ir::Add *x) { PrintBinaryOp("ADD", x); }
  void Visit(const ir::Sub *x) { PrintBinaryOp("SUB", x); }
  void Visit(const ir::Mul *x) { PrintBinaryOp("MUL", x); }
  void Visit(const ir::Div *x) { PrintBinaryOp("DIV", x); }
  void Visit(const ir::Mod *x) { PrintBinaryOp("MOD", x); }
  void Visit(const ir::EQ *x) { PrintBinaryOp("EQ", x); }
  void Visit(const ir::NE *x) { PrintBinaryOp("NE", x); }
  void Visit(const ir::LT *x) { PrintBinaryOp("LT", x); }
  void Visit(const ir::LE *x) { PrintBinaryOp("LE", x); }
  void Visit(const ir::GT *x) { PrintBinaryOp("GT", x); }
  void Visit(const ir::GE *x) { PrintBinaryOp("GE", x); }
  void Visit(const ir::And *x) { PrintBinaryOp("AND", x); }
  void Visit(const ir::Or *x) { PrintBinaryOp("OR", x); }
  void Visit(const ir::Max *x) { PrintBinaryOp("MAX", x); }
  void Visit(const ir::Min *x) { PrintBinaryOp("MIN", x); }

  template <typename IRN>
  void PrintBinaryOp(const std::string &op, const ir::BinaryOpNode<IRN> *x) {
    str_ += "_FPA_";
    ir::IrPrinter::Visit(x->a());
    str_ += op;
    ir::IrPrinter::Visit(x->b());
    str_ += "_BPA_";
  }
};

std::string Predicate2String(ir::Expr predicate) {
  std::stringstream ss;
  PredicatePrinter cond_printer(ss);
  cond_printer.Print(predicate);
  return ss.str();
}

static std::string CurTailFnName(const std::string &origin_fn_name) {
  const int MaxStrLength = 16383;
  if (origin_fn_name.length() <= MaxStrLength) {
    return origin_fn_name;
  }
  VLOG(6) << "Function name too long. Curtail and concat hash.";
  const std::string new_fn_name =
      origin_fn_name.substr(0, MaxStrLength) +
      std::to_string(std::hash<std::string>()(origin_fn_name));
  return new_fn_name;
}

std::string
detail::CollectBucketStrategyHostFunctionVisitor::GenDeviceKernelName(
    const std::string &fn_name, ir::Expr predicate) {
  std::string cond_str = Predicate2String(predicate);
  // replace '-' with 'NEG'
  size_t pos = cond_str.find("-", 0);
  const std::string replacement = "NEG";
  while (pos != std::string::npos) {
    cond_str.replace(pos, 1, replacement);
    pos = cond_str.find("-", pos + replacement.length());
  }
  VLOG(3) << "predicate string: " << cond_str;
  // NOTE(chenxi67): The kernel name is too long to be supported in cuda12.3 so
  // we need to curtail it.
  const std::string new_fn_name = CurTailFnName(fn_name);
  return new_fn_name + "__COND_" + cond_str + "__kernel";
}

void detail::CollectBucketStrategyHostFunctionVisitor::ProcessLoweredFunc(
    ir::LoweredFunc func, ir::Expr predicate) {
  VLOG(4) << "Process Lowered Func" << func;
  ir::_LoweredFunc_ *func_node = func.As<ir::_LoweredFunc_>();
  PADDLE_ENFORCE_NOT_NULL(
      func_node,
      ::common::errors::InvalidArgument(
          "The provided function could not be cast to a lowered function. "
          "Please ensure the function is valid."));
  if (!func_node->cuda_axis_info.valid()) {
    func_node->cuda_axis_info.set_valid(true);
  }
  // process device func
  device_module_builder.AddFunctionWithoutOptim(
      CreateDeviceFunction(func, predicate));
  // process host func
  ir::Var kernel_ptr(GenDeviceKernelName(func_node->name, predicate),
                     type_of<std::string>());

  std::optional<Expr> shared_mem_bytes;
  cinn::common::DefaultDeviceTarget().arch.Match(
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
        CINN_NOT_IMPLEMENTED;
      },
      [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
        shared_mem_bytes = CalculateSharedMemory(func);
#endif
      },
      [&](common::HygonDCUArchHIP) {
#ifdef CINN_WITH_HIP
        shared_mem_bytes = CalculateSharedMemory(func);
#endif
      },
      [&](common::HygonDCUArchSYCL) {
#ifdef CINN_WITH_SYCL
        shared_mem_bytes = Expr(0);
#endif
      });

  VLOG(6) << "Add a call node for func_node->name " << func_node->name << "\n"
          << "grid_dim: (" << func_node->cuda_axis_info.grid_dim(0) << ", "
          << func_node->cuda_axis_info.grid_dim(1) << ", "
          << func_node->cuda_axis_info.grid_dim(2) << "), "
          << "block_dim: (" << func_node->cuda_axis_info.block_dim(0) << ", "
          << func_node->cuda_axis_info.block_dim(1) << ", "
          << func_node->cuda_axis_info.block_dim(2) << "), "
          << "shared_mem: " << shared_mem_bytes.value();
  std::optional<const char *> call_kernel;
  cinn::common::DefaultDeviceTarget().arch.Match(
      [&](std::variant<common::UnknownArch, common::X86Arch, common::ARMArch>) {
        CINN_NOT_IMPLEMENTED;
      },
      [&](common::NVGPUArch) {
        call_kernel = runtime::intrinsic::call_cuda_kernel;
      },
      [&](common::HygonDCUArchHIP) {
        call_kernel = runtime::intrinsic::call_hip_kernel;
      },
      [&](common::HygonDCUArchSYCL) {
        call_kernel = runtime::intrinsic::call_sycl_kernel;
      });
  // TODO(Dmovic): use new ir when backend update done.
  ir::Expr call_extern_api =
      ir::Call::Make(Void(),
                     call_kernel.value(),
                     {kernel_ptr,
                      kernel_args_,
                      kernel_args_num_,
                      func_node->cuda_axis_info.grid_dim(0),   // grid_x
                      func_node->cuda_axis_info.grid_dim(1),   // grid_y
                      func_node->cuda_axis_info.grid_dim(2),   // grid_z
                      func_node->cuda_axis_info.block_dim(0),  // block_x
                      func_node->cuda_axis_info.block_dim(1),  // block_y
                      func_node->cuda_axis_info.block_dim(2),  // block_z
                      shared_mem_bytes.value(),                // shared_mem
                      kernel_stream_},
                     {},
                     ir::CallType::Extern,
                     ir::FunctionRef(),
                     0);

  // create memset calls for temp_spaces if needed
  std::vector<ir::stmt::StmtRef> call_kernel_stmts;
  for (auto &temp_space : func_node->temp_spaces) {
    if (temp_space.need_zero_init()) {
      ir::Expr size = common::cast(temp_space.size(), common::UInt(64));
      ir::Expr call_get_arg =
          lang::CallExtern(runtime::intrinsic::get_item_in_cuda_kernel_args,
                           {kernel_args_, ir::Expr(temp_space.arg_idx())});
      ir::Expr call_memset = lang::CallExtern(
          runtime::intrinsic::call_cuda_memset,
          {call_get_arg, ir::Expr(1), ir::Expr(0), size, kernel_stream_});
      call_kernel_stmts.push_back(ir::stmt::Evaluate(call_memset));
    }
  }
  call_kernel_stmts.push_back(ir::stmt::Evaluate(call_extern_api));
  auto call_extern_api_block = ir::stmt::BlockRef(call_kernel_stmts);

  if (buckets_.empty()) {
    buckets_.emplace_back(
        ir::stmt::IfThenElse(predicate, call_extern_api_block));
  } else {
    auto false_expr = buckets_.back();
    buckets_.pop_back();
    buckets_.emplace_back(ir::stmt::IfThenElse(
        predicate,
        call_extern_api_block,
        ir::stmt::BlockRef(std::vector<ir::stmt::StmtRef>{false_expr})));
  }

  // create infer shape calls for temp_spaces
  std::vector<ir::stmt::StmtRef> temp_space_infer_shape_stmts;
  for (int i = 0; i < func_node->temp_spaces.size(); ++i) {
    ir::Var tensor_shape_args(TENSOR_SHAPE_ARGS, type_of<int64_t **>());
    ir::Expr size =
        common::cast(func_node->temp_spaces[i].size(), common::Int(64));
    ir::Expr call_set_value =
        lang::CallExtern(runtime::intrinsic::infer_shape_set_value,
                         {ir::Expr(func_node->num_output_tensors + i),
                          ir::Expr(0),
                          size,
                          tensor_shape_args});
    temp_space_infer_shape_stmts.push_back(ir::stmt::Evaluate(call_set_value));
  }
  if (!temp_space_infer_shape_stmts.empty()) {
    ir::stmt::BlockRef if_body =
        ir::stmt::BlockRef(temp_space_infer_shape_stmts);
    if (temp_space_infer_shape_body_.defined()) {
      temp_space_infer_shape_body_ = ir::stmt::IfThenElse(
          predicate,
          if_body,
          ir::stmt::BlockRef(
              std::vector<ir::stmt::StmtRef>{temp_space_infer_shape_body_}));
    } else {
      temp_space_infer_shape_body_ = ir::stmt::IfThenElse(predicate, if_body);
    }
  }
}

void detail::CollectBucketStrategyHostFunctionVisitor::ProcessArgs(
    ir::LoweredFunc func) {
  const std::vector<ir::Argument> &args = func->args;
  for (int i = 0; i < args.size(); ++i) {
    if (args[i].is_var()) {
      ir::Expr call_get_value_in_kernel_args =
          ir::Call::Make(Int(64),
                         runtime::intrinsic::get_value_in_cuda_kernel_args,
                         {kernel_args_, ir::Expr(i)},
                         {},
                         ir::CallType::Extern,
                         ir::FunctionRef(),
                         0);
      ir::Expr let_symbol = ir::ir_utils::IRCopy(args[i].var_arg());
      let_symbol->set_type(type_of<int64_t>());
      ir::stmt::StmtRef stmt =
          ir::stmt::Let(let_symbol, call_get_value_in_kernel_args);
      arg_defs_.push_back(stmt);
    }
  }
}

ir::LoweredFunc
detail::CollectBucketStrategyHostFunctionVisitor::CreateDeviceFunction(
    ir::LoweredFunc expr, ir::Expr predicate) {
  auto copied = ir::ir_utils::IRCopy(expr);
  copied->name = GenDeviceKernelName(copied->name, predicate);
  return copied;
}

}  // namespace backends
}  // namespace cinn
