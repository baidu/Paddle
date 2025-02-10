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

#include "paddle/cinn/ir/lowered_func.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/common/enforce.h"

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/functional.h"
#include "paddle/cinn/utils/string.h"

PD_DECLARE_bool(cinn_runtime_display_debug_info);

namespace cinn {
namespace ir {

using cinn::common::bfloat16;
using cinn::common::float16;

const _LoweredFunc_* LoweredFunc::operator->() const {
  return As<_LoweredFunc_>();
}
_LoweredFunc_* LoweredFunc::operator->() { return As<_LoweredFunc_>(); }

LoweredFunc _LoweredFunc_::Make(const std::string& name,
                                const std::vector<Argument>& args,
                                const Expr& body,
                                const std::vector<ir::Buffer>& temp_bufs) {
  auto* n = make_shared<_LoweredFunc_>();
  n->name = name;
  n->args = args;
  n->body = body;
  n->temp_bufs = temp_bufs;

  n->CheckValid();
  n->PrepareAllocOutputBufferExprs();
  n->PrepareCreateTempBufferExprs();
  n->PrepareAllocTempBufferExprs();
  n->AllocTempBuffer();
  bool with_expr_gen_tensor = false;
  n->PrepareBufferCastExprs(with_expr_gen_tensor);
  n->PrepareArgumentExprs();
  n->PrepareDeallocTempBufferExprs();
  n->PrepareDeallocOutputBufferExprs();
  return LoweredFunc(n);
}

LoweredFunc _LoweredFunc_::Make(const std::string& name,
                                const std::vector<Argument>& args,
                                const stmt::BlockRef& body,
                                const std::vector<ir::Buffer>& temp_bufs) {
  auto* n = make_shared<_LoweredFunc_>();
  n->name = name;
  n->args = args;
  n->body_block = body;
  n->temp_bufs = temp_bufs;

  n->CheckValid();
  n->PrepareAllocOutputBufferExprs();
  n->PrepareCreateTempBufferExprs();
  n->PrepareAllocTempBufferExprs();
  n->AllocTempBuffer();
  bool with_expr_gen_tensor = false;
  n->PrepareBufferCastExprs(with_expr_gen_tensor);
  n->PrepareArgumentExprs();
  n->PrepareDeallocTempBufferExprs();
  n->PrepareDeallocOutputBufferExprs();
  return LoweredFunc(n);
}

LoweredFunc _LoweredFunc_::Make(const std::string& name,
                                const std::vector<Argument>& args,
                                const Expr& body) {
  auto* n = make_shared<_LoweredFunc_>();
  n->name = name;
  n->args = args;
  n->body = body;
  return LoweredFunc(n);
}

LoweredFunc _LoweredFunc_::Make(const std::string& name,
                                const std::vector<Argument>& args,
                                const stmt::BlockRef& body) {
  auto* n = make_shared<_LoweredFunc_>();
  n->name = name;
  n->args = args;
  n->body_block = body;
  return LoweredFunc(n);
}

void _LoweredFunc_::CheckValid() const {
  // check there is at least one output
  int out_count = 0;
  int in_count = 0;
  for (auto& arg : args) {
    in_count += arg.is_input();
    out_count += arg.is_output();
  }
  PADDLE_ENFORCE_GT(
      out_count,
      0,
      ::common::errors::InvalidArgument(
          "At least one output argument is needed for a function."));
}

std::vector<Expr*> _LoweredFunc_::expr_fields() { return {&body}; }
std::vector<const Expr*> _LoweredFunc_::expr_fields() const { return {&body}; }

void _LoweredFunc_::PrepareCudaAxisInfoFromBody() {
  std::set<Expr> bound_for_exprs =
      ir::ir_utils::CollectIRNodes(body, [](const Expr* expr) {
        const ir::For* for_expr = expr->As<ir::For>();
        return for_expr != nullptr && for_expr->is_binded();
      });

  if (bound_for_exprs.empty()) {
    device_api = ir::DeviceAPI::GPU;
    cuda_axis_info.set_grid_dim(0, 1);
    cuda_axis_info.set_block_dim(0, 1);
    cuda_axis_info.set_valid(true);
    return;
  }

  // bound_for_exprs.empty() is false
  for (const Expr& expr : bound_for_exprs) {
    const ir::For* for_expr = expr.As<ir::For>();
    if (for_expr->for_type() == ir::ForType::GPUBlock) {
      cuda_axis_info.set_grid_dim(for_expr->bind_info().offset,
                                  for_expr->extent);
    } else if (for_expr->for_type() == ir::ForType::GPUThread) {
      cuda_axis_info.set_block_dim(for_expr->bind_info().offset,
                                   for_expr->extent);
    }
  }
  device_api = ir::DeviceAPI::GPU;
  cuda_axis_info.set_valid(true);
}

void _LoweredFunc_::PrepareAllocOutputBufferExprs() {
  PADDLE_ENFORCE_EQ(alloc_output_buffer_exprs.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Duplicate prepare the allocate buffer for outputs."));
  std::set<std::string> buffer_names;
  for (auto& arg : args) {
    if (arg.is_output()) {
      PADDLE_ENFORCE_EQ(
          arg.type().valid(),
          true,
          ::common::errors::InvalidArgument(
              "Argument ['%s']'s type should be set.", arg.name()));
      if (arg.is_buffer() &&
          !buffer_names.count(arg.name())) {  // only buffer need allocation.
        buffer_names.insert(arg.name());      // Avoid duplicate
        alloc_output_buffer_exprs.push_back(
            Alloc::Make(arg.buffer_arg(),
                        arg.buffer_arg()->type(),
                        arg.buffer_arg()->shape,
                        Expr(),
                        Expr()));
      }
    }
  }
}

std::vector<ir::stmt::StmtRef> _LoweredFunc_::PrepareAxisRangeAssumptionStmts()
    const {
  std::vector<ir::stmt::StmtRef> assumption_stmts;

  const auto AssumeAxisLT = [&](std::string axis, const Expr& dim_size) {
    if (!dim_size.defined()) {
      return;
    }
    if (dim_size == common::make_const(1)) {
      return;
    }
    Expr expr_lt = LT::Make(Var(axis), dim_size);
    Expr call_lt = Call::Make(Void(),
                              runtime::intrinsic::cuda_builtin_assume,
                              {expr_lt},
                              {},
                              CallType::Intrinsic);
    assumption_stmts.push_back(ir::stmt::Evaluate(call_lt));
  };

  AssumeAxisLT("blockIdx.x", cuda_axis_info.grid_dim(0));
  AssumeAxisLT("blockIdx.y", cuda_axis_info.grid_dim(1));
  AssumeAxisLT("blockIdx.z", cuda_axis_info.grid_dim(2));
  AssumeAxisLT("threadIdx.x", cuda_axis_info.block_dim(0));
  AssumeAxisLT("threadIdx.y", cuda_axis_info.block_dim(1));
  AssumeAxisLT("threadIdx.z", cuda_axis_info.block_dim(2));

  return assumption_stmts;
}

std::vector<Expr> _LoweredFunc_::PrepareAllocTempBufferExprs() const {
  std::vector<Expr> alloc_temp_buffer_exprs;
  for (auto& temp_buf : temp_bufs) {
    if (!temp_buf->shape.empty() && temp_buf->type() != Void()) {
      alloc_temp_buffer_exprs.push_back(Alloc::Make(
          temp_buf, temp_buf->type(), temp_buf->shape, Expr(), Expr()));
    }
  }
  return alloc_temp_buffer_exprs;
}

std::vector<ir::stmt::StmtRef> _LoweredFunc_::PrepareAllocTempBufferStmts()
    const {
  std::vector<ir::stmt::StmtRef> alloc_temp_buffer_exprs;
  for (auto& temp_buf : temp_bufs) {
    if (!temp_buf->shape.empty() && temp_buf->type() != Void()) {
      alloc_temp_buffer_exprs.push_back(ir::stmt::Alloc(
          temp_buf, temp_buf->type(), temp_buf->shape, Expr(), Expr()));
    }
  }
  return alloc_temp_buffer_exprs;
}

std::vector<Expr> _LoweredFunc_::PrepareDeallocTempBufferExprs() const {
  std::vector<Expr> dealloc_temp_buffer_exprs;
  for (auto& temp_buf : temp_bufs) {
    if (!temp_buf->shape.empty() && temp_buf->type() != Void()) {
      dealloc_temp_buffer_exprs.push_back(Free::Make(temp_buf));
    }
  }
  return dealloc_temp_buffer_exprs;
}

std::vector<ir::stmt::StmtRef> _LoweredFunc_::PrepareDeallocTempBufferStmts()
    const {
  std::vector<ir::stmt::StmtRef> dealloc_temp_buffer_exprs;
  for (auto& temp_buf : temp_bufs) {
    if (!temp_buf->shape.empty() && temp_buf->type() != Void()) {
      dealloc_temp_buffer_exprs.push_back(ir::stmt::Free(temp_buf));
    }
  }
  return dealloc_temp_buffer_exprs;
}

std::vector<Expr> _LoweredFunc_::PrepareCreateTempBufferExprs() const {
  std::vector<Expr> create_temp_buffer_exprs;
  for (auto& temp_buf : temp_bufs) {
    if (!temp_buf->shape.empty() && temp_buf->type() != Void()) {
      auto expr = ir::intrinsics::BufferCreate::Make(temp_buf);
      auto buffer_ptr_type =
          Type()
              .set_customized_type(cinn::common::customized_type::kbuffer_t)
              .set_cpp_handle();
      Var variable = ir::_Var_::Make(temp_buf->name, buffer_ptr_type);
      expr = ir::Let::Make(variable, expr);
      create_temp_buffer_exprs.push_back(expr);
    }
  }
  return create_temp_buffer_exprs;
}

std::vector<Expr> _LoweredFunc_::CudaPrepareAllocTempBufferExprs() const {
  std::vector<Expr> alloc_output_buffer_exprs;
  for (auto temp_buf : temp_bufs) {
    if (utils::StartsWith(temp_buf->name, "_")) {
      temp_buf->name = temp_buf->name.substr(1);
    }
    if (!temp_buf->shape.empty() && temp_buf->type() != Void()) {
      alloc_output_buffer_exprs.push_back(Alloc::Make(
          temp_buf, temp_buf->type(), temp_buf->shape, Expr(), Expr()));
    }
  }
  return alloc_output_buffer_exprs;
}

void _LoweredFunc_::PrepareDeallocOutputBufferExprs() {
  PADDLE_ENFORCE_EQ(dealloc_output_buffer_exprs.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Duplicate prepare the allocate buffer for outputs."));

  std::set<std::string> buffer_names;
  for (auto& arg : args) {
    if (arg.is_output()) {
      PADDLE_ENFORCE_EQ(
          arg.type().valid(),
          true,
          ::common::errors::InvalidArgument(
              "Argument ['%s']'s type should be set.", arg.name()));
      if (arg.is_buffer() &&
          !buffer_names.count(arg.name())) {  // only buffer need allocation.
        buffer_names.insert(arg.name());      // Avoid duplicate
        dealloc_output_buffer_exprs.push_back(Free::Make(arg.buffer_arg()));
      }
    }
  }
}

void _LoweredFunc_::AllocTempBuffer() {}

void _LoweredFunc_::PrepareBufferCastExprs(bool with_expr_gen_tensor) {
  buffer_data_cast_exprs.clear();
  // collect write.
  auto write_teller = ir::ir_utils::CollectTensorNeedsWrite(&body);

  auto tensors = CollectAllTensorReference(with_expr_gen_tensor);
  std::sort(tensors.begin(),
            tensors.end(),
            [](const Tensor& a, const Tensor& b) { return a->name < b->name; });

  VLOG(3) << "Function used " << tensors.size() << " buffers";
  for (auto& tensor : tensors) {
    auto* node = tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument(
            "Failed to convert tensor to ir::_Tensor_. The tensor might be "
            "invalid or of an incorrect type."));
    if (!tensor->buffer.defined()) continue;

    Type value_type = tensor->type().ElementOf();
    bool is_const = !write_teller.count(tensor->name);
    value_type.set_cpp_handle();
    value_type.set_cpp_const(is_const);
    Var variable = _Var_::Make(tensor->name, value_type);

    Expr body =
        is_const
            ? ir::intrinsics::BufferGetDataConstHandle::Make(tensor->buffer)
            : ir::intrinsics::BufferGetDataHandle::Make(tensor->buffer);

    Type target_type = is_const ? tensor->buffer->dtype.PointerOf().ConstOf()
                                : tensor->buffer->dtype.PointerOf();
    body = ir::Cast::Make(target_type, body);
    auto let = Let::Make(variable, body);

    buffer_data_cast_exprs.push_back(let);
  }
}

std::vector<ir::stmt::StmtRef> _LoweredFunc_::CudaAliasVarStmts() const {
  std::unordered_set<std::string> args_buffer;
  for (auto arg : args) {
    args_buffer.insert(arg.name());
  }
  // collect write.
  std::vector<ir::stmt::StmtRef> res;
  auto write_teller = ir::ir_utils::CollectTensorNeedsWrite(&body);

  auto tensors = CollectAllTensorReference();
  std::sort(tensors.begin(),
            tensors.end(),
            [](const Tensor& a, const Tensor& b) { return a->name < b->name; });

  for (auto& tensor : tensors) {
    auto* node = tensor.As<ir::_Tensor_>();
    PADDLE_ENFORCE_NOT_NULL(
        node,
        ::common::errors::InvalidArgument(
            "Failed to convert tensor to ir::_Tensor_. The tensor might be "
            "invalid or of an incorrect type."));
    if (!tensor->buffer.defined()) {
      continue;
    }
    if (tensor->name == tensor->buffer->name.substr(1) ||
        args_buffer.count(tensor->buffer->name) == 0) {
      continue;
    }
    Type value_type = tensor->type().ElementOf();
    bool is_const = !write_teller.count(tensor->name);
    value_type.set_cpp_handle();
    value_type.set_cpp_const(is_const);
    Var variable = _Var_::Make(tensor->name, value_type);
    Var body = Var(tensor->buffer->name.substr(1), value_type);

    auto let = ir::stmt::Let(variable, body);

    res.push_back(let);
  }
  return res;
}

void _LoweredFunc_::PrepareArgumentExprs() {
  // Seems a CINN func.
  if (args.front().is_var() &&
      args.front().var_arg()->type() == type_of<cinn_pod_value_t*>())
    return;

  // type of `void*`
  auto void_ptr_array_type =
      Type().with_type(Type::type_t::Void).set_cpp_handle();
  // type of `cinn_buffer_t*`
  auto buffer_ptr_type =
      Type()
          .set_customized_type(cinn::common::customized_type::kbuffer_t)
          .set_cpp_handle();
  // type of `const cinn_buffer_t*`
  auto const_buffer_ptr_type = buffer_ptr_type.with_cpp_const();
  PADDLE_ENFORCE_NE(buffer_ptr_type.is_cpp_const(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The buffer pointer type should not be const."));
  Var args_passed_in("_args", type_of<void*>());
  auto pod_value_ptr =
      cinn::common::CastIfNeeded(args_passed_in, type_of<cinn_pod_value_t*>());

  if (FLAGS_cinn_runtime_display_debug_info) {
    argument_prepare_exprs.push_back(runtime::IntrinsicCall(
        Void(),
        runtime::intrinsic::print_debug_args_repr,
        {pod_value_ptr, cinn::common::make_const(Int(32), args.size())}));
  }

  /*
   * Get something like:
   *
   * const cinn_buffer_t* _A = args[0];
   * cinn_buffer_t* _B = (cinn_buffer_t*)args[1];
   * int M = (int)arg[2];
   */

  // We just has two kinds of argument types, first is `cinn_buffer_t*`, second
  // is `const cinn_buffer_t*`, do not need a `any` type support currently.
  for (int i = 0; i < args.size(); i++) {
    auto& arg = args[i];
    // cast arg to cinn_pod_value_t*

    // something like `_args[0]`
    Expr load_expr = Load::Make(
        pod_value_ptr, {cinn::common::make_const(static_cast<int32_t>(i))});
    PADDLE_ENFORCE_EQ(load_expr.type(),
                      type_of<cinn_pod_value_t>(),
                      ::common::errors::InvalidArgument(
                          "The type of load_expr should be cinn_pod_value_t"));
    load_expr = ir::intrinsics::GetAddr::Make(load_expr);

    Var _arg;
    bool is_const = arg.is_input();

    if (arg.is_buffer()) {
      auto buffer_type = is_const ? const_buffer_ptr_type : buffer_ptr_type;
      _arg = Var(arg.name(), buffer_type);
    } else if (arg.is_var()) {
      _arg = Var(arg.name(), arg.var_arg()->type());
    } else {
      CINN_NOT_IMPLEMENTED
    }

    PADDLE_ENFORCE_EQ(
        _arg->type().valid(),
        true,
        ::common::errors::InvalidArgument("Argument's type should be set."));

    Expr pod_cast_expr;

    if (arg.is_buffer()) {
      pod_cast_expr = ir::intrinsics::PodValueToX::Make(
          load_expr, type_of<cinn_buffer_t*>());
    } else if (arg.type() == type_of<int8_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int8_t>());
    } else if (arg.type() == type_of<int16_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int16_t>());
    } else if (arg.type() == type_of<int32_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int32_t>());
    } else if (arg.type() == type_of<int64_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int64_t>());
    } else if (arg.type() == type_of<uint8_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<uint8_t>());
    } else if (arg.type() == type_of<uint16_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<uint16_t>());
    } else if (arg.type() == type_of<uint32_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<uint32_t>());
    } else if (arg.type() == type_of<uint64_t>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<uint64_t>());
    } else if (arg.type() == type_of<bfloat16>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<bfloat16>());
    } else if (arg.type() == type_of<float16>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<float16>());
    } else if (arg.type() == type_of<float>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<float>());
    } else if (arg.type() == type_of<double>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<double>());
    } else if (arg.type() == type_of<bool>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<bool>());
    } else if (arg.type() == type_of<void*>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<void*>());
    } else if (arg.type() == type_of<int32_t*>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int32_t*>());
    } else if (arg.type() == type_of<int32_t**>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int32_t**>());
    } else if (arg.type() == type_of<int64_t**>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<int64_t**>());
    } else if (arg.type() == type_of<void**>()) {
      pod_cast_expr =
          ir::intrinsics::PodValueToX::Make(load_expr, type_of<void**>());
    } else {
      LOG(ERROR) << "Not supported type [" << arg.type() << "]";
      CINN_NOT_IMPLEMENTED
    }

    VLOG(6) << "args " << i << "convert";
    Expr let_expr = Let::Make(_arg, pod_cast_expr);
    PADDLE_ENFORCE_EQ(let_expr.type().valid(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The let expression's type should be set."));
    argument_prepare_exprs.push_back(let_expr);
  }
}

std::vector<Tensor> _LoweredFunc_::CollectAllTensorReference(
    bool with_expr_gen_tensor) const {
  std::set<Expr> tensor_exprs =
      with_expr_gen_tensor
          ? ir::ir_utils::CollectIRNodes(
                body, [](const Expr* expr) { return expr->As<ir::_Tensor_>(); })
          : cinn::utils::VectorToSet(ir::ir_utils::CollectIRNodesWithoutTensor(
                body,
                [](const Expr* expr) { return expr->As<ir::_Tensor_>(); }));

  std::vector<Tensor> tensors;
  // remove the duplicate tensor by their name.
  std::set<std::string> names;

  for (const Expr& expr : tensor_exprs) {
    Expr& _expr = *const_cast<Expr*>(&expr);
    Tensor b(_expr.As<_Tensor_>());
    if (names.count(b->name)) continue;
    tensors.push_back(b);
    names.insert(b->name);
  }

  return tensors;
}

ir::Buffer Argument::buffer_arg() const {
  PADDLE_ENFORCE_EQ(
      is_buffer(),
      true,
      ::common::errors::InvalidArgument(
          "The argument is not a buffer. Unable to return buffer_arg_."));
  return buffer_arg_;
}

ir::Var Argument::var_arg() const {
  PADDLE_ENFORCE_EQ(
      is_var(),
      true,
      ::common::errors::InvalidArgument(
          "The argument is not a variable. Unable to return var_arg_."));
  return var_arg_;
}

void Argument::set_buffer(const ir::Buffer& x) {
  PADDLE_ENFORCE_EQ(
      !is_var(),
      true,
      ::common::errors::InvalidArgument("The buffer is already a variable."));
  buffer_arg_ = x;
}

void Argument::set_var(const ir::Var& x) {
  PADDLE_ENFORCE_EQ(
      !is_buffer(),
      true,
      ::common::errors::InvalidArgument("The buffer is already a buffer."));
  var_arg_ = x;
}

Argument::Argument(const ir::Buffer& buffer, Argument::IO io) {
  set_buffer(buffer);
  this->io = io;
}

Type Argument::type() const {
  if (is_var())
    return var_arg()->type();
  else if (is_buffer())
    return buffer_arg()->type();
  else
    CINN_NOT_IMPLEMENTED
}

std::string Argument::name() const {
  if (is_buffer())
    return buffer_arg()->name;
  else if (is_var())
    return var_arg()->name;
  else
    CINN_NOT_IMPLEMENTED
  return "";
}

Argument::Argument(const ir::Var& var, Argument::IO io) {
  set_var(var);
  this->io = io;
}

std::string Argument::human_readable() const {
  std::stringstream os;
  os << "<Argument: " << name() << " ";
  os << (is_input() ? "R" : "W");
  os << ">";
  return os.str();
}

std::ostream& operator<<(std::ostream& os, const CudaAxisInfo& x) {
  os << "<grid:" << x.grid_dim(0) << ", " << x.grid_dim(1) << ", "
     << x.grid_dim(2) << ">";
  os << "<block:" << x.block_dim(0) << ", " << x.block_dim(1) << ", "
     << x.block_dim(2) << ">";
  return os;
}

void CudaAxisInfo::set_grid_dim(int offset, int64_t x) {
  valid_ = true;
  PADDLE_ENFORCE_LT(
      offset,
      3,
      ::common::errors::InvalidArgument("The offset should be less than 3."));
  grid_dims_[offset] = ir::Expr(x);
}

void CudaAxisInfo::set_block_dim(int offset, int64_t x) {
  valid_ = true;
  PADDLE_ENFORCE_LT(
      offset,
      3,
      ::common::errors::InvalidArgument("The offset should be less than 3."));
  block_dims_[offset] = ir::Expr(x);
}

void CudaAxisInfo::set_grid_dim(int offset, ir::Expr x) {
  valid_ = true;
  PADDLE_ENFORCE_LT(
      offset,
      3,
      ::common::errors::InvalidArgument("The offset should be less than 3."));
  grid_dims_[offset] = x;
}

void CudaAxisInfo::set_block_dim(int offset, ir::Expr x) {
  valid_ = true;
  PADDLE_ENFORCE_LT(
      offset,
      3,
      ::common::errors::InvalidArgument("The offset should be less than 3."));
  block_dims_[offset] = x;
}

ir::Expr CudaAxisInfo::grid_dim(int offset) const {
  PADDLE_ENFORCE_EQ(
      valid_,
      true,
      ::common::errors::InvalidArgument("CudaAxisInfo is not valid. This check "
                                        "failed in grid_dim() method."));
  PADDLE_ENFORCE_LT(
      offset,
      3,
      ::common::errors::InvalidArgument("The offset should be less than 3."));
  return grid_dims_[offset];
}

ir::Expr CudaAxisInfo::block_dim(int offset) const {
  PADDLE_ENFORCE_EQ(
      valid_,
      true,
      ::common::errors::InvalidArgument("CudaAxisInfo is not valid. This check "
                                        "failed in block_dim() method."));
  PADDLE_ENFORCE_LT(
      offset,
      3,
      ::common::errors::InvalidArgument("The offset should be less than 3."));
  return block_dims_[offset];
}

}  // namespace ir
}  // namespace cinn
