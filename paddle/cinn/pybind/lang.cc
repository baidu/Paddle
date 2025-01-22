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

#include <absl/types/variant.h>
#include <pybind11/functional.h>

#include <memory>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/stmt_converter.h"
#include "paddle/cinn/lang/buffer.h"
#include "paddle/cinn/lang/builtin.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/pybind/bind_utils.h"

namespace py = pybind11;

namespace cinn::pybind {
using cinn::common::Type;
using lang::Placeholder;
using py::arg;
using utils::GetStreamCnt;
using utils::StringFormat;

namespace {
void BindBuffer(py::module *);
void BindPlaceholder(py::module *);
void BindCompute(py::module *);
void BindModule(py::module *);
void BindBuiltin(py::module *);

void BindBuffer(py::module *m) {
  py::class_<lang::Buffer> buffer(*m, "Buffer");
  buffer
      .def(py::init<ir::Type, const std::string &>(),
           py::arg("type"),
           py::arg("name") = "")
      .def(py::init<const ir::Buffer &>())
      .def("buffer", &lang::Buffer::buffer);
}

void BindCompute(py::module *m) {
#define MAKE_COMPUTE_FN(__fn)                      \
  py::overload_cast<const std::vector<ir::Expr> &, \
                    __fn,                          \
                    const std::string &,           \
                    const std::vector<ir::Expr> &>(&lang::Compute)

#define DEFINE_COMPUTE(__fn)    \
  m->def("compute",             \
         MAKE_COMPUTE_FN(__fn), \
         arg("domin"),          \
         arg("fn"),             \
         arg("name") = "",      \
         arg("shape") = std::vector<ir::Expr>())

  // DEFINE_COMPUTE(std::function<ir::Expr()>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr)>);
  DEFINE_COMPUTE(std::function<ir::Expr(const std::vector<ir::Expr> &)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr, ir::Expr)>);
  // DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr, ir::Expr,
  // ir::Expr)>); DEFINE_COMPUTE(std::function<ir::Expr(ir::Expr, ir::Expr,
  // ir::Expr, ir::Expr, ir::Expr)>);
  DEFINE_COMPUTE(lang::compute_handler_t);

#undef DEFINE_COMPUTE
#undef MAKE_COMPUTE_FN

  py::class_<lang::ReturnType> return_type(*m, "ReturnType");
  return_type.def_readwrite("type", &lang::ReturnType::type)
      .def_readwrite("dims", &lang::ReturnType::dims)
      .def_readwrite("name", &lang::ReturnType::name);

  m->def("call_lowered",
         py::overload_cast<const std::string &,
                           const std::vector<ir::Expr> &,
                           const std::vector<lang::ReturnType> &>(
             &lang::CallLowered));
  m->def("call_extern",
         py::overload_cast<
             const std::string &,
             const std::vector<ir::Expr> &,
             const std::map<std::string,
                            absl::variant<int, float, bool, std::string>> &>(
             &lang::CallExtern));
}

void BindModule(py::module *m) {
  py::class_<ir::Module /*, ir::IrNodeRef*/> module(*m, "Module");

  module.def("target", &ir::Module::target)
      .def("buffers", &ir::Module::buffers)
      .def("functions", &ir::Module::functions)
      .def("submodules", &ir::Module::submodules)
      .def("compile", &ir::Module::Compile)
      .def("get_c_code", [](const ir::Module &self) -> std::string {
        backends::CodeGenC codegen(cinn::common::DefaultHostTarget());
        codegen.SetInlineBuiltinCodes(false);
        return codegen.Compile(self, backends::CodeGenC::OutputKind::CImpl);
      });

  py::class_<ir::Module::Builder> builder(module, "Builder");
  builder.def(py::init<const std::string &, const cinn::common::Target &>())
      .def("add_function",
           [](ir::Module::Builder &self, ir::LoweredFunc func) {
             self.GetTargetArch().Match(
                 [&](common::UnknownArch) { LOG(FATAL) << "NotImplemented"; },
                 [&](common::X86Arch) {
                   // Do nothing
                 },
                 [&](common::ARMArch) {
                   // Do nothing
                 },
                 [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
                   ir::SetCudaAxisInfo(func);
                   ir::stmt::BlockRef func_body_block =
                       ir::ConvertExprBlockToStmtBlock(func->body);
                   VLOG(6) << " Before OptimizeExprGPU in lang: \n"
                           << func_body_block;
                   optim::OptimizeExprGPU(func_body_block);
                   VLOG(6) << "After OptimizeExprGPU in lang: \n"
                           << func_body_block;
                   func->body =
                       ir::ConvertStmtBlockToExprBlock(func_body_block);
#endif
                 },
                 [&](std::variant<common::HygonDCUArchHIP,
                                  common::HygonDCUArchSYCL>) {
                   PADDLE_THROW(::common::errors::Unimplemented(
                       "CINN old obsolete code!"));
                 });
             self.AddFunction(func);
           })
      .def("add_buffer", &ir::Module::Builder::AddBuffer)
      .def("build", &ir::Module::Builder::Build);
}

class PlaceholderWrapper {
 public:
#define DEFINE_PLACEHOLDER(__dtype, __type) \
  if (dtype == #__dtype)                    \
  placeholder_ = std::make_unique<Placeholder<__type>>(name, shape)

#define INIT_PLACEHOLDER              \
  DEFINE_PLACEHOLDER(int32, int32_t); \
  DEFINE_PLACEHOLDER(int64, int64_t); \
  DEFINE_PLACEHOLDER(float32, float); \
  DEFINE_PLACEHOLDER(float64, double)

  PlaceholderWrapper(absl::string_view dtype,
                     const std::string &name,
                     const std::vector<int> &shape) {
    INIT_PLACEHOLDER;
  }

  PlaceholderWrapper(absl::string_view dtype,
                     const std::string &name,
                     const std::vector<ir::Expr> &shape) {
    INIT_PLACEHOLDER;
  }
#undef INIT_PLACEHOLDER
#undef DEFINE_PLACEHOLDER

  ir::Type type() const {
    return absl::visit([](auto &v) { return v->type(); }, placeholder_);
  }

  ir::Tensor tensor() const {
    return absl::visit([](auto &v) { return v->tensor(); }, placeholder_);
  }

  ir::Expr operator()(ir::Expr a) const {
    return absl::visit([&](auto &v) { return (*v)(a); }, placeholder_);
  }

  ir::Expr operator()(ir::Expr a, ir::Expr b) const {
    return absl::visit([&](auto &v) { return (*v)(a, b); }, placeholder_);
  }

  ir::Expr operator()(ir::Expr a, ir::Expr b, ir::Expr c) const {
    return absl::visit([&](auto &v) { return (*v)(a, b, c); }, placeholder_);
  }

  ir::Expr operator()(const std::vector<ir::Expr> &indices) const {
    return absl::visit([&](auto &v) { return (*v)(indices); }, placeholder_);
  }

  operator ir::Tensor() {
    return absl::visit([&](auto &v) { return ir::Tensor(*v); }, placeholder_);
  }
  operator ir::Expr() {
    return absl::visit([&](auto &v) { return ir::Expr(*v); }, placeholder_);
  }

 private:
  template <typename... Ts>
  using PlaceholderVariant = absl::variant<std::unique_ptr<Placeholder<Ts>>...>;

  PlaceholderVariant<int, int64_t, float, double> placeholder_;
};

void BindPlaceholder(py::module *m) {
  py::class_<PlaceholderWrapper> placeholder(*m, "Placeholder");
  placeholder
      .def(py::init<absl::string_view,
                    const std::string &,
                    const std::vector<int> &>())
      .def(py::init<absl::string_view,
                    const std::string &,
                    const std::vector<ir::Expr> &>())
      .def("type", &PlaceholderWrapper::type)
      .def("tensor", &PlaceholderWrapper::tensor)
      .def("__call__",
           [](PlaceholderWrapper &self, ir::Expr a) {
             return self(std::move(a));
           })
      .def("__call__",
           [](PlaceholderWrapper &self, ir::Expr a, ir::Expr b) {
             return self(std::move(a), std::move(b));
           })
      .def("__call__",
           [](PlaceholderWrapper &self, ir::Expr a, ir::Expr b, ir::Expr c) {
             return self(std::move(a), std::move(b), std::move(c));
           })
      .def("__call__",
           [](PlaceholderWrapper &self, const std::vector<ir::Expr> &indices) {
             return self(indices);
           })
      .def("to_expr", [](PlaceholderWrapper &self) { return ir::Expr(self); })
      .def("to_tensor",
           [](PlaceholderWrapper &self) { return ir::Tensor(self); });

  m->def("create_placeholder",
         static_cast<ir::Tensor (*)(
             const std::vector<Expr> &, Type, const std::string &)>(
             &lang::CreatePlaceHolder));
  m->def("create_placeholder",
         static_cast<ir::Tensor (*)(
             const std::vector<int> &, Type, const std::string &)>(
             &lang::CreatePlaceHolder));
}

void BindBuiltin(py::module *m) {
  m->def("reduce_sum",
         &lang::ReduceSum,
         py::arg("e"),
         py::arg("reduce_axis"),
         py::arg("init") = Expr());
  m->def("reduce_mul", &lang::ReduceMul);
  m->def("reduce_max", &lang::ReduceMax);
  m->def("reduce_min", &lang::ReduceMin);
  m->def("reduce_all", &lang::ReduceAll);
  m->def("reduce_any", &lang::ReduceAny);
}

}  // namespace

void BindLang(py::module *m) {
  BindBuffer(m);
  BindPlaceholder(m);
  BindCompute(m);
  BindModule(m);
  BindBuiltin(m);
}
}  // namespace cinn::pybind
