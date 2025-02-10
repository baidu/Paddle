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

#include "paddle/fluid/pybind/pir.h"
#include "paddle/fluid/pybind/pir_utils.h"

#include <Python.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/gpu/fused_bn_add_act_pass.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/fluid/pir/utils/name_analysis.h"
#include "paddle/fluid/pybind/control_flow_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/core/visitors.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pir_to_py_code_converter.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#endif

using paddle::dialect::ApiBuilder;
using paddle::dialect::DenseTensorArrayType;
using paddle::dialect::DenseTensorType;
using paddle::dialect::DistDenseTensorType;
using paddle::dialect::DistTypeInterface;
using paddle::dialect::IfOp;
using paddle::dialect::PyLayerOp;
using paddle::dialect::SelectedRowsType;
using paddle::dialect::SparseCooTensorType;
using paddle::dialect::SparseCsrTensorType;
using paddle::dialect::WhileOp;
using pir::TuplePopOp;

using paddle::dialect::IntArrayAttribute;
using paddle::dialect::OperationDistAttribute;
using paddle::dialect::PlaceAttribute;
using paddle::dialect::TensorDistAttribute;
using pir::ArrayAttribute;
using pir::Attribute;
using pir::Block;
using pir::BlockArgument;
using pir::BoolAttribute;
using pir::CloneOptions;
using pir::Int32Attribute;
using pir::Int64Attribute;
using pir::IrContext;
using pir::IrMapping;
using pir::Operation;
using pir::OpOperand;
using pir::OpResult;
using pir::Pass;
using pir::PassManager;
using pir::Program;
using pir::StrAttribute;
using pir::Type;
using pir::Value;
using pir::VectorType;
using pybind11::return_value_policy;

namespace name_analysis = pir::utils::name_analysis;

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(pir_apply_shape_optimization_pass);

namespace paddle {
namespace pybind {

PyTypeObject *g_ir_value_pytype = nullptr;

void BindOpsAPI(pybind11::module *module);

pir::Value FakeValue() {
  // create a fake value to simplify `ForwardBackwardSplit`.
  return pir::Value(nullptr);
}

bool IsFakeValue(const pir::Value &value) {
  // create a fake value to simplify `ForwardBackwardSplit`.
  return value.impl() == nullptr || !value.type();
}

inline int64_t GetProgramInt64Attr(const std::shared_ptr<Program> &program,
                                   const std::string &attr_name,
                                   int64_t default_value = 0) {
  auto op = program->module_op();
  if (op->HasAttribute(attr_name)) {
    auto val = op->attribute(attr_name).dyn_cast<pir::Int64Attribute>().data();
    return val;
  } else {
    return default_value;
  }
}

inline void SetProgramInt64Attr(std::shared_ptr<Program> program,
                                const std::string &attr_name,
                                int64_t value) {
  auto op = program->module_op();
  op->set_attribute(
      attr_name, pir::Int64Attribute::get(pir::IrContext::Instance(), value));
}

std::string GetValueInfo(Value v) {
  if (v.impl() == nullptr) {
    return "nullptr value";
  }
  std::stringstream ss;
  if (auto op_result = v.dyn_cast<OpResult>()) {
    ss << "define_op_name=" << op_result.owner()->name();
    ss << ", index=" << op_result.index();
  } else if (auto arg = v.dyn_cast<BlockArgument>()) {
    if (arg.is_kwarg()) {
      ss << "keyword block_arg, keyword = " << arg.keyword();
    } else {
      ss << "position block_arg, index = " << arg.index();
    }
  }
  if (!v.type()) {
    ss << ", dtype=<<NULL TYPE>>";
  } else {
    ss << ", dtype=" << v.type();
    if (v.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
      ss << ", place="
         << v.type()
                .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
                .place();
    }
  }
  auto stop_gradient = v.attribute<BoolAttribute>(kAttrStopGradients);
  if (stop_gradient && !stop_gradient.data()) {
    ss << ", stop_gradient=False";
  } else {
    ss << ", stop_gradient=True";
  }
  return ss.str();
}

py::object Clone(const Program &self, IrMapping *p_mapper = nullptr) {
  IrMapping mapper;
  if (p_mapper == nullptr) {
    p_mapper = &mapper;
  }
  auto src_obj = py::cast(self);
  auto new_obj = py::cast(self.Clone(*p_mapper));
  for (auto item : src_obj.attr("__dict__").cast<py::dict>()) {
    new_obj.attr(item.first.cast<std::string>().c_str()) = item.second;
  }
  return new_obj;
}

bool SomeInSet(const std::vector<pir::Value> &vec,
               const std::set<pir::Value> &set) {
  for (auto &v : vec) {
    if (set.find(v) != set.end()) {
      return true;
    }
  }
  return false;
}

pir::Value AppendDataOp(pir::Block *block,
                        const pir::Value &value,
                        const std::string &name,
                        const pir::Operation &origin_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto op_info = ctx->GetRegisteredOpInfo(paddle::dialect::DataOp::name());
  pir::AttributeMap attribute_map = {
      {"name", StrAttribute::get(ctx, name)},
      {"shape",
       paddle::dialect::IntArrayAttribute::get(
           ctx, phi::IntArray(phi::vectorize(GetValueDims(value))))},
      {"dtype",
       paddle::dialect::DataTypeAttribute::get(ctx, pir::GetValueDtype(value))},
      {"place", PlaceAttribute::get(ctx, phi::Place())}};
  std::vector<pir::Type> output_types{value.type()};
  pir::Operation *operation =
      pir::Operation::Create({}, attribute_map, output_types, op_info);

  block->insert(origin_op, operation);
  return operation->result(0);
}
std::vector<pir::Value> GetRealOpInputs(pir::Operation *op) {
  if (op->isa<paddle::dialect::IfOp>() ||
      op->isa<paddle::dialect::PyLayerOp>()) {
    return pir::GetUsedExternalValue(*op);
  } else if (op->isa<paddle::dialect::WhileOp>()) {
    paddle::dialect::WhileOp whileop = op->dyn_cast<paddle::dialect::WhileOp>();
    auto value_vector = op->operands_source();
    auto value_vector2 = pir::GetUsedExternalValue(whileop.body());
    value_vector.insert(
        value_vector.end(), value_vector2.begin(), value_vector2.end());
    return value_vector;
  } else {
    return op->operands_source();
  }
}
/*
  Variables in input_vars will be the pruned program's inputs,
  and variables in output_vars will be the pruned program's outputs.
  Therefore, the pruning logic includes replacing the input of
  input_vars with the data op, and then preserving all connected
  ops starting from output_vars.

  Note: The returned program is the original program.
  If you do not want the original program to be modified,
  please pass in a cloned result.
*/
void PruneWithInput(const std::vector<pir::Value> &input_vars,
                    const std::vector<pir::Value> &output_vars,
                    Program *prog) {
  auto global_block = prog->block();
  std::vector<pir::Value> new_input_vars;
  if (!input_vars.empty()) {
    std::vector<pir::Value> new_input_vars;
    for (uint64_t idx = 0; idx < input_vars.size(); idx++) {
      auto input = input_vars[idx];
      auto origin_op = input.defining_op();
      std::string name = name_analysis::TryGetValueFirstName(input).value_or(
          "input_" + std::to_string(idx));
      auto new_input = AppendDataOp(global_block, input, name, *origin_op);
      input.ReplaceAllUsesWith(new_input);
      new_input_vars.push_back(new_input);
    }
  }
  VLOG(6) << "program after add new feed op = " << *prog;
  auto total_ops_list = global_block->ops();
  std::vector<pir::Operation *> total_ops(total_ops_list.begin(),
                                          total_ops_list.end());
  std::vector<bool> intersection_op_flags(total_ops.size(), true);
  std::set<pir::Value> output_vars_set(output_vars.begin(), output_vars.end());
  for (uint32_t index = total_ops.size() - 1; index != (uint32_t)(-1);
       --index) {
    auto op = total_ops[index];
    auto op_results = op->results();
    if (SomeInSet(op_results, output_vars_set)) {
      for (auto &operand : GetRealOpInputs(op)) {
        output_vars_set.insert(operand);
      }
    } else {
      VLOG(6) << "delete op " << index << ", name is " << op->name();
      intersection_op_flags[index] = false;
    }
  }

  std::set<pir::Value> input_vars_set(new_input_vars.begin(),
                                      new_input_vars.end());
  std::vector<pir::Operation *> remove_ops;
  for (uint32_t index = total_ops.size() - 1; index != (uint32_t)(-1);
       --index) {
    auto op = total_ops[index];
    if (!intersection_op_flags[index]) {
      auto op_results = op->results();
      if (!input_vars_set.empty() && SomeInSet(op_results, input_vars_set)) {
        PADDLE_THROW(common::errors::InvalidArgument(
            "The input_var create by: '{%s}' is not involved in the "
            "output_vars calculation"
            "Please remove it from input_vars.",
            op->name()));
      }
      global_block->erase(*op);
    }
  }
}

void SetIsTestAttr(const std::shared_ptr<Program> &prog) {
  for (auto &op : prog->block()->ops()) {
    if (op->HasAttribute("is_test")) {
      op->set_attribute(
          "is_test", pir::BoolAttribute::get(pir::IrContext::Instance(), true));
    }
  }
}

using ComputeReturnType = std::variant<float,
                                       double,
                                       int32_t,
                                       int64_t,
                                       bool,
                                       std::string,
                                       std::vector<int32_t>,
                                       std::vector<int64_t>,
                                       std::vector<float>,
                                       phi::DataType,
                                       phi::Place>;
ComputeReturnType CastPyObjectToAny(const pybind11::object &obj,
                                    const std::string &type_name) {
  static const std::unordered_map<
      std::string,
      std::function<ComputeReturnType(const pybind11::object &)>>
      type_casters = {
          {"float",
           [](const pybind11::object &obj) { return obj.cast<float>(); }},
          {"double",
           [](const pybind11::object &obj) { return obj.cast<double>(); }},
          {"int32",
           [](const pybind11::object &obj) { return obj.cast<int32_t>(); }},
          {"int64",
           [](const pybind11::object &obj) { return obj.cast<int64_t>(); }},
          {"bool",
           [](const pybind11::object &obj) { return obj.cast<bool>(); }},
          {"string",
           [](const pybind11::object &obj) { return obj.cast<std::string>(); }},
          {"vector<int32>",
           [](const pybind11::object &obj) {
             return obj.cast<std::vector<int32_t>>();
           }},
          {"vector<int64>",
           [](const pybind11::object &obj) {
             return obj.cast<std::vector<int64_t>>();
           }},
          {"vector<float>",
           [](const pybind11::object &obj) {
             return obj.cast<std::vector<float>>();
           }},
          {"datatype",
           [](const pybind11::object &obj) {
             return obj.cast<phi::DataType>();
           }},
          {"place",
           [](const pybind11::object &obj) { return obj.cast<phi::Place>(); }}};

  auto it = type_casters.find(type_name);
  if (it == type_casters.end()) {
    throw std::runtime_error("Unsupported type: " + type_name);
  }
  return it->second(obj);
}

void BindProgram(py::module *m) {
  static int64_t global_prog_seed = 0;
  py::class_<Program, std::shared_ptr<Program>> program(
      *m, "Program", py::dynamic_attr(), R"DOC(
    Create Python Program. Program is an abstraction of model structure, divided into
    computational graphs and weights. The Program has a main block that stores the computational
    graphs.

    A set of Program usually contains startup program and main program.
    A startup program is set to contain some initial work, eg. initialize the ``Parameter``, and the main
    program will contain the network structure and vars for train.

    A set of Program can be used for test or train, in train program ,
    Paddle will contain all content to build a train network,  in test
    program Paddle will prune some content which is irrelevant to test, eg.
    backward ops and vars.

    **Notes**:
        **we have** :ref:`api_paddle_static_default_startup_program` **and** :ref:`api_paddle_static_default_main_program`
        **by default, a pair of them will shared the parameters. The** :ref:`api_paddle_static_default_startup_program` **only run once to initialize parameters,**
        :ref:`api_paddle_static_default_main_program` **run in every mini batch and adjust the weights.**

    Returns:
        Program: An empty Program.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> main_program = static.Program()
            >>> startup_program = static.Program()
            >>> with static.program_guard(main_program=main_program, startup_program=startup_program):
            ...    x = static.data(name="x", shape=[-1, 784], dtype='float32')
            ...    y = static.data(name="y", shape=[-1, 1], dtype='int32')
            ...    z = static.nn.fc(name="fc", x=x, size=10, activation="relu")
            >>> print("main program is: {}".format(main_program))
            >>> print("start up program is: {}".format(startup_program))
  )DOC");
  program
      .def(py::init([]() {
        auto prog = std::make_shared<Program>(pir::IrContext::Instance());
        SetProgramInt64Attr(prog, "random_seed", global_prog_seed);
        return prog;
      }))
      .def("__str__",
           [](const std::shared_ptr<Program> &self) {
             std::ostringstream print_stream;
             self->Print(print_stream);
             return print_stream.str();
           })
      .def("__repr__",
           [](const std::shared_ptr<Program> &self) {
             std::ostringstream print_stream;
             self->Print(print_stream);
             return print_stream.str();
           })
      .def("parameters_num",
           [](const std::shared_ptr<Program> &self) {
             return self->parameters_num();
           })
      .def("set_is_test_attr",
           [](const std::shared_ptr<Program> &self) { SetIsTestAttr(self); })
      .def("set_parameters_from",
           [](const std::shared_ptr<Program> &self,
              const std::shared_ptr<Program> &other) {
             self->set_parameters(other->parameters());
           })
      .def(
          "global_block",
          [](std::shared_ptr<Program> self) { return self->block(); },
          return_value_policy::reference)
      .def("clone", [](Program &self) { return Clone(self); })
      .def("clone",
           [](Program &self, IrMapping &ir_mapper) {
             return Clone(self, &ir_mapper);
           })
      .def(
          "copy_to_block",
          [](std::shared_ptr<Program> self,
             pir::IrMapping &mapper,
             Block *block) { return self->CopyToBlock(mapper, block); },
          return_value_policy::reference)
      .def(
          "list_vars",
          [](std::shared_ptr<Program> self) {
            std::vector<pir::Value> vars;
            for (auto op : self->block()->ops()) {
              for (auto var : op->results()) {
                vars.push_back(var);
              }
            }
            return vars;
          },
          return_value_policy::reference)
      .def(
          "global_block",
          [](const std::shared_ptr<Program> &self) { return self->block(); },
          return_value_policy::reference)
      .def_property(
          "random_seed",
          [](const std::shared_ptr<Program> &self) {
            return GetProgramInt64Attr(self, "random_seed", 0);
          },
          [](std::shared_ptr<Program> self, int64_t random_seed) {
            SetProgramInt64Attr(self, "random_seed", random_seed);
          })
      .def_property(
          "_seed",
          [](const std::shared_ptr<Program> &self) {
            return GetProgramInt64Attr(self, "random_seed", 0);
          },
          [](std::shared_ptr<Program> self, int64_t random_seed) {
            SetProgramInt64Attr(self, "random_seed", random_seed);
          })
      .def("global_seed",
           [](std::shared_ptr<Program> self, int64_t random_seed) {
             global_prog_seed = random_seed;
             SetProgramInt64Attr(self, "random_seed", random_seed);
           })
      .def_property_readonly(
          "num_blocks",
          [](const std::shared_ptr<Program> &self) {
            size_t num_blocks = 0;
            auto top_level_op = self->module_op();
            for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
              auto &region = top_level_op->region(i);
              num_blocks += region.size();
            }
            return num_blocks;
          })
      .def_property_readonly(
          "blocks",
          [](const std::shared_ptr<Program> &self) {
            // Note: We only return global block currently.
            py::list op_list;
            op_list.append(self->block());
            return op_list;
          },
          return_value_policy::reference)
      .def(
          "get_value_by_op_id",
          [](Program &self, py::object op_ids) {
            std::vector<int> op_ids_list;
            if (py::isinstance<py::int_>(op_ids)) {
              op_ids_list.push_back(op_ids.cast<int>());
            } else if (py::isinstance<py::list>(op_ids)) {
              for (auto item : op_ids) {
                op_ids_list.push_back(item.cast<int>());
              }
            } else {
              PADDLE_THROW(
                  "Invalid op_ids format. Please provide either a single "
                  "integer or a list of integers.");
            }

            std::list<Operation *> all_ops = self.block()->get_recursive_ops();
            std::vector<pir::Value> value_list;

            for (auto op : all_ops) {
              if (std::find(op_ids_list.begin(), op_ids_list.end(), op->id()) !=
                  op_ids_list.end()) {
                for (auto value : op->results()) {
                  value_list.push_back(value);
                }
              }
            }

            if (value_list.empty()) {
              PADDLE_THROW(
                  "Can't find the corresponding opresult from the op ids");
            }
            return value_list;
          })
      .def("get_output_value_by_name",
           [](Program &self, const std::string &name) {
             return name_analysis::GetOutputValueByName(self, name);
           })
      .def("get_parameter_value_by_name",
           [](Program &self, const std::string &name) {
             return name_analysis::GetParameterValueByName(self, name);
           })
      .def("get_all_parameter_values",
           [](Program &self) {
             return name_analysis::GetAllParameterValues(self);
           })
      .def("num_ops", [](Program &self) { return self.num_ops(); })
      .def(
          "_state_dict",
          [](std::shared_ptr<Program> self,
             const std::string &mode = "all",
             const framework::Scope &scope = framework::Scope()) {
            std::unordered_map<std::string, phi::DenseTensor> state_dict_all;
            std::unordered_map<std::string, phi::DenseTensor> state_dict_param;
            std::unordered_map<std::string, phi::DenseTensor> state_dict_opt;
            for (auto op : self->block()->ops()) {
              for (auto var : op->results()) {
                auto is_persistable =
                    var.attribute<BoolAttribute>(kAttrIsPersistable);
                if (is_persistable && is_persistable.data()) {
                  if (var.defining_op()->isa<::pir::ParameterOp>()) {
                    std::string var_name =
                        name_analysis::GetValueFirstName(var);
                    auto tensor =
                        scope.FindVar(var_name)->GetMutable<phi::DenseTensor>();
                    state_dict_param[var_name] = *tensor;
                    state_dict_all[var_name] = *tensor;
                  } else if (var.defining_op()
                                 ->isa<paddle::dialect::DataOp>()) {
                    std::string var_name =
                        name_analysis::GetValueFirstName(var);
                    auto tensor =
                        scope.FindVar(var_name)->GetMutable<phi::DenseTensor>();
                    state_dict_opt[var_name] = *tensor;
                    state_dict_all[var_name] = *tensor;
                  }
                }
              }
            }
            if (mode == "all") {
              return state_dict_all;
            } else if (mode == "param") {
              return state_dict_param;
            } else if (mode == "opt") {
              return state_dict_opt;
            } else {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The mode is not supported."));
            }
          })
      .def(
          "set_state_dict",
          [](std::shared_ptr<Program> self,
             const std::unordered_map<std::string, phi::DenseTensor>
                 &state_dict,
             const framework::Scope &scope = framework::Scope(),
             bool copy_tensor = false) {
            for (auto item : state_dict) {
              auto var = scope.FindVar(item.first);
              if (var == nullptr) {
                PADDLE_THROW(common::errors::NotFound(
                    "The variable %s is not found.", item.first));
              } else {
                if (copy_tensor) {
                  auto *mutable_tensor = var->GetMutable<phi::DenseTensor>();
                  paddle::framework::TensorCopy(
                      item.second, item.second.place(), mutable_tensor);
                } else {
                  *var->GetMutable<phi::DenseTensor>() = item.second;
                }
              }
            }
          },
          py::arg("state_dict"),
          py::arg("scope"),
          py::arg("copy_tensor") = false)
      .def(
          "_prune",
          [](Program &self, std::vector<pir::Value> output_vars) {
            std::vector<pir::Value> input_vars;
            PruneWithInput(input_vars, output_vars, &self);
            return &self;
          },
          py::arg("targets"),
          "A description for the _prune method")
      .def(
          "_prune_with_input",
          [](Program &self,
             std::vector<pir::Value> input_vars,
             std::vector<pir::Value> output_vars) {
            PruneWithInput(input_vars, output_vars, &self);
            return &self;
          },
          py::arg("feeded_vars"),
          py::arg("targets"))
      .def("_sync_with_cpp", [](const std::shared_ptr<Program> &self) {
        // It's not need _sync_with_cpp in pir, but it's necessary in old static
        // graph. Add empty function to avoid python call error.
      });
}

void RefreshOpStopgradients(Operation *op) {
  if (op->num_operands() == 0 || op->isa<pir::ParameterOp>() ||
      op->isa<paddle::dialect::UniformOp>()) {
    return;
  } else if (op->isa<pir::SliceOp>()) {
    op->dyn_cast<pir::SliceOp>().RefreshStopGradients();
  } else if (op->isa<pir::SplitOp>()) {
    op->dyn_cast<pir::SplitOp>().RefreshStopGradients();
  } else {
    RefreshStopGradientsDefaultly(op);
  }
}

void BindBlock(py::module *m) {
  py::class_<Block> block(*m, "Block", R"DOC(
    In IR, a Block has a list of Operation and can represent a sub computational graph.

    Notes:
        The constructor of Block should not be invoked directly. You can
        use `Program.block()` to get a block.
  )DOC");
  block.def("empty", &Block::empty)
      .def(
          "__str__",
          [](Block &self) {
            std::ostringstream print_stream;
            pir::IrPrinter printer(print_stream);
            printer.PrintBlock(self);
            return print_stream.str();
          },
          return_value_policy::reference)
      .def(
          "front",
          [](Block &self) { return &self.front(); },
          return_value_policy::reference)
      .def(
          "back",
          [](Block &self) { return &self.back(); },
          return_value_policy::reference)
      .def_property_readonly(
          "parent_op",
          [](Block &self) { return self.GetParentOp(); },
          return_value_policy::reference)
      .def_property_readonly(
          "program",
          [](Block &self) { return self.GetParentOp()->GetParentProgram(); },
          return_value_policy::reference)
      .def_property_readonly(
          "parent_block",
          [](Block &self) { return self.GetParentOp()->GetParent(); },
          return_value_policy::reference)
      .def_property_readonly("ops",
                             [](Block &self) -> py::list {
                               py::list op_list;
                               for (auto &op : self) {
                                 op_list.append(&op);
                               }
                               return op_list;
                             })
      .def("num_ops", [](Block &self) { return self.num_ops(); })
      .def(
          "__enter__",
          [](Block &self) -> Block & {
            ApiBuilder::Instance().PushInsertionPoint();
            ApiBuilder::Instance().SetInsertionPointToBlockEnd(&self);
            return self;
          },
          return_value_policy::reference)
      .def("__exit__",
           [](Block &self, py::object, py::object, py::object) {
             ApiBuilder::Instance().LoadInsertionPoint();
           })
      .def("__len__", [](Block &self) { return self.size(); })
      .def("args", &Block::args, return_value_policy::reference)
      .def("kwargs", &Block::kwargs, return_value_policy::reference)
      .def("add_arg", &Block::AddArg)
      .def("add_kwarg", &Block::AddKwarg)
      .def("erase_kwarg", &Block::EraseKwarg)
      .def("get_values_by_op_idx",
           [](Block &self, const py::list &op_idxs) -> py::list {
             py::list value_list;
             auto it = self.begin();
             std::set<int> idxs_set;
             for (py::handle item : op_idxs) {
               idxs_set.insert(item.cast<int>());
             }
             for (int i = 0; it != self.end(); ++i, ++it) {
               if (idxs_set.find(i) != idxs_set.end()) {
                 for (uint32_t j = 0; j < it->num_results(); ++j) {
                   value_list.append(static_cast<pir::Value>(it->result(j)));
                 }
               }
             }
             return value_list;
           })
      .def("remove_op",
           [](Block &self, const Operation &op) { self.erase(op); })
      .def(
          "move_op",
          [](Block &self, Operation *op, uint32_t offset) {
            Block::Iterator position = self.begin();
            std::advance(position, offset);
            op->MoveTo(&self, position);
          },
          R"DOC(
          Move an op to a specific position (block.begin() + offset).

          Args:
              op (pir.Operation): the operator to be moved.
              offset (uint32_t) : offset relative to the begin of the block

          Returns:
              None

        )DOC")
      .def(
          "move_op_to_block_end",
          [](Block &self, Operation *op) { op->MoveTo(&self, self.end()); },
          R"DOC(
            Move an op to the end of the block.

            Args:
                op (pir.Operation): The operator to be moved.

            Returns:
                None

            )DOC")
      .def("all_parameters",
           [](Block &self) -> py::list {
             py::list param_list;
             for (auto &op : self) {
               if (op.name() == "builtin.parameter" &&
                   op.HasAttribute(kAttrIsPersistable)) {
                 auto attrs = op.attribute(kAttrIsPersistable)
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .AsVector();
                 for (uint32_t i = 0; i < attrs.size(); i++) {
                   bool is_persistable =
                       attrs[i].dyn_cast<pir::BoolAttribute>().data();
                   if (is_persistable) {
                     param_list.append(static_cast<pir::Value>(op.result(i)));
                   }
                 }
               }
             }
             return param_list;
           })
      .def("refresh_stopgradient",
           [](Block &self) {
             for (auto &op : self) {
               RefreshOpStopgradients(&op);
             }
           })
      .def("_sync_with_cpp", [](const Block &self) {
        // It's not need _sync_with_cpp in pir, but it's necessary in old static
        // graph. Add empty function to avoid python call error.
      });
}

void BindIrMapping(py::module *m) {
  py::class_<IrMapping> ir_mapping(*m, "IrMapping");
  ir_mapping.def(py::init<>())
      .def("look_up",
           [](IrMapping &self, Value from) { return self.Lookup(from); })
      .def("has", [](IrMapping &self, Value from) { return self.Has(from); })
      .def("add",
           [](IrMapping &self, Value from, Value to) {
             self.Add<Value>(from, to);
           })
      .def("size",
           [](IrMapping &self) { return self.GetMutableMap<Value>().size(); });
}

void BindCloneOptions(py::module *m) {
  py::class_<CloneOptions> clone_options(*m, "CloneOptions");
  clone_options.def(
      "__init__",
      [](CloneOptions &self,
         bool clone_regions,
         bool clone_operands,
         bool clone_successors) {
        new (&self)
            CloneOptions(clone_regions, clone_operands, clone_successors);
      },
      return_value_policy::reference);
}

void BindOperation(py::module *m) {
  py::class_<Operation> op(*m, "Operation", R"DOC(
    In IR, all the operation are represented by Operation, and Operation
    is regarded as a build in an instruction of a Block. Users can call
    python api to describe their neural network.

    Notes:
        The constructor of operator should not be invoked directly. Use
        python api, for example: paddle.mean for building mean operation.

  )DOC");
  op.def("name", &Operation::name)
      .def("get_parent_block",
           &Operation::GetParent,
           return_value_policy::reference)
      .def("num_operands", &Operation::num_operands)
      .def("num_results", &Operation::num_results)
      .def("num_regions", &Operation::num_regions)
      .def("id", &Operation::id)
      .def("operand", &Operation::operand)
      .def("result",
           [](Operation &self, uint32_t index) {
             return static_cast<pir::Value>(self.result(index));
           })
      .def("operand_source", &Operation::operand_source)
      .def("operands", &Operation::operands)
      .def("results",
           [](Operation &self) -> py::list {
             py::list value_list;
             for (uint32_t i = 0; i < self.num_results(); i++) {
               value_list.append(static_cast<pir::Value>(self.result(i)));
             }
             return value_list;
           })
      .def(
          "blocks",
          [](Operation &self) { return &self.blocks(); },
          return_value_policy::reference)
      .def("has_attr", &Operation::HasAttribute)
      .def("str_attr",
           [](Operation &self, const std::string &attr_name) -> py::object {
             auto str_attr = self.attribute<StrAttribute>(attr_name);
             if (str_attr) {
               return py::cast(str_attr.AsString());
             } else {
               return py::cast<py::none>(Py_None);
             }
           })
      .def("int_attr",
           [](Operation &self, const std::string &attr_name) -> py::object {
             auto int_attr = self.attribute<Int32Attribute>(attr_name);
             if (int_attr) {
               return py::cast(int_attr.data());
             } else {
               return py::cast<py::none>(Py_None);
             }
           })
      .def("set_bool_attr",
           [](Operation &self, std::string &attr_name, bool flag) {
             self.set_attribute(
                 attr_name,
                 pir::BoolAttribute::get(pir::IrContext::Instance(), flag));
           })
      .def("set_int_array_attr",
           [](Operation &self,
              std::string &attr_name,
              const std::vector<int64_t> &val) {
             auto attr = IntArrayAttribute::get(pir::IrContext::Instance(),
                                                phi::IntArray(val));
             self.set_attribute(attr_name, attr);
           })
      .def("set_str_array_attr",
           [](Operation &self,
              std::string &attr_name,
              const std::vector<std::string> &val) {
             std::vector<Attribute> val_attr;
             for (auto &str : val) {
               val_attr.emplace_back(
                   StrAttribute::get(pir::IrContext::Instance(), str));
             }
             auto attr =
                 pir::ArrayAttribute::get(pir::IrContext::Instance(), val_attr);
             self.set_attribute(attr_name, attr);
           })
      .def("set_str_attr",
           [](Operation &self, std::string &attr_name, std::string &val) {
             self.set_attribute(
                 attr_name, StrAttribute::get(pir::IrContext::Instance(), val));
           })
      .def("set_int_attr",
           [](Operation &self, std::string &attr_name, const int &val) {
             self.set_attribute(
                 attr_name,
                 pir::Int32Attribute::get(pir::IrContext::Instance(), val));
           })
      .def("erase_attr",
           [](Operation &self, std::string &attr_name) {
             self.erase_attribute(attr_name);
           })
      .def("attrs",
           [](Operation &self) -> py::dict {
             py::dict attrs_dict;
             for (auto &pair : self.attributes()) {
               // SymbolAttribute is only used in PIR, no need to pass to Python
               if (pair.second.isa<pir::shape::SymbolAttribute>()) continue;
               if (pair.first == kAttrOpDistAttr) {
                 attrs_dict[pair.first.c_str()] =
                     pair.second.dyn_cast<OperationDistAttribute>();
               } else {
                 attrs_dict[pair.first.c_str()] =
                     paddle::dialect::GetAttributeData(pair.second);
               }
             }
             return attrs_dict;
           })
      .def("copy_attrs_from",
           [](Operation &self, Operation &other) {
             for (auto &pair : other.attributes()) {
               self.set_attribute(pair.first, pair.second);
             }
           })
      .def("set_execution_stream",
           [](Operation &self, const std::string &exe_stream) {
             self.set_attribute(
                 "execution_stream",
                 StrAttribute::get(pir::IrContext::Instance(), exe_stream));
           })
      .def("set_scheduling_priority",
           [](Operation &self, int64_t priority) {
             self.set_attribute("scheduling_priority",
                                pir::Int64Attribute::get(
                                    pir::IrContext::Instance(), priority));
           })
      .def("operands_source",
           [](Operation &self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self.num_operands(); i++) {
               op_list.append(self.operand_source(i));
             }
             return op_list;
           })
      .def("get_input_names",
           [](Operation &self) -> py::list {
             if (self.HasInterface<paddle::dialect::OpYamlInfoInterface>() ==
                 false) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Currently, we can only get input names of Operation that "
                   "has OpYamlInfoInterface"));
             }

             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto inputs_info = std::get<0>(yaml_interface.GetOpInfo());
             for (auto &input_info : inputs_info) {
               op_list.append(input_info.name);
             }
             return op_list;
           })
      .def("get_attr_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto attrs_info = std::get<1>(yaml_interface.GetOpInfo());
             for (auto &attr_info : attrs_info) {
               op_list.append(attr_info.name);
             }
             return op_list;
           })
      .def("get_output_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto outputs_info = std::get<2>(yaml_interface.GetOpInfo());
             for (auto &output_info : outputs_info) {
               op_list.append(output_info.name);
             }
             return op_list;
           })
      .def("get_output_intermediate_status",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto outputs_info = std::get<2>(yaml_interface.GetOpInfo());
             for (auto &output_info : outputs_info) {
               op_list.append(output_info.intermediate);
             }
             return op_list;
           })
      .def("get_input_grad_semantics",
           [](Operation &self) -> py::list {
             if (self.HasInterface<paddle::dialect::OpYamlInfoInterface>() ==
                 false) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Currently, we can only get input grad semantics of "
                   "Operation that "
                   "has OpYamlInfoInterface"));
             }
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto inputs_grad_info = std::get<0>(yaml_interface.GetOpInfo());
             for (auto &input_grad_info : inputs_grad_info) {
               op_list.append(input_grad_info.with_grad_semantic);
             }
             return op_list;
           })
      .def("replace_all_uses_with",
           [](Operation &self, const std::vector<Value> &values) {
             self.ReplaceAllUsesWith(values);
           })
      .def("as_if_op",
           [](Operation &self) { return PyIfOp(self.dyn_cast<IfOp>()); })
      .def("as_pylayer_op",
           [](Operation &self) -> PyLayerOp {
             auto pylayer_op = self.dyn_cast<PyLayerOp>();
             if (!pylayer_op) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Can't cast non-pylayer_op type Operation to PyLayerOp."));
             }
             return pylayer_op;
           })
      .def("as_while_op",
           [](Operation &self) { return PyWhileOp(self.dyn_cast<WhileOp>()); })
      .def(
          "as_tuple_pop_op",
          [](Operation &self) -> TuplePopOp {
            auto tuple_pop_op = self.dyn_cast<TuplePopOp>();
            if (!tuple_pop_op) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "Can't cast non-tuple_pop_op type Operation to TuplePopOp."));
            }
            return tuple_pop_op;
          })
      .def("__repr__",

           [](Operation &self) {
             std::ostringstream print_stream;
             print_stream << "Operation(";
             self.Print(print_stream);
             print_stream << ")";
             return print_stream.str();
           })
      .def(
          "clone",
          [](Operation &self, IrMapping &ir_mapping, CloneOptions options) {
            auto op = self.Clone(ir_mapping, options);
            return ApiBuilder::Instance().GetBuilder()->Insert(op);
          },
          return_value_policy::reference)
      .def("erase", &Operation::Erase)
      .def("move_before",
           [](Operation &self, Operation &other) {
             self.MoveTo(other.GetParent(), Block::Iterator{other});
           })
      .def_property(
          "callstack",
          [](Operation &self) -> py::list {
            py::list callstack_list;
            if (!self.HasAttribute(paddle::framework::OpProtoAndCheckerMaker::
                                       OpCreationCallstackAttrName())) {
              return callstack_list;
            }
            pir::Attribute op_callstack = self.attribute<pir::Attribute>(
                paddle::framework::OpProtoAndCheckerMaker::
                    OpCreationCallstackAttrName());
            PADDLE_ENFORCE(op_callstack.isa<pir::ArrayAttribute>(),
                           common::errors::PreconditionNotMet(
                               "The callstack of operation `%s` should be an "
                               "array attribute.",
                               self.name()));
            auto op_callstack_array_attr =
                op_callstack.dyn_cast<pir::ArrayAttribute>();
            for (size_t i = 0; i < op_callstack_array_attr.size(); ++i) {
              PADDLE_ENFORCE(
                  op_callstack_array_attr.at(i).isa<StrAttribute>(),
                  common::errors::PreconditionNotMet(
                      "The callstack info of operation `%s` should be array of "
                      "string attribute.",
                      self.name()));
              callstack_list.append(op_callstack_array_attr.at(i)
                                        .dyn_cast<StrAttribute>()
                                        .AsString());
            }
            return callstack_list;
          },
          [](Operation &self,
             const std::vector<std::string> &callstack) -> void {
            std::vector<pir::Attribute> op_callstack_infos;
            for (auto str : callstack) {
              op_callstack_infos.push_back(
                  StrAttribute::get(pir::IrContext::Instance(), str));
            }

            self.set_attribute(
                paddle::framework::OpProtoAndCheckerMaker::
                    OpCreationCallstackAttrName(),
                pir::ArrayAttribute::get(pir::IrContext::Instance(),
                                         op_callstack_infos));
          })
      .def_property(
          "dist_attr",
          [](Operation &self) -> py::object {
            if (self.HasAttribute(kAttrOpDistAttr)) {
              return py::cast(
                  self.attribute<OperationDistAttribute>(kAttrOpDistAttr));
            } else {
              return py::cast<py::none>(Py_None);
            }
          },
          [](Operation &self, OperationDistAttribute op_dist_attr) {
            self.set_attribute(kAttrOpDistAttr, op_dist_attr);
          })
      .def_property(
          "op_role",
          [](Operation &self) -> py::object {
            auto int_attr = self.attribute<Int32Attribute>("op_role");
            if (int_attr) {
              return py::cast(int_attr.data());
            } else {
              return py::cast(-1);
            }
          },
          [](Operation &self, const int &op_role) {
            self.set_attribute(
                "op_role",
                Int32Attribute::get(pir::IrContext::Instance(), op_role));
          })
      .def_property(
          "chunk_id",
          [](Operation &self) -> py::object {
            auto int_attr = self.attribute<Int32Attribute>("chunk_id");
            if (int_attr) {
              return py::cast(int_attr.data());
            } else {
              return py::cast(-1);
            }
          },
          [](Operation &self, const int &chunk_id) {
            self.set_attribute(
                "chunk_id",
                Int32Attribute::get(pir::IrContext::Instance(), chunk_id));
          })
      .def("is_no_need_buffer",
           [](Operation &self, const Value &operand_source) -> bool {
             paddle::dialect::OpYamlInfoInterface op_info_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             std::unique_ptr<paddle::dialect::OpYamlInfoParser> info_parser(
                 nullptr);
             if (op_info_interface) {
               info_parser =
                   std::make_unique<paddle::dialect::OpYamlInfoParser>(
                       op_info_interface.GetOpInfo(),
                       paddle::dialect::IsLegacyOp(self.name()));
               auto &no_need_buffer_ids = info_parser->NoNeedBufferIds();
               for (auto no_need_buffer_id : no_need_buffer_ids) {
                 if (operand_source == self.operand_source(no_need_buffer_id)) {
                   return true;
                 }
               }
             }
             return false;
           });
  py::class_<Operation::BlockContainer> block_container(
      *m, "Operation_BlockContainer", R"DOC(
    The Operation_BlockContainer only use to walk all blocks in the operation.
     )DOC");
  block_container.def(
      "__iter__",
      [](Operation::BlockContainer &self) {
        return py::make_iterator(self.begin(), self.end());
      },
      py::keep_alive<0, 1>());
}

py::str Value2String(Value self) {
  std::ostringstream print_stream;
  print_stream << "Value(";
  print_stream << GetValueInfo(self);
  print_stream << ")";
  return print_stream.str();
}

const phi::DDim &GetTensorDims(Type type) {
  if (!type) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The type used to get dims is nullptr."));
  }
  if (auto dense_type = type.dyn_cast<DenseTensorType>()) {
    return dense_type.dims();
  } else if (auto select_rows_type = type.dyn_cast<SelectedRowsType>()) {
    return select_rows_type.dims();
  } else if (auto sparse_coo_tensor_type =
                 type.dyn_cast<SparseCooTensorType>()) {
    return sparse_coo_tensor_type.dims();
  } else if (auto sparse_csr_tensr_type =
                 type.dyn_cast<SparseCsrTensorType>()) {
    return sparse_csr_tensr_type.dims();
  } else if (auto dense_array_type = type.dyn_cast<DenseTensorArrayType>()) {
    return dense_array_type.dims();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get shape for dense and selsect rows type."));
  }
}
const phi::DDim &GetValueDims(Value value) {
  return GetTensorDims(value.type());
}

pir::Value apply(Value self, py::object func) {
  py::gil_scoped_acquire gil;
  auto stop_gradient = self.attribute<BoolAttribute>(kAttrStopGradients);
  if (stop_gradient && !stop_gradient.data()) {
    PADDLE_THROW(common::errors::Unavailable(
        "Cannot apply function on a tensor that required gradient."));
  }
  PyObject *py_func = func.release().ptr();
  Py_INCREF(py_func);
  PyObject *res = nullptr;
  try {
    py::object obj = py::cast(self);
    PyObject *tmp_self = obj.release().ptr();
    Py_INCREF(tmp_self);
    res = PyObject_CallFunctionObjArgs(py_func, tmp_self, nullptr);
    Py_DECREF(tmp_self);
  } catch (std::exception &e) {
    PADDLE_THROW(common::errors::Unavailable(
        "Apply function of Tensor raises an exception: %s.", e.what()));
  } catch (...) {
    PADDLE_THROW(common::errors::Fatal(
        "Apply function of Tensor raises an unknown exception."));
  }
  if (res == Py_None) {
    return self;
  }
  auto out = CastPyArg2Value(res, "", 0, false);
  Py_DECREF(py_func);
  Py_DECREF(res);
  return out;
}

#define DEF_VALUE_BOOL_PROPERTY(name)                                         \
  def_property(                                                               \
      name,                                                                   \
      [](Value self) {                                                        \
        auto bool_data = self.attribute<BoolAttribute>(name);                 \
        return bool_data && bool_data.data();                                 \
      },                                                                      \
      [](Value self, bool bool_data) {                                        \
        self.set_attribute(                                                   \
            name, BoolAttribute::get(pir::IrContext::Instance(), bool_data)); \
      })

#define DEF_VALUE_STOP_GRADIENT_PROPERTY(name)                                \
  def_property(                                                               \
      name,                                                                   \
      [](Value self) {                                                        \
        auto bool_data = self.attribute<BoolAttribute>(name);                 \
        return !bool_data || bool_data.data();                                \
      },                                                                      \
      [](Value self, bool bool_data) {                                        \
        self.set_attribute(                                                   \
            name, BoolAttribute::get(pir::IrContext::Instance(), bool_data)); \
      })

#define DEF_VALUE_POINTER_PROPERTY(name)                                     \
  def_property(                                                              \
      name,                                                                  \
      [](Value self) -> py::object {                                         \
        auto prop_ptr = self.property(name);                                 \
        if (!prop_ptr) {                                                     \
          return py::cast<py::none>(Py_None);                                \
        }                                                                    \
        auto py_data = reinterpret_cast<PyObject *>(prop_ptr);               \
        py::object obj =                                                     \
            py::reinterpret_borrow<py::object>(py::handle(py_data));         \
        return obj;                                                          \
      },                                                                     \
      [](Value self, py::object obj) {                                       \
        pir::PropertiesDeleter deleter = [](void *python_obj) {              \
          Py_DECREF(python_obj);                                             \
        };                                                                   \
        PyObject *pointer_data = obj.release().ptr();                        \
        pir::Property value_property(reinterpret_cast<void *>(pointer_data), \
                                     deleter);                               \
        self.set_property(name, value_property);                             \
      })

void BindValue(py::module *m) {
  py::class_<Value> value(*m,
                          "Value",
                          R"DOC(
    Value class represents the SSA value in the IR system. It is a directed edge
    and a base class.

    Notes:
        The constructor of Value should not be invoked directly. Value can be automatically constructed
        when build network.

  )DOC");
  g_ir_value_pytype = reinterpret_cast<PyTypeObject *>(value.ptr());
  value.def(py::init<>())
      .def(py::init([](Value value) { return value; }))
      .def_property_readonly(
          "block",
          [](Value self) {
            if (auto op_result = self.dyn_cast<OpResult>()) {
              return op_result.owner()->GetParent();
            }
            return self.dyn_cast<BlockArgument>().owner();
          },
          return_value_policy::reference)
      .def_property_readonly(
          "id",
          [](Value self) {
            if (self.impl() == nullptr) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "Currently, we can only get id of Value whose impl "
                  "is not nullptr"));
            } else {
              std::stringstream ss;
              ss << std::hex << self.impl();
              return ss.str();
            }
          })
      .def_property(
          "name",
          [](Value self) -> std::string {
            return name_analysis::GetValueFirstName(self);
          },
          [](Value self, const std::string &name) {
            name_analysis::SetValueName(self, name);
          })
      .def_property_readonly(
          "has_name",
          [](Value self) {
            return name_analysis::TryGetValueFirstName(self).has_value();
          })
      // Return all Maybe names of given Value, for example:
      // DataOp("var_1") -> %0 -> shadow_output("output_2")
      // Return ["var_1", "output_2"]
      .def_property_readonly("_names",
                             [](Value self) -> py::list {
                               std::vector<std::string> names =
                                   name_analysis::GetValueAllNames(self);
                               return py::cast(names);
                             })
      .def_property(
          "shape",
          [](Value self) { return phi::vectorize(GetValueDims(self)); },
          [](Value self, const std::vector<int> &shape) {
            PADDLE_THROW(common::errors::InvalidArgument(
                "can't set shape when building static graph"));
          })
      .def_property(
          "_local_shape",
          [](Value self) {
            if (!self.type().isa<DistDenseTensorType>()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "_local_shape is only for distdense tensor."));
            }
            return phi::vectorize(
                self.type().dyn_cast<DistDenseTensorType>().local_ddim());
          },
          [](Value self, const std::vector<int> &shape) {
            PADDLE_THROW(common::errors::InvalidArgument(
                "can't set _local_shape when building static graph"));
          })
      .def_property(
          "dtype",
          [](Value self) { return pir::GetValueDtype(self); },
          [](Value self, phi::DataType dtype) {
            PADDLE_THROW(common::errors::InvalidArgument(
                "can't set dtype when building static graph"));
          })
      .def_property(
          "place_attr",
          [](Value self) -> phi::Place {
            auto place_attr = self.attribute<PlaceAttribute>("place");
            return place_attr ? place_attr.data() : phi::Place();
          },
          [](Value self, const phi::Place &place) {
            // auto place = CastPyArg2Place(place_obj.release().ptr(), 1);
            auto place_attr =
                dialect::PlaceAttribute::get(pir::IrContext::Instance(), place);
            self.set_attribute("place", place_attr);
          })
      .def("initialized",
           [](Value self) {
             if (self.impl() == nullptr || self.type().storage() == nullptr) {
               return false;
             } else {
               return true;
             }
           })
      .DEF_VALUE_STOP_GRADIENT_PROPERTY("stop_gradient")
      .DEF_VALUE_BOOL_PROPERTY("trainable")
      .DEF_VALUE_BOOL_PROPERTY("persistable")
      .DEF_VALUE_BOOL_PROPERTY("need_clip")
      .DEF_VALUE_BOOL_PROPERTY("is_distributed")
      .DEF_VALUE_BOOL_PROPERTY("is_parameter")
      .DEF_VALUE_POINTER_PROPERTY("optimize_attr")
      .DEF_VALUE_POINTER_PROPERTY("regularizer")
      .DEF_VALUE_POINTER_PROPERTY("do_model_average")
      .def("all_used_ops",
           [](Value &self) -> py::list {
             py::list op_list;
             for (auto it = self.use_begin(); it != self.use_end(); ++it) {
               op_list.append(it.owner());
             }
             return op_list;
           })
      .def("all_used_ops_in_same_block",
           [](Value &self) -> py::list {
             py::list op_list;
             for (auto it = self.use_begin(); it != self.use_end(); ++it) {
               pir::Operation *used_op = it.owner();
               while (used_op->GetParent() != self.defining_op()->GetParent() &&
                      used_op->GetParent()->GetParentOp()) {
                 used_op = used_op->GetParent()->GetParentOp();
               }
               op_list.append(used_op);
             }
             return op_list;
           })
      .def(
          "get_defining_op",
          [](Value self) -> pir::Operation * { return self.defining_op(); },
          return_value_policy::reference)
      .def("type", &Value::type)
      .def("index",
           [](Value self) -> uint32_t {
             if (auto op_result = self.dyn_cast<OpResult>()) {
               return op_result.index();
             } else if (auto arg = self.dyn_cast<BlockArgument>()) {
               if (!arg.is_kwarg()) {
                 return arg.index();
               }
             }
             PADDLE_THROW(common::errors::InvalidArgument(
                 "only support accessing index from op_result or positional "
                 "block arg."));
           })
      .def("is_dense_tensor_type",
           [](Value self) { return self.type().isa<DenseTensorType>(); })
      .def("is_selected_row_type",
           [](Value self) { return self.type().isa<SelectedRowsType>(); })
      .def("is_sparse_coo_tensor_type",
           [](Value self) { return self.type().isa<SparseCooTensorType>(); })
      .def("is_sparse_csr_tensor_type",
           [](Value self) { return self.type().isa<SparseCsrTensorType>(); })
      .def("is_dense_tensor_array_type",
           [](Value self) { return self.type().isa<DenseTensorArrayType>(); })
      .def("is_dist_dense_tensor_type",
           [](Value self) { return self.type().isa<DistDenseTensorType>(); })
      .def("value_assign", [](Value &self, Value value) { self = value; })
      .def("replace_all_uses_with",
           [](Value self, Value value) { self.ReplaceAllUsesWith(value); })
      .def("replace_grad_users_with",
           [](Value self,
              Value value,
              std::unordered_set<Operation *> &grad_ops) {
             for (auto it = self.use_begin(); it != self.use_end();) {
               auto use_op = it.owner();
               if (grad_ops.find(use_op) != grad_ops.end()) {
                 (it++)->set_source(value);
               } else {
                 it++;
               }
             }
           })
      .def("set_type", [](Value self, Type type) { self.set_type(type); })
      .def("first_use", &Value::first_use, return_value_policy::reference)
      .def("has_one_use", &Value::HasOneUse)
      .def("use_empty", &Value::use_empty)
      .def("apply", &apply)
      .def("is_same", &Value::operator==)
      .def("hash", [](Value self) { return std::hash<pir::Value>{}(self); })
      .def("_rename", &name_analysis::RenameValue)
      .def("_has_only_one_name",
           [](Value self) -> bool {
             return name_analysis::HasOnlyOneValueName(self);
           })
      .def("detach",
           [](Value self) {
             auto share_data_op =
                 ApiBuilder::Instance()
                     .GetBuilder()
                     ->Build<paddle::dialect::ShareData_Op>(self);
             auto out = share_data_op.out();
             out.set_attribute(
                 kAttrStopGradients,
                 BoolAttribute::get(pir::IrContext::Instance(), true));
             return out;
           })
      .def("__repr__", &Value2String)
      .def("is_combine",
           [](Value self) { return self.type().isa<pir::VectorType>(); })
      .def("is_dist",
           [](Value self) { return self.type().isa<DistTypeInterface>(); })
      // The function will calculate the new local shape based on the global
      // shape and the dist_attr argument.
      .def("update_dist_attr",
           [](Value &self, Attribute dist_attr) {
             self.set_type(dialect::CvtToPirDistType(self.type(), dist_attr));
           })
      .def("is_coalesced",
           [](Value self) {
             auto sparse_coo_tensor_type =
                 self.type().dyn_cast<SparseCooTensorType>();
             if (sparse_coo_tensor_type) {
               return sparse_coo_tensor_type.coalesced();
             } else {
               PADDLE_THROW(common::errors::InvalidType(
                   "Method is_coalesced only support sparse coo tensor."));
             }
           })
      .def_property_readonly(
          "process_mesh",
          [](Value &self) -> py::object {
            auto type = self.type();
            if (auto dist_type = type.dyn_cast<DistTypeInterface>()) {
              return py::cast(dist_type.tensor_dist_attr()
                                  .process_mesh_attr()
                                  .process_mesh());
            } else {
              return py::cast<py::none>(Py_None);
            }
          })
      .def("_clone",
           [](Value self) {
             // Return a new value owned by python side
             return self;
           })
      .def("sparse_dim",
           [](Value self) -> int32_t {
             auto op_result = self.dyn_cast<OpResult>();
             pir::Operation *operation = op_result.owner();
             if (self.type().isa<SparseCooTensorType>() &&
                 operation->name() == "pd_op.sparse_coo_tensor_sp") {
               std::vector<Value> sources = operation->operands_source();
               Value non_zero_indices = sources[1];
               return phi::vectorize(GetValueDims(non_zero_indices))[0];
             } else if (self.type().isa<SparseCsrTensorType>()) {
               PADDLE_THROW(common::errors::InvalidType(
                   "SparseCsrTensor is unsupported in pir mode."));
             } else {
               return 0;
             }
           })
      .def("dense_dim", [](Value self) -> int32_t {
        auto op_result = self.dyn_cast<OpResult>();
        pir::Operation *operation = op_result.owner();
        if (self.type().isa<SparseCooTensorType>() &&
            operation->name() == "pd_op.sparse_coo_tensor_sp") {
          std::vector<Value> sources = operation->operands_source();
          Value non_zero_indices = sources[1];
          int32_t dims = phi::vectorize(GetValueDims(self)).size();
          return dims - phi::vectorize(GetValueDims(non_zero_indices))[0];
        } else if (self.type().isa<SparseCsrTensorType>()) {
          PADDLE_THROW(common::errors::InvalidType(
              "SparseCsrTensor is unsupported in pir mode."));
        } else {
          return phi::vectorize(GetValueDims(self)).size();
        }
      });
}

void BindOpOperand(py::module *m) {
  py::class_<OpOperand> op_operand(*m,
                                   "OpOperand",
                                   R"DOC(
    OpOperand class represents the op_operand (input) of operation.

    Notes:
        The constructor of OpOperand should not be invoked directly. OpOperand can be automatically constructed
        when build network.

  )DOC");
  op_operand.def("source", [](OpOperand &self) { return self.source(); })
      .def("set_source",
           [](OpOperand &self, Value *value) {
             value ? self.set_source(*value) : self.set_source(nullptr);
           })
      .def("owner", &OpOperand::owner, return_value_policy::reference)
      .def("index", &OpOperand::index);
}

bool GetValueBoolAttr(Value value, const std::string &attr_name) {
  auto bool_attr = value.attribute<BoolAttribute>(attr_name);
  return !bool_attr || bool_attr.data();
}

void BindType(py::module *m) {
  py::class_<Type> ir_type(*m, "Type");
  ir_type.def("__eq__", &Type::operator==)
      .def_property(
          "shape",
          [](Type self) { return phi::vectorize(GetTensorDims(self)); },
          [](Type self, const std::vector<int> &shape) {
            PADDLE_THROW(common::errors::InvalidArgument(
                "can't set shape when building static graph"));
          })
      .def_property(
          "dtype",
          [](Type self) { return GetTensorDtype(self); },
          [](Type self, phi::DataType dtype) {
            PADDLE_THROW(common::errors::InvalidArgument(
                "can't set dtype when building static graph"));
          })
      .def_property(
          "_local_shape",
          [](Type self) {
            if (!self.isa<DistDenseTensorType>()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "_local_shape is only for distdense tensor."));
            }
            return phi::vectorize(
                self.dyn_cast<DistDenseTensorType>().local_ddim());
          },
          [](Type self, const std::vector<int> &shape) {
            PADDLE_THROW(common::errors::InvalidArgument(
                "can't set _local_shape when building static graph"));
          })
      .def("as_vec_type",
           [](Type self) -> py::object {
             if (auto vec_type = self.dyn_cast<VectorType>()) {
               return py::cast(vec_type);
             }
             return py::cast<py::none>(Py_None);
           })
      .def("as_dist_type",
           [](Type &self) -> py::object {
             if (auto dist_type = self.dyn_cast<DistTypeInterface>()) {
               return py::cast(dist_type);
             }
             return py::cast<py::none>(Py_None);
           })
      .def("__str__", [](Type &self) {
        std::ostringstream print_stream;
        print_stream << self;
        return print_stream.str();
      });

  m->def("create_shaped_type",
         [](Type &type, const std::vector<int64_t> &shape) -> Type {
           if (type.isa<DenseTensorType>()) {
             DenseTensorType src_type = type.dyn_cast<DenseTensorType>();
             DenseTensorType dst_type =
                 DenseTensorType::get(pir::IrContext::Instance(),
                                      src_type.dtype(),
                                      phi::make_ddim(shape),
                                      src_type.data_layout(),
                                      src_type.lod(),
                                      src_type.offset());
             return dst_type;
           } else if (type.isa<SelectedRowsType>()) {
             SelectedRowsType src_type = type.dyn_cast<SelectedRowsType>();
             SelectedRowsType dst_type =
                 SelectedRowsType::get(pir::IrContext::Instance(),
                                       src_type.dtype(),
                                       phi::make_ddim(shape),
                                       src_type.data_layout(),
                                       src_type.lod(),
                                       src_type.offset());
             return dst_type;
           } else {
             PADDLE_THROW(common::errors::InvalidArgument(
                 "Currently, we can only set shape for dense tensor"));
           }
         });
}
void BindVectorType(py::module *m) {
  py::class_<VectorType, Type> vec_type(*m, "VectorType");
  vec_type.def("as_list", &VectorType::data);
  m->def("create_vec_type", [](std::vector<Type> &types) {
    return VectorType::get(pir::IrContext::Instance(), types);
  });
}
void BindAttribute(py::module *m) {
  py::class_<Attribute> ir_attr(*m, "Attribute", py::module_local());
  ir_attr.def(py::init<>())
      .def("__bool__", [](Attribute &self) { return static_cast<bool>(self); })
      .def("__eq__", &Attribute::operator==)
      .def("__str__",
           [](Attribute &self) {
             std::ostringstream print_stream;
             print_stream << self;
             return print_stream.str();
           })
      .def("as_tensor_dist_attr",
           [](Attribute &self) -> py::object {
             if (auto dist_attr = self.dyn_cast<TensorDistAttribute>()) {
               return py::cast(dist_attr);
             }
             return py::cast<py::none>(Py_None);
           })
      .def("as_array_attr", [](Attribute &self) -> py::object {
        if (auto array_attr = self.dyn_cast<ArrayAttribute>()) {
          return py::cast(array_attr);
        }
        return py::cast<py::none>(Py_None);
      });
  py::class_<ArrayAttribute, Attribute> array_attr(*m, "ArrayAttribute");
  array_attr.def("__len__", [](ArrayAttribute &self) { return self.size(); })
      .def("__getitem__",
           [](ArrayAttribute &self, int idx) { return self.at(idx); });
}

struct PyInsertionPoint {
  pir::InsertionPoint value;
};
void BindInsertionPoint(pybind11::module *m) {
  py::class_<PyInsertionPoint> ir_insertion_point(*m, "InsertionPoint", R"DOC(
    InsertionPoint class represents the insertion point in the Builder.)DOC");
  ir_insertion_point
      .def(
          "next",
          [](PyInsertionPoint &self) -> Operation & {
            if (self.value.second == self.value.first->end()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the end and can't call "
                  "next()."));
            }
            return *(self.value.second++);
          },
          return_value_policy::reference)
      .def(
          "prev",
          [](PyInsertionPoint &self) -> Operation & {
            if (self.value.second == self.value.first->begin()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the begin and can't call "
                  "prev()."));
            }
            return *(--self.value.second);
          },
          return_value_policy::reference)
      .def(
          "get_operation",
          [](PyInsertionPoint &self) -> Operation & {
            if (self.value.second == self.value.first->begin()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the begin."));
            } else if (self.value.second == self.value.first->end()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the end."));
            }
            return *(self.value.second);
          },
          return_value_policy::reference)
      .def(
          "block",
          [](PyInsertionPoint &self) { return self.value.first; },
          return_value_policy::reference);
}

template <typename F, typename S>
void range_block_do(const Block *block,
                    std::pair<size_t, size_t> range,
                    F fn,
                    S skip_fn) {
  auto [start, end] = range;
  if (start >= end) {
    return;
  }
  auto it = block->begin();
  std::advance(it, start);
  for (size_t i = start; i < end && it != block->end(); ++i, ++it) {
    if (skip_fn(it)) {
      continue;
    }
    fn(it);
  }
}

template <typename F>
void range_block_do(const Block *block, std::pair<size_t, size_t> range, F fn) {
  range_block_do(block, range, fn, [](Operation *op) { return false; });
}

std::map<int, int> GetOpInplaceInfo(const pir::Operation *op) {
  std::map<int, int> inplace_info;
  if (!op->HasTrait<paddle::dialect::InplaceTrait>()) {
    return inplace_info;
  }
  pir::IrContext *ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<StrAttribute>().AsString();
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));

  for (size_t i = 0; i < op->num_results(); ++i) {
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string &inplace_name = yaml_parser.InplaceName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(inplace_name);
    }
    if (yaml_parser.HasView(value_name)) {
      const std::string &view_name = yaml_parser.ViewName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(view_name);
    }
  }

  return inplace_info;
}

std::pair<std::vector<pir::Value>, std::unordered_set<pir::Value>>
AnalysisMiddleVariable(const Program &program,
                       const std::vector<pir::Value> &forward_inputs,
                       const std::vector<pir::Value> &backward_outputs,
                       const std::pair<size_t, size_t> &forward_range,
                       const std::pair<size_t, size_t> &backward_range) {
  std::vector<pir::Value> middle_values;

  std::unordered_set<pir::Value> backward_used_values;
  std::unordered_set<pir::Value> x_or_param(forward_inputs.begin(),
                                            forward_inputs.end());
  for (const auto &value : backward_outputs) {
    backward_used_values.insert(value);
  }
  range_block_do(
      program.block(), backward_range, [&backward_used_values](Operation *op) {
        pir::Walk(op, [&](Operation *inner_op) {
          for (auto &t : inner_op->operands()) {
            backward_used_values.insert(t.source());
          }
        });
      });

  range_block_do(
      program.block(),
      forward_range,
      [&middle_values, &backward_used_values, &x_or_param](Operation *op) {
        pir::Walk(op, [&](Operation *inner_op) {
          for (auto &t : inner_op->results()) {
            auto v = Value(t.Value::impl());
            if (backward_used_values.count(v) && !x_or_param.count(v)) {
              middle_values.push_back(v);
            }
          }
        });
      });
  return std::make_pair(middle_values, backward_used_values);
}

void mapping_value(const std::vector<pir::Value> &origin,
                   const std::unordered_map<pir::Value, pir::Value> &value_map,
                   std::vector<pir::Value> &out) {  // NOLINT
  std::transform(origin.begin(),
                 origin.end(),
                 std::back_inserter(out),
                 [&value_map](const pir::Value &v) {
                   if (v.impl() == nullptr) return Value(nullptr);
                   if (!value_map.count(v)) {
                     VLOG(2) << "mapping value found v is not exist. may not "
                                "used by backward program.";
                     return Value(nullptr);
                   }
                   return value_map.at(v);
                 });
}

using SplitedProgram = std::vector<std::shared_ptr<Program>>;
using SplitedAttribute = std::map<std::string, std::vector<pir::Value>>;
using SplitedResult = std::pair<SplitedProgram, SplitedAttribute>;

static auto GetNoNeedBufferValue(const ::pir::Block *whole_block,
                                 std::pair<size_t, size_t> range) {
  // filter no need buffer values.
  std::unordered_set<::pir::Value> need_buffer_values;
  std::unordered_set<::pir::Value> no_need_buffer_values;
  range_block_do(
      whole_block, range, [&need_buffer_values](::pir::Operation *op) {
        // NOTE(SigureMo): We should process the CombineOp in it's users.
        if (op->isa<pir::CombineOp>()) {
          return;
        }
        if (op->HasInterface<paddle::dialect::OpYamlInfoInterface>() == false) {
          // not a OpYamlInfoInterface, can't have no_need_buffer.
          for (const auto &operand : op->operands_source()) {
            need_buffer_values.insert(operand);
          }
        } else {
          auto opinfo =
              op->dyn_cast<paddle::dialect::OpYamlInfoInterface>().GetOpInfo();
          int counter = 0;
          for (const auto &op_input_info : std::get<0>(opinfo)) {
            auto value = op->operand_source(counter);
            if (!op_input_info.no_need_buffer) {
              need_buffer_values.insert(value);
              if (!IsFakeValue(value) && value.defining_op() &&
                  value.defining_op()->isa<pir::CombineOp>()) {
                for (const auto &combine_value :
                     value.defining_op()->operands_source()) {
                  need_buffer_values.insert(combine_value);
                }
              }
            }
            counter += 1;
          }
        }
      });
  range_block_do(whole_block,
                 range,
                 [&need_buffer_values,
                  &no_need_buffer_values](const ::pir::Operation *op) {
                   for (const auto &operand : op->operands_source()) {
                     if (need_buffer_values.count(operand) == 0) {
                       no_need_buffer_values.insert(operand);
                     }
                   }
                 });
  return std::vector<::pir::Value>(no_need_buffer_values.begin(),
                                   no_need_buffer_values.end());
}

using ValueMap = std::pair<std::vector<pir::Value>, std::vector<pir::Value>>;
std::pair<std::shared_ptr<Program>, ValueMap> CloneProgram(
    const Program &program) {
  // Limitation of this function:
  // 1. don't support Parameters.
  pir::IrMapping mapper;
  auto cloned_program = program.Clone(mapper);
  std::vector<pir::Value> associated_array_key, associated_array_value;
  for (auto &pair : mapper.GetMap<pir::Value>()) {
    associated_array_key.push_back(pair.first);
    associated_array_value.push_back(pair.second);
  }
  return std::make_pair(
      cloned_program,
      std::make_pair(associated_array_key, associated_array_value));
}

void AppendShadowOutput(Program *program,
                        const pir::Value &value,
                        const std::string &name,
                        size_t start_point) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
  pir::AttributeMap attribute_map = {
      {"output_name", StrAttribute::get(ctx, name)},
  };
  pir::Operation *operation =
      pir::Operation::Create({value}, attribute_map, {}, op_info);
  auto position = program->block()->begin();
  std::advance(position, start_point);
  if (position == program->block()->end()) {
    program->block()->push_back(operation);
  } else {
    program->block()->insert(position, operation);
  }
}

int AppendShadowOutputs(Program *program,
                        const std::vector<pir::Value> &outputs,
                        int start_point,
                        std::string name_prefix) {
  int counter = 0;
  std::unordered_set<pir::Value> added_value;
  for (const auto &value : outputs) {
    if (!added_value.count(value) || IsFakeValue(value)) {
      std::string shadow_output_name =
          name_analysis::TryGetValueFirstName(value).value_or(
              name_prefix + std::to_string(counter));
      AppendShadowOutput(
          program, value, shadow_output_name, start_point + counter);
      counter += 1;
      added_value.insert(value);
    }
  }
  // return the inserted op.
  return counter;
}

SplitedResult SplitForwardBackward(
    const Program &program,
    const std::vector<pir::Value> &forward_inputs,
    const std::vector<pir::Value> &forward_params,
    const std::vector<pir::Value> &forward_outputs,
    const std::vector<pir::Value> &forward_inputs_grads,
    const std::vector<pir::Value> &forward_params_grads,
    const std::vector<pir::Value> &forward_outputs_grads,
    const std::pair<size_t, size_t> &forward_range,
    const std::pair<size_t, size_t> &backward_range) {
  std::vector<pir::Value> forward_in_out_values;
  for (auto &v :
       std::vector({&forward_inputs, &forward_outputs, &forward_params})) {
    forward_in_out_values.insert(
        forward_in_out_values.end(), v->begin(), v->end());
  }
  std::vector<pir::Value> backward_out_values;
  for (auto &v : std::vector({&forward_inputs_grads, &forward_params_grads})) {
    backward_out_values.insert(backward_out_values.end(), v->begin(), v->end());
  }

  std::vector<pir::Value> fx, fp, fm, fo, bx, bp, bm, bo_g, bx_g, bp_g, bo;
  std::vector<pir::Value> no_need_buffer_values;
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto forward_program = std::make_shared<Program>(ctx);
  auto backward_program = std::make_shared<Program>(ctx);
  std::vector<pir::Value> middle_values;
  std::unordered_set<pir::Value> backward_used_values;
  std::tie(middle_values, backward_used_values) =
      AnalysisMiddleVariable(program,
                             forward_in_out_values,
                             backward_out_values,
                             forward_range,
                             backward_range);

  pir::Block &backward_block = *backward_program->block();
  bool has_backward = forward_inputs_grads.size() > 0 ||
                      forward_params_grads.size() > 0 ||
                      forward_outputs_grads.size() > 0;

  // forward program construct.
  VLOG(4) << "start create forward program.";
  pir::IrMapping forward_mapper;
  auto clone_options = pir::CloneOptions::All();
  range_block_do(
      program.block(),
      forward_range,
      [&forward_mapper, &forward_program, &clone_options](Operation *op) {
        auto *cloned_op = op->Clone(forward_mapper, clone_options);
        forward_program->block()->push_back(cloned_op);
      },
      // Skip the ShadowOutputOp.
      /*skip_fn=*/[](Operation *op) { return op->isa<pir::ShadowOutputOp>(); });
  auto &forward_value_map = forward_mapper.GetMutableMap<pir::Value>();

  // backward program construct.
  // Step1. insert data op for inputs_values and middle_values
  pir::IrMapping backward_mapper;
  auto &backward_value_map = backward_mapper.GetMutableMap<pir::Value>();

  auto create_output_fn = [&ctx](
                              const std::unordered_map<Value, Value> &value_map,
                              const std::shared_ptr<Program> &program,
                              const std::string &prefix) {
    auto counter = std::make_shared<size_t>(0);
    return [&ctx, &value_map, &program, &prefix, counter](const pir::Value &v) {
      // NOTE(SigureMo): Ensure counter++ executed in each iteration.
      auto default_name = prefix + std::to_string((*counter)++);
      if (v.impl() == nullptr) {
        return;
      }
      const pir::Value &new_value = value_map.at(v);
      std::string shadow_output_name =
          name_analysis::TryGetValueFirstName(new_value).value_or(default_name);
      auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
      pir::AttributeMap attribute_map = {
          {"output_name", StrAttribute::get(ctx, shadow_output_name)},
      };
      pir::Operation *operation =
          pir::Operation::Create({new_value}, attribute_map, {}, op_info);
      program->block()->push_back(operation);
    };
  };

  VLOG(4) << "start create forward outputs, inserting shadow_output ops.";
  std::for_each(
      middle_values.begin(),
      middle_values.end(),
      create_output_fn(forward_value_map, forward_program, "middle_"));
  std::for_each(
      forward_outputs.begin(),
      forward_outputs.end(),
      create_output_fn(forward_value_map, forward_program, "output_"));

  auto create_kwarg_fn = [&backward_block,
                          &backward_used_values,
                          &backward_value_map,
                          &forward_value_map](const std::string &prefix) {
    auto counter = std::make_shared<size_t>(0);
    return [&backward_block,
            &backward_used_values,
            &backward_value_map,
            &forward_value_map,
            &prefix,
            counter](const pir::Value &v) {
      // NOTE(SigureMo): Ensure counter++ executed in each iteration.
      auto default_name = prefix + std::to_string((*counter)++);
      if (v && !backward_value_map.count(v) &&
          (backward_used_values.count(v))) {
        backward_value_map[v] = backward_block.AddKwarg(
            name_analysis::TryGetValueFirstName(forward_value_map[v])
                .value_or(default_name),
            v.type());
      }
    };
  };

  if (has_backward) {
    VLOG(4) << "start create backward inputs, creating keyword argument.";
    VLOG(4) << "Create keyword argument for backward program: fo";
    std::for_each(forward_outputs.begin(),
                  forward_outputs.end(),
                  create_kwarg_fn("output_"));
    VLOG(4) << "Create keyword argument for backward program: fx";
    std::for_each(forward_inputs.begin(),
                  forward_inputs.end(),
                  create_kwarg_fn("input_"));
    VLOG(4) << "Create keyword argument for backward program: fp";
    std::for_each(forward_params.begin(),
                  forward_params.end(),
                  create_kwarg_fn("param_"));
    VLOG(4) << "Create keyword argument for backward program: fm";
    std::for_each(
        middle_values.begin(), middle_values.end(), create_kwarg_fn("middle_"));
    VLOG(4) << "Create keyword argument for backward program: fo_g";
    std::for_each(forward_outputs_grads.begin(),
                  forward_outputs_grads.end(),
                  create_kwarg_fn("output_grad_"));
    VLOG(4) << "Create keyword argument for backward program end.";
  }

  // Step2. copy backward ops .
  VLOG(4) << "start copy backward ops";
  range_block_do(
      program.block(),
      backward_range,
      [&backward_mapper, &backward_program, &clone_options](Operation *op) {
        auto *cloned_op = op->Clone(backward_mapper, clone_options);
        backward_program->block()->push_back(cloned_op);
      },
      // Skip the ShadowOutputOp.
      /*skip_fn=*/[](Operation *op) { return op->isa<pir::ShadowOutputOp>(); });
  VLOG(4) << "start create backward outputs, inserting shadow_output ops.";
  if (has_backward) {
    std::for_each(
        forward_inputs_grads.begin(),
        forward_inputs_grads.end(),
        create_output_fn(backward_value_map, backward_program, "input_grad_"));
    std::for_each(
        forward_params_grads.begin(),
        forward_params_grads.end(),
        create_output_fn(backward_value_map, backward_program, "param_grad_"));
  }

  VLOG(4) << "forward_value_map.size() is " << forward_value_map.size();
  VLOG(4) << "backward_value_map.size() is " << backward_value_map.size();
  if (FLAGS_print_ir) {
    std::ostringstream print_stream;
    print_stream << "ForwardProgram is :\n";
    forward_program->Print(print_stream);
    print_stream << "BackwardProgram is:\n";
    backward_program->Print(print_stream);
    std::cout << "Splited Program (fwd | bwd): \n"
              << print_stream.str() << std::endl;
  }

  // construct all attributes we needed.

  mapping_value(middle_values, forward_value_map, fm);    // write 'fm'
  mapping_value(middle_values, backward_value_map, bm);   // write 'bm'
  mapping_value(forward_inputs, forward_value_map, fx);   // write 'fx'
  mapping_value(forward_inputs, backward_value_map, bx);  // write 'bx'
  mapping_value(forward_params, forward_value_map, fp);   // write 'fp'
  mapping_value(forward_params, backward_value_map, bp);  // write 'bp'
  mapping_value(forward_outputs, forward_value_map, fo);  // write 'fo'
  mapping_value(
      forward_inputs_grads, backward_value_map, bx_g);  // write 'bx_g'
  mapping_value(
      forward_params_grads, backward_value_map, bp_g);  // write 'bp_g'
  mapping_value(
      forward_outputs_grads, backward_value_map, bo_g);    // write 'bo_g'
  mapping_value(forward_outputs, backward_value_map, bo);  // write 'bo'
  mapping_value(GetNoNeedBufferValue(program.block(), backward_range),
                forward_value_map,
                no_need_buffer_values);  // write 'no_need_buffers'

  std::map<std::string, std::vector<pir::Value>> attr = {
      {"fx", fx},
      {"fp", fp},
      {"fm", fm},
      {"fo", fo},
      {"bx", bx},
      {"bp", bp},
      {"bm", bm},
      {"bo_g", bo_g},
      {"bx_g", bx_g},
      {"bp_g", bp_g},
      {"no_need_buffers", no_need_buffer_values},
      {"bo", bo}};
  std::vector<std::shared_ptr<Program>> programs = {forward_program,
                                                    backward_program};
  return std::make_pair(programs, attr);
}

pir::Type CreateSelectedRowsTypeByDenseTensor(pir::Type dense_tensor_type) {
  if (dense_tensor_type.isa<DenseTensorType>()) {
    DenseTensorType type = dense_tensor_type.dyn_cast<DenseTensorType>();
    return SelectedRowsType::get(pir::IrContext::Instance(),
                                 type.dtype(),
                                 type.dims(),
                                 type.data_layout(),
                                 type.lod(),
                                 type.offset());
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, input is not a dense tensor type."));
  }
}

pir::Type CreateDistDenseTensorTypeByDenseTensor(
    const pir::Type &gdense_tensor_type,
    const std::vector<int> &lshape,
    const phi::distributed::ProcessMesh &mesh,
    const std::vector<int64_t> &dims_mapping) {
  if (gdense_tensor_type.isa<DenseTensorType>()) {
    DenseTensorType type = gdense_tensor_type.dyn_cast<DenseTensorType>();
    paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;
    paddle::dialect::TensorDistAttribute tensor_dist_attr =
        paddle::dialect::TensorDistAttribute::get(
            pir::IrContext::Instance(), mesh, dims_mapping, partial_status);
    return DistDenseTensorType::get(pir::IrContext::Instance(),
                                    type,
                                    tensor_dist_attr,
                                    phi::make_ddim(lshape));
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, input is not a dense tensor type are not supported."));
  }
}

static void inline CreateVariableIfNotExist(
    const std::vector<pir::Value> &var_list,
    framework::Scope *scope,
    const framework::Executor *exe = nullptr) {
  size_t len = var_list.size();

  for (size_t i = 0; i < len; ++i) {
    pir::Value value = var_list[i];
    std::string para_name = name_analysis::GetValueFirstName(value);
    auto var = scope->FindVar(para_name);
    if (var == nullptr) {
      PADDLE_ENFORCE_NOT_NULL(exe,
                              common::errors::InvalidArgument(
                                  "Parameter not Initialized, "
                                  "Please set argument [executor] not None "
                                  "or run startup program first"));
      var = scope->Var(para_name);
      auto *tensor_temp = var->GetMutable<phi::DenseTensor>();
      tensor_temp->Resize(
          common::make_ddim(phi::vectorize(GetValueDims(value))));
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      const phi::DeviceContext *dev_ctx = nullptr;
      dev_ctx = pool.Get(exe->GetPlace());
      dev_ctx->Alloc(tensor_temp, pir::GetValueDtype(value));
    }
  }
  return;
}

void BindUtils(pybind11::module *m) {
  m->def("create_loaded_parameter", CreateVariableIfNotExist);
  m->def("clone_program", CloneProgram);
  m->def("get_op_inplace_info", GetOpInplaceInfo);
  m->def("split_program", SplitForwardBackward);
  m->def("append_shadow_outputs", AppendShadowOutputs);
  m->def("append_shadow_output", AppendShadowOutput);
  m->def("fake_value", FakeValue);
  m->def("is_fake_value", IsFakeValue);
  m->def("get_current_insertion_point", []() -> PyInsertionPoint {
    return {ApiBuilder::Instance().GetCurrentInsertionPoint()};
  });
  m->def("set_insertion_point", [](const PyInsertionPoint &insertion_point) {
    ApiBuilder::Instance().SetInsertionPoint(insertion_point.value);
  });
  m->def("set_insertion_point",
         [](Operation *op) { ApiBuilder::Instance().SetInsertionPoint(op); });
  m->def("set_insertion_point_after", [](Operation *op) {
    ApiBuilder::Instance().SetInsertionPointAfter(op);
  });
  m->def("set_insertion_point_to_block_end", [](Block *block) {
    ApiBuilder::Instance().SetInsertionPointToBlockEnd(block);
  });
  m->def("reset_insertion_point_to_start",
         []() { ApiBuilder::Instance().ResetInsertionPointToStart(); });
  m->def("reset_insertion_point_to_end",
         []() { ApiBuilder::Instance().ResetInsertionPointToEnd(); });
  m->def("set_chunk_id",
         [](int chunk_id) { ApiBuilder::Instance().SetChunkId(chunk_id); });
  m->def("get_chunk_id", []() { return ApiBuilder::Instance().GetChunkId(); });
  m->def("set_op_role",
         [](int op_role) { ApiBuilder::Instance().SetOpRole(op_role); });
  m->def("get_op_role", []() { return ApiBuilder::Instance().GetOpRole(); });
  m->def("register_paddle_dialect", []() {
    pir::IrContext::Instance()
        ->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  });
  m->def("register_dist_dialect", []() {
    pir::IrContext::Instance()
        ->GetOrRegisterDialect<paddle::dialect::DistDialect>();
  });
  m->def("create_selected_rows_type_by_dense_tensor",
         CreateSelectedRowsTypeByDenseTensor);
  m->def("create_dist_dense_tensor_type_by_dense_tensor",
         CreateDistDenseTensorTypeByDenseTensor);
  m->def(
      "translate_to_pir",
      [](const ::paddle::framework::ProgramDesc &legacy_program) {
        std::shared_ptr<Program> ret =
            paddle::TranslateLegacyProgramToProgram(legacy_program);
        return ret;
      },
      R"DOC(
        Convert Fluid Program to New IR Program.

        Args:

            legacy_program (ProgramDesc): The Fluid Program that will be converted.

        Returns:
            Program: The New IR Program

        Raises:
            PreconditionNotMet: If legacy_program has multi block will raise error.

        Examples:
            .. code-block:: python

                >>> import os
                >>> # Paddle will remove this flag in the next version
                >>> pir_flag = 'FLAGS_enable_pir_in_executor'
                >>> os.environ[pir_flag] = 'True'

                >>> import paddle
                >>> from paddle import pir
                >>> paddle.enable_static()

                >>> x = paddle.randn([4, 4])
                >>> main_program, start_program = (
                ...    paddle.static.Program(),
                ...    paddle.static.Program(),
                ...)

                >>> with paddle.static.program_guard(main_program, start_program):
                ...    x_s = paddle.static.data('x', [4, 4], x.dtype)
                ...    x_s.stop_gradient = False
                ...    y_s = paddle.matmul(x_s, x_s)
                ...    z_s = paddle.add(y_s, y_s)
                ...    k_s = paddle.tanh(z_s)
                >>> pir_program = pir.translate_to_pir(main_program.desc)

                >>> print(pir_program)
                {
                 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,is_persistable:[false],name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,4],stop_gradient:[false]} : () -> builtin.tensor<4x4xf32>
                 (%1) = "pd_op.matmul" (%0, %0) {is_persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%2) = "pd_op.add" (%1, %1) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%3) = "pd_op.tanh" (%2) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                }


      )DOC");
  m->def(
      "check_unregistered_ops",
      [](const framework::ProgramDesc &legacy_program) {
        pir::IrContext *ctx = pir::IrContext::Instance();
        return paddle::translator::CheckUnregisteredOperation(ctx,
                                                              legacy_program);
      },
      R"DOC(
      Check unregistered operators in paddle dialect.

      Args:
        legacy_program (ProgramDesc): The Fluid Program that need checked.
      Returns:
        list[str] : List of unregistered operators in paddle dialect, the name is expressed by origin op name.
    )DOC");
  m->def(
      "translate_to_pir_with_param_map",
      [](const framework::ProgramDesc &legacy_program) {
        auto ir_ctx = pir::IrContext::Instance();
        auto program = std::make_shared<pir::Program>(ir_ctx);
        translator::ProgramTranslator program_translator(&legacy_program,
                                                         program.get());
        program_translator.Translate();
        return std::make_pair(program, program_translator.VarDesc2Value());
      },
      R"DOC(
        Convert Fluid Program to New IR Program and get the mappings of VarDesc -> pir::Value.

        Args:

            legacy_program (ProgramDesc): The Fluid Program that will be converted.

        Returns:
            Program: The New IR Program
            dict[str, pir::Value]: Mapping between VarDesc(by name) and pir::Value.

        Raises:
            PreconditionNotMet: If legacy_program has multi block will raise error.

        Examples:
            .. code-block:: python

                >>> import os
                >>> # Paddle will remove this flag in the next version
                >>> pir_flag = 'FLAGS_enable_pir_in_executor'
                >>> os.environ[pir_flag] = 'True'

                >>> import paddle
                >>> from paddle import pir
                >>> paddle.enable_static()

                >>> x = paddle.randn([4, 4])
                >>> main_program, start_program = (
                ...     paddle.static.Program(),
                ...     paddle.static.Program(),
                ... )

                >>> with paddle.static.program_guard(main_program, start_program):
                ...     x_s = paddle.static.data('x', [4, 4], x.dtype)
                ...     x_s.stop_gradient = False
                ...     y_s = paddle.matmul(x_s, x_s)
                ...     z_s = paddle.add(y_s, y_s)
                ...     k_s = paddle.tanh(z_s)
                >>> pir_program, mappings = pir.translate_to_pir_with_param_map(main_program.desc)

                >>> print(pir_program)
                {
                 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,is_persistable:[false],name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,4],stop_gradient:[false]} : () -> builtin.tensor<4x4xf32>
                 (%1) = "pd_op.matmul" (%0, %0) {is_persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%2) = "pd_op.add" (%1, %1) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%3) = "pd_op.tanh" (%2) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                }

                >>> print(mappings)
                {'matmul_v2_0.tmp_0': [Value(define_op_name=pd_op.matmul, index=0, dtype=builtin.tensor<4x4xf32>)], 'x': [Value(define_op_name=pd_op.data, index=0, dtype=builtin.tensor<4x4xf32>)], 'tanh_0.tmp_0': [Value(define_op_name=pd_op.tanh, index=0, dtype=builtin.tensor<4x4xf32>)], 'elementwise_add_0': [Value(define_op_name=pd_op.add, index=0, dtype=builtin.tensor<4x4xf32>)]}
    )DOC");
  m->def("clear_cinn_compilation_cache", []() {
#ifdef PADDLE_WITH_CINN
    pybind11::gil_scoped_release release;
    VLOG(4) << "clear CINN CompilationCache and free BackendResource.";
    cinn::hlir::framework::CompilationCache::Instance().Clear();
#endif
  });

  m->def("cinn_compilation_cache_size", []() {
#ifdef PADDLE_WITH_CINN
    pybind11::gil_scoped_release release;
    VLOG(4) << "clear CINN CompilationCache and free BackendResource.";
    return cinn::hlir::framework::CompilationCache::Instance().Size();
#endif
  });
}

namespace {

void ApplyCinnPass(Program &program) {  // NOLINT
#ifdef PADDLE_WITH_CINN
  auto CreatePassManager = [&]() -> std::shared_ptr<pir::PassManager> {
    pir::IrContext *ctx = pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
    ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
    auto pass_manager = std::make_shared<pir::PassManager>(ctx);
    if (FLAGS_print_ir) {
      pass_manager->EnableIRPrinting();
    }
    auto &shape_analysis = pir::ShapeAnalysisManager::Instance().Get(&program);
    pass_manager->SetValueReplacedHook([&](pir::Value from, pir::Value to) {
      shape_analysis.ShareShapeOrData(from, to);
    });
    return pass_manager;
  };
  cinn::dialect::ir::ApplyCinnPass(&program, CreatePassManager);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "Currently we only support CINN Pass for Pir under @to_static, please "
      "compile PaddlePaddle with CINN"));
#endif
}

void CheckInferSymbolicIfNeed(Program &program) {  // NOLINT
#ifdef PADDLE_WITH_CINN
  auto CreatePassManager = [&]() -> std::shared_ptr<pir::PassManager> {
    pir::IrContext *ctx = pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
    ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
    auto pass_manager = std::make_shared<pir::PassManager>(ctx);
    if (FLAGS_print_ir) {
      pass_manager->EnableIRPrinting();
    }
    return pass_manager;
  };
  cinn::dialect::ir::CheckInferSymbolicIfNeed(&program, CreatePassManager);
#else
  // Do nothing.
#endif
}

}  // namespace

void InferSymbolicShapePass(
    std::shared_ptr<pir::PassManager> &pass_manager,  // NOLINT
    pir::Program &program) {                          // NOLINT
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  if (FLAGS_pir_apply_shape_optimization_pass) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }
}

std::shared_ptr<Program> ApplyCommonSubexpressionEliminationPass(
    std::shared_ptr<Program> program) {
  pir::PassManager pm(pir::IrContext::Instance(), 2);
  pm.AddPass(pir::CreateCommonSubexpressionEliminationPass());
  pm.Run(program.get());
  if (FLAGS_print_ir) {
    std::cout
        << "IR After CommonSubexpressionEliminationPass -------------------"
        << std::endl;
    std::cout << *program << std::endl;
  }
  return program;
}

void ApplyReduceAsToSumPass(
    std::shared_ptr<pir::PassManager> &pass_manager,  // NOLINT
    pir::Program &program) {                          // NOLINT
#ifdef PADDLE_WITH_CINN
  pass_manager->AddPass(cinn::dialect::ir::CreateReduceAsToSumPass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "Currently we only support ReduceAsToSumPass Pass for Pir under "
      "@to_static, please "
      "compile PaddlePaddle with CINN"));
#endif
}

std::shared_ptr<Program> ApplyFusedBnAddActPass(
    std::shared_ptr<Program> program) {
  pir::PassManager pm(pir::IrContext::Instance(), 3);
  pm.AddPass(pir::CreateFusedBnAddActPass());
  pm.Run(program.get());
  if (FLAGS_print_ir) {
    std::cout << "IR After FusedBnAddActPass -------------------" << std::endl;
    std::cout << *program << std::endl;
  }
  return program;
}

void BindIrPass(pybind11::module *m) {
  m->def("apply_cinn_pass", ApplyCinnPass);
  m->def("check_infer_symbolic_if_need", CheckInferSymbolicIfNeed);
  m->def("infer_symbolic_shape_pass", InferSymbolicShapePass);
  m->def("apply_cse_pass", ApplyCommonSubexpressionEliminationPass);
  m->def("apply_bn_add_act_pass", ApplyFusedBnAddActPass);
  m->def("reduce_as_sum_pass", ApplyReduceAsToSumPass);

  py::class_<Pass, std::shared_ptr<Pass>> pass(*m,
                                               "Pass",
                                               R"DOC(
    Pass class.

  )DOC");
  pass.def("name", &Pass::name)
      .def("opt_level",
           [](const Pass &self) { return self.pass_info().opt_level; })
      .def("dependents",
           [](const Pass &self) { return self.pass_info().dependents; });
}

void BindPassManager(pybind11::module *m) {
  py::class_<PassManager, std::shared_ptr<PassManager>> pass_manager(
      *m,
      "PassManager",
      R"DOC(
    A class that manages all passes.

  )DOC");
  pass_manager
      .def(py::init([](uint8_t opt_level) {
             return std::make_unique<PassManager>(pir::IrContext::Instance(),
                                                  opt_level);
           }),
           py::arg("opt_level") = 2)
      .def("add_pass",
           [](PassManager &self,
              const std::string &pass_name,
              const std::unordered_map<std::string, py::object> attrs = {}) {
             auto pass = pir::PassRegistry::Instance().Get(pass_name);
             for (const auto &attr : attrs) {
               if (py::isinstance<py::str>(attr.second)) {
                 pass->Set(attr.first,
                           new std::string(attr.second.cast<std::string>()));
               } else if (py::isinstance<py::bool_>(attr.second)) {
                 pass->Set(attr.first, new bool(attr.second.cast<bool>()));
               } else if (py::isinstance<py::int_>(attr.second)) {
                 pass->Set(attr.first, new int(attr.second.cast<int>()));
               } else if (py::isinstance<py::float_>(attr.second)) {
                 pass->Set(attr.first, new float(attr.second.cast<float>()));
               } else if (py::isinstance<framework::Scope>(attr.second)) {
                 pass->SetNotOwned(attr.first,
                                   attr.second.cast<framework::Scope *>());
               } else if (py::isinstance<phi::GPUPlace>(attr.second)) {
                 pass->Set(attr.first,
                           new phi::Place(attr.second.cast<phi::GPUPlace>()));
               } else {
                 PADDLE_THROW(common::errors::InvalidArgument(
                     "The pass attr is not supported this type."));
               }
             }
             self.AddPass(std::move(pass));
           })
      .def("register_pass",
           [](PassManager &self,
              const std::string &pass_name,
              std::shared_ptr<paddle::drr::DrrPatternContext> pattern_ctx) {
             using AutoFinalPass =
                 paddle::drr::AutoDrrPass<paddle::drr::AutoDrrPattern>;
             // Instead of using static PassRegistrar which may cause lifetime
             // issues during program termination, directly register the pass to
             // PassRegistry. This approach provides better control over object
             // lifetime management and avoids potential segmentation faults
             // during static destruction.
             self.AddPass(
                 std::make_unique<AutoFinalPass>(pass_name, pattern_ctx));
           })
      .def("passes",
           [](PassManager &self) {
             std::vector<std::string> pass_names;
             for (const auto &pass : self.passes()) {
               pass_names.emplace_back(pass->name());
             }
             return pass_names;
           })
      .def("run", [](PassManager &self, Program *p) { self.Run(p); })
      .def("empty", &PassManager::empty)
      .def("clear", &PassManager::clear)
      .def("enable_ir_printing",
           [](PassManager &self) { self.EnableIRPrinting(); })
      .def("enable_print_statistics",
           [](PassManager &self) { self.EnablePrintStatistics(); });
}

void BindDrrPatternContext(pybind11::module *m) {
  // bind NormalAttribute
  pybind11::class_<drr::NormalAttribute> normal_attribute(*m,
                                                          "NormalAttribute");

  // bind ComputeAttribute
  pybind11::class_<drr::ComputeAttribute> compute_attribute(*m,
                                                            "ComputeAttribute");

  // bind Tensor
  pybind11::class_<drr::Tensor> tensor(*m,
                                       "Tensor",
                                       R"DOC(
        register Tensor for DRR.
    )DOC");

  // bind Op
  pybind11::class_<drr::Op> op(*m,
                               "Op",
                               R"DOC(
        Represents an operation in the DRR framework.
    )DOC");
  op.def(
      "__call__",
      [](drr::Op *self,
         const std::vector<drr::Tensor> &input_tensors,
         const std::vector<drr::Tensor> &output_tensors) {
        std::vector<const drr::Tensor *> input_ptrs;
        std::vector<const drr::Tensor *> output_ptrs;

        for (const auto &t : input_tensors) {
          input_ptrs.push_back(&t);
        }
        for (const auto &t : output_tensors) {
          output_ptrs.push_back(&t);
        }
        (*self)(input_ptrs, output_ptrs);
      },
      pybind11::arg("input_tensors"),
      pybind11::arg("output_tensors"),
      "Call the operation with an input tensor and return the output tensor.");

  // bind DrrPatternContext
  pybind11::class_<drr::DrrPatternContext,
                   std::shared_ptr<drr::DrrPatternContext>>
      drr_pattern_context(*m,
                          "DrrPatternContext",
                          R"DOC(
    A class that manages DRR (Dynamic Rewrite Rule) pattern context.

  )DOC");
  drr_pattern_context.def(pybind11::init<>())
      .def("SourcePattern", &drr::DrrPatternContext::SourcePattern);

  // bind drr::SourcePattern
  pybind11::class_<drr::SourcePattern, std::shared_ptr<drr::SourcePattern>>
      source_pattern(*m,
                     "SourcePattern",
                     R"DOC(
      Represents a source pattern for matching in the DRR framework.

  )DOC");
  source_pattern.def("ResultPattern", &drr::SourcePattern::ResultPattern)
      .def(
          "Op",
          [](drr::SourcePattern &self,
             const std::string &op_type,
             const std::unordered_map<std::string, drr::Attribute> &attributes =
                 {}) { return self.Op(op_type, attributes); },
          pybind11::return_value_policy::reference_internal,
          pybind11::arg("op_type"),
          pybind11::arg("attributes") =
              std::unordered_map<std::string, drr::Attribute>())
      .def(
          "Tensor",
          [](drr::SourcePattern &self, const std::string &name) {
            return self.Tensor(name);
          },
          pybind11::return_value_policy::reference_internal,
          pybind11::arg("name"))
      .def(
          "InputNoneTensor",
          [](drr::ResultPattern &self) { return self.InputNoneTensor(); },
          pybind11::return_value_policy::reference_internal)
      .def(
          "OutputNoneTensor",
          [](drr::ResultPattern &self) { return self.OutputNoneTensor(); },
          pybind11::return_value_policy::reference_internal)
      .def(
          "Attr",
          [](drr::SourcePattern &self, const std::string &attr_name) {
            return self.Attr(attr_name);
          },
          pybind11::return_value_policy::reference_internal,
          pybind11::arg("attr_name"))
      .def("AddConstraint",
           [](drr::SourcePattern &self, const pybind11::function &py_func) {
             // wrap pyfunction -> cpp function
             paddle::drr::ConstraintFunction cpp_func =
                 [py_func](const paddle::drr::MatchContext &context) -> bool {
               try {
                 pybind11::object py_context = pybind11::cast(context);
                 pybind11::object result = py_func(py_context);

                 bool ret = result.cast<bool>();
                 return ret;
               } catch (const pybind11::error_already_set &e) {
                 std::cerr << "Python error in AddConstraint callback: "
                           << e.what() << std::endl;
                 throw;
               }
             };
             self.AddConstraint(cpp_func);
           })
      .def("AddPostProcess",
           [](drr::SourcePattern &self, const pybind11::function &py_func) {
             // wrap pyfunction -> cpp function
             paddle::drr::PostProcessFunction cpp_func =
                 [py_func](const paddle::drr::MatchContext &context) -> void {
               try {
                 pybind11::object py_context = pybind11::cast(context);
                 py_func(py_context);
               } catch (const pybind11::error_already_set &e) {
                 std::cerr << "Python error in AddPostProcess callback: "
                           << e.what() << std::endl;
                 throw;
               }
             };
             self.AddPostProcess(cpp_func);
           });

  // bind MatchContext
  pybind11::class_<drr::MatchContext, std::shared_ptr<drr::MatchContext>>
      match_context(*m,
                    "MatchContext",
                    R"DOC(
        Represents the context of a match in the DRR framework.
    )DOC");
  match_context
      .def(
          "Tensor",
          [](drr::MatchContext &self, std::string &tensor_name) -> pir::Value {
            return self.Tensor(tensor_name);
          },
          pybind11::return_value_policy::reference_internal,
          pybind11::arg("tensor_name"))
      // Attr
      .def(
          "StrAttr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<std::string>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "BoolAttr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<bool>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "Int32Attr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<int32_t>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "Int64Attr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<int64_t>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "Float32Attr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<float>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "DoubleAttr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<double>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "VectorInt32Attr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<std::vector<int32_t>>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "VectorInt64Attr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<std::vector<int64_t>>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "VectorFloat32Attr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<std::vector<int32_t>>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "DataTypeAttr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<phi::DataType>(value_name);
          },
          pybind11::arg("value_name"))
      .def(
          "PlaceAttr",
          [](drr::MatchContext &self, const std::string &value_name) {
            return self.Attr<phi::Place>(value_name);
          },
          pybind11::arg("value_name"));

  // bind drr::ResultPattern
  pybind11::class_<drr::ResultPattern, std::shared_ptr<drr::ResultPattern>>
      result_pattern(*m,
                     "ResultPattern",
                     R"DOC(
      Represents a result pattern for matching in the DRR framework

  )DOC");

  result_pattern
      .def(
          "Op",
          [](drr::ResultPattern &self,
             const std::string &op_type,
             const std::unordered_map<std::string, drr::Attribute> &attributes =
                 {}) { return self.Op(op_type, attributes); },
          pybind11::return_value_policy::reference_internal,
          pybind11::arg("op_type"),
          pybind11::arg("attributes") =
              std::unordered_map<std::string, drr::Attribute>())
      .def(
          "InputNoneTensor",
          [](drr::ResultPattern &self) { return self.InputNoneTensor(); },
          pybind11::return_value_policy::reference_internal)
      .def(
          "OutputNoneTensor",
          [](drr::ResultPattern &self) { return self.OutputNoneTensor(); },
          pybind11::return_value_policy::reference_internal)
      .def(
          "Tensor",
          [](drr::ResultPattern &self, const std::string &name) {
            return self.Tensor(name);
          },
          pybind11::return_value_policy::reference_internal,
          pybind11::arg("name"))
      // Attr
      .def(
          "StrAttr",
          [](drr::ResultPattern &self, const std::string &value) {
            return self.StrAttr(value);
          },
          pybind11::arg("value"))
      .def(
          "BoolAttr",
          [](drr::ResultPattern &self, bool value) {
            return self.BoolAttr(value);
          },
          pybind11::arg("value"))
      .def(
          "Int32Attr",
          [](drr::ResultPattern &self, int32_t value) {
            return self.Int32Attr(value);
          },
          pybind11::arg("value"))
      .def(
          "Int64Attr",
          [](drr::ResultPattern &self, int64_t value) {
            return self.Int64Attr(value);
          },
          pybind11::arg("value"))
      .def(
          "Float32Attr",
          [](drr::ResultPattern &self, float value) {
            return self.Float32Attr(value);
          },
          pybind11::arg("value"))
      .def(
          "VectorInt32Attr",
          [](drr::ResultPattern &self, const std::vector<int32_t> &value) {
            return self.VectorInt32Attr(value);
          },
          pybind11::arg("value"))
      .def(
          "VectorInt64Attr",
          [](drr::ResultPattern &self, const std::vector<int64_t> &value) {
            return self.VectorInt64Attr(value);
          },
          pybind11::arg("value"))
      .def(
          "VectorFloat32Attr",
          [](drr::ResultPattern &self, const std::vector<float> &value) {
            return self.VectorFloatAttr(value);
          },
          pybind11::arg("value"))
      .def(
          "DataTypeAttr",
          [](drr::ResultPattern &self, const std::string &value) {
            return self.DataTypeAttr(value);
          },
          pybind11::arg("value"))
      .def(
          "PlaceAttr",
          [](drr::ResultPattern &self, const std::string &value) {
            return self.PlaceAttr(value);
          },
          pybind11::arg("value"))
      .def(
          "DataLayoutAttr",
          [](drr::ResultPattern &self, const std::string &value) {
            return self.DataLayoutAttr(value);
          },
          pybind11::arg("value"))
      .def(
          "ComputeAttr",
          [](drr::ResultPattern &self, pybind11::function py_func) {
            paddle::drr::AttrComputeFunc cpp_func =
                [py_func](
                    const paddle::drr::MatchContext &context) -> std::any {
              try {
                pybind11::object py_context = pybind11::cast(context);
                pybind11::object py_result = py_func(py_context);
                pybind11::tuple result_tuple =
                    py_result.cast<pybind11::tuple>();
                pybind11::object result = result_tuple[0];
                std::string type_name = result_tuple[1].cast<std::string>();
                auto any_result = CastPyObjectToAny(result, type_name);

                return std::visit(
                    [](auto &&value) -> std::any { return std::any(value); },
                    any_result);
              } catch (const pybind11::error_already_set &e) {
                std::cerr << "Python error in ComputeAttr callback: "
                          << e.what() << std::endl;
                throw;
              }
            };
            return self.ComputeAttr(cpp_func);
          },
          pybind11::arg("py_func"));

  m->def("value_is_persistable", [](const pir::Value &value) {
    return pir::ValueIsPersistable(value);
  });
}

void BindShapeOrDataDimExprs(pybind11::module *m) {
  py::class_<symbol::ShapeOrDataDimExprs,
             std::shared_ptr<symbol::ShapeOrDataDimExprs>>
      shape_or_data_dim_exprs(*m, "ShapeOrDataDimExprs", R"DOC(
      A class that store the shape or data of value.
    )DOC");
  shape_or_data_dim_exprs
      .def("shape",
           &symbol::ShapeOrDataDimExprs::shape,
           return_value_policy::reference)
      .def("data",
           &symbol::ShapeOrDataDimExprs::data,
           return_value_policy::reference)
      .def(
          "is_equal",
          [](symbol::ShapeOrDataDimExprs &self,
             std::vector<int64_t> expect_shape,
             std::vector<int64_t> expect_data = {}) -> bool {
            VLOG(3) << "Start compare shape and data.";

            const auto &CompareFunc =
                [&](const std::vector<int64_t> &expect,
                    const std::vector<symbol::DimExpr> &actual,
                    const std::string &compare_type) -> bool {
              const auto PrintExpectAndActual = [&](const std::string &prefix) {
                std::ostringstream sout;
                sout << prefix << " expect: [";
                std::copy(expect.begin(),
                          expect.end(),
                          std::ostream_iterator<int64_t>(sout, ","));
                sout << "]" << std::endl;

                sout << prefix << " actual:" << actual << std::endl;
                LOG(ERROR) << sout.str();
              };

              if (actual.size() != expect.size()) {
                LOG(ERROR) << compare_type << " expect size " << expect.size()
                           << " is not equal to actual size " << actual.size()
                           << " . The detailed infermation is as follows:";
                PrintExpectAndActual(compare_type);
                return false;
              } else if (actual.empty()) {
                return true;
              }

              for (size_t i = 0; i < actual.size(); i++) {
                if (!actual.at(i).isa<int64_t>()) {
                  PrintExpectAndActual(compare_type);
                  PADDLE_THROW(common::errors::InvalidArgument(
                      "In OpTest, only supports cases where the type of "
                      "DimExpr "
                      "is int64_t."));
                  return false;
                }
                if (actual.at(i) != expect.at(i)) {
                  LOG(ERROR)
                      << compare_type << " expect[" << i
                      << "]: " << expect.at(i) << " is not equal to actual["
                      << i << "]: " << actual.at(i)
                      << " . The detailed infermation is as follows:";
                  PrintExpectAndActual(compare_type);
                  return false;
                }
              }
              return true;
            };

            // compare shape
            const std::vector<symbol::DimExpr> &actual_shape = self.shape();
            const bool shape_status =
                CompareFunc(expect_shape, actual_shape, "shape");
            // compare data
            const std::optional<std::vector<symbol::DimExpr>> &actual_data_ =
                self.data();
            if (actual_data_.has_value()) {
              PADDLE_ENFORCE_LE(actual_shape.size(),
                                1,
                                common::errors::Unimplemented(
                                    "Now data dim expr is not supported for "
                                    "multi-dim shape."));
              const std::vector<symbol::DimExpr> actual_data =
                  actual_data_.value();
              const bool data_status =
                  CompareFunc(expect_data, actual_data, "data");
              return shape_status && data_status;
            }
            return shape_status;
          });
}

void BindShapeConstraintIRAnalysis(pybind11::module *m) {
  m->def(
      "get_shape_constraint_ir_analysis",
      [](const pir::Program *program) -> pir::ShapeConstraintIRAnalysis & {
        return pir::ShapeAnalysisManager::Instance().Get(program);
      },
      return_value_policy::reference);
  m->def("all_ops_defined_symbol_infer",
         [](const pir::Program *program) -> bool {
           // check that all ops have defined the InferSymbolicShapeInterface
           bool flag = true;
           for (pir::Operation &op : *(program->block())) {
             pir::InferSymbolicShapeInterface infer_interface =
                 op.dyn_cast<pir::InferSymbolicShapeInterface>();
             if (!infer_interface) {
               LOG(ERROR) << "The op: " << op.name()
                          << " does not implement InferSymbolicShapeInterface.";
               flag = false;
             }
           }
           return flag;
         });

  py::class_<pir::ShapeConstraintIRAnalysis,
             std::shared_ptr<pir::ShapeConstraintIRAnalysis>>
      shape_constraint_ir_analysis(*m, "ShapeConstraintIRAnalysis", R"DOC(
      A class that store the shape information of all operators.
    )DOC");
  shape_constraint_ir_analysis
      .def("get_shape_or_data_for_var",
           &pir::ShapeConstraintIRAnalysis::GetShapeOrDataForValue,
           return_value_policy::reference)
      .def("set_shape_or_data_for_var",
           &pir::ShapeConstraintIRAnalysis::SetShapeOrDataForValue)
      .def("register_symbol_cstr_from_shape_analysis",
           &pir::ShapeConstraintIRAnalysis::
               RegisterSymbolConstraintFromShapeAnalysis);
}

void BindPir(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  BindProgram(&ir_module);
  BindBlock(&ir_module);
  BindValue(&ir_module);
  BindIrMapping(&ir_module);
  BindCloneOptions(&ir_module);
  BindOperation(&ir_module);
  BindOpOperand(&ir_module);
  BindType(&ir_module);
  BindVectorType(&ir_module);
  BindAttribute(&ir_module);
  BindInsertionPoint(&ir_module);
  BindUtils(&ir_module);
  BindIrPass(&ir_module);
  BindPassManager(&ir_module);
  BindControlFlowApi(&ir_module);
  BindShapeOrDataDimExprs(&ir_module);
  BindShapeConstraintIRAnalysis(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindOpsAPI(&ops_modules);
  BindDrrPatternContext(&ir_module);
}

}  // namespace pybind
}  // namespace paddle
