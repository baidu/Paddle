// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_set>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/fluid/pybind/eager_generator.h"
#include "paddle/fluid/pybind/eager_legacy_op_function_generator.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/utils/string/string_helper.h"

// phi
#include "paddle/phi/kernels/declarations.h"

static std::string LegalizeVarName(const std::string& var_name) {
  std::string ret = var_name;
  std::replace(ret.begin(), ret.end(), '@', '_');  // replace all '-' to '_'
  return ret;
}
// clang-format off
const char* OUT_INITIALIZER_TEMPLATE =
    R"({"%s", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}})";
const char* OUT_DUPLICABLE_INITIALIZER_TEMPLATE = R"({"%s", ConstructDuplicableOutput(%s)})";

const char* INPUT_INITIALIZER_TEMPLATE = R"({"%s", {%s}})";
const char* INPUT_LIST_INITIALIZER_TEMPLATE = R"({"%s", %s})";

const char* INPUT_INITIALIZER_TEMPLATE_WITH_NULL = R"(
    if (%s != nullptr) {
      ins["%s"] = {%s};
    }
)";

const char* INPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST = R"(
    if (%s.size() != 0) {
      ins["%s"] = %s;
    }
)";

const char* OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL = R"(
    outs["%s"] = {%s};
)";

const char* OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST = R"(
    outs["%s"] = %s;
)";
// if inputs is list, no need {}
const char* ARG_OUT_NUM = R"(%sNum)";
const char* ARG_OUT_NUM_TYPE = R"(size_t )";

const char* IN_VAR_TYPE = R"(py::handle)";
const char* IN_VAR_LIST_TYPE = R"(py::handle)";

const char* OUT_VAR_TYPE = R"(std::shared_ptr<imperative::VarBase>)";
const char* OUT_VAR_LIST_TYPE = R"(std::vector<std::shared_ptr<imperative::VarBase>>)";

const char* CAST_VAR_TEMPLATE = R"(
    auto& %s = GetTensorFromArgs("%s", "%s", args, %d, %s);)";

const char* CAST_VAR_LIST_TEMPLATE = R"(
    auto %s = GetTensorListFromArgs("%s", "%s", args, %d, %s);)";

const char* CAST_VAR_PTR_TEMPLATE = R"(
    auto %s = GetTensorPtrFromArgs("%s", "%s", args, %d, %s);)";

const char* CAST_VAR_PTR_LIST_TEMPLATE = R"(
    auto %s = GetTensorPtrListFromArgs("%s", "%s", args, %d, %s);)";

const char* CAST_SIZE_T_TEMPLATE = R"(
    auto %s = GetUnsignedLongFromArgs("%s", "%s", args, %d, %s);)";

const char* ARG_TEMPLATE = R"(const %s& %s)";

const char* RETURN_TUPLE_TYPE = R"(std::tuple<%s>)";
const char* RETURN_TUPLE_TEMPLATE = R"(std::make_tuple(%s))";
const char* RETURN_LIST_TEMPLATE = R"(outs["%s"])";
const char* RETURN_TEMPLATE = R"(outs["%s"][0])";

const char* FUNCTION_ARGS = R"(%s, const py::args& args)";
const char* FUNCTION_ARGS_NO_INPUT = R"(const py::args& args)";

const char* HANDLE_VIEW_BETWEEN_INPUT_AND_OUTPUT = R"(
    if (ins.count("%s") && outs.count("%s")) {
      HandleViewBetweenInputAndOutput(ins["%s"][0], outs["%s"][0]);
    })";

const char* OP_FUNCTION_TEMPLATE =
R"(
static PyObject * %s(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try {
    print_mem_info_guard guard("%s");
    %s
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs("%s", args, %d, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    %s
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    %s
  } catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
})";

const char* PYBIND_ITEM_TEMPLATE = R"(  {"%s", (PyCFunction)(void(*)(void))%s, METH_VARARGS | METH_KEYWORDS, "C++ interface function for %s in dygraph."},)";

// These operators will skip automatic code generation and
// need to be handwritten in CUSTOM_HANDWRITE_OP_FUNC_FILE
std::unordered_set<std::string> CUSTOM_HANDWRITE_OPS_SET = {"run_program"};

// clang-format on
static inline bool FindInsMap(const std::string& op_type,
                              const std::string& in_name) {
  return op_ins_map[op_type].count(in_name);
}

static inline bool FindOutsMap(const std::string& op_type,
                               const std::string& out_name) {
  return op_outs_map[op_type].count(out_name);
}

static inline bool FindPassingOutsMap(const std::string& op_type,
                                      const std::string& out_name) {
  return op_passing_outs_map[op_type].count(out_name);
}

static inline bool FindViewOpMap(const std::string& op_type) {
  return view_op_map.count(op_type);
}

static inline std::string TempName(const std::string& name) {
  return name + '_';
}

std::string GenerateOpFunctionsBody(
    const paddle::framework::proto::OpProto* op_proto,
    std::string func_name,
    std::map<std::string, std::string> inplace_map = {}) {
  auto& op_type = op_proto->type();
  std::string input_args = "";
  std::string call_api_str = "";
  std::string ins_initializer_with_null = "";
  std::string py_arg = "";
  int arg_idx = 0;
  int input_args_num = 0;
  std::string ins_cast_str = "";
  std::string view_strategy_str = "";
  if (!inplace_map.empty()) {
    // change call_api_str for inplace op
    call_api_str = "auto out = " + op_type + "__dygraph_function(";
  } else {
    call_api_str = "auto out = " + op_type + "_dygraph_function(";
  }
  for (auto& input : op_proto->inputs()) {
    auto& in_name = input.name();
    // skip those dispensable inputs, like ResidualData in conv2d
    if (input.dispensable() && !FindInsMap(op_type, in_name)) {
      continue;
    }
    const auto in_type = input.duplicable() ? IN_VAR_LIST_TYPE : IN_VAR_TYPE;
    auto input_arg = paddle::string::Sprintf(
        ARG_TEMPLATE, in_type, TempName(LegalizeVarName(in_name)));
    input_args += input_arg;
    input_args += ",";
    input_args_num++;
    const auto in_cast_type =
        input.duplicable() ? CAST_VAR_LIST_TEMPLATE : CAST_VAR_TEMPLATE;
    auto dispensable = input.dispensable() ? "true" : "false";
    ins_cast_str += paddle::string::Sprintf(in_cast_type,
                                            LegalizeVarName(in_name),
                                            op_type,
                                            in_name,
                                            arg_idx++,
                                            dispensable);

    call_api_str += LegalizeVarName(in_name) + ", ";
  }

  if (!input_args.empty() && input_args.back() == ',') {
    input_args.pop_back();
  }

  // Generate outs initializer
  std::string outs_initializer = "{";
  std::string outs_initializer_with_null = "";
  std::string return_str = "";

  for (auto& output : op_proto->outputs()) {
    auto& out_name = output.name();

    // skip those dispensable outputs
    if (output.dispensable() && !FindOutsMap(op_type, out_name)) {
      continue;
    }
    const auto out_type =
        output.duplicable() ? OUT_VAR_LIST_TYPE : OUT_VAR_TYPE;

    if (FindPassingOutsMap(op_type, out_name)) {
      if (!input_args.empty()) {
        input_args += ",";
      }
      input_args += out_type;
      input_args += LegalizeVarName(out_name);
      input_args_num++;

      if (output.dispensable()) {
        const auto out_template =
            output.duplicable() ? OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST
                                : OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL;
        outs_initializer_with_null +=
            paddle::string::Sprintf(out_template, out_name, out_name);
      } else {
        const auto out_template = output.duplicable()
                                      ? INPUT_LIST_INITIALIZER_TEMPLATE
                                      : INPUT_INITIALIZER_TEMPLATE;
        outs_initializer += paddle::string::Sprintf(
            out_template, out_name, LegalizeVarName(out_name));
        outs_initializer += ",";
      }

      const auto in_cast_type = output.duplicable() ? CAST_VAR_PTR_LIST_TEMPLATE
                                                    : CAST_VAR_PTR_TEMPLATE;
      auto dispensable = output.dispensable() ? "true" : "false";
      ins_cast_str += paddle::string::Sprintf(in_cast_type,
                                              LegalizeVarName(out_name),
                                              op_type,
                                              out_name,
                                              arg_idx++,
                                              dispensable);

      call_api_str += LegalizeVarName(out_name) + ", ";
    } else {
      // There are few Operators that have duplicable output, like `Out` in
      // split op. We need to specify the number of variables for the
      // duplicable output, as the argument OutNum;
      if (output.duplicable()) {
        if (!input_args.empty()) {
          input_args += ",";
        }
        auto out_num_str =
            paddle::string::Sprintf(ARG_OUT_NUM, LegalizeVarName(out_name));
        input_args += ARG_OUT_NUM_TYPE;
        input_args += out_num_str;
        input_args_num++;
        outs_initializer += paddle::string::Sprintf(
            OUT_DUPLICABLE_INITIALIZER_TEMPLATE, out_name, out_num_str);

        auto dispensable = output.dispensable() ? "true" : "false";
        ins_cast_str += paddle::string::Sprintf(CAST_SIZE_T_TEMPLATE,
                                                out_num_str,
                                                op_type,
                                                out_num_str,
                                                arg_idx++,
                                                dispensable);
        call_api_str += out_num_str + ", ";
      } else {
        outs_initializer +=
            paddle::string::Sprintf(OUT_INITIALIZER_TEMPLATE, out_name);
      }
      outs_initializer += ",";
    }
  }
  call_api_str += "attrs);";
  if (outs_initializer.back() == ',') {
    outs_initializer.pop_back();
    // return_str.pop_back();
  }
  outs_initializer += "}";
  if (FindViewOpMap(op_type)) {
    std::string view_input_name = view_op_map[op_type].first;
    std::string view_output_name = view_op_map[op_type].second;
    view_strategy_str +=
        paddle::string::Sprintf(HANDLE_VIEW_BETWEEN_INPUT_AND_OUTPUT,
                                view_input_name,
                                view_output_name,
                                view_input_name,
                                view_output_name);
  }
  if (!inplace_map.empty()) {
    // For inplace op, Use the input PyObject directly.
    return_str = "std::map<ssize_t, ssize_t> inplace_var_idx_map;\n";
    for (auto& inplace_pair : inplace_map) {
      // Find index of inplace tensor, and directly use input PyObject.
      std::string inplace_arg_name = inplace_pair.second;
      std::string inplace_return_name = inplace_pair.first;
      const char* RETURN_INPLACE_TENSOR_TEMPLATE =
          "    ssize_t arg_id = "
          "GetIdxFromCoreOpsInfoMap(core_ops_legacy_args_info, "
          "\"%s\", \"%s\");\n"
          "    ssize_t return_id = "
          "GetIdxFromCoreOpsInfoMap(core_ops_legacy_returns_info, \"%s\", "
          "\"%s\");\n"
          "    inplace_var_idx_map[return_id] = arg_id;";
      return_str += paddle::string::Sprintf(RETURN_INPLACE_TENSOR_TEMPLATE,
                                            op_type,
                                            inplace_arg_name,
                                            op_type,
                                            inplace_return_name);
    }
    return_str += "    return ToPyObject(out, args, inplace_var_idx_map);";
  } else {
    return_str = "return ToPyObject(out);";
  }

  std::string function_args = "";
  if (!input_args.empty()) {
    function_args = FUNCTION_ARGS_NO_INPUT;
  } else {
    function_args = paddle::string::Sprintf(FUNCTION_ARGS, input_args);
  }

  // generate op function body
  auto op_function_str = paddle::string::Sprintf(OP_FUNCTION_TEMPLATE,
                                                 func_name,
                                                 op_type,
                                                 ins_cast_str,
                                                 op_type,
                                                 input_args_num,
                                                 call_api_str,
                                                 return_str);

  return op_function_str;
}

static std::string GenerateCoreOpsInfoMap() {
  std::string result =
      "static PyObject * eager_get_core_ops_args_info(PyObject *self) {\n"
      "  PyThreadState *tstate = nullptr;\n"
      "  try {\n"
      "    return ToPyObject(core_ops_legacy_args_info);\n"
      "  } catch(...) {\n"
      "    if (tstate) {\n"
      "      PyEval_RestoreThread(tstate);\n"
      "    }\n"
      "    ThrowExceptionToPython(std::current_exception());\n"
      "    return nullptr;\n"
      "  }\n"
      "}\n"
      "\n"
      "static PyObject * eager_get_core_ops_args_type_info(PyObject *self) {\n"
      "  PyThreadState *tstate = nullptr;\n"
      "  try {\n"
      "    return ToPyObject(core_ops_legacy_args_type_info);\n"
      "  } catch(...) {\n"
      "    if (tstate) {\n"
      "      PyEval_RestoreThread(tstate);\n"
      "    }\n"
      "    ThrowExceptionToPython(std::current_exception());\n"
      "    return nullptr;\n"
      "  }\n"
      "}\n"
      "\n"
      "static PyObject * eager_get_core_ops_returns_info(PyObject *self) {\n"
      "  PyThreadState *tstate = nullptr;\n"
      "  try {\n"
      "    return ToPyObject(core_ops_legacy_returns_info);\n"
      "  } catch(...) {\n"
      "    if (tstate) {\n"
      "      PyEval_RestoreThread(tstate);\n"
      "    }\n"
      "    ThrowExceptionToPython(std::current_exception());\n"
      "    return nullptr;\n"
      "  }\n"
      "}\n";

  return result;
}

static std::tuple<std::vector<std::string>, std::vector<std::string>>
GenerateOpFunctions() {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list, bind_function_list;
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
  for (auto& pair : op_info_map) {
    auto& op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto& op_type = op_proto->type();
    // Skip operators that will be handwritten in CUSTOM_HANDWRITE_OP_FUNC_FILE.
    if (CUSTOM_HANDWRITE_OPS_SET.count(op_type)) {
      continue;
    }
    // Skip the sparse op
    if (op_type.compare(0, 7, "sparse_") == 0 && op_type != "sparse_momentum" &&
        op_type != "sparse_attention") {
      continue;
    }
    // Skip operator which is not inherit form OperatorWithKernel, like while,
    // since only OperatorWithKernel can run in dygraph mode.
    // if the phi lib contains op kernel, we still generate ops method
    if (!all_kernels.count(op_type) &&
        !phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_type)) {
      continue;
    }
    std::string func_name = "eager_legacy_api_" + op_type;
    std::string op_function_str =
        GenerateOpFunctionsBody(op_proto, func_name, {});

    // generate pybind item
    auto bind_function_str = paddle::string::Sprintf(
        PYBIND_ITEM_TEMPLATE, op_type, func_name, op_type);

    op_function_list.emplace_back(std::move(op_function_str));
    bind_function_list.emplace_back(std::move(bind_function_str));

    // NOTE(pangyoki): Inplace Strategy.
    // In this case, output will reuse input varbase.
    // Dygraph mode needs to be aligned with the in-place strategy in static
    // mode, and the mapping relationships between output and input that have
    // been defined in static graph mode should be used in dygraph mode.
    // Find which ops need to use Inplace strategy in static graph mode, and get
    // the mapping relationship between Inplace output and input.
    auto& infer_inplace =
        paddle::framework::OpInfoMap::Instance().Get(op_type).infer_inplace_;
    std::map<std::string, std::string> inplace_map;
    // `sum` op has duplicate input. Don't consider adding inplace strategy
    // for `sum` in temporary.
    if (infer_inplace && !special_inplace_op_set.count(op_type)) {
      // Inplace OP: op_type_.
      // The inplace OP needs a new implementation method.
      auto in_to_outs = infer_inplace(true);
      for (auto& inplace_pair : in_to_outs) {
        inplace_map[inplace_pair.second] = inplace_pair.first;
      }

      std::string inplace_op_type = op_type + "_";
      std::string inplace_func_name = "eager_legacy_api_" + inplace_op_type;
      std::string inplace_op_function_str =
          GenerateOpFunctionsBody(op_proto, inplace_func_name, inplace_map);

      // generate pybind item
      auto inplace_bind_function_str =
          paddle::string::Sprintf(PYBIND_ITEM_TEMPLATE,
                                  inplace_op_type,
                                  inplace_func_name,
                                  inplace_op_type);

      op_function_list.emplace_back(std::move(inplace_op_function_str));
      bind_function_list.emplace_back(std::move(inplace_bind_function_str));
    }
  }

  return std::make_tuple(op_function_list, bind_function_list);
}

int run_legacy_generator(int argc, char* argv[]) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  // We need a fake device to trigger the registration of the common kernel and
  // generate api
  paddle::operators::RegisterCustomDeviceCommonKernel("fake_device");
#endif

  const std::string str = "\"paddle/fluid/eager/api/generated/fluid_generated/";
  std::vector<std::string> headers{
      "<Python.h>",
      "\"paddle/fluid/platform/enforce.h\"",
      str + "dygraph_forward_api.h\"",
      "\"paddle/fluid/pybind/eager_utils.h\"",
      "\"paddle/phi/core/platform/profiler/event_tracing.h\"",
      "\"paddle/fluid/pybind/exception.h\"",
      "\"paddle/fluid/pybind/op_function_common.h\"",
      "\"paddle/fluid/pybind/eager_legacy_custom_python_api.h\"",
      "\"paddle/fluid/pybind/eager.h\""};

  std::ofstream out(argv[1], std::ios::out);

  for (auto& header : headers) {
    out << "#include  " + header + "\n";
  }

  out << "\n\n";

  auto op_funcs = GenerateOpFunctions();
  auto core_ops_infos = GenerateCoreOpsInfoMap();
  std::string core_ops_infos_registry =
      "  {\"get_core_ops_args_info\", "
      "(PyCFunction)(void(*)(void))eager_get_core_ops_args_info, METH_NOARGS, "
      "\"C++ interface function for eager_get_core_ops_args_info.\"},\n"
      "  {\"get_core_ops_args_type_info\", "
      "(PyCFunction)(void(*)(void))eager_get_core_ops_args_type_info, "
      "METH_NOARGS, "
      "\"C++ interface function for eager_get_core_ops_args_type_info.\"},\n"
      "  {\"get_core_ops_returns_info\", "
      "(PyCFunction)(void(*)(void))eager_get_core_ops_returns_info, "
      "METH_NOARGS, \"C++ interface function for "
      "eager_get_core_ops_returns_info.\"},\n";

  out << "namespace paddle {\n"
      << "namespace pybind {\n\n";
  out << core_ops_infos;
  out << paddle::string::join_strings(std::get<0>(op_funcs), '\n');
  out << "\n\n";

  out << "static PyMethodDef ExtestMethods[] = {\n"
      << paddle::string::join_strings(std::get<1>(op_funcs), '\n') << "\n"
      << core_ops_infos_registry << "\n  {nullptr,nullptr,0,nullptr}"
      << "};\n\n";

  out << "void BindEagerOpFunctions(pybind11::module *module) {\n"
      << "  InitOpsAttrTypeMap();\n"
      << "  auto m = module->def_submodule(\"ops\");\n"
      << "  auto legacy = m.def_submodule(\"legacy\");\n"
      << "  if (PyModule_AddFunctions(legacy.ptr(), ExtestMethods) < 0) {\n"
      << "    PADDLE_THROW(common::errors::Fatal (\"Add functions to "
         "core.eager.ops failed!\"));\n"
      << "  }\n\n"
      << "  if (PyModule_AddFunctions(legacy.ptr(), CustomEagerMethods) < "
         "0) {\n"
      << "    PADDLE_THROW(common::errors::Fatal (\"Add functions to "
         "core.eager.ops failed!\"));\n"
      << "  }\n\n"

      << "  BindFinalStateEagerOpFunctions(&m);\n"
      << "}\n\n"
      << "} // namespace pybind\n"
      << "} // namespace paddle\n";

  out.close();

  return 0;
}
