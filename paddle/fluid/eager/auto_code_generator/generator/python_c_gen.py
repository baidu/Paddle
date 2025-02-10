# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from codegen_utils import (
    FunctionGeneratorBase,
    GeneratorBase,
    GetForwardFunctionName,
    GetInplacedFunctionName,
    IsVectorTensorType,
)

#########################
# Global Configurations #
#########################
skipped_forward_api_names = {
    "scale_grad",
    "push_gpups_sparse",
    "embedding_grad",
    "multiply_grad",
    "cudnn_lstm_grad",
    "conv2d_grad",
    "pull_sparse_v2_grad",
}


def SkipAPIGeneration(forward_api_name):
    return forward_api_name in skipped_forward_api_names


atype_to_parsing_function = {
    "bool": "CastPyArg2Boolean",
    "int": "CastPyArg2Int",
    "long": "CastPyArg2Long",
    "int64_t": "CastPyArg2Long",
    "float": "CastPyArg2Float",
    "double": "CastPyArg2Double",
    "std::string": "CastPyArg2String",
    "std::vector<bool>": "CastPyArg2Booleans",
    "std::vector<int>": "CastPyArg2Ints",
    "std::vector<long>": "CastPyArg2Longs",
    "std::vector<int64_t>": "CastPyArg2Longs",
    "std::vector<float>": "CastPyArg2Floats",
    "std::vector<double>": "CastPyArg2Float64s",
    "std::vector<std::string>": "CastPyArg2Strings",
    "paddle::experimental::Scalar": "CastPyArg2Scalar",
    "std::vector<phi::Scalar>": "CastPyArg2ScalarArray",
    "paddle::experimental::IntArray": "CastPyArg2IntArray",
    "paddle::Place": "CastPyArg2Place",
    "phi::DataType": "CastPyArg2DataType",
}


def FindParsingFunctionFromAttributeType(atype):
    if atype not in atype_to_parsing_function.keys():
        raise AssertionError(
            f"Unable to find {atype} in atype_to_parsing_function."
        )

    return atype_to_parsing_function[atype]


########################
# Refactored Functions #
########################
PARSE_PYTHON_C_TENSORS_TEMPLATE = (
    '    auto {} = {}("{}", "{}", args, {}, {});\n'
)

PARSE_PYTHON_C_TENSOR_REF_TEMPLATE = (
    '    auto& {} = {}("{}", "{}", args, {}, {});\n'
)

CONVERT_TO_DISTTENSOR_AND_PARSE_PYTHON_C_TENSORS_TEMPLATE = (
    '    {} = {}("{}", "{}", args, {}, {}, mesh);\n'
)

CONVERT_INPUT_TENSORS_TO_DIST_TENSOR_WITH_SINGLE_TENSOR_TEMPLATE = """
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (egr::InputsContainDistTensor(&mesh{input_names})) {{
      egr::ConvertAllInputsToDistTensor(mesh{input_single_tensor_names});
      {optional_and_vector_convert_code}
    }}
"""

CONVERT_INPUT_TENSORS_TO_DIST_TENSOR_WITHOUT_SINGLE_TENSOR_TEMPLATE = """
    const phi::distributed::ProcessMesh* mesh = nullptr;
    if (egr::InputsContainDistTensor(&mesh{input_names})) {{
      {optional_and_vector_convert_code}
    }}
"""

PARSE_PYTHON_C_ARGS_TEMPLATE = """    PyObject* {}_obj = PyTuple_GET_ITEM(args, {});
    {} {} = {}({}_obj, \"{}\", {});
"""


RECORD_EVENT_TEMPLATE = (
    'phi::RecordEvent {}("{} {}", phi::TracerEventType::UserDefined, 1);'
)


RETURN_INPLACE_PYOBJECT_TEMPLATE = """
    inplace_var_idx_map[{}] = {};
"""


PYTHON_C_FUNCTION_TEMPLATE = """
PyObject * eager_api_{}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  {}
  PyThreadState *tstate = nullptr;
  try {{
    VLOG(6) << "Running Eager Final State API: {}";

    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);
    // Get EagerTensors from args
{}
    // Parse Attributes if needed
{}
    tstate = PyEval_SaveThread();

    // Set Device ID
{}
    // Call dygraph function
    {}

    PyEval_RestoreThread(tstate);
    tstate = nullptr;
{}
  }} catch(...) {{
    if (tstate) {{
      PyEval_RestoreThread(tstate);
    }}
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }}
}}
"""

NOAMP_DYGRAPH_FUNCTION_TEMPLATE = "decltype({}({})) ad_func_out = {}({});"


FUNCTION_SET_DEVICE_TEMPLATE = """{}
    SetPythonStack();
    if (phi::is_gpu_place(place)) {{
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::SetDeviceId(place.device);
      VLOG(4) <<"CurrentDeviceId: " << phi::backends::gpu::GetCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    }}
    if (phi::is_custom_place(place)) {{
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      phi::DeviceManager::SetDevice(place);
      VLOG(4) <<"CurrentDeviceId: " << phi::DeviceManager::GetDevice(place.GetDeviceType()) << " from " << (int)place.device;
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with CUSTOM_DEVICE if use CustomPlace."));
#endif
    }}
    if (phi::is_xpu_place(place)) {{
#if defined(PADDLE_WITH_XPU)
      phi::backends::xpu::SetXPUDeviceId(place.device);
      VLOG(4) <<"CurrentDeviceId: " << phi::backends::xpu::GetXPUCurrentDeviceId() << " from " << (int)place.device;
#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    }}
"""

FUNCTION_NAME_TEMPLATE = "{}{}{}"


PYTHON_C_FUNCTION_REG_TEMPLATE = '  {{"{}{}", (PyCFunction)(void(*)(void)) {}eager_api_{}, METH_VARARGS | METH_KEYWORDS, "C++ interface function for {} in dygraph."}},\n'


PYTHON_C_WRAPPER_TEMPLATE = """
#include <Python.h>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/api/include/strings_api.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/pybind/eager_custom_python_api.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_op_function.h"
namespace paddle {{
namespace pybind {{

{}

static PyMethodDef EagerFinalStateMethods[] = {{
{}
}};

void BindFinalStateEagerOpFunctions(pybind11::module *module) {{
  if (PyModule_AddFunctions(module->ptr(), EagerFinalStateMethods) < 0) {{
    PADDLE_THROW(common::errors::Fatal ("Add functions to core.eager.ops failed!"));
  }}

  if (PyModule_AddFunctions(module->ptr(), CustomEagerFinalStateMethods) < 0) {{
    PADDLE_THROW(common::errors::Fatal ("Add functions to core.eager.ops failed!"));
  }}
}}

}} // namespace pybind
}} // namespace paddle
"""


CORE_OPS_INFO = """
static PyObject * eager_get_core_ops_args_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try {
      return ToPyObject(core_ops_args_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}

static PyObject * eager_get_core_ops_args_type_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try {
      return ToPyObject(core_ops_args_type_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}

static PyObject * eager_get_core_ops_returns_info(PyObject *self) {
    PyThreadState *tstate = nullptr;
    try {
      return ToPyObject(core_ops_returns_info);
    }
    catch(...) {
      if (tstate) {
        PyEval_RestoreThread(tstate);
      }
      ThrowExceptionToPython(std::current_exception());
      return nullptr;
    }
}
"""


CORE_OPS_INFO_REGISTRY = """
  {\"get_core_ops_args_info\", (PyCFunction)(void(*)(void))eager_get_core_ops_args_info, METH_NOARGS, \"C++ interface function for eager_get_core_ops_args_info.\"},
  {\"get_core_ops_args_type_info\", (PyCFunction)(void(*)(void))eager_get_core_ops_args_type_info, METH_NOARGS, \"C++ interface function for eager_get_core_ops_args_type_info.\"},
  {\"get_core_ops_returns_info\", (PyCFunction)(void(*)(void))eager_get_core_ops_returns_info, METH_NOARGS, \"C++ interface function for eager_get_core_ops_returns_info.\"},
"""

NAMESPACE_WRAPPER_TEMPLATE = """namespace {} {{
    {}
}}
"""

PYTHON_C_H_TEMPLATE = """
#pragma once

#include <Python.h>

// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

namespace paddle {{
namespace pybind {{

{body}

}} // namespace pybind
}} // namespace paddle
"""

PYTHON_C_FUNCTION_DECLARE_TEMPLATE = """
PyObject *eager_api_{name}(PyObject *self, PyObject *args, PyObject *kwargs);
"""


#####################
# Generator Classes #
#####################
class PythonCSingleFunctionGenerator(FunctionGeneratorBase):
    def __init__(self, forward_api_contents, namespace):
        # Members from Parent:
        # self.namespace
        # self.forward_api_contents
        # self.forward_api_name
        # self.orig_forward_inputs_list
        # self.orig_forward_attrs_list
        # self.orig_forward_returns_list
        # self.forward_inputs_position_map
        # self.forward_outputs_position_map
        # self.optional_inputs
        # self.no_need_buffers
        # self.intermediate_outputs
        # self.forward_inplace_map
        FunctionGeneratorBase.__init__(self, forward_api_contents, namespace)

        self.is_forward_only = True

        # Generated Results
        self.python_c_function_str = ""
        self.python_c_function_reg_str = ""
        self.python_c_function_declare_str = ""

    def CollectIsForwardOnly(self):
        forward_api_contents = self.forward_api_contents
        self.is_forward_only = (
            False if 'backward' in forward_api_contents.keys() else True
        )

    def GeneratePythonCFunction(self):
        namespace = self.namespace
        forward_inplace_map = self.forward_inplace_map
        forward_api_name = self.forward_api_name
        orig_forward_attrs_list = self.orig_forward_attrs_list
        forward_inputs_position_map = self.forward_inputs_position_map
        forward_outputs_position_map = self.forward_outputs_position_map
        optional_inputs = self.optional_inputs
        is_forward_only = self.is_forward_only

        inplace_args_pos_map = {}
        inplace_returns_pos_map = {}
        # Generate Python-C Tensors Parsing Logic
        get_eager_tensor_str = ""
        input_names = ""
        input_single_tensor_names = ""
        for name, (ttype, pos) in forward_inputs_position_map.items():
            input_names = input_names + ", " + name
            if forward_inplace_map and name in forward_inplace_map.keys():
                inplace_args_pos_map[name] = pos
            is_optional = name in optional_inputs
            if IsVectorTensorType(ttype):
                if is_optional:
                    get_eager_tensor_str += (
                        PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                            name,
                            "GetOptionalTensorListFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "true",
                        )
                    )
                else:
                    get_eager_tensor_str += (
                        PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                            name,
                            "GetTensorListFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "false",
                        )
                    )
            else:
                if is_optional:
                    get_eager_tensor_str += (
                        PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                            name,
                            "GetOptionalTensorFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "true",
                        )
                    )
                else:
                    input_single_tensor_names = (
                        input_single_tensor_names + ", " + name
                    )
                    get_eager_tensor_str += (
                        PARSE_PYTHON_C_TENSOR_REF_TEMPLATE.format(
                            name,
                            "GetTensorFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "false",
                        )
                    )
        # No inputs, skip convert to DistTensor
        if len(input_names) > 0:
            optional_and_vector_convert_code = ""
            for name, (ttype, pos) in forward_inputs_position_map.items():
                is_optional = name in optional_inputs
                if IsVectorTensorType(ttype):
                    if is_optional:
                        optional_and_vector_convert_code += CONVERT_TO_DISTTENSOR_AND_PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                            name,
                            "GetOptionalTensorListFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "true",
                        )
                    else:
                        optional_and_vector_convert_code += CONVERT_TO_DISTTENSOR_AND_PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                            name,
                            "GetTensorListFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "false",
                        )
                else:
                    if is_optional:
                        optional_and_vector_convert_code += CONVERT_TO_DISTTENSOR_AND_PARSE_PYTHON_C_TENSORS_TEMPLATE.format(
                            name,
                            "GetOptionalTensorFromArgs",
                            forward_api_name,
                            name,
                            pos,
                            "true",
                        )

            if len(input_single_tensor_names) > 0:
                get_eager_tensor_str += CONVERT_INPUT_TENSORS_TO_DIST_TENSOR_WITH_SINGLE_TENSOR_TEMPLATE.format(
                    input_names=input_names,
                    input_single_tensor_names=input_single_tensor_names,
                    optional_and_vector_convert_code=optional_and_vector_convert_code,
                )
            else:
                get_eager_tensor_str += CONVERT_INPUT_TENSORS_TO_DIST_TENSOR_WITHOUT_SINGLE_TENSOR_TEMPLATE.format(
                    input_names=input_names,
                    optional_and_vector_convert_code=optional_and_vector_convert_code,
                )
        if forward_inplace_map:
            for name, (ttype, pos) in forward_outputs_position_map.items():
                if name in forward_inplace_map.values():
                    inplace_returns_pos_map[name] = pos

        parse_attributes_str = ""
        expected_place_str = (
            "    auto place = egr::Controller::Instance().GetExpectedPlace();\n"
        )

        # Generate Python-C Attributes Parsing Logic
        for name, atype, _, pos in orig_forward_attrs_list:
            parsing_function_name = FindParsingFunctionFromAttributeType(atype)
            # Used input argument place if specified from Python frontend.
            if (
                len(expected_place_str) != 0
                and parsing_function_name == "CastPyArg2Place"
            ):
                expected_place_str = ""
                assert (
                    name == "place"
                ), "Only support 'place' as template argument name in FUNCTION_SET_DEVICE_TEMPLATE."

            parse_attributes_str += PARSE_PYTHON_C_ARGS_TEMPLATE.format(
                name,
                pos,
                atype,
                name,
                parsing_function_name,
                name,
                forward_api_name,
                pos,
            )

        set_device_str = FUNCTION_SET_DEVICE_TEMPLATE.format(expected_place_str)

        # Generate Dygraph Function Call Logic
        num_args = len(forward_inputs_position_map.keys()) + len(
            orig_forward_attrs_list
        )
        dygraph_function_call_list = ["" for i in range(num_args)]
        for name, (_, pos) in forward_inputs_position_map.items():
            dygraph_function_call_list[pos] = f"{name}"
        for name, _, _, pos in orig_forward_attrs_list:
            dygraph_function_call_list[pos] = f"{name}"
        dygraph_function_call_str = ",".join(dygraph_function_call_list)

        # Generate Python-C Function Definitions
        fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
            "::", namespace, GetForwardFunctionName(forward_api_name)
        )

        return_str = "    return ToPyObject(ad_func_out);"

        # Generate Record Event for performance profiling
        pythonc_record_event_str = RECORD_EVENT_TEMPLATE.format(
            "pythonc_record_event", forward_api_name, "pybind_imperative_func"
        )

        noamp_dygraph_function_str = NOAMP_DYGRAPH_FUNCTION_TEMPLATE.format(
            fwd_function_name,
            dygraph_function_call_str,
            fwd_function_name,
            dygraph_function_call_str,
        )

        # Generate Python-C Function Definition
        self.python_c_function_str = PYTHON_C_FUNCTION_TEMPLATE.format(
            forward_api_name,
            pythonc_record_event_str,
            forward_api_name,
            get_eager_tensor_str,
            parse_attributes_str,
            set_device_str,
            noamp_dygraph_function_str,
            return_str,
        )
        self.python_c_function_declare_str = (
            PYTHON_C_FUNCTION_DECLARE_TEMPLATE.format(name=forward_api_name)
        )

        # Set prefix of forward_api_name to avoid conflicts
        prefix = self.namespace.strip("::")
        forward_api_name_prefix = "" if prefix == "" else prefix + "_"

        # Generate Python-C Function Registration
        self.python_c_function_reg_str = PYTHON_C_FUNCTION_REG_TEMPLATE.format(
            forward_api_name_prefix,
            forward_api_name,
            namespace,
            forward_api_name,
            forward_api_name,
        )

        if forward_inplace_map:
            inplaced_forward_api_name = GetInplacedFunctionName(
                self.forward_api_name
            )
            inplaced_fwd_function_name = FUNCTION_NAME_TEMPLATE.format(
                "::",
                namespace,
                GetForwardFunctionName(inplaced_forward_api_name),
            )

            inplace_noamp_dygraph_function_str = (
                NOAMP_DYGRAPH_FUNCTION_TEMPLATE.format(
                    inplaced_fwd_function_name,
                    dygraph_function_call_str,
                    inplaced_fwd_function_name,
                    dygraph_function_call_str,
                )
            )

            return_str = "    std::map<ssize_t, ssize_t> inplace_var_idx_map;"
            for inplace_input, inplace_output in forward_inplace_map.items():
                return_str += RETURN_INPLACE_PYOBJECT_TEMPLATE.format(
                    inplace_returns_pos_map[inplace_output],
                    inplace_args_pos_map[inplace_input],
                )
            return_str += (
                "    return ToPyObject(ad_func_out, args, inplace_var_idx_map);"
            )

            # Generate Python-C Function Definition
            python_c_inplace_func_str = PYTHON_C_FUNCTION_TEMPLATE.format(
                inplaced_forward_api_name,
                pythonc_record_event_str,
                inplaced_forward_api_name,
                get_eager_tensor_str,
                parse_attributes_str,
                set_device_str,
                inplace_noamp_dygraph_function_str,
                return_str,
            )

            python_c_function_declare_str = (
                PYTHON_C_FUNCTION_DECLARE_TEMPLATE.format(
                    name=inplaced_forward_api_name
                )
            )

            python_c_inplace_func_reg_str = (
                PYTHON_C_FUNCTION_REG_TEMPLATE.format(
                    forward_api_name_prefix,
                    inplaced_forward_api_name,
                    namespace,
                    inplaced_forward_api_name,
                    inplaced_forward_api_name,
                )
            )

            # self.forward_api_name ending with '_' means it only has inplace api
            if self.forward_api_name[-1] == '_':
                self.python_c_function_str = python_c_inplace_func_str
                self.python_c_function_declare_str = (
                    python_c_function_declare_str
                )
                # Generate Python-C Function Registration
                self.python_c_function_reg_str = python_c_inplace_func_reg_str
            elif "backward_op" not in self.forward_api_contents:
                self.python_c_function_str += python_c_inplace_func_str
                self.python_c_function_declare_str += (
                    python_c_function_declare_str
                )
                # Generate Python-C Function Registration
                self.python_c_function_reg_str += python_c_inplace_func_reg_str

    def run(self):
        # Initialized is_forward_only
        self.CollectIsForwardOnly()

        # Initialized optional_inputs
        self.ParseDispensable()

        # Initialized forward_inplace_map
        self.ParseForwardInplaceInfo()

        # Initialized orig_forward_inputs_list, orig_forward_returns_list, orig_forward_attrs_list
        self.CollectOriginalForwardInfo()

        if SkipAPIGeneration(self.forward_api_name):
            return False

        # Initialized forward_inputs_position_map, forward_outputs_position_map
        self.DetermineForwardPositionMap(
            self.orig_forward_inputs_list, self.orig_forward_returns_list
        )

        # Code Generation
        self.GeneratePythonCFunction()

        return True


class PythonCGenerator(GeneratorBase):
    def __init__(self, path):
        # Parent members:
        # self.namespace
        # self.api_yaml_path
        # self.forward_api_list
        GeneratorBase.__init__(self, api_yaml_path)

        # Generated Result
        self.python_c_functions_str = ""
        self.python_c_functions_reg_str = ""
        self.python_c_function_declare_str = ""

    def GeneratePythonCFunctions(self):
        namespace = self.namespace

        forward_api_list = self.forward_api_list
        for forward_api_content in forward_api_list:
            if "backward_op" in forward_api_content and forward_api_content[
                "backward_op"
            ].endswith(('double_grad', 'triple_grad', 'grad_grad', '_grad_')):
                continue
            f_generator = PythonCSingleFunctionGenerator(
                forward_api_content, namespace
            )
            status = f_generator.run()

            if status:
                self.python_c_functions_str += (
                    f_generator.python_c_function_str + "\n"
                )
                self.python_c_functions_reg_str += (
                    f_generator.python_c_function_reg_str
                )
                self.python_c_function_declare_str += (
                    f_generator.python_c_function_declare_str
                )

    def AttachNamespace(self):
        namespace = self.namespace
        python_c_functions_str = self.python_c_functions_str

        if namespace != "":
            if namespace.endswith("::"):
                namespace = namespace[:-2]
            self.python_c_functions_str = NAMESPACE_WRAPPER_TEMPLATE.format(
                namespace, python_c_functions_str
            )
            self.python_c_function_declare_str = (
                NAMESPACE_WRAPPER_TEMPLATE.format(
                    namespace, self.python_c_function_declare_str
                )
            )

    def run(self):
        # Infer namespace from yaml_path
        self.InferNameSpace()

        # Read Yaml file
        self.ParseForwardYamlContents()

        # Code Generation
        self.GeneratePythonCFunctions()

        # Wrap with namespace
        self.AttachNamespace()


##########################
# Code Generation Helper #
##########################
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Eager Code Generator Args Parser'
    )
    parser.add_argument('--api_yaml_path', type=str)
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--header_path', type=str)

    args = parser.parse_args()
    return args


def GenerateCoreOpsInfoMap():
    return CORE_OPS_INFO, CORE_OPS_INFO_REGISTRY


def GeneratePythonCWrappers(python_c_function_str, python_c_function_reg_str):
    (
        core_ops_infos_definition,
        core_ops_infos_registry,
    ) = GenerateCoreOpsInfoMap()

    python_c_function_str += core_ops_infos_definition
    python_c_function_reg_str += core_ops_infos_registry
    python_c_function_reg_str += "  {nullptr,nullptr,0,nullptr}"

    python_c_str = PYTHON_C_WRAPPER_TEMPLATE.format(
        python_c_function_str, python_c_function_reg_str
    )

    return python_c_str


def GeneratePythonCFile(filepath, python_c_str):
    with open(filepath, 'a') as f:
        f.write(python_c_str)


if __name__ == "__main__":
    args = ParseArguments()
    api_yaml_paths = args.api_yaml_path.split(",")

    generated_python_c_functions = ""
    generated_python_c_registration = ""
    generated_python_c_functions_header = ""
    for i in range(len(api_yaml_paths)):
        api_yaml_path = api_yaml_paths[i]

        py_c_generator = PythonCGenerator(api_yaml_path)
        py_c_generator.run()

        generated_python_c_functions += (
            py_c_generator.python_c_functions_str + "\n"
        )
        generated_python_c_registration += (
            py_c_generator.python_c_functions_reg_str
        )
        generated_python_c_functions_header += (
            py_c_generator.python_c_function_declare_str
        )

    python_c_str = GeneratePythonCWrappers(
        generated_python_c_functions, generated_python_c_registration
    )

    source_path = args.source_path
    header_path = args.header_path
    for path in [source_path, header_path]:
        if os.path.exists(path):
            os.remove(path)

    GeneratePythonCFile(source_path, python_c_str)
    GeneratePythonCFile(
        header_path,
        PYTHON_C_H_TEMPLATE.format(body=generated_python_c_functions_header),
    )
