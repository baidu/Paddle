# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import re

from api_gen import (
    INTARRAY_ATTRIBUTE,
    NAMESPACE_TEMPLATE,
    OP_INPUT,
    VECTOR_TYPE,
    CodeGen,
)

H_FILE_TEMPLATE = """

#pragma once

#include <Python.h>

// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

{body}

"""

API_DECLARE_TEMPLATE = """
PyObject *static_api_{name}(PyObject *self, PyObject *args, PyObject *kwargs);
"""


CPP_FILE_TEMPLATE = """

#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/fluid/pybind/op_callstack_utils.h"


{body}

"""

NO_MUTABLE_ATTR_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

        // Get Value from args
        {inputs}

        // Parse Attributes
        {attrs}

        // Call ir static api
        CallStackRecorder callstack_recorder("{api_name}");
        callstack_recorder.Record();
        auto static_api_out = paddle::dialect::{api_name}({args});
        callstack_recorder.AttachToOps();
        return ToPyObject(static_api_out);
    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

NO_OUTPUT_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

        // Get Value from args
        {inputs}

        // Parse Attributes
        {attrs}

        // Call ir static api
        CallStackRecorder callstack_recorder("{api_name}");
        callstack_recorder.Record();
        paddle::dialect::{api_name}({args});
        callstack_recorder.AttachToOps();
        Py_RETURN_NONE;
    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

INPUT_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});
        auto {name} = {cast_func}({name}_obj, "{api_name}", {index}, {dispensable});"""

NO_MUTABLE_ATTR_CAST_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});
        {type} {name} = {cast_func}({name}_obj, "{api_name}", {index});"""

MUTABLE_ATTR_API_IMPL_TEMPLATE = """
PyObject *static_api_{api_name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
    try {{
        VLOG(6) << "Add {api_name} op into program";
        VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

        // Get Value from args
        {inputs}

        // Parse Attributes
        {attrs_py_obj}

        // Check for mutable attrs
        {init_attrs}
        {cast_attrs}

        // Call ir static api
        CallStackRecorder callstack_recorder("{api_name}");
        callstack_recorder.Record();
        auto static_api_out = paddle::dialect::{api_name}({args_with_mutable_attrs});
        callstack_recorder.AttachToOps();
        return ToPyObject(static_api_out);


    }} catch (...) {{
        ThrowExceptionToPython(std::current_exception());
        return nullptr;
    }}
}}
"""

INIT_ATTRS_TEMPLATE = """
       {type} {name};
"""
MUTABLE_ATTR_TEMPLATE = """
        if (PyObject_CheckIRValue({name}_obj)){{
            {mutable_cast_attrs}
        }}else{{
            {no_mutable_cast_attrs}
        }}"""

MUTABLE_ATTR_LIST_TEMPLATE = """
        if (PyObject_CheckIRValue({name}_obj)){{
           {mutable_cast_attrs}
        }}else if (PyObject_CheckIRVectorOfValue({name}_obj)){{
           {mutable_vector_cast_attrs}
        }}else{{
           {no_mutable_cast_attrs}
        }}"""

MUTABLE_ATTR_OBJ_TEMPLATE = """
        PyObject *{name}_obj = PyTuple_GET_ITEM(args, {index});"""

MUTABLE_ATTR_CAST_TEMPLATE = """
            {type} {name_} = {cast_func}({name}_obj, "{api_name}", {index});"""

FULL_OP_TEMPLATE = """
            {name} = paddle::dialect::full(std::vector<int64_t>{{1}}, {name}_tmp, phi::DataType::{phi_datatype}, phi::CPUPlace());
"""

FULL_INT_ARRAY_OP_TEMPLATE = """
            {name} = paddle::dialect::full_int_array({name}_tmp, phi::DataType::{phi_datatype}, phi::CPUPlace());
"""

BUILTIN_STACK_OP_TEMPLATE = """
            {name} = paddle::dialect::stack({name}_tmp, /*axis*/0);
"""
TYPE_TO_FUNC_MAP = {
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
    "phi::Place": "CastPyArg2Place",
    "Place": "CastPyArg2Place",
    "phi::DataType": "CastPyArg2DataTypeDirectly",
}

TYPE_TO_PHI_DATATYPE_MAP = {
    "bool": "BOOL",
    "int": "INT32",
    "long": "INT64",
    "int64_t": "INT64",
    "float": "FLOAT32",
    "double": "FLOAT64",
    "std::vector<bool>": "BOOL",
    "std::vector<int>": "INT32",
    "std::vector<long>": "INT64",
    "std::vector<int64_t>": "INT64",
    "std::vector<float>": "FLOAT32",
    "std::vector<double>": "FLOAT64",
}

MANUAL_STATIC_OP_FUNCTION_LIST = ['full']


class PythonCCodeGen(CodeGen):
    def __init__(self) -> None:
        super().__init__()

    def _gen_one_declare(self, op_name):
        return API_DECLARE_TEMPLATE.format(name=op_name)

    def _gen_h_file(self, op_info_items, namespaces, h_file_path):
        declare_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if self._need_skip(op_info, op_name):
                    continue
                sparse_op_name_suffix = "_sp" if op_info.is_sparse_op else ''
                sparse_op_inplace_name_suffix = (
                    "sp_" if op_info.is_sparse_op else ''
                )
                if op_name[-1] == "_":
                    declare_str += self._gen_one_declare(
                        op_name + sparse_op_inplace_name_suffix
                    )
                else:
                    declare_str += self._gen_one_declare(
                        op_name + sparse_op_name_suffix
                    )

        body = declare_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(h_file_path, 'w') as f:
            f.write(H_FILE_TEMPLATE.format(body=body))

    def _gen_inputs(self, op_info, op_name):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        optional_list = op_info.input_optional_list
        assert len(name_list) == len(type_list) == len(optional_list)
        ret = ''
        for i, (name, type, optional) in enumerate(
            zip(name_list, type_list, optional_list)
        ):
            if optional == 'true':
                cast_func = (
                    'CastPyArg2OptionalVectorOfValue'
                    if VECTOR_TYPE in type
                    else 'CastPyArg2OptionalValue'
                )
                dispensable = "true"
            else:
                cast_func = (
                    'CastPyArg2VectorOfValue'
                    if VECTOR_TYPE in type
                    else 'CastPyArg2Value'
                )
                dispensable = "false"
            ret += INPUT_TEMPLATE.format(
                name=name,
                index=i,
                cast_func=cast_func,
                api_name=op_name,
                dispensable=dispensable,
            )
        return ret

    def _gen_attrs_without_mutable(self, op_info, op_name):
        input_size = len(op_info.input_name_list)
        name_list = op_info.attribute_name_list
        type_list = op_info.attribute_build_arg_type_list
        assert len(name_list) == len(type_list)
        ret = ''
        for i, (name, type) in enumerate(zip(name_list, type_list)):
            type = type.replace('const ', '').replace('&', '')
            cast_func = TYPE_TO_FUNC_MAP[type]
            ret += NO_MUTABLE_ATTR_CAST_TEMPLATE.format(
                name=name,
                index=input_size + i,
                type=type,
                cast_func=cast_func,
                api_name=op_name,
            )
        return ret

    def _gen_attrs_py_obj_with_mutable(self, op_info):
        input_size = len(op_info.input_name_list)
        name_list = op_info.attribute_name_list
        ret = ''
        for i, name in enumerate(name_list):
            ret += MUTABLE_ATTR_OBJ_TEMPLATE.format(
                name=name, index=input_size + i
            )
        return ret

    def _gen_init_mutable_attrs(self, op_info):
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        ret = ''
        for name in mutable_attr_name_list:
            ret += INIT_ATTRS_TEMPLATE.format(type=OP_INPUT, name=name)

        return ret

    def _gen_cast_attrs(self, op_info, op_name):
        input_size = len(op_info.input_name_list)
        attr_name_list = op_info.attribute_name_list
        attr_type_list = op_info.attribute_build_arg_type_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        mutable_attr_type_list = op_info.mutable_attribute_type_list
        assert len(attr_name_list) == len(attr_type_list)
        ret = ''
        for i, (name, type) in enumerate(zip(attr_name_list, attr_type_list)):
            type = type.replace('const ', '').replace('&', '')
            cast_func = TYPE_TO_FUNC_MAP[type]

            if name in mutable_attr_name_list:
                phi_dtype = TYPE_TO_PHI_DATATYPE_MAP[type]
                if (
                    mutable_attr_type_list[mutable_attr_name_list.index(name)][
                        0
                    ]
                    == INTARRAY_ATTRIBUTE
                ):
                    mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='',
                        name_=name,
                        name=name,
                        cast_func='CastPyArg2Value',
                        api_name=op_name,
                        index=input_size + i,
                    )

                    mutable_vector_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='std::vector<pir::Value>',
                        name_=name + '_tmp',
                        name=name,
                        cast_func='CastPyArg2VectorOfValue',
                        api_name=op_name,
                        index=input_size + i,
                    )
                    mutable_vector_cast_str += BUILTIN_STACK_OP_TEMPLATE.format(
                        name=name
                    )

                else:
                    mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                        type='',
                        name_=name,
                        name=name,
                        cast_func='CastPyArg2Value',
                        api_name=op_name,
                        index=input_size + i,
                    )

                no_mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                    type=type,
                    name_=name + '_tmp',
                    name=name,
                    cast_func=cast_func,
                    api_name=op_name,
                    index=input_size + i,
                )

                if (
                    mutable_attr_type_list[mutable_attr_name_list.index(name)][
                        0
                    ]
                    == INTARRAY_ATTRIBUTE
                ):
                    no_mutable_cast_str += FULL_INT_ARRAY_OP_TEMPLATE.format(
                        name=name,
                        phi_datatype=phi_dtype,
                    )
                    ret += MUTABLE_ATTR_LIST_TEMPLATE.format(
                        name=name,
                        mutable_cast_attrs=mutable_cast_str,
                        mutable_vector_cast_attrs=mutable_vector_cast_str,
                        no_mutable_cast_attrs=no_mutable_cast_str,
                    )
                else:
                    no_mutable_cast_str += FULL_OP_TEMPLATE.format(
                        name=name,
                        phi_datatype=phi_dtype,
                    )
                    ret += MUTABLE_ATTR_TEMPLATE.format(
                        name=name,
                        mutable_cast_attrs=mutable_cast_str,
                        no_mutable_cast_attrs=no_mutable_cast_str,
                    )
            else:
                mutable_cast_str = MUTABLE_ATTR_CAST_TEMPLATE.format(
                    type=type,
                    name_=name,
                    name=name,
                    cast_func=cast_func,
                    api_name=op_name,
                    index=input_size + i,
                )
                ret += mutable_cast_str

        return ret

    def _gen_one_impl(self, op_info, op_name):
        input_name_list = op_info.input_name_list
        output_name_list = op_info.output_name_list
        attr_name_list = op_info.attribute_name_list
        mutable_attr_name_list = op_info.mutable_attribute_name_list
        no_mutable_attr_name_list = op_info.non_mutable_attribute_name_list

        if len(output_name_list) == 0:
            ret = NO_OUTPUT_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                inputs=self._gen_inputs(op_info, op_name),
                attrs=self._gen_attrs_without_mutable(op_info, op_name),
                args=', '.join(input_name_list + attr_name_list),
            )
        elif len(mutable_attr_name_list) > 0:
            ret = MUTABLE_ATTR_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                inputs=self._gen_inputs(op_info, op_name),
                attrs_py_obj=self._gen_attrs_py_obj_with_mutable(op_info),
                init_attrs=self._gen_init_mutable_attrs(op_info),
                cast_attrs=self._gen_cast_attrs(op_info, op_name),
                args_with_mutable_attrs=', '.join(
                    input_name_list
                    + mutable_attr_name_list
                    + no_mutable_attr_name_list
                ),
            )
        else:
            ret = NO_MUTABLE_ATTR_API_IMPL_TEMPLATE.format(
                api_name=op_name,
                inputs=self._gen_inputs(op_info, op_name),
                attrs=self._gen_attrs_without_mutable(op_info, op_name),
                args=', '.join(input_name_list + attr_name_list),
            )
        ret = re.sub(r' +\n', '', ret)
        return ret

    def _need_skip(self, op_info, op_name):
        return (
            super()._need_skip(op_info, op_name)
            or op_name.endswith('xpu')
            or op_name in MANUAL_STATIC_OP_FUNCTION_LIST
        )

    def _gen_cpp_file(self, op_info_items, namespaces, cpp_file_path):
        impl_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if self._need_skip(op_info, op_name):
                    continue
                sparse_op_name_suffix = "_sp" if op_info.is_sparse_op else ''
                sparse_op_inplace_name_suffix = (
                    "sp_" if op_info.is_sparse_op else ''
                )
                if op_name[-1] == "_":
                    impl_str += self._gen_one_impl(
                        op_info, op_name + sparse_op_inplace_name_suffix
                    )
                else:
                    impl_str += self._gen_one_impl(
                        op_info, op_name + sparse_op_name_suffix
                    )
        body = impl_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body))


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Dialect Python C Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--python_c_def_h_file', type=str)
    parser.add_argument('--python_c_def_cc_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    python_c_def_h_file = args.python_c_def_h_file
    python_c_def_cc_file = args.python_c_def_cc_file

    code_gen = PythonCCodeGen()
    code_gen.gen_h_and_cpp_file(
        op_yaml_files,
        op_compat_yaml_file,
        namespaces,
        python_c_def_h_file,
        python_c_def_cc_file,
    )
