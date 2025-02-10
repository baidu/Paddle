# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import yaml
from api_base import BaseAPI


class BackwardAPI(BaseAPI):
    def __init__(self, backward_item_yaml):
        super().__init__(backward_item_yaml)
        self.check_args(backward_item_yaml['forward'])
        self.no_need_buffer = self.parse_no_need_buffer(backward_item_yaml)

    def get_api_name(self, api_item_yaml):
        return api_item_yaml['backward_op']

    def parse_forward_config(self, forward_config):
        # api_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
        result = re.search(
            r"(?P<op>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->\s*(?P<outputs>.+)",
            forward_config,
        )
        api = result.group('op')
        (
            _,
            outputs,
            _,
        ) = self.parse_output(self.api, result.group('outputs'))
        outputs = [item.split('@')[0] for item in outputs]
        fw_inputs, fw_attrs = self.parse_input_and_attr(
            api, result.group('args')
        )

        return api, fw_inputs, fw_attrs, outputs

    def parse_no_need_buffer(self, api_item_yaml):
        no_need_buffer = []
        if 'no_need_buffer' in api_item_yaml:
            no_need_buffer = [
                item.strip()
                for item in api_item_yaml['no_need_buffer'].split(',')
            ]
        return no_need_buffer

    def check_args(self, forward_config):
        # parse the forward and backward config
        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            forward_config
        )

        # check the inputs of backward
        for input in self.inputs['names']:
            if input not in fw_inputs['names'] and input not in fw_outputs:
                if input.endswith('_grad'):
                    original_name = input[:-5]
                    assert (
                        original_name in fw_outputs
                    ), f"{self.api} : Input Tensor error: the input tensor({input}) of backward should be an input or output or grad of output in forward api. \
                         Please check the forward of {self.api} in yaml."

        # check the attributes of backward
        for attr in self.attrs['names']:
            assert (
                attr in fw_attrs['names']
                and self.attrs['attr_info'][attr][0]
                == fw_attrs['attr_info'][attr][0]
            ) or self.attrs['attr_info'][attr][
                1
            ] is not None, f"{self.api} : Attribute error: The attribute({attr}) of backward isn't consistent with forward api or doesn't have default value. \
                 Please check the args of {self.api} in yaml."

        # check the output of backward
        assert len(self.outputs['types']) <= len(
            fw_inputs['names']
        ), f"{self.api} : Output error: The number of outputs should be less then the number of inputs of forward api. \
             Please check the output of {self.api} in yaml."

    def get_declare_args(self, inplace_flag=False):
        return self.get_define_args()

    def get_define_args(self, inplace_flag=False):
        out_type_map = {
            'Tensor': 'Tensor*',
            'std::vector<Tensor>': 'std::vector<Tensor*>',
        }
        inputs_and_attrs = super().get_define_args()
        outs = []
        for i, name in enumerate(self.outputs['names']):
            outs.append(
                out_type_map[self.outputs['types'][i]]
                + ' '
                + name.split('@')[0]
            )
        result = inputs_and_attrs + ', ' + ", ".join(outs)
        return result

    def gene_return_code(self):
        return ""

    def gene_api_declaration(self):
        if not self.is_base_api and not self.is_only_composite_api:
            invoke_func_name = self.invoke.split('(')[0]
            if (not invoke_func_name.endswith("_grad")) and (
                not invoke_func_name.endswith('_impl')
            ):
                return ""
        api_func_name = self.get_api_func_name()
        api_declaration = f"""
PADDLE_API void {api_func_name}({self.get_declare_args()});
"""
        return api_declaration

    def gene_kernel_backend_select(self):
        all_no_need_buffer = True
        for in_name in self.inputs['names']:
            if in_name not in self.no_need_buffer:
                all_no_need_buffer = False

        if all_no_need_buffer:
            return """
  kernel_backend = ParseBackend(egr::Controller::Instance().GetExpectedPlace());
"""
        else:
            return super().gene_kernel_backend_select()

    def get_return_type(self, inplace_flag=False):
        return 'void'

    def gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
        kernel_output = []
        output_names = []
        output_create = ""

        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
            inplace_assign = (
                " = " + self.inplace_map[self.outputs['names'][0]]
                if inplace_flag
                and self.inplace_map is not None
                and self.outputs['names'][0] in self.inplace_map
                else ""
            )
            output_create = ""
            set_out_func = (
                'SetKernelOutput'
                if out_tensor_type_list is None
                or out_tensor_type_list[0] == 'dense'
                else 'SetSelectedRowsKernelOutput'
            )
            if out_dtype_list[0] == 'std::vector<Tensor>':
                assert (
                    self.outputs['out_size_expr'] is not None
                ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}(&{self.outputs['names'][0]});"""
                )

            else:
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}({self.outputs['names'][0]});"""
                )

        elif len(out_dtype_list) > 1:
            output_create = ""
            for i, out_type_item in enumerate(out_dtype_list):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
                set_out_func = (
                    'SetKernelOutput'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else 'SetSelectedRowsKernelOutput'
                )
                if out_type_item == 'Tensor':
                    if (
                        inplace_flag
                        and self.inplace_map is not None
                        and self.outputs['names'][i] in self.inplace_map
                    ):
                        output_create = (
                            output_create
                            + f"""
{code_indent}  *{self.outputs['names'][i]} = {self.inplace_map[self.outputs['names'][i]]};"""
                        )

                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({self.outputs['names'][i]});"""
                    )

                else:
                    if (
                        inplace_flag
                        and self.inplace_map is not None
                        and self.outputs['names'][i] in self.inplace_map
                    ):
                        output_create = (
                            output_create
                            + f"""
{code_indent}  *{self.outputs['names'][i]} = {self.inplace_map[self.outputs['names'][i]]};"""
                        )

                    assert (
                        self.outputs['out_size_expr'][i] is not None
                    ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}(&{self.outputs['names'][i]});"""
                    )

        else:
            raise ValueError(
                f"{self.api} : Output error: the output should not be empty."
            )

        return kernel_output, output_names, output_create

    def gene_invoke_code(self, invoke_code, params_code):
        invoke_func_name = invoke_code.split('(')[0].strip()
        if invoke_func_name.endswith('_grad') or invoke_func_name.endswith(
            '_impl'
        ):
            return f"""
PADDLE_API {self.get_return_type()} {self.api}({params_code}) {{
  {invoke_code};
}}"""

        else:
            return ""


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path, fw_header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "{fw_header_file_path}"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/fusion.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

PD_DECLARE_bool(conv2d_disable_cudnn);
COMMON_DECLARE_int32(low_precision_op_list);
COMMON_DECLARE_bool(benchmark);
"""


def backward_api_namespace():
    return (
        """
namespace paddle {
namespace experimental {

""",
        """

}  // namespace experimental
}  // namespace paddle
""",
    )


def generate_backward_api(
    backward_yaml_path,
    is_fused_backward_yaml,
    header_file_path,
    source_file_path,
):
    bw_apis = []
    for each_api_yaml in backward_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                bw_apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = backward_api_namespace()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = (
        "paddle/phi/api/backward/fused_backward_api.h"
        if is_fused_backward_yaml
        else "paddle/phi/api/backward/backward_api.h"
    )
    include_fw_header_file = (
        "paddle/phi/api/include/fused_api.h"
        if is_fused_backward_yaml
        else "paddle/phi/api/include/api.h"
    )
    source_file.write(
        source_include(include_header_file, include_fw_header_file)
    )
    source_file.write(namespace[0])
    # not all fused ops support dygraph
    if is_fused_backward_yaml is True:
        new_bw_apis = [
            bw_api
            for bw_api in bw_apis
            if "support_dygraph_mode" in bw_api
            and bw_api["support_dygraph_mode"] is True
        ]
        bw_apis = new_bw_apis

    for bw_api in bw_apis:
        bw_api = BackwardAPI(bw_api)
        header_file.write(bw_api.gene_api_declaration())
        source_file.write(bw_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ backward API files'
    )
    parser.add_argument(
        '--backward_yaml_path',
        help='path to backward yaml file',
        nargs='+',
        default=['paddle/phi/ops/yaml/backward.yaml'],
    )

    parser.add_argument(
        '--is_fused_backward_yaml',
        help='flag of fused backward yaml',
        action='store_true',
    )

    parser.add_argument(
        '--backward_header_path',
        help='output of generated backward header code file',
        default='paddle/phi/api/backward/backward_api.h',
    )

    parser.add_argument(
        '--backward_source_path',
        help='output of generated backward source code file',
        default='paddle/phi/api/lib/backward_api.cc',
    )

    options = parser.parse_args()

    backward_yaml_path = options.backward_yaml_path
    is_fused_backward_yaml = options.is_fused_backward_yaml
    header_file_path = options.backward_header_path
    source_file_path = options.backward_source_path

    generate_backward_api(
        backward_yaml_path,
        is_fused_backward_yaml,
        header_file_path,
        source_file_path,
    )


if __name__ == '__main__':
    main()
