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
import math
import os
from pathlib import Path

import yaml
from filters import (
    assert_dense_or_sr,
    cartesian_prod_mapping,
    delete_last_underline,
    find_optional_inputs_name,
    get_infer_var_type_func,
    to_composite_grad_opmaker_name,
    to_input_name,
    to_int_array_tensor_name,
    to_int_array_tensors_name,
    to_op_attr_type,
    to_opmaker_name,
    to_opmaker_name_cstr,
    to_pascal_case,
    to_scalar_tensor_name,
    to_variable_names,
)
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from parse_utils import to_named_dict
from tests_utils import (
    is_base_op,
    is_composite_op,
    is_initializer_list,
    is_only_composite_op,
    is_scalar,
    is_vec,
    supports_inplace,
    supports_no_need_buffer,
)

file_loader = FileSystemLoader(Path(__file__).parent / "templates")
env = Environment(
    loader=file_loader,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
    extensions=['jinja2.ext.do'],
)
env.filters["to_op_attr_type"] = to_op_attr_type
env.filters["to_opmaker_name"] = to_opmaker_name
env.filters["to_pascal_case"] = to_pascal_case
env.filters["to_scalar_tensor_name"] = to_scalar_tensor_name
env.filters["to_int_array_tensor_name"] = to_int_array_tensor_name
env.filters["to_int_array_tensors_name"] = to_int_array_tensors_name
env.filters["to_input_name"] = to_input_name
env.filters["to_opmaker_name_cstr"] = to_opmaker_name_cstr
env.filters["cartesian_prod_mapping"] = cartesian_prod_mapping
env.filters["to_composite_grad_opmaker_name"] = to_composite_grad_opmaker_name
env.filters["to_variable_names"] = to_variable_names
env.filters["get_infer_var_type_func"] = get_infer_var_type_func
env.filters["assert_dense_or_sr"] = assert_dense_or_sr
env.filters["find_optional_inputs_name"] = find_optional_inputs_name
env.tests["base_op"] = is_base_op
env.tests["composite_op"] = is_composite_op
env.tests["only_composite_op"] = is_only_composite_op
env.tests["vec"] = is_vec
env.tests["scalar"] = is_scalar
env.tests["initializer_list"] = is_initializer_list
env.tests["supports_inplace"] = supports_inplace
env.tests["supports_no_need_buffer"] = supports_no_need_buffer


def restruct_io(op):
    op["input_dict"] = to_named_dict(op["inputs"])
    op["attr_dict"] = to_named_dict(op["attrs"])
    op["output_dict"] = to_named_dict(op["outputs"])
    return op


def process_scalar(op_item, scalar_configs):
    scalar_map = {
        'Scalar': 'Scalar',
        'Scalar(float)': 'float',
        'Scalar(double)': 'double',
        'Scalar(int)': 'int',
        'Scalar(int64_t)': 'int64_t',
    }
    if scalar_configs is not None:
        for attr_item in op_item['attrs']:
            if attr_item['name'] in scalar_configs:
                attr_type = attr_item['typename']
                assert (
                    attr_type in scalar_map
                ), f"{op_item['name']}'s scalar in op_compat.yaml is error, the data_type of {attr_item['name']} is expected to be one of Scalar, Scalar(float), Scalar(int) or Scalar(int64_t), but now is {attr_type}."

                scalar_config = scalar_configs[attr_item['name']]
                attr_item['is_support_tensor'] = (
                    True if scalar_config.get('support_tensor') else False
                )
                attr_item['data_type'] = (
                    scalar_config['data_type']
                    if 'data_type' in scalar_config
                    else scalar_map[attr_type]
                )
                if (
                    attr_type == 'Scalar(double)'
                    and attr_item['data_type'] == 'std::string'
                    and 'default_value' in attr_item
                ):
                    attr_item['default_value'] = (
                        '"' + attr_item['default_value'] + '"'
                    )
                if attr_item['is_support_tensor'] is False:
                    if 'tensor_name' in scalar_config:
                        attr_item['tensor_name'] = scalar_config['tensor_name']


def process_int_array(op_item, int_array_configs):
    data_type_map = {
        'int': 'std::vector<int>',
        'int64_t': 'std::vector<int64_t>',
    }
    if int_array_configs is not None:
        for attr_item in op_item['attrs']:
            if attr_item['name'] in int_array_configs:
                attr_type = attr_item['typename']
                assert (
                    attr_item['typename'] == "IntArray"
                ), f"{op_item['name']}'s int_array in op_compat.yaml is error, the data_type of {attr_item['name']} is expected to be one of IntArray, but now is {attr_type}."

                int_array_config = int_array_configs[attr_item['name']]
                attr_item['is_support_tensor'] = (
                    True if int_array_config.get('support_tensor') else False
                )
                attr_item['data_type'] = (
                    data_type_map[int_array_config['data_type']]
                    if 'data_type' in int_array_config
                    else 'std::vector<int64_t>'
                )
                if attr_item['is_support_tensor'] is False:
                    attr_item['manual_flag'] = True
                    if 'tensor_name' in int_array_config:
                        attr_item['tensor_name'] = int_array_config[
                            'tensor_name'
                        ]
                    if 'tensors_name' in int_array_config:
                        attr_item['tensors_name'] = int_array_config[
                            'tensors_name'
                        ]


def add_composite_info(ops, backward_ops, backward_op_dict):
    # add backward composite name in forward
    for op in ops + backward_ops:
        if (
            op["backward"] in backward_op_dict
            and "composite" in backward_op_dict[op["backward"]]
        ):
            op["backward_composite"] = op["backward"]
        else:
            op["backward_composite"] = None

        # add whether only composite
        if (
            op["backward_composite"] is not None
            and "invoke" not in backward_op_dict[op["backward"]]
            and "kernel" not in backward_op_dict[op["backward"]]
        ):
            op["only_backward_composite"] = True
        else:
            op["only_backward_composite"] = False


# add fluid name in ops and backward ops info
def add_fluid_name(dict_list):
    for item in dict_list:
        item["fluid_name"] = item["name"]


# add fluid name of op and params for OpMaker
def add_compat_name(op_fluid_map_list, forward_op_dict, backward_op_dict):
    def get_phi_and_fluid_op_name(op_item):
        names = op_item.split('(')
        if len(names) == 1:
            return names[0].strip(), names[0].strip()
        else:
            return names[0].strip(), names[1].split(')')[0].strip()

    def add_op_param_name(op_args, args_alias_map):
        for item in op_args:
            if item['name'] in args_alias_map:
                item['fluid_name'] = args_alias_map[item['name']]
            else:
                item['fluid_name'] = item['name']

    def add_grad_args_name(op_args, args_alias_map):
        for item in op_args:
            if (
                item['name'].endswith('_grad')
                and item['name'][:-5] in args_alias_map
            ):
                args_alias_map[item['name']] = (
                    args_alias_map[item['name'][:-5]] + '_grad'
                )
                item['fluid_name'] = args_alias_map[item['name'][:-5]] + '_grad'
            elif (
                item['name'].endswith('_grad')
                and item['name'][:-5] not in args_alias_map
            ):
                item['fluid_name'] = item['name']

    def get_param_list_alias(param_list, args_map):
        return [
            args_map[param] if param in args_map else param
            for param in param_list
        ]

    def update_common_params_name(
        op_item, args_name_map, scalar_configs, int_array_configs
    ):
        if op_item.get('inplace'):
            inplace_map = {}
            for key, val in op_item['inplace'].items():
                if key in args_map:
                    key = args_map[key]
                if val in args_map:
                    val = args_map[val]
                inplace_map[key] = val
            op_item['inplace'] = inplace_map
        if op_item.get('no_need_buffer'):
            op_item['no_need_buffer'] = get_param_list_alias(
                op_item['no_need_buffer'], args_map
            )
        if op_item.get('data_transform'):
            data_trans_item = op_item['data_transform']
            if 'skip_transform' in data_trans_item:
                data_trans_item['skip_transform'] = get_param_list_alias(
                    data_trans_item['skip_transform'], args_map
                )
            if 'support_trans_dtype' in data_trans_item:
                data_trans_item['support_trans_dtype'] = get_param_list_alias(
                    data_trans_item['support_trans_dtype'], args_map
                )

        process_scalar(op_item, scalar_configs)
        process_int_array(op_item, int_array_configs)

        if 'invoke' in op_item:
            op_item['invoke']['args'] = [
                (
                    args_map[param.strip()]
                    if param.strip() in args_map
                    else param.strip()
                )
                for param in op_item['invoke']['args'].split(',')
            ]
            return
        elif 'composite' in op_item and 'kernel' not in op_item:
            return

        op_item['infer_meta']['param'] = get_param_list_alias(
            op_item['infer_meta']['param'], args_name_map
        )
        op_item['kernel']['param'] = get_param_list_alias(
            op_item['kernel']['param'], args_name_map
        )
        if op_item['kernel']['data_type']:
            op_item['kernel']['data_type']['candidates'] = get_param_list_alias(
                op_item['kernel']['data_type']['candidates'], args_name_map
            )
        if op_item['kernel']['backend']:
            op_item['kernel']['backend']['candidates'] = get_param_list_alias(
                op_item['kernel']['backend']['candidates'], args_name_map
            )
        if op_item['kernel']['layout']:
            op_item['kernel']['layout']['candidates'] = get_param_list_alias(
                op_item['kernel']['layout']['candidates'], args_name_map
            )

    def add_grad_op_compat_name(grad_op_item, args_name_map):
        add_op_param_name(grad_op_item['inputs'], args_name_map)
        add_op_param_name(grad_op_item['outputs'], args_name_map)
        add_op_param_name(grad_op_item['attrs'], args_name_map)
        add_op_param_name(grad_op_item['forward']['inputs'], args_name_map)
        add_op_param_name(grad_op_item['forward']['outputs'], args_name_map)
        add_op_param_name(grad_op_item['forward']['attrs'], args_name_map)
        add_grad_args_name(grad_op_item['inputs'], args_map)
        add_grad_args_name(grad_op_item['outputs'], args_map)

    for op_args in op_fluid_map_list:
        new_op_name, op_name = get_phi_and_fluid_op_name(op_args['op'])
        if new_op_name not in forward_op_dict:
            continue
        forward_op_item = forward_op_dict[new_op_name]
        has_backward = True if forward_op_item['backward'] else False
        if has_backward:
            backward_op_item = backward_op_dict[forward_op_item['backward']]
        if new_op_name != op_name:
            forward_op_item['op_name'] = op_name

        # add complex promote information
        if "complex_promote" in op_args:
            forward_op_item["complex_promote"] = op_args["complex_promote"]
            if has_backward:
                backward_op_item["complex_promote"] = op_args["complex_promote"]
        scalar_configs = None
        int_array_configs = None
        if 'scalar' in op_args:
            scalar_configs = op_args['scalar']
        if 'int_array' in op_args:
            int_array_configs = op_args['int_array']
        if 'extra' in op_args and 'outputs' in op_args['extra']:
            for out_item in forward_op_item['outputs']:
                if out_item['name'] in op_args['extra']['outputs']:
                    out_item['is_extra'] = True
        if 'extra' in op_args and 'inputs' in op_args['extra']:
            for input_item in forward_op_item['inputs']:
                if input_item['name'] in op_args['extra']['inputs']:
                    input_item['is_extra'] = True

        key_set = ['inputs', 'attrs', 'outputs']
        args_map = {}
        for key in key_set:
            if key in op_args:
                args_map.update(op_args[key])
                for args_item in forward_op_item[key]:
                    if args_item['name'] in op_args[key]:
                        if (
                            scalar_configs
                            and args_item['name'] in scalar_configs
                        ):
                            scalar_configs[op_args[key][args_item['name']]] = (
                                scalar_configs[args_item['name']]
                            )
                        if (
                            int_array_configs
                            and args_item['name'] in int_array_configs
                        ):
                            int_array_configs[
                                op_args[key][args_item['name']]
                            ] = int_array_configs[args_item['name']]
                        args_item['fluid_name'] = op_args[key][
                            args_item['name']
                        ]
        update_common_params_name(
            forward_op_item, args_map, scalar_configs, int_array_configs
        )

        if has_backward:
            # update fluid info in backward
            add_grad_op_compat_name(backward_op_item, args_map)
            update_common_params_name(
                backward_op_item, args_map, scalar_configs, int_array_configs
            )

            if 'backward' not in op_args:
                continue

            backward_op_list = op_args['backward'].split(',')
            phi_bw_op_name, bw_op_name = get_phi_and_fluid_op_name(
                backward_op_list[0]
            )
            if (
                forward_op_item["backward_composite"] is not None
                and phi_bw_op_name != bw_op_name
            ):
                forward_op_item["backward_composite"] = bw_op_name
            forward_op_item['backward'] = bw_op_name
            backward_op_item['op_name'] = bw_op_name

            # for double grad
            if len(backward_op_list) > 1:
                (
                    phi_double_grad_op_name,
                    double_grad_op_name,
                ) = get_phi_and_fluid_op_name(backward_op_list[1])
                double_grad_item = backward_op_dict[phi_double_grad_op_name]
                if (
                    backward_op_item["backward_composite"] is not None
                    and phi_double_grad_op_name != double_grad_op_name
                ):
                    backward_op_item["backward_composite"] = double_grad_op_name
                backward_op_item['backward'] = double_grad_op_name
                double_grad_item['op_name'] = double_grad_op_name
                add_grad_op_compat_name(double_grad_item, args_map)
                update_common_params_name(
                    double_grad_item,
                    args_map,
                    scalar_configs,
                    int_array_configs,
                )

                # for triple grad
                if len(backward_op_list) > 2:
                    (
                        phi_triple_grad_op_name,
                        triple_grad_op_name,
                    ) = get_phi_and_fluid_op_name(backward_op_list[2])
                    triple_grad_item = backward_op_dict[phi_triple_grad_op_name]
                    if (
                        double_grad_item["backward_composite"] is not None
                        and phi_triple_grad_op_name != triple_grad_op_name
                    ):
                        double_grad_item["backward_composite"] = (
                            triple_grad_op_name
                        )
                    double_grad_item['backward'] = triple_grad_op_name
                    triple_grad_item['op_name'] = triple_grad_op_name
                    add_grad_op_compat_name(triple_grad_item, args_map)
                    update_common_params_name(
                        triple_grad_item,
                        args_map,
                        scalar_configs,
                        int_array_configs,
                    )


def process_invoke_op(forward_op_dict, backward_op_dict):
    for bw_op in backward_op_dict.values():
        if 'invoke' in bw_op:
            invoke_op = bw_op['invoke']['func']
            args_list = bw_op['invoke']['args']
            args_index = 0
            # backward invoke forward
            if invoke_op in forward_op_dict:
                reuse_op = forward_op_dict[invoke_op]
                bw_op['invoke']['func'] = reuse_op['op_name']
                bw_op['invoke']['inputs'] = []
                bw_op['invoke']['attrs'] = []
                bw_op['invoke']['outputs'] = []
                for input_item in reuse_op['inputs']:
                    bw_op['invoke']['inputs'].append(
                        {
                            'fluid_name': input_item['fluid_name'],
                            'name': input_item['name'],
                            'value': args_list[args_index],
                        }
                    )
                    args_index = args_index + 1
                bw_fluid_attrs_set = [
                    item['fluid_name'] for item in bw_op['attrs']
                ]
                for attr in reuse_op['attrs']:
                    if args_index < len(args_list):
                        attr_value = (
                            f'this->GetAttr("{args_list[args_index]}")'
                            if args_list[args_index] in bw_fluid_attrs_set
                            else args_list[args_index]
                        )
                        bw_op['invoke']['attrs'].append(
                            {
                                'name': attr['name'],
                                'fluid_name': attr['fluid_name'],
                                'value': attr_value,
                            }
                        )
                        args_index = args_index + 1
                    else:
                        break
                for idx, output_item in enumerate(reuse_op['outputs']):
                    bw_op['invoke']['outputs'].append(
                        {
                            'name': output_item['name'],
                            'fluid_name': output_item['fluid_name'],
                            'value': bw_op['outputs'][idx]['fluid_name'],
                        }
                    )


def parse_drop_empty_grad(op_fluid_list: list, bw_op_dict: dict):
    for op_comp_map in op_fluid_list:
        if 'drop_empty_grad' in op_comp_map:
            bw_names = [
                bw_name.split('(')[0].strip()
                for bw_name in op_comp_map['backward'].split(',')
            ]
            new_bw_names = [
                bw_name for bw_name in bw_names if bw_name in bw_op_dict
            ]
            if len(new_bw_names) != 0:
                bws_has_out_grad = False
                for bw_name in bw_names:
                    # static_ops.yaml and ops.yaml use the common op_compat.yaml
                    for out_grad in op_comp_map['drop_empty_grad']:
                        if out_grad in bw_op_dict[bw_name]['output_dict']:
                            bw_op_dict[bw_name]['output_dict'][out_grad][
                                'drop_empty_grad'
                            ] = False
                            bws_has_out_grad = True
                assert (
                    bws_has_out_grad
                ), f'''{bw_names} with {op_comp_map['drop_empty_grad']} is not existed in output_dict '''


def parse_get_expected_kerneltype(
    op_fluid_list: list, fw_op_dict: dict, bw_op_dict: dict
):
    for op_comp_map in op_fluid_list:
        if 'get_expected_kernel_type' in op_comp_map:
            fw_name = op_comp_map['op'].split('(')[0].strip()
            # deal the last underline of function name in op_comp_map['get_expected_kernel_type']
            new_get_expected_kernel_type_func_map = {}
            for key, value in op_comp_map['get_expected_kernel_type'].items():
                new_get_expected_kernel_type_func_map[
                    delete_last_underline(key)
                ] = value
            op_comp_map['get_expected_kernel_type'] = (
                new_get_expected_kernel_type_func_map
            )
            if fw_name in op_comp_map['get_expected_kernel_type']:
                # static_ops.yaml and ops.yaml use the common op_compat.yaml
                if fw_name in fw_op_dict:
                    fw_op_dict[fw_name]["get_expected_kernel_type"] = (
                        op_comp_map['get_expected_kernel_type'][fw_name]
                    )
            if "backward" in op_comp_map:
                bw_names = [
                    bw_name.split('(')[0].strip()
                    for bw_name in op_comp_map['backward'].split(',')
                ]
                for bw_name in bw_names:
                    # static_ops.yaml and ops.yaml use the common op_compat.yaml
                    if (
                        bw_name in bw_op_dict
                        and bw_name in op_comp_map['get_expected_kernel_type']
                    ):
                        bw_op_dict[bw_name]["get_expected_kernel_type"] = (
                            op_comp_map['get_expected_kernel_type'][bw_name]
                        )


def parse_keep_signature(
    op_fluid_list: list, fw_op_dict: dict, bw_op_dict: dict
):
    for op_comp_map in op_fluid_list:
        if 'manual_signature' in op_comp_map:
            for op_name in op_comp_map['manual_signature']:
                op_name_without_last_underline = delete_last_underline(op_name)
                if op_name_without_last_underline in fw_op_dict:
                    fw_op_dict[op_name_without_last_underline][
                        "manual_signature"
                    ] = True
                elif op_name_without_last_underline in bw_op_dict:
                    bw_op_dict[op_name_without_last_underline][
                        "manual_signature"
                    ] = True


def split_ops_list(ops, backward_op_dict, split_num):
    new_ops_list = []
    new_bw_ops_list = []
    list_size = math.ceil(len(ops) / split_num)
    tmp_ops_list = []
    tmp_bw_ops_list = []
    for idx, op in enumerate(ops):
        tmp_ops_list.append(op)
        current_op = op
        while (
            'backward' in current_op
            and current_op['backward'] in backward_op_dict
        ):
            tmp_bw_ops_list.append(backward_op_dict[current_op['backward']])
            current_op = backward_op_dict[current_op['backward']]
        if (idx + 1) % list_size == 0 or idx == len(ops) - 1:
            new_ops_list.append(tmp_ops_list)
            new_bw_ops_list.append(tmp_bw_ops_list)
            tmp_ops_list = []
            tmp_bw_ops_list = []
    return new_ops_list, new_bw_ops_list


def to_phi_and_fluid_op_name_without_underline(op_item):
    '''
    If the op_name ends with '_', delete the last '_'. For an example, 'sgd_' becomes 'sgd
    '''
    names = op_item.split('(')
    if len(names) == 1:
        op_kernel_name = delete_last_underline(names[0].strip())
        return op_kernel_name
    else:
        op_name = delete_last_underline(names[0].strip())
        kernel_name = delete_last_underline(names[1].split(')')[0].strip())
        return op_name + '(' + kernel_name + ')'


def main(
    ops_yaml_path,
    backward_yaml_path,
    op_compat_yaml_path,
    op_version_yaml_path,
    output_op_path,
    output_arg_map_path,
    ops_exclude_yaml_path,
    backward_exclude_yaml_path,
):
    with open(ops_yaml_path, "rt") as f:
        ops = yaml.safe_load(f)
        ops = [restruct_io(op) for op in ops]
    forward_op_dict = to_named_dict(ops, True)
    with open(backward_yaml_path, "rt") as f:
        backward_ops = yaml.safe_load(f)
        backward_ops = [restruct_io(op) for op in backward_ops]
    backward_op_dict = to_named_dict(backward_ops, True)
    with open(op_version_yaml_path, "rt") as f:
        op_versions = yaml.safe_load(f)
    # add op version info into op
    for op_version in op_versions:
        if op_version['op'] in forward_op_dict:
            forward_op_dict[op_version['op']]['version'] = op_version['version']

    with open(op_compat_yaml_path, "rt") as f:
        op_fluid_map_list = yaml.safe_load(f)
        for op_args in op_fluid_map_list:
            op_args["op"] = to_phi_and_fluid_op_name_without_underline(
                op_args["op"]
            )

    # exclude ops in specific yaml file
    if ops_exclude_yaml_path is not None:
        with open(ops_exclude_yaml_path, "rt", encoding='utf-8') as f:
            exclude_ops = yaml.safe_load(f)
            exclude_ops = [op.rstrip('_') for op in exclude_ops]
        for op_name in exclude_ops:
            if op_name in forward_op_dict:
                del forward_op_dict[op_name]
        ops = [op for op in ops if op['name'] not in exclude_ops]
    if backward_exclude_yaml_path is not None:
        with open(backward_exclude_yaml_path, "rt", encoding='utf-8') as f:
            exclude_backward_ops = yaml.safe_load(f)
            exclude_ops = [op.rstrip('_') for op in exclude_ops]
        for op_name in exclude_backward_ops:
            if op_name in backward_op_dict:
                del backward_op_dict[op_name]
        backward_ops = [
            bw_op
            for bw_op in backward_ops
            if bw_op['name'] not in exclude_backward_ops
        ]

    for op in ops:
        op['op_name'] = op['name']
        add_fluid_name(op['inputs'])
        add_fluid_name(op['attrs'])
        add_fluid_name(op['outputs'])
    for bw_op in backward_ops:
        bw_op['op_name'] = bw_op['name']
        add_fluid_name(bw_op['inputs'])
        add_fluid_name(bw_op['attrs'])
        add_fluid_name(bw_op['outputs'])
        add_fluid_name(bw_op['forward']['inputs'])
        add_fluid_name(bw_op['forward']['attrs'])
        add_fluid_name(bw_op['forward']['outputs'])
        for bw_output in bw_op['outputs']:
            bw_output['drop_empty_grad'] = True

    # deal the drop_empty_grad of bw_op by op_compat.yaml
    parse_drop_empty_grad(op_fluid_map_list, backward_op_dict)

    parse_get_expected_kerneltype(
        op_fluid_map_list, forward_op_dict, backward_op_dict
    )

    parse_keep_signature(op_fluid_map_list, forward_op_dict, backward_op_dict)

    add_composite_info(ops, backward_ops, backward_op_dict)

    add_compat_name(op_fluid_map_list, forward_op_dict, backward_op_dict)

    # prepare for invoke case
    process_invoke_op(forward_op_dict, backward_op_dict)

    # fill backward field for an op if another op claims it as forward
    for name, backward_op in backward_op_dict.items():
        forward_name = backward_op["forward"]["name"]
        if forward_name in backward_op_dict:
            forward_op = backward_op_dict[forward_name]
            if forward_op["backward"] is None:
                forward_op["backward"] = name

    op_dict = {}
    op_dict.update(forward_op_dict)
    op_dict.update(backward_op_dict)
    if len(ops) == 0 and len(backward_ops) == 0:
        if os.path.isfile(output_op_path):
            os.remove(output_op_path)
        if os.path.isfile(output_arg_map_path):
            os.remove(output_arg_map_path)
        return
    op_template = env.get_template('op.c.j2')

    backward_fluid_op_dict = {}
    for bw_op in backward_ops:
        backward_fluid_op_dict[bw_op['op_name']] = bw_op
    output_op_files_num = len(output_op_path)
    new_ops_list, new_bw_ops_list = split_ops_list(
        ops, backward_fluid_op_dict, output_op_files_num
    )
    for idx, output_op_file in enumerate(output_op_path):
        with open(output_op_file, "wt") as f:
            msg = op_template.render(
                ops=new_ops_list[idx],
                backward_ops=new_bw_ops_list[idx],
                op_dict=op_dict,
            )
            f.write(msg)

    ks_template = env.get_template('ks.c.j2')
    with open(output_arg_map_path, 'wt') as f:
        msg = ks_template.render(ops=ops, backward_ops=backward_ops)
        f.write(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate operator file from op yaml."
    )
    parser.add_argument(
        '--ops_yaml_path', type=str, help="parsed ops yaml file."
    )
    parser.add_argument(
        '--backward_yaml_path', type=str, help="parsed backward ops yaml file."
    )
    parser.add_argument(
        '--op_compat_yaml_path', type=str, help="ops args compat yaml file."
    )
    parser.add_argument(
        '--op_version_yaml_path', type=str, help="ops version yaml file."
    )
    parser.add_argument(
        "--output_op_path",
        type=str,
        nargs='+',
        help="path to save generated operators.",
    )
    parser.add_argument(
        "--output_arg_map_path",
        type=str,
        help="path to save generated argument mapping functions.",
    )
    parser.add_argument(
        "--ops_exclude_yaml_path",
        type=str,
        default=None,
        help="yaml file to exclude ops",
    )
    parser.add_argument(
        "--backward_exclude_yaml_path",
        type=str,
        default=None,
        help="yaml file to exclude backward ops",
    )

    args = parser.parse_args()
    main(
        args.ops_yaml_path,
        args.backward_yaml_path,
        args.op_compat_yaml_path,
        args.op_version_yaml_path,
        args.output_op_path,
        args.output_arg_map_path,
        args.ops_exclude_yaml_path,
        args.backward_exclude_yaml_path,
    )
