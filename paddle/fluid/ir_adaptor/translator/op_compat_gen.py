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

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

file_loader = FileSystemLoader(Path(__file__).parent)
env = Environment(
    loader=file_loader,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
    extensions=['jinja2.ext.do'],
)


def OpNameNormalizerInitialization(
    op_compat_yaml_file: str = "",
    sparse_op_yaml_file: str = "",
    output_source_file: str = "",
) -> None:
    def to_phi_and_fluid_op_name(op_item):
        # Template: - op : phi_name (fluid_name)
        names = op_item.split('(')
        if len(names) == 1:
            phi_fluid_name = names[0].strip()
            return phi_fluid_name, phi_fluid_name
        else:
            phi_name = names[0].strip()
            fluid_name = names[1].split(')')[0].strip()
            return phi_name, fluid_name

    with open(op_compat_yaml_file, "r") as f:
        op_compat_infos = yaml.safe_load(f)
    op_name_mappings: dict[str, str] = {}
    op_arg_name_mappings: dict[str, dict[str, str]] = {}
    op_mutable_attributes: dict[str, set[str]] = {}
    op_mutable_attribute_infos: dict[str, dict[str, list[str]]] = {}

    for op_compat_item in op_compat_infos:

        def insert_new_mappings(op_name_str: str) -> str:
            normalized_name, legacy_name = to_phi_and_fluid_op_name(op_name_str)
            if normalized_name == legacy_name:
                return normalized_name, legacy_name
            op_name_mappings[legacy_name] = normalized_name
            return normalized_name, legacy_name

        def insert_new_arg_mappings(op_name: str, arg_mapping: dict[str, str]):
            if op_name is None:
                return
            if op_name not in op_arg_name_mappings:
                op_arg_name_mappings[op_name] = {}
            op_arg_name_mappings[op_name].update(arg_mapping)

        def insert_new_mutable_attributes(
            op_name: str, mutable_attribute_infos: dict[str, dict[str, str]]
        ):
            if op_name not in op_mutable_attributes:
                op_mutable_attributes[op_name] = set()
            if op_name not in op_mutable_attribute_infos:
                op_mutable_attribute_infos[op_name] = {}
            for (
                attribute_name,
                mutable_attribute_info,
            ) in mutable_attribute_infos.items():
                op_mutable_attributes[op_name].add(attribute_name)
                op_mutable_attribute_infos[op_name][attribute_name] = []
                for k, v in mutable_attribute_info.items():
                    if k == 'tensor_name' or k == 'tensors_name':
                        op_mutable_attribute_infos[op_name][
                            attribute_name
                        ].insert(0, v)

        _, legacy_name = insert_new_mappings(op_compat_item["op"])
        dygraph_backward_op_names = []
        if "backward" in op_compat_item:
            backward_op_name_mapping_paris = op_compat_item["backward"].split(
                ","
            )
            for pair in backward_op_name_mapping_paris:
                _, dygraph_backward_op_name = insert_new_mappings(pair)
                dygraph_backward_op_names.append(dygraph_backward_op_name)

        if "inputs" in op_compat_item:
            insert_new_arg_mappings(legacy_name, op_compat_item["inputs"])
            for backward_op in dygraph_backward_op_names:
                insert_new_arg_mappings(backward_op, op_compat_item["inputs"])

        if "attrs" in op_compat_item:
            insert_new_arg_mappings(legacy_name, op_compat_item["attrs"])
            for backward_op in dygraph_backward_op_names:
                insert_new_arg_mappings(backward_op, op_compat_item["attrs"])
        if "outputs" in op_compat_item:
            insert_new_arg_mappings(legacy_name, op_compat_item["outputs"])
            for backward_op in dygraph_backward_op_names:
                insert_new_arg_mappings(backward_op, op_compat_item["outputs"])

        if "int_array" in op_compat_item:
            insert_new_mutable_attributes(
                legacy_name, op_compat_item["int_array"]
            )
            for backward_op in dygraph_backward_op_names:
                insert_new_mutable_attributes(
                    backward_op, op_compat_item["int_array"]
                )

        if "scalar" in op_compat_item:
            insert_new_mutable_attributes(legacy_name, op_compat_item["scalar"])
            for backward_op in dygraph_backward_op_names:
                insert_new_mutable_attributes(
                    backward_op, op_compat_item["scalar"]
                )

    # special mapping list
    op_name_mappings["deformable_conv_v1"] = "deformable_conv"
    op_name_mappings["deformable_conv_v1_grad"] = "deformable_conv_grad"
    op_arg_name_mappings["deformable_conv_v1"] = {
        "x": "Input",
        "offset": "Offset",
        "filter": "Filter",
        "mask": "Mask",
        "out": "Output",
    }

    op_name_mappings["lookup_table"] = "embedding"
    op_arg_name_mappings["lookup_table"] = {
        "x": "Ids",
        "weight": "W",
        "out": "Out",
    }

    op_arg_name_mappings["set_value_grad"]["values_grad"] = "ValueTensor@GRAD"
    op_arg_name_mappings["fetch"] = {"x": "X"}
    op_arg_name_mappings["elementwise_add_grad_grad"] = {
        "y": "Y",
        "grad_out": "DOut",
        "grad_x_grad": "DDX",
        "grad_y_grad": "DDY",
        "grad_out_grad": "DDOut",
    }
    op_arg_name_mappings["batch_norm_grad_grad"] = {
        "scale_grad": "DScale",
        "x_grad": "DX",
        "grad_out_grad": "DDY",
        "out_mean": "OutMean",
        "out_variance": "OutVariance",
        "grad_x_grad": "DDX",
        "grad_scale_grad": "DDScale",
        "grad_bias_grad": "DDBias",
        "grad_out": "DY",
    }
    op_arg_name_mappings["matmul"] = {
        "x": "X",
        "y": "Y",
        "out": "Out",
        "transpose_x": "transpose_X",
        "transpose_y": "transpose_Y",
    }

    op_arg_name_mappings["matrix_rank"] = {
        "x": "X",
        "atol_tensor": "TolTensor",
        "out": "Out",
    }
    op_arg_name_mappings['fused_softmax_mask_grad'].update({"out": "Softmax"})
    op_arg_name_mappings['push_sparse_v2'].update(
        {"out_grad_in": "Out@GRAD", "out_grad_out": "Out@GRAD"}
    )
    op_arg_name_mappings['push_box_sparse'].update(
        {"out_grad_in": "Out@GRAD", "out_grad_out": "Out@GRAD"}
    )
    op_arg_name_mappings['push_gpups_sparse'].update(
        {"out_grad": "Out@GRAD", "out_grad_grad": "Out@GRAD"}
    )

    sparse_op_yaml_files = sparse_op_yaml_file.split(",")
    for yaml_file in sparse_op_yaml_files:
        with open(yaml_file, 'r') as f:
            sparse_ops_items = yaml.safe_load(f)
            for sparse_op in sparse_ops_items:
                if yaml_file.endswith("sparse_ops.yaml"):
                    op_name = sparse_op['op']
                else:
                    op_name = sparse_op['backward_op']
                if op_name[-1] == "_":
                    op_name_mappings["sparse_" + op_name[:-1]] = op_name + 'sp_'
                else:
                    op_name_mappings["sparse_" + op_name] = op_name + '_sp'

    op_mutable_attributes["reduce_mean"] = set()
    op_mutable_attributes["reduce_mean"].add("axis")
    op_mutable_attribute_infos["reduce_mean"] = {}
    op_mutable_attribute_infos["reduce_mean"]["axis"] = []

    op_mutable_attributes["reduce_mean_grad"] = set()
    op_mutable_attributes["reduce_mean_grad"].add("axis")
    op_mutable_attribute_infos["reduce_mean_grad"] = {}
    op_mutable_attribute_infos["reduce_mean_grad"]["axis"] = []

    op_name_normalizer_template = env.get_template("op_compat_info.cc.j2")
    with open(output_source_file, 'wt') as f:
        op_compat_definition = op_name_normalizer_template.render(
            op_name_pairs=op_name_mappings,
            op_arg_name_pairs=op_arg_name_mappings,
            op_mutable_attributes=op_mutable_attributes,
            op_mutable_attribute_infos=op_mutable_attribute_infos,
        )
        f.write(op_compat_definition)


# =====================================
# Script parameter parsing
# =====================================
def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate OP Compatible info Files By Yaml'
    )
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--sparse_op_yaml_file', type=str)
    parser.add_argument('--output_source_file', type=str)
    return parser.parse_args()


# =====================================
# Main
# =====================================
if __name__ == "__main__":
    # parse arguments
    args = ParseArguments()
    OpNameNormalizerInitialization(**vars(args))
