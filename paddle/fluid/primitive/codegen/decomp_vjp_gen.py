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
import hashlib
import pathlib
import sys

import jinja2
import yaml

# fmt: off
# import from paddle/fluid/operators/generator
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'operators/generator')
)
import filters as op_gen_filters
import tests_utils as op_gen_tests
from parse_utils import to_named_dict

# import from paddle/fluid/pir/dialect/op_generator/api_gen.py
sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[2] / 'pir/dialect/op_generator')
)

from decomp_interface_gen_op_list import (
    decomp_vjp_interface_implementation_gen_op_list,
)
from gen_utils import attr_types_map, to_pascal_case
from type_mapping import output_type_map

# fmt: on


VJPS_BLACK_LIST = [
    'reshape_grad',
    'add_n_grad',
    'fused_attention_grad',
]

BACKENDS_BLACK_LIST = [
    'accuracy_check',
    'copy_to',
    'add_n_grad',
    "allclose",
    "isclose",
    "p_send",
    "assert",
    "embedding_sparse_grad",
    "embedding_grad",
    "full",
    "partial_send",
    "push_dense",
    "comm_init_all",
]

# which is prim vjp
# op only has backward decomp_vjp rules
PRIM_VJP = [
    'abs_grad',
    'add_grad',
    'amax_grad',
    'amin_grad',
    'angle_grad',
    'argsort_grad',
    'assign_grad',
    'atan_grad',
    'atan2_grad',
    'cast_grad',
    'ceil_grad',
    'concat_grad',
    'cos_grad',
    'cumprod_grad',
    'cumsum_grad',
    'divide_grad',
    'dot_grad',
    'elementwise_pow_grad',
    'erf_grad',
    'exp_grad',
    'expm1_grad',
    'expand_grad',
    'floor_grad',
    'fmax_grad',
    'fmin_grad',
    'gather_grad',
    'gather_nd_grad',
    'kron_grad',
    'kthvalue_grad',
    'log_grad',
    'logcumsumexp_grad',
    'logsumexp_grad',
    'masked_select_grad',
    'matmul_grad',
    'max_grad',
    'maximum_grad',
    'minimum_grad',
    'multiply_grad',
    'pad_grad',
    'pow_grad',
    'prod_grad',
    'put_along_axis_grad',
    'reduce_as_grad',
    'reshape_grad',
    'roll_grad',
    'rsqrt_grad',
    'scale_grad',
    'scatter_grad',
    'scatter_nd_add_grad',
    'sigmoid_grad',
    'sin_grad',
    'slice_grad',
    'squeeze_grad',
    'split_grad',
    'sqrt_grad',
    'square_grad',
    'subtract_grad',
    'sum_grad',
    'take_along_axis_grad',
    'tanh_grad',
    'tile_grad',
    'topk_grad',
    'transpose_grad',
    'trunc_grad',
    'unsqueeze_grad',
    'where_grad',
]

# which op is custom_vjp?
# op has forward decomp_rules and backward decomp_vjp rules
CUSTOM_VJP = [
    'batch_norm_grad',
    'bce_loss_grad',
    'dropout_grad',
    'gelu_grad',
    'group_norm_grad',
    'hardsigmoid_grad',
    'hardswish_grad',
    'instance_norm_grad',
    'layer_norm_grad',
    'leaky_relu_grad',
    'mean_grad',
    'relu_grad',
    'relu6_grad',
    'silu_grad',
    'softmax_grad',
    'softsign_grad',
    'stack_grad',
    'swish_grad',
    'swiglu_grad',
    'p_norm_grad',
]  # custom vjp list of composite op

VJP_COMPS = PRIM_VJP + CUSTOM_VJP


def load(path: pathlib.Path):
    """Load config from yaml file.

    Args:
        path (pathlib.Path): The path of yaml config.

    Returns:
        dict: The config info.

    """
    with open(path, 'rt') as f:
        return yaml.safe_load(f)


def render(src_dir: pathlib.Path, dst_dir: pathlib.Path, *args, **kwargs):
    """Render and save Jinja2 templates to the destination directory.

    Args:
        src_dir (pathlib.Path): The source directory containing Jinja2 templates.
        dst_dir (pathlib.Path): The destination directory to save rendered files.
        *args: Additional positional arguments passed to the `render` function.
        **kwargs: Additional keyword arguments passed to the `render` function.

    Returns:
        None
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(src_dir),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
        extensions=['jinja2.ext.do'],
    )
    env.filters.update(
        {
            'to_paddle_attr_type': op_gen_filters.to_paddle_attr_type,
            'to_paddle_input_type': op_gen_filters.to_paddle_input_type,
            'to_paddle_output_type': op_gen_filters.to_paddle_output_type,
            'trip_intermediate': op_gen_filters.filter_intermediate,
        }
    )
    env.tests.update(
        {
            'scalar': op_gen_tests.is_scalar,
            'intarray': op_gen_tests.is_intarray,
            'datatype': op_gen_tests.is_datatype,
            'exist_mutable_attribute': op_gen_tests.exist_mutable_attribute,
            'mutable_attribute': op_gen_tests.is_mutable_attribute,
            'only_composite_op': op_gen_tests.is_only_composite_op,
        }
    )
    for tpl in env.list_templates(
        filter_func=lambda name: ".h" in name or ".cc" in name
    ):
        save(
            env.get_template(tpl).render(*args, **kwargs),
            dst_dir / tpl.rstrip('.j2'),
        )


def render_decomp_vjp(
    src_dir: pathlib.Path, dst_dir: pathlib.Path, *args, **kwargs
):
    """Render and save Jinja2 templates to the destination directory.

    Args:
        src_dir (pathlib.Path): The source directory containing Jinja2 templates.
        dst_dir (pathlib.Path): The destination directory to save rendered files.
        *args: Additional positional arguments passed to the `render` function.
        **kwargs: Additional keyword arguments passed to the `render` function.

    Returns:
        None
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(src_dir),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
        extensions=['jinja2.ext.do'],
    )
    env.filters.update(
        {
            'to_paddle_attr_type': op_gen_filters.to_paddle_attr_type,
            'to_paddle_input_type': op_gen_filters.to_paddle_input_type,
            'to_paddle_output_type': op_gen_filters.to_paddle_output_type,
            'trip_intermediate': op_gen_filters.filter_intermediate,
        }
    )
    env.tests.update(
        {
            'scalar': op_gen_tests.is_scalar,
            'intarray': op_gen_tests.is_intarray,
            'datatype': op_gen_tests.is_datatype,
            'exist_mutable_attribute': op_gen_tests.exist_mutable_attribute,
            'mutable_attribute': op_gen_tests.is_mutable_attribute,
            'only_composite_op': op_gen_tests.is_only_composite_op,
        }
    )

    decomp_temp = "decomp/generated_decomp_vjp.j2"
    save(
        env.get_template(decomp_temp).render(*args, **kwargs),
        pathlib.Path(dst_dir),
    )


def save(content: str, path: pathlib.Path):
    """Saves the given string contents to a file in the specified path.

    Args:
        content (str): The string content that needs to be saved.
        path (pathlib.Path): The path to save the file, a Pathlib path object

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    dst_content = ''
    if path.is_file():
        with open(path, 'r') as f:
            dst_content = f.read()

    if (
        hashlib.md5(content.encode("UTF-8")).hexdigest()
        != hashlib.md5(dst_content.encode("UTF-8")).hexdigest()
    ):
        with open(path, 'w') as f:
            f.write(content)
            print(f"Generate source file {path}")


def get_inplace_api(apis):
    inplace_apis = []
    for api in apis:
        if (
            'inplace' in api
            and api['inplace'] is not None
            and not api['name'].endswith('_')
        ):
            inplace_api = api.copy()
            inplace_api['name'] = api['name'] + '_'
            inplace_apis.append(inplace_api)
    return inplace_apis


def filter_compat_info(items):
    for item in items:
        item['op'] = item['op'].split('(')[0].strip()
        if 'backward' in item:
            item_backwards = item['backward'].split(',')
            for idx, item_backward in enumerate(item_backwards):
                item_backward = item_backward.split('(')[0].strip()
                item_backwards[idx] = item_backward
            item['backward'] = (
                ','.join(item_backwards)
                if len(item_backwards) > 0
                else item_backwards[0]
            )


def extend_compat_info(apis, compats):
    for api in apis:
        if api['name'].endswith('sp') or api['name'].endswith('sp_'):
            continue
        attrs = api["attrs"]
        for attr in attrs:
            if op_gen_tests.is_scalar(
                attr['typename']
            ) or op_gen_tests.is_intarray(attr['typename']):
                attr["support_tensor"] = False
    apis_dict = to_named_dict(apis)
    for compat_item in compats:
        fwd_op_name = compat_item["op"]
        if fwd_op_name not in apis_dict:
            continue
        fwd_api = apis_dict[fwd_op_name]
        backward_op_names = []
        while fwd_op_name is not None and fwd_op_name in apis_dict:
            backward_op_names.append(apis_dict[fwd_op_name]['backward'])
            fwd_op_name = apis_dict[fwd_op_name]['backward']
        backward_apis = []
        for backward_op_name in backward_op_names:
            if backward_op_name in apis_dict:
                backward_apis.append(apis_dict[backward_op_name])
        support_tensor_attrs_names = []
        compat_attrs_data_type = {}
        if 'scalar' in compat_item and compat_item['op'] != "pow":
            for attr_name, attr_info in compat_item['scalar'].items():
                if (
                    'support_tensor' in attr_info
                    and attr_info['support_tensor'] is True
                    or 'tensor_name' in attr_info
                ):
                    support_tensor_attrs_names.append(attr_name)
                if 'data_type' in attr_info:
                    compat_attrs_data_type.update(
                        {attr_name: attr_info['data_type']}
                    )
        if 'int_array' in compat_item:
            for attr_name, attr_info in compat_item['int_array'].items():
                if (
                    'support_tensor' in attr_info
                    and attr_info['support_tensor'] is True
                    or 'tensor_name' in attr_info
                    or 'tensors_name' in attr_info
                ):
                    support_tensor_attrs_names.append(attr_name)
        if len(support_tensor_attrs_names) > 0:
            for api in [fwd_api, *backward_apis]:
                attrs = api["attrs"]
                for attr in attrs:
                    if attr['name'] in support_tensor_attrs_names:
                        attr['support_tensor'] = True
        for api in [fwd_api, *backward_apis]:
            attrs = api["attrs"]
            for attr in attrs:
                if attr['name'] in compat_attrs_data_type:
                    attr['data_type'] = compat_attrs_data_type[attr['name']]
    return apis


def process_backward_invoke_info(apis):
    apis_dict = to_named_dict(apis)
    for api in apis:
        if api['is_fwd']:
            continue
        if 'invoke' in api and api['invoke']['func'] in apis_dict:
            args = api['invoke']['args'].split(',')
            args = [arg.strip() for arg in args]
            attrs_dict = to_named_dict(api['attrs'])
            inputs_dict = to_named_dict(api['inputs'])
            arg_inputs = []
            arg_attrs = []
            for arg in args:
                if arg in inputs_dict:
                    arg_inputs.append(arg)
                elif arg in attrs_dict and attrs_dict[arg].get(
                    "support_tensor", False
                ):
                    arg_inputs.append(arg + '_')
                else:
                    arg_attrs.append(arg)
            args = arg_inputs + arg_attrs
            api['invoke']['args'] = ', '.join(args)


def process_optional_inplace_output_info(apis):
    for api in apis:
        inputs_dict = to_named_dict(api['inputs'])
        for output in api['outputs']:
            if not api['is_fwd']:
                return
            else:
                if (
                    api.get("inplace", None)
                    and output['name'] in api['inplace']
                    and inputs_dict[api['inplace'][output['name']]]['optional']
                ):
                    output['optional'] = True
                else:
                    output['optional'] = False


def update_apis(op_yaml_items, update_yaml_file):
    with open(update_yaml_file, "r") as f:
        update_apis = yaml.safe_load(f)
    for i in range(len(op_yaml_items)):
        for update_api in update_apis:
            if op_yaml_items[i]['name'] == update_api['name']:
                op_yaml_items[i] = update_api
                break


def gen(
    prim_path: pathlib.Path,
    fwd_path: pathlib.Path,
    rev_path: pathlib.Path,
    compat_path: pathlib.Path,
    fwd_pd_op_path: pathlib.Path,
    update_fwd_pd_op_path: pathlib.Path,
    rev_pd_op_path: pathlib.Path,
    fused_op_path: pathlib.Path,
    fused_rev_path: pathlib.Path,
    sparse_op_path: pathlib.Path,
    sparse_rev_op_path: pathlib.Path,
    templates_dir: pathlib.Path,
    destination_dir: pathlib.Path,
    decomp_vjp_destination_dir: pathlib.Path,
):
    """The `gen` load jinja2 templates and relative config info, use jinja2
    templating engine to generate c++ code, and save the code into destination.

    Args:
        prim_path (pathlib.Path): The YAML file path of the primitive API.
        fwd_path (pathlib.Path):  The YAML file path of the forward API.
        rev_path (pathlib.Path): The YAML file path of the backward API.
        compat_path: (pathlib.Path): The YAML file path of the ops compat.
        fwd_pd_op_path (pathlib.Path): The YAML file path of the ir forward API.
        update_fwd_pd_op_path (pathlib.Path): The YAML file path of the ir update_ops.
        rev_pd_op_path (pathlib.Path): The YAML file path of the ir backward API.
        fused_op_path (pathlib.Path): The YAML file path of the fused API.
        fused_rev_path (pathlib.Path): The YAML file path of the fused backward API.
        sparse_op_path (pathlib.Path): The YAML file path of the sparse API.
        sparse_rev_op_path (pathlib.Path): The YAML file path of the sparse backward API.
        templates_dir (pathlib.Path): The directory of the templates.
        destination_dir (pathlib.Path): The Directory of the generated file.

    Returns:
        None
    """
    (
        prims,
        fwds,
        revs,
        compats,
        ir_fwds,
        ir_revs,
        ir_update_fwds,
        fused_fwds,
        fused_revs,
        sparse_fwds,
        sparse_revs,
    ) = (
        load(prim_path),
        load(fwd_path),
        load(rev_path),
        load(compat_path),
        load(fwd_pd_op_path),
        load(rev_pd_op_path),
        load(update_fwd_pd_op_path),
        load(fused_op_path),
        load(fused_rev_path),
        load(sparse_op_path),
        load(sparse_rev_op_path),
    )
    filter_compat_info(compats)
    for sparse_op in sparse_fwds:
        if sparse_op['name'].endswith("_"):
            sparse_op['name'] += 'sp_'
            if sparse_op['backward'] is not None:
                sparse_op['backward'] += '_sp'
        else:
            sparse_op['name'] += '_sp'
            if sparse_op['backward'] is not None:
                sparse_op['backward'] += '_sp'
    fwd_apis = fwds + ir_fwds + ir_update_fwds + fused_fwds + sparse_fwds

    for sparse_op in sparse_revs:
        sparse_op['name'] += '_sp'
        if sparse_op['forward']['name'].endswith("_"):
            sparse_op['forward']['name'] += 'sp_'
            if sparse_op.get('invoke') is not None:
                sparse_op['invoke']['func'] += 'sp_'
        else:
            sparse_op['forward']['name'] += '_sp'
            if sparse_op.get('invoke') is not None:
                sparse_op['invoke']['func'] += '_sp'
    apis = [{**api, **{'is_fwd': True}} for api in fwd_apis]
    apis = apis + [
        {**api, **{'is_fwd': False}}
        for api in revs + ir_revs + fused_revs + sparse_revs
    ]
    apis = [
        (
            {**api, **{'is_prim': True}}
            if api['name'] in prims
            else {**api, **{'is_prim': False}}
        )
        for api in apis
    ]

    apis = extend_compat_info(apis, compats)
    apis = apis + get_inplace_api(apis)
    process_backward_invoke_info(apis)
    process_optional_inplace_output_info(apis)

    apis = [
        {**api, **{'class_name': to_pascal_case(api["name"]) + "Op"}}
        for api in apis
    ]

    for item in apis:
        for attr_item in item["attrs"]:
            if attr_item["typename"] not in attr_types_map.keys():
                raise TypeError
            attr_item["mapped_type"] = attr_types_map[attr_item["typename"]]
        for out_item in item["outputs"]:
            if out_item["typename"] not in output_type_map.keys():
                name = out_item["typename"]
                raise TypeError(f"err type {name}")
            if out_item["optional"]:
                out_item["mapped_type"] = (
                    "paddle::optional<"
                    + output_type_map[out_item["typename"]]
                    + ">"
                )
            else:
                out_item["mapped_type"] = output_type_map[out_item["typename"]]

    render(
        templates_dir,
        destination_dir,
        apis=apis,
        backend_black_list=BACKENDS_BLACK_LIST,
        vjp_black_list=VJPS_BLACK_LIST,
        vjp_comp_white_list=VJP_COMPS,
    )
    render_decomp_vjp(
        templates_dir,
        decomp_vjp_destination_dir,
        apis=apis,
        backend_black_list=BACKENDS_BLACK_LIST,
        vjp_black_list=VJPS_BLACK_LIST,
        vjp_comp_white_list=VJP_COMPS,
        decomp_vjp_white_list=decomp_vjp_interface_implementation_gen_op_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Static Primitive API'
    )
    parser.add_argument(
        '--prim_path',
        type=str,
        help='The primitive API yaml file.',
    )
    parser.add_argument(
        '--fwd_path', type=str, help='The parsed ops yaml file.'
    )
    parser.add_argument(
        '--rev_path', type=str, help='The parsed ops yaml file.'
    )
    parser.add_argument(
        '--compat_path',
        type=str,
        help='The parsed ops compat yaml file.',
    )
    parser.add_argument(
        '--fwd_pd_op_path',
        type=str,
        help='The ir forward ops parsed  yaml file.',
    )
    parser.add_argument(
        '--update_fwd_pd_op_path',
        type=str,
        help='The ir update forward ops parsed  yaml file.',
    )
    parser.add_argument(
        '--rev_pd_op_path',
        type=str,
        help='The ir backward ops parsed  yaml file.',
    )
    parser.add_argument(
        '--fused_op_path',
        type=str,
        help='The parsed fused forward ops yaml file.',
    )
    parser.add_argument(
        '--fused_rev_op_path',
        type=str,
        help='The parsed fused backward ops yaml file.',
    )
    parser.add_argument(
        '--sparse_op_path',
        type=str,
        help='The parsed sparse forward ops yaml file.',
    )
    parser.add_argument(
        '--sparse_rev_op_path',
        type=str,
        help='The parsed sparse backward ops yaml file.',
    )
    parser.add_argument(
        '--templates_dir',
        type=str,
        help='JinJa2 templates base directory.',
    )
    parser.add_argument(
        '--destination_dir',
        type=str,
        help='Destination base directory for generated file.',
    )
    parser.add_argument(
        '--decomp_vjp_destination_dir',
        type=str,
        help='Destination base directory for generated file.',
    )
    args = parser.parse_args()

    gen(
        pathlib.Path(args.prim_path),
        pathlib.Path(args.fwd_path),
        pathlib.Path(args.rev_path),
        pathlib.Path(args.compat_path),
        pathlib.Path(args.fwd_pd_op_path),
        pathlib.Path(args.update_fwd_pd_op_path),
        pathlib.Path(args.rev_pd_op_path),
        pathlib.Path(args.fused_op_path),
        pathlib.Path(args.fused_rev_op_path),
        pathlib.Path(args.sparse_op_path),
        pathlib.Path(args.sparse_rev_op_path),
        pathlib.Path(args.templates_dir),
        pathlib.Path(args.destination_dir),
        pathlib.Path(args.decomp_vjp_destination_dir),
    )
