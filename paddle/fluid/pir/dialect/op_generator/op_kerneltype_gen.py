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

OP_GET_KERNEL_TYPE_FOR_VAR_TEMPLATE = """
phi::DataType {op_name}::GetKernelTypeForVar(
    const std::string& var_name,
    const phi::DataType& tensor_dtype,
    const phi::DataType& expected_kernel_dtype) {{
  VLOG(4) << "Get KernelType for Var of op: {op_name}";
  {data_transform_check}{complex_promote_check}
  return expected_kernel_dtype;
}}
"""

OP_DATA_TRANSFORM_CHECK_TEMPLATE = """
{skip_trans}{support_trans}
"""

OP_SKIP_TRANSFORM_CHECK_TEMPLATE = """
  // deal skip data transform
  if ({skip_transform_check}){{
    return expected_kernel_dtype;
  }}
"""

OP_SUPPORT_TRANSFORM_CHECK_TEMPLATE = """
  // deal support data transform
  VLOG(8) << "SUPPORT_TRANSFORM";
  if ({support_transform_check}){{
    return tensor_dtype;
  }}
"""

OP_COMPLEX_PROMOTE_CHECK_TEMPLATE = """
  // deal complex_promote
  if (phi::IsComplexType(expected_kernel_dtype)) {{
    // only promote inputs’s types when contains complex input
    return tensor_dtype;
  }}
"""


def get_data_transform_check_str(op_data_transform_map):
    skip_trans_str = ""
    support_trans_str = ""
    if op_data_transform_map is not None:
        args = None
        if "skip_transform" in op_data_transform_map:
            args = op_data_transform_map["skip_transform"]
            if args is not None:
                if_cond_args = []
                for skip_arg in args:
                    if_cond_args.append(f'var_name == "{skip_arg}"')
                skip_trans_str = OP_SKIP_TRANSFORM_CHECK_TEMPLATE.format(
                    skip_transform_check=' || '.join(if_cond_args)
                )
        if "support_trans_dtype" in op_data_transform_map:
            args = op_data_transform_map["support_trans_dtype"]
            # TODO:(chenxi67) complete SUPPORT logic
            if args is not None:
                if_cond_args = []
                for support_arg in args:
                    if_cond_args.append(f'var_name == "{support_arg}"')
            if args is not None:
                support_trans_str = OP_SUPPORT_TRANSFORM_CHECK_TEMPLATE.format(
                    support_transform_check=' || '.join(if_cond_args)
                    # support_dtype_name=args.join(", ")
                )

    return OP_DATA_TRANSFORM_CHECK_TEMPLATE.format(
        skip_trans=skip_trans_str,
        support_trans=support_trans_str,
    )


def get_complex_promote_check_str(op_compat_item):
    complex_promote_check_str = ""
    if (
        op_compat_item is not None
        and "complex_promote" in op_compat_item
        and op_compat_item["complex_promote"] is not None
    ):
        complex_promote_check_str = OP_COMPLEX_PROMOTE_CHECK_TEMPLATE
    return complex_promote_check_str


def gen_kernel_type_for_var_str(
    op_class_name, op_data_transform_map, op_kernel_map, op_compat_item
):
    complex_promote_check_str = get_complex_promote_check_str(op_compat_item)
    data_transform_check_str = get_data_transform_check_str(
        op_data_transform_map
    )

    return OP_GET_KERNEL_TYPE_FOR_VAR_TEMPLATE.format(
        op_name=op_class_name,
        data_transform_check=data_transform_check_str,
        complex_promote_check=complex_promote_check_str,
    )
