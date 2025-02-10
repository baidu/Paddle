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
import os

from api_gen import NAMESPACE_TEMPLATE, CodeGen

CPP_FILE_TEMPLATE = """
#include <pybind11/pybind11.h>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/pybind/eager_op_function.h"
#include "paddle/fluid/pybind/manual_static_op_function.h"
#include "paddle/fluid/pybind/static_op_function.h"
#include "paddle/phi/core/enforce.h"

{body}

"""

NAMESPACE_INNER_TEMPLATE = """
{function_impl}

static PyMethodDef OpsAPI[] = {{
{ops_api}
{{nullptr, nullptr, 0, nullptr}}
}};

void BindOpsAPI(pybind11::module *module) {{
  if (PyModule_AddFunctions(module->ptr(), OpsAPI) < 0) {{
    PADDLE_THROW(common::errors::Fatal("Add C++ api to core.ops failed!"));
  }}
  if (PyModule_AddFunctions(module->ptr(), ManualOpsAPI) < 0) {{
    PADDLE_THROW(common::errors::Fatal("Add C++ api to core.ops failed!"));
  }}
}}
"""

FUNCTION_IMPL_TEMPLATE = """
static PyObject *{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {{
    VLOG(6) << "Call static_api_{name}";
    return static_api_{name}(self, args, kwargs);
  }} else {{
    VLOG(6) << "Call eager_api_{name}";
    return eager_api_{name}(self, args, kwargs);
  }}
}}"""


STATIC_ONLY_FUNCTION_IMPL_TEMPLATE = """
static PyObject *{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  VLOG(6) << "Call static_api_{name}";
  return static_api_{name}(self, args, kwargs);
}}"""

SPARSE_FUNCTION_IMPL_TEMPLATE = """
static PyObject *sparse_{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  if (egr::Controller::Instance().GetCurrentTracer() == nullptr) {{
    VLOG(6) << "Call static_api_{name}";
    return static_api_{name}{name_suffix}(self, args, kwargs);
  }} else {{
    VLOG(6) << "Call eager_api_{name}";
    return sparse::eager_api_{name}(self, args, kwargs);
  }}
}}"""

SPARSE_STATIC_ONLY_FUNCTION_IMPL_TEMPLATE = """
static PyObject *sparse_{name}(PyObject *self, PyObject *args, PyObject *kwargs) {{
  VLOG(6) << "Call static_api_{name}";
  return static_api_{name}{name_suffix}(self, args, kwargs);
}}"""

OPS_API_TEMPLATE = """
{{"{name}", (PyCFunction)(void (*)(void)){name}, METH_VARARGS | METH_KEYWORDS, "C++ interface function for {name}."}},"""

SPARSE_OPS_API_TEMPLATE = """
{{"sparse_{name}", (PyCFunction)(void (*)(void))sparse_{name}, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_{name}."}},"""

NEED_GEN_STATIC_ONLY_APIS = [
    'c_softmax_with_cross_entropy',
    'c_softmax_with_multi_label_cross_entropy',
    'distributed_fused_lamb_init',
    'distributed_fused_lamb_init_',
    'fetch',
    'fused_embedding_eltwise_layernorm',
    'fused_fc_elementwise_layernorm',
    'fused_multi_transformer_xpu',
    'fused_scale_bias_relu_conv_bn',
    'fused_scale_bias_add_relu',
    'fused_dconv_drelu_dbn',
    'fused_dot_product_attention',
    'fusion_transpose_flatten_concat',
    'skip_layernorm',
    'generate_sequence_xpu',
    'layer_norm_act_xpu',
    'memcpy',
    'memcpy_d2h_multi_io',
    'batch_norm_',
    'multi_encoder_xpu',
    'multihead_matmul',
    'squeeze_excitation_block',
    'yolo_box_xpu',
    'fusion_gru',
    'fusion_seqconv_eltadd_relu',
    'fusion_seqexpand_concat_fc',
    'fused_conv2d_add_act',
    'fusion_repeated_fc_relu',
    'fusion_squared_mat_sub',
    'fused_attention',
    'fused_feedforward',
    'fc',
    'self_dp_attention',
    'get_tensor_from_selected_rows',
    'print',
    'assign_value',
    'share_data_',
    'onednn_to_paddle_layout',
    'lars_momentum_',
    'lrn',
    'multi_gru',
    'matmul_with_flatten',
    'moving_average_abs_max_scale',
    'moving_average_abs_max_scale_',
    'quantize_linear',
    'quantize_linear_',
    'dequantize_linear',
    'dequantize_linear_',
    'coalesce_tensor_',
    'send_v2',
    'recv_v2',
    'sequence_expand',
    'sequence_softmax',
    'qkv_unpack_mha',
    'hash',
    'beam_search_decode',
    'nop',
    'nop_',
    'lod_reset_grad_',
]

NO_NEED_GEN_STATIC_ONLY_APIS = [
    'add_n_',
    'anchor_generator',
    'batch_fc',
    'barrier',
    'c_split',
    'comm_init_all',
    'decayed_adagrad',
    'distributed_push_sparse',
    'distributed_lookup_table',
    'dgc_momentum',
    'dgc',
    'dpsgd',
    'embedding_grad_sparse',
    'faster_tokenizer',
    'ftrl',
    'fused_adam_',
    'fused_batch_norm_act_',
    'fused_bn_add_activation_',
    'fused_elemwise_activation',
    'fused_elemwise_add_activation',
    'fused_scale_bias_relu_conv_bn',
    'fused_scale_bias_add_relu',
    'fused_token_prune',
    'fused_dconv_drelu_dbn',
    'fused_dot_product_attention',
    'fused_elementwise_add',
    'fused_elementwise_div',
    'fused_elementwise_mul',
    'fused_elementwise_sub',
    'fused_embedding_fc_lstm',
    'fused_gate_attention',
    'fused_multi_transformer_int8',
    'fused_seqpool_cvm',
    'fusion_group',
    'fusion_lstm',
    'fusion_seqpool_cvm_concat',
    'nce',
    'legacy_reshape',
    'legacy_reshape_',
    'legacy_reshape_grad',
    'lookup_table',
    'lrn',
    'lod_reset',
    'lod_reset_',
    'max_pool2d_v2',
    'partial_sum',
    'pull_gpups_sparse',
    'pull_gpups_sparse_',
    'push_gpups_sparse',
    'push_gpups_sparse_',
    'random_routing',
    'rnn_',
    'row_conv',
    'seed',
    'shadow_feed',
    'shadow_feed_tensors',
    'sparse_momentum',
    'sync_comm_stream',
    'sync_comm_stream_',
    'soft_relu',
    'match_matrix_tensor',
    'c_scatter',
    "cross_entropy_grad2",
    'push_sparse_v2',
    'push_sparse_v2_',
    'pull_sparse_v2',
    'partial_concat',
    'partial_send',
    'partial_recv',
    'partial_allgather',
    'partial_allgather_',
    'gemm_epilogue',
    'push_dense',
    'legacy_matmul',
    'legacy_matmul_grad',
    'legacy_matmul_double_grad',
    'global_scatter',
    'global_gather',
    'pull_box_sparse',
    'pull_box_sparse_',
    'push_box_sparse',
    'push_box_sparse_',
    'send_and_recv',
    'send_and_recv_',
    'straight_through_estimator',
    "multiply_grad",
    "scale_grad",
    "conv2d_grad",
]


class OpsAPIGen(CodeGen):
    def __init__(self) -> None:
        super().__init__()

    def _need_skip(self, op_info, op_name):
        if op_name.endswith("_grad"):
            if op_name.endswith(("double_grad", "_grad_grad", "triple_grad")):
                return True
            if op_name[:-5] in NO_NEED_GEN_STATIC_ONLY_APIS:
                return True
        if op_name.endswith("_grad_"):
            return True

        return (
            super()._need_skip(op_info, op_name)
            or op_name.endswith('xpu')
            or op_name in NO_NEED_GEN_STATIC_ONLY_APIS
        )

    def _gen_one_function_impl(self, name):
        if name.endswith('_grad'):
            fwd_name = name[:-5]
            if fwd_name in NEED_GEN_STATIC_ONLY_APIS:
                return STATIC_ONLY_FUNCTION_IMPL_TEMPLATE.format(name=name)
            else:
                return FUNCTION_IMPL_TEMPLATE.format(name=name)
        else:
            if name in NEED_GEN_STATIC_ONLY_APIS:
                return STATIC_ONLY_FUNCTION_IMPL_TEMPLATE.format(name=name)
            else:
                return FUNCTION_IMPL_TEMPLATE.format(name=name)

    def _gen_sparse_one_function_impl(self, name, name_suffix):
        return SPARSE_FUNCTION_IMPL_TEMPLATE.format(
            name=name, name_suffix=name_suffix
        )

    def _gen_one_ops_api(self, name):
        return OPS_API_TEMPLATE.format(name=name)

    def _gen_sparse_one_ops_api(self, name):
        return SPARSE_OPS_API_TEMPLATE.format(name=name)

    def gen_cpp_file(
        self, op_yaml_files, op_compat_yaml_file, namespaces, cpp_file_path
    ):
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)
        op_info_items = self._parse_yaml(op_yaml_files, op_compat_yaml_file)
        function_impl_str = ''
        ops_api_str = ''
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                if self._need_skip(op_info, op_name):
                    continue
                if op_info.is_sparse_op:
                    op_name_suffix = "sp_" if op_name[-1] == "_" else "_sp"
                    function_impl_str += self._gen_sparse_one_function_impl(
                        op_name, op_name_suffix
                    )
                    ops_api_str += self._gen_sparse_one_ops_api(op_name)
                else:
                    function_impl_str += self._gen_one_function_impl(op_name)
                    ops_api_str += self._gen_one_ops_api(op_name)

        inner_body = NAMESPACE_INNER_TEMPLATE.format(
            function_impl=function_impl_str, ops_api=ops_api_str
        )

        body = inner_body
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
    parser.add_argument('--ops_api_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
    ops_api_file = args.ops_api_file

    code_gen = OpsAPIGen()
    code_gen.gen_cpp_file(
        op_yaml_files, op_compat_yaml_file, namespaces, ops_api_file
    )
