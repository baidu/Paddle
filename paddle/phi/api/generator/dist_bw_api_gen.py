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

import dist_api_gen
import yaml
from backward_api_gen import BackwardAPI
from dist_api_gen import DistForwardAPI

######################
# Code Gen Templates #
######################

MAIN_DIST_BRANCH_TEMPLATE = """
  // Auto Parallel condition
  if (run_auto_parallel) {{
    // 1. InferSpmd (Infer DistAttr of Inputs&Outputs){}
    // 2. Create Temporary Output & Prepare Dist and Dense Output{}
    // 3. Infer DistTensor's Global Shape{}\n
    // 4. Set Output Dist Attr For Default Impl{}\n
    if (rank_is_in_current_mesh) {{
      // 5. Select Kernel{}
      // 6. Reshard Input{}\n
      // 7. PrepareData (DataTransform & Prepare Dense Input){}
      // 8. RecordOpInfoSupplement{}
      // 9. Infer Local DenseTensor Meta{}
      // 10. DenseTensor Kernel Call{}
      // 11. Fallback{}
    }}
    // 12. Reshard Kernel Output to API output{}\n
    // 13. Return
    {}
  }}
"""

# 1. Create API Outputs
SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD = """
    auto dist_out = SetKernelDistOutput({});
    auto dense_out = dist_out->unsafe_mutable_value();
"""
SINGLE_OUT_CREATION_TEMPLATE_WITH_SPMD = """
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out =
        CreateKernelDistOutput({}, !rank_is_in_current_mesh, spmd_info.second[0]);
    phi::distributed::DistTensor* dist_out = shared_dist_out.get();
    phi::DenseTensor* dense_out = nullptr;
    if (dist_out) {{
      dense_out = dist_out->unsafe_mutable_value();
      if (dense_out && !rank_is_in_current_mesh && !dist_out->defined()) {{
        *dense_out = phi::DenseTensor(
            std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
            phi::DenseTensorMeta());
    }}
    }}
"""
SINGLE_OUT_CREATION_TEMPLATE = """
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out =
        CreateKernelDistOutput({}, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out = shared_dist_out.get();
    phi::DenseTensor* dense_out = nullptr;
    if (dist_out) {{
      dense_out = dist_out->unsafe_mutable_value();
      if (dense_out && !rank_is_in_current_mesh && !dist_out->defined()) {{
      *dense_out = phi::DenseTensor(
                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                phi::DenseTensorMeta());
    }}
    }}
"""
VECTOR_OUT_CREATION_TEMPLATE_WITH_NO_SPMD = """
    auto dist_out = SetKernelDistOutput({name});
    std::vector<phi::DenseTensor*> dense_out(dist_out.size(), nullptr);
    for (size_t i=0; i<dist_out.size(); i++) {{
      if (dist_out[i]) {{
        dense_out[i] = dist_out[i]->unsafe_mutable_value();
        if (dense_out[i] && !rank_is_in_current_mesh && !dist_out[i]->defined()) {{
            *dense_out[i] = phi::DenseTensor(
                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                phi::DenseTensorMeta());
        }}
      }}
    }}
"""

VECTOR_OUT_CREATION_TEMPLATE_WITH_SPMD = """
    auto shared_dist_out = CreateKernelDistOutput({name}, !rank_is_in_current_mesh, spmd_info.second[0]);
    std::vector<phi::distributed::DistTensor*> dist_out;
    for(auto& e: shared_dist_out){{
      dist_out.push_back(e.get());
    }}
    std::vector<phi::DenseTensor*> dense_out(dist_out.size(), nullptr);
    for (size_t i=0; i<dist_out.size(); i++) {{
      if (dist_out[i]) {{
        dense_out[i] = dist_out[i]->unsafe_mutable_value();
        if (dense_out[i] && !rank_is_in_current_mesh && !dist_out[i]->defined()) {{
            *dense_out[i] = phi::DenseTensor(
                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                phi::DenseTensorMeta());
        }}
      }}
    }}
"""


VECTOR_OUT_CREATION_TEMPLATE = """
    auto shared_dist_out = CreateKernelDistOutput({name}, !rank_is_in_current_mesh);
    std::vector<phi::distributed::DistTensor*> dist_out;
    for(auto& e: shared_dist_out){{
      dist_out.push_back(e.get());
    }}
    std::vector<phi::DenseTensor*> dense_out(dist_out.size(), nullptr);
    for (size_t i=0; i<dist_out.size(); i++) {{
      if (dist_out[i]) {{
        dense_out[i] = dist_out[i]->unsafe_mutable_value();
        if (dense_out[i] && !rank_is_in_current_mesh && !dist_out[i]->defined()) {{
            *dense_out[i] = phi::DenseTensor(
                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                phi::DenseTensorMeta());
        }}
      }}
    }}
"""


INPLACE_OUT_CREATION_TEMPLATE = """
    *{} = {};
"""
MULTI_SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD = """
    auto dist_out_{idx} = SetKernelDistOutput({name});
    auto dense_out_{idx} = dist_out_{idx} ? dist_out_{idx}->unsafe_mutable_value() : nullptr;
    if (dense_out_{idx} && !rank_is_in_current_mesh && !dist_out_{idx}->defined()) {{
      *dense_out_{idx} = phi::DenseTensor(
        std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
        phi::DenseTensorMeta());
    }}
"""
MULTI_SINGLE_OUT_CREATION_TEMPLATE_WITH_SPMD = """
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_{idx} =
        CreateKernelDistOutput({name}, !rank_is_in_current_mesh, spmd_info.second[{idx}]);
    phi::distributed::DistTensor* dist_out_{idx} = shared_dist_out_{idx}.get();
    phi::DenseTensor* dense_out_{idx} = dist_out_{idx} ? dist_out_{idx}->unsafe_mutable_value() : nullptr;
    if (dense_out_{idx} && !rank_is_in_current_mesh && !dist_out_{idx}->defined()) {{
      *dense_out_{idx} = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }}
"""
MULTI_SINGLE_OUT_CREATION_TEMPLATE = """
    std::shared_ptr<phi::distributed::DistTensor> shared_dist_out_{idx} =
        CreateKernelDistOutput({name}, !rank_is_in_current_mesh);
    phi::distributed::DistTensor* dist_out_{idx} = shared_dist_out_{idx}.get();
    phi::DenseTensor* dense_out_{idx} = dist_out_{idx} ? dist_out_{idx}->unsafe_mutable_value() : nullptr;
    if (dense_out_{idx} && !rank_is_in_current_mesh && !dist_out_{idx}->defined()) {{
      *dense_out_{idx} = phi::DenseTensor(
          std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
          phi::DenseTensorMeta());
    }}
"""
MULTI_VECTOR_OUT_CREATION_TEMPLATE = """
    auto dist_out_{i} = SetKernelDistOutput({name});
    std::vector<phi::DenseTensor*> dense_out_{i}(dist_out_{i}.size(), nullptr);
    for (size_t i = 0; i < dist_out_{i}.size(); i++) {{
      if (dist_out_{i}[i]) {{
        dense_out_{i}[i] = const_cast<phi::DenseTensor*>(&dist_out_{i}[i]->value());
        if (dense_out_{i}[i] && !rank_is_in_current_mesh && !dist_out_{i}[i]->defined()) {{
            *dense_out_{i}[i]= phi::DenseTensor(
                std::make_shared<phi::Allocation>(nullptr, 0, phi::distributed::GetDefaultPlace()),
                phi::DenseTensorMeta());
        }}
      }}
    }}
"""

# 9. Reshard Output
RESHARD_SINGLE_OUTPUT_TEMPLATE = """
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out, {}, "{}");"""

RESHARD_MULTI_SINGLE_OUTPUT_TEMPLATE = """
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out_{}, {}, "{}");"""

RESHARD_VECTOR_OUTPUT_TEMPLATE = """
      ReshardKernelOutputToApiOutput(dev_ctx, shared_dist_out, {}, "{}");"""

NONEED_TO_RESHARD_OUTPUT_TEMPLATE = """
    // API `{}` does not need to reshard output."""

SET_LOCAL_SHAPE_TEMPLATE = """
      {meta_tensor}.set_dims(phi::make_ddim(local_shape));"""


class DistBackwardAPI(DistForwardAPI, BackwardAPI):
    def __init__(self, backward_item_yaml):
        BackwardAPI.__init__(self, backward_item_yaml)
        self.forward_config = backward_item_yaml['forward']
        self.init_dist_api_members()

    # override DistForwardAPI's method
    def generate_output_creation_code(self) -> str:
        # backward api only need to generate kernel outputs
        output_num = len(self.outputs['types'])
        output_creation_code = ""
        output_creation_code += "\n    phi::DeviceContext* dev_ctx = nullptr;"
        if output_num == 1:
            self.dist_output_args.append('dist_out')
            self.dense_output_args.append('dense_out')
            if self.outputs['types'][0] == 'Tensor':
                if self.infer_meta['spmd_rule'] is not None:
                    output_creation_code += (
                        SINGLE_OUT_CREATION_TEMPLATE_WITH_SPMD.format(
                            self.outputs['names'][0]
                        )
                    )
                elif self.generate_general_infer_spmd is True:
                    output_creation_code += SINGLE_OUT_CREATION_TEMPLATE.format(
                        self.outputs['names'][0]
                    )
                else:
                    output_creation_code += (
                        SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD.format(
                            self.outputs['names'][0]
                        )
                    )
            elif self.outputs['types'][0] == 'std::vector<Tensor>':
                if self.infer_meta['spmd_rule'] is not None:
                    output_creation_code += (
                        VECTOR_OUT_CREATION_TEMPLATE_WITH_SPMD.format(
                            name=self.outputs['names'][0]
                        )
                    )
                elif self.generate_general_infer_spmd is True:
                    output_creation_code += VECTOR_OUT_CREATION_TEMPLATE.format(
                        name=self.outputs['names'][0]
                    )
                else:
                    output_creation_code += (
                        VECTOR_OUT_CREATION_TEMPLATE_WITH_NO_SPMD.format(
                            name=self.outputs['names'][0]
                        )
                    )
            else:
                self.vector_output_size_assertion_check()
        elif output_num > 1:
            for i, out_type in enumerate(self.outputs['types']):
                self.dist_output_args.append(f'dist_out_{i}')
                self.dense_output_args.append(f'dense_out_{i}')
                if out_type == 'Tensor':
                    if self.infer_meta['spmd_rule'] is not None:
                        output_creation_code += (
                            MULTI_SINGLE_OUT_CREATION_TEMPLATE_WITH_SPMD.format(
                                name=self.outputs['names'][i], idx=i
                            )
                        )
                    elif self.generate_general_infer_spmd is True:
                        output_creation_code += (
                            MULTI_SINGLE_OUT_CREATION_TEMPLATE.format(
                                name=self.outputs['names'][i], idx=i
                            )
                        )
                    else:
                        output_creation_code += (
                            MULTI_SINGLE_OUT_CREATION_TEMPLATE_NO_SPMD.format(
                                name=self.outputs['names'][i], idx=i
                            )
                        )
                elif out_type == 'std::vector<Tensor>':
                    output_creation_code += (
                        MULTI_VECTOR_OUT_CREATION_TEMPLATE.format(
                            i=i, name=self.outputs['names'][i]
                        )
                    )
                else:
                    self.vector_output_size_assertion_check()
        else:
            raise ValueError(
                f"{self.api} : Output error: the output should not be empty."
            )

        return output_creation_code

    def generate_bw_infer_local_shape_code(self) -> str:
        arg_name = self.infer_meta['local_shape']
        assert arg_name in self.outputs['names'], (
            f"Auto Parallel will calculate local_shape for {arg_name} "
            f"in {self.api}, but {arg_name} is not found in its outputs."
        )

        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            self.forward_config
        )
        # shape_type = self.attrs['attr_info'][shape_name][0]
        # out_name = self.dist_output_args[0]
        dist_out_name = self.dist_output_args[
            self.outputs['names'].index(arg_name)
        ]
        shape_type = self.get_shape_type(fw_attrs['attr_info'])
        return dist_api_gen.CALCULATE_LOCAL_SHAPE_TEMPLATE.format(
            out_name=dist_out_name,
            out_dist_attr=(
                "PADDLE_GET_CONST(phi::distributed::TensorDistAttr, spmd_info.second[0]);"
                if self.infer_meta['spmd_rule']
                else f"phi::distributed::TensorDistAttr(common::vectorize({dist_out_name}->dims()))"
            ),
            dtype=shape_type,
            op_name=self.kernel['func'][0],
        )

    def generate_infer_meta_code(self) -> str:
        (
            infer_meta_func_code,
            input_args_code,
            output_decl_code,
            output_args_code,
        ) = self.generate_infer_meta_func_and_args_code()

        infer_meta_code = ""

        if self.infer_meta['global_shape'] is not None:
            for i, out_name in enumerate(self.outputs['names']):
                if out_name == self.infer_meta[
                    'global_shape'
                ] and self.need_to_generate_code_for_inplace_impl(i):
                    infer_meta_code += dist_api_gen.SET_DIMS_TEMPLATE.format(
                        dst=self.dist_output_args[i],
                        src=(
                            self.dist_output_args[i] + '_tmp'
                            if i > 0
                            else self.dist_output_args[i]
                        ),
                    )

        infer_meta_code = (
            infer_meta_code
            + dist_api_gen.INFER_META_TEMPLATE.format(
                infer_meta_func_code, input_args_code, output_args_code
            )
        )
        # TODO(GhostScreaming): kernel like reshape need calculate local_shape
        if self.infer_meta['local_shape'] is not None:
            infer_meta_code += self.generate_bw_infer_local_shape_code()
            infer_meta_code += SET_LOCAL_SHAPE_TEMPLATE.format(
                meta_tensor="meta_" + self.dense_output_args[0]
            )

        return output_decl_code + infer_meta_code

    # override DistForwardAPI's method
    def generate_return_code(self) -> str:
        return "return;"

    # override BaseAPI's method
    def get_api_func_name(self):
        return self.api

    # override BaseAPI's method
    # The method lookup order are: (DistBackwardAPI.__mro__)
    # <class '__main__.DistBackwardAPI'>,
    # <class 'dist_api_gen.DistForwardAPI'>,
    # <class 'api_gen.ForwardAPI'>,
    # <class 'backward_api_gen.BackwardAPI'>,
    # <class 'api_base.BaseAPI'>,
    # <class 'object'>
    # if don't override it, the ForwardAPI's gene_output will be called
    def gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
        return BackwardAPI.gene_output(
            self,
            out_dtype_list,
            out_tensor_type_list,
            code_indent,
            inplace_flag,
        )

    # override BaseAPI's method
    def get_return_type(self, inplace_flag=False):
        return BackwardAPI.get_return_type(self)

    # override BaseAPI's method
    def gene_return_code(self):
        return ""

    # override BaseAPI's method
    def gene_api_declaration(self) -> str:
        return BackwardAPI.gene_api_declaration(self)

    def generate_reshard_output_code(self):
        reshard_output_code = ""
        if self.generate_infer_spmd is True:
            output_num = len(self.outputs['types'])
            if output_num == 1:
                if self.outputs['types'][0] == 'Tensor':
                    reshard_output_code += (
                        RESHARD_SINGLE_OUTPUT_TEMPLATE.format(
                            self.outputs['names'][0], self.outputs['names'][0]
                        )
                    )
                elif self.outputs['types'][0] == 'std::vector<Tensor>':
                    reshard_output_code += (
                        RESHARD_VECTOR_OUTPUT_TEMPLATE.format(
                            self.outputs['names'][0], self.outputs['names'][0]
                        )
                    )
                else:
                    self.vector_output_size_assertion_check()
            elif output_num > 1:
                for i, out_type in enumerate(self.outputs['types']):
                    if out_type == 'Tensor':
                        reshard_output_code += (
                            RESHARD_MULTI_SINGLE_OUTPUT_TEMPLATE.format(
                                i,
                                self.outputs['names'][i],
                                self.outputs['names'][i],
                            )
                        )
                    else:
                        self.vector_output_size_assertion_check()
            else:
                raise ValueError(
                    f"{self.api} : Output error: the output should not be empty."
                )
        else:
            reshard_output_code += NONEED_TO_RESHARD_OUTPUT_TEMPLATE.format(
                self.kernel['func'][0]
            )
            # do nothing
            pass

        return reshard_output_code

    def generate_auto_parallel_branch(self) -> str:
        # if no tensor input, do not generate auto parallel branch
        if len(self.inputs['names']) == 0:
            return ""
        infer_spmd_code = self.generate_infer_spmd_code()
        output_creation_code = self.generate_output_creation_code()
        infer_global_shape_code = self.generate_infer_global_shape_code()
        output_dist_attr_setting = self.generate_output_dist_attr_setting()
        kernel_selection_code = self.generate_kernel_selection_code()
        reshard_input_code = self.generate_reshard_input_code()
        (
            prepare_data_code,
            input_name_tensor_map,
        ) = self.generate_prepare_data_code()
        record_op_info_supplement_code = (
            self.generate_record_op_info_supplement(
                input_name_tensor_map, '    ', True
            )
        )
        infer_meta_code = self.generate_infer_meta_code()
        kernel_call_code = self.generate_kernel_call_code()
        fallback_code = self.generate_fallback_code()
        reshard_output_code = self.generate_reshard_output_code()
        return_code = self.generate_return_code()

        return MAIN_DIST_BRANCH_TEMPLATE.format(
            infer_spmd_code,
            output_creation_code,
            infer_global_shape_code,
            output_dist_attr_setting,
            kernel_selection_code,
            reshard_input_code,
            prepare_data_code,
            record_op_info_supplement_code,
            infer_meta_code,
            kernel_call_code,
            fallback_code,
            reshard_output_code,
            return_code,
        )


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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#endif

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
        dist_bw_api = DistBackwardAPI(bw_api)
        header_file.write(dist_bw_api.gene_api_declaration())
        if is_fused_backward_yaml is True:
            source_file.write(dist_bw_api.gene_api_code())
        else:
            source_file.write(dist_bw_api.gene_api_code())

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
