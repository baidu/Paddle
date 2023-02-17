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

import yaml
from api_gen import ForwardAPI

inplace_out_type_map = {
    "Tensor": "Tensor&",
    "std::vector<Tensor>": "std::vector<Tensor>&",
}

inplace_optional_out_type_map = {
    "Tensor": "paddle::optional<Tensor>&",
    "std::vector<Tensor>": "paddle::optional<std::vector<Tensor>>&",
}

indent = "  "


operants_base_include = """// Generated by paddle/phi/api/yaml/generator/tensor_gen.py

#pragma once

#include "paddle/phi/api/include/tensor.h"

"""

operants_base_start = """
namespace paddle {

namespace operants {

using Tensor = paddle::experimental::Tensor;

class TensorOperantsBase {
 public:
  virtual ~TensorOperantsBase() = default;
"""


operants_base_end = """};

}  // namespace operants
}  // namespace paddle

"""


operants_header_include = """// Generated by paddle/phi/api/yaml/generator/tensor_gen.py

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"

"""

operants_header_start = """
namespace paddle {

namespace operants {

class PhiTensorOperants : public TensorOperantsBase {
 private:
  DISABLE_COPY_AND_ASSIGN(PhiTensorOperants);

 public:
  PhiTensorOperants() = default;
"""


operants_header_end = """};

}  // namespace operants
}  // namespace paddle

"""


operants_source_include = """// Generated by paddle/phi/api/yaml/generator/tensor_gen.py

#include "paddle/phi/api/include/tensor_operants.h"

#include "paddle/phi/api/include/api.h"

"""


operants_source_start = """
namespace paddle {

namespace operants {
"""


operants_source_end = """
}  // namespace operants
}  // namespace paddle

"""


operants_manager_header_include = """// Generated by paddle/phi/api/yaml/generator/tensor_gen.py

#pragma once

#include "paddle/phi/api/include/operants_base.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"

"""

operants_manager_header_start = """
namespace paddle {

using Tensor = paddle::experimental::Tensor;
using TensorOperantsBase = paddle::operants::TensorOperantsBase;

/**
 * [ Why need OperantsManager? ]
 *
 * Ideally, overloading tensor operators should call Tensor API directly.
 * However, we faced two problems:
 *
 * 1. Support multiple modes: Tensor operator overloading needs to support
 * [static mode / autograd mode / custom operator mode] at the same time.
 *
 * 2. Decouple phi and fluid: Tensor belongs to the phi library, but it relies
 * upon functions in fluid when overloading Tensor operators.
 *
 * We design OperantsManager to solve these two problems:
 *
 * 1. use `FLAGS_tensor_operants_mode` to handle overloading mode, set this flag
 * at the entry point of each mode:
 *
 * - FLAGS_tensor_operants_mode = "static": at the construction function of
 * `CompositeGradOpMakerBase`.
 * - FLAGS_tensor_operants_mode = "eager": at the beginning of dygraph_function.
 * - FLAGS_tensor_operants_mode = "phi": at the beginning of the
 * `eager_api_run_custom_op` function in eager mode and at the beginning of
 * calling kernels in static mode.
 *
 * In order to guarantee the performance, OperantsManager holds three pointers
 * to identify each mode respectively.
 *
 * 2. Decouple phi with the help of the polymorphism mechanism,
 * TensorOperantsBase derives three child classes: PhiTensorOperants,
 * EagerTensorOperants, and StaticTensorOperants. We set eager and static tensor
 * operants at the fluid library and set phi operants at the phi library.
 *
 */
class OperantsManager {
 private:
  OperantsManager() = default;
  DISABLE_COPY_AND_ASSIGN(OperantsManager);

 public:
  std::unique_ptr<TensorOperantsBase> eager_operants{nullptr};
  std::unique_ptr<TensorOperantsBase> static_operants{nullptr};
  std::unique_ptr<TensorOperantsBase> phi_operants{nullptr};

 public:
  static OperantsManager& Instance();
"""


operants_manager_header_end = """};

}  // namespace paddle

"""


operants_manager_source_include = """// Generated by paddle/phi/api/yaml/generator/tensor_gen.py

#include "paddle/phi/api/include/operants_manager.h"

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

"""


operants_manager_source_start = """
DECLARE_string(tensor_operants_mode);

namespace paddle {

OperantsManager& OperantsManager::Instance() {
  static OperantsManager g_op_manager;
  return g_op_manager;
}
"""


operants_manager_source_end = """
}  // namespace paddle

"""


class OperantsAPI(ForwardAPI):
    def __init__(self, api_item_yaml, prims=()):
        super().__init__(api_item_yaml)
        self.is_prim_api = False
        if self.get_api_func_name() in prims:
            self.is_prim_api = True

    def gene_operants_base(self):
        api_func_name = self.get_api_func_name()
        if api_func_name[-1] != '_':
            return f"""
{indent}virtual {self.get_return_type()} {api_func_name}({self.get_declare_args()}) = 0;
"""
        else:
            return f"""
{indent}virtual {self.get_return_type(inplace_flag=True)} {api_func_name}({self.get_declare_args(inplace_flag=True)}) = 0;
"""

    def gene_operants_declaration(self):
        api_func_name = self.get_api_func_name()
        if api_func_name[-1] != '_':
            return f"""
{indent}{self.get_return_type()} {api_func_name}({self.get_declare_args()});
"""
        else:
            return f"""
{indent}{self.get_return_type(inplace_flag=True)} {api_func_name}({self.get_declare_args(inplace_flag=True)});
"""

    def gene_operants_implementation(self):
        func_name = self.get_api_func_name()
        func_args = self.inputs['names'] + self.attrs['names']
        func_args_code = ", ".join(func_args)
        # func decalaration
        if func_name[-1] != '_':
            return f"""
{self.get_return_type()} PhiTensorOperants::{func_name}({self.get_define_args()}) {{
{indent}return paddle::experimental::{func_name}({func_args_code});
}}
"""
        else:
            return f"""
{self.get_return_type(inplace_flag=True)} PhiTensorOperants::{func_name}({self.get_define_args(inplace_flag=True)}) {{
{indent}return paddle::experimental::{func_name}({func_args_code});
}}

"""

    def gene_operants_manager_code(self):
        func_name = self.get_api_func_name()
        func_args = self.inputs['names'] + self.attrs['names']
        func_args_code = ", ".join(func_args)
        return f"""
  if (FLAGS_tensor_operants_mode == "eager") {{
    PADDLE_ENFORCE_NE(
        this->eager_operants.get(),
        nullptr,
        phi::errors::Unavailable("The eager_operants pointer of "
                                 "OperantsManager is not initialized"));
    VLOG(4) << "OperantsManager reaches eager mode";
    return this->eager_operants->{func_name}({func_args_code});
  }} else if (FLAGS_tensor_operants_mode == "static") {{
    PADDLE_ENFORCE_NE(
        this->static_operants.get(),
        nullptr,
        phi::errors::Unavailable("The static_operants pointer of "
                                 "OperantsManager is not initialized"));
    VLOG(4) << "OperantsManager reaches static mode";
    return this->static_operants->{func_name}({func_args_code});
  }} else if (FLAGS_tensor_operants_mode == "phi") {{
    PADDLE_ENFORCE_NE(
        this->phi_operants.get(),
        nullptr,
        phi::errors::Unavailable(
            "The phi_operants pointer of OperantsManager is not initialized"));
    VLOG(4) << "OperantsManager reaches phi mode";
    return this->phi_operants->{func_name}({func_args_code});
  }} else {{
    PADDLE_THROW(phi::errors::Unimplemented(
        "FLAGS_tensor_operants_mode is not nitialized, please set "
        "FLAGS_tensor_operants_mode first, which currently supports eager, "
        "phi, and static mode"));
  }}
"""

    def gene_operants_manager_implementation(self):
        func_name = self.get_api_func_name()
        # func decalaration
        if func_name[-1] != '_':
            return f"""
{self.get_return_type()} OperantsManager::{func_name}({self.get_define_args()}) {{{self.gene_operants_manager_code()}}}
"""
        else:
            return f"""
{self.get_return_type(inplace_flag=True)} OperantsManager::{func_name}({self.get_define_args(inplace_flag=True)}) {{
{self.gene_operants_manager_code()}
}}
"""


def generate_tensor_operants_api(
    api_yaml_path,
    operants_base_path,
    operants_header_path,
    operants_source_path,
    operants_manager_header_path,
    operants_manager_source_path,
    api_prim_path,
):
    apis = []

    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    operants_base_file = open(operants_base_path, 'w')
    operants_header_file = open(operants_header_path, 'w')
    operants_source_file = open(operants_source_path, 'w')
    operants_manager_header_file = open(operants_manager_header_path, 'w')
    operants_manager_source_file = open(operants_manager_source_path, 'w')

    operants_base_file.write(operants_base_include)
    operants_base_file.write(operants_base_start)
    operants_header_file.write(operants_header_include)
    operants_header_file.write(operants_header_start)
    operants_source_file.write(operants_source_include)
    operants_source_file.write(operants_source_start)
    operants_manager_header_file.write(operants_manager_header_include)
    operants_manager_header_file.write(operants_manager_header_start)
    operants_manager_source_file.write(operants_manager_source_include)
    operants_manager_source_file.write(operants_manager_source_start)

    with open(api_prim_path, 'rt') as f:
        api_prims = yaml.safe_load(f)
        # white list temporarily
        api_prims = ('add', 'subtract', 'multiply', 'divide')

    for api in apis:
        operants_api = OperantsAPI(api, api_prims)
        if operants_api.is_prim_api:
            operants_base_file.write(operants_api.gene_operants_base())
            operants_header_file.write(operants_api.gene_operants_declaration())
            operants_source_file.write(
                operants_api.gene_operants_implementation()
            )
            operants_manager_header_file.write(
                operants_api.gene_operants_declaration()
            )
            operants_manager_source_file.write(
                operants_api.gene_operants_manager_implementation()
            )

    operants_base_file.write(operants_base_end)
    operants_header_file.write(operants_header_end)
    operants_source_file.write(operants_source_end)
    operants_manager_header_file.write(operants_manager_header_end)
    operants_manager_source_file.write(operants_manager_source_end)

    operants_base_file.close()
    operants_header_file.close()
    operants_source_file.close()
    operants_manager_header_file.close()
    operants_manager_source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        help='path to api yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--operants_base_path',
        help='output of generated operants_base header code file',
        default='paddle/phi/api/include/operants_base.h',
    )

    parser.add_argument(
        '--phi_tensor_operants_header_path',
        help='output of generated phi_tensor_operants header code file',
        default='paddle/phi/api/include/tensor_operants.h',
    )

    parser.add_argument(
        '--phi_tensor_operants_source_path',
        help='output of generated phi_tensor_operants source code file',
        default='paddle/phi/api/lib/tensor_operants.cc',
    )

    parser.add_argument(
        '--operants_manager_header_path',
        help='output of generated operants_manager header code file',
        default='paddle/phi/api/include/operants_manager.h',
    )

    parser.add_argument(
        '--operants_manager_source_path',
        help='output of generated operants_manager source code file',
        default='paddle/phi/api/lib/operants_manager.cc',
    )

    parser.add_argument(
        '--api_prim_yaml_path',
        help='Primitive API list yaml file.',
        default='paddle/fluid/prim/api/api.yaml',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    operants_base_path = options.operants_base_path
    operants_header_path = options.phi_tensor_operants_header_path
    operants_source_path = options.phi_tensor_operants_source_path
    operants_manager_header_path = options.operants_manager_header_path
    operants_manager_source_path = options.operants_manager_source_path
    api_prim_yaml_path = options.api_prim_yaml_path

    generate_tensor_operants_api(
        api_yaml_path,
        operants_base_path,
        operants_header_path,
        operants_source_path,
        operants_manager_header_path,
        operants_manager_source_path,
        api_prim_yaml_path,
    )


if __name__ == '__main__':
    main()
