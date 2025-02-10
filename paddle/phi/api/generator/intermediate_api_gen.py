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

import yaml
from api_gen import ForwardAPI, backward_api_black_list
from dist_api_gen import DistForwardAPI
from sparse_api_gen import SparseAPI


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""#include "{header_file_path}"

#include <memory>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/sparse/binary.h"
#include "paddle/phi/infermeta/sparse/multiary.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

#ifdef PADDLE_WITH_DISTRIBUTE
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#endif

COMMON_DECLARE_int32(low_precision_op_list);
COMMON_DECLARE_bool(benchmark);
"""


def api_namespace():
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


def sparse_namespace():
    return (
        """
namespace sparse {
""",
        """
}  // namespace sparse
""",
    )


def generate_intermediate_api(
    api_yaml_path,
    sparse_api_yaml_path,
    dygraph_header_file_path,
    dygraph_source_file_path,
    gen_dist_branch,
):
    dygraph_header_file = open(dygraph_header_file_path, 'w')
    dygraph_source_file = open(dygraph_source_file_path, 'w')

    namespace = api_namespace()
    sparse_namespace_pair = sparse_namespace()

    dygraph_header_file.write("#pragma once\n")
    dygraph_header_file.write(header_include())
    dygraph_header_file.write(namespace[0])

    dygraph_include_header_file = "paddle/phi/api/lib/dygraph_api.h"
    dygraph_source_file.write(source_include(dygraph_include_header_file))
    dygraph_source_file.write(namespace[0])

    apis = []
    for each_api_yaml in api_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                apis.extend(api_list)

    for api in apis:
        forward_api = (
            DistForwardAPI(api) if gen_dist_branch else ForwardAPI(api)
        )
        if forward_api.is_dygraph_api:
            dygraph_header_file.write(forward_api.gene_api_declaration())
            dygraph_source_file.write(forward_api.gene_api_code())

    dygraph_header_file.write(sparse_namespace_pair[0])
    dygraph_source_file.write(sparse_namespace_pair[0])
    sparse_apis = []
    for each_sparse_api_yaml in sparse_api_yaml_path:
        with open(each_sparse_api_yaml, 'r') as f:
            sparse_api_list = yaml.load(f, Loader=yaml.FullLoader)
            if sparse_api_list:
                sparse_apis.extend(sparse_api_list)

    for api in sparse_apis:
        sparse_api = SparseAPI(api)
        if sparse_api.api in backward_api_black_list:
            continue
        if sparse_api.is_dygraph_api:
            dygraph_header_file.write(sparse_api.gene_api_declaration())
            dygraph_source_file.write(sparse_api.gene_api_code())

    dygraph_header_file.write(sparse_namespace_pair[1])
    dygraph_header_file.write(namespace[1])

    dygraph_source_file.write(sparse_namespace_pair[1])
    dygraph_source_file.write(namespace[1])

    dygraph_header_file.close()
    dygraph_source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ Sparse API files'
    )
    parser.add_argument(
        '--api_yaml_path',
        nargs='+',
        help='path to api yaml file',
        default=['paddle/phi/ops/yaml/ops.yaml'],
    )

    parser.add_argument(
        '--sparse_api_yaml_path',
        nargs='+',
        help='path to sparse api yaml file',
        default='paddle/phi/ops/yaml/sparse_ops.yaml',
    )

    parser.add_argument(
        '--dygraph_api_header_path',
        help='output of generated dygraph api header code file',
        default='paddle/phi/api/lib/dygraph_api.h',
    )

    parser.add_argument(
        '--dygraph_api_source_path',
        help='output of generated dygraph api source code file',
        default='paddle/phi/api/lib/dygraph_api.cc',
    )

    parser.add_argument(
        '--gen_dist_branch',
        help='whether generate distributed branch code',
        dest='gen_dist_branch',
        action='store_true',
    )

    options = parser.parse_args()

    api_yaml_path = options.api_yaml_path
    sparse_api_yaml_path = options.sparse_api_yaml_path
    dygraph_header_file_path = options.dygraph_api_header_path
    dygraph_source_file_path = options.dygraph_api_source_path
    gen_dist_branch = options.gen_dist_branch

    generate_intermediate_api(
        api_yaml_path,
        sparse_api_yaml_path,
        dygraph_header_file_path,
        dygraph_source_file_path,
        gen_dist_branch,
    )


if __name__ == '__main__':
    main()
