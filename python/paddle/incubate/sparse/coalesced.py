#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import _C_ops
from paddle.fluid.framework import core, dygraph_only

__all__ = [
    'coalesced',
]


@dygraph_only
def coalesced(x):
    r"""
    the coalesced operator include sorted and merge, after coalesced, the indices of x is sorted and unique, .

    Args:
        x (Tensor): the input SparseCooTensor.

    Returns:
        Tensor: return the SparseCooTensor after coalesced.

    Examples:

    ..  code-block:: python

        import paddle
        from paddle.incubate import sparse
        from paddle.fluid.framework import _test_eager_guard

        with _test_eager_guard():
            indices = [[0, 0, 1], [1, 1, 2]]
            values = [1.0, 2.0, 3.0]
            sp_x = sparse.sparse_coo_tensor(indices, values)
            sp_x = sparse.coalesced(sp_x)
            print(sp_x.indices())
            #[[0, 1], [1, 2]]
            print(sp_x.values())
            #[3.0, 3.0]
	"""
    return _C_ops.final_state_sparse_coalesced(x)
