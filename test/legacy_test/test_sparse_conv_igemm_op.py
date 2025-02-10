# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import unittest

import numpy as np

import paddle
from paddle import sparse
from paddle.base import core

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "only test when CUDA is available",
)
class TestSparseConvImplicitGemm(unittest.TestCase):
    def test_SubmConv2D_igemm(self):
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[4], [5], [10], [7]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )
        sparse_input.stop_gradient = False

        subm_conv2d = paddle.sparse.nn.SubmConv2D(
            1,
            1,
            3,
            padding=1,
            stride=1,
            data_format='NHWC',
            key='subm_conv_2d',
            backend='igemm',
        )
        # set weight to all ones
        subm_conv2d.weight = paddle.create_parameter(
            (3, 3, 1, 1),
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0),
        )

        # check the forward
        sparse_out = subm_conv2d(sparse_input)
        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())
        np.testing.assert_array_equal(
            correct_out_values, sparse_out.values().numpy()
        )

        # check x_grad
        x_grad = paddle.grad(sparse_out, sparse_input)[0]
        correct_x_grad_values = [[2], [2], [4], [2]]
        np.testing.assert_array_equal(indices, x_grad.indices().numpy())
        np.testing.assert_array_equal(
            correct_x_grad_values, x_grad.values().numpy()
        )

        # check weight_grad
        loss = sparse_out.to_dense().mean()
        loss.backward()
        correct_weight_grad_values = [
            [[[0.33333334]], [[0.0]], [[0.16666667]]],
            [[[0.0]], [[0.8333334]], [[0.0]]],
            [[[0.25]], [[0.0]], [[0.5833334]]],
        ]
        np.testing.assert_allclose(
            correct_weight_grad_values, subm_conv2d.weight.grad.numpy()
        )

    def test_SubmConv3D_igemm(self):
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[4], [5], [10], [7]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )
        sparse_input.stop_gradient = False

        subm_conv3d = paddle.sparse.nn.SubmConv3D(
            1,
            1,
            (1, 3, 3),
            padding=1,
            stride=1,
            data_format='NDHWC',
            key='subm_conv',
            backend='igemm',
        )
        # set weight to all ones
        subm_conv3d.weight = paddle.create_parameter(
            (1, 3, 3, 1, 1),
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0),
        )

        # check the forward
        sparse_out = subm_conv3d(sparse_input)
        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())
        np.testing.assert_array_equal(
            correct_out_values, sparse_out.values().numpy()
        )

        # check x_grad
        x_grad = paddle.grad(sparse_out, sparse_input)[0]
        correct_x_grad_values = [[2], [2], [4], [2]]
        np.testing.assert_array_equal(indices, x_grad.indices().numpy())
        np.testing.assert_array_equal(
            correct_x_grad_values, x_grad.values().numpy()
        )

        # check weight_grad
        loss = sparse_out.to_dense().mean()
        loss.backward()
        correct_weight_grad_values = [
            [
                [[[0.33333334]], [[0.0]], [[0.16666667]]],
                [[[0.0]], [[0.8333334]], [[0.0]]],
                [[[0.25]], [[0.0]], [[0.5833334]]],
            ]
        ]
        np.testing.assert_allclose(
            correct_weight_grad_values, subm_conv3d.weight.grad.numpy()
        )

    def test_submconv2d_igemm(self):
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[5], [6], [11], [8]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )
        sparse_input.stop_gradient = False

        weight = paddle.ones((3, 3, 1, 1), dtype='float32')
        weight.stop_gradient = False
        bias = paddle.ones((1), dtype='float32')
        bias.stop_gradient = False
        sparse_out = paddle.sparse.nn.functional.subm_conv2d_igemm(
            sparse_input,
            weight,
            bias,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            data_format="NHWC",
            key='subm_conv_2d',
        )

        # check the forward
        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())
        np.testing.assert_array_equal(
            correct_out_values, sparse_out.values().numpy()
        )

        # check x_grad
        x_grad = paddle.grad(sparse_out, sparse_input)[0]
        correct_x_grad_values = [[2], [2], [4], [2]]
        np.testing.assert_array_equal(indices, x_grad.indices().numpy())
        np.testing.assert_array_equal(
            correct_x_grad_values, x_grad.values().numpy()
        )

        # check weight_grad
        loss = sparse_out.to_dense().mean()
        loss.backward()
        correct_weight_grad_values = [
            [[[0.33333334]], [[0.0]], [[0.16666667]]],
            [[[0.0]], [[0.8333334]], [[0.0]]],
            [[[0.25]], [[0.0]], [[0.5833334]]],
        ]
        np.testing.assert_allclose(
            correct_weight_grad_values, weight.grad.numpy()
        )
        correct_bias_grad_values = [0.33333334]
        np.testing.assert_allclose(correct_bias_grad_values, bias.grad.numpy())

    def test_submconv3d_igemm_forward(self):
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[5], [6], [11], [8]]
        sparse_input = paddle.sparse.sparse_coo_tensor(
            indices, values, dense_shape, False
        )
        sparse_input.stop_gradient = False

        weight = paddle.ones((1, 3, 3, 1, 1), dtype='float32')
        weight.stop_gradient = False
        bias = paddle.ones((1), dtype='float32')
        bias.stop_gradient = False
        sparse_out = paddle.sparse.nn.functional.subm_conv3d_igemm(
            sparse_input,
            weight,
            bias,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            data_format="NDHWC",
            key='subm_conv_3d',
        )

        # check the forward
        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())
        np.testing.assert_array_equal(
            correct_out_values, sparse_out.values().numpy()
        )

        # check x_grad
        x_grad = paddle.grad(sparse_out, sparse_input)[0]
        correct_x_grad_values = [[2], [2], [4], [2]]
        np.testing.assert_array_equal(indices, x_grad.indices().numpy())
        np.testing.assert_array_equal(
            correct_x_grad_values, x_grad.values().numpy()
        )

        # check weight_grad
        loss = sparse_out.to_dense().mean()
        loss.backward()
        correct_weight_grad_values = [
            [
                [[[0.33333334]], [[0.0]], [[0.16666667]]],
                [[[0.0]], [[0.8333334]], [[0.0]]],
                [[[0.25]], [[0.0]], [[0.5833334]]],
            ]
        ]
        np.testing.assert_allclose(
            correct_weight_grad_values, weight.grad.numpy()
        )
        correct_bias_grad_values = [0.33333334]
        np.testing.assert_allclose(correct_bias_grad_values, bias.grad.numpy())

    def test_multi_input(self):
        indices_1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        indices_2 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [0, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices_1 = paddle.to_tensor(indices_1, dtype='int32')
        indices_2 = paddle.to_tensor(indices_2, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values_1 = [[4], [5], [10], [7]]
        correct_out_values_2 = [[1], [5], [9], [7]]
        sparse_input_1 = paddle.sparse.sparse_coo_tensor(
            indices_1, values, dense_shape, False
        )
        sparse_input_2 = paddle.sparse.sparse_coo_tensor(
            indices_2, values, dense_shape, False
        )

        subm_conv3d = paddle.sparse.nn.SubmConv3D(
            1,
            1,
            (1, 3, 3),
            padding=1,
            stride=1,
            data_format='NDHWC',
            key='subm_conv',
            backend='igemm',
        )
        # set weight to all ones
        subm_conv3d.weight = paddle.create_parameter(
            (1, 3, 3, 1, 1),
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0),
        )

        sparse_out = subm_conv3d(sparse_input_1)
        np.testing.assert_array_equal(indices_1, sparse_out.indices().numpy())
        np.testing.assert_array_equal(
            correct_out_values_1, sparse_out.values().numpy()
        )

        sparse_out = subm_conv3d(sparse_input_2)

        # the output shape of subm_conv is same as input shape
        np.testing.assert_array_equal(indices_2, sparse_out.indices().numpy())
        np.testing.assert_array_equal(
            correct_out_values_2, sparse_out.values().numpy()
        )


class TestStatic(unittest.TestCase):

    def test3d(self):
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(
                name='indices', shape=[4, 4], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[4, 1], dtype='float32'
            )
            dense_shape = [1, 1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

            weight_shape = [1, 3, 3, 1, 1]
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype='float32'
            )
            bias_shape = [1]
            bias = paddle.static.data(
                name='bias', shape=bias_shape, dtype='float32'
            )
            out = sparse.nn.functional.subm_conv3d_igemm(
                sp_x,
                weight,
                bias,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                data_format="NDHWC",
            )
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()

            exe = paddle.static.Executor()

            indices_data = [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 2],
                [1, 3, 2, 3],
            ]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array(
                [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
            ).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')

            fetch = exe.run(
                feed={
                    'indices': indices_data,
                    'values': values_data,
                    'weight': weight_data,
                    'bias': bias_data,
                },
                fetch_list=[out, out_indices, out_values],
                return_numpy=True,
            )
            correct_out_values = [[5.0], [6.0], [11.0], [8.0]]
            np.testing.assert_array_equal(correct_out_values, fetch[2])
        paddle.disable_static()

    def test2d(self):
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(
                name='indices', shape=[3, 4], dtype='int32'
            )
            values = paddle.static.data(
                name='values', shape=[4, 1], dtype='float32'
            )
            dense_shape = [1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)

            weight_shape = [3, 3, 1, 1]
            weight = paddle.static.data(
                name='weight', shape=weight_shape, dtype='float32'
            )
            bias_shape = [1]
            bias = paddle.static.data(
                name='bias', shape=bias_shape, dtype='float32'
            )
            out = sparse.nn.functional.subm_conv2d_igemm(
                sp_x,
                weight,
                bias,
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                data_format="NHWC",
            )
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()

            exe = paddle.static.Executor()

            indices_data = [
                [0, 0, 0, 0],
                [0, 0, 1, 2],
                [1, 3, 2, 3],
            ]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array(
                [[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]
            ).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')

            fetch = exe.run(
                feed={
                    'indices': indices_data,
                    'values': values_data,
                    'weight': weight_data,
                    'bias': bias_data,
                },
                fetch_list=[out, out_indices, out_values],
                return_numpy=True,
            )
            correct_out_values = [[5.0], [6.0], [11.0], [8.0]]
            np.testing.assert_array_equal(correct_out_values, fetch[2])
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
