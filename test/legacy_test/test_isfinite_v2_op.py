# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from utils import static_guard

import paddle
from paddle import base, static


def run_static(x_np, dtype, op_str, use_gpu=False):
    with static_guard():
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        place = paddle.CPUPlace()
        if use_gpu and base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        exe = base.Executor(place)
        with static.program_guard(main_program, startup_program):
            x = paddle.static.data(name='x', shape=x_np.shape, dtype=dtype)
            res = getattr(paddle, op_str)(x)
            static_result = exe.run(feed={'x': x_np}, fetch_list=[res])
        return static_result


def run_dygraph(x_np, op_str, use_gpu=True):
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    x = paddle.to_tensor(x_np)
    dygraph_result = getattr(paddle, op_str)(x)
    return dygraph_result


def run_eager(x_np, op_str, use_gpu=True):
    with paddle.base.dygraph.guard():
        place = paddle.CPUPlace()
        if use_gpu and base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)

        x = paddle.to_tensor(x_np)
        dygraph_result = getattr(paddle, op_str)(x)
        return dygraph_result


def np_data_generator(
    low, high, np_shape, type, sv_list, op_str, *args, **kwargs
):
    x_np = np.random.uniform(low, high, np_shape).astype(getattr(np, type))
    # x_np.shape[0] >= len(sv_list)
    if type in ['float16', 'float32', 'float64']:
        for i, v in enumerate(sv_list):
            x_np[i] = v
    if type in ['complex64', 'complex128']:
        for i, v in enumerate(sv_list):
            x_np[i].real = v
            x_np[i].imag = v
    ori_shape = x_np.shape
    x_np = x_np.reshape((np.prod(ori_shape),))
    np.random.shuffle(x_np)
    x_np = x_np.reshape(ori_shape)
    result_np = getattr(np, op_str)(x_np)
    return x_np, result_np


TEST_META_DATA = [
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [8, 17, 5, 6, 7],
        'type': 'float16',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [11, 17],
        'type': 'float32',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [2, 3, 4, 5],
        'type': 'float64',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0,
        'high': 100,
        'np_shape': [11, 17, 10],
        'type': 'int32',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0,
        'high': 999,
        'np_shape': [132],
        'type': 'int64',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [11, 17],
        'type': 'complex64',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [2, 3, 4, 5],
        'type': 'complex128',
        'sv_list': [np.inf, np.nan],
    },
]

TEST_META_DATA_ADDITIONAL = [
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [2, 3, 4, 5],
        'type': 'int8',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0,
        'high': 100,
        'np_shape': [11, 17, 10],
        'type': 'int16',
        'sv_list': [np.inf, np.nan],
    },
    {
        'low': 0,
        'high': 999,
        'np_shape': [132],
        'type': 'uint8',
        'sv_list': [np.inf, np.nan],
    },
]

TEST_META_DATA2 = [
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [11, 17],
        'type': 'float32',
        'sv_list': [-np.inf, np.inf],
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [2, 3, 4, 5],
        'type': 'float64',
        'sv_list': [np.inf, -np.inf],
    },
    {
        'low': 0,
        'high': 999,
        'np_shape': [132],
        'type': 'uint8',
        'sv_list': [-np.inf, np.inf],
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [2, 3, 4, 5],
        'type': 'int8',
        'sv_list': [-np.inf, np.inf],
    },
    {
        'low': 0,
        'high': 100,
        'np_shape': [11, 17, 10],
        'type': 'int16',
        'sv_list': [np.inf, -np.inf],
    },
    {
        'low': 0,
        'high': 100,
        'np_shape': [11, 17, 10],
        'type': 'int32',
        'sv_list': [-np.inf, np.inf],
    },
    {
        'low': 0,
        'high': 999,
        'np_shape': [132],
        'type': 'int64',
        'sv_list': [np.inf, -np.inf],
    },
]

TEST_META_DATA3 = [
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [8, 17, 5, 6, 7],
        'type': 'float16',
        'sv_list': [np.inf, -np.inf],
    },
]


def test(test_case, op_str, use_gpu=False, data_set=TEST_META_DATA):
    for meta_data in data_set:
        meta_data = dict(meta_data)
        meta_data['op_str'] = op_str
        x_np, result_np = np_data_generator(**meta_data)

        dygraph_result = run_dygraph(x_np, op_str, use_gpu).numpy()
        eager_result = run_eager(x_np, op_str, use_gpu).numpy()

        test_case.assertTrue((dygraph_result == result_np).all())
        test_case.assertTrue((eager_result == result_np).all())

        def test_static_or_pir_mode():
            (static_result,) = run_static(
                x_np, meta_data['type'], op_str, use_gpu
            )
            test_case.assertTrue((static_result == result_np).all())

        test_static_or_pir_mode()


def test_bf16(test_case, op_str):
    x_np = np.array([float('inf'), -float('inf'), 2.0, 3.0])
    result_np = getattr(np, op_str)(x_np)

    place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    x = paddle.to_tensor(x_np, dtype='bfloat16')
    dygraph_result = getattr(paddle, op_str)(x).numpy()

    test_case.assertTrue((dygraph_result == result_np).all())


class TestCPUNormal(unittest.TestCase):
    def test_inf(self):
        test(self, 'isinf')

    def test_nan(self):
        test(self, 'isnan')

    def test_finite(self):
        test(self, 'isfinite')

    def test_inf_additional(self):
        test(self, 'isinf', data_set=TEST_META_DATA_ADDITIONAL)

    def test_posinf(self):
        test(self, 'isposinf', data_set=TEST_META_DATA2)

    def test_neginf(self):
        test(self, 'isneginf', data_set=TEST_META_DATA2)


class TestCUDANormal(unittest.TestCase):
    def test_inf(self):
        test(self, 'isinf', True)

    def test_nan(self):
        test(self, 'isnan', True)

    def test_finite(self):
        test(self, 'isfinite', True)

    def test_inf_additional(self):
        test(self, 'isinf', True, data_set=TEST_META_DATA_ADDITIONAL)

    def test_posinf(self):
        test(self, 'isposinf', True, data_set=TEST_META_DATA2)

    def test_neginf(self):
        test(self, 'isneginf', True, data_set=TEST_META_DATA2)


@unittest.skipIf(
    not base.core.is_compiled_with_cuda()
    or not base.core.is_float16_supported(base.core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestCUDAFP16(unittest.TestCase):
    def test_posinf(self):
        test(self, 'isposinf', True, data_set=TEST_META_DATA3)

    def test_neginf(self):
        test(self, 'isneginf', True, data_set=TEST_META_DATA3)


@unittest.skipIf(
    not base.core.is_compiled_with_cuda()
    or not base.core.is_bfloat16_supported(base.core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestCUDABFP16(unittest.TestCase):
    def test_posinf(self):
        test_bf16(self, 'isposinf')

    def test_neginf(self):
        test_bf16(self, 'isneginf')


class TestError(unittest.TestCase):

    def test_bad_input(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):

            def test_isinf_bad_x():
                x = [1, 2, 3]
                result = paddle.isinf(x)

            self.assertRaises(TypeError, test_isinf_bad_x)

            def test_isnan_bad_x():
                x = [1, 2, 3]
                result = paddle.isnan(x)

            self.assertRaises(TypeError, test_isnan_bad_x)

            def test_isfinite_bad_x():
                x = [1, 2, 3]
                result = paddle.isfinite(x)

            self.assertRaises(TypeError, test_isfinite_bad_x)

            def test_isposinf_bad_x():
                x = [1, 2, 3]
                result = paddle.isposinf(x)

            self.assertRaises(TypeError, test_isposinf_bad_x)

            def test_isneginf_bad_x():
                x = [1, 2, 3]
                result = paddle.isneginf(x)

            self.assertRaises(TypeError, test_isneginf_bad_x)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
