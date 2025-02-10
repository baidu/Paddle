# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op_test import OpTest
from utils import dygraph_guard, static_guard

import paddle
from paddle import base, static
from paddle.base import core

paddle.enable_static()
SEED = 2049
np.random.seed(SEED)


def np_matrix_rank_atol_rtol(x, atol=None, rtol=None, hermitian=False):
    if (atol is None) and (rtol is None):
        return np.linalg.matrix_rank(x, hermitian=hermitian)
    use_default_tol = False
    if atol is None:
        atol = np.array(0.0).astype(x.dtype)
    if rtol is None:
        rtol = np.array(0.0).astype(x.dtype)
        use_default_tol = True
    atol, rtol = np.broadcast_arrays(atol, rtol)
    _, sv, _ = np.linalg.svd(x)
    sv_max = sv.max(axis=-1)
    if use_default_tol:
        rtol_T = np.finfo(x.dtype).eps * max(x.shape[-2], x.shape[-1]) * sv_max
        rtol_default = np.full(x.shape[:-2], rtol_T, x.dtype)
        rtol = np.where(atol == 0.0, rtol_default, rtol * sv_max)
    else:
        rtol = rtol * sv_max
    tol = np.maximum(atol, rtol)
    return np.linalg.matrix_rank(x, tol, hermitian=hermitian)


def matrix_rank_atol_rtol_wrapper(x, atol=None, rtol=None, hermitian=False):
    return paddle.linalg.matrix_rank(x, None, hermitian, atol, rtol)


class TestMatrixRankAtolRtolOP(OpTest):
    def setUp(self):
        self.python_api = matrix_rank_atol_rtol_wrapper
        self.op_type = "matrix_rank_atol_rtol"
        self.init_data()
        self.process_data()
        self.inputs = {'x': self.x}
        self.inputs['atol'] = self.atol
        self.inputs['rtol'] = self.rtol
        self.attrs = {'hermitian': self.hermitian}
        self.outputs = {'out': self.out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.atol = 0.05
        self.rtol = None
        self.hermitian = True

    def process_data(self):
        self.out = np_matrix_rank_atol_rtol(
            self.x, self.atol, self.rtol, self.hermitian
        )
        if self.atol is None:
            self.atol = np.full([], 0.0, self.x.dtype)
        if isinstance(self.atol, (float, int)):
            self.atol = np.full([], self.atol, self.x.dtype)

        if self.rtol is None:
            self.rtol = np.full([], 0.0, self.x.dtype)
        if isinstance(self.rtol, (float, int)):
            self.rtol = np.full([], self.rtol, self.x.dtype)
        self.atol, self.rtol = np.broadcast_arrays(self.atol, self.rtol)


class TestMatrixRankAtolRtolOP1(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(3, dtype=np.float32)
        self.atol = None
        self.rtol = 0.05
        self.hermitian = True


class TestMatrixRankAtolRtolOP2(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.atol = np.random.random([3, 4]).astype(self.x.dtype)
        self.rtol = None
        self.hermitian = False


class TestMatrixRankAtolRtolOP3(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(3, 4, 5, 6).astype(np.float32)
        self.atol = None
        self.rtol = np.random.random([3, 4]).astype(self.x.dtype)
        self.hermitian = False


class TestMatrixRankAtolRtolOP4(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(1, 10).astype(np.float32)
        self.atol = 0.2
        self.rtol = 1.1
        self.hermitian = False


class TestMatrixRankAtolRtolOP5(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.random.rand(5, 1).astype(np.float64)
        self.atol = np.random.random([1, 4]).astype(self.x.dtype)
        self.rtol = np.random.random([1, 4]).astype(self.x.dtype)
        self.hermitian = False


class TestMatrixRankAtolRtolOP6(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.atol = np.random.random([200, 200]).astype(self.x.dtype)
        self.rtol = 0.8
        self.hermitian = True


class TestMatrixRankAtolRtolOP7(TestMatrixRankAtolRtolOP):
    def init_data(self):
        self.x = np.eye(200, dtype=np.float64)
        self.atol = np.random.random([200, 1]).astype(self.x.dtype)
        self.rtol = np.random.random([200, 200]).astype(self.x.dtype)
        self.hermitian = True


class TestMatrixRankAtolRtolAPI(unittest.TestCase):
    def test_dygraph(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            paddle.disable_static(place)

            # atol: float, rtol: None
            x_np = np.eye(10, dtype=np.float32)
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.015, rtol=None, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=0.015, rtol=None
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: float, rtol: None
            x_np = np.ones([3, 4, 5, 5])
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.5, rtol=None, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=0.5, rtol=None
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: None, rtol: float
            x_np = np.eye(10, dtype=np.float32)
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=None, rtol=1.1, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=None, rtol=1.1
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: float, rtol: float
            x_np = np.eye(10, dtype=np.float32)
            x_pd = paddle.to_tensor(x_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.2, rtol=0.05, hermitian=True
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=True, atol=0.2, rtol=0.05
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: float
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            atol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=0.01, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=0.01
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: float, rtol: tensor
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            rtol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            rtol_pd = paddle.to_tensor(rtol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=0.01, rtol=rtol_np, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=0.01, rtol=rtol_pd
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: tensor
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            atol_np = np.random.random([3, 4]).astype(np.float32)
            rtol_np = np.random.random([3, 4]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rtol_pd = paddle.to_tensor(rtol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=rtol_np, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=rtol_pd
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: tensor; broadcast_shape
            x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
            atol_np = np.random.random([3, 1]).astype(np.float32)
            rtol_np = np.random.random([3, 1]).astype(np.float32)
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rtol_pd = paddle.to_tensor(rtol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=rtol_np, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=rtol_pd
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

            # atol: tensor, rtol: None; atol specified as 0
            x_np = np.random.rand(3, 4, 5, 5)
            atol_np = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0, 10.0],
                    [10.0, 0.0, 10.0, 0.0],
                ]
            )
            x_pd = paddle.to_tensor(x_np)
            atol_pd = paddle.to_tensor(atol_np)
            rank_np = np_matrix_rank_atol_rtol(
                x_np, atol=atol_np, rtol=None, hermitian=False
            )
            rank_pd = paddle.linalg.matrix_rank(
                x_pd, hermitian=False, atol=atol_pd, rtol=None
            )
            np.testing.assert_allclose(rank_np, rank_pd, rtol=1e-05)

    def test_static(self):
        paddle.enable_static()
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            # atol: float, rtol: None
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.015, rtol=None, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=0.015, rtol=None
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: None, rtol: float
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=None, rtol=1.1, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=None, rtol=1.1
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: float, rtol: None
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.ones([3, 4, 5, 5])
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.5, rtol=None, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=0.5, rtol=None
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: float, rtol: float
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.05, rtol=0.2, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=0.05, rtol=0.2
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: tensor, rtol: float
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 8], dtype='float64'
                )
                atol_np = np.random.random([3, 4]).astype(np.float32)
                atol_pd = paddle.static.data(
                    name="Atol", shape=[3, 4], dtype='float32'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=atol_np, rtol=0.02, hermitian=False
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=False, atol=atol_pd, rtol=0.02
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Atol": atol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: float, rtol: tensor
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.random.rand(3, 4, 7, 8).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=[3, 4, 7, 8], dtype='float64'
                )
                rtol_np = np.random.random([3, 4]).astype(np.float32)
                rtol_pd = paddle.static.data(
                    name="Rtol", shape=[3, 4], dtype='float32'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=0.02, rtol=rtol_np, hermitian=False
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=False, atol=0.02, rtol=rtol_pd
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Rtol": rtol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: tensor, rtol: tensor
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.eye(10).astype(np.float64)
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                atol_np = np.array([0.02]).astype(np.float32)
                atol_pd = paddle.static.data(
                    name="Atol", shape=atol_np.shape, dtype='float32'
                )
                rtol_np = np.array([0.02]).astype(np.float32)
                rtol_pd = paddle.static.data(
                    name="Rtol", shape=rtol_np.shape, dtype='float32'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=atol_np, rtol=rtol_np, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=atol_pd, rtol=rtol_pd
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Atol": atol_np, "Rtol": rtol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)

        for place in places:
            # atol: tensor, rtol: None; atol specified as 0
            with static.program_guard(static.Program(), static.Program()):
                x_np = np.ones([3, 4, 5, 5])
                x_pd = paddle.static.data(
                    name="X", shape=x_np.shape, dtype='float64'
                )
                atol_np = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [10.0, 10.0, 10.0, 10.0],
                        [10.0, 0.0, 10.0, 0.0],
                    ]
                )
                atol_pd = paddle.static.data(
                    name="Atol", shape=atol_np.shape, dtype='float64'
                )
                rank_np = np_matrix_rank_atol_rtol(
                    x_np, atol=atol_np, rtol=None, hermitian=True
                )
                rank_pd = paddle.linalg.matrix_rank(
                    x_pd, hermitian=True, atol=atol_pd, rtol=None
                )
                exe = base.Executor(place)
                fetches = exe.run(
                    feed={"X": x_np, "Atol": atol_np},
                    fetch_list=[rank_pd],
                )
                np.testing.assert_allclose(fetches[0], rank_np, rtol=1e-05)


class TestMatrixRankError(unittest.TestCase):
    def test_errors(self):
        x = paddle.eye(10)
        with self.assertRaises(ValueError):
            paddle.linalg.matrix_rank(x, tol=0.2, hermitian=True, atol=0.2)

        with self.assertRaises(ValueError):
            paddle.linalg.matrix_rank(x, tol=0.2, hermitian=True, rtol=0.2)


class TestMatrixRankAtolRtolZeroSizeTensor(unittest.TestCase):

    def _get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        return places

    def _test_matrix_rank_static(self, place, atol, rtol):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_valid = paddle.static.data(
                    name='x_valid', shape=[2, 0, 6, 6], dtype='float32'
                )

                y_valid = paddle.linalg.matrix_rank(
                    x_valid, atol=atol, rtol=rtol
                )

                exe = paddle.static.Executor(place)
                res_valid = exe.run(
                    feed={'x_valid': np.zeros((2, 0, 6, 6), dtype='float32')},
                    fetch_list=[y_valid],
                )
                self.assertEqual(res_valid[0].shape, tuple(x_valid.shape[:-2]))

    def _test_matrix_rank_dynamic(self, atol, rtol):
        with dygraph_guard():
            x_valid = paddle.full((2, 0, 6, 6), 1.0, dtype='float32')
            x_invalid1 = paddle.full((0, 0), 1.0, dtype='float32')
            x_invalid2 = paddle.full((2, 3, 0, 0), 1.0, dtype='float32')
            self.assertRaises(
                ValueError,
                paddle.linalg.matrix_rank,
                x_invalid1,
                atol=atol,
                rtol=rtol,
            )
            self.assertRaises(
                ValueError,
                paddle.linalg.matrix_rank,
                x_invalid2,
                atol=atol,
                rtol=rtol,
            )

            y_valid = paddle.linalg.matrix_rank(x_valid, atol=atol, rtol=rtol)
            self.assertEqual(y_valid.shape, x_valid.shape[:-2])

    def test_matrix_rank_tensor(self):
        atol = 0.2
        rtol = 0.2
        for place in [paddle.CPUPlace()]:
            self._test_matrix_rank_static(place, atol, rtol)
        self._test_matrix_rank_dynamic(atol, rtol)


if __name__ == '__main__':
    unittest.main()
