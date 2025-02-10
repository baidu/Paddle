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

import math
import unittest

import numpy as np
import scipy.stats
from distribution import config
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
from test_distribution import DistributionNumpy

import paddle
from paddle import base
from paddle.distribution import Normal

np.random.seed(2022)


class InitDataContextManager:
    def __init__(self, in_pir, prog):
        self.in_pir = in_pir
        self.prog = prog

    def __enter__(self):
        if self.in_pir:
            self.guard = paddle.pir_utils.IrGuard()
            self.guard.__enter__()
            self.program_guard = paddle.static.program_guard(self.prog)
            self.program_guard.__enter__()
        else:
            self.program_guard = base.program_guard(self.prog)
            self.program_guard.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.in_pir:
            self.program_guard.__exit__(exc_type, exc_value, traceback)
            self.guard.__exit__(exc_type, exc_value, traceback)
        else:
            self.program_guard.__exit__(exc_type, exc_value, traceback)


class NormalNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        self.loc = np.array(loc)
        self.scale = np.array(scale)
        self._complex_gaussian = False
        if str(self.loc.dtype) not in [
            'float32',
            'float64',
            'complex64',
            'complex128',
        ]:
            self.loc = self.loc.astype('float32')
            self.scale = self.scale.astype('float32')
        if str(self.loc.dtype) in ['complex64', 'complex128']:
            self._complex_gaussian = True

    def sample(self, shape):
        shape = tuple(shape) + (self.loc + self.scale).shape
        if self._complex_gaussian:
            eps = np.vectorize(complex)(
                np.random.randn(*shape), np.random.randn(*shape)
            )
        else:
            eps = np.random.randn(*shape)
        return self.loc + (eps * self.scale)

    def log_prob(self, value):
        var = self.scale * self.scale
        log_scale = np.log(self.scale)
        if self._complex_gaussian:
            return (
                -((value - self.loc).conj() * (value - self.loc)) / (var)
                - 2.0 * log_scale
                - math.log(math.pi)
            )
        else:
            return (
                -((value - self.loc) * (value - self.loc)) / (2.0 * var)
                - log_scale
                - math.log(math.sqrt(2.0 * math.pi))
            )

    def probs(self, value):
        var = self.scale * self.scale
        if self._complex_gaussian:
            return np.exp(
                -1.0 * ((value - self.loc).conj() * (value - self.loc)) / (var)
            ) / (math.pi * var)
        else:
            return np.exp(
                -1.0 * ((value - self.loc) * (value - self.loc)) / (2.0 * var)
            ) / (math.sqrt(2 * math.pi) * self.scale)

    def entropy(self):
        if self._complex_gaussian:
            return 1.0 + np.log(math.pi) + 2.0 * np.log(self.scale)
        else:
            return (
                0.5
                + 0.5 * np.log(np.array(2.0 * math.pi).astype(self.loc.dtype))
                + np.log(self.scale)
            )

    def kl_divergence(self, other):
        var_ratio = self.scale / other.scale
        var_ratio = var_ratio * var_ratio
        t1 = (self.loc - other.loc) / other.scale
        if self._complex_gaussian:
            t1 = t1.conj() * t1
            return var_ratio + t1 - 1 - np.log(var_ratio)
        else:
            t1 = t1 * t1
            return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


class NormalTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=2, dims=3):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = base.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = base.CUDAPlace(0)
            self.gpu_id = 0

        self.batch_size = batch_size
        self.dims = dims
        self.init_numpy_data(self.batch_size, self.dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(self.batch_size, self.dims)

        paddle.enable_static()
        self.test_program = base.Program()
        with paddle.pir_utils.IrGuard():
            self.test_pir_program = paddle.static.Program()

        self.executor = base.Executor(self.place)

    def init_numpy_data(self, batch_size, dims):
        # loc ans scale are 'float'
        self.loc_np = (np.random.ranf() - 0.5) * 4
        self.scale_np = (np.random.ranf() - 0.5) * 4
        while self.scale_np < 0:
            self.scale_np = (np.random.ranf() - 0.5) * 4
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (np.random.ranf() - 0.5) * 4
        self.other_scale_np = (np.random.ranf() - 0.5) * 4
        while self.other_scale_np < 0:
            self.other_scale_np = (np.random.ranf() - 0.5) * 4
        self.values_np = np.random.ranf(1).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1], dtype='float32'
            )

    def compare_with_numpy(self, fetch_list, sample_shape=7, tolerance=1e-6):
        sample, entropy, log_prob, probs, kl = fetch_list

        np_normal = NormalNumpy(self.loc_np, self.scale_np)
        np_sample = np_normal.sample([sample_shape])
        np_entropy = np_normal.entropy()
        np_lp = np_normal.log_prob(self.values_np)
        np_p = np_normal.probs(self.values_np)
        np_other_normal = NormalNumpy(self.other_loc_np, self.other_scale_np)
        np_kl = np_normal.kl_divergence(np_other_normal)

        # Because assign op does not support the input of numpy.ndarray whose dtype is FP64.
        # When loc and scale are FP64 numpy.ndarray, we need to use assign op to convert it
        #  to FP32 Tensor. And then use cast op to convert it to a FP64 Tensor.
        # There is a loss of accuracy in this conversion.
        # So set the tolerance from 1e-6 to 1e-4.
        log_tolerance = 1e-4
        np.testing.assert_equal(sample.shape, np_sample.shape)
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            log_prob, np_lp, rtol=log_tolerance, atol=log_tolerance
        )
        np.testing.assert_allclose(
            probs, np_p, rtol=log_tolerance, atol=log_tolerance
        )
        np.testing.assert_allclose(
            kl, np_kl, rtol=log_tolerance, atol=log_tolerance
        )

    def test_normal_distribution_dygraph(self, sample_shape=7, tolerance=1e-6):
        paddle.disable_static(self.place)
        normal = Normal(self.dynamic_loc, self.dynamic_scale)

        sample = normal.sample([sample_shape]).numpy()
        entropy = normal.entropy().numpy()
        log_prob = normal.log_prob(self.dynamic_values).numpy()
        probs = normal.probs(self.dynamic_values).numpy()
        other_normal = Normal(self.dynamic_other_loc, self.dynamic_other_scale)
        kl = normal.kl_divergence(other_normal).numpy()

        fetch_list = [sample, entropy, log_prob, probs, kl]
        self.compare_with_numpy(fetch_list)

    def run_old_ir_normal_distribution_static(self, sample_shape):
        with base.program_guard(self.test_program, paddle.static.Program()):
            normal = Normal(self.static_loc, self.static_scale)

            sample = normal.sample([sample_shape])
            entropy = normal.entropy()
            log_prob = normal.log_prob(self.static_values)
            probs = normal.probs(self.static_values)
            other_normal = Normal(
                self.static_other_loc, self.static_other_scale
            )
            kl = normal.kl_divergence(other_normal)

            fetch_list = [sample, entropy, log_prob, probs, kl]

            feed_vars = {
                'loc': self.loc_np,
                'scale': self.scale_np,
                'values': self.values_np,
                'other_loc': self.other_loc_np,
                'other_scale': self.other_scale_np,
            }

            self.executor.run(base.default_startup_program())
            fetch_list = self.executor.run(
                program=self.test_program, feed=feed_vars, fetch_list=fetch_list
            )

            self.compare_with_numpy(fetch_list)

    def run_pir_normal_distribution_static(self, sample_shape):
        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(
                self.test_pir_program, paddle.static.Program()
            ):
                normal = Normal(self.static_loc, self.static_scale)

                sample = normal.sample([sample_shape])
                entropy = normal.entropy()
                log_prob = normal.log_prob(self.static_values)
                probs = normal.probs(self.static_values)
                other_normal = Normal(
                    self.static_other_loc, self.static_other_scale
                )
                kl = normal.kl_divergence(other_normal)

                fetch_list = [sample, entropy, log_prob, probs, kl]

                feed_vars = {
                    'loc': self.loc_np,
                    'scale': self.scale_np,
                    'values': self.values_np,
                    'other_loc': self.other_loc_np,
                    'other_scale': self.other_scale_np,
                }
                self.executor.run(paddle.static.default_startup_program())
                fetch_list = self.executor.run(
                    program=self.test_pir_program,
                    feed=feed_vars,
                    fetch_list=fetch_list,
                )

            self.compare_with_numpy(fetch_list)

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        self.init_static_data(self.batch_size, self.dims, in_pir=False)
        self.run_old_ir_normal_distribution_static(sample_shape)

        self.init_static_data(self.batch_size, self.dims, in_pir=True)
        self.run_pir_normal_distribution_static(sample_shape)


class ComplexNormalTest(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc is 'complex' and scale is 'float'
        m = int((np.random.ranf() - 0.5) * 4)
        self.loc_np = m + m * 1j
        self.scale_np = (np.random.ranf() - 0.5) * 4
        while self.scale_np < 0:
            self.scale_np = (np.random.ranf() - 0.5) * 4
        # used to construct another Normal object to calculate kl_divergence
        m2 = int((np.random.ranf() - 0.5) * 4)
        self.other_loc_np = m2 + m2 * 1j
        self.other_scale_np = int((np.random.ranf() - 0.5) * 4)
        while self.other_scale_np < 0:
            self.other_scale_np = int((np.random.ranf() - 0.5) * 4)
        v1 = np.random.ranf(1)
        v2 = np.random.ranf(1)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        self.init_static_data(self.batch_size, self.dims, in_pir=True)
        self.run_pir_normal_distribution_static(sample_shape)


class NormalTest2(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are 'int'
        self.loc_np = int((np.random.ranf() - 0.5) * 8)
        self.scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.scale_np < 0:
            self.scale_np = int((np.random.ranf() - 0.5) * 8)
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = int((np.random.ranf() - 0.5) * 8)
        self.other_scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.other_scale_np < 0:
            self.other_scale_np = int((np.random.ranf() - 0.5) * 8)
        self.values_np = np.random.ranf(1).astype('float32')


class ComplexNormalTest2(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc is 'complex' and scale is 'int'
        m = int((np.random.ranf() - 0.5) * 8)
        self.loc_np = m + m * 1j
        self.scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.scale_np <= 0:
            self.scale_np = int((np.random.ranf() - 0.5) * 8)
        # used to construct another Normal object to calculate kl_divergence
        m2 = int((np.random.ranf() - 0.5) * 8)
        self.other_loc_np = m2 + m2 * 1j
        self.other_scale_np = (np.random.ranf() - 0.5) * 8
        while self.other_scale_np < 0:
            self.other_scale_np = (np.random.ranf() - 0.5) * 8
        v1 = np.random.ranf(1)
        v2 = np.random.ranf(1)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        self.init_static_data(self.batch_size, self.dims, in_pir=True)
        self.run_pir_normal_distribution_static(sample_shape)


class NormalTest3(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: loc is float, scale is numpy.ndarray with dtype 'float32'.
        self.loc_np = (np.random.ranf() - 0.5) * 4
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (np.random.ranf() - 0.5) * 4
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float32'
            )


class ComplexNormalTest3(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: loc is complex, scale is numpy.ndarray with dtype 'float32'.
        m = (np.random.ranf() - 0.5) * 4
        self.loc_np = m + m * 1j
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')
        # used to construct another Normal object to calculate kl_divergence
        m2 = (np.random.ranf() - 0.5) * 4
        self.other_loc_np = m2 + m2 * 1j
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex64'
            )


class NormalTest4(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'float32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float32'
            )


class ComplexNormalTest4(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'complex64' and 'float32'.
        m = np.random.randn(batch_size, dims)
        self.loc_np = np.vectorize(complex)(m, m).astype('complex64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = np.vectorize(complex)(m2, m2).astype('complex64')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex64'
            )


class NormalTest5(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'float64'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float64'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float64'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np

        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float64'
            )


class ComplexNormalTest5(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'complex128' and 'float64'.
        m = np.random.randn(batch_size, dims)
        self.loc_np = np.vectorize(complex)(m, m).astype('complex128')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex128')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = np.vectorize(complex)(m2, m2).astype('complex128')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float64'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float64'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(
            self.values_np, dtype='complex128'
        )

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np

        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex128'
            )


class NormalTest6(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np)
        self.dynamic_scale = paddle.to_tensor(self.scale_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(self.other_loc_np)
        self.dynamic_other_scale = paddle.to_tensor(self.other_scale_np)

    def init_static_data(self, batch_size, dims, in_pir=False):
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_loc = paddle.static.data(
                name='loc', shape=[-1, dims], dtype='float32'
            )
            self.static_scale = paddle.static.data(
                name='scale', shape=[-1, dims], dtype='float32'
            )
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float32'
            )
            self.static_other_loc = paddle.static.data(
                name='other_loc', shape=[-1, dims], dtype='float32'
            )
            self.static_other_scale = paddle.static.data(
                name='other_scale', shape=[-1, dims], dtype='float32'
            )


class ComplexNormalTest6(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc is Tensor with dtype 'VarType.COMPLEX64'.
        m = np.random.randn(batch_size, dims)
        self.loc_np = np.vectorize(complex)(m, m).astype('complex64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = np.vectorize(complex)(m2, m2).astype('complex64')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np)
        self.dynamic_scale = paddle.to_tensor(self.scale_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(self.other_loc_np)
        self.dynamic_other_scale = paddle.to_tensor(self.other_scale_np)

    def init_static_data(self, batch_size, dims, in_pir=False):
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_loc = paddle.static.data(
                name='loc', shape=[-1, dims], dtype='complex64'
            )
            self.static_scale = paddle.static.data(
                name='scale', shape=[-1, dims], dtype='float32'
            )
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex64'
            )
            self.static_other_loc = paddle.static.data(
                name='other_loc', shape=[-1, dims], dtype='complex64'
            )
            self.static_other_scale = paddle.static.data(
                name='other_scale', shape=[-1, dims], dtype='float32'
            )


class NormalTest7(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP64'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float64'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float64'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='float64')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='float64'
        )
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64'
        )

    def init_static_data(self, batch_size, dims, in_pir=False):
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_loc = paddle.static.data(
                name='loc', shape=[-1, dims], dtype='float64'
            )
            self.static_scale = paddle.static.data(
                name='scale', shape=[-1, dims], dtype='float64'
            )
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float64'
            )
            self.static_other_loc = paddle.static.data(
                name='other_loc', shape=[-1, dims], dtype='float64'
            )
            self.static_other_scale = paddle.static.data(
                name='other_scale', shape=[-1, dims], dtype='float64'
            )


class ComplexNormalTest7(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.COMPLEX128'.
        m = np.random.randn(batch_size, dims)
        self.loc_np = np.vectorize(complex)(m, m).astype('complex128')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex128')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = np.vectorize(complex)(m2, m2).astype('complex128')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float64'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float64'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='complex128')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(
            self.values_np, dtype='complex128'
        )
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='complex128'
        )
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64'
        )

    def init_static_data(self, batch_size, dims, in_pir=False):
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_loc = paddle.static.data(
                name='loc', shape=[-1, dims], dtype='complex128'
            )
            self.static_scale = paddle.static.data(
                name='scale', shape=[-1, dims], dtype='float64'
            )
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex128'
            )
            self.static_other_loc = paddle.static.data(
                name='other_loc', shape=[-1, dims], dtype='complex128'
            )
            self.static_other_scale = paddle.static.data(
                name='other_scale', shape=[-1, dims], dtype='float64'
            )


class NormalTest8(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP64'. value's dtype is 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float64'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float64'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='float64')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='float64'
        )
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64'
        )

    def init_static_data(self, batch_size, dims, in_pir=False):
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_loc = paddle.static.data(
                name='loc', shape=[-1, dims], dtype='float64'
            )
            self.static_scale = paddle.static.data(
                name='scale', shape=[-1, dims], dtype='float64'
            )
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float32'
            )
            self.static_other_loc = paddle.static.data(
                name='other_loc', shape=[-1, dims], dtype='float64'
            )
            self.static_other_scale = paddle.static.data(
                name='other_scale', shape=[-1, dims], dtype='float64'
            )


class ComplexNormalTest8(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc is Tensor with dtype 'VarType.COMPLEX128'. value's dtype is 'VarType.COMPLEX64'.
        m = np.random.randn(batch_size, dims)
        self.loc_np = np.vectorize(complex)(m, m).astype('complex128')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = np.vectorize(complex)(m2, m2).astype('complex128')
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float64'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float64'
            )

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='complex128')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='complex128'
        )
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64'
        )

    def init_static_data(self, batch_size, dims, in_pir=False):
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_loc = paddle.static.data(
                name='loc', shape=[-1, dims], dtype='complex128'
            )
            self.static_scale = paddle.static.data(
                name='scale', shape=[-1, dims], dtype='float64'
            )
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex64'
            )
            self.static_other_loc = paddle.static.data(
                name='other_loc', shape=[-1, dims], dtype='complex128'
            )
            self.static_other_scale = paddle.static.data(
                name='other_scale', shape=[-1, dims], dtype='float64'
            )


class NormalTest9(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are list.
        self.loc_np = (
            np.random.randn(batch_size, dims).astype('float32').tolist()
        )
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = self.scale_np.tolist()
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (
            np.random.randn(batch_size, dims).astype('float32').tolist()
        )
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )
        self.other_scale_np = self.other_scale_np.tolist()

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float32'
            )


class ComplexNormalTest9(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are list.
        m = np.random.randn(batch_size, dims)
        self.loc_np = np.vectorize(complex)(m, m).astype('complex64').tolist()
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = self.scale_np.tolist()
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = (
            np.vectorize(complex)(m2, m2).astype('complex64').tolist()
        )
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )
        self.other_scale_np = self.other_scale_np.tolist()

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex64'
            )

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        self.init_static_data(self.batch_size, self.dims, in_pir=True)
        self.run_pir_normal_distribution_static(sample_shape)


class NormalTest10(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are tuple.
        self.loc_np = tuple(
            np.random.randn(batch_size, dims).astype('float32').tolist()
        )
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = tuple(self.scale_np.tolist())
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = tuple(
            np.random.randn(batch_size, dims).astype('float32').tolist()
        )
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )
        self.other_scale_np = tuple(self.other_scale_np.tolist())

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='float32'
            )


class ComplexNormalTest10(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are tuple.
        m = np.random.randn(batch_size, dims)
        self.loc_np = tuple(
            np.vectorize(complex)(m, m).astype('complex64').tolist()
        )
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = tuple(self.scale_np.tolist())
        v1 = np.random.randn(batch_size, dims)
        v2 = np.random.randn(batch_size, dims)
        self.values_np = np.vectorize(complex)(v1, v2).astype('complex64')
        # used to construct another Normal object to calculate kl_divergence
        m2 = np.random.randn(batch_size, dims)
        self.other_loc_np = tuple(
            np.vectorize(complex)(m2, m2).astype('complex64').tolist()
        )
        self.other_scale_np = np.random.randn(batch_size, dims).astype(
            'float32'
        )
        while not np.all(self.other_scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size, dims).astype(
                'float32'
            )
        self.other_scale_np = tuple(self.other_scale_np.tolist())

    def init_static_data(self, batch_size, dims, in_pir=False):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        manager = InitDataContextManager(
            in_pir, self.test_pir_program if in_pir else self.test_program
        )
        with manager as mgr:
            self.static_values = paddle.static.data(
                name='values', shape=[-1, dims], dtype='complex64'
            )

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        self.init_static_data(self.batch_size, self.dims, in_pir=True)
        self.run_pir_normal_distribution_static(sample_shape)


def kstest(loc, scale, samples):
    # Uses the Kolmogorov-Smirnov test for goodness of fit.
    ks, _ = scipy.stats.kstest(
        samples, scipy.stats.norm(loc=loc, scale=scale).cdf
    )
    return ks < 0.02


def make_complex_normal_loc(mean, dtype='complex64'):
    return np.vectorize(complex)(mean, mean).astype(dtype)


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale'),
    [
        ('sample', xrand((4,)), xrand((4,))),
        ('complex-sample', make_complex_normal_loc(xrand((4,))), xrand((4,))),
    ],
)
class TestNormalSampleDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.paddle_normal = Normal(loc=self.loc, scale=self.scale)
        n = 100000
        self.sample_shape = (n,)
        self.samples = self.paddle_normal.sample(self.sample_shape).numpy()
        self._complex_normal = self.loc.dtype in [np.complex64, np.complex128]

    def test_sample(self):
        samples_mean = self.samples.mean(axis=0)
        samples_var = self.samples.var(axis=0)
        np.testing.assert_allclose(
            samples_mean, self.paddle_normal.mean, rtol=0.1, atol=0
        )
        np.testing.assert_allclose(
            samples_var, self.paddle_normal.variance, rtol=0.1, atol=0
        )
        if self._complex_normal:
            samples_var_real = self.samples.real.var(axis=0)
            samples_var_imag = self.samples.imag.var(axis=0)
            np.testing.assert_allclose(
                samples_var_real,
                self.paddle_normal.variance / 2.0,
                rtol=0.1,
                atol=0,
            )
            np.testing.assert_allclose(
                samples_var_imag,
                self.paddle_normal.variance / 2.0,
                rtol=0.1,
                atol=0,
            )

        batch_shape = (self.loc + self.scale).shape
        self.assertEqual(self.samples.shape, self.sample_shape + batch_shape)

        if not self._complex_normal:
            for i in range(len(self.scale)):
                self.assertTrue(
                    kstest(self.loc[i], self.scale[i], self.samples[:, i])
                )
        else:
            for i in range(len(self.scale)):
                var_i = self.scale[i] ** 2
                self.assertTrue(
                    kstest(
                        self.loc[i].real,
                        np.sqrt(var_i / 2.0),
                        self.samples[:, i].real,
                    )
                )
                self.assertTrue(
                    kstest(
                        self.loc[i].imag,
                        np.sqrt(var_i / 2.0),
                        self.samples[:, i].imag,
                    )
                )


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale'),
    [
        ('sample', xrand((4,)), xrand((4,))),
        ('complex-sample', make_complex_normal_loc(xrand((4,))), xrand((4,))),
    ],
    test_pir=True,
)
class TestNormalSampleStatic(unittest.TestCase):
    def build_program(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            n = 100000
            self.sample_shape = (n,)
            self.paddle_normal = Normal(loc=loc, scale=scale)
            mean = self.paddle_normal.mean
            variance = self.paddle_normal.variance
            samples = self.paddle_normal.sample(self.sample_shape)
        fetch_list = [mean, variance, samples]
        self.feeds = {'loc': self.loc, 'scale': self.scale}

        executor.run(startup_program)
        [self.mean, self.variance, self.samples] = executor.run(
            main_program, feed=self.feeds, fetch_list=fetch_list
        )
        self._complex_normal = self.loc.dtype in [np.complex64, np.complex128]

    def setUp(self):
        if self.test_pir:
            with paddle.pir_utils.IrGuard():
                self.build_program()
        else:
            self.build_program()

    def test_sample(self):
        samples_mean = self.samples.mean(axis=0)
        samples_var = self.samples.var(axis=0)
        np.testing.assert_allclose(samples_mean, self.mean, rtol=0.1, atol=0)
        np.testing.assert_allclose(samples_var, self.variance, rtol=0.1, atol=0)
        if self._complex_normal:
            samples_var_real = self.samples.real.var(axis=0)
            samples_var_imag = self.samples.imag.var(axis=0)
            np.testing.assert_allclose(
                samples_var_real, self.variance / 2.0, rtol=0.1, atol=0
            )
            np.testing.assert_allclose(
                samples_var_imag, self.variance / 2.0, rtol=0.1, atol=0
            )

        batch_shape = (self.loc + self.scale).shape
        self.assertEqual(self.samples.shape, self.sample_shape + batch_shape)

        if not self._complex_normal:
            for i in range(len(self.scale)):
                self.assertTrue(
                    kstest(self.loc[i], self.scale[i], self.samples[:, i])
                )
        else:
            for i in range(len(self.scale)):
                var_i = self.scale[i] ** 2
                self.assertTrue(
                    kstest(
                        self.loc[i].real,
                        np.sqrt(var_i / 2.0),
                        self.samples[:, i].real,
                    )
                )
                self.assertTrue(
                    kstest(
                        self.loc[i].imag,
                        np.sqrt(var_i / 2.0),
                        self.samples[:, i].imag,
                    )
                )


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale'),
    [
        ('rsample', xrand((4,)), xrand((4,))),
        ('complex-sample', make_complex_normal_loc(xrand((4,))), xrand((4,))),
    ],
)
class TestNormalRSampleDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self._complex_normal = self.loc.dtype in [np.complex64, np.complex128]
        self.loc = paddle.to_tensor(self.loc)
        self.scale = paddle.to_tensor(self.scale)
        self.loc.stop_gradient = False
        self.scale.stop_gradient = False
        self.paddle_normal = Normal(loc=self.loc, scale=self.scale)
        n = 100000
        self.rsample_shape = [n]
        self.rsamples = self.paddle_normal.rsample(self.rsample_shape)

    def test_rsample(self):
        rsamples_mean = self.rsamples.mean(axis=0)
        rsamples_var = self.rsamples.numpy().var(axis=0)
        np.testing.assert_allclose(
            rsamples_mean, self.paddle_normal.mean, rtol=0.1, atol=0
        )
        np.testing.assert_allclose(
            rsamples_var, self.paddle_normal.variance, rtol=0.1, atol=0
        )
        if self._complex_normal:
            samples_var_real = self.rsamples.real().var(axis=0)
            samples_var_imag = self.rsamples.imag().var(axis=0)
            np.testing.assert_allclose(
                samples_var_real,
                self.paddle_normal.variance / 2.0,
                rtol=0.1,
                atol=0,
            )
            np.testing.assert_allclose(
                samples_var_imag,
                self.paddle_normal.variance / 2.0,
                rtol=0.1,
                atol=0,
            )

        batch_shape = (self.loc + self.scale).shape
        self.assertEqual(self.rsamples.shape, self.rsample_shape + batch_shape)

        if not self._complex_normal:
            for i in range(len(self.scale)):
                self.assertTrue(
                    kstest(self.loc[i], self.scale[i], self.rsamples[:, i])
                )
        else:
            for i in range(len(self.scale)):
                var_i = self.scale[i].numpy() ** 2
                self.assertTrue(
                    kstest(
                        self.loc[i].real().numpy(),
                        np.sqrt(var_i / 2.0),
                        self.rsamples[:, i].real().numpy(),
                    )
                )
                self.assertTrue(
                    kstest(
                        self.loc[i].imag().numpy(),
                        np.sqrt(var_i / 2.0),
                        self.rsamples[:, i].imag().numpy(),
                    )
                )

    def test_backpropagation(self):
        grads = paddle.grad([self.rsamples], [self.loc, self.scale])
        self.assertEqual(len(grads), 2)
        self.assertEqual(grads[0].dtype, self.loc.dtype)
        self.assertEqual(grads[0].shape, self.loc.shape)
        self.assertEqual(grads[1].dtype, self.scale.dtype)
        self.assertEqual(grads[1].shape, self.scale.shape)


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale'),
    [
        ('rsample', xrand((4,)), xrand((4,))),
        ('complex-sample', make_complex_normal_loc(xrand((4,))), xrand((4,))),
    ],
    test_pir=True,
)
class TestNormalRSampleStatic(unittest.TestCase):
    def build_program(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            n = 100000
            self.rsample_shape = (n,)
            self.paddle_normal = Normal(loc=loc, scale=scale)
            mean = self.paddle_normal.mean
            variance = self.paddle_normal.variance
            rsamples = self.paddle_normal.rsample(self.rsample_shape)
        fetch_list = [mean, variance, rsamples]
        self.feeds = {'loc': self.loc, 'scale': self.scale}

        executor.run(startup_program)
        [self.mean, self.variance, self.rsamples] = executor.run(
            main_program, feed=self.feeds, fetch_list=fetch_list
        )
        self._complex_normal = self.loc.dtype in [np.complex64, np.complex128]

    def setUp(self):
        if self.test_pir:
            with paddle.pir_utils.IrGuard():
                self.build_program()
        else:
            self.build_program()

    def test_rsample(self):
        rsamples_mean = self.rsamples.mean(axis=0)
        rsamples_var = self.rsamples.var(axis=0)
        np.testing.assert_allclose(rsamples_mean, self.mean, rtol=0.1, atol=0)
        np.testing.assert_allclose(
            rsamples_var, self.variance, rtol=0.1, atol=0
        )
        if self._complex_normal:
            samples_var_real = self.rsamples.real.var(axis=0)
            samples_var_imag = self.rsamples.imag.var(axis=0)
            np.testing.assert_allclose(
                samples_var_real, self.variance / 2.0, rtol=0.1, atol=0
            )
            np.testing.assert_allclose(
                samples_var_imag, self.variance / 2.0, rtol=0.1, atol=0
            )

        batch_shape = (self.loc + self.scale).shape
        self.assertEqual(self.rsamples.shape, self.rsample_shape + batch_shape)

        if not self._complex_normal:
            for i in range(len(self.scale)):
                self.assertTrue(
                    kstest(self.loc[i], self.scale[i], self.rsamples[:, i])
                )
        else:
            for i in range(len(self.scale)):
                var_i = self.scale[i] ** 2
                self.assertTrue(
                    kstest(
                        self.loc[i].real,
                        np.sqrt(var_i / 2.0),
                        self.rsamples[:, i].real,
                    )
                )
                self.assertTrue(
                    kstest(
                        self.loc[i].imag,
                        np.sqrt(var_i / 2.0),
                        self.rsamples[:, i].imag,
                    )
                )


if __name__ == '__main__':
    unittest.main()
