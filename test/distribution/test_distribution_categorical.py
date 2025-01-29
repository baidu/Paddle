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

import unittest

import numpy as np
from test_distribution import DistributionNumpy

import paddle
from paddle import base
from paddle.distribution import Categorical, Distribution, Normal, Uniform

np.random.seed(2022)


class CategoricalNumpy(DistributionNumpy):
    def __init__(self, logits):
        self.logits = np.array(logits).astype('float32')

    def entropy(self):
        logits = self.logits - np.max(self.logits, axis=-1, keepdims=True)
        e_logits = np.exp(logits)
        z = np.sum(e_logits, axis=-1, keepdims=True)
        prob = e_logits / z
        return -1.0 * np.sum(prob * (logits - np.log(z)), axis=-1)

    def kl_divergence(self, other):
        logits = self.logits - np.max(self.logits, axis=-1, keepdims=True)
        other_logits = other.logits - np.max(
            other.logits, axis=-1, keepdims=True
        )
        e_logits = np.exp(logits)
        other_e_logits = np.exp(other_logits)
        z = np.sum(e_logits, axis=-1, keepdims=True)
        other_z = np.sum(other_e_logits, axis=-1, keepdims=True)
        prob = e_logits / z
        return np.sum(
            prob * (logits - np.log(z) - other_logits + np.log(other_z)),
            axis=-1,
            keepdims=True,
        )


class CategoricalTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=3, dims=5):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = base.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = base.CUDAPlace(0)
            self.gpu_id = 0

        self.batch_size = batch_size
        self.dims = dims
        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = base.Program()
        self.executor = base.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float32')
        self.other_logits_np = np.random.rand(batch_size, dims).astype(
            'float32'
        )
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        # dist_shape = logits_shape[:-1], it represents the number of
        #  different distributions.
        self.dist_shape = [batch_size]
        # sample shape represents the number of samples
        self.sample_shape = [2, 4]
        # value used in probs and log_prob method
        # If value is 1-D and logits is 2-D or higher dimension, value will be
        #  broadcasted to have the same number of distributions with logits.
        # If value is 2-D or higher dimentsion, it should have the same number
        #  of distributions with logtis. ``value[:-1] = logits[:-1]
        self.value_shape = [3]

    def init_dynamic_data(self, batch_size, dims):
        self.logits = paddle.to_tensor(self.logits_np)
        self.other_logits = paddle.to_tensor(self.other_logits_np)
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with base.program_guard(self.test_program):
            self.logits_static = paddle.static.data(
                name='logits', shape=self.logits_shape, dtype='float32'
            )
            self.other_logits_static = paddle.static.data(
                name='other_logits', shape=self.logits_shape, dtype='float32'
            )
            self.value_static = paddle.static.data(
                name='value', shape=self.value_shape, dtype='int64'
            )

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.dist_shape + self.value_shape)
        for i in range(self.batch_size):
            for j in range(3):
                np_probs[i][j] = probability[i][self.value_np[j]]
        return np_probs

    def compare_with_numpy(self, fetch_list, tolerance=1e-6):
        sample, entropy, kl, probs, log_prob = fetch_list
        log_tolerance = 1e-4

        np.testing.assert_equal(
            sample.shape, self.sample_shape + self.dist_shape
        )

        np_categorical = CategoricalNumpy(self.logits_np)
        np_other_categorical = CategoricalNumpy(self.other_logits_np)
        np_entropy = np_categorical.entropy()
        np_kl = np_categorical.kl_divergence(np_other_categorical)

        np.testing.assert_allclose(
            entropy, np_entropy, rtol=log_tolerance, atol=log_tolerance
        )
        np.testing.assert_allclose(
            kl, np_kl, rtol=log_tolerance, atol=log_tolerance
        )

        probability = np.exp(self.logits_np) / np.sum(
            np.exp(self.logits_np), axis=-1, keepdims=True
        )
        np_probs = self.get_numpy_selected_probs(probability)
        np_log_prob = np.log(np_probs)

        np.testing.assert_allclose(
            probs, np_probs, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            log_prob, np_log_prob, rtol=tolerance, atol=tolerance
        )

    def test_categorical_distribution_dygraph(self, tolerance=1e-6):
        paddle.disable_static(self.place)
        categorical = Categorical(self.logits)
        other_categorical = Categorical(self.other_logits)

        sample = categorical.sample(self.sample_shape).numpy()
        entropy = categorical.entropy().numpy()
        kl = categorical.kl_divergence(other_categorical).numpy()
        probs = categorical.probs(self.value).numpy()
        log_prob = categorical.log_prob(self.value).numpy()

        fetch_list = [sample, entropy, kl, probs, log_prob]
        self.compare_with_numpy(fetch_list)

    def test_categorical_distribution_static(self, tolerance=1e-6):
        paddle.enable_static()
        with base.program_guard(self.test_program):
            categorical = Categorical(self.logits_static)
            other_categorical = Categorical(self.other_logits_static)

            sample = categorical.sample(self.sample_shape)
            entropy = categorical.entropy()
            kl = categorical.kl_divergence(other_categorical)
            probs = categorical.probs(self.value_static)
            log_prob = categorical.log_prob(self.value_static)

            fetch_list = [sample, entropy, kl, probs, log_prob]

        feed_vars = {
            'logits': self.logits_np,
            'other_logits': self.other_logits_np,
            'value': self.value_np,
        }

        self.executor.run(base.default_startup_program())
        fetch_list = self.executor.run(
            program=self.test_program, feed=feed_vars, fetch_list=fetch_list
        )

        self.compare_with_numpy(fetch_list)


class CategoricalTest2(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor with dtype Float64
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float64')
        self.other_logits_np = np.random.rand(batch_size, dims).astype(
            'float64'
        )
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        self.dist_shape = [batch_size]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def init_static_data(self, batch_size, dims):
        with base.program_guard(self.test_program):
            self.logits_static = paddle.static.data(
                name='logits', shape=self.logits_shape, dtype='float64'
            )
            self.other_logits_static = paddle.static.data(
                name='other_logits', shape=self.logits_shape, dtype='float64'
            )
            self.value_static = paddle.static.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class CategoricalTest3(CategoricalTest):
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D numpy.ndarray with dtype Float32
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = self.logits_np
        self.other_logits = self.other_logits_np
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with base.program_guard(self.test_program):
            self.logits_static = self.logits_np
            self.other_logits_static = self.other_logits_np
            self.value_static = paddle.static.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class CategoricalTest4(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D numpy.ndarray with dtype Float64
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float64')
        self.other_logits_np = np.random.rand(batch_size, dims).astype(
            'float64'
        )
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        self.dist_shape = [batch_size]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def init_dynamic_data(self, batch_size, dims):
        self.logits = self.logits_np
        self.other_logits = self.other_logits_np
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with base.program_guard(self.test_program):
            self.logits_static = self.logits_np
            self.other_logits_static = self.other_logits_np
            self.value_static = paddle.static.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


# test shape of logits and value used in probs and log_prob method
class CategoricalTest5(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 1-D Tensor
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(dims).astype('float32')
        self.other_logits_np = np.random.rand(dims).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [dims]
        self.dist_shape = []
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.value_shape)
        for i in range(3):
            np_probs[i] = probability[self.value_np[i]]
        return np_probs


class CategoricalTest6(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor
        # value used in probs and log_prob method has the same number of batches with input
        self.logits_np = np.random.rand(3, 5).astype('float32')
        self.other_logits_np = np.random.rand(3, 5).astype('float32')
        self.value_np = np.array([[2, 1], [0, 3], [2, 3]]).astype('int64')

        self.logits_shape = [3, 5]
        self.dist_shape = [3]
        self.sample_shape = [2, 4]
        self.value_shape = [3, 2]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.value_shape)
        for i in range(3):
            for j in range(2):
                np_probs[i][j] = probability[i][self.value_np[i][j]]
        return np_probs


class CategoricalTest7(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 3-D Tensor
        # value used in probs and log_prob method has the same number of distributions with input
        self.logits_np = np.random.rand(3, 2, 5).astype('float32')
        self.other_logits_np = np.random.rand(3, 2, 5).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [3, 2, 5]
        self.dist_shape = [3, 2]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.dist_shape + self.value_shape)
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    np_probs[i][j][k] = probability[i][j][self.value_np[k]]
        return np_probs


class CategoricalTest8(CategoricalTest):
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D list
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = self.logits_np.tolist()
        self.other_logits = self.other_logits_np.tolist()
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with base.program_guard(self.test_program):
            self.logits_static = self.logits_np.tolist()
            self.other_logits_static = self.other_logits_np.tolist()
            self.value_static = paddle.static.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class CategoricalTest9(CategoricalTest):
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D tuple
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = tuple(self.logits_np.tolist())
        self.other_logits = tuple(self.other_logits_np.tolist())
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with base.program_guard(self.test_program):
            self.logits_static = tuple(self.logits_np.tolist())
            self.other_logits_static = tuple(self.other_logits_np.tolist())
            self.value_static = paddle.static.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class DistributionTestError(unittest.TestCase):
    def test_distribution_error(self):
        distribution = Distribution()

        self.assertRaises(NotImplementedError, distribution.sample)
        self.assertRaises(NotImplementedError, distribution.entropy)

        normal = Normal(0.0, 1.0)
        self.assertRaises(
            NotImplementedError, distribution.kl_divergence, normal
        )

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = paddle.tensor.create_tensor(dtype="float32")
        self.assertRaises(
            NotImplementedError, distribution.log_prob, value_tensor
        )
        self.assertRaises(NotImplementedError, distribution.probs, value_tensor)

    def test_normal_error(self):
        paddle.enable_static()
        normal = Normal(0.0, 1.0)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, normal.log_prob, value)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, normal.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, normal.sample, shape)

        seed = 1.0
        # type of seed must be int
        self.assertRaises(TypeError, normal.sample, [2, 3], seed)

        normal_other = Uniform(1.0, 2.0)
        # type of other must be an instance of Normal
        self.assertRaises(TypeError, normal.kl_divergence, normal_other)

    def test_uniform_error(self):
        paddle.enable_static()
        uniform = Uniform(0.0, 1.0)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, uniform.log_prob, value)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, uniform.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, uniform.sample, shape)

        seed = 1.0
        # type of seed must be int
        self.assertRaises(TypeError, uniform.sample, [2, 3], seed)

    def test_categorical_error(self):
        paddle.enable_static()

        categorical = Categorical([0.4, 0.6])

        value = [1, 0]
        # type of value must be variable
        self.assertRaises(AttributeError, categorical.log_prob, value)

        value = [1, 0]
        # type of value must be variable
        self.assertRaises(AttributeError, categorical.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, categorical.sample, shape)

        categorical_other = Uniform(1.0, 2.0)
        # type of other must be an instance of Categorical
        self.assertRaises(
            TypeError, categorical.kl_divergence, categorical_other
        )

        def test_shape_not_match_error():
            # shape of value must match shape of logits
            # value_shape[:-1] == logits_shape[:-1]
            paddle.disable_static()
            logits = paddle.rand([3, 5])
            cat = Categorical(logits)
            value = paddle.to_tensor([[2, 1, 3], [3, 2, 1]], dtype='int64')
            cat.log_prob(value)

        self.assertRaises(ValueError, test_shape_not_match_error)


if __name__ == '__main__':
    unittest.main()
