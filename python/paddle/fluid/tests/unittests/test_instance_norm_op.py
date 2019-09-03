#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
from op_test import OpTest


def _reference_instance_norm_naive(x, scale, bias, epsilon, momentum,
                                   use_global_stats, mean, var):
    x_shape = x.shape
    if len(x_shape) == 2:
        x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
    n, c, h, w = x.shape
    mean, var = mean, var

    if use_global_stats:
        mean = np.tile(mean, n)
        var = np.tile(var, n)

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    x_norm = (x - mean_tile) / np.sqrt(var_tile + epsilon).astype('float32')
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))
    bias_tile = np.reshape(bias, (1, c, 1, 1))
    bias_tile = np.tile(bias_tile, (n, 1, h, w))
    y = scale_tile * x_norm + bias_tile
    if len(x_shape) == 2:
        y = np.reshape(y, x_shape)
    return y, mean, var


def _reference_instance_norm_test(x, scale, bias, epsilon):
    x_shape = x.shape
    if len(x_shape) == 2:
        x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
    n, c, h, w = x.shape
    mean_tile = np.zeros(x.shape).astype(np.float32)
    var_tile = np.ones(x.shape).astype(np.float32)

    x_norm = (x - mean_tile) / np.sqrt(var_tile + epsilon)
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))
    bias_tile = np.reshape(bias, (1, c, 1, 1))
    bias_tile = np.tile(bias_tile, (n, 1, h, w))
    y = scale_tile * x_norm + bias_tile
    if len(x_shape) == 2:
        y = np.reshape(y, x_shape)
    return y


def _reference_instance_norm_grad(x, d_y, scale, mean, var, epsilon,
                                  use_global_stats):
    # d_scale = sum(d_y * (x-mean) / sqrt(var+epsilon))
    # d_offset = sum(d_y)
    # d_x = scale / sqrt(var+epsilon) * (d_y - np.mean(d_y, axis=(2,3)) - (x-mean)/sqrt(var+epsilon)* np.mean(y_grad * (x-mean)/sqrt(var+epsilon), axis=(2,3)))
    n, c, h, w = x.shape

    if use_global_stats:
        mean = np.tile(mean, n)
        var = 1. / np.sqrt(var + epsilon)
        var = np.tile(var, n)

    d_bias = np.sum(d_y, axis=(0, 2, 3))

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    d_scale = np.sum(d_y * (x - mean_tile) * var_tile, axis=(0, 2, 3))
    var_inv = var_tile
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))

    if use_global_stats:
        d_x = scale_tile * var_inv * d_y
    else:
        d_x = scale_tile * var_inv * (
            d_y - np.mean(
                d_y, axis=(2, 3), keepdims=True) -
            (x - mean_tile) * var_inv * np.mean(
                d_y * (x - mean_tile) * var_inv, axis=(2, 3), keepdims=True))
    return d_x, d_scale, d_bias


def _cal_mean_variance(x, epsilon, mean_shape):
    mean = np.reshape(np.mean(x, axis=(2, 3)), mean_shape)
    var = np.reshape(np.var(x, axis=(2, 3)), mean_shape)
    return mean, var


class TestInstanceNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.momentum = 0.9
        self.epsilon = 1e-5
        self.init_test_case()

    def init_test_case(self):
        ### if self.use_global_stats = True in training, fetch_list need to delete saved_mean and saved_variance
        self.use_global_stats = False
        self.no_grad_set = set()
        if self.use_global_stats:
            self.fetch_list = [
                'y', 'mean', 'variance', 'x@GRAD', 'scale@GRAD', 'bias@GRAD'
            ]
        else:
            self.fetch_list = [
                'y', 'mean', 'variance', 'saved_mean', 'saved_variance',
                'x@GRAD', 'scale@GRAD', 'bias@GRAD'
            ]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def set_global_mean_var(self, mean_shape, x):
        mean, variance = _cal_mean_variance(x, self.epsilon, mean_shape)
        mean_pre = np.zeros(mean_shape).astype(np.float32)
        variance_pre = np.ones(mean_shape).astype(np.float32)

        ## compute global mean and variance
        if self.use_global_stats:
            _, _, h, w = x.shape
            momentum = self.momentum
            mean = (1. - momentum) * mean + momentum * mean_pre
            unbias_var = variance * (h * w) / (h * w - 1)  ## ubbias variance
            variance = (1. - momentum) * variance + momentum * variance_pre
        return mean, variance

    def test_forward_backward(self):
        def test_with_place(place, shape):
            epsilon = self.epsilon
            momentum = self.momentum
            n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]
            mean_shape = [n * c]

            np.random.seed(137)
            x = np.random.random_sample(shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean, variance = self.set_global_mean_var(mean_shape, x)
            d_y = np.random.random_sample(shape).astype(np.float32)

            if self.use_global_stats:
                saved_mean = np.zeros(n * c).astype(np.float32)
                saved_variance = np.zeros(n * c).astype(np.float32)

                mean = np.reshape(mean, (n, c))
                mean = np.mean(mean, 0)
                variance = np.reshape(variance, (n, c))
                variance = np.mean(variance, 0)

            y, saved_mean, variance_tmp = _reference_instance_norm_naive(
                x, scale, bias, epsilon, self.momentum, self.use_global_stats,
                mean, variance)

            if self.use_global_stats == False:
                mean_out = np.zeros(mean_shape).astype(np.float32)
                variance_out = np.ones(mean_shape).astype(np.float32)

                mean_out = saved_mean * (1. - momentum) + momentum * mean_out
                unbias_var = variance_tmp * (h * w) / (
                    h * w - 1)  ## ubbias variance
                variance_out = unbias_var * (1. - momentum
                                             ) + momentum * variance_out

                saved_variance = 1 / np.sqrt(variance_tmp + epsilon)

                mean = np.reshape(mean_out, (n, c))
                mean = np.mean(mean, 0)
                variance = np.reshape(variance_out, (n, c))
                variance = np.mean(variance, 0)

            if self.use_global_stats:
                d_x, d_scale, d_bias = _reference_instance_norm_grad(
                    x, d_y, scale, mean, variance, epsilon,
                    self.use_global_stats)
            else:
                d_x, d_scale, d_bias = _reference_instance_norm_grad(
                    x, d_y, scale, saved_mean, saved_variance, epsilon,
                    self.use_global_stats)

            var_dict = locals()
            var_dict['y@GRAD'] = d_y
            var_dict['x@GRAD'] = d_x
            var_dict['scale@GRAD'] = d_scale
            var_dict['bias@GRAD'] = d_bias

            var_names = [
                'x', 'scale', 'bias', 'mean', 'variance', 'y', 'saved_mean',
                'saved_variance'
            ]
            ground_truth = {name: var_dict[name] for name in var_names}

            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape)
                in_op = block.append_op(
                    type="instance_norm",
                    inputs={
                        "X": block.var("x"),
                        "Scale": block.var("scale"),
                        "Bias": block.var("bias"),
                        "Mean": block.var("mean"),
                        "Variance": block.var("variance")
                    },
                    outputs={
                        "Y": block.var("y"),
                        "MeanOut": block.var("mean"),
                        "VarianceOut": block.var("variance"),
                        "SavedMean": block.var("saved_mean"),
                        "SavedVariance": block.var("saved_variance")
                    },
                    attrs={
                        "momentum": momentum,
                        "epsilon": epsilon,
                        "is_test": False,
                        "use_global_stats": self.use_global_stats
                    })

                block.create_var(name="y@GRAD", dtype='float32', shape=y.shape)

                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    in_op.desc, self.no_grad_set, [])
                grad_op_desc = grad_op_desc_list[0]
                new_op_desc = block.desc.append_op()
                new_op_desc.copy_from(grad_op_desc)
                for var_name in grad_op_desc.output_arg_names():
                    block.desc.var(var_name.encode("ascii"))
                grad_op_desc.infer_var_type(block.desc)
                grad_op_desc.infer_shape(block.desc)
                for arg in grad_op_desc.output_arg_names():
                    grad_var = block.desc.find_var(arg.encode("ascii"))
                    grad_var.set_dtype(core.VarDesc.VarType.FP32)

                exe = fluid.Executor(place)
                out = exe.run(
                    program,
                    feed={
                        name: var_dict[name]
                        for name in
                        ['x', 'scale', 'bias', 'mean', 'variance', 'y@GRAD']
                    },
                    fetch_list=self.fetch_list)

            for id, name in enumerate(self.fetch_list):
                self.__assert_close(var_dict[name], out[id], name)
            print("op test forward passes: ", str(place))

        places = [core.CPUPlace()]

        if core.is_compiled_with_cuda() and core.op_support_gpu(
                "instance_norm"):
            places.append(core.CUDAPlace(0))
        for place in places:
            test_with_place(place, [2, 3, 4, 5])


class TestCase1(TestInstanceNormOpTraining):
    def init_test_case(self):
        ### if self.use_global_stats = True in training, fetch_list need to delete saved_mean and saved_variance
        self.use_global_stats = True
        self.no_grad_set = set()
        if self.use_global_stats:
            self.fetch_list = [
                'y', 'mean', 'variance', 'x@GRAD', 'scale@GRAD', 'bias@GRAD'
            ]
        else:
            self.fetch_list = [
                'y', 'mean', 'variance', 'saved_mean', 'saved_variance',
                'x@GRAD', 'scale@GRAD', 'bias@GRAD'
            ]


class TestInstanceNormOpInference(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-5
        self.is_test = True
        self.momentum = 0.9

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def test_forward_backward(self):
        def test_with_place(place, shape):
            epsilon = self.epsilon
            n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]
            mean_shape = [n * c]

            np.random.seed()
            x = np.random.random_sample(shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean = np.zeros(mean_shape).astype(np.float32)
            variance = np.ones(mean_shape).astype(np.float32)

            y = _reference_instance_norm_test(x, scale, bias, epsilon)
            saved_mean = mean
            saved_variance = variance

            var_dict = locals()
            var_names = [
                'x', 'scale', 'bias', 'mean', 'variance', 'y', 'saved_mean',
                'saved_variance'
            ]
            ground_truth = {name: var_dict[name] for name in var_names}
            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape)
                in_op = block.append_op(
                    type="instance_norm",
                    inputs={
                        "X": block.var("x"),
                        "Scale": block.var("scale"),
                        "Bias": block.var("bias"),
                        "Mean": block.var("mean"),
                        "Variance": block.var("variance")
                    },
                    outputs={
                        "Y": block.var("y"),
                        "MeanOut": block.var("mean"),
                        "VarianceOut": block.var("variance"),
                        "SavedMean": block.var("saved_mean"),
                        "SavedVariance": block.var("saved_variance")
                    },
                    attrs={
                        "momentum": self.momentum,
                        "epsilon": epsilon,
                        "is_test": True
                    }),

                exe = fluid.Executor(place)
                out = exe.run(
                    program,
                    feed={
                        name: var_dict[name]
                        for name in ['x', 'scale', 'bias', 'mean', 'variance']
                    },
                    fetch_list=['y'])

                self.__assert_close(y, out[0], 'y')
                print("op test forward passes: ", str(place))

        places = [core.CPUPlace()]

        if core.is_compiled_with_cuda() and core.op_support_gpu(
                "instance_norm"):
            places.append(core.CUDAPlace(0))
        for place in places:
            test_with_place(place, [3, 7, 4, 5])


if __name__ == '__main__':
    unittest.main()
