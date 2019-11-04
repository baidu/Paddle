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

import numpy as np
import unittest
from py_precise_roi_pool import PyPrRoIPool
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard


class TestPRROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.prRoIPool = PyPrRoIPool()
        self.outs = self.prRoIPool.compute(
            self.x, self.rois, self.output_channels, self.spatial_scale,
            self.pooled_height, self.pooled_width).astype('float32')
        self.inputs = {'X': self.x, 'ROIs': (self.rois[:, 1:5], self.rois_lod)}
        self.attrs = {
            'output_channels': self.output_channels,
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width
        }
        self.outputs = {'Out': self.outs}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 26
        self.width = 24

        self.x_dim = [self.batch_size, self.channels, self.height, self.width]

        self.spatial_scale = 1.0 / 4.0
        self.output_channels = self.channels
        self.pooled_height = 2
        self.pooled_width = 2

        self.x = np.random.random(self.x_dim).astype('float32')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.uniform(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y1 = np.random.uniform(
                    0, self.height // self.spatial_scale - self.pooled_height)

                x2 = np.random.uniform(x1 + self.pooled_width,
                                       self.width // self.spatial_scale)
                y2 = np.random.uniform(y1 + self.pooled_height,
                                       self.height // self.spatial_scale)
                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype('float32')

    def setUp(self):
        self.op_type = 'prroi_pool'
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_backward(self):
        self.check_grad(['X'], 'Out', max_relative_error=1.0)

    def run_net(self, place):
        with program_guard(Program(), Program()):
            x = fluid.layers.data(
                name="X",
                shape=[self.channels, self.height, self.width],
                dtype="float32")
            rois = fluid.layers.data(
                name="ROIs", shape=[4], dtype="float32", lod_level=1)
            output = fluid.layers.prroi_pool(x, rois, 0.25, 2, 2)
            loss = fluid.layers.mean(output)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)
            input_x = fluid.create_lod_tensor(self.x, [], place)
            input_rois = fluid.create_lod_tensor(self.rois[:, 1:5],
                                                 self.rois_lod, place)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            exe.run(fluid.default_main_program(),
                    {'X': input_x,
                     "ROIs": input_rois})

    def test_net(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.run_net(place)

    def test_errors(self):
        with program_guard(Program(), Program()):
            x = fluid.layers.data(
                name="x", shape=[245, 30, 30], dtype="float32")
            rois = fluid.layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            # spatial_scale must be float type
            self.assertRaises(TypeError, fluid.layers.prroi_pool, x, rois, 2, 7,
                              7)
            # pooled_height must be int type
            self.assertRaises(TypeError, fluid.layers.prroi_pool, x, rois, 0.25,
                              0.7, 7)
            # pooled_width must be int type
            self.assertRaises(TypeError, fluid.layers.prroi_pool, x, rois, 0.25,
                              7, 0.7)


class TestPRROIPoolOpTensorRoIs(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.prRoIPool = PyPrRoIPool()
        self.outs = self.prRoIPool.compute(
            self.x, self.rois, self.output_channels, self.spatial_scale,
            self.pooled_height, self.pooled_width).astype('float64')

        self.rois_index = np.array(self.rois_lod).reshape([-1]).astype(np.int64)
        self.inputs = {
            'X': self.x,
            'ROIs': self.rois[:, 1:5],
            'BatchRoINums': self.rois_index
        }
        self.attrs = {
            'output_channels': self.output_channels,
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width
        }
        self.outputs = {'Out': self.outs}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 26
        self.width = 24

        self.x_dim = [self.batch_size, self.channels, self.height, self.width]

        self.spatial_scale = 1.0 / 4.0
        self.output_channels = self.channels
        self.pooled_height = 2
        self.pooled_width = 2

        self.x = np.random.random(self.x_dim).astype('float64')

    def make_rois(self):
        rois = []
        self.rois_lod = []
        for bno in range(self.batch_size):
            self.rois_lod.append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.uniform(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y1 = np.random.uniform(
                    0, self.height // self.spatial_scale - self.pooled_height)

                x2 = np.random.uniform(x1 + self.pooled_width,
                                       self.width // self.spatial_scale)
                y2 = np.random.uniform(y1 + self.pooled_height,
                                       self.height // self.spatial_scale)
                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype('float64')

    def setUp(self):
        self.op_type = 'prroi_pool'
        self.set_data()

    def test_check_output(self):
        self.check_output()
        place = fluid.CPUPlace()
        self.check_output_with_place(place, 1e-5, None, False, False)
        place = fluid.CUDAPlace(0)
        self.check_output_with_place(place, 1e-5, None, False, False)

    def test_backward(self):
        place = fluid.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X'],
            'Out',
            max_relative_error=0.005,
            numeric_grad_delta=0.005)

    def run_net(self, place):
        with program_guard(Program(), Program()):
            x = fluid.layers.data(
                name="X",
                shape=[self.channels, self.height, self.width],
                dtype="float64")
            rois = fluid.layers.data(name="ROIs", shape=[4], dtype="float64")
            rois_index = fluid.layers.data(
                name='rois_idx', shape=[], dtype="int64")
            output = fluid.layers.prroi_pool(
                x, rois, 0.25, 2, 2, batch_roi_nums=rois_index)
            loss = fluid.layers.mean(output)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            exe.run(fluid.default_main_program(), {
                'X': self.x,
                "ROIs": self.rois[:, 1:5],
                "rois_idx": self.rois_index
            })

    def test_net(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self.run_net(place)

    def test_errors(self):
        with program_guard(Program(), Program()):
            x = fluid.layers.data(
                name="x", shape=[245, 30, 30], dtype="float32")
            rois = fluid.layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            # spatial_scale must be float type
            self.assertRaises(TypeError, fluid.layers.prroi_pool, x, rois, 2, 7,
                              7)
            # pooled_height must be int type
            self.assertRaises(TypeError, fluid.layers.prroi_pool, x, rois, 0.25,
                              0.7, 7)
            # pooled_width must be int type
            self.assertRaises(TypeError, fluid.layers.prroi_pool, x, rois, 0.25,
                              7, 0.7)


if __name__ == '__main__':
    unittest.main()
