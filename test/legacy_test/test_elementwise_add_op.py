#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import unittest
import warnings

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
import paddle.distributed as dist
from paddle import base
from paddle.base import core
from paddle.base.layer_helper import LayerHelper


class TestElementwiseAddOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.public_python_api = paddle.add
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.if_check_prim()
        self.if_enable_cinn()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def check_dygraph(self):
        return not self.use_mkldnn and self.axis == -1

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_output(
            check_dygraph=self.check_dygraph(),
            check_pir=self.check_dygraph(),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_normal(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_dygraph=self.check_dygraph(),
            check_prim=self.check_prim,
            check_prim_pir=self.check_dygraph(),
            check_pir=self.check_dygraph(),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_ignore_x(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=self.check_dygraph(),
            check_prim=self.check_prim,
            check_prim_pir=self.check_dygraph(),
            check_pir=self.check_dygraph(),
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_ignore_y(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=self.check_dygraph(),
            check_prim=self.check_prim,
            check_prim_pir=self.check_dygraph(),
            check_pir=self.check_dygraph(),
            check_pir_onednn=self.check_pir_onednn,
        )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float64

    def init_axis(self):
        self.axis = -1

    def if_check_prim(self):
        self.check_prim = self.axis == -1

    def if_enable_cinn(self):
        pass


class TestElementwiseAddOp_ZeroDim1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_ZeroDim2(TestElementwiseAddOp_ZeroDim1):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_ZeroDim3(TestElementwiseAddOp_ZeroDim1):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.out = np.add(self.x, self.y)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16ElementwiseAddOp(TestElementwiseAddOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place,
            atol=1e-3,
            check_dygraph=self.check_dygraph(),
            check_pir=self.check_dygraph(),
        )

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X', 'Y'], 'Out', check_prim=True)

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or core.cudnn_version() < 8100
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "only support compiled with CUDA and cudnn version need larger than 8.1.0 and device's compute capability is at least 8.0",
)
class TestBF16ElementwiseAddOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.public_python_api = paddle.add
        self.prim_op_type = "prim"
        self.dtype = np.uint16

        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        self.out = np.add(self.x, self.y)

        self.axis = -1

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(self.x)),
            'Y': OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(self.y)),
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': False}
        self.outputs = {'Out': convert_float_to_uint16(self.out)}
        self.if_enable_cinn()

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X', 'Y'],
            'Out',
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_ignore_x(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def if_enable_cinn(self):
        self.enable_cinn = False


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseAddOp_scalar(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestFP16ElementwiseAddOp_scalar(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1,1) to test broadcast."
)
class TestElementwiseAddOp_scalar2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1,1) to test broadcast."
)
class TestFP16ElementwiseAddOp_scalar2(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_Vector(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestFP16ElementwiseAddOp_Vector(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_broadcast_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 0

    def if_check_prim(self):
        self.check_prim = False


@skip_check_grad_ci(
    reason="the numerical method is not accurate enough on fp16"
)
class TestFP16ElementwiseAddOp_broadcast_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 0

    # In paddle2.0 api we don't have axis parameter in add,
    # so we can't check prim when axis is not -1 by default.
    def if_check_prim(self):
        self.check_prim = self.axis == -1

    # Because the numerical method is not accurate enough on fp16,
    # so we do not test the grad on fp16
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


class TestElementwiseAddOp_broadcast_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 100, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 100, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 1

    def if_check_prim(self):
        self.check_prim = False


class TestFP16ElementwiseAddOp_broadcast_1(
    TestFP16ElementwiseAddOp_broadcast_0
):
    def init_input_output(self):
        self.x = np.random.rand(2, 100, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 100, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_broadcast_2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 100)
        self.python_api = paddle.add


class TestFP16ElementwiseAddOp_broadcast_2(
    TestFP16ElementwiseAddOp_broadcast_0
):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 100)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_broadcast_3(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 1).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 1


class TestFP16ElementwiseAddOp_broadcast_3(
    TestFP16ElementwiseAddOp_broadcast_0
):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_broadcast_4(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 1, 2).astype(self.dtype)
        self.y = np.random.rand(100, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 0


class TestFP16ElementwiseAddOp_broadcast_4(
    TestFP16ElementwiseAddOp_broadcast_0
):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 1, 2).astype(self.dtype)
        self.y = np.random.rand(100, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 0


class TestElementwiseAddOp_broadcast_5(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 3, 12).astype(self.dtype)
        self.y = np.random.rand(10, 1, 12).astype(self.dtype)
        self.out = self.x + self.y


class TestFP16ElementwiseAddOp_broadcast_5(
    TestFP16ElementwiseAddOp_broadcast_0
):
    def init_input_output(self):
        self.x = np.random.rand(10, 3, 12).astype(self.dtype)
        self.y = np.random.rand(10, 1, 12).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_broadcast_6(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 12, 3, 5).astype(self.dtype)
        self.y = np.random.rand(2, 12, 1, 5).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_broadcast_7(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 1, 20, 5).astype(self.dtype)
        self.y = np.random.rand(20, 5, 1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestFP16ElementwiseAddOp_broadcast_6(
    TestFP16ElementwiseAddOp_broadcast_0
):
    def init_input_output(self):
        self.x = np.random.rand(2, 12, 3, 5).astype(self.dtype)
        self.y = np.random.rand(2, 12, 1, 5).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12)

    def init_axis(self):
        self.axis = 1


@skip_check_grad_ci(
    reason="the numerical method is not accurate enough on fp16."
)
class TestFP16ElementwiseAddOp_rowwise_add_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12)

    def init_axis(self):
        self.axis = 1

    # Because the numerical method is not accurate enough on fp16,
    # so we do not test the grad on fp16
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ignore_x(self):
        pass

    def test_check_grad_ignore_y(self):
        pass


class TestElementwiseAddOp_rowwise_add_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 100, 1).astype(self.dtype)
        self.y = np.random.rand(100, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 100, 1)


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestFP16ElementwiseAddOp_rowwise_add_1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_channelwise_add(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestFP16ElementwiseAddOp_channelwise_add(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_commonuse_add1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(1, 1, 100).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseFP16AddOp_commonuse_add1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(1, 1, 100).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_commonuse_add2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 3, 1, 4).astype(self.dtype)
        self.y = np.random.rand(10, 1, 12, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_xsize_lessthan_ysize_add(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 12).astype(self.dtype)
        self.y = np.random.rand(2, 2, 10, 12).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = 2


class TestElementwiseAddOp_same_shape_ysize_large(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 1, 12).astype(self.dtype)
        self.y = np.random.rand(10, 2, 12).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = 0

# --------------------------- Add fp16 cpu ---------------------------
def create_test_fp16_cpu(parent):
    class TestAddFloat16_CPU(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            place = core.CPUPlace()
            self.check_output_with_place(
                place,
                atol=1e-3,
                check_dygraph=self.check_dygraph(),
                check_pir=self.check_dygraph(),
            )

create_test_fp16_cpu(TestFP16ElementwiseAddOp)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_scalar)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_scalar2)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_Vector)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_0)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_1)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_2)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_3)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_4)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_5)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_broadcast_6)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_rowwise_add_0)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_rowwise_add_1)
create_test_fp16_cpu(TestFP16ElementwiseAddOp_channelwise_add)
create_test_fp16_cpu(TestElementwiseFP16AddOp_commonuse_add1)

# --------------------------------------------------------------------

class TestAddApi(unittest.TestCase):
    def _executed_api(self, x, y, name=None):
        return paddle.add(x, y, name)

    def test_name(self):
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(base.Program()):
                x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
                y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')

                y_1 = self._executed_api(x, y, name='add_res')
                self.assertEqual(('add_res' in y_1.name), True)

    def test_declarative(self):
        with base.program_guard(base.Program()):

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32'),
                }

            x = paddle.static.data(name="x", shape=[3], dtype='float32')
            y = paddle.static.data(name="y", shape=[3], dtype='float32')
            z = self._executed_api(x, y)

            place = base.CPUPlace()
            exe = base.Executor(place)
            z_value = exe.run(feed=gen_data(), fetch_list=[z])
            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual((z_value == z_expected).all(), True)

    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = self._executed_api(x, y)
            np_z = z.numpy()
            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual((np_z == z_expected).all(), True)


class TestAddInplaceApi(TestAddApi):
    def _executed_api(self, x, y, name=None):
        return x.add_(y, name)


class TestAddInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype('float')
        self.y_numpy = np.random.rand(3, 4).astype('float')

    def test_broadcast_success(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.add_(y)
        numpy_result = self.x_numpy + self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestAddInplaceBroadcastSuccess2(TestAddInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype('float')
        self.y_numpy = np.random.rand(3, 1).astype('float')


class TestAddInplaceBroadcastSuccess3(TestAddInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype('float')
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype('float')


class TestAddInplaceBroadcastError(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(3, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')

    def test_broadcast_errors(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        def broadcast_shape_error():
            x.add_(y)

        self.assertRaises(ValueError, broadcast_shape_error)
        paddle.enable_static()


class TestAddInplaceBroadcastError2(TestAddInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestAddInplaceBroadcastError3(TestAddInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(5, 2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestComplexElementwiseAddOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.dtype = np.complex128
        self.shape = (2, 3, 4, 5)
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x + self.y

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"), check_pir=True)

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_pir=True)


class TestRealComplexElementwiseAddOp(TestComplexElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(self.dtype)
        self.out = self.x + self.y


class TestBoolAddFloatElementwiseAddop(unittest.TestCase):
    def test_static_add(self):
        paddle.enable_static()
        a = 1.5
        b = paddle.full([4, 5, 6], True, dtype='bool')
        c = a + b
        self.assertTrue(c.dtype == paddle.float32)
        with paddle.pir_utils.IrGuard():
            a = 1.5
            b = paddle.full([4, 5, 6], True, dtype='bool')
            c = a + b
            self.assertTrue(c.dtype == core.DataType.FLOAT32)

    def test_dygraph_add(self):
        paddle.disable_static()
        a = 1.5
        b = paddle.full([2], True, dtype='bool')
        # special case: scalar + tensor(bool)
        c = a + b
        self.assertTrue(c.dtype == paddle.float32)

        np_a = np.random.random((2, 3, 4)).astype(np.float64)
        np_b = np.random.random((2, 3, 4)).astype(np.float64)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: tensor + tensor
        expect_out = np_a + np_b
        actual_out = tensor_a + tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor + scalar
        expect_out = np_a + 1
        actual_out = tensor_a + 1
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: scalar + tenor
        expect_out = 1 + np_a
        actual_out = 1 + tensor_a
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


class TestElementwiseAddop1(unittest.TestCase):
    def test_dygraph_add(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray + tenor
        expect_out = np_a + np_b
        actual_out = np_a + tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor + nparray
        actual_out = tensor_a + np_b
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


class TestTensorAddNumpyScalar(unittest.TestCase):
    def test_float32_add(self):
        paddle.disable_static()
        a = paddle.full([4, 5, 6], 1.5, dtype='float32')
        b = np.array([1.5], dtype='float32')[0]
        c = a + b
        self.assertTrue(c.dtype == paddle.float32)

    def test_float16_add(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        a = paddle.full([4, 5, 6], 1.5, dtype='float16')
        b = np.array([1.5], dtype='float16')[0]
        c = a + b
        self.assertTrue(c.dtype == paddle.float16)


class TestTensorAddAPIWarnings(unittest.TestCase):
    def test_warnings(self):
        with paddle.pir_utils.OldIrGuard():
            with warnings.catch_warnings(record=True) as context:
                warnings.simplefilter("always")

                paddle.enable_static()
                helper = LayerHelper("elementwise_add")
                data = paddle.static.data(
                    name='data', shape=[None, 3, 32, 32], dtype='float32'
                )
                out = helper.create_variable_for_type_inference(
                    dtype=data.dtype
                )
                os.environ['FLAGS_print_extra_attrs'] = "1"
                helper.append_op(
                    type="elementwise_add",
                    inputs={'X': data, 'Y': data},
                    outputs={'Out': out},
                    attrs={'axis': 1, 'use_mkldnn': False},
                )
                self.assertTrue(
                    "op elementwise_add's attr axis = 1 is not the default value: -1"
                    in str(context[-1].message)
                )
                os.environ['FLAGS_print_extra_attrs'] = "0"


class TestTensorFloat32Bfloat16OrFloat16Add(unittest.TestCase):
    def _float32_bfloat16_or_float16_add(self, y_dtype):
        paddle.disable_static()
        test_num = 5
        val_range = 10000
        shapes = []
        for i in range(test_num):
            shape = [np.random.randint(val_range), np.random.randint(val_range)]
            shapes.append(shape)

        for i, shape in enumerate(shapes):
            x = paddle.randn(list(shape), dtype=paddle.float32)
            x_copy = copy.deepcopy(x)
            y = paddle.randn(list(shape), dtype=y_dtype)
            x.add_(y)
            x_copy.add_(paddle.cast(y, paddle.float32))
            np.testing.assert_equal(x.numpy(), x_copy.numpy())
            del x, x_copy


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or core.cudnn_version() < 8100
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "only support compiled with CUDA and cudnn version need larger than 8.1.0 and device's compute capability is at least 8.0",
)
class TestTensorFloat32Bfloat16Add(TestTensorFloat32Bfloat16OrFloat16Add):
    def test_float32_bfloat16_add(self):
        place = core.CUDAPlace(0)
        with base.dygraph.base.guard(place=place):
            self._float32_bfloat16_or_float16_add(y_dtype=paddle.bfloat16)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.cudnn_version() < 8100,
    "only support compiled with CUDA and cudnn version need larger than 8.1.0",
)
class TestTensorFloat32Float16Add(TestTensorFloat32Bfloat16OrFloat16Add):
    def test_float32_float16_add(self):
        place = core.CUDAPlace(0)
        with base.dygraph.base.guard(place=place):
            self._float32_bfloat16_or_float16_add(y_dtype=paddle.float16)


class TestElementwiseAddOpAutoParallel(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.public_python_api = paddle.add
        self.prim_op_type = "prim"
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.if_check_prim()
        self.if_enable_cinn()
        self.init_placements()
        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }

        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def check_dygraph(self):
        return not self.use_mkldnn and self.axis == -1

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_auto_parallel=True,
        )

    def init_placements(self):
        self.placements = {
            "X": [dist.Shard(0)],
            "Y": [dist.Replicate()],
        }

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [16, 32]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [16, 32]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float64

    def init_axis(self):
        self.axis = -1

    def if_check_prim(self):
        self.check_prim = self.axis == -1

    def if_enable_cinn(self):
        pass


class TestElementwiseAddOpAutoParallelXShardBroadcast(
    TestElementwiseAddOpAutoParallel
):
    def init_placements(self):
        self.placements = {
            "X": [dist.Shard(0)],
            "Y": [dist.Replicate()],
        }

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [8, 16]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 8, 16]).astype(self.dtype)
        self.out = np.add(self.x, self.y)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseAddOpAutoParallelXYShard(TestElementwiseAddOpAutoParallel):
    def init_placements(self):
        self.placements = {
            "X": [dist.Shard(0)],
            "Y": [dist.Shard(1)],
        }

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X', 'Y'], 'Out', check_auto_parallel=True
        )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [16, 32]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [16, 32]).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOpAutoParallelXYShardBroadcast(
    TestElementwiseAddOpAutoParallelXYShard
):
    def init_placements(self):
        self.placements = {
            "X": [dist.Shard(0)],
            "Y": [dist.Replicate()],
        }

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place, ['X', 'Y'], 'Out', check_auto_parallel=True
        )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [8, 16]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 8, 16]).astype(self.dtype)
        self.out = np.add(self.x, self.y)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
