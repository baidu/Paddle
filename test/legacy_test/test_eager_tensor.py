#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import itertools
import unittest
import warnings

import numpy as np
from utils import dygraph_guard

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.utils.dlpack import DLDeviceType


class TestEagerTensor(unittest.TestCase):
    def setUp(self):
        self.shape = [512, 1234]
        self.dtype = np.float32
        self.array = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_to_tensor(self):
        def check_with_place(place):
            with base.dygraph.guard():
                paddle.set_default_dtype("float32")
                # set_default_dtype should not take effect on int
                x = paddle.to_tensor(1, place=place, stop_gradient=False)
                np.testing.assert_array_equal(x.numpy(), [1])
                self.assertNotEqual(x.dtype, paddle.float32)

                y = paddle.to_tensor(2, place=x.place)
                self.assertEqual(str(x.place), str(y.place))

                # set_default_dtype should not take effect on numpy
                x = paddle.to_tensor(
                    np.array([1.2]).astype("float16"),
                    place=place,
                    stop_gradient=False,
                )
                np.testing.assert_array_equal(
                    x.numpy(), np.array([1.2], "float16")
                )
                self.assertEqual(x.dtype, paddle.float16)

                # set_default_dtype take effect on int
                x = paddle.to_tensor(1, place=place)
                self.assertTrue(x.dtype, paddle.int64)

                # set_default_dtype take effect on float
                x = paddle.to_tensor(1.2, place=place, stop_gradient=False)
                np.testing.assert_array_equal(
                    x.numpy(), np.array([1.2]).astype("float32")
                )
                self.assertEqual(x.dtype, paddle.float32)
                clone_x = x.clone()
                np.testing.assert_array_equal(
                    clone_x.numpy(), np.array([1.2]).astype("float32")
                )
                self.assertEqual(clone_x.dtype, paddle.float32)
                y = clone_x**2
                y.backward()
                np.testing.assert_array_equal(
                    x.grad.numpy(), np.array([2.4]).astype("float32")
                )
                y = x.cpu()
                self.assertEqual(y.place.__repr__(), "Place(cpu)")
                if core.is_compiled_with_cuda():
                    y = x.pin_memory()
                    self.assertEqual(y.place.__repr__(), "Place(gpu_pinned)")
                    y = x.cuda()
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    y = x.cuda(None)
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    y = x.cuda(device_id=0)
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    y = x.cuda(blocking=False)
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    y = x.cuda(blocking=True)
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    y = x.cuda(device_id=0, blocking=True)
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    y = x.cuda(device_id=0, blocking=False)
                    self.assertEqual(y.place.__repr__(), "Place(gpu:0)")
                    with self.assertRaises(ValueError):
                        y = x.cuda("test")

                # support 'dtype' is core.VarType
                x = paddle.rand((2, 2))
                y = paddle.to_tensor([2, 2], dtype=x.dtype)
                self.assertEqual(y.dtype, paddle.float32)

                # set_default_dtype take effect on complex
                x = paddle.to_tensor(1 + 2j, place=place, stop_gradient=False)
                np.testing.assert_array_equal(x.numpy(), [1 + 2j])
                self.assertEqual(x.dtype, paddle.complex64)

                paddle.set_default_dtype("float64")
                x = paddle.to_tensor(1.2, place=place, stop_gradient=False)
                np.testing.assert_array_equal(x.numpy(), [1.2])
                self.assertEqual(x.dtype, paddle.float64)

                x = paddle.to_tensor(1 + 2j, place=place, stop_gradient=False)
                np.testing.assert_array_equal(x.numpy(), [1 + 2j])
                self.assertEqual(x.dtype, paddle.complex128)

                x = paddle.to_tensor(
                    1, dtype="float32", place=place, stop_gradient=False
                )
                np.testing.assert_array_equal(x.numpy(), [1.0])
                self.assertEqual(x.dtype, paddle.float32)
                self.assertEqual(x.shape, [])
                self.assertEqual(x.stop_gradient, False)
                self.assertEqual(x.type, core.VarDesc.VarType.DENSE_TENSOR)

                x = paddle.to_tensor(
                    (1, 2), dtype="float32", place=place, stop_gradient=False
                )
                x = paddle.to_tensor(
                    [1, 2], dtype="float32", place=place, stop_gradient=False
                )
                np.testing.assert_array_equal(x.numpy(), [1.0, 2.0])
                self.assertEqual(x.dtype, paddle.float32)
                self.assertIsNone(x.grad)
                self.assertEqual(x.shape, [2])
                self.assertEqual(x.stop_gradient, False)
                self.assertEqual(x.type, core.VarDesc.VarType.DENSE_TENSOR)

                x = paddle.to_tensor(
                    self.array,
                    dtype="float32",
                    place=place,
                    stop_gradient=False,
                )
                np.testing.assert_array_equal(x.numpy(), self.array)
                self.assertEqual(x.dtype, paddle.float32)
                self.assertEqual(x.shape, self.shape)
                self.assertEqual(x.stop_gradient, False)
                self.assertEqual(x.type, core.VarDesc.VarType.DENSE_TENSOR)

                y = paddle.to_tensor(x)
                y = paddle.to_tensor(y, dtype="float64", place=place)
                np.testing.assert_array_equal(y.numpy(), self.array)
                self.assertEqual(y.dtype, paddle.float64)
                self.assertEqual(y.shape, self.shape)
                self.assertEqual(y.stop_gradient, True)
                self.assertEqual(y.type, core.VarDesc.VarType.DENSE_TENSOR)
                z = x + y
                np.testing.assert_array_equal(z.numpy(), 2 * self.array)

                x = paddle.to_tensor(
                    [1 + 2j, 1 - 2j], dtype="complex64", place=place
                )
                y = paddle.to_tensor(x)
                np.testing.assert_array_equal(x.numpy(), [1 + 2j, 1 - 2j])
                self.assertEqual(y.dtype, paddle.complex64)
                self.assertEqual(y.shape, [2])

                paddle.set_default_dtype("float32")
                x = paddle.randn([3, 4])
                x_array = np.array(x)
                self.assertEqual(x_array.shape, x.numpy().shape)
                self.assertEqual(x_array.dtype, x.numpy().dtype)
                np.testing.assert_array_equal(x_array, x.numpy())

                x = paddle.to_tensor(1.0, place=place)
                self.assertEqual(x.item(), 1.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.randn([3, 2, 2])
                self.assertTrue(isinstance(x.item(5), float))
                self.assertTrue(isinstance(x.item(1, 0, 1), float))
                self.assertEqual(x.item(5), x.item(1, 0, 1))
                np.testing.assert_array_equal(
                    x.item(1, 0, 1), x.numpy().item(1, 0, 1)
                )

                x = paddle.to_tensor([[1.111111, 2.222222, 3.333333]])
                self.assertEqual(x.item(0, 2), x.item(2))
                self.assertAlmostEqual(x.item(2), 3.333333)
                self.assertTrue(isinstance(x.item(0, 2), float))

                x = paddle.to_tensor(1.0, dtype="float64")
                self.assertEqual(x.item(), 1.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.to_tensor(1.0, dtype="float16")
                self.assertEqual(x.item(), 1.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.to_tensor(1, dtype="uint8")
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype="int8")
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype="int16")
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype="int32")
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype="int64")
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(True)
                self.assertEqual(x.item(), True)
                self.assertTrue(isinstance(x.item(), bool))

                x = paddle.to_tensor(1 + 1j)
                self.assertEqual(x.item(), 1 + 1j)
                self.assertTrue(isinstance(x.item(), complex))

                # empty tensor
                x = paddle.to_tensor([])
                self.assertEqual(x.shape, [0])
                expected_result = np.array([], dtype="float32")
                self.assertEqual(x.numpy().shape, expected_result.shape)
                np.testing.assert_array_equal(x.numpy(), expected_result)

                numpy_array = np.random.randn(3, 4)
                # covert core.DenseTensor to paddle.Tensor
                dense_tensor = paddle.base.core.DenseTensor()
                place = paddle.base.framework._current_expected_place()
                dense_tensor.set(numpy_array, place)
                x = paddle.to_tensor(dense_tensor)
                np.testing.assert_array_equal(x.numpy(), numpy_array)
                self.assertEqual(x.type, core.VarDesc.VarType.DENSE_TENSOR)
                self.assertEqual(str(x.place), str(place))

                # covert core.DenseTensor to paddle.Tensor
                x = paddle.to_tensor(numpy_array)
                dlpack = x.value().get_tensor()._to_dlpack()
                tensor_from_dlpack = paddle.base.core.from_dlpack(dlpack)
                x = paddle.to_tensor(tensor_from_dlpack)
                np.testing.assert_array_equal(x.numpy(), numpy_array)
                self.assertEqual(x.type, core.VarDesc.VarType.DENSE_TENSOR)

                # test dtype=bfloat16
                x = paddle.to_tensor(-1e6, dtype=paddle.bfloat16)
                self.assertEqual(x.dtype, paddle.bfloat16)
                self.assertTrue(x == -999424.0)
                self.assertTrue(x.item() == -999424.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.to_tensor([-1e6, -1e6, -1e6], dtype="bfloat16")
                self.assertEqual(x.dtype, paddle.bfloat16)
                self.assertTrue(x[0] == -999424.0)
                self.assertTrue(x[1] == -999424.0)
                self.assertTrue(x[2] == -999424.0)

                x = paddle.to_tensor(
                    -1e6, dtype=paddle.bfloat16, stop_gradient=False
                )
                self.assertEqual(x.dtype, paddle.bfloat16)
                self.assertTrue(x == -999424.0)
                y = x * x
                y.backward()
                self.assertTrue(x.grad == -999424.0 * 2)

                # test default_type=bfloat16
                paddle.set_default_dtype("bfloat16")
                x = paddle.to_tensor(-1e6)
                self.assertEqual(x.dtype, paddle.bfloat16)
                self.assertTrue(x == -999424.0)
                self.assertTrue(x.item() == -999424.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.to_tensor([-1e6, -1e6, -1e6])
                self.assertEqual(x.dtype, paddle.bfloat16)
                self.assertTrue(x[0] == -999424.0)
                self.assertTrue(x[1] == -999424.0)
                self.assertTrue(x[2] == -999424.0)

                x = paddle.to_tensor(-1e6, stop_gradient=False)
                self.assertEqual(x.dtype, paddle.bfloat16)
                self.assertTrue(x == -999424.0)
                y = x * x
                y.backward()
                self.assertTrue(x.grad == -999424.0 * 2)
                paddle.set_default_dtype("float32")

                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item()
                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item(18)
                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item(1, 2)
                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item(2, 1, 2)
                with self.assertRaises(TypeError):
                    paddle.to_tensor("test")
                with self.assertRaises(TypeError):
                    paddle.to_tensor(1, dtype="test")
                with self.assertRaises(ValueError):
                    paddle.to_tensor([[1], [2, 3]])
                with self.assertRaises(ValueError):
                    paddle.to_tensor([[1], [2, 3]], place="test")
                with self.assertRaises(ValueError):
                    paddle.to_tensor([[1], [2, 3]], place=1)

        check_with_place(core.CPUPlace())
        check_with_place("cpu")
        if core.is_compiled_with_cuda():
            check_with_place(core.CUDAPinnedPlace())
            check_with_place("gpu_pinned")
            check_with_place(core.CUDAPlace(0))
            check_with_place("gpu:0")

    def test_to_tensor_not_change_input_stop_gradient(self):
        with paddle.base.dygraph.guard(core.CPUPlace()):
            a = paddle.zeros([1024])
            a.stop_gradient = False
            b = paddle.to_tensor(a)
            self.assertEqual(a.stop_gradient, False)
            self.assertEqual(b.stop_gradient, True)

    def test_to_tensor_change_place(self):
        if core.is_compiled_with_cuda():
            a_np = np.random.rand(1024, 1024)
            with paddle.base.dygraph.guard(core.CPUPlace()):
                a = paddle.to_tensor(a_np, place=paddle.CUDAPinnedPlace())
                a = paddle.to_tensor(a)
                self.assertEqual(a.place.__repr__(), "Place(cpu)")

            with paddle.base.dygraph.guard(core.CUDAPlace(0)):
                a = paddle.to_tensor(a_np, place=paddle.CUDAPinnedPlace())
                a = paddle.to_tensor(a)
                self.assertEqual(a.place.__repr__(), "Place(gpu:0)")

            with paddle.base.dygraph.guard(core.CUDAPlace(0)):
                a = paddle.to_tensor(a_np, place=paddle.CPUPlace())
                a = paddle.to_tensor(a, place=paddle.CUDAPinnedPlace())
                self.assertEqual(a.place.__repr__(), "Place(gpu_pinned)")

    def test_to_tensor_with_densetensor(self):
        if core.is_compiled_with_cuda():
            a_np = np.random.rand(1024, 1024)
            with paddle.base.dygraph.guard(core.CPUPlace()):
                dense_tensor = core.DenseTensor()
                dense_tensor.set(a_np, core.CPUPlace())
                a = paddle.to_tensor(dense_tensor)
                np.testing.assert_array_equal(a_np, a.numpy())

            with paddle.base.dygraph.guard(core.CUDAPlace(0)):
                dense_tensor = core.DenseTensor()
                dense_tensor.set(a_np, core.CUDAPlace(0))
                a = paddle.to_tensor(dense_tensor, place=core.CPUPlace())
                np.testing.assert_array_equal(a_np, a.numpy())
                self.assertTrue(a.place.__repr__(), "Place(cpu)")

    def test_to_tensor_attributes(self):
        var = paddle.to_tensor(self.array)
        np.testing.assert_array_equal(var.numpy(), self.array)
        # default value
        self.assertEqual(var.persistable, False)
        self.assertEqual(var.stop_gradient, True)
        self.assertEqual(var.shape, self.shape)
        self.assertEqual(var.dtype, paddle.float32)
        self.assertEqual(var.type, core.VarDesc.VarType.DENSE_TENSOR)

    def test_list_to_tensor(self):
        array = [[[1, 2], [1, 2], [1.0, 2]], [[1, 2], [1, 2], [1, 2]]]
        var = paddle.to_tensor(array, dtype="int32")
        np.testing.assert_array_equal(var.numpy(), array)
        self.assertEqual(var.shape, [2, 3, 2])
        self.assertEqual(var.dtype, paddle.int32)
        self.assertEqual(var.type, core.VarDesc.VarType.DENSE_TENSOR)

    def test_tuple_to_tensor(self):
        array = (((1, 2), (1, 2), (1, 2)), ((1, 2), (1, 2), (1, 2)))
        var = paddle.to_tensor(array, dtype="float32")
        np.testing.assert_array_equal(var.numpy(), array)
        self.assertEqual(var.shape, [2, 3, 2])
        self.assertEqual(var.dtype, paddle.float32)
        self.assertEqual(var.type, core.VarDesc.VarType.DENSE_TENSOR)

    def test_tensor_to_tensor(self):
        t = base.Tensor()
        t.set(np.random.random((1024, 1024)), paddle.CPUPlace())
        var = paddle.to_tensor(t)
        np.testing.assert_array_equal(t, var.numpy())

    def test_leaf_tensor(self):
        with base.dygraph.guard():
            x = paddle.to_tensor(np.random.uniform(-1, 1, size=[10, 10]))
            self.assertTrue(x.is_leaf)
            y = x + 1
            self.assertTrue(y.is_leaf)

            x = paddle.to_tensor(
                np.random.uniform(-1, 1, size=[10, 10]), stop_gradient=False
            )
            self.assertTrue(x.is_leaf)
            y = x + 1
            self.assertFalse(y.is_leaf)

            linear = paddle.nn.Linear(10, 10)
            input = paddle.to_tensor(
                np.random.uniform(-1, 1, size=[10, 10]).astype("float32"),
                stop_gradient=False,
            )
            self.assertTrue(input.is_leaf)

            out = linear(input)
            self.assertTrue(linear.weight.is_leaf)
            self.assertTrue(linear.bias.is_leaf)
            self.assertFalse(out.is_leaf)

    def test_detach(self):
        with base.dygraph.guard():
            x = paddle.to_tensor([1.0], dtype="float64", stop_gradient=False)
            detach_x = x.detach()
            self.assertTrue(detach_x.stop_gradient, True)

            cmp_float = (
                np.allclose if core.is_compiled_with_rocm() else np.array_equal
            )
            detach_x[:] = 10.0
            self.assertTrue(cmp_float(x.numpy(), [10.0]))

            y = x**2
            y.backward()
            self.assertTrue(cmp_float(x.grad.numpy(), [20.0]))
            self.assertIsNone(detach_x.grad)

            detach_x.stop_gradient = (
                False  # Set stop_gradient to be False, supported auto-grad
            )
            z = 3 * detach_x**2
            z.backward()
            self.assertTrue(cmp_float(x.grad.numpy(), [20.0]))
            self.assertTrue(cmp_float(detach_x.grad.numpy(), [60.0]))

            with self.assertRaises(ValueError):
                detach_x[:] = 5.0

            detach_x.stop_gradient = True

            # Due to sharing of data with origin Tensor, There are some unsafe operations:
            with self.assertRaises(RuntimeError):
                y = 2**x
                detach_x[:] = 5.0
                y.backward()

    def test_write_property(self):
        with base.dygraph.guard():
            var = paddle.to_tensor(self.array)

            self.assertEqual(var.name, "generated_tensor_0")
            var.name = "test"
            self.assertEqual(var.name, "test")

            self.assertEqual(var.persistable, False)
            var.persistable = True
            self.assertEqual(var.persistable, True)

            self.assertEqual(var.stop_gradient, True)
            var.stop_gradient = False
            self.assertEqual(var.stop_gradient, False)

    def test_deep_copy(self):
        with base.dygraph.guard():
            empty_var = core.eager.Tensor()
            empty_var_copy = copy.deepcopy(empty_var)
            self.assertEqual(
                empty_var.stop_gradient, empty_var_copy.stop_gradient
            )
            self.assertEqual(empty_var.persistable, empty_var_copy.persistable)
            self.assertEqual(empty_var.type, empty_var_copy.type)
            self.assertEqual(empty_var.dtype, empty_var_copy.dtype)

            x = paddle.to_tensor([2.0], stop_gradient=False)
            y = paddle.to_tensor([3.0], stop_gradient=False)
            z = x * y
            memo = {}
            x_copy = copy.deepcopy(x, memo)
            y_copy = copy.deepcopy(y, memo)

            self.assertEqual(x_copy.stop_gradient, y_copy.stop_gradient)
            self.assertEqual(x_copy.persistable, y_copy.persistable)
            self.assertEqual(x_copy.type, y_copy.type)
            self.assertEqual(x_copy.dtype, y_copy.dtype)
            np.testing.assert_array_equal(x.numpy(), x_copy.numpy())
            np.testing.assert_array_equal(y.numpy(), y_copy.numpy())

            self.assertNotEqual(id(x), id(x_copy))
            np.testing.assert_array_equal(x.numpy(), [2.0])

            with self.assertRaises(ValueError):
                x_copy[:] = 5.0

            x_copy2 = copy.deepcopy(x, memo)
            y_copy2 = copy.deepcopy(y, memo)
            self.assertEqual(id(x_copy), id(x_copy2))
            self.assertEqual(id(y_copy), id(y_copy2))

            # test copy selected rows
            x = core.eager.Tensor(
                core.VarDesc.VarType.FP32,
                [3, 100],
                "selected_rows",
                core.VarDesc.VarType.SELECTED_ROWS,
                True,
            )

            selected_rows = x.value().get_selected_rows()
            selected_rows.get_tensor().set(
                np.random.rand(3, 100), core.CPUPlace()
            )
            selected_rows.set_height(10)
            selected_rows.set_rows([3, 5, 7])
            x_copy = copy.deepcopy(x)

            self.assertEqual(x_copy.stop_gradient, x.stop_gradient)
            self.assertEqual(x_copy.persistable, x.persistable)
            self.assertEqual(x_copy.type, x.type)
            self.assertEqual(x_copy.dtype, x.dtype)

            copy_selected_rows = x_copy.value().get_selected_rows()
            self.assertEqual(
                copy_selected_rows.height(), selected_rows.height()
            )
            self.assertEqual(copy_selected_rows.rows(), selected_rows.rows())
            np.testing.assert_array_equal(
                np.array(copy_selected_rows.get_tensor()),
                np.array(selected_rows.get_tensor()),
            )

    def test_deep_copy_0size_tensor(self):
        x = paddle.to_tensor(np.array([]))
        x_copy = copy.deepcopy(x)
        self.assertEqual(x_copy.stop_gradient, x.stop_gradient)
        self.assertEqual(x_copy.persistable, x.persistable)
        self.assertEqual(x_copy.type, x.type)
        self.assertEqual(x_copy.dtype, x.dtype)
        self.assertEqual(x_copy.shape, x.shape)
        self.assertEqual(str(x_copy.place), str(x.place))
        np.testing.assert_array_equal(x.numpy(), x_copy.numpy())

    # test some patched methods
    def test_set_value(self):
        var = paddle.to_tensor(self.array)
        tmp1 = np.random.uniform(0.1, 1, [2, 2, 3]).astype(self.dtype)
        self.assertRaises(AssertionError, var.set_value, tmp1)

        tmp2 = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        var.set_value(tmp2)
        np.testing.assert_array_equal(var.numpy(), tmp2)

    def test_to_string(self):
        var = paddle.to_tensor(self.array)
        self.assertTrue(isinstance(str(var), str))

    def test_element_size(self):
        with base.dygraph.guard():
            x = paddle.to_tensor(1, dtype="bool")
            self.assertEqual(x.element_size(), 1)

            x = paddle.to_tensor(1, dtype="float16")
            self.assertEqual(x.element_size(), 2)

            x = paddle.to_tensor(1, dtype="float32")
            self.assertEqual(x.element_size(), 4)

            x = paddle.to_tensor(1, dtype="float64")
            self.assertEqual(x.element_size(), 8)

            x = paddle.to_tensor(1, dtype="int8")
            self.assertEqual(x.element_size(), 1)

            x = paddle.to_tensor(1, dtype="int16")
            self.assertEqual(x.element_size(), 2)

            x = paddle.to_tensor(1, dtype="int32")
            self.assertEqual(x.element_size(), 4)

            x = paddle.to_tensor(1, dtype="int64")
            self.assertEqual(x.element_size(), 8)

            x = paddle.to_tensor(1, dtype="uint8")
            self.assertEqual(x.element_size(), 1)

            x = paddle.to_tensor(1, dtype="complex64")
            self.assertEqual(x.element_size(), 8)

            x = paddle.to_tensor(1, dtype="complex128")
            self.assertEqual(x.element_size(), 16)

    def test_backward(self):
        var = paddle.to_tensor(self.array)
        var.stop_gradient = False
        loss = F.relu(var)
        loss.backward()
        grad_var = var._grad_ivar()
        self.assertEqual(grad_var.shape, self.shape)

    def test_gradient(self):
        var = paddle.to_tensor(self.array)
        var.stop_gradient = False
        loss = F.relu(var)
        loss.backward()
        grad_var = var.gradient()
        self.assertEqual(grad_var.shape, self.array.shape)

    def _test_slice(self):
        w = paddle.to_tensor(
            np.random.random((784, 100, 100)).astype("float64")
        )

        for i in range(3):
            nw = w[i]
            self.assertEqual((100, 100), tuple(nw.shape))

        nw = w[:]
        self.assertEqual((784, 100, 100), tuple(nw.shape))

        nw = w[:, :]
        self.assertEqual((784, 100, 100), tuple(nw.shape))

        nw = w[:, :, -1]
        self.assertEqual((784, 100), tuple(nw.shape))

        nw = w[1, 1, 1]

        self.assertEqual(len(nw.shape), 0)

        nw = w[:, :, :-1]
        self.assertEqual((784, 100, 99), tuple(nw.shape))

        tensor_array = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        ).astype("float32")
        var = paddle.to_tensor(tensor_array)
        var1 = var[0, 1, 1]
        var2 = var[1:]
        var3 = var[0:1]
        var4 = var[::-1]
        var5 = var[1, 1:, 1:]
        var_reshape = paddle.reshape(var, [3, -1, 3])
        var6 = var_reshape[:, :, -1]
        var7 = var[:, :, :-1]
        var8 = var[:1, :1, :1]
        var9 = var[:-1, :-1, :-1]
        var10 = var[::-1, :1, :-1]
        var11 = var[:-1, ::-1, -1:]
        var12 = var[1:2, 2:, ::-1]
        var13 = var[2:10, 2:, -2:-1]
        var14 = var[1:-1, 0:2, ::-1]
        var15 = var[::-1, ::-1, ::-1]
        var16 = var[-4:4]
        var17 = var[:, 0, 0:0]
        var18 = var[:, 1:1:2]

        vars = [
            var,
            var1,
            var2,
            var3,
            var4,
            var5,
            var6,
            var7,
            var8,
            var9,
            var10,
            var11,
            var12,
            var13,
            var14,
            var15,
            var16,
            var17,
            var18,
        ]
        local_out = [var.numpy() for var in vars]

        np.testing.assert_array_equal(local_out[1], tensor_array[0, 1, 1:2])
        np.testing.assert_array_equal(local_out[2], tensor_array[1:])
        np.testing.assert_array_equal(local_out[3], tensor_array[0:1])
        np.testing.assert_array_equal(local_out[4], tensor_array[::-1])
        np.testing.assert_array_equal(local_out[5], tensor_array[1, 1:, 1:])
        np.testing.assert_array_equal(
            local_out[6], tensor_array.reshape((3, -1, 3))[:, :, -1]
        )
        np.testing.assert_array_equal(local_out[7], tensor_array[:, :, :-1])
        np.testing.assert_array_equal(local_out[8], tensor_array[:1, :1, :1])
        np.testing.assert_array_equal(local_out[9], tensor_array[:-1, :-1, :-1])
        np.testing.assert_array_equal(
            local_out[10], tensor_array[::-1, :1, :-1]
        )
        np.testing.assert_array_equal(
            local_out[11], tensor_array[:-1, ::-1, -1:]
        )
        np.testing.assert_array_equal(
            local_out[12], tensor_array[1:2, 2:, ::-1]
        )
        np.testing.assert_array_equal(
            local_out[13], tensor_array[2:10, 2:, -2:-1]
        )
        np.testing.assert_array_equal(
            local_out[14], tensor_array[1:-1, 0:2, ::-1]
        )
        np.testing.assert_array_equal(
            local_out[15], tensor_array[::-1, ::-1, ::-1]
        )
        np.testing.assert_array_equal(local_out[16], tensor_array[-4:4])
        np.testing.assert_array_equal(local_out[17], tensor_array[:, 0, 0:0])
        np.testing.assert_array_equal(local_out[18], tensor_array[:, 1:1:2])

    def _test_slice_for_tensor_attr(self):
        tensor_array = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
            ]
        ).astype("float32")

        var = paddle.to_tensor(tensor_array)

        one = paddle.ones(shape=[], dtype="int32")
        two = paddle.full(shape=[], fill_value=2, dtype="int32")
        negative_one = paddle.full(shape=[], fill_value=-1, dtype="int32")
        four = paddle.full(shape=[], fill_value=4, dtype="int32")

        var = paddle.to_tensor(tensor_array)
        var1 = var[0, one, one]
        var2 = var[one:]
        var3 = var[0:one]
        var4 = var[::negative_one]
        var5 = var[one, one:, one:]
        var_reshape = paddle.reshape(var, [3, negative_one, 3])
        var6 = var_reshape[:, :, negative_one]
        var7 = var[:, :, :negative_one]
        var8 = var[:one, :one, :1]
        var9 = var[:-1, :negative_one, :negative_one]
        var10 = var[::negative_one, :one, :negative_one]
        var11 = var[:negative_one, ::-1, negative_one:]
        var12 = var[one:2, 2:, ::negative_one]
        var13 = var[two:10, 2:, -2:negative_one]
        var14 = var[1:negative_one, 0:2, ::negative_one]
        var15 = var[::negative_one, ::-1, ::negative_one]
        var16 = var[-4:4]

        vars = [
            var,
            var1,
            var2,
            var3,
            var4,
            var5,
            var6,
            var7,
            var8,
            var9,
            var10,
            var11,
            var12,
            var13,
            var14,
            var15,
            var16,
        ]
        local_out = [var.numpy() for var in vars]

        np.testing.assert_array_equal(local_out[1], tensor_array[0, 1, 1:2])
        np.testing.assert_array_equal(local_out[2], tensor_array[1:])
        np.testing.assert_array_equal(local_out[3], tensor_array[0:1])
        np.testing.assert_array_equal(local_out[4], tensor_array[::-1])
        np.testing.assert_array_equal(local_out[5], tensor_array[1, 1:, 1:])
        np.testing.assert_array_equal(
            local_out[6], tensor_array.reshape((3, -1, 3))[:, :, -1]
        )
        np.testing.assert_array_equal(local_out[7], tensor_array[:, :, :-1])
        np.testing.assert_array_equal(local_out[8], tensor_array[:1, :1, :1])
        np.testing.assert_array_equal(local_out[9], tensor_array[:-1, :-1, :-1])
        np.testing.assert_array_equal(
            local_out[10], tensor_array[::-1, :1, :-1]
        )
        np.testing.assert_array_equal(
            local_out[11], tensor_array[:-1, ::-1, -1:]
        )
        np.testing.assert_array_equal(
            local_out[12], tensor_array[1:2, 2:, ::-1]
        )
        np.testing.assert_array_equal(
            local_out[13], tensor_array[2:10, 2:, -2:-1]
        )
        np.testing.assert_array_equal(
            local_out[14], tensor_array[1:-1, 0:2, ::-1]
        )
        np.testing.assert_array_equal(
            local_out[15], tensor_array[::-1, ::-1, ::-1]
        )
        np.testing.assert_array_equal(local_out[16], tensor_array[-4:4])

    def _test_for_getitem_ellipsis_index(self):
        shape = (64, 3, 5, 256)
        np_fp32_value = np.random.random(shape).astype("float32")
        np_int_value = np.random.randint(1, 100, shape)

        var_fp32 = paddle.to_tensor(np_fp32_value)
        var_int = paddle.to_tensor(np_int_value)

        def assert_getitem_ellipsis_index(var_tensor, var_np):
            var = [
                var_tensor[..., 0].numpy(),
                var_tensor[..., 1, 0].numpy(),
                var_tensor[0, ..., 1, 0].numpy(),
                var_tensor[1, ..., 1].numpy(),
                var_tensor[2, ...].numpy(),
                var_tensor[2, 0, ...].numpy(),
                var_tensor[2, 0, 1, ...].numpy(),
                var_tensor[...].numpy(),
                var_tensor[:, ..., 100].numpy(),
            ]

            np.testing.assert_array_equal(var[0], var_np[..., 0])
            np.testing.assert_array_equal(var[1], var_np[..., 1, 0])
            np.testing.assert_array_equal(var[2], var_np[0, ..., 1, 0])
            np.testing.assert_array_equal(var[3], var_np[1, ..., 1])
            np.testing.assert_array_equal(var[4], var_np[2, ...])
            np.testing.assert_array_equal(var[5], var_np[2, 0, ...])
            np.testing.assert_array_equal(var[6], var_np[2, 0, 1, ...])
            np.testing.assert_array_equal(var[7], var_np[...])
            np.testing.assert_array_equal(var[8], var_np[:, ..., 100])

        var_fp32 = paddle.to_tensor(np_fp32_value)
        var_int = paddle.to_tensor(np_int_value)

        assert_getitem_ellipsis_index(var_fp32, np_fp32_value)
        assert_getitem_ellipsis_index(var_int, np_int_value)

        # test 1 dim tensor
        var_one_dim = paddle.to_tensor([1, 2, 3, 4])
        np.testing.assert_array_equal(
            var_one_dim[..., 0].numpy(), np.array([1])
        )

    def _test_none_index(self):
        shape = (8, 64, 5, 256)
        np_value = np.random.random(shape).astype("float32")
        var_tensor = paddle.to_tensor(np_value)

        var = [
            var_tensor[1, 0, None].numpy(),
            var_tensor[None, ..., 1, 0].numpy(),
            var_tensor[:, :, :, None].numpy(),
            var_tensor[1, ..., 1, None].numpy(),
            var_tensor[2, ..., None, None].numpy(),
            var_tensor[None, 2, 0, ...].numpy(),
            var_tensor[None, 2, None, 1].numpy(),
            var_tensor[None].numpy(),
            var_tensor[0, 0, None, 0, 0, None].numpy(),
            var_tensor[None, None, 0, ..., None].numpy(),
            var_tensor[..., None, :, None].numpy(),
            var_tensor[0, 1:10:2, None, None, ...].numpy(),
        ]

        np.testing.assert_array_equal(var[0], np_value[1, 0, None])
        np.testing.assert_array_equal(var[1], np_value[None, ..., 1, 0])
        np.testing.assert_array_equal(var[2], np_value[:, :, :, None])
        np.testing.assert_array_equal(var[3], np_value[1, ..., 1, None])
        np.testing.assert_array_equal(var[4], np_value[2, ..., None, None])
        np.testing.assert_array_equal(var[5], np_value[None, 2, 0, ...])
        np.testing.assert_array_equal(var[6], np_value[None, 2, None, 1])
        np.testing.assert_array_equal(var[7], np_value[None])
        np.testing.assert_array_equal(var[8], np_value[0, 0, None, 0, 0, None])
        np.testing.assert_array_equal(
            var[9], np_value[None, None, 0, ..., None]
        )
        np.testing.assert_array_equal(var[10], np_value[..., None, :, None])

        # TODO(zyfncg) there is a bug of dimensions when slice step > 1 and
        #              indices has int type
        # self.assertTrue(
        #     np.array_equal(var[11], np_value[0, 1:10:2, None, None, ...]))

    def _test_bool_index(self):
        shape = (4, 2, 5, 64)
        np_value = np.random.random(shape).astype("float32")
        var_tensor = paddle.to_tensor(np_value)
        index = [
            [True, True, True, True],
            [True, False, True, True],
            [True, False, False, True],
            [False, 0, 1, True, True],
            [False, False, False, False],
        ]
        index2d = np.array(
            [[True, True], [False, False], [True, False], [True, True]]
        )
        tensor_index = paddle.to_tensor(index2d)
        var = [
            var_tensor[index[0]].numpy(),
            var_tensor[index[1]].numpy(),
            var_tensor[index[2]].numpy(),
            var_tensor[index[3]].numpy(),
            var_tensor[paddle.to_tensor(index[0])].numpy(),
            var_tensor[tensor_index].numpy(),
            var_tensor[paddle.to_tensor(index[4])].numpy(),
        ]
        np.testing.assert_array_equal(var[0], np_value[index[0]])
        np.testing.assert_array_equal(var[1], np_value[index[1]])
        np.testing.assert_array_equal(var[2], np_value[index[2]])
        np.testing.assert_array_equal(var[3], np_value[index[3]])
        np.testing.assert_array_equal(var[4], np_value[index[0]])
        np.testing.assert_array_equal(var[5], np_value[index2d])
        np.testing.assert_array_equal(var[6], np_value[index[4]])
        np.testing.assert_array_equal(
            var_tensor[var_tensor > 0.67], np_value[np_value > 0.67]
        )
        np.testing.assert_array_equal(
            var_tensor[var_tensor < 0.55], np_value[np_value < 0.55]
        )

        with self.assertRaises(IndexError):
            var_tensor[[True, False]]
        with self.assertRaises(IndexError):
            var_tensor[[True, False, False, False, False]]
        with self.assertRaises(IndexError):
            var_tensor[paddle.to_tensor([[True, False, False, False]])]

    def _test_scalar_bool_index(self):
        shape = (1, 2, 5, 64)
        np_value = np.random.random(shape).astype("float32")
        var_tensor = paddle.to_tensor(np_value)
        index = [True]
        tensor_index = paddle.to_tensor(index)
        var = [
            var_tensor[tensor_index].numpy(),
        ]
        np.testing.assert_array_equal(var[0], np_value[index])

    def _test_for_var(self):
        np_value = np.random.random((30, 100, 100)).astype("float32")
        w = paddle.to_tensor(np_value)

        for i, e in enumerate(w):
            np.testing.assert_array_equal(e.numpy(), np_value[i])

    def _test_numpy_index(self):
        array = np.arange(120).reshape([4, 5, 6])
        t = paddle.to_tensor(array)
        np.testing.assert_array_equal(t[np.longlong(0)].numpy(), array[0])
        np.testing.assert_array_equal(
            t[np.longlong(0) : np.longlong(4) : np.longlong(2)].numpy(),
            array[0:4:2],
        )
        np.testing.assert_array_equal(t[np.int64(0)].numpy(), array[0])
        np.testing.assert_array_equal(
            t[np.int32(1) : np.int32(4) : np.int32(2)].numpy(), array[1:4:2]
        )
        np.testing.assert_array_equal(
            t[np.int16(0) : np.int16(4) : np.int16(2)].numpy(), array[0:4:2]
        )

    def _test_list_index(self):
        # case1:
        array = np.arange(120).reshape([6, 5, 4])
        x = paddle.to_tensor(array)
        py_idx = [[0, 2, 0, 1, 3], [0, 0, 1, 2, 0]]

        idx = [paddle.to_tensor(py_idx[0]), paddle.to_tensor(py_idx[1])]
        np.testing.assert_array_equal(x[idx].numpy(), array[np.array(py_idx)])
        np.testing.assert_array_equal(
            x[py_idx].numpy(), array[np.array(py_idx)]
        )
        # case2:
        tensor_x = paddle.to_tensor(
            np.zeros(12).reshape(2, 6).astype(np.float32)
        )
        tensor_y1 = paddle.zeros([1], dtype="int32") + 2
        tensor_y2 = paddle.zeros([1], dtype="int32") + 5
        tensor_x[:, tensor_y1:tensor_y2] = 42
        res = tensor_x.numpy()
        exp = np.array(
            [
                [0.0, 0.0, 42.0, 42.0, 42.0, 0.0],
                [0.0, 0.0, 42.0, 42.0, 42.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(res, exp)

        # case3:
        row = np.array([0, 1, 2])
        col = np.array([2, 1, 3])
        np.testing.assert_array_equal(array[row, col], x[row, col].numpy())

    def test_slice(self):
        with base.dygraph.guard():
            self._test_slice()
            self._test_slice_for_tensor_attr()
            self._test_for_var()
            self._test_for_getitem_ellipsis_index()
            self._test_none_index()
            self._test_bool_index()
            self._test_scalar_bool_index()
            self._test_numpy_index()
            self._test_list_index()

            var = paddle.to_tensor(self.array)
            np.testing.assert_array_equal(var[1, :].numpy(), self.array[1, :])
            np.testing.assert_array_equal(var[::-1].numpy(), self.array[::-1])

            with self.assertRaises(IndexError):
                y = var[self.shape[0]]

            with self.assertRaises(IndexError):
                y = var[0 - self.shape[0] - 1]

            with self.assertRaises(IndexError):
                mask = np.array([1, 0, 1, 0], dtype=bool)
                var[paddle.to_tensor([0, 1]), mask]

    def test_tensor_to_np(self):
        with base.dygraph.guard():
            var = paddle.to_tensor(self.array)
            np.testing.assert_array_equal(var.numpy(), var.numpy(False))

    def test_tensor_as_np(self):
        with base.dygraph.guard():
            var = paddle.to_tensor(self.array)
            np.testing.assert_array_equal(var.numpy(), np.array(var))
            np.testing.assert_array_equal(
                var.numpy(), np.array(var, dtype=np.float32)
            )

    def test_if(self):
        with base.dygraph.guard():
            var1 = paddle.to_tensor(np.array([[[0]]]))
            var2 = paddle.to_tensor(np.array([[[1]]]))

            var1_bool = False
            var2_bool = False

            if var1:
                var1_bool = True

            if var2:
                var2_bool = True

            assert not var1_bool, "if var1 should be false"
            assert var2_bool, "if var2 should be true"
            assert not bool(var1), "bool(var1) is False"
            assert bool(var2), "bool(var2) is True"

    def test_tensor_str(self):
        paddle.enable_static()
        paddle.disable_static(paddle.CPUPlace())
        paddle.seed(10)
        a = paddle.rand([10, 20])
        paddle.set_printoptions(4, 100, 3)
        a_str = str(a)

        expected = """Tensor(shape=[10, 20], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[0.2727, 0.5489, 0.8655, ..., 0.2916, 0.8525, 0.9000],
        [0.3806, 0.8996, 0.0928, ..., 0.9535, 0.8378, 0.6409],
        [0.1484, 0.4038, 0.8294, ..., 0.0148, 0.6520, 0.4250],
        ...,
        [0.3426, 0.1909, 0.7240, ..., 0.4218, 0.2676, 0.5679],
        [0.5561, 0.2081, 0.0676, ..., 0.9778, 0.3302, 0.9559],
        [0.2665, 0.8483, 0.5389, ..., 0.4956, 0.6862, 0.9178]])"""

        self.assertEqual(a_str, expected)

    def test_tensor_str2(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor([[1.5111111, 1.0], [0, 0]])
        a_str = str(a)

        expected = """Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[1.5111, 1.    ],
        [0.    , 0.    ]])"""

        self.assertEqual(a_str, expected)

    def test_tensor_str3(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor([[-1.5111111, 1.0], [0, -0.5]])
        a_str = str(a)

        expected = """Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-1.5111,  1.    ],
        [ 0.    , -0.5000]])"""

        self.assertEqual(a_str, expected)

    def test_tensor_str_scaler(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor(np.array(False))
        a_str = str(a)

        expected = """Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
       False)"""

        self.assertEqual(a_str, expected)

    def test_tensor_str_shape_with_zero(self):
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.ones((10, 10))
        y = paddle.nonzero(x == 0)
        a_str = str(y)

        expected = """Tensor(shape=[0, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [])"""

        self.assertEqual(a_str, expected)

    def test_tensor_str_linewidth(self):
        paddle.disable_static(paddle.CPUPlace())
        paddle.seed(2021)
        x = paddle.rand([128])
        paddle.set_printoptions(
            precision=4, threshold=1000, edgeitems=3, linewidth=80
        )
        a_str = str(x)

        expected = """Tensor(shape=[128], dtype=float32, place=Place(cpu), stop_gradient=True,
       [0.3759, 0.0278, 0.2489, 0.3110, 0.9105, 0.7381, 0.1905, 0.4726, 0.2435,
        0.9142, 0.3367, 0.7243, 0.7664, 0.9915, 0.2921, 0.1363, 0.8096, 0.2915,
        0.9564, 0.9972, 0.2573, 0.2597, 0.3429, 0.2484, 0.9579, 0.7003, 0.4126,
        0.4274, 0.0074, 0.9686, 0.9910, 0.0144, 0.6564, 0.2932, 0.7114, 0.9301,
        0.6421, 0.0538, 0.1273, 0.5771, 0.9336, 0.6416, 0.1832, 0.9311, 0.7702,
        0.7474, 0.4479, 0.3382, 0.5579, 0.0444, 0.9802, 0.9874, 0.3038, 0.5640,
        0.2408, 0.5489, 0.8866, 0.1006, 0.5881, 0.7560, 0.7928, 0.8604, 0.4670,
        0.9285, 0.1482, 0.4541, 0.1307, 0.6221, 0.4902, 0.1147, 0.4415, 0.2987,
        0.7276, 0.2077, 0.7551, 0.9652, 0.4369, 0.2282, 0.0047, 0.2934, 0.4308,
        0.4190, 0.1442, 0.3650, 0.3056, 0.6535, 0.1211, 0.8721, 0.7408, 0.4220,
        0.5937, 0.3123, 0.9198, 0.0275, 0.5338, 0.4622, 0.7521, 0.3609, 0.4703,
        0.1736, 0.8976, 0.7616, 0.3756, 0.2416, 0.2907, 0.3246, 0.4305, 0.5717,
        0.0735, 0.0361, 0.5534, 0.4399, 0.9260, 0.6525, 0.3064, 0.4573, 0.9210,
        0.8269, 0.2424, 0.7494, 0.8945, 0.7098, 0.8078, 0.4707, 0.5715, 0.7232,
        0.4678, 0.5047])"""

        self.assertEqual(a_str, expected)

    def test_tensor_str_linewidth2(self):
        paddle.disable_static(paddle.CPUPlace())
        paddle.seed(2021)
        x = paddle.rand([128])
        paddle.set_printoptions(precision=4, linewidth=160, sci_mode=True)
        a_str = str(x)

        expected = """Tensor(shape=[128], dtype=float32, place=Place(cpu), stop_gradient=True,
       [3.7587e-01, 2.7798e-02, 2.4891e-01, 3.1097e-01, 9.1053e-01, 7.3811e-01, 1.9045e-01, 4.7258e-01, 2.4354e-01, 9.1415e-01, 3.3666e-01, 7.2428e-01,
        7.6640e-01, 9.9146e-01, 2.9215e-01, 1.3625e-01, 8.0957e-01, 2.9153e-01, 9.5642e-01, 9.9718e-01, 2.5732e-01, 2.5973e-01, 3.4292e-01, 2.4841e-01,
        9.5794e-01, 7.0029e-01, 4.1260e-01, 4.2737e-01, 7.3788e-03, 9.6863e-01, 9.9102e-01, 1.4416e-02, 6.5640e-01, 2.9318e-01, 7.1136e-01, 9.3008e-01,
        6.4209e-01, 5.3849e-02, 1.2730e-01, 5.7712e-01, 9.3359e-01, 6.4155e-01, 1.8320e-01, 9.3110e-01, 7.7021e-01, 7.4736e-01, 4.4793e-01, 3.3817e-01,
        5.5794e-01, 4.4412e-02, 9.8023e-01, 9.8735e-01, 3.0376e-01, 5.6397e-01, 2.4082e-01, 5.4893e-01, 8.8659e-01, 1.0065e-01, 5.8812e-01, 7.5600e-01,
        7.9280e-01, 8.6041e-01, 4.6701e-01, 9.2852e-01, 1.4821e-01, 4.5410e-01, 1.3074e-01, 6.2210e-01, 4.9024e-01, 1.1466e-01, 4.4154e-01, 2.9868e-01,
        7.2758e-01, 2.0766e-01, 7.5508e-01, 9.6522e-01, 4.3688e-01, 2.2823e-01, 4.7394e-03, 2.9342e-01, 4.3083e-01, 4.1902e-01, 1.4416e-01, 3.6500e-01,
        3.0560e-01, 6.5350e-01, 1.2115e-01, 8.7206e-01, 7.4081e-01, 4.2203e-01, 5.9372e-01, 3.1230e-01, 9.1979e-01, 2.7486e-02, 5.3383e-01, 4.6224e-01,
        7.5211e-01, 3.6094e-01, 4.7034e-01, 1.7355e-01, 8.9763e-01, 7.6165e-01, 3.7557e-01, 2.4157e-01, 2.9074e-01, 3.2458e-01, 4.3049e-01, 5.7171e-01,
        7.3509e-02, 3.6087e-02, 5.5341e-01, 4.3993e-01, 9.2601e-01, 6.5248e-01, 3.0640e-01, 4.5727e-01, 9.2104e-01, 8.2688e-01, 2.4243e-01, 7.4937e-01,
        8.9448e-01, 7.0981e-01, 8.0783e-01, 4.7065e-01, 5.7154e-01, 7.2319e-01, 4.6777e-01, 5.0465e-01])"""

        self.assertEqual(a_str, expected)

    def test_tensor_str_bf16(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor([[1.5, 1.0], [0, 0]])
        a = paddle.cast(a, dtype=paddle.bfloat16)
        paddle.set_printoptions(precision=4)
        a_str = str(a)

        expected = """Tensor(shape=[2, 2], dtype=bfloat16, place=Place(cpu), stop_gradient=True,
       [[1.5000, 1.    ],
        [0.    , 0.    ]])"""

        self.assertEqual(a_str, expected)

    def test_print_tensor_dtype(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.rand([1])
        a_str = str(a.dtype)

        expected = "paddle.float32"

        self.assertEqual(a_str, expected)

    def test___cuda_array_interface__(self):
        """test Tensor.__cuda_array_interface__"""
        with dygraph_guard():
            # raise AttributeError for cpu tensor.
            cpu_place = paddle.CPUPlace()
            cpu_tensor = paddle.rand([3, 3]).to(device=cpu_place)
            self.assertRaises(
                AttributeError,
                getattr,
                cpu_tensor,
                '__cuda_array_interface__',
            )

            if paddle.device.is_compiled_with_cuda():
                gpu_place = paddle.CUDAPlace(0)
                # raise AttributeError for sparse tensor.
                sparse_tensor = (
                    paddle.rand([3, 3]).to(device=gpu_place).to_sparse_coo(2)
                )
                self.assertRaises(
                    AttributeError,
                    getattr,
                    sparse_tensor,
                    '__cuda_array_interface__',
                )

                # strides should be None if contiguous
                tensor = paddle.randn([3, 3]).to(device=gpu_place)
                interface = tensor.__cuda_array_interface__
                self.assertIsNone(interface["strides"])

                # strides should be tuple of int if not contiguous
                tensor = paddle.randn([10, 10]).to(device=gpu_place)
                tensor = tensor[::2]
                interface = tensor.__cuda_array_interface__
                self.assertEqual(interface["strides"], (80, 4))

                # data_ptr should be 0 if tensor is 0-size
                tensor = paddle.randn([0, 10]).to(device=gpu_place)
                interface = tensor.__cuda_array_interface__
                self.assertEqual(interface["data"][0], 0)

                # raise AttributeError for tensor that requires grad.
                tensor = paddle.randn([3, 3]).to(device=gpu_place)
                tensor.stop_gradient = False
                self.assertRaises(
                    RuntimeError,
                    getattr,
                    tensor,
                    '__cuda_array_interface__',
                )

                # check supports of dtypes
                for dtype in [
                    paddle.complex64,
                    paddle.complex128,
                    paddle.bfloat16,
                    paddle.float16,
                    paddle.float32,
                    paddle.float64,
                    paddle.uint8,
                    paddle.int8,
                    paddle.int16,
                    paddle.int32,
                    paddle.int64,
                    paddle.bool,
                ]:
                    tensor = (
                        paddle.uniform([10, 10], min=-10.0, max=10.0)
                        .to(device=gpu_place)
                        .astype(dtype)
                    )
                    interface = tensor.__cuda_array_interface__
                    self.assertIn("typestr", interface)
                    self.assertIsInstance(interface["typestr"], str)

                    self.assertIn("shape", interface)
                    self.assertIsInstance(interface["shape"], tuple)

                    self.assertIn("strides", interface)
                    self.assertTrue(
                        isinstance(interface["strides"], tuple)
                        or interface["strides"] is None
                    )

                    self.assertIn("data", interface)
                    self.assertIsInstance(interface["data"], tuple)
                    self.assertEqual(len(interface["data"]), 2)

                    self.assertIn("version", interface)
                    self.assertEqual(interface["version"], 2)

    def test_to_tensor_from___cuda_array_interface__(self):
        # only test warning message here for cuda tensor of other framework is not supported in Paddle test, more tests code can be referenced: https://github.com/PaddlePaddle/Paddle/pull/69913
        with dygraph_guard():
            with warnings.catch_warnings(record=True) as w:
                x = paddle.to_tensor([1, 2, 3])
                paddle.to_tensor(x)
                flag = paddle.tensor.creation._warned_in_to_tensor
                self.assertTrue(flag)

    def test_dlpack_device(self):
        """test Tensor.__dlpack_device__"""
        with dygraph_guard():
            # test CPU
            tensor_cpu = paddle.to_tensor([1, 2, 3], place=base.CPUPlace())
            device_type, device_id = tensor_cpu.__dlpack_device__()
            self.assertEqual(device_type, DLDeviceType.kDLCPU)
            self.assertEqual(device_id, None)

            # test CUDA
            if paddle.is_compiled_with_cuda():
                tensor_cuda = paddle.to_tensor(
                    [1, 2, 3], place=base.CUDAPlace(0)
                )
                device_type, device_id = tensor_cuda.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDA)
                self.assertEqual(device_id, 0)

            # test CUDA Pinned
            if paddle.is_compiled_with_cuda():
                tensor_pinned = paddle.to_tensor(
                    [1, 2, 3], place=base.CUDAPinnedPlace()
                )
                device_type, device_id = tensor_pinned.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDAHost)
                self.assertEqual(device_id, None)

            # test XPU
            if paddle.is_compiled_with_xpu():
                tensor_xpu = paddle.to_tensor([1, 2, 3], place=base.XPUPlace(0))
                device_type, device_id = tensor_xpu.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLOneAPI)
                self.assertEqual(device_id, 0)

            # zero_dim
            # test CPU
            tensor = paddle.to_tensor(5.0, place=base.CPUPlace())
            device_type, device_id = tensor.__dlpack_device__()
            self.assertEqual(device_type, DLDeviceType.kDLCPU)
            self.assertEqual(device_id, None)

            # test CUDA
            if paddle.is_compiled_with_cuda():
                tensor_cuda = paddle.to_tensor(5.0, place=base.CUDAPlace(0))
                device_type, device_id = tensor_cuda.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDA)
                self.assertEqual(device_id, 0)

            # test XPU
            if paddle.is_compiled_with_xpu():
                tensor_xpu = paddle.to_tensor(5.0, place=base.XPUPlace(0))
                device_type, device_id = tensor_xpu.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLOneAPI)
                self.assertEqual(device_id, 0)

            # zero_size
            # test CPU
            tensor = paddle.to_tensor(
                paddle.zeros([0, 10]), place=base.CPUPlace()
            )
            device_type, device_id = tensor.__dlpack_device__()
            self.assertEqual(device_type, DLDeviceType.kDLCPU)
            self.assertEqual(device_id, None)

            # test CUDA
            if paddle.is_compiled_with_cuda():
                tensor_cuda = paddle.to_tensor(
                    paddle.zeros([0, 10]), place=base.CUDAPlace(0)
                )
                device_type, device_id = tensor_cuda.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLCUDA)
                self.assertEqual(device_id, 0)

            # test XPU
            if paddle.is_compiled_with_xpu():
                tensor_xpu = paddle.to_tensor(
                    paddle.zeros([0, 10]), place=base.XPUPlace(0)
                )
                device_type, device_id = tensor_xpu.__dlpack_device__()
                self.assertEqual(device_type, DLDeviceType.kDLOneAPI)
                self.assertEqual(device_id, 0)

    def test_tensor__format__(self):
        # test for floating point scalar
        for width in range(0, 5):
            paddle_scalar = paddle.randn([])
            numpy_scalar = paddle_scalar.numpy()
            format_spec = f".{width}f"
            self.assertEqual(
                paddle_scalar.__format__(format_spec),
                numpy_scalar.__format__(format_spec),
            )
            format_spec = f".{width}e"
            self.assertEqual(
                paddle_scalar.__format__(format_spec),
                numpy_scalar.__format__(format_spec),
            )
            format_spec = f".{width}g"
            self.assertEqual(
                paddle_scalar.__format__(format_spec),
                numpy_scalar.__format__(format_spec),
            )

            format_spec = "{:.{}f}"
            self.assertEqual(
                format_spec.format(paddle_scalar, width),
                format_spec.format(numpy_scalar, width),
            )

        # test for integer scalar
        for width in range(0, 5):
            paddle_scalar = paddle.uniform([], min=-100, max=100).to("int64")
            numpy_scalar = paddle_scalar.numpy()
            format_spec = f"{width}d"
            self.assertEqual(
                paddle_scalar.__format__(format_spec),
                numpy_scalar.__format__(format_spec),
            )
            format_spec = f"{width}o"
            self.assertEqual(
                paddle_scalar.__format__(format_spec),
                numpy_scalar.__format__(format_spec),
            )
            format_spec = f"{width}x"
            self.assertEqual(
                paddle_scalar.__format__(format_spec),
                numpy_scalar.__format__(format_spec),
            )

            format_spec = "{:{}d}"
            self.assertEqual(
                format_spec.format(paddle_scalar, width),
                format_spec.format(numpy_scalar, width),
            )

        # test for tensor that ndim > 0, expected to raise TypeError
        paddle_scalar = paddle.uniform([1], min=-100, max=100)
        self.assertRaises(TypeError, paddle_scalar.__format__, ".3f")

        # test for float scalar but format_spec is 'd', expected to raise ValueError
        paddle_scalar = paddle.uniform([], min=-100, max=100)
        self.assertRaises(ValueError, paddle_scalar.__format__, "3d")


class TestEagerTensorSetitem(unittest.TestCase):
    def func_setUp(self):
        self.set_dtype()
        self.tensor_x = paddle.to_tensor(np.ones((4, 2, 3)).astype(self.dtype))
        self.np_value = np.random.random((2, 3)).astype(self.dtype)
        self.tensor_value = paddle.to_tensor(self.np_value)

    def set_dtype(self):
        self.dtype = "int32"

    def _test(self, value):
        id_origin = id(self.tensor_x)
        self.tensor_x[0] = value
        if isinstance(value, (int, float)):
            result = np.zeros((2, 3)).astype(self.dtype) + value

        else:
            result = self.np_value

        np.testing.assert_array_equal(self.tensor_x[0].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))

        self.tensor_x[1:2] = value
        np.testing.assert_array_equal(self.tensor_x[1].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))

        self.tensor_x[...] = value
        np.testing.assert_array_equal(self.tensor_x[3].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))

    def func_test_value_tensor(self):
        self._test(self.tensor_value)

    def test_value_tensor(self):
        self.func_setUp()
        self.func_test_value_tensor()

    def func_test_value_numpy(self):
        self._test(self.np_value)

    def test_value_numpy(self):
        self.func_setUp()
        self.func_test_value_numpy()

    def func_test_value_int(self):
        self._test(10)

    def test_value_int(self):
        self.func_setUp()
        self.func_test_value_int()


class TestEagerTensorSetitemInt64(TestEagerTensorSetitem):
    def set_dtype(self):
        self.dtype = "int64"


class TestEagerTensorSetitemFp32(TestEagerTensorSetitem):
    def set_dtype(self):
        self.dtype = "float32"

    def func_test_value_float(self):
        paddle.disable_static()
        self._test(3.3)

    def test_value_float(self):
        self.func_setUp()
        self.func_test_value_float()


class TestEagerTensorSetitemFp64(TestEagerTensorSetitem):
    def set_dtype(self):
        self.dtype = "float64"


class TestEagerTensorSetitemBoolIndex(unittest.TestCase):
    def func_setUp(self):
        paddle.disable_static()
        self.set_dtype()
        self.set_input()

    def set_input(self):
        self.tensor_x = paddle.to_tensor(np.ones((4, 2, 3)).astype(self.dtype))
        self.np_value = np.random.random((2, 3)).astype(self.dtype)
        self.tensor_value = paddle.to_tensor(self.np_value)

    def set_dtype(self):
        self.dtype = "int32"

    def _test(self, value):
        paddle.disable_static()

        id_origin = id(self.tensor_x)
        index_1 = paddle.to_tensor(np.array([True, False, False, False]))
        self.tensor_x[index_1] = value

        if isinstance(value, (int, float)):
            result = np.zeros((2, 3)).astype(self.dtype) + value

        else:
            result = self.np_value

        np.testing.assert_array_equal(self.tensor_x[0].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))

        index_2 = paddle.to_tensor(np.array([False, True, False, False]))
        self.tensor_x[index_2] = value
        np.testing.assert_array_equal(self.tensor_x[1].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))

        index_3 = paddle.to_tensor(np.array([True, True, True, True]))
        self.tensor_x[index_3] = value
        np.testing.assert_array_equal(self.tensor_x[3].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))

    def func_test_value_tensor(self):
        paddle.disable_static()
        self._test(self.tensor_value)

    def test_value_tensor(self):
        self.func_setUp()
        self.func_test_value_tensor()

    def func_test_value_numpy(self):
        paddle.disable_static()
        self._test(self.np_value)

    def test_value_numpy(self):
        self.func_setUp()
        self.func_test_value_numpy()

    def func_test_value_int(self):
        paddle.disable_static()
        self._test(10)

    def test_value_int(self):
        self.func_setUp()
        self.func_test_value_int()


class TestEagerTensorSetitemBoolScalarIndex(unittest.TestCase):
    def set_input(self):
        self.tensor_x = paddle.to_tensor(np.ones((1, 2, 3)).astype(self.dtype))
        self.np_value = np.random.random((2, 3)).astype(self.dtype)
        self.tensor_value = paddle.to_tensor(self.np_value)

    def _test(self, value):
        paddle.disable_static()
        self.assertEqual(self.tensor_x.inplace_version, 0)

        id_origin = id(self.tensor_x)
        index = paddle.to_tensor(np.array([True]))
        self.tensor_x[index] = value
        self.assertEqual(self.tensor_x.inplace_version, 1)

        if isinstance(value, (int, float)):
            result = np.zeros((2, 3)).astype(self.dtype) + value

        else:
            result = self.np_value

        np.testing.assert_array_equal(self.tensor_x[0].numpy(), result)
        self.assertEqual(id_origin, id(self.tensor_x))


class TestEagerTensorInplaceVersion(unittest.TestCase):
    def test_setitem(self):
        paddle.disable_static()

        var = paddle.ones(shape=[4, 2, 3], dtype="float32")
        self.assertEqual(var.inplace_version, 0)

        var[1] = 1
        self.assertEqual(var.inplace_version, 1)

        var[1:2] = 1
        self.assertEqual(var.inplace_version, 2)

    def test_bump_inplace_version(self):
        paddle.disable_static()
        var = paddle.ones(shape=[4, 2, 3], dtype="float32")
        self.assertEqual(var.inplace_version, 0)

        var._bump_inplace_version()
        self.assertEqual(var.inplace_version, 1)

        var._bump_inplace_version()
        self.assertEqual(var.inplace_version, 2)


class TestEagerTensorSlice(unittest.TestCase):
    def test_slice(self):
        paddle.disable_static()
        np_x = np.random.random((3, 8, 8))
        x = paddle.to_tensor(np_x, dtype="float64")
        actual_x = x._slice(0, 1)
        actual_x = paddle.to_tensor(actual_x)
        self.assertEqual(actual_x.numpy().all(), np_x[0:1].all())


class TestEagerTensorClear(unittest.TestCase):
    def test_clear(self):
        paddle.disable_static()
        np_x = np.random.random((3, 8, 8))
        x = paddle.to_tensor(np_x, dtype="float64")
        x._clear()
        self.assertEqual(str(x), "Tensor(Not initialized)")


class TestEagerTensorOffset(unittest.TestCase):
    def test_offset(self):
        paddle.disable_static()
        np_x = np.random.random((3, 8, 8))
        x = paddle.to_tensor(np_x, dtype="float64")
        expected_offset = 0
        actual_x = x._slice(expected_offset, 1)
        actual_x = paddle.to_tensor(actual_x)
        self.assertEqual(actual_x._offset(), expected_offset)


class TestEagerTensorShareBufferTo(unittest.TestCase):
    def test_share_buffer_To(self):
        paddle.disable_static()
        np_src = np.random.random((3, 8, 8))
        src = paddle.to_tensor(np_src, dtype="float64")
        # empty_var
        dst = core.eager.Tensor()
        src._share_buffer_to(dst)
        self.assertEqual(src._is_shared_buffer_with(dst), True)


class TestEagerTensorTo(unittest.TestCase):
    def func_setUp(self):
        paddle.disable_static()
        self.np_x = np.random.random((3, 8, 8))
        self.x = paddle.to_tensor(self.np_x, dtype="float32")

    def func_test_private_to_api(self):
        x_double = self.x._to(dtype="double")
        self.assertEqual(x_double.dtype, paddle.float64)
        np.testing.assert_allclose(self.np_x, x_double, rtol=1e-05)

        x_ = self.x._to()
        self.assertEqual(self.x.dtype, paddle.float32)
        np.testing.assert_allclose(self.np_x, x_, rtol=1e-05)

        if paddle.base.is_compiled_with_cuda():
            x_gpu = self.x._to(device=paddle.CUDAPlace(0))
            self.assertTrue(x_gpu.place.is_gpu_place())
            self.assertEqual(x_gpu.place.gpu_device_id(), 0)

            x_gpu0 = self.x._to(device="gpu:0")
            self.assertTrue(x_gpu0.place.is_gpu_place())
            self.assertEqual(x_gpu0.place.gpu_device_id(), 0)

            x_gpu1 = self.x._to(device="gpu:0", dtype="float64")
            self.assertTrue(x_gpu1.place.is_gpu_place())
            self.assertEqual(x_gpu1.place.gpu_device_id(), 0)
            self.assertEqual(x_gpu1.dtype, paddle.float64)

            x_gpu2 = self.x._to(device="gpu:0", dtype="float16")
            self.assertTrue(x_gpu2.place.is_gpu_place())
            self.assertEqual(x_gpu2.place.gpu_device_id(), 0)
            self.assertEqual(x_gpu2.dtype, paddle.float16)

        x_cpu = self.x._to(device=paddle.CPUPlace())
        self.assertTrue(x_cpu.place.is_cpu_place())

        x_cpu0 = self.x._to(device="cpu")
        self.assertTrue(x_cpu0.place.is_cpu_place())

        x_cpu1 = self.x._to(device=paddle.CPUPlace(), dtype="float64")
        self.assertTrue(x_cpu1.place.is_cpu_place())
        self.assertEqual(x_cpu1.dtype, paddle.float64)

        x_cpu2 = self.x._to(device="cpu", dtype="float16")
        self.assertTrue(x_cpu2.place.is_cpu_place())
        self.assertEqual(x_cpu2.dtype, paddle.float16)

        self.assertRaises(ValueError, self.x._to, device=1)
        self.assertRaises(AssertionError, self.x._to, blocking=1)

    def func_test_public_to_api(self):
        # test for Tensor.to
        dtypes = [
            paddle.int32,
            paddle.int64,
            paddle.float32,
            paddle.float64,
            paddle.complex64,
        ]
        places = [paddle.CPUPlace()]
        if paddle.base.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        for src_place, src_dtype in itertools.product(places, dtypes):
            src = paddle.to_tensor(
                [1.0, 2.0, 3.0], dtype=src_dtype, place=src_place
            )
            self.assertEqual(src.dtype, src_dtype)
            self.assertEqual(str(src.place), str(src_place))
            src_data_ptr = src.data_ptr()

            for dst_place, dst_dtype in itertools.product(places, dtypes):
                dst = src.to(dtype=dst_dtype, device=dst_place)
                # test for non-inplace operation
                self.assertEqual(src_data_ptr, src.data_ptr())

                # test for correctness of Tensor.to
                self.assertEqual(dst.dtype, dst_dtype)
                self.assertEqual(str(dst.place), str(dst_place))
                dst_data_ptr = dst.data_ptr()

                if src_place == dst_place and src_dtype == dst_dtype:
                    # just return self
                    self.assertEqual(src_data_ptr, dst_data_ptr)
                    self.assertEqual(src.dtype, dst.dtype)
                    self.assertEqual(str(src.place), str(dst.place))
                elif src_place != dst_place and src_dtype == dst_dtype:
                    # creating new tensor from src
                    self.assertNotEqual(src_data_ptr, dst_data_ptr)
                    self.assertEqual(src.dtype, dst.dtype)
                    self.assertNotEqual(str(src.place), str(dst.place))
                elif src_place == dst_place and src_dtype != dst_dtype:
                    # creating new tensor from src
                    self.assertNotEqual(src_data_ptr, dst_data_ptr)
                    self.assertNotEqual(src.dtype, dst.dtype)
                    self.assertEqual(str(src.place), str(dst.place))
                else:
                    # creating new tensor from src
                    self.assertNotEqual(src_data_ptr, dst_data_ptr)
                    self.assertNotEqual(src.dtype, dst.dtype)
                    self.assertNotEqual(str(src.place), str(dst.place))

    def test_to_api(self):
        self.func_setUp()
        self.func_test_private_to_api()
        self.func_test_public_to_api()


class TestEagerTensorInitEagerTensorFromTensorWithDevice(unittest.TestCase):
    def test_tensor_init(self):
        paddle.disable_static()
        t = base.Tensor()
        np_x = np.random.random((3, 8, 8))
        t.set(np_x, base.CPUPlace())

        if paddle.base.is_compiled_with_cuda():
            device = paddle.CUDAPlace(0)
            tmp = base.core.eager.Tensor(t, device)
            self.assertTrue(tmp.place.is_gpu_place())
            self.assertEqual(tmp.numpy().all(), np_x.all())

        device = paddle.CPUPlace()
        tmp = base.core.eager.Tensor(t, device)
        self.assertEqual(tmp.numpy().all(), np_x.all())


class TestEagerTensorNumel(unittest.TestCase):
    def test_numel_normal(self):
        paddle.disable_static()
        np_x = np.random.random((3, 8, 8))
        x = paddle.to_tensor(np_x, dtype="float64")
        x_actual_numel = x._numel()
        x_expected_numel = np.prod((3, 8, 8))
        self.assertEqual(x_actual_numel, x_expected_numel)

    def test_numel_without_holder(self):
        paddle.disable_static()
        x_without_holder = core.eager.Tensor()
        x_actual_numel = x_without_holder._numel()
        self.assertEqual(x_actual_numel, 0)


class TestEagerTensorCopyGradientFrom(unittest.TestCase):
    def test_copy_gradient_from(self):
        paddle.disable_static()
        np_x = np.random.random((2, 2))
        np_y = np.random.random((2, 2))
        x = paddle.to_tensor(np_x, dtype="float64", stop_gradient=False)
        y = paddle.to_tensor(np_y, dtype="float64")
        out = x + x
        out.backward()
        x._copy_gradient_from(y)
        self.assertEqual(x.grad.numpy().all(), np_y.all())


class TestEagerTensorGradNameValue(unittest.TestCase):
    def test_eager_tensor_grad_name_value(self):
        a_np = np.array([2, 3]).astype("float32")
        a = paddle.to_tensor(a_np)
        a.stop_gradient = False
        b = a**2
        self.assertIsNone(a._grad_value())
        b.backward()
        # Note, for new dygraph, there are no generated grad name, so we skip the name check.
        self.assertIsNotNone(a._grad_value())


class TestDenseTensorToTensor(unittest.TestCase):
    def test_same_place_data_ptr_consistency(self):
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            x = paddle.rand([3, 5]).to(device=place)
            x_dense = x.get_tensor()
            y = paddle.to_tensor(x_dense, place=place)

            self.assertEqual(x.data_ptr(), y.data_ptr())


class TestSetDynamicAttributeToEagerTensorInstance(unittest.TestCase):
    def test_set_dynamic_attribute_to_eager_tensor_instance_create_via_constructor(
        self,
    ):
        tensor_instance = paddle.to_tensor(1.0)
        tensor_instance._custom_id = 0
        self.assertEqual(tensor_instance._custom_id, 0)
        self.assertEqual(tensor_instance.__dict__["_custom_id"], 0)

    def test_set_dynamic_attribute_to_eager_tensor_instance_create_via_to_pyobject(
        self,
    ):
        original_tensor = paddle.to_tensor(-1.0)
        tensor_instance = paddle.abs(original_tensor)
        tensor_instance._custom_flag = True
        self.assertEqual(tensor_instance._custom_flag, True)
        self.assertEqual(tensor_instance.__dict__["_custom_flag"], True)


if __name__ == "__main__":
    unittest.main()
