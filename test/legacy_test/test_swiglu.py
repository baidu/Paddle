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

import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.incubate.nn.functional import swiglu as fused_swiglu_impl


def swiglu(x, y, out_grad):
    if isinstance(x, np.ndarray):
        x = paddle.to_tensor(x)
        y = paddle.to_tensor(y)
        out_grad = paddle.to_tensor(out_grad)

    origin_x = x.detach().clone()
    origin_x.stop_gradient = False
    x = origin_x

    origin_y = y.detach().clone()
    origin_y.stop_gradient = False
    y = origin_y

    dtype = x.dtype
    need_convert = False
    assert dtype == y.dtype
    output_dtype = dtype
    if paddle.is_compiled_with_cuda():
        if dtype in [paddle.float16, paddle.bfloat16]:
            output_dtype = paddle.float32
            x = x.astype(output_dtype)
            y = y.astype(output_dtype)
            need_convert = True

    out = F.silu(x) * y
    if need_convert:
        out = out.astype(dtype)
    out.backward(out_grad)
    ret = [
        out.astype(output_dtype),
        origin_x.grad.astype(output_dtype),
        origin_y.grad.astype(output_dtype),
    ]
    return ret


def fused_swiglu(x, y, out_grad):
    x = x.detach().clone()
    x.stop_gradient = False
    if y is not None:
        y = y.detach().clone()
        y.stop_gradient = False
    out = fused_swiglu_impl(x, y)
    out.backward(out_grad)

    output_dtype = x.dtype
    if paddle.is_compiled_with_cuda():
        if x.dtype in [paddle.float16, paddle.bfloat16]:
            output_dtype = paddle.float32
    ret = [
        out.astype(output_dtype),
    ]
    if y is not None:
        x_grad, y_grad = x.grad, y.grad
    else:
        x_grad, y_grad = paddle.split(x.grad, 2, axis=-1)

    ret.append(x_grad.astype(output_dtype))
    ret.append(y_grad.astype(output_dtype))
    return ret


tol_map = {
    paddle.float64: [1e-8, 1e-8],
    paddle.float32: [1e-6, 1e-6],
    paddle.float16: [1e-3, 1e-3],
    paddle.bfloat16: [1e-3, 1e-3],
}


class TestSwiGLUDygraph(unittest.TestCase):
    def check_dygraph_impl(self, device, shape, dtype):
        x = paddle.randn(shape, dtype=dtype)
        y = paddle.randn(shape, dtype=dtype)
        out_grad = paddle.randn(shape, dtype=dtype)

        ret1 = swiglu(x, y, out_grad)
        ret2 = fused_swiglu(x, y, out_grad)
        ret3 = fused_swiglu(paddle.concat([x, y], axis=-1), None, out_grad)

        atol, rtol = tol_map[dtype]
        err_msg = (
            f"Failed when device = {device}, dtype = {dtype}, shape = {shape}"
        )
        for t1, t2, t3 in zip(ret1, ret2, ret3):
            t1, t2, t3 = t1.numpy(), t2.numpy(), t3.numpy()
            np.testing.assert_allclose(
                t1, t2, atol=atol, rtol=rtol, err_msg=err_msg
            )
            np.testing.assert_equal(t2, t3, err_msg=err_msg)

    def check_dygraph(self, shape):
        metas = [('cpu', paddle.float32), ('cpu', paddle.float64)]
        if paddle.is_compiled_with_cuda():
            metas.append(('gpu', paddle.float32))
            metas.append(('gpu', paddle.float64))
            metas.append(('gpu', paddle.float16))
            prop = paddle.device.cuda.get_device_properties()
            if prop.major >= 8:
                metas.append(('gpu', paddle.bfloat16))

        for device, dtype in metas:
            origin_device = paddle.get_device()
            paddle.set_device(device)
            for with_split in [True]:
                self.check_dygraph_impl(device, shape, dtype)
            paddle.set_device(origin_device)

    def check_static_graph(self, shape, dtype="float32"):
        x = paddle.static.data(name='x', shape=shape, dtype=dtype)
        y = paddle.static.data(name='y', shape=shape, dtype=dtype)
        concated_x = paddle.static.data(
            name='concated_x',
            shape=[*shape[:-1], shape[-1] * 2],
            dtype=dtype,
        )
        out1 = fused_swiglu_impl(x, y)
        out2 = fused_swiglu_impl(concated_x)

        concated_x_np = np.random.random(concated_x.shape).astype(dtype)
        x_np, y_np = np.split(concated_x_np, 2, axis=-1)

        exe = paddle.static.Executor()
        t1, t2 = exe.run(
            feed={'x': x_np, 'y': y_np, 'concated_x': concated_x_np},
            fetch_list=[out1, out2],
        )
        np.testing.assert_equal(t1, t2)

    def check_main(self, shape):
        self.check_dygraph(shape)
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            self.check_static_graph(shape)
        paddle.disable_static()

    def test_main(self):
        self.check_main([8, 100])
        self.check_main([4, 101])


class TestSwigluOp(OpTest):
    def config(self):
        self.x_shape = (8, 128)
        self.check_auto_parallel = True

    def setUp(self):
        self.config()
        self.op_type = "swiglu"
        self.prim_op_type = "comp"
        self.python_api = fused_swiglu_impl
        self.public_python_api = fused_swiglu_impl
        x = np.random.uniform(-1, 1, self.x_shape).astype("float64")
        y = np.random.uniform(-1, 1, self.x_shape).astype("float64")
        out_grad = np.random.uniform(-1, 1, self.x_shape).astype("float64")
        res = swiglu(x, y, out_grad)
        self.inputs = {'x': x, 'y': y}
        self.outputs = {'out': res[0].numpy()}
        self.placements = {
            'x': [dist.Shard(1)],
            'y': [dist.Shard(1)],
            'out': [dist.Shard(1)],
        }

    def test_check_output(self):
        self.check_output(check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['x', 'y'],
            'out',
            check_auto_parallel=self.check_auto_parallel,
            check_dygraph=1,
            check_prim_pir=True,
        )


class TestSwigluOp2(TestSwigluOp):
    def setUp(self):
        self.config()
        self.op_type = "swiglu"
        self.prim_op_type = "comp"
        self.python_api = fused_swiglu_impl
        self.public_python_api = fused_swiglu_impl
        x = np.random.uniform(-1, 1, self.x_shape).astype("float64")
        tmp_inputs = np.split(x, 2, axis=-1)
        x = tmp_inputs[0]
        y = tmp_inputs[1]
        out_grad = np.random.uniform(-1, 1, x.shape).astype("float64")
        res = swiglu(x, y, out_grad)
        self.inputs = {'x': x, 'y': y}
        self.outputs = {'out': res[0].numpy()}
        self.placements = {
            'x': [dist.Shard(1)],
            'y': [dist.Shard(1)],
            'out': [dist.Shard(1)],
        }


@unittest.skipIf(
    not paddle.base.core.is_compiled_with_dist(),
    "The spmd rule is should be tested with distributed=ON",
)
class TestSwigluSpmd(unittest.TestCase):
    def setUp(self):
        self.kernel = 'swiglu'
        self.rule = paddle.base.core.get_phi_spmd_rule(self.kernel)
        x_shape = [64, 32]
        process_mesh = dist.ProcessMesh(mesh=[0, 1, 2, 3])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.y_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

    def test_input_x_y(self):
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.y_dist_tensor_spec
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])

    def test_input_x_unshard_last_dim(self):
        x_shape = [64, 32]
        process_mesh = dist.ProcessMesh(mesh=[0, 1, 2, 3])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [0, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, DistTensorSpec()
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1])


if __name__ == "__main__":
    unittest.main()
