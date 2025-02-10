#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core
from paddle.framework import in_dynamic_mode


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool2D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
    data_type=np.float64,
):
    if data_type == np.float64 and core.is_compiled_with_rocm():
        data_type = np.float32
    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    if adaptive:
        H_out, W_out = ksize
    else:
        H_out = (
            (H - ksize[0] + 2 * paddings[0] + strides[0] - 1) // strides[0] + 1
            if ceil_mode
            else (H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        )
        W_out = (
            (W - ksize[1] + 2 * paddings[1] + strides[1] - 1) // strides[1] + 1
            if ceil_mode
            else (W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        )
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            if adaptive:
                r_start = adaptive_start_index(i, H, ksize[0])
                r_end = adaptive_end_index(i, H, ksize[0])
                c_start = adaptive_start_index(j, W, ksize[1])
                c_end = adaptive_end_index(j, W, ksize[1])
            else:
                r_start = np.max((i * strides[0] - paddings[0], 0))
                r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
                c_start = np.max((j * strides[1] - paddings[1], 0))
                c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
    return out


def avg_pool2D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
    data_type=np.float64,
):
    if data_type == np.float64 and core.is_compiled_with_rocm():
        data_type = np.float32
    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    if adaptive:
        H_out, W_out = ksize
    else:
        H_out = (
            (H - ksize[0] + 2 * paddings[0] + strides[0] - 1) // strides[0] + 1
            if ceil_mode
            else (H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        )
        W_out = (
            (W - ksize[1] + 2 * paddings[1] + strides[1] - 1) // strides[1] + 1
            if ceil_mode
            else (W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        )
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            if adaptive:
                r_start = adaptive_start_index(i, H, ksize[0])
                r_end = adaptive_end_index(i, H, ksize[0])
                c_start = adaptive_start_index(j, W, ksize[1])
                c_end = adaptive_end_index(j, W, ksize[1])
            else:
                r_start = i * strides[0] - paddings[0]
                r_end = i * strides[0] + ksize[0] - paddings[0]
                c_start = j * strides[1] - paddings[1]
                c_end = j * strides[1] + ksize[1] - paddings[1]
                field_size = (r_end - r_start) * (c_end - c_start)
                r_start = np.max((r_start, 0))
                r_end = np.min((r_end, H))
                c_start = np.max((c_start, 0))
                c_end = np.min((c_end, W))

            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            if exclusive or adaptive:
                field_size = (r_end - r_start) * (c_end - c_start)

            if data_type == np.int8 or data_type == np.uint8:
                out[:, :, i, j] = (
                    np.rint(np.sum(x_masked, axis=(2, 3)) / field_size)
                ).astype(data_type)
            else:
                out[:, :, i, j] = (
                    np.sum(x_masked, axis=(2, 3)) / field_size
                ).astype(data_type)
    return out


def pool2D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=True,
    adaptive=False,
    data_format='NCHW',
    pool_type="max",
    padding_algorithm="EXPLICIT",
    norm_type=0,
):
    if norm_type == float("inf"):
        pool_type = 'max'

    # update paddings
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(
            input_shape, pool_size, pool_stride
        ):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(
                ((out_size - 1) * stride_size + filter_size - input_size, 0)
            )
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    if isinstance(padding_algorithm, str):
        padding_algorithm = padding_algorithm.upper()
        if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
            raise ValueError(
                f"Unknown Attr(padding_algorithm): '{padding_algorithm}'. "
                "It can only be 'SAME' or 'VALID'."
            )

        if padding_algorithm == "VALID":
            paddings = [0, 0, 0, 0]
            if ceil_mode is not False:
                raise ValueError(
                    'When Attr(pool_padding) is "VALID", Attr(ceil_mode)'
                    " must be False. "
                    "Received ceil_mode: True."
                )
        elif padding_algorithm == "SAME":
            input_data_shape = []
            if data_format == "NCHW":
                input_data_shape = x.shape[2:4]
            elif data_format == "NHWC":
                input_data_shape = x.shape[1:3]
            paddings = _get_padding_with_SAME(input_data_shape, ksize, strides)

    assert len(paddings) == 2 or len(paddings) == 4
    is_sys = True if len(paddings) == 2 else False

    N = x.shape[0]
    C, H, W = (
        [x.shape[1], x.shape[2], x.shape[3]]
        if data_format == 'NCHW'
        else [x.shape[3], x.shape[1], x.shape[2]]
    )

    if global_pool == 1:
        ksize = [H, W]
        paddings = [0 for _ in range(len(paddings))]

    pad_h_up = paddings[0] if is_sys else paddings[0]
    pad_h_down = paddings[0] if is_sys else paddings[1]
    pad_w_left = paddings[1] if is_sys else paddings[2]
    pad_w_right = paddings[1] if is_sys else paddings[3]

    if adaptive:
        H_out, W_out = ksize
    else:
        H_out = (
            (H - ksize[0] + pad_h_up + pad_h_down + strides[0] - 1)
            // strides[0]
            + 1
            if ceil_mode
            else (H - ksize[0] + pad_h_up + pad_h_down) // strides[0] + 1
        )
        W_out = (
            (W - ksize[1] + pad_w_left + pad_w_right + strides[1] - 1)
            // strides[1]
            + 1
            if ceil_mode
            else (W - ksize[1] + pad_w_left + pad_w_right) // strides[1] + 1
        )

    out = (
        np.zeros((N, C, H_out, W_out))
        if data_format == 'NCHW'
        else np.zeros((N, H_out, W_out, C))
    )
    for i in range(H_out):
        if adaptive:
            in_h_start = adaptive_start_index(i, H, ksize[0])
            in_h_end = adaptive_end_index(i, H, ksize[0])
        else:
            in_h_start = np.max((i * strides[0] - pad_h_up, 0))
            in_h_end = np.min((i * strides[0] + ksize[0] - pad_h_up, H))

        for j in range(W_out):
            if adaptive:
                in_w_start = adaptive_start_index(j, W, ksize[1])
                in_w_end = adaptive_end_index(j, W, ksize[1])
            else:
                in_h_start = i * strides[0] - pad_h_up
                in_w_start = j * strides[1] - pad_w_left
                in_h_end = i * strides[0] + ksize[0] - pad_h_up
                in_w_end = j * strides[1] + ksize[1] - pad_w_left

                field_size = (in_h_end - in_h_start) * (in_w_end - in_w_start)
                in_h_start = np.max((in_h_start, 0))
                in_w_start = np.max((in_w_start, 0))
                in_h_end = np.min((in_h_end, H))
                in_w_end = np.min((in_w_end, W))

            if data_format == 'NCHW':
                x_masked = x[:, :, in_h_start:in_h_end, in_w_start:in_w_end]
                if pool_type == 'avg':
                    if exclusive or adaptive:
                        field_size = (in_h_end - in_h_start) * (
                            in_w_end - in_w_start
                        )

                    #                         if (exclusive or adaptive) else (ksize[0] * ksize[1])
                    out[:, :, i, j] = np.sum(x_masked, axis=(2, 3)) / field_size
                elif pool_type == 'max':
                    out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
                else:  # lp_pool2d
                    if norm_type == 0:
                        out[:, :, i, j] = 1
                    else:
                        out[:, :, i, j] = np.power(
                            np.sum(np.power(x_masked, norm_type), axis=(2, 3)),
                            1.0 / norm_type,
                        )
            elif data_format == 'NHWC':
                x_masked = x[:, in_h_start:in_h_end, in_w_start:in_w_end, :]
                if pool_type == 'avg':
                    if exclusive or adaptive:
                        field_size = (in_h_end - in_h_start) * (
                            in_w_end - in_w_start
                        )
                    out[:, i, j, :] = np.sum(x_masked, axis=(1, 2)) / field_size
                elif pool_type == 'max':
                    out[:, i, j, :] = np.max(x_masked, axis=(1, 2))
                else:  # lp_pool2d
                    if norm_type == 0:
                        out[:, i, j, :] = 1
                    else:
                        out[:, i, j, :] = np.power(
                            np.sum(np.power(x_masked, norm_type), axis=(2, 3)),
                            1.0 / norm_type,
                        )
    return out


def pool2d_wrapper_not_use_cudnn(
    X,
    ksize=[],
    strides=[],
    paddings=[],
    ceil_mode=False,
    exclusive=True,
    data_format="NCDHW",
    pooling_type="max",
    global_pooling=False,
    adaptive=False,
    padding_algorithm="EXPLICIT",
):
    if in_dynamic_mode():
        X = X._use_gpudnn(False)
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.pool2d(
        X,
        ksize,
        strides,
        paddings,
        ceil_mode,
        exclusive,
        data_format,
        pooling_type,
        global_pooling,
        adaptive,
        padding_algorithm,
    )


def pool2d_wrapper_use_cudnn(
    X,
    ksize=[],
    strides=[],
    paddings=[],
    ceil_mode=False,
    exclusive=True,
    data_format="NCDHW",
    pooling_type="max",
    global_pooling=False,
    adaptive=False,
    padding_algorithm="EXPLICIT",
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.pool2d(
        X,
        ksize,
        strides,
        paddings,
        ceil_mode,
        exclusive,
        data_format,
        pooling_type,
        global_pooling,
        adaptive,
        padding_algorithm,
    )


def lp_pool2d_wrapper(
    X,
    ksize=[],
    strides=[],
    paddings=[],
    ceil_mode=False,
    exclusive=True,
    data_format="NCDHW",
    pooling_type="lp",
    global_pooling=False,
    adaptive=False,
    padding_algorithm="EXPLICIT",
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.lp_pool2d(
        X,
        ksize,
        strides,
        paddings,
        ceil_mode,
        exclusive,
        data_format,
        pooling_type,
        global_pooling,
        adaptive,
        padding_algorithm,
        2,
    )


class TestPool2D_Op_Mixin:
    def setUp(self):
        self.op_type = "pool2d"
        self.use_cudnn = False
        self.init_kernel_type()
        self.use_mkldnn = False
        self.init_data_type()
        self.init_test_case()
        self.padding_algorithm = "EXPLICIT"
        self.init_paddings()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()
        self.init_data_format()
        self.init_shape()

        if self.is_bfloat16_op():
            input = np.random.random(self.shape).astype(np.float32)
        else:
            input = np.random.random(self.shape).astype(self.dtype)

        output = pool2D_forward_naive(
            input,
            self.ksize,
            self.strides,
            self.paddings,
            self.global_pool,
            self.ceil_mode,
            self.exclusive,
            self.adaptive,
            self.data_format,
            self.pool_type,
            self.padding_algorithm,
        )

        if self.is_bfloat16_op():
            output = convert_float_to_uint16(output)
            self.inputs = {'X': convert_float_to_uint16(input)}
        else:
            output = output.astype(self.dtype)
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(input)}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            'exclusive': self.exclusive,
            'adaptive': self.adaptive,
            "padding_algorithm": self.padding_algorithm,
        }

        self.outputs = {'Out': output}

        if self.use_cudnn:
            self.python_api = pool2d_wrapper_use_cudnn
        else:
            self.python_api = pool2d_wrapper_not_use_cudnn

    def has_cudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.has_cudnn():
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place,
                atol=1e-5,
                check_dygraph=(not self.use_mkldnn),
                check_cinn=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        else:
            self.check_output(
                check_dygraph=(not self.use_mkldnn),
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        # TODO(wangzhongpu): support onednn op in dygraph mode
        if self.has_cudnn() and self.pool_type != "max":
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                {'X'},
                'Out',
                check_dygraph=(not self.use_mkldnn),
                check_cinn=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        elif self.pool_type != "max":
            self.check_grad(
                {'X'},
                'Out',
                max_relative_error=0.07,
                check_dygraph=(not self.use_mkldnn),
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_shape(self):
        self.shape = [2, 3, 5, 5]

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_kernel_type(self):
        self.use_cudnn = False

    def init_data_type(self):
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False

    def init_exclusive(self):
        self.exclusive = True

    def init_adaptive(self):
        self.adaptive = False


class TestPool2D_Op(TestPool2D_Op_Mixin, OpTest):
    pass


class TestLPPool2D_Op(TestPool2D_Op):
    def setUp(self):
        self.op_type = "lp_pool2d"
        self.use_cudnn = False
        self.init_kernel_type()
        self.use_mkldnn = False
        self.init_data_type()
        self.init_test_case()
        self.padding_algorithm = "EXPLICIT"
        self.init_paddings()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()
        self.init_data_format()
        self.init_shape()
        self.norm_type = 2
        self.pool_type = 'lp'

        if self.is_bfloat16_op():
            input = np.random.random(self.shape).astype(np.float32)
        else:
            input = np.random.random(self.shape).astype(self.dtype)

        output = pool2D_forward_naive(
            input,
            self.ksize,
            self.strides,
            self.paddings,
            self.global_pool,
            self.ceil_mode,
            self.exclusive,
            self.adaptive,
            self.data_format,
            self.pool_type,
            self.padding_algorithm,
            self.norm_type,
        )

        if self.is_bfloat16_op():
            output = convert_float_to_uint16(output)
            self.inputs = {'x': convert_float_to_uint16(input)}
        else:
            output = output.astype(self.dtype)
            self.inputs = {'x': OpTest.np_dtype_to_base_dtype(input)}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'kernel_size': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            "padding_algorithm": self.padding_algorithm,
            'norm_type': self.norm_type,
        }

        self.outputs = {'out': output}

        self.python_api = lp_pool2d_wrapper

    def has_cudnn(self):
        return False

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            {'x'},
            'out',
            max_relative_error=0.07,
            check_dygraph=(not self.use_mkldnn),
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


class TestCase1(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [0, 0]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase2(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [1, 1]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase3(TestPool2D_Op):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase4(TestCase1):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase5(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


# --------------------test pool2d cudnn--------------------


def create_test_cudnn_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

    cls_name = "{}_{}".format(parent.__name__, "CUDNNOp")
    TestCUDNNCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNCase


create_test_cudnn_class(TestPool2D_Op)
create_test_cudnn_class(TestCase1)
create_test_cudnn_class(TestCase2)
create_test_cudnn_class(TestCase3)
create_test_cudnn_class(TestCase4)
create_test_cudnn_class(TestCase5)

# --------------------test pool2d cudnn_fp16--------------------


def create_test_cudnn_fp16_class(parent, check_grad=True):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNFp16Case(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = np.float16

        def test_check_output(self):
            # TODO(wangzhongpu): support onednn op in dygraph mode
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(
                        place,
                        check_dygraph=(not self.use_mkldnn),
                        check_cinn=True,
                        check_pir_onednn=self.check_pir_onednn,
                    )

        def test_check_grad(self):
            # TODO(wangzhongpu): support onednn op in dygraph mode
            place = core.CUDAPlace(0)
            if (
                core.is_float16_supported(place)
                and self.pool_type != "max"
                and check_grad
            ):
                self.check_grad_with_place(
                    place,
                    {'X'},
                    'Out',
                    check_dygraph=(not self.use_mkldnn),
                    check_cinn=True,
                    check_pir_onednn=self.check_pir_onednn,
                )

    cls_name = "{}_{}".format(parent.__name__, "CUDNNFp16Op")
    TestCUDNNFp16Case.__name__ = cls_name
    globals()[cls_name] = TestCUDNNFp16Case


def create_test_fp16_class(parent, check_grad=True):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestFp16Case(parent):
        def init_kernel_type(self):
            self.use_cudnn = False
            self.dtype = np.float16

        def test_check_output(self):
            # TODO(wangzhongpu): support onednn op in dygraph mode
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(
                        place,
                        check_dygraph=(not self.use_mkldnn),
                        check_cinn=True,
                        check_pir_onednn=self.check_pir_onednn,
                    )

        def test_check_grad(self):
            # TODO(wangzhongpu): support onednn op in dygraph mode
            place = core.CUDAPlace(0)
            if (
                core.is_float16_supported(place)
                and self.pool_type != "max"
                and check_grad
            ):
                self.check_grad_with_place(
                    place,
                    {'X'},
                    'Out',
                    check_dygraph=(not self.use_mkldnn),
                    check_cinn=True,
                    check_pir_onednn=self.check_pir_onednn,
                )

    cls_name = "{}_{}".format(parent.__name__, "Fp16Op")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


def create_test_bf16_class(parent, check_grad=True):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestBf16Case(parent):
        def init_kernel_type(self):
            self.use_cuda = True
            self.dtype = np.uint16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                self.check_output_with_place(
                    place,
                    check_dygraph=(not self.use_mkldnn),
                    check_cinn=True,
                    check_pir_onednn=self.check_pir_onednn,
                )

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if self.pool_type != "max" and check_grad:
                self.check_grad_with_place(
                    place,
                    {'X'},
                    'Out',
                    check_dygraph=(not self.use_mkldnn),
                    check_cinn=True,
                    check_pir_onednn=self.check_pir_onednn,
                )

    cls_name = "{}_{}".format(parent.__name__, "Bf16Op")
    TestBf16Case.__name__ = cls_name
    globals()[cls_name] = TestBf16Case


create_test_cudnn_fp16_class(TestPool2D_Op)
create_test_cudnn_fp16_class(TestCase1)
create_test_cudnn_fp16_class(TestCase2)
create_test_cudnn_fp16_class(TestCase3)
create_test_cudnn_fp16_class(TestCase4)
create_test_cudnn_fp16_class(TestCase5)

create_test_fp16_class(TestPool2D_Op)
create_test_fp16_class(TestCase1)
create_test_fp16_class(TestCase2)
create_test_fp16_class(TestCase3)
create_test_fp16_class(TestCase4)
create_test_fp16_class(TestCase5)

create_test_bf16_class(TestPool2D_Op)
create_test_bf16_class(TestCase1)
create_test_bf16_class(TestCase2)
create_test_bf16_class(TestCase3)
create_test_bf16_class(TestCase4)
create_test_bf16_class(TestCase5)
# --------------------test pool2d use ceil mode--------------------


def create_test_cudnn_use_ceil_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestPool2DUseCeilCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{}_{}".format(parent.__name__, "CUDNNOpCeilMode")
    TestPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestPool2DUseCeilCase


create_test_cudnn_use_ceil_class(TestPool2D_Op)
create_test_cudnn_use_ceil_class(TestCase1)


def create_test_use_ceil_class(parent):
    class TestPool2DUseCeilCase(parent):
        def init_ceil_mode(self):
            self.ceil_mode = True

    cls_name = "{}_{}".format(parent.__name__, "CeilModeCast")
    TestPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestPool2DUseCeilCase


create_test_use_ceil_class(TestCase1)
create_test_use_ceil_class(TestCase2)


class TestAvgInclude(TestCase2):
    def init_exclusive(self):
        self.exclusive = False


class TestCUDNNAvgInclude(TestCase2):
    def init_kernel_type(self):
        self.use_cudnn = True

    def init_exclusive(self):
        self.exclusive = False


class TestAvgPoolAdaptive(TestCase1):
    def init_adaptive(self):
        self.adaptive = True


class TestAvgPoolAdaptiveAsyOutSize(TestCase1):
    def init_adaptive(self):
        self.adaptive = True

    def init_shape(self):
        self.shape = [8, 3, 6, 6]

    def init_test_case(self):
        self.ksize = [2, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0, 0, 0]


# -------test pool2d with asymmetric padding-----


class TestPool2D_AsyPadding(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 5, 5]


class TestCase1_AsyPadding(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 0]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase2_AsyPadding(TestCase2):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase3_AsyPadding(TestCase3):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 5, 5]


class TestCase4_AsyPadding(TestCase4):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 0, 1, 0]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCase5_AsyPadding(TestCase5):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [2, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


create_test_cudnn_class(TestPool2D_AsyPadding)
create_test_cudnn_class(TestCase1_AsyPadding)
create_test_cudnn_class(TestCase2_AsyPadding)
create_test_cudnn_class(TestCase3_AsyPadding)
create_test_cudnn_class(TestCase4_AsyPadding)
create_test_cudnn_class(TestCase5_AsyPadding)

create_test_cudnn_fp16_class(TestPool2D_AsyPadding)
create_test_cudnn_fp16_class(TestCase1_AsyPadding)
create_test_cudnn_fp16_class(TestCase2_AsyPadding)
create_test_cudnn_fp16_class(TestCase3_AsyPadding)
create_test_cudnn_fp16_class(TestCase4_AsyPadding)
create_test_cudnn_fp16_class(TestCase5_AsyPadding)

create_test_fp16_class(TestPool2D_AsyPadding)
create_test_fp16_class(TestCase1_AsyPadding)
create_test_fp16_class(TestCase2_AsyPadding)
create_test_fp16_class(TestCase3_AsyPadding)
create_test_fp16_class(TestCase4_AsyPadding)
create_test_fp16_class(TestCase5_AsyPadding)

create_test_bf16_class(TestPool2D_AsyPadding)
create_test_bf16_class(TestCase1_AsyPadding)
create_test_bf16_class(TestCase2_AsyPadding)
create_test_bf16_class(TestCase3_AsyPadding)
create_test_bf16_class(TestCase4_AsyPadding)
create_test_bf16_class(TestCase5_AsyPadding)

create_test_cudnn_use_ceil_class(TestPool2D_AsyPadding)
create_test_cudnn_use_ceil_class(TestCase1_AsyPadding)

create_test_use_ceil_class(TestCase1_AsyPadding)
create_test_use_ceil_class(TestCase2_AsyPadding)


class TestAvgInclude_AsyPadding(TestCase2):
    def init_exclusive(self):
        self.exclusive = False

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 2, 1, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestCUDNNAvgInclude_AsyPadding(TestCase2):
    def init_kernel_type(self):
        self.use_cudnn = True

    def init_exclusive(self):
        self.exclusive = False

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [2, 1, 1, 1]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


class TestAvgPoolAdaptive_AsyPadding(TestCase1):
    def init_adaptive(self):
        self.adaptive = True

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1, 0, 2]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]


# ----------- test channel_last --------------
class TestPool2D_channel_last(TestPool2D_Op):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase1_channel_last(TestCase1):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase2_channel_last(TestCase2):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase3_channel_last(TestCase3):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase4_channel_last(TestCase4):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase5_channel_last(TestCase5):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


create_test_cudnn_class(TestPool2D_channel_last)
create_test_cudnn_class(TestCase1_channel_last)
create_test_cudnn_class(TestCase2_channel_last)
create_test_cudnn_class(TestCase3_channel_last)
create_test_cudnn_class(TestCase4_channel_last)
create_test_cudnn_class(TestCase5_channel_last)

create_test_cudnn_fp16_class(TestPool2D_channel_last)
create_test_cudnn_fp16_class(TestCase1_channel_last)
create_test_cudnn_fp16_class(TestCase2_channel_last)
create_test_cudnn_fp16_class(TestCase3_channel_last)
create_test_cudnn_fp16_class(TestCase4_channel_last)
create_test_cudnn_fp16_class(TestCase5_channel_last)

create_test_fp16_class(TestPool2D_channel_last)
create_test_fp16_class(TestCase1_channel_last)
create_test_fp16_class(TestCase2_channel_last)
create_test_fp16_class(TestCase3_channel_last)
create_test_fp16_class(TestCase4_channel_last)
create_test_fp16_class(TestCase5_channel_last)

create_test_bf16_class(TestPool2D_channel_last)
create_test_bf16_class(TestCase1_channel_last)
create_test_bf16_class(TestCase2_channel_last)
create_test_bf16_class(TestCase3_channel_last)
create_test_bf16_class(TestCase4_channel_last)
create_test_bf16_class(TestCase5_channel_last)

create_test_cudnn_use_ceil_class(TestPool2D_channel_last)
create_test_cudnn_use_ceil_class(TestCase1_channel_last)

create_test_use_ceil_class(TestCase1_channel_last)
create_test_use_ceil_class(TestCase2_channel_last)


class TestCase5_Max(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.has_cudnn() and self.pool_type == "max":
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                {'X'},
                'Out',
                max_relative_error=1.00,
                check_cinn=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )
        elif self.pool_type == "max":
            self.check_grad(
                {'X'},
                'Out',
                max_relative_error=1.00,
                check_cinn=True,
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )


class TestCase5_channel_last_Max(TestCase5_Max):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


create_test_cudnn_class(TestCase5_Max)
create_test_cudnn_class(TestCase5_channel_last_Max)


class TestAvgInclude_channel_last(TestCase2_channel_last):
    def init_exclusive(self):
        self.exclusive = False


class TestCUDNNAvgInclude_channel_last(TestCase2_channel_last):
    def init_kernel_type(self):
        self.use_cudnn = True

    def init_exclusive(self):
        self.exclusive = False


class TestAvgPoolAdaptive_channel_last(TestCase1_channel_last):
    def init_adaptive(self):
        self.adaptive = True


class TestPool2D_AsyPadding_channel_last(TestPool2D_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase1_AsyPadding_channel_last(TestCase1_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase2_AsyPadding_channel_last(TestCase2_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase3_AsyPadding_channel_last(TestCase3_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 5, 5, 3]


class TestCase4_AsyPadding_channel_last(TestCase4_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCase5_AsyPadding_channel_last(TestCase5_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


create_test_cudnn_class(TestPool2D_AsyPadding_channel_last)
create_test_cudnn_class(TestCase1_AsyPadding_channel_last)
create_test_cudnn_class(TestCase2_AsyPadding_channel_last)
create_test_cudnn_class(TestCase3_AsyPadding_channel_last)
create_test_cudnn_class(TestCase4_AsyPadding_channel_last)
create_test_cudnn_class(TestCase5_AsyPadding_channel_last)

create_test_cudnn_fp16_class(TestPool2D_AsyPadding_channel_last)
create_test_cudnn_fp16_class(TestCase1_AsyPadding_channel_last)
create_test_cudnn_fp16_class(TestCase2_AsyPadding_channel_last)
create_test_cudnn_fp16_class(TestCase3_AsyPadding_channel_last)
create_test_cudnn_fp16_class(TestCase4_AsyPadding_channel_last)
create_test_cudnn_fp16_class(TestCase5_AsyPadding_channel_last)

create_test_fp16_class(TestPool2D_AsyPadding_channel_last)
create_test_fp16_class(TestCase1_AsyPadding_channel_last)
create_test_fp16_class(TestCase2_AsyPadding_channel_last)
create_test_fp16_class(TestCase3_AsyPadding_channel_last)
create_test_fp16_class(TestCase4_AsyPadding_channel_last)
create_test_fp16_class(TestCase5_AsyPadding_channel_last)

create_test_bf16_class(TestPool2D_AsyPadding_channel_last)
create_test_bf16_class(TestCase1_AsyPadding_channel_last)
create_test_bf16_class(TestCase2_AsyPadding_channel_last)
create_test_bf16_class(TestCase3_AsyPadding_channel_last)
create_test_bf16_class(TestCase4_AsyPadding_channel_last)
create_test_bf16_class(TestCase5_AsyPadding_channel_last)

create_test_cudnn_use_ceil_class(TestPool2D_AsyPadding_channel_last)
create_test_cudnn_use_ceil_class(TestCase1_AsyPadding_channel_last)

create_test_use_ceil_class(TestCase1_AsyPadding_channel_last)
create_test_use_ceil_class(TestCase2_AsyPadding_channel_last)


class TestAvgInclude_AsyPadding_channel_last(TestAvgInclude_AsyPadding):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestCUDNNAvgInclude_AsyPadding_channel_last(
    TestCUDNNAvgInclude_AsyPadding
):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


class TestAvgPoolAdaptive_AsyPadding_channel_last(
    TestAvgPoolAdaptive_AsyPadding
):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


# test paddings: SAME VALID


def create_test_padding_SAME_class(parent):
    class TestPaddingSAMECase(parent):
        def init_paddings(self):
            self.paddings = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{}_{}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSAMECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSAMECase


create_test_padding_SAME_class(TestPool2D_Op)
create_test_padding_SAME_class(TestCase1)
create_test_padding_SAME_class(TestCase2)
create_test_padding_SAME_class(TestCase3)
create_test_padding_SAME_class(TestCase4)
create_test_padding_SAME_class(TestCase5)

create_test_padding_SAME_class(TestPool2D_channel_last)
create_test_padding_SAME_class(TestCase1_channel_last)
create_test_padding_SAME_class(TestCase2_channel_last)
create_test_padding_SAME_class(TestCase3_channel_last)
create_test_padding_SAME_class(TestCase4_channel_last)
create_test_padding_SAME_class(TestCase5_channel_last)


def create_test_cudnn_padding_SAME_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNPaddingSAMECase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_paddings(self):
            self.paddings = [1, 1]
            self.padding_algorithm = "SAME"

    cls_name = "{}_{}".format(parent.__name__, "CudnnPaddingSAMEOp")
    TestCUDNNPaddingSAMECase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingSAMECase


create_test_cudnn_padding_SAME_class(TestPool2D_Op)
create_test_cudnn_padding_SAME_class(TestCase1)
create_test_cudnn_padding_SAME_class(TestCase2)
create_test_cudnn_padding_SAME_class(TestCase3)
create_test_cudnn_padding_SAME_class(TestCase4)
create_test_cudnn_padding_SAME_class(TestCase5)

create_test_cudnn_padding_SAME_class(TestPool2D_channel_last)
create_test_cudnn_padding_SAME_class(TestCase1_channel_last)
create_test_cudnn_padding_SAME_class(TestCase2_channel_last)
create_test_cudnn_padding_SAME_class(TestCase3_channel_last)
create_test_cudnn_padding_SAME_class(TestCase4_channel_last)
create_test_cudnn_padding_SAME_class(TestCase5_channel_last)


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.paddings = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{}_{}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


create_test_padding_VALID_class(TestPool2D_Op)
create_test_padding_VALID_class(TestCase1)
create_test_padding_VALID_class(TestCase2)
create_test_padding_VALID_class(TestCase3)
create_test_padding_VALID_class(TestCase4)
create_test_padding_VALID_class(TestCase5)

create_test_padding_VALID_class(TestPool2D_channel_last)
create_test_padding_VALID_class(TestCase1_channel_last)
create_test_padding_VALID_class(TestCase2_channel_last)
create_test_padding_VALID_class(TestCase3_channel_last)
create_test_padding_VALID_class(TestCase4_channel_last)
create_test_padding_VALID_class(TestCase5_channel_last)


def create_test_cudnn_padding_VALID_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNPaddingVALIDCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_paddings(self):
            self.paddings = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{}_{}".format(parent.__name__, "CudnnPaddingVALIDOp")
    TestCUDNNPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingVALIDCase


create_test_cudnn_padding_VALID_class(TestPool2D_Op)
create_test_cudnn_padding_VALID_class(TestCase1)
create_test_cudnn_padding_VALID_class(TestCase2)
create_test_cudnn_padding_VALID_class(TestCase3)
create_test_cudnn_padding_VALID_class(TestCase4)
create_test_cudnn_padding_VALID_class(TestCase5)

create_test_cudnn_padding_VALID_class(TestPool2D_channel_last)
create_test_cudnn_padding_VALID_class(TestCase1_channel_last)
create_test_cudnn_padding_VALID_class(TestCase2_channel_last)
create_test_cudnn_padding_VALID_class(TestCase3_channel_last)
create_test_cudnn_padding_VALID_class(TestCase4_channel_last)
create_test_cudnn_padding_VALID_class(TestCase5_channel_last)


class TestCase1_strides(TestCase1):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 2]

    def init_shape(self):
        self.shape = [2, 3, 4, 5]


create_test_cudnn_class(TestCase1_strides)
create_test_padding_SAME_class(TestCase1_strides)
create_test_cudnn_padding_SAME_class(TestCase1_strides)


if __name__ == '__main__':
    unittest.main()
