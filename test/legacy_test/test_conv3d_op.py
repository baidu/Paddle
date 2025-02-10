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
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_numeric_gradient,
)
from testsuite import create_op

import paddle
from paddle.base import core


def conv3d_forward_naive(
    input,
    filter,
    group,
    conv_param,
    padding_algorithm='EXPLICIT',
    data_format="NCDHW",
):
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError(
            f"Unknown Attr(padding_algorithm): '{padding_algorithm}'. "
            "It can only be 'SAME' or 'VALID'."
        )

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            f"Unknown Attr(data_format): '{data_format}' ."
            "It can only be 'NCDHW' or 'NDHWC'."
        )

    channel_last = data_format == "NDHWC"
    if channel_last:
        input = np.transpose(input, [0, 4, 1, 2, 3])

    in_n, in_c, in_d, in_h, in_w = input.shape

    f_n, f_c, f_d, f_h, f_w = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group

    stride, pad, dilation = (
        conv_param['stride'],
        conv_param['pad'],
        conv_param['dilations'],
    )

    # update pad and dilation
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

    ksize = filter.shape[2:5]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilation = [1, 1, 1]
        input_data_shape = input.shape[2:5]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_d_0, pad_d_1 = pad[0], pad[0]
    pad_h_0, pad_h_1 = pad[1], pad[1]
    pad_w_0, pad_w_1 = pad[2], pad[2]
    if len(pad) == 6:
        pad_d_0, pad_d_1 = pad[0], pad[1]
        pad_h_0, pad_h_1 = pad[2], pad[3]
        pad_w_0, pad_w_1 = pad[4], pad[5]

    out_d = (
        1
        + (in_d + pad_d_0 + pad_d_1 - (dilation[0] * (f_d - 1) + 1))
        // stride[0]
    )
    out_h = (
        1
        + (in_h + pad_h_0 + pad_h_1 - (dilation[1] * (f_h - 1) + 1))
        // stride[1]
    )
    out_w = (
        1
        + (in_w + pad_w_0 + pad_w_1 - (dilation[2] * (f_w - 1) + 1))
        // stride[2]
    )

    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    d_block_d = dilation[0] * (f_d - 1) + 1
    d_block_h = dilation[1] * (f_h - 1) + 1
    d_block_w = dilation[2] * (f_w - 1) + 1

    input_pad = np.pad(
        input,
        (
            (0, 0),
            (0, 0),
            (pad_d_0, pad_d_1),
            (pad_h_0, pad_h_1),
            (pad_w_0, pad_w_1),
        ),
        mode='constant',
        constant_values=0,
    )

    filter_dilation = np.zeros((f_n, f_c, d_block_d, d_block_h, d_block_w))
    filter_dilation[
        :,
        :,
        0 : d_block_d : dilation[0],
        0 : d_block_h : dilation[1],
        0 : d_block_w : dilation[2],
    ] = filter

    for d in range(out_d):
        for i in range(out_h):
            for j in range(out_w):
                for g in range(group):
                    input_pad_masked = input_pad[
                        :,
                        g * f_c : (g + 1) * f_c,
                        d * stride[0] : d * stride[0] + d_block_d,
                        i * stride[1] : i * stride[1] + d_block_h,
                        j * stride[2] : j * stride[2] + d_block_w,
                    ]

                    f_sub = filter_dilation[
                        g * sub_f_n : (g + 1) * sub_f_n, :, :, :, :
                    ]
                    for k in range(sub_out_c):
                        out[:, g * sub_out_c + k, d, i, j] = np.sum(
                            input_pad_masked * f_sub[k, :, :, :, :],
                            axis=(1, 2, 3, 4),
                        )
    if channel_last:
        out = np.transpose(out, [0, 2, 3, 4, 1])
    return out


def create_test_cudnn_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = (
                np.float32 if core.is_compiled_with_rocm() else np.float64
            )

    cls_name = "{}_{}".format(parent.__name__, "CUDNN")
    TestCUDNNCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNCase


def create_test_cudnn_bf16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    class TestConv3DCUDNNBF16(parent):
        def get_numeric_grad(self, place, check_name):
            scope = core.Scope()
            self._check_grad_helper()
            op = create_op(
                scope, self.op_type, self.inputs, self.outputs, self.attrs
            )
            return get_numeric_gradient(
                place, scope, op, self.inputs_fp32, check_name, ['Output']
            )

        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place,
                check_dygraph=(not self.use_mkldnn),
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

        def test_check_grad_no_filter(self):
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Input')

            self.check_grad_with_place(
                place,
                ['Input'],
                'Output',
                no_grad_set={'Filter'},
                check_dygraph=(not self.use_mkldnn),
                user_defined_grads=[numeric_grads],
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

        def test_check_grad_no_input(self):
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Filter')

            self.check_grad_with_place(
                place,
                ['Filter'],
                'Output',
                no_grad_set={'Input'},
                check_dygraph=(not self.use_mkldnn),
                user_defined_grads=[numeric_grads],
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            numeric_input_grads = self.get_numeric_grad(place, 'Input')
            numeric_filter_grads = self.get_numeric_grad(place, 'Filter')

            self.check_grad_with_place(
                place,
                ['Input', 'Filter'],
                'Output',
                user_defined_grads=[numeric_input_grads, numeric_filter_grads],
                check_dygraph=(not self.use_mkldnn),
                check_pir=True,
                check_pir_onednn=self.check_pir_onednn,
            )

    cls_name = "{}_{}".format(parent.__name__, "CUDNNBF16OP")
    TestConv3DCUDNNBF16.__name__ = cls_name
    globals()[cls_name] = TestConv3DCUDNNBF16


def create_test_padding_SAME_class(parent):
    class TestPaddingSAMECase(parent):
        def init_paddings(self):
            self.pad = [0, 0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{}_{}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSAMECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSAMECase


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.pad = [1, 1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{}_{}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def create_test_cudnn_padding_SAME_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNPaddingSAMECase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = (
                np.float32 if core.is_compiled_with_rocm() else np.float64
            )

        def init_paddings(self):
            self.pad = [1, 1, 1]
            self.padding_algorithm = "SAME"

    cls_name = "{}_{}".format(parent.__name__, "CudnnPaddingSAMEOp")
    TestCUDNNPaddingSAMECase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingSAMECase


def create_test_cudnn_padding_VALID_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCUDNNPaddingVALIDCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = (
                np.float32 if core.is_compiled_with_rocm() else np.float64
            )

        def init_paddings(self):
            self.pad = [1, 1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{}_{}".format(parent.__name__, "CudnnPaddingVALIDOp")
    TestCUDNNPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingVALIDCase


def create_test_channel_last_class(parent):
    class TestChannelLastCase(parent):
        def init_data_format(self):
            self.data_format = "NDHWC"

        def init_test_case_2(self):
            N, C, D, H, W = self.input_size
            self.input_size = [N, D, H, W, C]

    cls_name = "{}_{}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_cudnn_channel_last_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestCudnnChannelLastCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = (
                np.float32 if core.is_compiled_with_rocm() else np.float64
            )

        def init_data_format(self):
            self.data_format = "NDHWC"

        def init_test_case_2(self):
            N, C, D, H, W = self.input_size
            self.input_size = [N, D, H, W, C]

    cls_name = "{}_{}".format(parent.__name__, "CudnnChannelLast")
    TestCudnnChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestCudnnChannelLastCase


def conv3d_wrapper(
    x,
    weight,
    stride=1,
    padding=0,
    padding_algorithm="EXPLICIT",
    groups=1,
    dilation=1,
    data_format="NCDHW",
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    if padding_algorithm is None:
        padding_algorithm = "EXPLICIT"
    return paddle._C_ops.conv3d(
        x,
        weight,
        stride,
        padding,
        padding_algorithm,
        groups,
        dilation,
        data_format,
    )


class TestConv3DOp(OpTest):
    def setUp(self):
        self.op_type = "conv3d"
        self.python_api = conv3d_wrapper
        self.use_cudnn = False
        self.use_mkldnn = False
        self.data_format = "AnyLayout"
        self.dtype = np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv3d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilations': self.dilations,
        }

        if self.is_bfloat16_op():
            input = np.random.random(self.input_size).astype(np.float32)
            filter = np.random.random(self.filter_size).astype(np.float32)
        else:
            input = np.random.random(self.input_size).astype(self.dtype)
            filter = np.random.random(self.filter_size).astype(self.dtype)

        output = conv3d_forward_naive(
            input,
            filter,
            self.groups,
            conv3d_param,
        )

        if self.is_bfloat16_op():
            output = convert_float_to_uint16(output)
            self.inputs = {
                'Input': convert_float_to_uint16(input),
                'Filter': convert_float_to_uint16(filter),
            }
            self.inputs_fp32 = {
                'Input': OpTest.np_dtype_to_base_dtype(input),
                'Filter': OpTest.np_dtype_to_base_dtype(filter),
            }
        else:
            output = output.astype(self.dtype)
            self.inputs = {
                'Input': OpTest.np_dtype_to_base_dtype(input),
                'Filter': OpTest.np_dtype_to_base_dtype(filter),
            }

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
        }
        self.outputs = {'Output': output}

    def has_cudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        self.check_output_with_place(
            place,
            atol=1e-5,
            check_dygraph=(not self.use_mkldnn),
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad(self):
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad_with_place(
            place,
            {'Input', 'Filter'},
            'Output',
            max_relative_error=0.03,
            check_dygraph=(not self.use_mkldnn),
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_no_filter(self):
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad_with_place(
            place,
            ['Input'],
            'Output',
            max_relative_error=0.03,
            no_grad_set={'Filter'},
            check_dygraph=(not self.use_mkldnn),
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_no_input(self):
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad_with_place(
            place,
            ['Filter'],
            'Output',
            max_relative_error=0.03,
            no_grad_set={'Input'},
            check_dygraph=(not self.use_mkldnn),
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_test_case_2(self):
        pass

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass


class TestCase1(TestConv3DOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


class TestWithGroup1(TestConv3DOp):
    def init_group(self):
        self.groups = 3


class TestWithGroup2(TestCase1):
    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv3DOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithInput1x1Filter1x1(TestConv3DOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [40, 3, 1, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv3DOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 6, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 2, 2, 2]

    def init_dilation(self):
        self.dilations = [2, 2, 2]

    def init_group(self):
        self.groups = 3


# ---------------- Conv3DCUDNN ----------------


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNN(TestConv3DOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16CUDNN(TestConv3DOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(
                    place,
                    atol=2e-2,
                    check_pir=True,
                    check_pir_onednn=self.check_pir_onednn,
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestWithGroup1CUDNN(TestWithGroup1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16WithGroup1CUDNN(TestWithGroup1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(
                    place,
                    atol=2e-2,
                    check_pir=True,
                    check_pir_onednn=self.check_pir_onednn,
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestWithGroup2CUDNN(TestWithGroup2):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16WithGroup2CUDNN(TestWithGroup2):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(
                    place,
                    atol=2e-2,
                    check_pir=True,
                    check_pir_onednn=self.check_pir_onednn,
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestWith1x1CUDNN(TestWith1x1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16With1x1CUDNN(TestWith1x1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(
                    place,
                    atol=2e-2,
                    check_pir=True,
                    check_pir_onednn=self.check_pir_onednn,
                )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestWithInput1x1Filter1x1CUDNN(TestWithInput1x1Filter1x1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16WithInput1x1Filter1x1CUDNN(TestWithInput1x1Filter1x1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(
                    place,
                    atol=2e-2,
                    check_pir=True,
                    check_pir_onednn=self.check_pir_onednn,
                )


class TestCUDNNExhaustiveSearch(TestCUDNN):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.exhaustive_search = True
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64


# ----------------Conv3DCUDNN bf16----------------
create_test_cudnn_bf16_class(TestConv3DOp)
create_test_cudnn_bf16_class(TestWithGroup1)
create_test_cudnn_bf16_class(TestWithGroup2)
create_test_cudnn_bf16_class(TestWith1x1)
create_test_cudnn_bf16_class(TestWithInput1x1Filter1x1)


# ---- test asymmetric padding ----


class TestConv3DOp_2(OpTest):
    def setUp(self):
        self.op_type = "conv3d"
        self.python_api = conv3d_wrapper
        self.use_cudnn = False
        self.use_mkldnn = False
        self.data_format = "NCDHW"
        self.dtype = np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_data_format()
        self.init_test_case()
        self.init_paddings()

        self.init_test_case_2()

        conv3d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilations': self.dilations,
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        output = conv3d_forward_naive(
            input,
            filter,
            self.groups,
            conv3d_param,
            self.padding_algorithm,
            self.data_format,
        ).astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_base_dtype(input),
            'Filter': OpTest.np_dtype_to_base_dtype(filter),
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
        }
        self.outputs = {'Output': output}

    def has_cudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        self.check_output_with_place(
            place,
            atol=1e-5,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        self.check_grad_with_place(
            place,
            {'Input', 'Filter'},
            'Output',
            max_relative_error=0.03,
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['Input'],
            'Output',
            max_relative_error=0.03,
            no_grad_set={'Filter'},
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def test_check_grad_no_input(self):
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cudnn() else core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['Filter'],
            'Output',
            max_relative_error=0.03,
            no_grad_set={'Input'},
            check_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )

    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_test_case_2(self):
        pass

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass

    def init_paddings(self):
        self.pad = [0, 0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_data_format(self):
        self.data_format = "NCDHW"


class TestConv3DOp_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv3DOp_DiffDataInDiffDim(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.input_size = [2, 3, 4, 5, 5]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 4, 3]

    def init_paddings(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


create_test_padding_SAME_class(TestConv3DOp_DiffDataInDiffDim)
create_test_padding_VALID_class(TestConv3DOp_DiffDataInDiffDim)
create_test_channel_last_class(TestConv3DOp_DiffDataInDiffDim)


class TestCase1_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_paddings(self):
        self.pad = [0, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithGroup1_AsyPadding(TestConv3DOp_2):
    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [1, 1, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithGroup2_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [1, 1, 0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWith1x1_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 0, 1, 0, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithDilation_AsyPadding(TestConv3DOp_2):
    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 6, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 2, 2, 2]

    def init_dilation(self):
        self.dilations = [2, 2, 2]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 0, 1, 0, 1, 0]
        self.padding_algorithm = "EXPLICIT"


create_test_cudnn_class(TestConv3DOp_AsyPadding)
create_test_cudnn_class(TestWithGroup1_AsyPadding)
create_test_cudnn_class(TestWithGroup2_AsyPadding)
create_test_cudnn_class(TestWith1x1_AsyPadding)
create_test_cudnn_class(TestWithDilation_AsyPadding)

create_test_padding_SAME_class(TestConv3DOp_AsyPadding)
create_test_padding_SAME_class(TestWithGroup1_AsyPadding)
create_test_padding_SAME_class(TestWith1x1_AsyPadding)

create_test_padding_VALID_class(TestConv3DOp_AsyPadding)
create_test_padding_VALID_class(TestWithGroup1_AsyPadding)
create_test_padding_VALID_class(TestWith1x1_AsyPadding)

create_test_cudnn_padding_SAME_class(TestConv3DOp_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithGroup1_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWith1x1_AsyPadding)

create_test_cudnn_padding_VALID_class(TestConv3DOp_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithGroup1_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWith1x1_AsyPadding)

create_test_channel_last_class(TestConv3DOp_AsyPadding)
create_test_channel_last_class(TestWithGroup1_AsyPadding)
create_test_channel_last_class(TestWith1x1_AsyPadding)

create_test_channel_last_class(TestConv3DOp_AsyPadding)
create_test_channel_last_class(TestWithGroup1_AsyPadding)
create_test_channel_last_class(TestWith1x1_AsyPadding)

create_test_cudnn_channel_last_class(TestConv3DOp_AsyPadding)
create_test_cudnn_channel_last_class(TestWithGroup1_AsyPadding)
create_test_cudnn_channel_last_class(TestWith1x1_AsyPadding)

create_test_cudnn_channel_last_class(TestConv3DOp_AsyPadding)
create_test_cudnn_channel_last_class(TestWithGroup1_AsyPadding)
create_test_cudnn_channel_last_class(TestWith1x1_AsyPadding)

# FIXME(typhoonzero): find a way to determine if
# using cudnn > 6 in python
# class TestWithDilationCUDNN(TestWithDilation):
#     def init_op_type(self):
#         self.op_type = "conv3d"


# --------- test python API ---------------
class TestConv3DAPI(unittest.TestCase):
    def api_run(self):
        input_NDHWC = paddle.static.data(
            name="input_NDHWC",
            shape=[2, 5, 5, 5, 3],
            dtype="float32",
        )
        input_NDHWC_in_channel = 5

        input_NCDHW = paddle.static.data(
            name="input_NCDHW",
            shape=[2, 3, 5, 5, 3],
            dtype="float32",
        )
        input_NCDHW_in_channel = 3

        paddle.nn.Conv3D(
            in_channels=input_NCDHW_in_channel,
            out_channels=3,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=0,
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )(input_NCDHW)

        paddle.nn.Conv3D(
            in_channels=input_NCDHW_in_channel,
            out_channels=3,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[1, 2, 1, 0, 1, 0],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )(input_NCDHW)

        paddle.nn.Conv3D(
            in_channels=input_NCDHW_in_channel,
            out_channels=3,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )(input_NCDHW)

        paddle.nn.Conv3D(
            in_channels=input_NDHWC_in_channel,
            out_channels=3,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
            dilation=[1, 1, 1],
            groups=1,
            data_format="NDHWC",
        )(input_NDHWC)

        paddle.nn.Conv3D(
            in_channels=input_NCDHW_in_channel,
            out_channels=3,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding="SAME",
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )(input_NCDHW)

        paddle.nn.Conv3D(
            in_channels=input_NCDHW_in_channel,
            out_channels=3,
            kernel_size=[3, 3, 3],
            stride=[1, 1, 1],
            padding="VALID",
            dilation=[1, 1, 1],
            groups=1,
            data_format="NCDHW",
        )(input_NCDHW)

    def test_api(self):
        with paddle.pir_utils.OldIrGuard():
            self.api_run()
        with paddle.pir_utils.IrGuard():
            self.api_run()


class TestConv3DAPI_Error(unittest.TestCase):
    def test_api(self):
        with paddle.pir_utils.OldIrGuard():
            input = paddle.static.data(
                name="input",
                shape=[2, 5, 5, 5, 4],
                dtype="float32",
            )

            # ValueError: cudnn
            def run_1():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=3,
                    filter_size=3,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    use_cudnn=[0],
                    data_format="NCDHW",
                )

            self.assertRaises(ValueError, run_1)

            # ValueError: data_format
            def run_2():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=3,
                    filter_size=[3, 3, 3],
                    stride=[1, 1, 1],
                    padding=0,
                    dilation=[1, 1, 1],
                    groups=1,
                    use_cudnn=False,
                    data_format="NCHWC",
                )

            self.assertRaises(ValueError, run_2)

            # ValueError: padding
            def run_3():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=3,
                    filter_size=3,
                    stride=1,
                    padding="SAMEE",
                    dilation=1,
                    groups=1,
                    use_cudnn=False,
                    data_format="NCDHW",
                )

            self.assertRaises(ValueError, run_3)

            def run_4():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=3,
                    filter_size=3,
                    stride=1,
                    padding=[[0, 1], [0, 0], [0, 1], [0, 1], [0, 1]],
                    dilation=1,
                    groups=1,
                    use_cudnn=False,
                    data_format="NCDHW",
                )

            self.assertRaises(ValueError, run_4)

            def run_5():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=3,
                    filter_size=0,
                    stride=0,
                    padding=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                    dilation=1,
                    groups=1,
                    use_cudnn=False,
                    data_format="NDHWC",
                )

            self.assertRaises(ValueError, run_5)

            # ValueError: channel dimension
            x = paddle.static.data(
                name="x",
                shape=[2, 5, 5, 5, -1],
                dtype="float32",
            )

            def run_6():
                paddle.static.nn.conv3d(
                    input=x,
                    num_filters=3,
                    filter_size=3,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    use_cudnn=False,
                    data_format="NDHWC",
                )

            self.assertRaises(ValueError, run_6)

            # ValueError: groups
            def run_7():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=3,
                    filter_size=3,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=3,
                    use_cudnn=False,
                    data_format="NDHWC",
                )

            self.assertRaises(ValueError, run_7)

            # ValueError: filter num
            def run_8():
                paddle.static.nn.conv3d(
                    input=input,
                    num_filters=0,
                    filter_size=0,
                    stride=0,
                    padding=0,
                    dilation=0,
                    groups=1,
                    use_cudnn=False,
                    data_format="NDHWC",
                )

            self.assertRaises(ValueError, run_8)


class TestPIRConv3DAPI_Error(unittest.TestCase):
    def test_api(self):
        with paddle.pir_utils.IrGuard():
            input = paddle.static.data(
                name="input",
                shape=[2, 5, 5, 5, 4],
                dtype="float32",
            )
            input_NCDHW_in_channel = 5
            input_NDHWC_in_channel = 4

            # ValueError: cudnn
            # def run_1():
            #     model = paddle.nn.Conv3D(
            #         in_channels=input_NCDHW_in_channel,
            #         out_channels=3,
            #         kernel_size=3,
            #         stride=1,
            #         padding=0,
            #         dilation=1,
            #         groups=1,
            #         data_format="NCDHW",
            #     )
            #     model._use_cudnn = [0]
            #     model(input)
            #
            # self.assertRaises(ValueError, run_1)

            # ValueError: data_format
            def run_2():
                paddle.nn.Conv3D(
                    in_channels=input_NCDHW_in_channel,
                    out_channels=3,
                    kernel_size=[3, 3, 3],
                    stride=[1, 1, 1],
                    padding=0,
                    dilation=[1, 1, 1],
                    groups=1,
                    data_format="NCHWC",
                )(input)

            self.assertRaises(ValueError, run_2)

            # ValueError: padding
            def run_3():
                paddle.nn.Conv3D(
                    in_channels=input_NCDHW_in_channel,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding="SAMEE",
                    dilation=1,
                    groups=1,
                    data_format="NCDHW",
                )(input)

            self.assertRaises(ValueError, run_3)

            def run_4():
                paddle.nn.Conv3D(
                    in_channels=input_NCDHW_in_channel,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=[[0, 1], [0, 0], [0, 1], [0, 1], [0, 1]],
                    dilation=1,
                    groups=1,
                    data_format="NCDHW",
                )(input)

            self.assertRaises(ValueError, run_4)

            def run_5():
                paddle.nn.Conv3D(
                    in_channels=input_NDHWC_in_channel,
                    out_channels=3,
                    kernel_size=0,
                    stride=0,
                    padding=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                    dilation=1,
                    groups=1,
                    data_format="NDHWC",
                )(input)

            self.assertRaises(ValueError, run_5)

            # ValueError: channel dimension
            x = paddle.static.data(
                name="x",
                shape=[2, 5, 5, 5, -1],
                dtype="float32",
            )
            x_NCDHW_in_channel = 5
            x_NDHWC_in_channel = -1

            def run_6():
                paddle.nn.Conv3D(
                    in_channels=x_NDHWC_in_channel,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    data_format="NDHWC",
                )(x)

            self.assertRaises(AssertionError, run_6)

            # ValueError: groups
            def run_7():
                paddle.nn.Conv3D(
                    in_channels=x_NDHWC_in_channel,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=3,
                    data_format="NDHWC",
                )(x)

            self.assertRaises(ValueError, run_7)

            # ValueError: filter num
            def run_8():
                paddle.nn.Conv3D(
                    in_channels=x_NDHWC_in_channel,
                    out_channels=0,
                    kernel_size=0,
                    stride=0,
                    padding=0,
                    dilation=0,
                    groups=1,
                    data_format="NDHWC",
                )(x)

            self.assertRaises(AssertionError, run_8)


if __name__ == '__main__':
    unittest.main()
