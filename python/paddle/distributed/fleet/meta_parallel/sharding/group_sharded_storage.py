#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# The file has been adapted from fairscale file:
# https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/misc/param_bucket.py
# Git commit hash: 8acbec718f3c70a6b9785470bb9e05cd84fc3f8e
# We retain the following license from the original files:

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import paddle
from paddle.framework import core

from .group_sharded_utils import Type, cvt_to_device, device_guard


class BufferWarper(core.eager.Tensor):
    def __init__(self):
        super().__init__()
        self.need_clip = True
        self.is_distributed = False
        self.trainable = True


class InternalStorage:
    """
    This is a basic class, which is responsible for consolidating the basic storage tensor.

    """

    # Support integration parameter tensor
    def __init__(self, size, dtype, device, convert_cpu=False):
        self._params = []
        self._param_ids = []
        self._fill = 0
        self._device = device
        self._dtype = dtype

        # The flatten tensor
        size = [size] if isinstance(size, int) else size
        if convert_cpu:
            value = (
                np.zeros(size, dtype=np.float16)
                if Type.fp16.value == dtype
                else np.zeros(size, dtype=np.float32)
            )
            self.buffer = core.eager.Tensor(value=value, place=core.CPUPlace())
            if dtype == Type.bf16.value:
                self.buffer = paddle.cast(self.buffer, dtype=paddle.bfloat16)
        else:
            self.buffer = paddle.zeros(size, dtype=dtype)

        self.dev_id = (
            0
            if paddle.get_device() == "cpu"
            else int(paddle.get_device().split(":")[1])
        )

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        assert (
            self.buffer is not None
        ), "Cannot move a collapsed bucket, please rebuild it"
        assert (
            dtype == Type.fp32.value or Type.fp16.value
        ), "Conversion type is not supported now"

        if self._device != device:
            if device in paddle.device.get_all_custom_device_type():
                tmp_buffer = self.buffer._copy_to(
                    paddle.CustomPlace(device, self.dev_id), True
                )
            else:
                tmp_buffer = (
                    cvt_to_device(self.buffer, self.dev_id)
                    if device in ["gpu", "xpu"]
                    else self.buffer.cpu()
                )
            for param in self._params:
                param.clear_gradient(False)

            del self.buffer
            self.buffer = tmp_buffer
            self._device = device

        if dtype is not None:
            self.buffer = self.buffer.cast(dtype=dtype)
            self._dtype = dtype

    def warp_buffer(self):
        tmp_buffer = BufferWarper()
        self._buffer = self.buffer
        tmp_buffer.get_tensor()._share_data_with(self.buffer.get_tensor())
        self.buffer = tmp_buffer


class ParamStorage(InternalStorage):
    """
    This is a basic class to simplify the handling of parameter InternalStorages.
    """

    def __init__(self, size, dtype, device):
        super().__init__(size, dtype, device, convert_cpu=True)
        self.param2align = None

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """

        super().to(device, dtype)

        if keep_alignment:
            self._array_params()

    @paddle.autograd.no_grad()
    def add_rank_params(self, trainable_params, param2align, convert_gpu=True):
        """
        Add new parameters to the InternalStorage. Params becomes a view of this InternalStorage buffer.
        """

        assert all(
            id(param) not in self._param_ids for param in trainable_params
        ), "The same param cannot be checked in twice"
        assert self.buffer is not None

        self.param2align = param2align

        cpu_param_shape = []
        for param in trainable_params:
            p_shape = self._add_param_as_view(
                param, param2align[param.name], convert_gpu
            )
            cpu_param_shape.append(p_shape)

        if convert_gpu:
            if self._device in paddle.device.get_all_custom_device_type():
                self.buffer = self.buffer._copy_to(
                    paddle.CustomPlace(self._device, self.dev_id), True
                )
            else:
                # buffer convert from cpu to cuda
                self.buffer = cvt_to_device(self.buffer, self.dev_id)

        self._fill = 0

        for idx, param in enumerate(trainable_params):
            self._convert_buffer(
                param, cpu_param_shape[idx], param2align[param.name]
            )
            self._params.append(param)
            self._param_ids.append(id(param))

    @paddle.autograd.no_grad()
    def _add_param_as_view(self, param, align, convert_gpu=True):
        assert (
            param.dtype == self.buffer.dtype
        ), f"Different types for the InternalStorage and the param, cannot proceed: {param.dtype} - {self.buffer.dtype}"

        var_end = self._fill + param._numel()
        offset = var_end + align
        assert offset <= self.buffer._numel()

        p_shape = param.shape

        origin_state = param.stop_gradient
        param.stop_gradient = True
        param.flatten_()
        param.stop_gradient = origin_state

        # Copy the current param value

        with device_guard(self.dev_id, "cpu"):
            tmp_var = self.buffer._slice(self._fill, var_end)
            if convert_gpu:
                param_cpu = param.cpu()
                param._clear_data()
                tmp_var.set_value(param_cpu)
            else:
                tmp_var.set_value(param)
            del tmp_var

        self._fill = offset
        return p_shape

    @paddle.autograd.no_grad()
    def _convert_buffer(self, param, p_shape, align):
        var_end = self._fill + np.prod(p_shape).tolist()
        offset = var_end + align
        assert offset <= self.buffer._numel()

        # Convert the param value
        with device_guard(self.dev_id, self._device):
            tmp_tensor = self.buffer._slice(self._fill, var_end)
            tmp_tensor._share_buffer_to(param)
            param.get_tensor()._set_dims(p_shape)

        self._fill = offset

    @paddle.autograd.no_grad()
    def _array_params(self):
        """
        Given the parameters which have been registered previously, rebuild the whole InternalStorage.
        """
        assert len(self._params) > 0
        assert self.param2align is not None

        self._fill = 0
        for p in self._params:
            self._convert_buffer(p, p.shape, self.param2align[p.name])  # modify


class GradStorage(InternalStorage):
    """
    This is a basic class to simplify the handling of gradient InternalStorages
    """

    def __init__(
        self, size, dtype, device, destination, param2align, convert_cpu=False
    ):
        if isinstance(size, np.int64):
            size = size.tolist()
        super().__init__(size, dtype, device, convert_cpu)

        self._max_size = size
        self._release = False

        self.params_checked_in = 0
        self.destination = destination
        self._param2align = param2align
        self.sent = False

    def reset_checked_in(self):
        """Reset the counter of the parameter grads which have been checked in"""
        self.params_checked_in = 0
        self.sent = False

    @property
    def all_checked_in(self):
        """Judge all the expected gradient check-in happened"""
        return len(self._params) == self.params_checked_in

    def can_add_grad_view(self, param, align):
        """Is there enough InternalStorage to add this parameter gradient, and whether this param have already checked in."""
        return (
            self._fill + param._numel() + align <= self._max_size
            and id(param) not in self._param_ids
        )

    def to(self, device, dtype=None, keep_alignment=True):
        """
        Move the underlying buffer
        """
        if self._release:
            self.rebuild()

        super().to(device, dtype)

        if keep_alignment:
            self._array_grads()

    @paddle.autograd.no_grad()
    def add_grad(self, param, align):
        """
        Add a new parameter gradient to the InternalStorage. Param.grad becomes a view of this InternalStorage buffer.
        """

        assert (
            id(param) not in self._param_ids
        ), "The same gradients cannot be checked in twice"

        self._add_grad_as_view(param, align)
        self._params.append(param)
        self._param_ids.append(id(param))

    @paddle.autograd.no_grad()
    def manual_release(self):
        """
        Release the buffer from InternalStorage. The InternalStorage will need to be rebuilt before use.
        """
        if not self._release:
            for p in self._params:
                use_main_grad = hasattr(p, "main_grad")
                if use_main_grad and p.main_grad is not None:
                    p.main_grad._clear_data()
                    p.main_grad = None
                elif p.grad is not None:
                    p.clear_gradient(False)

            self.buffer = None
            self._fill = 0
            self.params_checked_in = 0
            self._release = True

    @paddle.autograd.no_grad()
    def rebuild(self):
        """
        Given the parameter gradients which have been registered previously, rebuild the whole InternalStorage.
        """

        if self._release:
            self.buffer = paddle.zeros([self._max_size], dtype=self._dtype)

            for p in self._params:
                self._add_grad_as_view(p, self._param2align[p.name])

            self._release = False

    @paddle.autograd.no_grad()
    def _array_grads(self):
        """
        Given the parameters gradients which have been registered previously, rebuild the whole InternalStorage.
        """
        if len(self._params) > 0:
            self._fill = 0
            for p in self._params:
                self._add_grad_as_view(p, self._param2align[p.name])

    @paddle.autograd.no_grad()
    def _add_grad_as_view(self, param, align):
        assert (
            param._numel() > 0
        ), "Cannot add a gradient to a released InternalStorage, please rebuild"

        use_main_grad = hasattr(param, "main_grad")
        if use_main_grad:
            assert self.buffer.dtype == paddle.float32
        else:
            assert param.dtype == self.buffer.dtype

        grad_end = self._fill + param._numel()
        offset = grad_end + align
        assert offset <= self.buffer._numel()

        # Copy the current grad value to InternalStorage
        with device_guard(self.dev_id, self._device):
            tmp_var = self.buffer._slice(self._fill, grad_end)
            tmp_var.get_tensor()._set_dims(param.shape)
            if not use_main_grad:
                param._copy_gradient_from(tmp_var)
            else:
                param.main_grad = tmp_var
            del tmp_var

        self._fill = offset
