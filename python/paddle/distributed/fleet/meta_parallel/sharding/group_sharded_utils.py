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

import contextlib
from enum import Enum
from types import MethodType

import numpy as np

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.base import core
from paddle.common_ops_import import dygraph_only
from paddle.nn import clip


class Taskflow:
    """
    Task flows, one way linked list for task acquisition.
    """

    def __init__(self, task, callback):
        self.task = task
        self.callback = callback


class Type(Enum):
    """
    Type of trainable parameters
    """

    fp16 = paddle.float16
    bf16 = paddle.bfloat16
    fp32 = paddle.float32


class GroupShardedClipGrad:
    def __init__(self, clip, device, group):
        self._clip = clip
        self._device = device
        self._group = group

    @paddle.autograd.no_grad()
    def _dygraph_clip(self, params_grads):
        sum_square_fp32, sum_square_fp16, sum_square_bfp16 = [], [], []
        unslice_params_fp32, unslice_params_fp16, unslice_params_bfp16 = (
            [],
            [],
            [],
        )

        for p, g in params_grads:
            p_slice = True  # using for slice parameter in sharding stage3
            if g is None or getattr(p, 'need_clip', True) is False:
                continue
            if hasattr(p, "unslice"):
                p_slice = False

            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.get_tensor_from_selected_rows(
                    clip.merge_selected_rows(g)
                )
            square = paddle.square(merge_grad)
            sum_square = paddle.sum(square)

            if p.dtype == paddle.float16:
                if p_slice:
                    sum_square_fp16.append(sum_square)
                else:
                    unslice_params_fp16.append(sum_square)
            elif p.dtype == paddle.float32:
                if p_slice:
                    sum_square_fp32.append(sum_square)
                else:
                    unslice_params_fp32.append(sum_square)
            elif p.dtype == paddle.bfloat16:
                if p_slice:
                    sum_square_bfp16.append(sum_square)
                else:
                    unslice_params_bfp16.append(sum_square)

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_fp16) == 0:
            global_norm_fp16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_fp16 = paddle.add_n(sum_square_fp16)
            global_norm_fp16 = paddle.cast(
                global_norm_fp16, dtype=paddle.float32
            )

        # global norm of non-distributed BFP16 params_and_grads
        if len(sum_square_bfp16) == 0:
            global_norm_bfp16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_bfp16 = paddle.add_n(sum_square_bfp16)
            global_norm_bfp16 = paddle.cast(
                global_norm_bfp16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads for unslice parameters
        if len(unslice_params_fp16) == 0:
            global_unslice_fp16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_unslice_fp16 = paddle.add_n(unslice_params_fp16)
            global_unslice_fp16 = paddle.cast(
                global_unslice_fp16, dtype=paddle.float32
            )

        # global norm of non-distributed BFP16 params_and_grads for unslice parameters
        if len(unslice_params_bfp16) == 0:
            global_unslice_bfp16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_unslice_bfp16 = paddle.add_n(unslice_params_bfp16)
            global_unslice_bfp16 = paddle.cast(
                global_unslice_bfp16, dtype=paddle.float32
            )

        # global norm of non-distributed FP32 params_and_grads
        if len(sum_square_fp32) == 0:
            global_norm_fp32 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_fp32 = paddle.add_n(sum_square_fp32)

        # global norm of non-distributed FP32 params_and_grads for unslice parameters
        if len(unslice_params_fp32) == 0:
            global_unslice_fp32 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_unslice_fp32 = paddle.add_n(unslice_params_fp32)

        global_unslice_var = (
            global_unslice_fp16 + global_unslice_fp32 + global_unslice_bfp16
        )

        global_norm_var = (
            global_norm_fp16 + global_norm_fp32 + global_norm_bfp16
        )

        # add all reduce to get global norm of distributed params_and_grads
        dev_id = int(self._device.split(":")[1])
        dev_type = self._device.split(':')[0]
        if paddle.device.get_device() == "cpu":
            if dev_type in paddle.device.get_all_custom_device_type():
                global_norm_var = global_norm_var._copy_to(
                    paddle.CustomPlace(dev_type, dev_id), True
                )
            elif dev_type == "xpu":
                global_norm_var = global_norm_var.to(self._device)
            else:
                global_norm_var = global_norm_var.cuda(dev_id)

        with device_guard(dev_id, self._device.split(":")[0]):
            paddle.distributed.all_reduce(global_norm_var, group=self._group)

        global_norm_var = paddle.sqrt(global_norm_var + global_unslice_var)
        max_global_norm = paddle.full(
            shape=[], dtype=global_norm_var.dtype, fill_value=self.clip_norm
        )

        clip_var = paddle.divide(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var, y=max_global_norm),
        )
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        for p, g in params_grads:
            if getattr(p, 'need_clip', True) is False or g is None:
                continue
            origin_state = g.stop_gradient
            g.stop_gradient = True
            if p.dtype == paddle.float16:
                g.scale_(clip_var_fp16)
            else:
                g.scale_(clip_var)
            g.stop_gradient = origin_state
            # p._reset_grad_inplace_version(True)

        return params_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


@contextlib.contextmanager
def device_guard(dev_id=0, device="cpu"):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device in ["gpu", "xpu"]:
        paddle.set_device(f"{device}:{dev_id}")
    elif device in paddle.device.get_all_custom_device_type():
        paddle.set_device(f"{device}:{dev_id}")

    try:
        yield
    finally:
        paddle.set_device(origin_device)


@dygraph_only
def GroupShardedScaler(scaler):
    def unscale_method(self, optimizer):
        if not self._enable:
            return
        param_grads = []
        param_grads_bfp16 = []
        param_grads_fp16 = []
        param_grads_fp32 = []
        if hasattr(optimizer, "update_slice"):
            optimizer.update_slice()
            optimizer.update_scaler = True

        if getattr(optimizer._optim, '_param_groups', None) and isinstance(
            optimizer._optim._param_groups[0], dict
        ):
            for group in optimizer._optim._param_groups:
                for param in group['params']:
                    tgt_grad = None
                    if (
                        hasattr(param, "main_grad")
                        and param.main_grad is not None
                    ):
                        tgt_grad = param.main_grad
                    elif param.grad is not None:
                        tgt_grad = param.grad
                    if tgt_grad is not None:
                        param_grads.append(tgt_grad)
                        if tgt_grad.dtype in [
                            core.VarDesc.VarType.FP16,
                            paddle.float16,
                        ]:
                            param_grads_fp16.append(tgt_grad)
                        elif tgt_grad.dtype in [paddle.bfloat16]:
                            param_grads_bfp16.append(tgt_grad)
                        else:
                            param_grads_fp32.append(tgt_grad)
        else:
            # inf found when step == 1
            if optimizer.offload and len(optimizer._optim._parameter_list) > 1:
                params_list = [optimizer.offload_params.buffer]
                optimizer._optim._parameter_list = params_list
                optimizer._optim._param_groups = params_list
            for param in optimizer._optim._parameter_list:
                tgt_grad = None
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    tgt_grad = param.main_grad
                elif param.grad is not None:
                    tgt_grad = param.grad
                if tgt_grad is not None:
                    param_grads.append(tgt_grad)
                    if tgt_grad.dtype in [
                        core.VarDesc.VarType.FP16,
                        paddle.float16,
                    ]:
                        param_grads_fp16.append(tgt_grad)
                    elif tgt_grad.dtype in [paddle.bfloat16]:
                        param_grads_bfp16.append(tgt_grad)
                    else:
                        param_grads_fp32.append(tgt_grad)

        temp_found_inf_fp16 = paddle.to_tensor(np.array([0]).astype(np.bool_))
        temp_found_inf_bfp16 = paddle.to_tensor(np.array([0]).astype(np.bool_))
        temp_found_inf_fp32 = paddle.to_tensor(np.array([0]).astype(np.bool_))

        device = paddle.get_device().split(":")[0]
        device = "cpu" if optimizer.offload else device
        dev_id = (
            0 if device == "cpu" else int(paddle.get_device().split(":")[1])
        )

        self._found_inf = self._temp_found_inf_value_false
        with device_guard(dev_id, device):
            if len(param_grads_bfp16):
                _legacy_C_ops.check_finite_and_unscale(
                    param_grads_bfp16,
                    self._scale,
                    param_grads_bfp16,
                    temp_found_inf_bfp16,
                )
                self._found_inf = _C_ops.bitwise_or(
                    self._found_inf, temp_found_inf_bfp16
                )
            if len(param_grads_fp16):
                _legacy_C_ops.check_finite_and_unscale(
                    param_grads_fp16,
                    self._scale,
                    param_grads_fp16,
                    temp_found_inf_fp16,
                )
                self._found_inf = _C_ops.bitwise_or(
                    self._found_inf, temp_found_inf_fp16
                )
            if len(param_grads_fp32):
                _legacy_C_ops.check_finite_and_unscale(
                    param_grads_fp32,
                    self._scale,
                    param_grads_fp32,
                    temp_found_inf_fp32,
                )
                self._found_inf = _C_ops.bitwise_or(
                    self._found_inf, temp_found_inf_fp32
                )

        self._found_inf = self._found_inf.cast("int32")

        paddle.distributed.all_reduce(
            self._found_inf, op=paddle.distributed.ReduceOp.MAX, group=None
        )

        self._found_inf = self._found_inf.cast("bool")

    scaler._unscale = MethodType(unscale_method, scaler)
    return scaler


def cvt_to_device(x, dev_id, blocking=True):
    """
    Copy data in x from cpu memory to supported device
    """
    if paddle.is_compiled_with_cuda():
        place = paddle.CUDAPlace(dev_id)
    elif paddle.is_compiled_with_xpu():
        place = paddle.XPUPlace(dev_id)
    else:
        supported_custom_devices = ["npu"]
        place = paddle.framework._current_expected_place()
        if place.get_device_type() not in supported_custom_devices:
            raise OSError(
                "Only supported compiled paddle with gpu/rocm and xpu, but current version is compiled with cpu."
            )
    return x._copy_to(place, blocking)
