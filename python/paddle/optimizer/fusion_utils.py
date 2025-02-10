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

import numpy as np

import paddle
import paddle.autograd as imperative_base
from paddle.framework import (
    _current_expected_place_,
)
from paddle.incubate.tensor.manipulation import (
    async_offload_with_offset,
    create_async_load,
)

alignment = {
    "gpu": 256,
    "npu": 256,
    "xpu": 256,
}

align = {
    paddle.float16.value: 2,
    paddle.bfloat16.value: 2,
    paddle.float32.value: 4,
}

__current_device_type__ = None


def get_current_device_type():
    global __current_device_type__
    if __current_device_type__ is None:
        if paddle.is_compiled_with_cuda():
            device_type = "gpu"
        elif paddle.is_compiled_with_xpu():
            device_type = "xpu"
        else:
            current_device = _current_expected_place_()
            try:
                device_type = current_device.get_device_type()
            except:
                device_type = "unknown"
        assert (
            device_type in alignment.keys()
        ), f"tensor fusion helper now only support {alignment.keys()}, but got device {device_type} instead."
        __current_device_type__ = device_type
    return __current_device_type__


def get_align(t):
    size = np.prod(t.shape) * align[t.dtype.value]
    remaining = size % alignment[get_current_device_type()]
    ali = (
        0
        if remaining == 0
        else alignment[get_current_device_type()] - remaining
    )
    align_ = ali // align[t.dtype.value]
    return align_


class FusionStorage:
    def __init__(
        self,
        accumulators,
        master_weights,
        merged_model_params=None,
        dtype=paddle.float32,
    ):
        assert isinstance(accumulators, dict), "accumulators must be a dict"
        assert isinstance(master_weights, dict), "master_weights must be a dict"
        assert (
            isinstance(merged_model_params, dict) or merged_model_params is None
        ), "merged_model_params must be a dict or None"
        self.accumulators = accumulators
        self.master_weights = master_weights
        self.merged_model_params = merged_model_params
        self.accumulators_meta = {}
        self.master_weights_meta = {}
        self.merged_model_params_meta = {}
        self.dtype = dtype
        self.buffer = None
        self.buffer_ipc_meta = None
        self.offset = 0
        self.build_buffer()
        self.mapping_tensor()

    @imperative_base.no_grad()
    def build_buffer(self):
        self.offset = 0

        for k, v in self.accumulators.items():
            if k not in self.accumulators_meta:
                self.accumulators_meta[k] = {}
            for para_name, var_tmp in v.items():
                assert var_tmp.dtype == self.dtype
                src_len = var_tmp._numel() + get_align(var_tmp)
                self.accumulators_meta[k][para_name] = {
                    "start": self.offset,
                    "end": self.offset + src_len,
                    "name": var_tmp.name,
                    "shape": var_tmp.shape,
                }
                self.offset += src_len

        for k, v in self.master_weights.items():
            assert v.dtype == self.dtype
            src_len = v._numel() + get_align(v)
            self.master_weights_meta[k] = {
                "start": self.offset,
                "end": self.offset + src_len,
                "name": v.name,
                "shape": v.shape,
            }
            self.offset += src_len

        if self.merged_model_params is not None:
            for k, v in self.merged_model_params.items():
                assert v.dtype == self.dtype
                src_len = v._numel() + get_align(v)
                self.merged_model_params_meta[k] = {
                    "start": self.offset,
                    "end": self.offset + src_len,
                    "name": v.name,
                    "shape": v.shape,
                }
                self.offset += src_len

        self.buffer = paddle.zeros((self.offset,), dtype=self.dtype)
        self.buffer_ipc_meta = self.buffer.value().get_tensor()._share_cuda()

    @imperative_base.no_grad()
    def mapping_tensor(self):
        for k, v in self.accumulators_meta.items():
            for para_name, meta in v.items():
                self.mapping_tensor_impl(
                    src=self.accumulators[k][para_name],
                    start=meta["start"],
                    end=meta["end"],
                )

        for k, v in self.master_weights_meta.items():
            self.mapping_tensor_impl(
                src=self.master_weights[k], start=v["start"], end=v["end"]
            )

        for k, v in self.merged_model_params_meta.items():
            self.mapping_tensor_impl(
                src=self.merged_model_params[k],
                start=v["start"],
                end=v["end"],
            )

    @imperative_base.no_grad()
    def mapping_tensor_impl(self, src, start, end):
        tensor_shape = src.shape
        stop_gradient = src.stop_gradient
        src.stop_gradient = True
        src.flatten_()
        paddle.assign(
            src,
            self.buffer._slice(start, end),
        )
        src.get_tensor()._set_dims(tensor_shape)
        src.stop_gradient = stop_gradient
        self.buffer._slice(start, end)._share_buffer_to(src)


class FusionStorageHelper:
    def __init__(
        self,
        accumulators_meta,
        master_weights_meta,
        merged_model_params_meta,
        buffer_ipc_meta,
    ):
        self.async_loader = create_async_load()
        self.accumulators_meta = None
        self.master_weights_meta = None
        self.merged_model_params_meta = None
        self.buffer_ipc_meta = None
        self.buffer = None
        self.cpu_buffer = None
        self.buffer_length = None
        self.tasks = []
        self.reset_meta(
            accumulators_meta,
            master_weights_meta,
            merged_model_params_meta,
            buffer_ipc_meta,
        )

    @imperative_base.no_grad()
    def reset_meta(
        self,
        accumulators_meta,
        master_weights_meta,
        merged_model_params_meta,
        buffer_ipc_meta,
    ):
        assert isinstance(
            accumulators_meta, dict
        ), "accumulators_meta must be a dict"
        self.accumulators_meta = accumulators_meta
        assert isinstance(
            master_weights_meta, dict
        ), "master_weights_meta must be a dict"
        self.master_weights_meta = master_weights_meta
        assert (
            isinstance(merged_model_params_meta, dict)
            or merged_model_params_meta is None
        ), "merged_model_params_meta must be a dict or None"
        self.merged_model_params_meta = merged_model_params_meta
        assert (
            isinstance(buffer_ipc_meta, tuple) and len(buffer_ipc_meta) == 7
        ), "buffer_ipc_meta must be a tuple with length 7"
        self.buffer_ipc_meta = buffer_ipc_meta

        self.buffer = paddle.to_tensor(
            paddle.base.core.LoDTensor._new_shared_cuda(self.buffer_ipc_meta)
        )
        self.cpu_buffer = self.buffer.pin_memory()
        self.buffer_length = self.buffer._numel()

    def sync_param(self):
        self.sync_partial_param(0, self.buffer_length)

    @imperative_base.no_grad()
    def sync_partial_param(self, start, end):
        assert isinstance(start, int), "start must be an integer"
        assert isinstance(end, int), "end must be an integer"
        assert start >= 0, "start must be non-negative"
        assert (
            end <= self.buffer_length
        ), "end must be less than or equal to the total buffer length"
        task = async_offload_with_offset(
            src_tensor=self.buffer,
            dst_tensor=self.cpu_buffer,
            src_offset=start,
            dst_offset=start,
            offload_size=(end - start),
            async_loader=self.async_loader,
        )
        self.tasks.append(task)

    def wait_all(self):
        if len(self.tasks) == 0:
            return
        last_task = self.tasks.pop(-1)
        while len(self.tasks) > 0:
            task = self.tasks.pop(0)
            task.cuda_wait()
        last_task.cpu_wait()

    def state_dict(self):
        state_dict = {"master_weights": {}}
        for k, v in self.accumulators_meta.items():
            for para_name, tensor_meta in v.items():
                var_tmp = self.restore_tensor_from_meta(tensor_meta)
                state_dict[var_tmp.name] = var_tmp
        for k, v in self.master_weights_meta.items():
            var_tmp = self.restore_tensor_from_meta(v)
            state_dict["master_weights"][k] = var_tmp
        if self.merged_model_params_meta:
            state_dict["merged_model_params"] = {}
            for k, v in self.merged_model_params_meta.items():
                var_tmp = self.restore_tensor_from_meta(v)
                state_dict["merged_model_params"][k] = var_tmp
        return state_dict

    @imperative_base.no_grad()
    def restore_tensor_from_meta(self, tensor_meta):
        shape = tensor_meta["shape"]
        name = tensor_meta["name"]
        start = tensor_meta["start"]
        end = tensor_meta["end"]
        tensor = self.cpu_buffer._slice(start, end)
        tensor.get_tensor()._set_dims(shape)
        tensor.name = name
        return tensor
