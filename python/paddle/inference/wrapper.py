#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

import paddle
from paddle.base import core
from paddle.base.core import (
    AnalysisConfig,
    PaddleDType,
    PaddleInferPredictor,
    PaddleInferTensor,
    PaddlePlace,
    convert_to_mixed_precision_bind,
)
from paddle.base.log_helper import get_logger

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import Unpack

    from paddle import Tensor

    class _WhiteList(TypedDict):
        white_list: set[str]


_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

DataType = PaddleDType
PlaceType = PaddlePlace
PrecisionType = AnalysisConfig.Precision
Config = AnalysisConfig
Tensor = PaddleInferTensor
Predictor = PaddleInferPredictor


def tensor_copy_from_cpu(self, data: npt.NDArray[Any] | list[str]) -> None:
    '''
    Support input type check based on tensor.copy_from_cpu.
    '''
    if isinstance(data, np.ndarray) or (
        isinstance(data, list) and len(data) > 0 and isinstance(data[0], str)
    ):
        self._copy_from_cpu_bind(data)
    else:
        raise TypeError(
            "In copy_from_cpu, we only support numpy ndarray and list[str] data type."
        )


def tensor_share_external_data(self, data: Tensor) -> None:
    '''
    Support input type check based on tensor.share_external_data.
    '''
    if isinstance(data, core.DenseTensor):
        self._share_external_data_bind(data)
    elif isinstance(data, paddle.Tensor):
        self._share_external_data_paddle_tensor_bind(data)
    elif isinstance(data, paddle.base.framework.Variable):
        raise TypeError(
            "The interface 'share_external_data' can only be used in dynamic graph mode. "
            "Maybe you called 'paddle.enable_static()' and you are in static graph mode now. "
            "Please use 'copy_from_cpu' instead."
        )
    else:
        raise TypeError(
            "In share_external_data, we only support Tensor and DenseTensor."
        )


def tensor_share_external_data_by_ptr_name(self, data, shape, dtype, place):
    '''
    Support tensor.share_external_data_by_ptr_name.
    '''
    self._share_external_data_by_ptr_name_bind(data, shape, dtype, place)


def convert_to_mixed_precision(
    model_file: str,
    params_file: str,
    mixed_model_file: str,
    mixed_params_file: str,
    mixed_precision: PrecisionType,
    backend: PlaceType,
    keep_io_types: bool = True,
    black_list: set[str] = set(),
    **kwargs: Unpack[_WhiteList],
) -> None:
    '''
    Convert a fp32 model to mixed precision model.

    Args:
        model_file: fp32 model file, e.g. inference.pdmodel.
        params_file: fp32 params file, e.g. inference.pdiparams.
        mixed_model_file: The storage path of the converted mixed-precision model.
        mixed_params_file: The storage path of the converted mixed-precision params.
        mixed_precision: The precision, e.g. PrecisionType.Half.
        backend: The backend, e.g. PlaceType.GPU.
        keep_io_types: Whether the model input and output dtype remains unchanged.
            Default is True.
        black_list: Operators that do not convert precision.
        kwargs: Supported keys including 'white_list'.
            - white_list: Operators that do convert precision.
    '''
    if backend is PlaceType.GPU and not core.is_compiled_with_cuda():
        _logger.error(
            "You should use PaddlePaddle compiled with GPU when backend set to PlaceType.GPU"
        )

    mixed_model_dirname = os.path.dirname(mixed_model_file)
    # Support mixed_params_file is empty, because some models don't have params, but convert_to_mixed_precision will call
    # constant_folding_pass, it will generate a new params file to save persistable vars, which is saved in the same
    # level file directory as the model file by default and ends in pdiparams.
    mixed_params_dirname = (
        os.path.dirname(mixed_params_file)
        if len(mixed_params_file) != 0
        else mixed_model_dirname
    )
    if not os.path.exists(mixed_params_dirname):
        os.makedirs(mixed_params_dirname)
    white_list = kwargs.get('white_list', set())
    convert_to_mixed_precision_bind(
        model_file,
        params_file,
        mixed_model_file,
        mixed_params_file,
        mixed_precision,
        backend,
        keep_io_types,
        black_list,
        white_list,
    )


Tensor.copy_from_cpu = tensor_copy_from_cpu
Tensor.share_external_data = tensor_share_external_data
Tensor.share_external_data_by_ptr_name = tensor_share_external_data_by_ptr_name
