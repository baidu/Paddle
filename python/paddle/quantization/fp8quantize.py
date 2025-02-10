# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import abc
import copy

from paddle.nn import Layer
from paddle.nn.quant.fp8format import (
    FP8ConvertibleQuantedLayer,
    FP8LinearQuanterDequanter,
)

from .base_quanter import BaseQuanter
from .config import QuantConfig
from paddleslim.quant.layers import FP8QuantedLinear
class FP8Quantization(metaclass=abc.ABCMeta):
    r"""
    Abstract class used to prepares a copy of the model for quantization calibration or quantization-aware training.
    Args:
        config(QuantConfig) - Quantization configuration
    """

    def __init__(self, config: QuantConfig):
        self._config = copy.deepcopy(config)

    @abc.abstractmethod
    def quantize(self, model: Layer, inplace=False):
        r"""Create a model for quantization-aware training or post-training quantization."""
        pass

    def convert(self, model: Layer, inplace=False):
        r"""Convert the quantization model to onnx style. And the converted
        model can be saved as inference model by calling paddle.jit.save.
        Args:
            model(Layer) - The quantized model to be covnerted.
            inplace(bool) - Whether to modify the model in-place.

        Return: The converted model

        Examples:
        .. code-block:: python
            import paddle
            from paddle.quantization import QAT, QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            from paddle.vision.models import LeNet

            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            q_config = QuantConfig(activation=quanter, weight=quanter)
            qat = QAT(q_config)
            model = LeNet()
            quantized_model = qat.quantize(model)
            converted_model = qat.convert(quantized_model)
            dummy_data = paddle.rand([1, 1, 32, 32], dtype="float32")
            paddle.jit.save(converted_model, "./quant_deploy", [dummy_data])
        """
        _model = model if inplace else copy.deepcopy(model)
        replaced = {}
        for name, child in _model.named_children():
            quant_dequant = None
            if isinstance(child, FP8ConvertibleQuantedLayer):
                child._convert()
            elif isinstance(child, BaseQuanter):
                quant_dequant = FP8LinearQuanterDequanter.from_quanter(child)
            else:
                self.convert(child, inplace=True)
            if quant_dequant is not None:
                replaced[name] = quant_dequant
        for key, value in replaced.items():
            _model._sub_layers[key] = value
        return _model

    def _convert_to_quant_layers(self, model: Layer, config: QuantConfig):
        replaced = {}
        for name, child in model.named_children():
            if (
                config._is_quantifiable(child)
                and type(child) in config.qat_layer_mappings
            ):
                #print(config._get_qat_layer(child))
                replaced[name] = FP8QuantedLinear(child,config._get_config_by_layer(child))
            else:
                self._convert_to_quant_layers(child, config)
        for key, value in replaced.items():
            model._sub_layers[key] = value
        
    def _insert_activation_observers(self, model: Layer, config: QuantConfig):
        replaced = {}
        for name, child in model.named_children():
            if config._need_observe(child):
                replaced[name] = config._get_observe_wrapper(child)
            else:
                self._insert_activation_observers(child, config)
        for key, value in replaced.items():
            model._sub_layers[key] = value

    def _details(self):
        return self._config.details()

    def __str__(self):
        return self._details()

    def __repr__(self):
        return self.__str__()

