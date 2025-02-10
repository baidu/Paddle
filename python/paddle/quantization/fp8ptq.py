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

import copy

from paddle.distributed import fleet
from paddle.nn import Layer

from .config import QuantConfig
from .fp8quantize import FP8Quantization

class FP8PTQ(FP8Quantization):
    """
    Applying post training quantization to the model.
    """

    def __init__(self, config: QuantConfig):
        super().__init__(config)

    def _is_parallel_training(self):
        try:
            if fleet.worker_num() > 2:
                return True
            else:
                return False
        except Exception:  # fleet is not initialized
            return False

    def quantize(self, model: Layer, inplace=False):
        r"""
        Create a model for post-training quantization.

        The quantization configuration will be propagated in the model.
        And it will insert observers into the model to collect and compute
        quantization parameters.

        Args:
            model(Layer) - The model to be quantized.
            inplace(bool) - Whether to modify the model in-place.

        Return: The prepared model for post-training quantization.

        Examples:
        .. code-block:: python
            from paddle.quantization import PTQ, QuantConfig
            from paddle.quantization.observers import AbsmaxObserver
            from paddle.vision.models import LeNet

            observer = AbsmaxObserver()
            q_config = QuantConfig(activation=observer, weight=observer)
            ptq = PTQ(q_config)
            model = LeNet()
            model.eval()
            quant_model = ptq.quantize(model)
            print(quant_model)
        """
        _model = model
        if not inplace:
            assert (
                not self._is_parallel_training()
            ), "'inplace' is not compatible with parallel training."
            _model = copy.deepcopy(model)
            _model.eval()
        assert (
            not model.training
        ), "Post-Training Quantization shoud not work on training models. Please set evaluation mode by model.eval()."
        self._config._specify(_model)
        self._convert_to_quant_layers(_model, self._config)
        self._insert_activation_observers(_model, self._config)
        return _model
