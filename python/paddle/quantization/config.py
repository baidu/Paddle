# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from typing import TYPE_CHECKING

import paddle
from paddle import nn

from .wrapper import ObserveWrapper
from ..nn.quant.format import LinearQuanterDequanter, LinearQuanter, LinearDequanter

if TYPE_CHECKING:
    from paddle.nn import Layer

    from .factory import QuanterFactory


# TODO: Implement quanted layer and fill the mapping dict
DEFAULT_QAT_LAYER_MAPPINGS: dict[Layer, Layer] = {
    nn.quant.Stub: nn.quant.stub.QuanterStub,
    nn.Linear: nn.quant.qat.QuantedLinear,
    nn.Conv2D: nn.quant.qat.QuantedConv2D,
}

DEFAULT_LEAVES = [nn.ReLU, nn.AvgPool2D]


class SingleLayerConfig:
    r"""
    Configure how to quantize the activations and weights of a single layer.

    Args:
        activation(QuanterFactory): The factory to create instance of quanter used to quantize activations.
        weight(QuanterFactory): The factory to create instance of quanter used to quantize weights.
    """

    def __init__(
        self, activation: QuanterFactory, weight: QuanterFactory
    ) -> None:
        self._activation = activation
        self._weight = weight

    @property
    def activation(self) -> QuanterFactory:
        return self._activation

    @property
    def weight(self) -> QuanterFactory:
        return self._weight

    def __str__(self):
        return f"activation: {self._activation}\nweight: {self._weight}"


class QuantConfig:
    r"""
    Configure how to quantize a model or a part of the model. It will map each layer to
    an instance of SingleLayerConfig by the settings. It provides diverse methods to set
    the strategies of quantization.

    Args:
        activation(QuanterFactory | None): The global quantizer used to quantize the activations.
        weight(QuanterFactory | None): The global quantizer used to quantize the weights.

    Examples:
        .. code-block:: python

            >>> from paddle.quantization import QuantConfig
            >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver

            >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
            >>> q_config = QuantConfig(activation=quanter, weight=quanter)
            >>> print(q_config)
            Global config:
            activation: FakeQuanterWithAbsMaxObserver(name=None,moving_rate=0.9,bit_length=8,dtype=float32)
            weight: FakeQuanterWithAbsMaxObserver(name=None,moving_rate=0.9,bit_length=8,dtype=float32)

    """

    def __init__(
        self, activation: QuanterFactory | None, weight: QuanterFactory | None
    ) -> None:
        if activation is None and weight is None:
            self._global_config = None
        else:
            self._global_config = SingleLayerConfig(activation, weight)
        self._layer2config = {}
        self._prefix2config = {}
        self._type2config = {}
        self._model = None
        self._qat_layer_mapping = copy.deepcopy(DEFAULT_QAT_LAYER_MAPPINGS)
        self._customized_qat_layer_mapping = {}

        self._customized_leaves = []

    def add_layer_config(
        self,
        layer: Layer | list[Layer],
        activation: QuanterFactory | None = None,
        weight: QuanterFactory | None = None,
    ) -> None:
        r"""
        Set the quantization config by layer. It has the highest priority among
        all the setting methods.

        Args:
            layer(Layer|list[Layer]]): One or a list of layers.
            activation(QuanterFactory | None): Quanter used for activations. Default is None.
            weight(QuanterFactory | None): Quanter used for weights. Default is None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.nn import Linear
                >>> from paddle.quantization import QuantConfig
                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver

                >>> class Model(paddle.nn.Layer):
                ...    def __init__(self):
                ...        super().__init__()
                ...        self.fc = Linear(576, 120)
                >>> model = Model()
                >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
                >>> q_config = QuantConfig(activation=None, weight=None)
                >>> q_config.add_layer_config([model.fc], activation=quanter, weight=quanter)
                >>> # doctest: +SKIP('random memory address')
                >>> print(q_config)
                Global config:
                None
                Layer prefix config:
                {'linear_0': <paddle.quantization.config.SingleLayerConfig object at 0x7fe41a680ee0>}

        """
        if isinstance(layer, list):
            for _element in layer:
                self.add_layer_config(
                    _element, activation=activation, weight=weight
                )
        else:
            self.add_name_config(
                layer.full_name(), activation=activation, weight=weight
            )

    def add_name_config(
        self,
        layer_name: str | list[str],
        activation: QuanterFactory | None = None,
        weight: QuanterFactory | None = None,
    ) -> None:
        r"""
        Set the quantization config by full name of layer. Its priority is
        lower than `add_layer_config`.

        Args:
            layer_name(str|list[str]): One or a list of layers' full name.
            activation(QuanterFactory | None): Quanter used for activations. Default is None.
            weight(QuanterFactory | None): Quanter used for weights. Default is None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.nn import Linear
                >>> from paddle.quantization import QuantConfig
                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver

                >>> class Model(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.fc = Linear(576, 120)
                >>> model = Model()
                >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
                >>> q_config = QuantConfig(activation=None, weight=None)
                >>> q_config.add_name_config([model.fc.full_name()], activation=quanter, weight=quanter)
                >>> # doctest: +SKIP('random memory address')
                >>> print(q_config)
                Global config:
                None
                Layer prefix config:
                {'linear_0': <paddle.quantization.config.SingleLayerConfig object at 0x7fe41a680fd0>}

        """
        if isinstance(layer_name, str):
            config = SingleLayerConfig(activation, weight)
            self._prefix2config[layer_name] = config
        if isinstance(layer_name, list):
            for _element in layer_name:
                self.add_name_config(
                    _element, activation=activation, weight=weight
                )

    def add_type_config(
        self,
        layer_type: type[Layer] | list[type[Layer]],
        activation: QuanterFactory | None = None,
        weight: QuanterFactory | None = None,
    ) -> None:
        r"""
        Set the quantization config by the type of layer. The `layer_type` should be
        subclass of `paddle.nn.Layer`. Its priority is lower than `add_layer_config`
        and `add_name_config`.

        Args:
            layer_type(type[Layer] | list[type[Layer]]): One or a list of layers' type. It should be subclass of
            `paddle.nn.Layer`. Python build-in function `type()` can be used to get the type of a layer.
            activation(QuanterFactory | None): Quanter used for activations. Default is None.
            weight(QuanterFactory | None): Quanter used for weights. Default is None.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.nn import Linear
                >>> from paddle.quantization import QuantConfig
                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver

                >>> class Model(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.fc = Linear(576, 120)
                >>> model = Model()
                >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
                >>> q_config = QuantConfig(activation=None, weight=None)
                >>> q_config.add_type_config([Linear], activation=quanter, weight=quanter)
                >>> # doctest: +SKIP('random memory address')
                >>> print(q_config)
                Global config:
                None
                Layer type config:
                {<class 'paddle.nn.layer.common.Linear'>: <paddle.quantization.config.SingleLayerConfig object at 0x7fe41a680a60>}

        """
        if isinstance(layer_type, type) and issubclass(
            layer_type, paddle.nn.Layer
        ):
            config = SingleLayerConfig(activation, weight)
            self._type2config[layer_type] = config
        if isinstance(layer_type, list):
            for _element in layer_type:
                self.add_type_config(
                    _element, activation=activation, weight=weight
                )

    def add_qat_layer_mapping(
        self, source: type[Layer], target: type[Layer]
    ) -> None:
        r"""
        Add rules converting layers to simulated quantization layers
        before quantization-aware training. It will convert layers
        with type `source` to layers with type `target`. `source` and
        `target` should be subclass of `paddle.nn.Layer`. And a default
        mapping is provided by property `default_qat_layer_mapping`.

        Args:
            source(type[Layer]): The type of layers that will be converted.
            target(type[Layer]): The type of layers that will be converted to.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.nn import Conv2D
                >>> from paddle.quantization import QuantConfig
                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
                >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
                >>> q_config = QuantConfig(activation=None, weight=None)
                >>> class CustomizedQuantedConv2D(paddle.nn.Layer):
                ...     def forward(self, x):
                ...         pass
                ...         # add some code for quantization simulation
                >>> q_config.add_qat_layer_mapping(Conv2D, CustomizedQuantedConv2D)
        """
        assert isinstance(source, type) and issubclass(
            source, paddle.nn.Layer
        ), "The source layer to be placed should be a subclass of paddle.nn.Layer"
        assert isinstance(target, type) and issubclass(
            target, paddle.nn.Layer
        ), "The target layer should be a subclass of paddle.nn.qat.Layer"
        self._qat_layer_mapping[source] = target
        self._customized_qat_layer_mapping[source] = target

    def add_customized_leaf(self, layer_type: type[Layer]) -> None:
        r"""
        Declare the customized layer as leaf of model for quantization.
        The leaf layer is quantized as one layer. The sublayers of
        leaf layer will not be quantized.

        Args:
            layer_type(type[Layer]): The type of layer to be declared as leaf.

        Examples:
            .. code-block:: python

                >>> from paddle.nn import Sequential
                >>> from paddle.quantization import QuantConfig
                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
                >>> q_config = QuantConfig(activation=None, weight=None)
                >>> q_config.add_customized_leaf(Sequential)

        """
        self._customized_leaves.append(layer_type)

    @property
    def customized_leaves(self) -> list[type[Layer]]:
        r"""
        Get all the customized leaves.
        """
        return self._customized_leaves

    def _need_observe(self, layer: Layer):
        r"""
        Whether the layer should be observed by observer.
        """
        return self._is_leaf(layer) and self._has_observer_config(layer) and self._is_not_converted_quanter(layer)

    def _is_not_converted_quanter(self, layer):
        if type(layer) == LinearDequanter or type(layer) == LinearQuanter or type(layer) == LinearQuanterDequanter:
            return False
        return True

    def _get_qat_layer(self, layer: Layer):
        q_config = self._get_config_by_layer(layer)

        target_type = self._customized_qat_layer_mapping.get(
            type(layer), self.qat_layer_mappings.get(type(layer))
        )
        return target_type(layer, q_config)

    def _has_observer_config(self, layer: Layer):
        r"""
        Whether the layer has been configured for activation quantization.
        """
        _config = self._get_config_by_layer(layer)
        return _config is not None and _config.activation is not None

    def _is_leaf(self, layer: Layer):
        return (
            self._is_default_leaf(layer)
            or self._is_real_leaf(layer)
            or self._is_customized_leaf(layer)
        )

    def _is_default_leaf(self, layer: Layer):
        return type(layer) in DEFAULT_LEAVES

    def _is_real_leaf(self, layer: Layer):
        r"""
        The leaf is real leaf when it has no sublayers.
        """
        return layer._sub_layers is None or len(layer._sub_layers) == 0

    def _is_customized_leaf(self, layer: Layer):
        return type(layer) in self.customized_leaves

    def _get_observer(self, layer: Layer):
        r"""
        Create an instance of observer or quanter according to the
        given layer's quantization config.
        """
        _config = self._get_config_by_layer(layer)
        _observer = None if _config is None else _config.activation
        return None if _observer is None else _observer._instance(layer)

    def _get_observe_wrapper(self, layer: Layer):
        _observer = self._get_observer(layer)
        return ObserveWrapper(_observer, layer)

    @property
    def qat_layer_mappings(self) -> dict[Layer, Layer]:
        return self._qat_layer_mapping

    @property
    def default_qat_layer_mapping(self) -> dict[Layer, Layer]:
        return DEFAULT_QAT_LAYER_MAPPINGS

    @property
    def global_config(self) -> SingleLayerConfig:
        return self._global_config

    def _get_config_by_layer(self, layer) -> SingleLayerConfig:
        return self._layer2config.get(layer, None)

    def _is_quantifiable(self, layer: Layer):
        r"""
        The layer is quantifiable when it configured by activation quanter/observer
        or weight quanter/observer.
        """
        return layer in self._layer2config

    def _specify(self, model: Layer):
        r"""
        Specify the quantization config of each sublayer in model.
        For each layer in sublayers of mode,
        1. Set the config by global config
        2. Overwrite the config with parents' config
        3. Overwrite the config with config set by layer's type
        4. Overwrite the config with config set by layer's full name
        5. Overwrite the config with config set by layer

        Args:
            model(Layer): The model to be specified by the config.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> from paddle.nn import Linear, Sequential
                >>> from paddle.quantization import QuantConfig
                >>> from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver

                >>> class Model(paddle.nn.Layer):
                ...     def __init__(self):
                ...         super().__init__()
                ...         self.fc = Sequential(Linear(576, 120),Linear(576, 120))
                >>> model = Model()
                >>> quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.9)
                >>> q_config = QuantConfig(activation=None, weight=None)
                >>> q_config.add_layer_config([model.fc], activation=quanter, weight=quanter)
                >>> q_config._specify(model)
        """
        self._model = model
        self._specify_helper(self._model)

    def _specify_helper(self, model: Layer):
        for child in model.children():
            layer_prefix = child.full_name()
            config = self._layer2config.get(model, self.global_config)

            config = self._type2config.get(type(child), config)
            config = self._prefix2config.get(layer_prefix, config)
            if config is not None:
                self._layer2config[child] = config
            self._specify_helper(child)
        return self

    def details(self) -> str:
        r"""
        Get the formatted details of current config.
        """
        if self._model is None:
            return self.__str__()
        return self._details_helper(self._model)

    def _details_helper(self, layer: Layer):
        sublayer_lines = []
        for name, sublayer in layer.named_children():
            sublayer_str = self._details_helper(sublayer)
            sublayer_str = self._addindent(sublayer_str, 2)
            if sublayer in self._layer2config:
                sublayer_lines.append(
                    '('
                    + name
                    + '): '
                    + sublayer_str
                    + ', '
                    + str(self._layer2config[sublayer])
                )

        final_str = layer.__class__.__name__ + '('
        if sublayer_lines:
            final_str += '\n  ' + '\n  '.join(sublayer_lines) + '\n'

        final_str += ')'
        return final_str

    def _addindent(self, string, indent):
        s1 = string.split('\n')
        if len(s1) == 1:
            return string
        s2 = []
        for idx, line in enumerate(s1):
            if idx > 0:
                s2.append(str((indent * ' ') + line))
        return s1[0] + '\n' + '\n'.join(s2)

    def __str__(self):
        result = ""
        result += f"Global config:\n{self._global_config}\n"
        if len(self._type2config) > 0:
            result += f"Layer type config:\n{self._type2config}\n"
        if len(self._prefix2config) > 0:
            result += f"Layer prefix config: \n{self._prefix2config}\n"
        return result
