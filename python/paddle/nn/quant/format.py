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
"""Define some layers used to export quantization model with ONNX style."""
from __future__ import annotations

import abc

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.base import unique_name
from paddle.framework import in_dynamic_mode, in_pir_mode

from ..layer.layers import Layer


def fake_fp8_quant(input, scale, axis=-1, type='e4m3'):
    # only support channelwise or tensorwise
    if axis >= 0:
        shape = [1] * len(input.shape)
        shape[axis] = scale.numel()
        scale = scale.reshape(shape)
    inp = input.astype("float32")

    if type == 'e4m3':
        return paddle.cast(
            (inp * 448 / scale).clip(-448, 448), "float8_e4m3fn"
        ).astype(
            input.dtype
        )  # clip then cast
    elif type == 'e5m2':
        return paddle.cast(
            (inp * 57344 / scale).clip(-57344, 57344), "float8_e5m2"
        ).astype(
            input.dtype
        )  # clip then cast
    else:
        raise NotImplementedError("only support e4m3 or e5m2 now")


def fake_fp8_dequant(input, scale, axis=-1, type='e4m3'):
    # only support channelwise or tensorwise
    if axis >= 0:
        shape = [1] * len(input.shape)
        shape[axis] = scale.numel()
        scale = scale.reshape(shape)
    if type == 'e4m3':
        return (input.astype("float32") / 448 * scale).astype(input.dtype)
    elif type == 'e5m2':
        return (input.astype("float32") / 57344 * scale).astype(input.dtype)
    else:
        raise NotImplementedError("only support e4m3 or e5m2 now")


class LinearQuanterDequanter(Layer):
    def __init__(self, quanter, dequanter):
        super().__init__()
        self._quanter = quanter
        self._dequanter = dequanter

    def forward(self, input):
        out = input
        if self._quanter is not None:
            out = self._quanter(out)
        if self._dequanter is not None:
            out = self._dequanter(out)
        return out

    @staticmethod
    def from_quanter(quanter):
        assert quanter is not None
        return LinearQuanterDequanter(
            LinearQuanter.from_quanter(quanter),
            LinearDequanter.from_quanter(quanter),
        )


class LinearQuanter(Layer):
    def __init__(
        self,
        scales,
        zero_point=None,
        quant_axis=None,
        bit_length=8,
        group_size=128,
    ):
        super().__init__()
        scales = paddle.to_tensor(scales, dtype="float32")
        scale_attr = paddle.framework.ParamAttr(
            name=paddle.utils.unique_name.generate('quant_dequant.scale'),
            initializer=paddle.nn.initializer.Constant(1.0),
            trainable=False,
        )
        self._scales = self.create_parameter(
            shape=scales.shape, attr=scale_attr, dtype="float32"
        )
        self._scales.set_value(scales)
        self.in_accum = paddle.to_tensor(0.0, dtype="float32")
        self.in_state = paddle.to_tensor(0.0, dtype="float32")
        zero_point = zero_point if zero_point is not None else zero_point
        zero_point = paddle.to_tensor(zero_point, dtype="float32")
        zp_attr = paddle.framework.ParamAttr(
            name=paddle.utils.unique_name.generate('quant_dequant.zero_point'),
            initializer=paddle.nn.initializer.Constant(0.0),
            trainable=False,
        )
        self._zero_point = self.create_parameter(
            shape=zero_point.shape, attr=zp_attr, dtype="float32"
        )
        self._zero_point.set_value(zero_point)
        self._quant_axis = -1 if quant_axis is None else quant_axis
        self._bit_length = bit_length
        self._group_size = group_size
        if isinstance(self._bit_length, tuple):
            if (
                self._bit_length[0] == 4
                and self._bit_length[1] == 3
                and len(self._bit_length) == 2
            ):
                self._qmin = -1 * 448
                self._qmax = 448
            elif (
                self._bit_length[0] == 5
                and self._bit_length[1] == 2
                and len(self._bit_length) == 2
            ):
                self._qmin = -1 * 57344
                self._qmax = 57344
            else:
                raise NotImplementedError(
                    "Currently, only float8_e4m3 and float8_e5m2 formats are supported. Please set quant_bits to (4,3) or (5,2) for the corresponding format."
                )
        else:
            self._qmax = (1 << (self._bit_length - 1)) - 1
            self._qmin = -1 * self._qmax - 1
        if isinstance(self._bit_length, tuple):
            self._bit_length = self._bit_length[0] + self._bit_length[1] + 1

    def forward(self, input):
        if in_dynamic_mode():
            if len(self._scales.shape) > 1:
                if self._zero_point.sum() != 0:
                    quant_weight = paddle.clip(
                        paddle.round(input.cast('float32') / self._scales)
                        + self._zero_point,
                        self._qmin,
                        self._qmax,
                    )
                else:
                    new_s = paddle.repeat_interleave(
                        self._scales, self._group_size, 0
                    )
                    new_zp = paddle.repeat_interleave(
                        self._zero_point, self._group_size, 0
                    )
                    quant_weight = paddle.clip(
                        paddle.round(input.cast('float32') / new_s * self._qmax)
                        + new_zp,
                        self._qmin,
                        self._qmax,
                    )
                return quant_weight.cast(input.dtype)

            if self._qmax == 448:
                return fake_fp8_quant(
                    input, self._scales, self._quant_axis, type='e4m3'
                )
            elif self._qmax == 57344:
                return fake_fp8_quant(
                    input, self._scales, self._quant_axis, type='e5m2'
                )

            return _legacy_C_ops.quantize_linear(
                input.cast('float32'),
                self._scales,
                self._zero_point,
                "quant_axis",
                self._quant_axis,
                "bit_length",
                self._bit_length,
                "qmin",
                self._qmin,
                "qmax",
                self._qmax,
            ).cast(input.dtype)
        if in_pir_mode():
            input.stop_gradient = True
            quant_out = paddle.pir.core.create_persistable_value(
                dtype='float32',
                shape=input.shape,
                name=unique_name.generate("quant_out"),
                initializer=paddle.nn.initializer.Constant(0.0),
                stop_gradient=True,
            )
            # TODO(xiaoluomi): need to add only observer pass for quantize_linear
            quant_out, out_state, out_accum, out_scale = _C_ops.quantize_linear(
                input,
                self._scales,
                self._zero_point,
                self.in_accum,
                self.in_state,
                self._quant_axis,
                self._bit_length,
                self._qmin,
                self._qmax,
                0,
                True,
                False,
            )
            return quant_out
        else:
            out = self._helper.create_variable_for_type_inference(input.dtype)
            self._helper.append_op(
                type='quantize_linear',
                inputs={
                    'X': input,
                    'Scale': self._scales,
                    'ZeroPoint': self._zero_point,
                },
                outputs={'Y': out},
                attrs={
                    'quant_axis': self._quant_axis,
                    'bit_length': self._bit_length,
                    'qmin': self._qmin,
                    'qmax': self._qmax,
                },
            )
            return out

    @staticmethod
    def from_quanter(quanter):
        return LinearQuanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )


class LinearDequanter(Layer):
    def __init__(
        self,
        scales,
        zero_point=None,
        quant_axis=None,
        bit_length=8,
        group_size=128,
    ):
        super().__init__()
        scales = paddle.to_tensor(scales, dtype="float32")
        scale_attr = paddle.framework.ParamAttr(
            name=paddle.utils.unique_name.generate('quant_dequant.scale'),
            initializer=paddle.nn.initializer.Constant(1.0),
            trainable=False,
        )
        self._scales = self.create_parameter(
            shape=scales.shape, attr=scale_attr, dtype="float32"
        )
        self._scales.set_value(scales)
        self.in_accum = paddle.to_tensor(0.0, dtype="float32")
        self.in_state = paddle.to_tensor(0.0, dtype="float32")
        zero_point = zero_point if zero_point is not None else zero_point
        zero_point = paddle.to_tensor(zero_point, dtype="float32")
        zp_attr = paddle.framework.ParamAttr(
            name=paddle.utils.unique_name.generate('quant_dequant.zero_point'),
            initializer=paddle.nn.initializer.Constant(0.0),
            trainable=False,
        )
        self._zero_point = self.create_parameter(
            shape=zero_point.shape, attr=zp_attr, dtype="float32"
        )
        self._zero_point.set_value(zero_point)
        self._quant_axis = -1 if quant_axis is None else quant_axis
        self._bit_length = bit_length
        self._group_size = group_size
        if isinstance(self._bit_length, tuple):
            if (
                self._bit_length[0] == 4
                and self._bit_length[1] == 3
                and len(self._bit_length) == 2
            ):
                self._qmin = -1 * 448
                self._qmax = 448
            elif (
                self._bit_length[0] == 5
                and self._bit_length[1] == 2
                and len(self._bit_length) == 2
            ):
                self._qmin = -1 * 57344
                self._qmax = 57344
            else:
                raise NotImplementedError(
                    "Currently, only float8_e4m3 and float8_e5m2 formats are supported. Please set quant_bits to (4,3) or (5,2) for the corresponding format."
                )
        else:
            self._qmax = (1 << (self._bit_length - 1)) - 1
            self._qmin = -1 * self._qmax - 1
        if isinstance(self._bit_length, tuple):
            self._bit_length = self._bit_length[0] + self._bit_length[1] + 1

    def forward(self, input):
        if in_dynamic_mode():
            if len(self._scales.shape) > 1:
                if self._zero_point.sum() != 0:
                    quant_dequant_weight = (
                        input.cast('float32') - self._zero_point
                    ) * self._scales
                else:
                    new_s = paddle.repeat_interleave(
                        self._scales, self._group_size, 0
                    )
                    new_zp = paddle.repeat_interleave(
                        self._zero_point, self._group_size, 0
                    )
                    quant_dequant_weight = (
                        (input.cast('float32') - new_zp) / self._qmax * new_s
                    )
                return quant_dequant_weight.cast(input.dtype)

            if self._qmax == 448:
                return fake_fp8_dequant(
                    input, self._scales, self._quant_axis, type='e4m3'
                )
            elif self._qmax == 57344:
                return fake_fp8_dequant(
                    input, self._scales, self._quant_axis, type='e5m2'
                )

            return _legacy_C_ops.dequantize_linear(
                input.cast('float32'),
                self._scales,
                self._zero_point,
                "quant_axis",
                self._quant_axis,
                "bit_length",
                self._bit_length,
                "qmin",
                self._qmin,
                "qmax",
                self._qmax,
            ).cast(input.dtype)
        if in_pir_mode():
            input.stop_gradient = True
            dequant_out = paddle.pir.core.create_persistable_value(
                dtype='float32',
                shape=input.shape,
                name=unique_name.generate("quant_out"),
                initializer=paddle.nn.initializer.Constant(0.0),
                stop_gradient=True,
            )
            # TODO(xiaoluomi): need to add only observer pass for dequantize_linear
            dequant_out, out_state, out_accum, out_scale = (
                _C_ops.dequantize_linear(
                    input,
                    self._scales,
                    self._zero_point,
                    self.in_accum,
                    self.in_state,
                    self._quant_axis,
                    self._bit_length,
                    self._qmin,
                    self._qmax,
                    0,
                    True,
                    False,
                )
            )
            return dequant_out
        else:
            out = self._helper.create_variable_for_type_inference(input.dtype)
            self._helper.append_op(
                type='dequantize_linear',
                inputs={
                    'X': input,
                    'Scale': self._scales,
                    'ZeroPoint': self._zero_point,
                },
                outputs={'Y': out},
                attrs={
                    'quant_axis': self._quant_axis,
                    'bit_length': self._bit_length,
                    'qmin': self._qmin,
                    'qmax': self._qmax,
                },
            )
            return out

    @staticmethod
    def from_quanter(quanter):
        return LinearDequanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )


class ConvertibleQuantedLayer(Layer, metaclass=abc.ABCMeta):
    r"""Abstract class to help convert quantized layer to inference model.
    It defines some functions to convert quantizers and observers to quantize
    or dequantize operators that maintain the quantization parameters used
    during inference.

    Examples:
        .. code-block:: python

            >>> # Given codes in ./customized_quanter.py
            >>> class CustomizedQuantedLayer(ConvertibleQuantedLayer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight_a = paddle.create_parameter(shape=[1], dtype='float32')
            ...         self.weight_b = paddle.create_parameter(shape=[1], dtype='float32')
            ...         self.quanter_for_weight_a = None
            ...         self.activation_weight = None
            ...
            ...     def forward(self, input):
            ...         qweight_a = self.quanter_for_weight_a(self.weight_a)
            ...         weight_b = self.weight_b
            ...         qinput = self.activation_weight(input)
            ...         # compute with qweight_a, weight_b and qinput.
            ...         return qweight * qinput + weight_b
            ...
            ...     def weights_to_quanters(self):
            ...         return [('weight_a', 'quanter_for_weight_a')]
            ...
            ...     def activation_quanters(self):
            ...         return ['activation_weight']
    """

    def __init__(self):
        super().__init__()
        self.converted = False

    @abc.abstractmethod
    def weights_to_quanters(self) -> list[tuple[str, str]]:
        r"""Get the name pairs of weights to be quantized and their corresponding
        quantizers. In the convert function of this abstract class, it will call
        the ‘weights_to_quanters’ function and do something as follows:
        For each pair, the quantizer will be converted to a quantize operator and
        a dequantize operator. Then, the weight will be quantized by the quantize
        operator. Finally, the quantize operator will be removed and the weights
        will be stored in integer data type.

        Returns: A list of name pairs. Each pair contains two names. The first is name of weight
        to be quantized and the second is name of corresponding quanter.
        """
        pass

    @abc.abstractmethod
    def activation_quanters(self) -> list[str]:
        r"""Get the names of quanters used to quantize activations.
        All the quanters or observers returned by this function will be converted to quantize
        and dequantize operators for deployment.
        Returns: A list of quanter names.
        """
        pass

    def _convert_quanter_to_qdq(self, quanter_name) -> LinearQuanterDequanter:
        r"""Convert quanter to an instance of LinearQuanterDequanter."""
        if not hasattr(self, quanter_name):
            return None
        quanter = getattr(self, quanter_name)
        if quanter is None:
            return None
        elif type(quanter) == LinearQuanterDequanter:
            return None
        quanter = LinearQuanterDequanter.from_quanter(quanter)
        setattr(self, quanter_name, quanter)
        self._sub_layers[quanter_name] = quanter
        return quanter

    def _quant_weights(self, weight_name, quanter):
        r"""Quantize the weight by given quanter."""
        weight = getattr(self, weight_name)
        qweight = quanter(weight)
        weight.set_value(qweight)

    def _convert(self, remain_weight=False):
        r"""Convert current layer to onnx style for inference."""
        assert not self.converted, "The model should be converted only once."
        for weight_name, quanter_name in self.weights_to_quanters():
            qdq = self._convert_quanter_to_qdq(quanter_name)
            if qdq is not None and remain_weight is False:
                self._quant_weights(weight_name, qdq._quanter)
                qdq._quanter = None
                qdq._sub_layers['_quanter'] = None

        for quanter_name in self.activation_quanters():
            self._convert_quanter_to_qdq(quanter_name)

        self.converted = True
