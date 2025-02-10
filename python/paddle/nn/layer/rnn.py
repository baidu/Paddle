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

import math
from collections.abc import Sequence
from functools import partial, reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

import paddle
from paddle import _C_ops, _legacy_C_ops, framework, in_dynamic_mode
from paddle.base.data_feeder import check_type, check_variable_and_dtype
from paddle.base.dygraph.base import NON_PERSISTABLE_VAR_NAME_SUFFIX
from paddle.base.framework import (
    default_startup_program,
    in_dynamic_or_pir_mode,
    program_guard,
)
from paddle.common_ops_import import Variable
from paddle.framework import core, in_pir_mode
from paddle.nn import (
    functional as F,
    initializer as I,
)
from paddle.tensor.manipulation import tensor_array_to_tensor

from .container import LayerList
from .layers import Layer

if TYPE_CHECKING:
    from typing import Literal

    from paddle import Tensor
    from paddle._typing import (
        DTypeLike,
        NestedStructure,
        ParamAttrLike,
        ShapeLike,
        TensorOrTensors,
    )

    _DirectionType = Literal["forward", "bidirect", "bidirectional"]
    _RNNType = Literal["LSTM", "GRU", "RNN_RELU", "RNN_TANH"]
    _ActivationType = Literal["tanh", "relu"]

__all__ = []


def rnn(
    cell: RNNCellBase,
    inputs: Tensor,
    initial_states: TensorOrTensors | None = None,
    sequence_length: Tensor | None = None,
    time_major: bool = False,
    is_reverse: bool = False,
    **kwargs: Any,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
    r"""
    rnn creates a recurrent neural network specified by RNNCell `cell`,
    which performs :code:`cell.call()` (for dygraph mode :code:`cell.forward`)
    repeatedly until reaches to the maximum length of `inputs`.

    Parameters:
        cell(RNNCellBase): An instance of `RNNCellBase`.
        inputs(Tensor): the input sequences.
            If time_major is True, the shape is
            `[time_steps, batch_size, input_size]`
            else the shape is `[batch_size, time_steps, input_size]`.
        initial_states(Tensor|tuple|list, optional): the initial state of the
            rnn cell. Tensor or a possibly nested structure of tensors. If not
            provided, `cell.get_initial_states` would be called to produce
            the initial state. Defaults to None.
        sequence_length (Tensor|None, optional): shape `[batch_size]`, dtype: int64
            or int32. The valid lengths of input sequences. Defaults to None.
            If `sequence_length` is not None, the inputs are treated as
            padded sequences. In each input sequence, elements whose time step
            index are not less than the valid length are treated as paddings.
        time_major (bool, optional): Whether the first dimension of the input means the
            time steps. Defaults to False.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Defaults to False.
        **kwargs: Additional keyword arguments to pass to `forward` of the cell.

    Returns:
        outputs (Tensor|list|tuple): the output sequence. Tensor or nested
            structure of Tensors.
            If `time_major` is True, the shape of each tensor in outputs is
            `[time_steps, batch_size, hidden_size]`, else
            `[batch_size, time_steps, hidden_size]`.
        final_states (Tensor|list|tuple): final states. A (possibly nested structure of)
            tensor[s], representing the final state for RNN. It has the same
            structure of initial state. Each tensor in final states has the same
            shape and dtype as the corresponding tensor in initial states.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> inputs = paddle.rand((4, 23, 16))
            >>> prev_h = paddle.randn((4, 32))

            >>> cell = paddle.nn.SimpleRNNCell(16, 32)
            >>> rnn = paddle.nn.RNN(cell)
            >>> outputs, final_states = rnn(inputs, prev_h)
            >>> print(outputs.shape)
            [4, 23, 32]
            >>> print(final_states.shape)
            [4, 32]

    """

    if in_dynamic_mode():
        return _rnn_dynamic_graph(
            cell,
            inputs,
            initial_states,
            sequence_length,
            time_major,
            is_reverse,
            **kwargs,
        )
    else:
        return _rnn_static_graph(
            cell,
            inputs,
            initial_states,
            sequence_length,
            time_major,
            is_reverse,
            **kwargs,
        )


class ArrayWrapper:
    def __init__(self, x: Tensor) -> None:
        self.array = [x]

    def append(self, x: Tensor) -> Self:
        self.array.append(x)
        return self

    def __getitem__(self, item: int) -> Tensor:
        return self.array.__getitem__(item)


def _maybe_copy(state: Tensor, new_state: Tensor, step_mask: Tensor) -> Tensor:
    """update rnn state or just pass the old state through"""
    new_state = paddle.tensor.math._multiply_with_axis(
        new_state, step_mask, axis=0
    ) + paddle.tensor.math._multiply_with_axis(state, (1 - step_mask), axis=0)
    return new_state


def _transpose_batch_time(x: Tensor) -> Tensor:
    perm = [1, 0, *list(range(2, len(x.shape)))]
    return paddle.transpose(x, perm)


def _rnn_dynamic_graph(
    cell,
    inputs,
    initial_states=None,
    sequence_length=None,
    time_major=False,
    is_reverse=False,
    **kwargs,
):
    time_step_index = 0 if time_major else 1
    flat_inputs = paddle.utils.flatten(inputs)
    time_steps = flat_inputs[0].shape[time_step_index]

    if initial_states is None:
        initial_states = cell.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )

    if not time_major:
        inputs = paddle.utils.map_structure(_transpose_batch_time, inputs)

    if sequence_length is not None:
        mask = paddle.static.nn.sequence_lod.sequence_mask(
            sequence_length, maxlen=time_steps, dtype=inputs.dtype
        )
        mask = paddle.transpose(mask, [1, 0])

    if is_reverse:
        inputs = paddle.utils.map_structure(
            lambda x: paddle.reverse(x, axis=[0]), inputs
        )
        mask = (
            paddle.reverse(mask, axis=[0])
            if sequence_length is not None
            else None
        )

    states = initial_states
    outputs = []
    for i in range(time_steps):
        step_inputs = paddle.utils.map_structure(lambda x: x[i], inputs)
        step_outputs, new_states = cell(step_inputs, states, **kwargs)
        if sequence_length is not None:
            new_states = paddle.utils.map_structure(
                partial(_maybe_copy, step_mask=mask[i]), states, new_states
            )
        states = new_states
        outputs = (
            paddle.utils.map_structure(lambda x: ArrayWrapper(x), step_outputs)
            if i == 0
            else paddle.utils.map_structure(
                lambda x, x_array: x_array.append(x), step_outputs, outputs
            )
        )

    final_outputs = paddle.utils.map_structure(
        lambda x: paddle.stack(x.array, axis=time_step_index), outputs
    )

    if is_reverse:
        final_outputs = paddle.utils.map_structure(
            lambda x: paddle.reverse(x, axis=time_step_index), final_outputs
        )

    final_states = new_states
    return final_outputs, final_states


def _rnn_static_graph(
    cell,
    inputs,
    initial_states=None,
    sequence_length=None,
    time_major=False,
    is_reverse=False,
    **kwargs,
):
    check_type(
        inputs, 'inputs', (Variable, list, tuple, paddle.pir.Value), 'rnn'
    )
    if isinstance(inputs, (list, tuple)):
        for i, input_x in enumerate(inputs):
            check_variable_and_dtype(
                input_x, 'inputs[' + str(i) + ']', ['float32', 'float64'], 'rnn'
            )
    check_type(
        initial_states,
        'initial_states',
        (Variable, list, tuple, type(None), paddle.pir.Value),
        'rnn',
    )

    check_type(
        sequence_length,
        'sequence_length',
        (Variable, type(None), paddle.pir.Value),
        'rnn',
    )

    def _switch_grad(x, stop=False):
        x.stop_gradient = stop
        return x

    if initial_states is None:
        initial_states = cell.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )
    initial_states = paddle.utils.map_structure(_switch_grad, initial_states)

    if not time_major:
        inputs = paddle.utils.map_structure(_transpose_batch_time, inputs)

    max_seq_len = paddle.shape(paddle.utils.flatten(inputs)[0])[0]
    max_seq_len = paddle.cast(max_seq_len, paddle.int32)
    if sequence_length is not None:
        mask = paddle.static.nn.sequence_lod.sequence_mask(
            sequence_length,
            maxlen=max_seq_len,
            dtype=paddle.utils.flatten(initial_states)[0].dtype,
        )
        mask = paddle.transpose(mask, [1, 0])
    if is_reverse:
        inputs = paddle.utils.map_structure(
            lambda x: paddle.reverse(x, axis=[0]), inputs
        )
        mask = (
            paddle.reverse(mask, axis=[0])
            if sequence_length is not None
            else None
        )

    with paddle.base.framework.device_guard("cpu"):
        start_i = paddle.zeros([], dtype="int64")
        end = max_seq_len

        end = paddle.cast(end, "int64")
        cond = start_i < end
    while_op = paddle.static.nn.control_flow.While(cond)

    out_array = paddle.tensor.create_array(
        dtype=paddle.utils.flatten(inputs)[0].dtype
    )

    init_array = paddle.utils.map_structure(
        lambda x: paddle.tensor.create_array(dtype=x.dtype), initial_states
    )

    paddle.utils.map_structure(
        lambda x, y: paddle.tensor.array_write(x, start_i, y),
        initial_states,
        init_array,
    )

    with while_op.block():
        step_in = inputs[start_i]
        # step_in = paddle.base.layers.Print( step_in, message="step in")
        pre_state = paddle.utils.map_structure(
            lambda x: paddle.tensor.array_read(x, start_i), init_array
        )
        outputs, new_states = cell(step_in, pre_state, **kwargs)
        assert isinstance(
            outputs, (paddle.base.framework.Variable, paddle.pir.Value)
        )
        paddle.utils.assert_same_structure(new_states, pre_state)
        if sequence_length is not None:
            step_mask = paddle.unsqueeze(mask[start_i], 1)
            # new_states = map_structure(
            #     partial(_maybe_copy, step_mask=step_mask),
            #     pre_state, new_states
            # )
            new_states = paddle.utils.map_structure(
                lambda x, y: (x * step_mask + y * (1.0 - step_mask)),
                new_states,
                pre_state,
            )

        paddle.tensor.array_write(outputs, start_i, out_array)

        with paddle.base.framework.device_guard("cpu"):
            start_i = paddle.tensor.increment(x=start_i, value=1)
        paddle.utils.map_structure(
            lambda x, y: paddle.tensor.array_write(x, start_i, y),
            new_states,
            init_array,
        )

        with paddle.base.framework.device_guard("cpu"):
            new_cond = paddle.tensor.less_than(start_i, end)
            paddle.assign(new_cond, cond)

    out, _ = tensor_array_to_tensor(out_array, axis=0, use_stack=True)

    all_state = paddle.utils.map_structure(
        lambda x: tensor_array_to_tensor(x, axis=0, use_stack=True)[0],
        init_array,
    )
    final_outputs = out
    final_states = paddle.utils.map_structure(lambda x: x[-1], all_state)

    if is_reverse:
        final_outputs = paddle.utils.map_structure(
            lambda x: paddle.reverse(x, axis=[0]), final_outputs
        )

    if not time_major:
        final_outputs = paddle.utils.map_structure(
            _transpose_batch_time, final_outputs
        )

    return (final_outputs, final_states)


def birnn(
    cell_fw: RNNCellBase,
    cell_bw: RNNCellBase,
    inputs: Tensor,
    initial_states: tuple[Tensor, Tensor] | list[Tensor] | None = None,
    sequence_length: Tensor | None = None,
    time_major: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    r"""
    birnn creates a bidirectional recurrent neural network specified by
    RNNCell `cell_fw` and `cell_bw`, which performs :code:`cell.call()`
    (for dygraph mode :code:`cell.forward`) repeatedly until reaches to
    the maximum length of `inputs` and then concat the outputs for both RNNs
    along the last axis.

    Parameters:
        cell_fw(RNNCellBase): An instance of `RNNCellBase`.
        cell_bw(RNNCellBase): An instance of `RNNCellBase`.
        inputs(Tensor): the input sequences.
            If time_major is True, the shape is
            `[time_steps, batch_size, input_size]`
            else the shape is `[batch_size, time_steps, input_size]`.
        initial_states(tuple|None, optional): A tuple of initial states of
            `cell_fw` and `cell_bw`.
            If not provided, `cell.get_initial_states` would be called to
            produce initial state for each cell. Defaults to None.
        sequence_length (Tensor|None, optional): shape `[batch_size]`, dtype: int64
            or int32. The valid lengths of input sequences. Defaults to None.
            If `sequence_length` is not None, the inputs are treated as
            padded sequences. In each input sequence, elements whose time step
            index are not less than the valid length are treated as paddings.
        time_major (bool): Whether the first dimension of the input means the
            time steps. Defaults to False.
        **kwargs: Additional keyword arguments to pass to `forward` of each cell.

    Returns:
        outputs (Tensor): the outputs of the bidirectional RNN. It is the
            concatenation of the outputs from the forward RNN and backward
            RNN along the last axis.
            If time_major is True, the shape is `[time_steps, batch_size, size]`,
            else the shape is `[batch_size, time_steps, size]`, where size is
            `cell_fw.hidden_size + cell_bw.hidden_size`.
        final_states (tuple): A tuple of the final states of the forward
            cell and backward cell.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> cell_fw = paddle.nn.LSTMCell(16, 32)
            >>> cell_bw = paddle.nn.LSTMCell(16, 32)
            >>> rnn = paddle.nn.BiRNN(cell_fw, cell_bw)
            >>> inputs = paddle.rand((2, 23, 16))
            >>> outputs, final_states = rnn(inputs)
            >>> print(outputs.shape)
            [2, 23, 64]
            >>> print(final_states[0][0].shape)
            [2, 32]

    """

    if initial_states is None:
        states_fw = cell_fw.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )
        states_bw = cell_fw.get_initial_states(
            batch_ref=inputs, batch_dim_idx=1 if time_major else 0
        )
    else:
        states_fw, states_bw = initial_states
    outputs_fw, states_fw = rnn(
        cell_fw,
        inputs,
        states_fw,
        sequence_length,
        time_major=time_major,
        **kwargs,
    )

    outputs_bw, states_bw = rnn(
        cell_bw,
        inputs,
        states_bw,
        sequence_length,
        time_major=time_major,
        is_reverse=True,
        **kwargs,
    )

    outputs = paddle.utils.map_structure(
        lambda x, y: paddle.concat([x, y], -1), outputs_fw, outputs_bw
    )

    final_states = (states_fw, states_bw)
    return outputs, final_states


def split_states(
    states: TensorOrTensors,
    bidirectional: bool = False,
    state_components: int = 1,
) -> list[Tensor] | list[list[Tensor]]:
    r"""
    Split states of RNN network into possibly nested list or tuple of
    states of each RNN cells of the RNN network.

    Parameters:
        states (Tensor|tuple|list): the concatenated states for RNN network.
            When `state_components` is 1, states in a Tensor with shape
            `(L*D, N, C)` where `L` is the number of layers of the RNN
            network, `D` is the number of directions of the RNN network(1
            for unidirectional RNNs and 2 for bidirectional RNNs), `N` is
            the batch size of the input to the RNN network, `C` is the
            hidden size of the RNN network.

            When `state_components` is larger than 1, `states` is a tuple of
            `state_components` Tensors that meet the requirements described
            above.

            For SimpleRNNs and GRUs, `state_components` is 1, and for LSTMs,
            `state_components` is 2.
        bidirectional (bool): whether the state is of a bidirectional RNN
            network. Defaults to False.
        state_components (int): the number of the components of the states. see
            `states` above. Defaults to 1.

    Returns:
        A nested list or tuple of RNN cell states.
        If `bidirectional` is True, it can be indexed twice to get an RNN
        cell state. The first index indicates the layer, the second index
        indicates the direction.
        If `bidirectional` is False, it can be indexed once to get an RNN
        cell state. The index indicates the layer.
        Note that if `state_components` is larger than 1, an RNN cell state
        can be indexed one more time to get a tensor of shape(N, C), where
        `N` is the batch size of the input to the RNN cell, and `C` is the
        hidden size of the RNN cell.
    """
    if state_components == 1:
        states = paddle.unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([paddle.unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def concat_states(
    states: Sequence[Tensor],
    bidirectional: bool = False,
    state_components: int = 1,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    Concatenate a possibly nested list or tuple of RNN cell states into a
    compact form.

    Parameters:
        states (list|tuple): a possibly nested list or tuple of RNN cell
            states.
            If `bidirectional` is True, it can be indexed twice to get an
            RNN cell state. The first index indicates the layer, the second
            index indicates the direction.
            If `bidirectional` is False, it can be indexed once to get an RNN
            cell state. The index indicates the layer.
            Note that if `state_components` is larger than 1, an RNN cell
            state can be indexed one more time to get a tensor of shape(N, C),
            where `N` is the batch size of the input to the RNN cell, and
            `C` is the hidden size of the RNN cell.
        bidirectional (bool): whether the state is of a bidirectional RNN
            network. Defaults to False.
        state_components (int): the number of the components of the states. see
            `states` above. Defaults to 1.

    Returns:
        Concatenated states for RNN network.
        When `state_components` is 1, states in a Tensor with shape
        `(L\*D, N, C)` where `L` is the number of layers of the RNN
        network, `D` is the number of directions of the RNN network(1 for
        unidirectional RNNs and 2 for bidirectional RNNs), `N` is the batch
        size of the input to the RNN network, `C` is the hidden size of the
        RNN network.

    """
    if state_components == 1:
        return paddle.stack(paddle.utils.flatten(states))
    else:
        states = paddle.utils.flatten(states)
        components = []
        for i in range(state_components):
            components.append(states[i::state_components])
        return tuple([paddle.stack(item) for item in components])


class RNNCellBase(Layer):
    r"""
    RNNCellBase is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def get_initial_states(
        self,
        batch_ref: Tensor,
        shape: NestedStructure[ShapeLike] | None = None,
        dtype: NestedStructure[DTypeLike] | None = None,
        init_value: float = 0.0,
        batch_dim_idx: int = 0,
    ) -> NestedStructure[Tensor]:
        r"""
        Generate initialized states according to provided shape, data type and
        value.

        Parameters:
            batch_ref (Tensor): A tensor, which shape would be used to
                determine the batch size, which is used to generate initial
                states. For `batch_ref`'s shape d, `d[batch_dim_idx]` is
                treated as batch size.
            shape (list|tuple|None, optional): A (possibly nested structure of) shape[s],
                where a shape is a list/tuple of integer. `-1` (for batch size)
                will be automatically prepended if a shape does not starts with
                it. If None, property `state_shape` will be used. Defaults to
                None.
            dtype (str|list|tuple|None, optional): A (possibly nested structure of)
                data type[s]. The structure must be same as that of `shape`,
                except when all tensors' in states has the same data type, a
                single data type can be used. If None and property `cell.state_shape`
                is not available, current default floating type of paddle is
                used. Defaults to None.
            init_value (float, optional): A float value used to initialize states.
                Defaults to 0.
            batch_dim_idx (int, optional): An integer indicating which
                dimension of the of `batch_ref` represents batch. Defaults to 0.

        Returns:
            init_states (Tensor|tuple|list): tensor of the provided shape and
                dtype, or list of tensors that each satisfies the requirements,
                packed in the same structure as `shape` and `type` does.
        """
        # TODO: use inputs and batch_size
        batch_ref = paddle.utils.flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            """For shape, list/tuple of integer is the finest-grained objection"""
            if isinstance(seq, (list, tuple)):
                if reduce(
                    lambda flag, x: isinstance(x, int) and flag, seq, True
                ):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return isinstance(seq, Sequence) and not isinstance(seq, str)

        class Shape:
            def __init__(self, shape):
                self.shape = (
                    list(shape) if shape[0] == -1 else ([-1, *list(shape)])
                )

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = paddle.utils.layers_utils.is_sequence
        paddle.utils.layers_utils.is_sequence = _is_shape_sequence
        states_shapes = paddle.utils.map_structure(
            lambda shape: Shape(shape), states_shapes
        )
        paddle.utils.layers_utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:
            states_dtypes = framework.get_default_dtype()
        if len(paddle.utils.flatten(states_dtypes)) == 1:
            dtype = paddle.utils.flatten(states_dtypes)[0]
            states_dtypes = paddle.utils.map_structure(
                lambda shape: dtype, states_shapes
            )
        fill_shapes = states_shapes
        if batch_ref.shape[batch_dim_idx] > 0:
            if isinstance(fill_shapes, list):
                for s in fill_shapes[0]:
                    s.shape[0] = batch_ref.shape[batch_dim_idx]
            elif isinstance(fill_shapes, tuple):
                for s in fill_shapes:
                    s.shape[0] = batch_ref.shape[batch_dim_idx]
            else:
                fill_shapes.shape[0] = batch_ref.shape[batch_dim_idx]
        else:
            if isinstance(fill_shapes, list):
                for s in fill_shapes[0]:
                    s.shape[0] = paddle.shape(batch_ref)[batch_dim_idx].item()
            elif isinstance(fill_shapes, tuple):
                for s in fill_shapes:
                    s.shape[0] = paddle.shape(batch_ref)[batch_dim_idx].item()
            else:
                fill_shapes.shape[0] = paddle.shape(batch_ref)[
                    batch_dim_idx
                ].item()

        init_states = paddle.utils.map_structure(
            lambda shape, dtype: paddle.full(
                shape=shape.shape,
                fill_value=init_value,
                dtype=dtype,
            ),
            fill_shapes,
            states_dtypes,
        )
        return init_states

    @property
    def state_shape(self) -> None:
        r"""
        Abstract method (property).
        Used to initialize states.
        A (possibly nested structure of) shape[s], where a shape is a
        list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementation for `state_shape` in the used cell."
        )

    @property
    def state_dtype(self) -> None:
        r"""
        Abstract method (property).
        Used to initialize states.
        A (possibly nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a single data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementation for `state_dtype` in the used cell."
        )


class SimpleRNNCell(RNNCellBase):
    r"""
    Elman RNN (SimpleRNN) cell. Given the inputs and previous states, it
    computes the outputs and updates states.

    The formula used is as follows:

    .. math::
        h_{t} & = act(W_{ih}x_{t} + b_{ih} + W_{hh}h_{t-1} + b_{hh})

        y_{t} & = h_{t}

    where :math:`act` is for :attr:`activation`.

    Please refer to `Finding Structure in Time
    <https://crl.ucsd.edu/~elman/Papers/fsit.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        activation (str, optional): The activation in the SimpleRNN cell.
            It can be `tanh` or `relu`. Defaults to `tanh`.
        weight_ih_attr (ParamAttr|None, optional): The parameter attribute for
            :math:`weight_ih`. Default: None.
        weight_hh_attr(ParamAttr|None, optional): The parameter attribute for
            :math:`weight_hh`. Default: None.
        bias_ih_attr (ParamAttr|None, optional): The parameter attribute for the
            :math:`bias_ih`. Default: None.
        bias_hh_attr (ParamAttr|None, optional): The parameter attribute for the
            :math:`bias_hh`. Default: None.
        name (str|None, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Variables:
        - **weight_ih** (Parameter): shape (hidden_size, input_size), input to hidden weight, corresponding to :math:`W_{ih}` in the formula.
        - **weight_hh** (Parameter): shape (hidden_size, hidden_size), hidden to hidden weight, corresponding to :math:`W_{hh}` in the formula.
        - **bias_ih** (Parameter): shape (hidden_size, ), input to hidden bias, corresponding to :math:`b_{ih}` in the formula.
        - **bias_hh** (Parameter): shape (hidden_size, ), hidden to hidden bias, corresponding to :math:`b_{hh}` in the formula.

    Inputs:
        - **inputs** (Tensor): shape `[batch_size, input_size]`, the input, corresponding to :math:`x_{t}` in the formula.
        - **states** (Tensor, optional): shape `[batch_size, hidden_size]`, the previous hidden state, corresponding to :math:`h_{t-1}` in the formula. When states is None, zero state is used. Defaults to None.

    Returns:
        - **outputs** (Tensor): shape `[batch_size, hidden_size]`, the output, corresponding to :math:`h_{t}` in the formula.
        - **states** (Tensor): shape `[batch_size, hidden_size]`, the new hidden state, corresponding to :math:`h_{t}` in the formula.

    Notes:
        All the weights and bias are initialized with `Uniform(-std, std)` by default. Where std = :math:`\frac{1}{\sqrt{hidden\_size}}`. For more information about parameter initialization, please refer to :ref:`api_paddle_ParamAttr`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn((4, 16))
            >>> prev_h = paddle.randn((4, 32))

            >>> cell = paddle.nn.SimpleRNNCell(16, 32)
            >>> y, h = cell(x, prev_h)
            >>> print(y.shape)
            [4, 32]

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: _ActivationType | str = "tanh",
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size of {self.__class__.__name__} must be greater than 0, but now equals to {hidden_size}"
            )
        std = 1.0 / math.sqrt(hidden_size)
        if weight_ih_attr is not False:
            self.weight_ih = self.create_parameter(
                (hidden_size, input_size),
                weight_ih_attr,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.weight_ih = self.create_parameter(
                (hidden_size, input_size),
                None,
                default_initializer=I.Constant(1.0),
            )
            self.weight_ih.stop_gradient = True

        if weight_hh_attr is not False:
            self.weight_hh = self.create_parameter(
                (hidden_size, hidden_size),
                weight_hh_attr,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.weight_hh = self.create_parameter(
                (hidden_size, hidden_size),
                None,
                default_initializer=I.Constant(1.0),
            )
            self.weight_hh.stop_gradient = True

        if bias_ih_attr is not False:
            self.bias_ih = self.create_parameter(
                (hidden_size,),
                bias_ih_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.bias_ih = self.create_parameter(
                (hidden_size,),
                None,
                is_bias=True,
                default_initializer=I.Constant(0.0),
            )
            self.bias_ih.stop_gradient = True

        if bias_hh_attr is not False:
            self.bias_hh = self.create_parameter(
                (hidden_size,),
                bias_hh_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.bias_hh = self.create_parameter(
                (hidden_size,),
                None,
                is_bias=True,
                default_initializer=I.Constant(0.0),
            )
            self.bias_hh.stop_gradient = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        if activation not in ["tanh", "relu"]:
            raise ValueError(
                "activation for SimpleRNNCell should be tanh or relu, "
                f"but get {activation}"
            )
        self.activation = activation
        self._activation_fn = paddle.tanh if activation == "tanh" else F.relu

    def forward(self, inputs: Tensor, states: Tensor | None = None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_h = states
        i2h = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = paddle.matmul(pre_h, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self._activation_fn(i2h + h2h)
        return h, h

    @property
    def state_shape(self) -> tuple[int]:
        return (self.hidden_size,)

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.activation != "tanh":
            s += ', activation={activation}'
        return s.format(**self.__dict__)


class LSTMCell(RNNCellBase):
    r"""
    Long-Short Term Memory(LSTM) RNN cell. Given the inputs and previous states,
    it computes the outputs and updates states.

    The formula used is as follows:

    .. math::
        i_{t} & = \sigma(W_{ii}x_{t} + b_{ii} + W_{hi}h_{t-1} + b_{hi})

        f_{t} & = \sigma(W_{if}x_{t} + b_{if} + W_{hf}h_{t-1} + b_{hf})

        o_{t} & = \sigma(W_{io}x_{t} + b_{io} + W_{ho}h_{t-1} + b_{ho})

        \widetilde{c}_{t} & = \tanh (W_{ig}x_{t} + b_{ig} + W_{hg}h_{t-1} + b_{hg})

        c_{t} & = f_{t} * c_{t-1} + i_{t} * \widetilde{c}_{t}

        h_{t} & = o_{t} * \tanh(c_{t})

        y_{t} & = h_{t}

    If `proj_size` is specified, the dimension of hidden state :math:`h_{t}` will be projected to `proj_size`:

    .. math::

        h_{t} = h_{t}W_{proj\_size}

    where :math:`\sigma` is the sigmoid function, and * is the elementwise
    multiplication operator.

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        weight_ih_attr(ParamAttr|None, optional): The parameter attribute for
            `weight_ih`. Default: None.
        weight_hh_attr(ParamAttr|None, optional): The parameter attribute for
            `weight_hh`. Default: None.
        bias_ih_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_ih`. Default: None.
        bias_hh_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_hh`. Default: None.
        proj_size (int, optional): If specified, the output hidden state
            will be projected to `proj_size`. `proj_size` must be smaller than
            `hidden_size`. Default: None.
        name (str|None, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Variables:
        - **weight_ih** (Parameter): shape (4 * hidden_size, input_size), input to hidden weight, which corresponds to the concatenation of :math:`W_{ii}, W_{if}, W_{ig}, W_{io}` in the formula.
        - **weight_hh** (Parameter): shape (4 * hidden_size, hidden_size), hidden to hidden weight, which corresponds to the concatenation of :math:`W_{hi}, W_{hf}, W_{hg}, W_{ho}` in the formula. If proj_size was specified, the shape will be (4 * hidden_size, proj_size).
        - **weight_ho** (Parameter, optional): shape (hidden_size, proj_size), project the hidden state.
        - **bias_ih** (Parameter): shape (4 * hidden_size, ), input to hidden bias, which corresponds to the concatenation of :math:`b_{ii}, b_{if}, b_{ig}, b_{io}` in the formula.
        - **bias_hh** (Parameter): shape (4 * hidden_size, ), hidden to hidden bias, which corresponds to the concatenation of :math:`b_{hi}, b_{hf}, b_{hg}, b_{ho}` in the formula.

    Inputs:
        - **inputs** (Tensor): shape `[batch_size, input_size]`, the input, corresponding to :math:`x_t` in the formula.
        - **states** (list|tuple, optional): a list/tuple of two tensors, each of shape `[batch_size, hidden_size]`, the previous hidden state, corresponding to :math:`h_{t-1}, c_{t-1}` in the formula. When states is None, zero state is used. Defaults to None.

    Returns:
        - **outputs** (Tensor). Shape `[batch_size, hidden_size]`, the output, corresponding to :math:`h_{t}` in the formula. If `proj_size` is specified, output shape will be `[batch_size, proj_size]`.
        - **states** (tuple). A tuple of two tensors, each of shape `[batch_size, hidden_size]`, the new hidden states, corresponding to :math:`h_{t}, c_{t}` in the formula.
            If `proj_size` is specified, shape of :math:`h_{t}` will be `[batch_size, proj_size]`.

    Notes:
        All the weights and bias are initialized with `Uniform(-std, std)` by
        default. Where std = :math:`\frac{1}{\sqrt{hidden\_size}}`. For more
        information about parameter initialization, please refer to :ref:`api_paddle_ParamAttr`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn((4, 16))
            >>> prev_h = paddle.randn((4, 32))
            >>> prev_c = paddle.randn((4, 32))

            >>> cell = paddle.nn.LSTMCell(16, 32)
            >>> y, (h, c) = cell(x, (prev_h, prev_c))

            >>> print(y.shape)
            [4, 32]
            >>> print(h.shape)
            [4, 32]
            >>> print(c.shape)
            [4, 32]

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        proj_size: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size of {self.__class__.__name__} must be greater than 0, but now equals to {hidden_size}"
            )
        if proj_size < 0:
            raise ValueError(
                f"proj_size of {self.__class__.__name__} must be greater than 0, but now equals to {hidden_size}"
            )

        if proj_size >= hidden_size:
            raise ValueError("proj_size must be smaller than hidden_size")

        std = 1.0 / math.sqrt(hidden_size)
        if weight_ih_attr is not False:
            self.weight_ih = self.create_parameter(
                (4 * hidden_size, input_size),
                weight_ih_attr,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.weight_ih = self.create_parameter(
                (4 * hidden_size, input_size),
                None,
                default_initializer=I.Constant(1.0),
            )
            self.weight_ih.stop_gradient = True
        if weight_hh_attr is not False:
            self.weight_hh = self.create_parameter(
                (4 * hidden_size, proj_size or hidden_size),
                weight_hh_attr,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.weight_hh = self.create_parameter(
                (4 * hidden_size, proj_size or hidden_size),
                None,
                default_initializer=I.Constant(1.0),
            )
            self.weight_hh.stop_gradient = True
        if bias_ih_attr is not False:
            self.bias_ih = self.create_parameter(
                (4 * hidden_size,),
                bias_ih_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.bias_ih = self.create_parameter(
                (4 * hidden_size,),
                None,
                is_bias=True,
                default_initializer=I.Constant(0.0),
            )
            self.bias_ih.stop_gradient = True
        if bias_hh_attr is not False:
            self.bias_hh = self.create_parameter(
                (4 * hidden_size,),
                bias_hh_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.bias_hh = self.create_parameter(
                (4 * hidden_size,),
                None,
                is_bias=True,
                default_initializer=I.Constant(0.0),
            )
            self.bias_hh.stop_gradient = True

        self.proj_size = proj_size
        if proj_size > 0:
            self.weight_ho = self.create_parameter(
                (hidden_size, proj_size),
                weight_hh_attr,
                default_initializer=I.Uniform(-std, std),
            )

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs: Tensor, states: Sequence[Tensor] | None = None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        pre_hidden, pre_cell = states
        gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        chunked_gates = paddle.split(gates, num_or_sections=4, axis=-1)

        i = self._gate_activation(chunked_gates[0])
        f = self._gate_activation(chunked_gates[1])
        o = self._gate_activation(chunked_gates[3])
        c = f * pre_cell + i * self._activation(chunked_gates[2])
        h = o * self._activation(c)
        if self.proj_size > 0:
            h = paddle.matmul(h, self.weight_ho)

        return h, (h, c)

    @property
    def state_shape(self) -> tuple[tuple[int], tuple[int]]:
        r"""
        The `state_shape` of LSTMCell is a tuple with two shapes:
        `((hidden_size, ), (hidden_size,))`. (-1 for batch size would be
        automatically inserted into shape). These two shapes correspond
        to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return ((self.hidden_size,), (self.proj_size or self.hidden_size,))

    def extra_repr(self) -> str:
        return '{input_size}, {hidden_size}'.format(**self.__dict__)


class GRUCell(RNNCellBase):
    r"""
    Gated Recurrent Unit (GRU) RNN cell. Given the inputs and previous states,
    it computes the outputs and updates states.

    The formula for GRU used is as follows:

    ..  math::

        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}h_{t-1} + b_{hr})

        z_{t} & = \sigma(W_{iz}x_{t} + b_{iz} + W_{hz}h_{t-1} + b_{hz})

        \widetilde{h}_{t} & = \tanh(W_{ic}x_{t} + b_{ic} + r_{t} * (W_{hc}h_{t-1} + b_{hc}))

        h_{t} & = z_{t} * h_{t-1} + (1 - z_{t}) * \widetilde{h}_{t}

        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid function, and * is the elementwise
    multiplication operator.

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size.
        hidden_size (int): The hidden size.
        weight_ih_attr(ParamAttr|None, optional): The parameter attribute for
            `weight_ih`. Default: None.
        weight_hh_attr(ParamAttr|None, optional): The parameter attribute for
            `weight_hh`. Default: None.
        bias_ih_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_ih`. Default: None.
        bias_hh_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_hh`. Default: None.
        name (str|None, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Variables:
        - **weight_ih** (Parameter): shape (3 * hidden_size, input_size), input to hidden weight, which corresponds to the concatenation of :math:`W_{ir}, W_{iz}, W_{ic}` in the formula.
        - **weight_hh** (Parameter): shape (3 * hidden_size, hidden_size), hidden to hidden weight, which corresponds to the concatenation of :math:`W_{hr}, W_{hz}, W_{hc}` in the formula.
        - **bias_ih** (Parameter): shape (3 * hidden_size, ), input to hidden bias, which corresponds to the concatenation of :math:`b_{ir}, b_{iz}, b_{ic}` in the formula.
        - **bias_hh** (Parameter): shape (3 * hidden_size, ), hidden to hidden bias, which corresponds to the concatenation of :math:`b_{hr}, b_{hz}, b_{hc}` in the formula.

    Inputs:
        - **inputs** (Tensor): A tensor with shape `[batch_size, input_size]`, corresponding to :math:`x_t` in the formula.
        - **states** (Tensor): A tensor with shape `[batch_size, hidden_size]`, corresponding to :math:`h_{t-1}` in the formula.

    Returns:
        - **outputs** (Tensor): shape `[batch_size, hidden_size]`, the output, corresponding to :math:`h_{t}` in the formula.
        - **states** (Tensor): shape `[batch_size, hidden_size]`, the new hidden state, corresponding to :math:`h_{t}` in the formula.

    Notes:
        All the weights and bias are initialized with `Uniform(-std, std)` by
        default. Where std = :math:`\frac{1}{\sqrt{hidden\_size}}`. For more
        information about parameter initialization, please refer to s:ref:`api_paddle_ParamAttr`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> x = paddle.randn((4, 16))
            >>> prev_h = paddle.randn((4, 32))

            >>> cell = paddle.nn.GRUCell(16, 32)
            >>> y, h = cell(x, prev_h)

            >>> print(y.shape)
            [4, 32]
            >>> print(h.shape)
            [4, 32]


    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size of {self.__class__.__name__} must be greater than 0, but now equals to {hidden_size}"
            )
        std = 1.0 / math.sqrt(hidden_size)
        if weight_ih_attr is not False:
            self.weight_ih = self.create_parameter(
                (3 * hidden_size, input_size),
                weight_ih_attr,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.weight_ih = self.create_parameter(
                (3 * hidden_size, input_size),
                None,
                default_initializer=I.Constant(1.0),
            )
            self.weight_ih.stop_gradient = True
        if weight_hh_attr is not False:
            self.weight_hh = self.create_parameter(
                (3 * hidden_size, hidden_size),
                weight_hh_attr,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.weight_hh = self.create_parameter(
                (3 * hidden_size, hidden_size),
                None,
                default_initializer=I.Constant(1.0),
            )
            self.weight_hh.stop_gradient = True

        if bias_ih_attr is not False:
            self.bias_ih = self.create_parameter(
                (3 * hidden_size,),
                bias_ih_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.bias_ih = self.create_parameter(
                (3 * hidden_size,),
                None,
                is_bias=True,
                default_initializer=I.Constant(0.0),
            )
            self.bias_ih.stop_gradient = True

        if bias_hh_attr is not False:
            self.bias_hh = self.create_parameter(
                (3 * hidden_size,),
                bias_hh_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std),
            )
        else:
            self.bias_hh = self.create_parameter(
                (3 * hidden_size,),
                None,
                is_bias=True,
                default_initializer=I.Constant(0.0),
            )
            self.bias_hh.stop_gradient = True

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(
        self, inputs: Tensor, states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)

        pre_hidden = states
        x_gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh

        x_r, x_z, x_c = paddle.split(x_gates, num_or_sections=3, axis=1)
        h_r, h_z, h_c = paddle.split(h_gates, num_or_sections=3, axis=1)

        r = self._gate_activation(x_r + h_r)
        z = self._gate_activation(x_z + h_z)
        c = self._activation(x_c + r * h_c)  # apply reset gate after mm
        h = (pre_hidden - c) * z + c

        return h, h

    @property
    def state_shape(self) -> tuple[int]:
        r"""
        The `state_shape` of GRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to the shape of :math:`h_{t-1}`.
        """
        return (self.hidden_size,)

    def extra_repr(self) -> str:
        return '{input_size}, {hidden_size}'.format(**self.__dict__)


class RNN(Layer):
    r"""
    Wrapper for RNN, which creates a recurrent neural network with an RNN cell.
    It performs :code:`cell.forward()` repeatedly until reaches to the maximum
    length of `inputs`.

    Parameters:
        cell(RNNCellBase): An instance of `RNNCellBase`.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Defaults to False.
        time_major (bool): Whether the first dimension of the input means the
            time steps. Defaults to False.

    Inputs:
        - **inputs** (Tensor): A (possibly nested structure of) tensor[s]. The input sequences. If time_major is False, the shape is `[batch_size, time_steps, input_size]`. If time_major is True, the shape is `[time_steps, batch_size, input_size]` where `input_size` is the input size of the cell.
        - **initial_states** (Tensor|list|tuple, optional): Tensor of a possibly nested structure of tensors, representing the initial state for the rnn cell. If not provided, `cell.get_initial_states` would be called to produce the initial states. Defaults to None.
        - **sequence_length** (Tensor, optional): shape `[batch_size]`, dtype: int64 or int32. The valid lengths of input sequences. Defaults to None.If `sequence_length` is not None, the inputs are treated as padded sequences. In each input sequence, elements whose time step index are not less than the valid length are treated as paddings.
        - **kwargs**: Additional keyword arguments to pass to `forward` of the cell.

    Outputs:
        - **outputs** (Tensor|list|tuple): the output sequences. If `time_major` is True, the shape is `[time_steps, batch_size, hidden_size]`, else `[batch_size, time_steps, hidden_size]`.
        - **final_states** (Tensor|list|tuple): final states of the cell. Tensor or a possibly nested structure of tensors which has the same structure with initial state. Each tensor in final states has the same shape and dtype as the corresponding tensor in initial states.

    Notes:
        This class is a low-level API for wrapping rnn cell into a RNN network.
        Users should take care of the state of the cell. If `initial_states` is
        passed to the `forward` method, make sure that it satisfies the
        requirements of the cell.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> inputs = paddle.rand((4, 23, 16))
            >>> prev_h = paddle.randn((4, 32))

            >>> cell = paddle.nn.SimpleRNNCell(16, 32)
            >>> rnn = paddle.nn.RNN(cell)
            >>> outputs, final_states = rnn(inputs, prev_h)

            >>> print(outputs.shape)
            [4, 23, 32]
            >>> print(final_states.shape)
            [4, 32]

    """

    def __init__(
        self,
        cell: RNNCellBase,
        is_reverse: bool = False,
        time_major: bool = False,
    ) -> None:
        super().__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            # for non-dygraph mode, `rnn` api uses cell.call
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major

    def forward(
        self,
        inputs: Tensor,
        initial_states: TensorOrTensors | None = None,
        sequence_length: Tensor = None,
        **kwargs: Any,
    ):
        final_outputs, final_states = rnn(
            self.cell,
            inputs,
            initial_states=initial_states,
            sequence_length=sequence_length,
            time_major=self.time_major,
            is_reverse=self.is_reverse,
            **kwargs,
        )
        return final_outputs, final_states


class BiRNN(Layer):
    r"""
    Wrapper for bidirectional RNN, which builds a bidirectional RNN given the
    forward rnn cell and backward rnn cell. A BiRNN applies forward RNN and
    backward RNN with corresponding cells separately and concats the outputs
    along the last axis.

    Parameters:
        cell_fw (RNNCellBase): A RNNCellBase instance used for forward RNN.
        cell_bw (RNNCellBase): A RNNCellBase instance used for backward RNN.
        time_major (bool, optional): Whether the first dimension of the input means the
            time steps. Defaults to False.

    Inputs:
        - **inputs** (Tensor): the input sequences of both RNN. If time_major is True, the shape of is `[time_steps, batch_size, input_size]`, else the shape is `[batch_size, time_steps, input_size]`, where input_size is the input size of both cells.
        - **initial_states** (list|tuple|None, optional): A tuple/list of the initial states of the forward cell and backward cell. Defaults to None. If not provided, `cell.get_initial_states` would be called to produce the initial states for each cell. Defaults to None.
        - **sequence_length** (Tensor|None, optional): shape `[batch_size]`, dtype: int64 or int32. The valid lengths of input sequences. Defaults to None. If `sequence_length` is not None, the inputs are treated as padded sequences. In each input sequence, elements whose time step index are not less than the valid length are treated as paddings.
        - **kwargs**: Additional keyword arguments. Arguments passed to `forward` for each cell.

    Outputs:
        - **outputs** (Tensor): the outputs of the bidirectional RNN. It is the concatenation of the outputs from the forward RNN and backward RNN along the last axis. If time_major is True, the shape is `[time_steps, batch_size, size]`, else the shape is `[batch_size, time_steps, size]`, where size is `cell_fw.hidden_size + cell_bw.hidden_size`.
        - **final_states** (tuple): A tuple of the final states of the forward cell and backward cell.

    Notes:
        This class is a low level API for wrapping rnn cells into a BiRNN
        network. Users should take care of the states of the cells.
        If `initial_states` is passed to the `forward` method, make sure that
        it satisfies the requirements of the cells.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> cell_fw = paddle.nn.LSTMCell(16, 32)
            >>> cell_bw = paddle.nn.LSTMCell(16, 32)
            >>> rnn = paddle.nn.BiRNN(cell_fw, cell_bw)

            >>> inputs = paddle.rand((2, 23, 16))
            >>> outputs, final_states = rnn(inputs)

            >>> print(outputs.shape)
            [2, 23, 64]
            >>> print(final_states[0][0].shape,len(final_states),len(final_states[0]))
            [2, 32] 2 2

    """

    def __init__(
        self,
        cell_fw: RNNCellBase,
        cell_bw: RNNCellBase,
        time_major: bool = False,
    ) -> None:
        super().__init__()
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        if cell_fw.input_size != cell_bw.input_size:
            raise ValueError(
                f"input size of forward cell({cell_fw.input_size}) does not equals"
                f"that of backward cell({cell_bw.input_size})"
            )
        for cell in [self.cell_fw, self.cell_bw]:
            if not hasattr(cell, "call"):
                # for non-dygraph mode, `rnn` api uses cell.call
                cell.call = cell.forward
        self.time_major = time_major

    def forward(
        self,
        inputs: Tensor,
        initial_states: tuple[Tensor, Tensor] | list[Tensor] | None = None,
        sequence_length: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if isinstance(initial_states, (list, tuple)):
            assert (
                len(initial_states) == 2
            ), "length of initial_states should be 2 when it is a list/tuple"

        outputs, final_states = birnn(
            self.cell_fw,
            self.cell_bw,
            inputs,
            initial_states,
            sequence_length,
            self.time_major,
            **kwargs,
        )
        return outputs, final_states


class RNNBase(LayerList):
    r"""
    RNNBase class for RNN networks. It provides `forward`, `flatten_parameters`
    and other common methods for SimpleRNN, LSTM and GRU.
    """

    def __init__(
        self,
        mode: _RNNType | str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        direction: _DirectionType | str = "forward",
        time_major: bool = False,
        dropout: float = 0.0,
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        proj_size: int = 0,
    ) -> None:
        super().__init__()
        bidirectional_list: list[str] = ["bidirectional", "bidirect"]
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction in bidirectional_list else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 2 if mode == "LSTM" else 1

        kwargs: dict[str, Any] = {
            "weight_ih_attr": weight_ih_attr,
            "weight_hh_attr": weight_hh_attr,
            "bias_ih_attr": bias_ih_attr,
            "bias_hh_attr": bias_hh_attr,
        }

        self.proj_size = proj_size
        if proj_size > 0:
            assert mode == 'LSTM'

        if mode == "LSTM":
            rnn_cls = LSTMCell
            kwargs["proj_size"] = proj_size
        elif mode == "GRU":
            rnn_cls = GRUCell
        elif mode == "RNN_RELU":
            rnn_cls = SimpleRNNCell
            kwargs["activation"] = 'relu'
        elif mode == "RNN_TANH":
            rnn_cls = SimpleRNNCell
            kwargs["activation"] = 'tanh'
        else:
            rnn_cls = SimpleRNNCell
            kwargs["activation"] = self.activation

        in_size = proj_size or hidden_size
        if direction in ["forward"]:
            is_reverse = False
            cell = rnn_cls(input_size, hidden_size, **kwargs)
            self.append(RNN(cell, is_reverse, time_major))
            for _ in range(1, num_layers):
                cell = rnn_cls(in_size, hidden_size, **kwargs)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction in bidirectional_list:
            cell_fw = rnn_cls(input_size, hidden_size, **kwargs)
            cell_bw = rnn_cls(input_size, hidden_size, **kwargs)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for _ in range(1, num_layers):
                cell_fw = rnn_cls(2 * in_size, hidden_size, **kwargs)
                cell_bw = rnn_cls(2 * in_size, hidden_size, **kwargs)
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward or bidirect (or bidirectional), "
                f"received direction = {direction}"
            )

        self.could_use_cudnn = True
        self.could_use_cudnn &= len(self.parameters()) == num_layers * 4 * (
            2 if direction in bidirectional_list else 1
        )

        # Expose params as RNN's attribute, which can make it compatible when
        # replacing small ops composed rnn with cpp rnn kernel.
        # Moreover, `jit.to_static` assumes params are added by current layer
        # and wouldn't include sublayer's params in current layer, which also
        # requires these params are added to current layer for `jit.save`.
        param_names = []
        for layer in range(self.num_layers):
            for num in range(self.num_directions):
                suffix = '_reverse' if num == 1 else ''
                param_names.extend(['weight_ih_l{}{}', 'weight_hh_l{}{}'])
                if bias_ih_attr is not False:
                    param_names.append('bias_ih_l{}{}')
                if bias_hh_attr is not False:
                    param_names.append('bias_hh_l{}{}')
                param_names = [x.format(layer, suffix) for x in param_names]
        for name, param in zip(param_names, self.parameters()):
            setattr(self, name, param)

        self.flatten_parameters()

    def flatten_parameters(self) -> None:
        """
        Resets parameter data pointer to address in continuous memory block for
        cudnn usage.
        """
        if self.could_use_cudnn:
            # layer.parameters() is depth first and ordered
            # for i in layer: for j in direct: w_ih, w_hh, b_ih, b_hh
            # need to reorganize to cudnn param layout:
            # all bias following all weights
            params = self.parameters(include_sublayers=False)
            shape = [np.prod(param.shape) for param in params]
            self._all_weights = [None] * len(params)
            for i, param in enumerate(params):
                offset = (
                    0
                    if i % 4 < 2
                    else (2 * self.num_layers * self.num_directions)
                )
                layer_idx = i // 4
                self._all_weights[offset + layer_idx * 2 + i % 2] = param
            # Wrap using a list to avoid registered into params and saving, maybe
            # need a better way to handle this later. Use `create_parameter` to
            # add both to main_program and startup_program for static-graph.
            # Use Constant initializer to avoid make effect on random generator.
            self._flat_weight = [
                self.create_parameter(
                    shape=[np.sum(shape)],
                    dtype=params[0].dtype,
                    default_initializer=I.Constant(0.0),
                )
            ]
            # dropout state may also can be hided and avoid saving
            # should dropout state be persistable for static-graph
            if in_pir_mode():
                self._dropout_state = paddle.pir.core.create_parameter(
                    dtype="uint8",
                    shape=[0],
                    name=f"dropout_state{NON_PERSISTABLE_VAR_NAME_SUFFIX}",
                    initializer=paddle.nn.initializer.Constant(value=0),
                    trainable=False,
                )
                self._dropout_state.stop_gradient = True
            else:
                self._dropout_state = self.create_variable(
                    dtype=core.VarDesc.VarType.UINT8,
                    name=f"dropout_state{NON_PERSISTABLE_VAR_NAME_SUFFIX}",
                )
            if in_dynamic_mode():
                with paddle.no_grad():
                    dtype = params[0].dtype
                    if isinstance(dtype, core.DataType):
                        dtype = paddle.base.framework.paddle_type_to_proto_type[
                            dtype
                        ]
                    _legacy_C_ops.coalesce_tensor(
                        self._all_weights,
                        self._all_weights,
                        self._flat_weight[0],
                        "copy_data",
                        True,
                        "use_align",
                        False,
                        "dtype",
                        dtype,
                    )
                    return
            # for static-graph, append coalesce_tensor into startup program
            with program_guard(
                default_startup_program(), default_startup_program()
            ):
                with paddle.no_grad():
                    if in_pir_mode():
                        _C_ops.coalesce_tensor(
                            self._all_weights,
                            params[0].dtype,
                            True,
                            False,
                            False,
                            0.0,
                            False,
                            -1,
                            -1,
                            [],
                            [],
                        )
                    else:
                        self._helper.append_op(
                            type="coalesce_tensor",
                            inputs={"Input": self._all_weights},
                            outputs={
                                "Output": self._all_weights,
                                "FusedOutput": self._flat_weight,
                            },
                            attrs={
                                "copy_data": True,
                                "use_align": False,
                                "dtype": params[0].dtype,
                            },
                        )

    def _cudnn_impl(
        self,
        inputs: Tensor,
        initial_states: TensorOrTensors | None,
        sequence_length: int | None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        if not self.time_major:
            inputs = paddle.tensor.transpose(inputs, [1, 0, 2])

        if in_dynamic_or_pir_mode():
            out, _, state = _C_ops.rnn(
                inputs,
                initial_states,
                self._all_weights,
                sequence_length,
                self._dropout_state,
                self.dropout,
                self.num_directions == 2,
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.mode,
                0,
                not self.training,
            )
        else:
            out = self._helper.create_variable_for_type_inference(inputs.dtype)
            state = [
                self._helper.create_variable_for_type_inference(inputs.dtype)
                for i in range(self.state_components)
            ]
            reserve = self._helper.create_variable_for_type_inference(
                dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
            )

            inputs = {
                'Input': inputs,
                'WeightList': self._all_weights,
                'PreState': initial_states,
                'SequenceLength': sequence_length,
            }
            attrs = {
                'dropout_prob': self.dropout,
                'is_bidirec': self.num_directions == 2,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'mode': self.mode,
                'is_test': not self.training,
            }

            outputs = {
                'Out': out,
                'State': state,
                'Reserve': reserve,
                'DropoutState': self._dropout_state,
            }
            self._helper.append_op(
                type="rnn", inputs=inputs, outputs=outputs, attrs=attrs
            )

        out = (
            paddle.tensor.transpose(out, [1, 0, 2])
            if not self.time_major
            else out
        )
        return out, tuple(state) if len(state) > 1 else state[0]

    def forward(
        self,
        inputs: Tensor,
        initial_states: TensorOrTensors | None = None,
        sequence_length: int | None = None,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        batch_index = 1 if self.time_major else 0
        dtype = inputs.dtype
        if initial_states is None:
            dims = ([self.proj_size or self.hidden_size], [self.hidden_size])
            fill_shape = [self.num_layers * self.num_directions, -1]
            if inputs.shape[batch_index] > 0:
                fill_shape[1] = inputs.shape[batch_index]
            else:
                fill_shape[1] = paddle.shape(inputs)[batch_index].item()
            initial_states = tuple(
                [
                    paddle.full(
                        shape=fill_shape + dims[i], fill_value=0, dtype=dtype
                    )
                    for i in range(self.state_components)
                ]
            )
        else:
            initial_states = (
                [initial_states]
                if isinstance(
                    initial_states, (paddle.static.Variable, paddle.pir.Value)
                )
                else initial_states
            )

        if self.could_use_cudnn and (
            not paddle.device.is_compiled_with_rocm() or sequence_length is None
        ):
            # Add CPU kernel and dispatch in backend later
            return self._cudnn_impl(inputs, initial_states, sequence_length)

        states = split_states(
            initial_states, self.num_directions == 2, self.state_components
        )
        final_states = []

        for i, rnn_layer in enumerate(self):
            if i > 0:
                inputs = F.dropout(
                    inputs,
                    self.dropout,
                    training=self.training,
                    mode="upscale_in_train",
                )
            outputs, final_state = rnn_layer(inputs, states[i], sequence_length)
            final_states.append(final_state)
            inputs = outputs

        final_states = concat_states(
            final_states, self.num_directions == 2, self.state_components
        )
        return outputs, final_states

    def extra_repr(self) -> str:
        main_str = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            main_str += ', num_layers={num_layers}'
        if self.time_major is not False:
            main_str += ', time_major={time_major}'
        if self.dropout != 0:
            main_str += ', dropout={dropout}'
        return main_str.format(**self.__dict__)


class SimpleRNN(RNNBase):
    r"""
    Multilayer Elman network(SimpleRNN). It takes input sequences and initial
    states as inputs, and returns the output sequences and the final states.

    Each layer inside the SimpleRNN maps the input sequences and initial states
    to the output sequences and final states in the following manner: at each
    step, it takes step inputs(:math:`x_{t}`) and previous
    states(:math:`h_{t-1}`) as inputs, and returns step outputs(:math:`y_{t}`)
    and new states(:math:`h_{t}`).

    .. math::

        h_{t} & = act(W_{ih}x_{t} + b_{ih} + W_{hh}h_{t-1} + b_{hh})

        y_{t} & = h_{t}

    where :math:`act` is for :attr:`activation`.

    Using key word arguments to construct is recommended.

    Parameters:
        input_size (int): The input size of :math:`x` for the first layer's cell.
        hidden_size (int): The hidden size of :math:`h` for each layer's cell.
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        direction (str, optional): The direction of the network. It can be "forward"
            or "bidirect"(or "bidirectional"). When "bidirect", the way to merge
            outputs of forward and backward is concatenating. Defaults to "forward".
        time_major (bool, optional): Whether the first dimension of the input
            means the time steps. If time_major is True, the shape of Tensor is
            [time_steps,batch_size,input_size], otherwise [batch_size, time_steps,input_size].
            Defaults to False. `time_steps` means the length of input sequence.
        dropout (float, optional): The dropout probability. Dropout is applied
            to the input of each layer except for the first layer. The range of
            dropout from 0 to 1. Defaults to 0.
        activation (str, optional): The activation in each SimpleRNN cell. It can be
            `tanh` or `relu`. Defaults to `tanh`.
        weight_ih_attr (ParamAttr|None, optional): The parameter attribute for
            `weight_ih` of each cell. Defaults to None.
        weight_hh_attr (ParamAttr|None, optional): The parameter attribute for
            `weight_hh` of each cell. Defaults to None.
        bias_ih_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_ih` of each cells. Defaults to None.
        bias_hh_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_hh` of each cells. Defaults to None.
        name (str|None, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Inputs:
        - **inputs** (Tensor): the input sequence. If `time_major` is True, the shape is `[time_steps, batch_size, input_size]`, else, the shape is `[batch_size, time_steps, input_size]`. `time_steps` means the length of the input sequence.
        - **initial_states** (Tensor, optional): the initial state. The shape is `[num_layers * num_directions, batch_size, hidden_size]`. If initial_state is not given, zero initial states are used.
        - **sequence_length** (Tensor, optional): shape `[batch_size]`, dtype: int64 or int32. The valid lengths of input sequences. Defaults to None. If `sequence_length` is not None, the inputs are treated as padded sequences. In each input sequence, elements whose time step index are not less than the valid length are treated as paddings.

    Returns:

        - **outputs** (Tensor): the output sequence. If `time_major` is True, the shape is `[time_steps, batch_size, num_directions * hidden_size]`, else, the shape is `[batch_size, time_steps, num_directions * hidden_size]`. Note that `num_directions` is 2 if direction is "bidirectional" else 1. `time_steps` means the length of the output sequence.

        - **final_states** (Tensor): final states. The shape is `[num_layers * num_directions, batch_size, hidden_size]`. Note that `num_directions` is 2 if direction is "bidirectional" (the index of forward states are 0, 2, 4, 6... and the index of backward states are 1, 3, 5, 7...), else 1.

    Variables:
        - **weight_ih_l[k]**: the learnable input-hidden weights of the k-th layer. If `k = 0`, the shape is `[hidden_size, input_size]`. Otherwise, the shape is `[hidden_size, num_directions * hidden_size]`.
        - **weight_hh_l[k]**: the learnable hidden-hidden weights of the k-th layer, with shape `[hidden_size, hidden_size]`.
        - **bias_ih_l[k]**: the learnable input-hidden bias of the k-th layer, with shape `[hidden_size]`.
        - **bias_hh_l[k]**: the learnable hidden-hidden bias of the k-th layer, with shape `[hidden_size]`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> rnn = paddle.nn.SimpleRNN(16, 32, 2)

            >>> x = paddle.randn((4, 23, 16))
            >>> prev_h = paddle.randn((2, 4, 32))
            >>> y, h = rnn(x, prev_h)

            >>> print(y.shape)
            [4, 23, 32]
            >>> print(h.shape)
            [2, 4, 32]


    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        direction: _DirectionType | str = "forward",
        time_major: bool = False,
        dropout: float = 0.0,
        activation: _ActivationType | str = "tanh",
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        name: str | None = None,
    ) -> None:
        if activation == "tanh":
            mode = "RNN_TANH"
        elif activation == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(f"Unknown activation '{activation}'")
        self.activation = activation
        super().__init__(
            mode,
            input_size,
            hidden_size,
            num_layers,
            direction,
            time_major,
            dropout,
            weight_ih_attr,
            weight_hh_attr,
            bias_ih_attr,
            bias_hh_attr,
            0,  # proj_size
        )


class LSTM(RNNBase):
    r"""
    Multilayer LSTM. It takes a sequence and an initial state as inputs, and
    returns the output sequences and the final states.

    Each layer inside the LSTM maps the input sequences and initial states
    to the output sequences and final states in the following manner: at each
    step, it takes step inputs(:math:`x_{t}`) and previous
    states(:math:`h_{t-1}, c_{t-1}`) as inputs, and returns step
    outputs(:math:`y_{t}`) and new states(:math:`h_{t}, c_{t}`).

    .. math::

        i_{t} & = \sigma(W_{ii}x_{t} + b_{ii} + W_{hi}h_{t-1} + b_{hi})

        f_{t} & = \sigma(W_{if}x_{t} + b_{if} + W_{hf}h_{t-1} + b_{hf})

        o_{t} & = \sigma(W_{io}x_{t} + b_{io} + W_{ho}h_{t-1} + b_{ho})

        \widetilde{c}_{t} & = \tanh (W_{ig}x_{t} + b_{ig} + W_{hg}h_{t-1} + b_{hg})

        c_{t} & = f_{t} * c_{t-1} + i_{t} * \widetilde{c}_{t}

        h_{t} & = o_{t} * \tanh(c_{t})

        y_{t} & = h_{t}

    If `proj_size` is specified, the dimension of hidden state :math:`h_{t}` will be projected to `proj_size`:

    .. math::

        h_{t} = h_{t}W_{proj\_size}

    where :math:`\sigma` is the sigmoid function, and * is the elementwise
    multiplication operator.

    Using key word arguments to construct is recommended.

    Parameters:
        input_size (int): The input size of :math:`x` for the first layer's cell.
        hidden_size (int): The hidden size of :math:`h` for each layer's cell.
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        direction (str, optional): The direction of the network. It can be "forward"
            or "bidirect"(or "bidirectional"). When "bidirect", the way to merge
            outputs of forward and backward is concatenating. Defaults to "forward".
        time_major (bool, optional): Whether the first dimension of the input
            means the time steps. If time_major is True, the shape of Tensor is
            [time_steps,batch_size,input_size], otherwise [batch_size, time_steps,input_size].
            Defaults to False. `time_steps` means the length of input sequence.
        dropout (float, optional): The dropout probability. Dropout is applied
            to the input of each layer except for the first layer. The range of
            dropout from 0 to 1. Defaults to 0.
        weight_ih_attr (ParamAttr|None, optional): The parameter attribute for
            `weight_ih` of each cell. Default: None.
        weight_hh_attr (ParamAttr|None, optional): The parameter attribute for
            `weight_hh` of each cell. Default: None.
        bias_ih_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_ih` of each cells. Default: None.
        bias_hh_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_hh` of each cells. Default: None.
        proj_size (int, optional): If specified, the output hidden state of each layer
            will be projected to `proj_size`. `proj_size` must be smaller than `hidden_size`.
            Default: 0.
        name (str|None, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Inputs:
        - **inputs** (Tensor): the input sequence. If `time_major` is True, the shape is `[time_steps, batch_size, input_size]`, else, the shape is `[batch_size, time_steps, input_size]`. `time_steps` means the length of the input sequence.
        - **initial_states** (list|tuple, optional): the initial state, a list/tuple of (h, c), the shape of each is `[num_layers * num_directions, batch_size, hidden_size]`. If initial_state is not given, zero initial states are used.
        - **sequence_length** (Tensor, optional): shape `[batch_size]`, dtype: int64 or int32. The valid lengths of input sequences. Defaults to None. If `sequence_length` is not None, the inputs are treated as padded sequences. In each input sequence, elements whose time step index are not less than the valid length are treated as paddings.

    Returns:

        - **outputs** (Tensor). The output sequence. If `time_major` is True, the shape is `[time_steps, batch_size, num_directions * hidden_size]`. If `proj_size` is specified, shape will be `[time_major, batch_size, num_directions * proj_size]`. If `time_major` is False, the shape is `[batch_size, time_steps, num_directions * hidden_size]`. Note that `num_directions` is 2 if direction is "bidirectional" else 1. `time_steps` means the length of the output sequence.
        - **final_states** (tuple). The final state, a tuple of two tensors, h and c. The shape of each is `[num_layers * num_directions, batch_size, hidden_size]`. If `proj_size` is specified, the last dimension of h will be proj_size.
            Note that `num_directions` is 2 if direction is "bidirectional" (the index of forward states are 0, 2, 4, 6... and the index of backward states are 1, 3, 5, 7...), else 1.

    Variables:
        - **weight_ih_l[k]**: the learnable input-hidden weights of the k-th layer. If `k = 0`, the shape is `[hidden_size, input_size]`. Otherwise, the shape is `[hidden_size, num_directions * hidden_size]`.
        - **weight_hh_l[k]**: the learnable hidden-hidden weights of the k-th layer, with shape `[hidden_size, hidden_size]`.
        - **bias_ih_l[k]**: the learnable input-hidden bias of the k-th layer, with shape `[hidden_size]`.
        - **bias_hh_l[k]**: the learnable hidden-hidden bias of the k-th layer, with shape `[hidden_size]`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> rnn = paddle.nn.LSTM(16, 32, 2)

            >>> x = paddle.randn((4, 23, 16))
            >>> prev_h = paddle.randn((2, 4, 32))
            >>> prev_c = paddle.randn((2, 4, 32))
            >>> y, (h, c) = rnn(x, (prev_h, prev_c))

            >>> print(y.shape)
            [4, 23, 32]
            >>> print(h.shape)
            [2, 4, 32]
            >>> print(c.shape)
            [2, 4, 32]


    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        direction: _DirectionType | str = "forward",
        time_major: bool = False,
        dropout: float = 0.0,
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        proj_size: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(
            "LSTM",
            input_size,
            hidden_size,
            num_layers,
            direction,
            time_major,
            dropout,
            weight_ih_attr,
            weight_hh_attr,
            bias_ih_attr,
            bias_hh_attr,
            proj_size,
        )


class GRU(RNNBase):
    r"""
    Multilayer GRU. It takes input sequence and initial states as inputs, and
    returns the output sequences and the final states.

    Each layer inside the GRU maps the input sequences and initial states
    to the output sequences and final states in the following manner: at each
    step, it takes step inputs(:math:`x_{t}`) and previous
    states(:math:`h_{t-1}`) as inputs, and returns step outputs(:math:`y_{t}`)
    and new states(:math:`h_{t}`).

    .. math::

        r_{t} & = \sigma(W_{ir}x_{t} + b_{ir} + W_{hr}h_{t-1} + b_{hr})

        z_{t} & = \sigma(W_{iz}x_{t} + b_{iz} + W_{hz}h_{t-1} + b_{hz})

        \widetilde{h}_{t} & = \tanh(W_{ic}x_{t} + b_{ic} + r_{t} * (W_{hc}h_{t-1} + b_{hc}))

        h_{t} & = z_{t} * h_{t-1} + (1 - z_{t}) * \widetilde{h}_{t}

        y_{t} & = h_{t}

    where :math:`\sigma` is the sigmoid function, and * is the elementwise
    multiplication operator.

    Using key word arguments to construct is recommended.

    Parameters:
        input_size (int): The input size of :math:`x` for the first layer's cell.
        hidden_size (int): The hidden size of :math:`h` for each layer's cell.
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.
        direction (str, optional): The direction of the network. It can be "forward"
            or "bidirect"(or "bidirectional"). When "bidirect", the way to merge
            outputs of forward and backward is concatenating. Defaults to "forward".
        time_major (bool, optional): Whether the first dimension of the input
            means the time steps. If time_major is True, the shape of Tensor is
            [time_steps,batch_size,input_size], otherwise [batch_size, time_steps,input_size].
            Defaults to False. `time_steps` means the length of input sequence.
        dropout (float, optional): The dropout probability. Dropout is applied
            to the input of each layer except for the first layer. The range of
            dropout from 0 to 1. Defaults to 0.
        weight_ih_attr (ParamAttr|None, optional): The parameter attribute for
            `weight_ih` of each cell. Default: None.
        weight_hh_attr (ParamAttr|None, optional): The parameter attribute for
            `weight_hh` of each cell. Default: None.
        bias_ih_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_ih` of each cells. Default: None.
        bias_hh_attr (ParamAttr|None, optional): The parameter attribute for the
            `bias_hh` of each cells. Default: None.
        name (str|None, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Inputs:
        - **inputs** (Tensor): the input sequence. If `time_major` is True, the shape is `[time_steps, batch_size, input_size]`, else, the shape is `[batch_size, time_steps, input_size]`. `time_steps` means the length of the input sequence.
        - **initial_states** (Tensor, optional): the initial state. The shape is `[num_layers * num_directions, batch_size, hidden_size]`. If initial_state is not given, zero initial states are used. Defaults to None.
        - **sequence_length** (Tensor, optional): shape `[batch_size]`, dtype: int64 or int32. The valid lengths of input sequences. Defaults to None. If `sequence_length` is not None, the inputs are treated as padded sequences. In each input sequence, elements whose time step index are not less than the valid length are treated as paddings.

    Returns:

        - **outputs** (Tensor): the output sequence. If `time_major` is True, the shape is `[time_steps, batch_size, num_directions * hidden_size]`, else, the shape is `[batch_size, time_steps, num_directions * hidden_size]`. Note that `num_directions` is 2 if direction is "bidirectional" else 1. `time_steps` means the length of the output sequence.

        - **final_states** (Tensor): final states. The shape is `[num_layers * num_directions, batch_size, hidden_size]`. Note that `num_directions` is 2 if direction is "bidirectional" (the index of forward states are 0, 2, 4, 6... and the index of backward states are 1, 3, 5, 7...), else 1.

    Variables:
        - **weight_ih_l[k]**: the learnable input-hidden weights of the k-th layer. If `k = 0`, the shape is `[hidden_size, input_size]`. Otherwise, the shape is `[hidden_size, num_directions * hidden_size]`.
        - **weight_hh_l[k]**: the learnable hidden-hidden weights of the k-th layer, with shape `[hidden_size, hidden_size]`.
        - **bias_ih_l[k]**: the learnable input-hidden bias of the k-th layer, with shape `[hidden_size]`.
        - **bias_hh_l[k]**: the learnable hidden-hidden bias of the k-th layer, with shape `[hidden_size]`.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> rnn = paddle.nn.GRU(16, 32, 2)

            >>> x = paddle.randn((4, 23, 16))
            >>> prev_h = paddle.randn((2, 4, 32))
            >>> y, h = rnn(x, prev_h)

            >>> print(y.shape)
            [4, 23, 32]
            >>> print(h.shape)
            [2, 4, 32]


    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        direction: _DirectionType | str = "forward",
        time_major: bool = False,
        dropout: float = 0.0,
        weight_ih_attr: ParamAttrLike | None = None,
        weight_hh_attr: ParamAttrLike | None = None,
        bias_ih_attr: ParamAttrLike | None = None,
        bias_hh_attr: ParamAttrLike | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            "GRU",
            input_size,
            hidden_size,
            num_layers,
            direction,
            time_major,
            dropout,
            weight_ih_attr,
            weight_hh_attr,
            bias_ih_attr,
            bias_hh_attr,
            0,  # proj_size
        )
