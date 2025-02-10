# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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


import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    build_size_tensor,
    build_start_tensor,
    cast_tensor,
    fix_negative_indices,
    get_axes_for_reduce_op,
    get_input_constant_value,
    get_shape_tensor_element,
    has_dynamic_shape,
    resize_to_1d,
    trt_concat,
    trt_expand,
    trt_floor_div,
    trt_gather,
    trt_less,
    trt_max,
    trt_min,
    trt_prod,
    trt_reshape,
    trt_shape,
    trt_sub,
    trt_sum,
)
from paddle.tensorrt.register import converter_registry

from ..util import get_trt_version_list


@converter_registry.register("pd_op.reshape", trt_version="trt_version_ge=8.0")
def reshape_converter(network, paddle_op, inputs):
    x = inputs[0]
    is_constant_shape = False
    shape = get_input_constant_value(paddle_op, inputs, 1)
    if shape is not None:
        reshape_dim = shape
        is_constant_shape = True
    elif isinstance(inputs[1], list):
        # shape tensor is a list value
        shape_tensor = trt_concat(network, inputs[1])
    else:
        # shape tensor is a value
        shape_tensor = inputs[1]

    if not is_constant_shape:
        shape_tensor = resize_to_1d(network, shape_tensor)

    layer = network.add_shuffle(x)
    if is_constant_shape:
        layer.reshape_dims = reshape_dim
    else:
        layer.set_input(1, shape_tensor)

    assert len(layer.get_output(0).shape) >= 0, (
        'When convert reshape op to TRT reshape layer, the rank of trt reshape output dims is less than 0, '
        'you should modify trt_config(a TensorRTConfig object) and set trt_config.disable_ops = ["pd_op.reshape"] to forbid this op.'
    )

    return layer.get_output(0)


@converter_registry.register("pd_op.gather", trt_version="8.x")
def gather_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    index_tensor = inputs[1]
    axis_value = get_input_constant_value(paddle_op, inputs, 2)[0]
    axis_value = int(axis_value)

    reshape_layer = network.add_shuffle(index_tensor)
    reshape_layer.reshape_dims = (-1,)
    gather_layer = network.add_gather(
        input_tensor, reshape_layer.get_output(0), axis_value
    )
    return gather_layer.get_output(0)


@converter_registry.register("pd_op.gather_nd", trt_version="8.x")
def gather_nd_converter(network, paddle_op, inputs):
    input_tensor, indices_tensor = inputs
    non_zero_layer = network.add_gather_v2(
        input_tensor, indices_tensor, trt.GatherMode.ND
    )
    non_zero_layer.num_elementwise_dims = 0
    return non_zero_layer.get_output(0)


@converter_registry.register("pd_op.flatten", trt_version="trt_version_ge=8.0")
def flatten_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    input_val_shape = paddle_op.operands()[0].source().shape
    dims = len(input_val_shape)

    start_axis = paddle_op.attrs().get("start_axis")
    stop_axis = paddle_op.attrs().get("stop_axis")

    flatten_layer = network.add_shuffle(input_val)

    if not has_dynamic_shape(input_val_shape):
        if start_axis < 0:
            start_axis += dims + 1
        if stop_axis < 0:
            stop_axis += dims + 1

        flatten_dim = 1
        final_shape = []

        for i, s in enumerate(input_val_shape):
            if i >= start_axis and i <= stop_axis:
                flatten_dim *= s
            elif i == stop_axis + 1:
                final_shape.append(flatten_dim)
                final_shape.append(s)
            else:
                final_shape.append(s)

        if stop_axis == len(input_val.shape) - 1:
            final_shape.append(flatten_dim)

        flatten_layer.reshape_dims = tuple(final_shape)
    else:
        input_shape_layer = network.add_shape(input_val)
        input_shape_layer.name = f"{input_val.name}_origin_shape"

        final_shapes = []
        # Shapes before start_axis
        if start_axis > 0:
            prefix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(0,),
                shape=(start_axis,),
                stride=(1,),
            )
            prefix_shape_layer.name = f"{input_val.name}_prefix_shape"
            final_shapes.append(prefix_shape_layer.get_output(0))

        flatten_shape_layer = network.add_slice(
            input_shape_layer.get_output(0),
            start=(start_axis,),
            shape=(stop_axis - start_axis + 1,),
            stride=(1,),
        )
        flatten_shape_layer.name = f"{input_val.name}_need_flatten"
        flatten_shape_layer = network.add_reduce(
            flatten_shape_layer.get_output(0),
            trt.ReduceOperation.PROD,
            axes=get_axes_for_reduce_op(0, False),
            keep_dims=True,
        )
        flatten_shape_layer.name = f"{input_val.name}_flatten_dim"
        final_shapes.append(flatten_shape_layer.get_output(0))

        # Shapes after stop_axis
        if stop_axis < len(input_val_shape) - 1:
            suffix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(stop_axis + 1,),
                shape=(len(input_val_shape) - stop_axis - 1,),
                stride=(1,),
            )
            suffix_shape_layer.name = f"{input_val.name}_suffix_shape"
            final_shapes.append(suffix_shape_layer.get_output(0))

        final_shape_layer = network.add_concatenation(final_shapes)
        final_shape_layer.axis = 0
        final_shape_layer.name = f"{input_val.name}_final_shape"
        flatten_layer.set_input(1, final_shape_layer.get_output(0))

    return flatten_layer.get_output(0)


# In the converter, pd_op.concat has three inputs, because builtin.combine has two inputs.
@converter_registry.register("pd_op.concat", trt_version="trt_version_ge=8.0")
def concat_converter(network, paddle_op, inputs):
    input_tensors = inputs[0]
    axis_tensor = inputs[1]
    concat_layer = network.add_concatenation(inputs=input_tensors)

    axis = get_input_constant_value(paddle_op, inputs, 1)[0]
    axis = int(axis)
    if axis < 0:
        axis = len(input_tensors[0].shape) + axis
    concat_layer.axis = axis

    return concat_layer.get_output(0)


@converter_registry.register(
    "pd_op.unsqueeze", trt_version="trt_version_ge=8.0"
)
@converter_registry.register(
    "pd_op.unsqueeze_", trt_version="trt_version_ge=8.0"
)
def unsqueeze_converter(network, paddle_op, inputs):
    x = inputs[0]
    input_dims = x.shape
    axes = get_input_constant_value(paddle_op, inputs, 1)
    assert (
        len(axes) > 0
    ), f"axes size should be > 0 in when convert unsqueeze op in TensorRT, but received len(axes) = {len(axes)}."

    should_unsqueeze = [False] * (len(input_dims) + len(axes))
    cur_out_rank = len(input_dims)
    for i in range(len(axes)):
        cur_out_rank += 1
        if axes[i] < 0:
            axes[i] += cur_out_rank

        # axes[i] is relative to cur_out_rank
        # we make [axes[i], cur_out_rank - 2] shift right
        # and make (axes[i]) to true!
        for j in range(cur_out_rank - 1, axes[i], -1):
            should_unsqueeze[j] = should_unsqueeze[j - 1]
        if axes[i] >= cur_out_rank:
            should_unsqueeze[cur_out_rank - 1] = True
        else:
            should_unsqueeze[axes[i]] = True

    gather_indices = []
    in_rank_i = 0
    for i in range(len(should_unsqueeze)):
        if should_unsqueeze[i]:
            gather_indices.append(len(input_dims))
            continue
        gather_indices.append(in_rank_i)
        in_rank_i += 1

    layer = network.add_shuffle(x)
    shape_tensor = trt_shape(network, x)
    all_one = [1] * len(axes)
    all_one_tensor = add_1D_constant_layer(network, all_one)
    concat_inputs = [shape_tensor, all_one_tensor]
    real_shape_tensor = trt_gather(
        network, trt_concat(network, concat_inputs), gather_indices
    )
    layer.set_input(1, real_shape_tensor)
    return layer.get_output(0)


@converter_registry.register("pd_op.squeeze", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.squeeze_", trt_version="trt_version_ge=8.0")
def squeeze_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    input_shape = input_val.shape
    input_shape_size = len(input_shape)

    # If input is weights, convert to TensorRT tensor
    if isinstance(input_val, trt.Weights):
        input_val = network.add_constant(input_shape, input_val).get_output(0)

    # Get axis
    axis = get_input_constant_value(paddle_op, inputs, 1)
    if len(axis) == 0:
        for i in range(input_shape_size):
            if input_shape[i] == -1:
                raise RuntimeError(
                    "The necessary attributes of the squeeze operator axis is missing"
                )
            elif input_shape[i] == 1:
                axis.append(i)
    else:
        # Verify that each axis to squeeze has size 1
        for a in axis:
            if a < 0:
                a += input_shape_size
            if input_shape[a] != 1:
                raise RuntimeError(
                    f"Cannot squeeze dimension {a} with size {input_shape[a]}. Only dimensions with size 1 can be squeezed."
                )

    axes_size = len(axis)
    if axes_size == 0:
        raise RuntimeError(
            f"axis.size should be >0 in pd_op.squeeze op in TensorRT, but received {axes_size}"
        )
    # Mark which dimensions to squeeze
    should_squeeze = [False] * input_shape_size
    for a in axis:
        should_squeeze[a] = True

    # Get dimensions to keep
    gather_indices = [
        i for i, squeeze in enumerate(should_squeeze) if not squeeze
    ]

    # Add Shuffle layer
    layer = network.add_shuffle(input_val)
    shape_tensor = trt_shape(network, input_val)
    real_shape_tensor = trt_gather(network, shape_tensor, gather_indices)
    layer.set_input(1, real_shape_tensor)

    return layer.get_output(0)


@converter_registry.register("pd_op.expand", trt_version="trt_version_ge=8.0")
def expand_converter(network, paddle_op, inputs):
    input = inputs[0]
    input_dims = input.shape
    rank = len(input_dims)
    paddle_shape_tensor = paddle_op.operands()[1].source()

    shape = get_input_constant_value(paddle_op, inputs, 1)
    if shape is not None:
        shape_tensor = add_1D_constant_layer(network, shape)
        shape_rank = len(shape)
    elif paddle_shape_tensor.type().as_vec_type():
        shape_tensors = inputs[1]
        shape_rank = len(shape_tensors)
        shape_tensor = trt_concat(network, shape_tensors)
    else:
        shape_tensor = inputs[1]
        shape_rank = shape_tensor.shape[0]
    return trt_expand(network, input, rank, shape_tensor, shape_rank)


@converter_registry.register(
    "pd_op.expand_as", trt_version="trt_version_ge=8.0"
)
def expand_as_converter(network, paddle_op, inputs):
    input = inputs[0]
    input_dims = input.shape
    rank = len(input_dims)
    y = paddle_op.operands()[1].source()

    if y.initialized():
        y_t = inputs[1]
        shape_tensor = trt_shape(network, y_t)
        shape_rank = len(y_t.shape)
    else:
        shape = paddle_op.attrs().get("target_shape")
        shape_tensor = add_1D_constant_layer(network, shape)
        shape_rank = len(shape)
    return trt_expand(network, input, rank, shape_tensor, shape_rank)


@converter_registry.register("pd_op.cast", trt_version="8.x")
@converter_registry.register("pd_op.cast_", trt_version="8.x")
def cast_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    out_dtype = int(paddle_op.attrs().get("dtype"))
    # Reference paddle/phi/common/data_type.h enum DataType
    if out_dtype == 1:
        out_dtype = trt.bool
    elif out_dtype == 7:
        out_dtype = trt.int32
    elif out_dtype == 9:
        out_dtype = trt.int32
    elif out_dtype == 10:
        out_dtype = trt.float32
    elif out_dtype == 11:
        out_dtype = trt.float32
    elif out_dtype == 15:
        out_dtype = trt.float16
    else:
        raise RuntimeError(
            f"cast converter currently doesn't support dtype: {out_dtype}"
        )
    cast_layer = network.add_identity(input_tensor)
    cast_layer.set_output_type(0, out_dtype)
    cast_layer.get_output(0).dtype = out_dtype
    return cast_layer.get_output(0)


@converter_registry.register("pd_op.slice", trt_version="trt_version_ge=8.0")
def slice_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    axes = paddle_op.attrs()["axes"]
    decrease_axis = paddle_op.attrs().get("decrease_axis")

    input_shape_tensor = trt_shape(network, input_tensor)
    input_rank = len(input_tensor.shape)

    starts_tensor = []
    ends_tensor = []
    for i in range(input_rank):
        starts_tensor.append(add_1D_constant_layer(network, 0))
        ends_tensor.append(
            get_shape_tensor_element(network, input_shape_tensor, i)
        )

    starts = get_input_constant_value(paddle_op, inputs, 1)
    if starts is not None:
        assert len(starts) == len(
            axes
        ), f"The size of this starts: {len(starts)} must be equal to the axes: {len(axes)}."
        for idx in range(len(axes)):
            if starts[idx] < 0:
                starts_tensor[axes[idx]] = trt_max(
                    network,
                    trt_sum(
                        network,
                        add_1D_constant_layer(network, starts[idx]),
                        get_shape_tensor_element(
                            network, input_shape_tensor, axes[idx]
                        ),
                    ),
                    add_1D_constant_layer(network, 0),
                )
            else:
                starts_tensor[axes[idx]] = trt_min(
                    network,
                    add_1D_constant_layer(network, starts[idx]),
                    get_shape_tensor_element(
                        network, input_shape_tensor, axes[idx]
                    ),
                )
    else:
        starts = inputs[1]
        for idx in range(len(axes)):
            starts_tensor[axes[idx]] = get_shape_tensor_element(
                network, starts, idx
            )

    ends = get_input_constant_value(paddle_op, inputs, 2)
    if ends is not None:
        assert len(ends) == len(
            axes
        ), f"The size of this ends: {len(ends)} must be equal to the axes: {len(axes)}."
        for idx in range(len(axes)):
            if ends[idx] < 0:
                ends_tensor[axes[idx]] = trt_max(
                    network,
                    trt_sum(
                        network,
                        add_1D_constant_layer(network, ends[idx]),
                        get_shape_tensor_element(
                            network, input_shape_tensor, axes[idx]
                        ),
                    ),
                    add_1D_constant_layer(network, 0),
                )
            else:
                ends_tensor[axes[idx]] = trt_min(
                    network,
                    add_1D_constant_layer(network, ends[idx]),
                    get_shape_tensor_element(
                        network, input_shape_tensor, axes[idx]
                    ),
                )
    else:
        ends = inputs[2]
        for idx in range(len(axes)):
            ends_tensor[axes[idx]] = get_shape_tensor_element(
                network, ends, idx
            )

    start_tensor_layer = network.add_concatenation(starts_tensor)
    start_tensor_layer.axis = 0
    start_tensor = start_tensor_layer.get_output(0)
    end_tensor_layer = network.add_concatenation(ends_tensor)
    end_tensor_layer.axis = 0
    end_tensor = end_tensor_layer.get_output(0)
    size_tensor = trt_sub(network, end_tensor, start_tensor)

    # Create Slice layer
    slice_layer = network.add_slice(
        input_tensor, [0] * input_rank, [0] * input_rank, [1] * input_rank
    )
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, size_tensor)

    output_tensor = slice_layer.get_output(0)

    # Handle decrease_axis
    if len(decrease_axis) > 0:
        gather_indices = []
        for i in range(input_rank):
            if i in decrease_axis:
                continue
            gather_indices.append(i)

        if len(gather_indices) == 0:
            # 0-dim tensor situation and shuffle layer will make its shape (1,) -> ()
            shuffle_layer = network.add_shuffle(output_tensor)
            shuffle_layer.reshape_dims = ()
        else:
            real_size_tensor = trt_gather(network, size_tensor, gather_indices)
            shuffle_layer = network.add_shuffle(output_tensor)
            shuffle_layer.set_input(1, real_size_tensor)

        output_tensor = shuffle_layer.get_output(0)

    return output_tensor


@converter_registry.register("pd_op.split_with_num", trt_version="8.x")
def split_with_num_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape_size = len(input_tensor.shape)

    # Handle the case where axis is of type pir::Value
    axis_value = get_input_constant_value(paddle_op, inputs, 1)
    if axis_value is not None:
        axis_tensor = add_1D_constant_layer(network, axis_value)
    else:
        axis_tensor = inputs[1]
        axis_tensor = cast_tensor(network, axis_tensor, trt.int32)

    num_splits = paddle_op.attrs().get("num")
    num_splits_tensor = add_1D_constant_layer(network, num_splits)

    # Get the dynamic shape of the input tensor
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)

    # Handle negative axis index
    input_shape_size_tensor = add_1D_constant_layer(network, input_shape_size)
    zero_tensor = add_1D_constant_layer(network, 0)

    is_negative_axis = trt_less(network, axis_tensor, zero_tensor)
    is_negative_axis_int = cast_tensor(network, is_negative_axis, trt.int32)

    axis_adjustment = trt_prod(
        network, is_negative_axis_int, input_shape_size_tensor
    )

    axis_tensor = trt_sum(network, axis_tensor, axis_adjustment)

    # Get the size of the dimension specified by axis
    input_axis_size = network.add_gather(
        input_shape_tensor, axis_tensor, axis=0
    ).get_output(0)

    # Compute the size of each split
    split_size = trt_floor_div(network, input_axis_size, num_splits_tensor)

    outputs = []
    current_offset = add_1D_constant_layer(network, 0)

    for idx in range(num_splits):
        idx_tensor = add_1D_constant_layer(network, idx)
        # Calculate the slice start and size
        start_tensor = build_start_tensor(
            network, input_shape_size, axis_tensor, current_offset
        )
        size_tensor = build_size_tensor(
            network,
            input_shape_size,
            axis_tensor,
            split_size,
            input_shape_tensor,
        )

        # Create Slice layer
        slice_layer = network.add_slice(
            input_tensor,
            [0] * input_shape_size,
            [0] * input_shape_size,
            [1] * input_shape_size,
        )
        slice_layer.set_input(1, start_tensor)
        slice_layer.set_input(2, size_tensor)

        outputs.append(slice_layer.get_output(0))

        # Update current_offset for the next slice
        current_offset = trt_sum(network, current_offset, split_size)

    return outputs


@converter_registry.register("pd_op.split", trt_version="8.x")
def split_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape = input_tensor.shape
    input_shape_size = len(input_shape)

    axis_value = get_input_constant_value(paddle_op, inputs, 2)
    if axis_value is not None:
        axis_tensor = add_1D_constant_layer(network, axis_value)
    else:
        axis_tensor = inputs[2]
        axis_tensor = cast_tensor(network, axis_tensor, trt.int32)

    # Retrieve and process sections
    sections_value = get_input_constant_value(paddle_op, inputs, 1)
    if sections_value is not None:
        section_list = [int(s) for s in sections_value]
        dynamic_sections = False
    else:
        sections_tensor = inputs[1]
        dynamic_sections = True

    # Get the dynamic shape of the input tensor
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)

    # Handle negative axis index
    input_shape_size_tensor = add_1D_constant_layer(network, input_shape_size)
    zero_tensor = add_1D_constant_layer(network, 0)

    is_negative_axis = trt_less(network, axis_tensor, zero_tensor)
    is_negative_axis_int = cast_tensor(network, is_negative_axis, trt.int32)

    axis_adjustment = trt_prod(
        network, is_negative_axis_int, input_shape_size_tensor
    )
    axis_tensor = trt_sum(network, axis_tensor, axis_adjustment)

    # Initialize output list
    outputs = []
    offset = add_1D_constant_layer(network, 0)

    if not dynamic_sections:
        for section_size in section_list:
            section_size_tensor = add_1D_constant_layer(network, section_size)

            # Build start_tensor
            start_tensor = build_start_tensor(
                network, input_shape_size, axis_tensor, offset
            )

            # Build size_tensor
            size_tensor = build_size_tensor(
                network,
                input_shape_size,
                axis_tensor,
                section_size_tensor,
                input_shape_tensor,
            )
            # Create Slice layer
            slice_layer = network.add_slice(
                input_tensor,
                [0] * input_shape_size,
                [0] * input_shape_size,
                [1] * input_shape_size,
            )
            slice_layer.set_input(1, start_tensor)
            slice_layer.set_input(2, size_tensor)

            outputs.append(slice_layer.get_output(0))

            # Update offset
            offset = network.add_elementwise(
                offset, section_size_tensor, trt.ElementWiseOperation.SUM
            ).get_output(0)
    else:
        # If sections is a dynamic tensor
        num_sections = sections_tensor.shape[0]
        if num_sections == -1:
            raise NotImplementedError("dynamic sections not support")
        num_sections = int(num_sections)

        for idx in range(num_sections):
            idx_tensor = add_1D_constant_layer(network, idx)

            # Get section_size_tensor = sections_tensor[idx]
            section_size_tensor = network.add_gather(
                sections_tensor, idx_tensor, axis=0
            ).get_output(0)

            # Build start_tensor
            start_tensor = build_start_tensor(
                network, input_shape_size, axis_tensor, offset
            )

            # Build size_tensor
            size_tensor = build_size_tensor(
                network,
                input_shape_size,
                axis_tensor,
                section_size_tensor,
                input_shape_tensor,
            )

            # Create Slice layer
            slice_layer = network.add_slice(
                input_tensor,
                [0] * input_shape_size,
                [0] * input_shape_size,
                [1] * input_shape_size,
            )
            slice_layer.set_input(1, start_tensor)
            slice_layer.set_input(2, size_tensor)

            outputs.append(slice_layer.get_output(0))

            # Update offset
            offset = network.add_elementwise(
                offset, section_size_tensor, trt.ElementWiseOperation.SUM
            ).get_output(0)

    return outputs


@converter_registry.register("pd_op.stack", trt_version="8.x")
def stack_converter(network, paddle_op, inputs):
    input_tensors = inputs[0]
    input_num = len(input_tensors)

    inputs = []
    for i in range(input_num):
        inputs.append(input_tensors[i])

    input_rank = len(input_tensors[0].shape)

    output_rank = input_rank + 1
    axis = paddle_op.attrs().get("axis")
    if axis < 0:
        axis += output_rank

    shape_tensor = network.add_shape(input_tensors[0]).get_output(0)
    shape_tensor_vec = []
    for i in range(output_rank):
        if i < axis:
            shape_tensor_vec.append(
                get_shape_tensor_element(network, shape_tensor, i)
            )
        elif i > axis:
            shape_tensor_vec.append(
                get_shape_tensor_element(network, shape_tensor, i - 1)
            )
        else:
            shape_tensor_vec.append(add_1D_constant_layer(network, 1))

    after_shape_tensor = network.add_concatenation(shape_tensor_vec).get_output(
        0
    )

    for i in range(input_num):
        shuffle_layer = network.add_shuffle(inputs[i])
        shuffle_layer.set_input(1, after_shape_tensor)
        reshaped_tensor = shuffle_layer.get_output(0)
        inputs[i] = reshaped_tensor

    concat_layer = network.add_concatenation(inputs)
    concat_layer.axis = axis
    output_tensor = concat_layer.get_output(0)

    # Because we change tensor to 1-dim in 0-dim tensor situation when use trt,
    # so after stack, output will become 2-dim, if paddle output is a 1d tensor, we need reshape it.
    if (
        len(paddle_op.results()[0].shape) == 1
        and paddle_op.results()[0].shape[0] != -1
    ):
        output_tensor = resize_to_1d(network, output_tensor)
    return output_tensor


@converter_registry.register("pd_op.tile", trt_version="8.x")
def tile_converter(network, paddle_op, inputs):
    input = inputs[0]
    input_shape = input.shape
    input_shape_tensor = network.add_shape(input).get_output(0)
    rank = len(input_shape)

    repeat_times = get_input_constant_value(paddle_op, inputs, 1)
    if repeat_times is not None:
        repeat_tensor = add_1D_constant_layer(network, repeat_times)
        repeat_rank = len(repeat_times)
    else:
        repeat_tensor = inputs[1]
        repeat_tensor = resize_to_1d(network, repeat_tensor)
        repeat_shape = paddle_op.operands()[1].source().shape
        repeat_rank = repeat_shape[0]

    if rank > repeat_rank:
        one_rank_tensor = add_1D_constant_layer(
            network, [1] * (rank - repeat_rank)
        )
        repeat_expand_tensor = trt_concat(
            network, [one_rank_tensor, repeat_tensor]
        )
    elif rank < repeat_rank:
        one_rank_tensor = add_1D_constant_layer(
            network, [1] * (repeat_rank - rank)
        )
        input_shape_tensor = trt_concat(
            network, [one_rank_tensor, input_shape_tensor]
        )
        input = trt_reshape(network, input, input_shape_tensor, "", True)
        repeat_expand_tensor = repeat_tensor
    else:
        repeat_expand_tensor = repeat_tensor

    start = [0] * max(rank, repeat_rank)
    stride = [1] * max(rank, repeat_rank)
    output_shape = [0] * max(rank, repeat_rank)
    output_shape_tensor = trt_prod(
        network, input_shape_tensor, repeat_expand_tensor
    )

    slice_layer = network.add_slice(input, start, output_shape, stride)
    slice_layer.set_input(2, output_shape_tensor)

    version_list = get_trt_version_list()
    if version_list >= [8, 6, 0]:
        slice_layer.mode = trt.SampleMode.WRAP
    else:
        slice_layer.mode = trt.SliceMode.WRAP

    return slice_layer.get_output(0)


@converter_registry.register(
    "pd_op.take_along_axis", trt_version="trt_version_ge=8.2"
)
def take_along_axis_converter(network, paddle_op, inputs):
    axis = paddle_op.attrs().get("axis", 0)
    input_tensor = inputs[0]
    index_tensor = inputs[1]

    input_dims = input_tensor.shape
    if axis < 0:
        axis += len(input_dims)

    gather_layer = network.add_gather_v2(
        input_tensor, index_tensor, trt.GatherMode.ELEMENT
    )
    gather_layer.axis = axis

    output_tensor = gather_layer.get_output(0)

    return output_tensor


@converter_registry.register("pd_op.strided_slice", trt_version="8.x")
def strided_slice_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    axes = paddle_op.attrs()["axes"]
    starts = get_input_constant_value(paddle_op, inputs, 1)
    ends = get_input_constant_value(paddle_op, inputs, 2)
    strides = get_input_constant_value(paddle_op, inputs, 3)

    input_shape = input_tensor.shape
    nchw_input_dims = len(input_shape)

    trt_start_dims = [0] * nchw_input_dims
    trt_end_dims = [0] * nchw_input_dims
    trt_size_dims = [0] * nchw_input_dims
    trt_step_dims = [1] * nchw_input_dims

    has_neg_indices = False

    for i, trt_axis in enumerate(axes):
        trt_start_dims[trt_axis] = starts[i]
        trt_end_dims[trt_axis] = ends[i]
        trt_step_dims[trt_axis] = strides[i]
        if starts[i] < 0 or ends[i] < 0:
            has_neg_indices = True

    shape_tensor = trt_shape(network, input_tensor)
    start_tensor = add_1D_constant_layer(network, trt_start_dims)
    if has_neg_indices:
        start_tensor = fix_negative_indices(network, shape_tensor, start_tensor)

    end_vec_tensor = []
    for i in range(len(trt_end_dims)):
        end_vec_tensor.append(
            get_shape_tensor_element(network, shape_tensor, i)
        )

    for i, trt_axis in enumerate(axes):
        if ends[i] >= 0:
            end_vec_tensor[trt_axis] = add_1D_constant_layer(network, ends[i])
        else:
            end_vec_tensor[trt_axis] = trt_sum(
                network,
                end_vec_tensor[trt_axis],
                add_1D_constant_layer(network, ends[i]),
            )

    size_tensor = trt_sub(
        network,
        start_tensor,
        trt_min(network, trt_concat(network, end_vec_tensor), shape_tensor),
    )
    zero_t = add_1D_constant_layer(network, 0)
    step_tensor = add_1D_constant_layer(network, trt_step_dims)
    size_tensor = trt_sub(
        network, zero_t, trt_floor_div(network, size_tensor, step_tensor)
    )

    layer = network.add_slice(
        input_tensor, trt_start_dims, trt_size_dims, trt_step_dims
    )
    layer.set_input(1, start_tensor)
    layer.set_input(2, size_tensor)
    layer.set_input(3, step_tensor)
    return layer.get_output(0)


@converter_registry.register("pd_op.roll", trt_version="8.x")
def roll_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    axis = paddle_op.attrs()["axis"]

    shifts = get_input_constant_value(paddle_op, inputs, 1)
    if shifts is None:
        shifts = inputs[1]

    axis_size = len(axis)
    input_shape_tensor = trt_shape(network, input_tensor)

    for i in range(axis_size):
        axi = axis[i]
        if isinstance(shifts, trt.ITensor):
            shift = get_shape_tensor_element(network, shifts, i)
            input_shift = shift
        else:
            shift = shifts[i]
            input_shift = add_1D_constant_layer(network, shift)
        input_axis = get_shape_tensor_element(network, input_shape_tensor, axi)

        # 1.sub_value mod input_axis
        input1 = trt_sub(network, input_axis, input_shift)
        tmp_div_res = trt_floor_div(network, input1, input_axis)
        tmp_prod_res = trt_prod(network, tmp_div_res, input_axis)
        start = trt_sub(network, input1, tmp_prod_res)
        # 2.avoid start less than 0,start mod input_axis
        start = trt_sum(network, start, input_axis)
        tmp_div_res1 = trt_floor_div(network, start, input_axis)
        tmp_prod_res1 = trt_prod(network, tmp_div_res1, input_axis)
        start = trt_sub(network, start, tmp_prod_res1)
        zero_tensor = add_1D_constant_layer(network, 0)
        step = add_1D_constant_layer(network, 1)
        # 3.make index_tensor0
        sub_qutient = trt_sub(network, input_axis, start)
        quotient_tensor = trt_floor_div(network, sub_qutient, step)
        start1 = get_shape_tensor_element(network, start, 0, is_scalar=True)
        fill_layer0 = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
        fill_layer0.set_input(0, quotient_tensor)
        fill_layer0.set_input(1, start1)
        fill_layer0.set_input(2, step)
        index_tensor0 = fill_layer0.get_output(0)
        # 4.make index_tensor1
        sub_qutient_tensor = trt_sub(network, start, zero_tensor)
        quotient_tensor = trt_floor_div(network, sub_qutient_tensor, step)
        start2 = add_1D_constant_layer(network, 0, is_scalar=True)
        fill_layer1 = network.add_fill(shape=(), op=trt.FillOperation.LINSPACE)
        fill_layer1.set_input(0, quotient_tensor)
        fill_layer1.set_input(1, start2)
        fill_layer1.set_input(2, step)
        index_tensor1 = fill_layer1.get_output(0)
        itensors = [index_tensor0, index_tensor1]
        concat_input_tensor = trt_concat(network, itensors)
        if i == 0:
            layer = network.add_gather(
                input=input_tensor, indices=concat_input_tensor, axis=axi
            )
        else:
            layer = network.add_gather(
                input=layer.get_output(0), indices=concat_input_tensor, axis=axi
            )

    return layer.get_output(0)


@converter_registry.register("pd_op.pad", trt_version="8.x")
def pad_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    paddings = paddle_op.attrs()["paddings"]
    pad_size = len(paddings)
    pre_pad = [paddings[pad_size - 4], paddings[pad_size - 2]]
    post_pad = [paddings[pad_size - 3], paddings[pad_size - 1]]
    layer = network.add_padding_nd(input_tensor, pre_pad, post_pad)
    return layer.get_output(0)


@converter_registry.register("pd_op.numel", trt_version="8.x")
def numel_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    shape_tensor = network.add_shape(input_tensor).get_output(0)
    layer = network.add_reduce(
        shape_tensor, trt.ReduceOperation.PROD, axes=1, keep_dims=False
    )
    return layer.get_output(0)
