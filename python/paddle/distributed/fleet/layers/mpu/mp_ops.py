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

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.autograd import PyLayer
from paddle.base.data_feeder import check_dtype, check_variable_and_dtype
from paddle.distributed import collective
from paddle.framework import LayerHelper, _create_tensor, in_dynamic_mode
from paddle.nn import Layer
from paddle.nn.utils import dygraph_utils
from paddle.utils.inplace_utils import inplace_apis_in_dygraph_only

from ....communication.reduce import ReduceOp, _get_reduce_op


class c_identity_eager(PyLayer):
    @staticmethod
    def forward(ctx, tensor, group, skip_c_identity_dynamic):
        ctx.group = group
        if skip_c_identity_dynamic:
            return tensor
        else:
            return _legacy_C_ops.c_identity(
                tensor,
                'use_calc_stream',
                True,
                'ring_id',
                group.id,
                'use_model_parallel',
                True,
            )

    @staticmethod
    def backward(ctx, dy):
        op_type = _get_reduce_op(ReduceOp.SUM, "_c_identity")
        ctx.group.process_group.all_reduce_on_calc_stream(dy, op_type)
        return dy


class c_split_eager(PyLayer):
    @staticmethod
    def forward(ctx, tensor, group, rank, nranks):
        ctx.group = group
        ctx.nranks = nranks
        return _legacy_C_ops.c_split(
            tensor,
            'use_calc_stream',
            True,
            'ring_id',
            group.id,
            'rank',
            rank,
            'nranks',
            nranks,
            'use_model_parallel',
            True,
        )

    @staticmethod
    def backward(ctx, dy):
        group = ctx.group
        out_shape = dy.shape
        out_shape[0] = out_shape[0] * ctx.nranks
        out = paddle.empty(out_shape, dtype=dy.dtype)
        group.process_group.all_gather_into_tensor_on_calc_stream(
            out,
            dy,
        )
        return out


def _c_identity(tensor, group=None, skip_c_identity_dynamic=False):
    """
    Return a copy of the tensor, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if in_dynamic_mode():
        return c_identity_eager.apply(tensor, group, skip_c_identity_dynamic)
    else:
        op_type = 'c_identity'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

        check_variable_and_dtype(
            tensor,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            '_c_identity',
        )

        helper.append_op(
            type=op_type,
            inputs={'X': tensor},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': True,
                'use_model_parallel': True,
            },
        )
        return out


def _c_concat(tensor, group=None):
    """
    Return allgather of the tensor, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return
    group = collective._get_default_group() if group is None else group
    ring_id = group.id

    global_rank = collective._get_global_env().rank
    rank = group.rank
    nranks = group.nranks

    if in_dynamic_mode():
        return _legacy_C_ops.c_concat(
            tensor,
            'ring_id',
            ring_id,
            'use_calc_stream',
            True,
            'rank',
            rank,
            'nranks',
            nranks,
            'use_model_parallel',
            True,
        )
    else:
        op_type = 'c_concat'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

        check_variable_and_dtype(
            tensor,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            '_c_concat',
        )

        helper.append_op(
            type=op_type,
            inputs={'X': tensor},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': True,
                'use_model_parallel': True,
                'nranks': nranks,
                'rank': rank,
            },
        )
        return out


def _c_split(tensor, group=None):
    """
    Split tensor evenly among all members, mainly used with model parallel.

    Args:
        tensor (Tensor): The input Tensor. Its data type
            should be float16, float32, float64, int32 or int64.
        rank (int): The rank of the current process.
        group (int): The id of the process group to work on.

    Returns:
        Tensor.
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    global_rank = collective._get_global_env().rank
    rank = global_rank if group is None else group.get_group_rank(global_rank)
    nranks = (
        collective._get_global_env().world_size
        if group is None
        else group.nranks
    )

    if in_dynamic_mode():
        return c_split_eager.apply(tensor, group, rank, nranks)
    else:
        op_type = 'c_split'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

        check_variable_and_dtype(
            tensor,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            '_c_split',
        )

        helper.append_op(
            type=op_type,
            inputs={'X': tensor},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': True,
                'rank': rank,
                'nranks': nranks,
                'use_model_parallel': True,
            },
        )
        return out


class mp_allreduce_eager(PyLayer):
    @staticmethod
    def forward(
        ctx,
        tensor,
        group,
        use_calc_stream,
        use_model_parallel,
        op,
        skip_c_identity_dynamic,
    ):
        ctx.ring_id = group.id
        ctx.skip_c_identity_dynamic = skip_c_identity_dynamic

        if use_calc_stream:
            op_type = _get_reduce_op(op, "_mp_allreduce")
            group.process_group.all_reduce_on_calc_stream(tensor, op_type)
            return tensor
        else:
            return _legacy_C_ops.c_allreduce_sum_(
                tensor,
                'use_calc_stream',
                use_calc_stream,
                'ring_id',
                group.id,
            )

    @staticmethod
    def backward(ctx, dy):
        if ctx.skip_c_identity_dynamic:
            return dy
        else:
            return _legacy_C_ops.c_identity(
                dy,
                'use_calc_stream',
                True,
                'ring_id',
                ctx.ring_id,
                'use_model_parallel',
                True,
            )


def _mp_allreduce(
    tensor,
    op=ReduceOp.SUM,
    group=None,
    use_calc_stream=True,
    use_model_parallel=True,
    skip_c_identity_dynamic=False,
):
    """[it is same as allreduce above, but it supports model parallel. And it support inplace strategy]"""
    if group is not None and not group.is_member():
        return

    if in_dynamic_mode():
        group = collective._get_default_group() if group is None else group
        assert op == ReduceOp.SUM, f"Unknown parameter: {op}."
        return mp_allreduce_eager.apply(
            tensor,
            group,
            use_calc_stream,
            use_model_parallel,
            op,
            skip_c_identity_dynamic,
        )
    else:
        ring_id = 0 if group is None else group.id
        op_type = 'mp_allreduce_sum'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)

        check_variable_and_dtype(
            tensor,
            'tensor',
            ['float16', 'float32', 'float64', 'int32', 'int64', 'uint16'],
            op_type,
        )

        helper.append_op(
            type=op_type,
            inputs={'X': tensor},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': use_calc_stream,
            },
        )
        return out


def _c_lookup_table(table, index, start_index=0, vocab_size=-1, name=None):
    """
    Lookup table according to index.

    Args:
        table (Tensor): The input Tensor. Its data type
            should be float16, float32, float64.
        index (Tensor): The index to lookup table.
        start_index (int): The initial index for table range.
        name (string): The name of the api

    Returns:
        Tensor.
    """
    if in_dynamic_mode():
        return _C_ops.c_embedding(table, index, start_index, vocab_size)
    else:
        op_type = 'c_embedding'
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='table')
        check_variable_and_dtype(index, 'input', ['int32', 'int64'], op_type)
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='c_embedding',
            inputs={'Ids': index, 'W': table},
            outputs={'Out': tmp},
            attrs={"start_index": start_index, "vocab_size": vocab_size},
        )
        return tmp


class _Linear(Layer):
    """
    Linear
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.name = name

    def forward(self, input):
        out = _linear(
            x=input, weight=self.weight, bias=self.bias, name=self.name
        )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return f'in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, dtype={self._dtype}{name_str}'


def _c_softmax_with_cross_entropy(
    logits,
    label,
    group=None,
    return_softmax=False,
    ignore_index=-100,
):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    global_rank = collective._get_global_env().rank
    rank = global_rank if group is None else group.get_group_rank(global_rank)
    nranks = (
        collective._get_global_env().world_size
        if group is None
        else group.nranks
    )

    input_shape = list(logits.shape)
    label_shape = list(label.shape)
    input_dims = len(input_shape)
    label_dims = len(label_shape)
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f'Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got input_dims{input_dims}, label_dims{label_dims})'
        )
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=-1)
        label_shape = list(label.shape)
    if label_shape[-1] < 1 or label_shape[-1] > input_shape[-1] * nranks:
        raise ValueError(
            f'Expected label_shape[-1] >= 1 and label_shape[-1] <= input_shape[-1] * nranks\
             (got label_shape[-1] = {label_shape[-1]}, input_shape[-1] = {input_shape[-1]})'
        )

    if in_dynamic_mode():
        softmax, loss = _legacy_C_ops.c_softmax_with_cross_entropy(
            logits,
            label,
            'ring_id',
            ring_id,
            'rank',
            rank,
            'nranks',
            nranks,
            'ignore_index',
            ignore_index,
        )
        if not return_softmax:
            return loss
        else:
            return loss, softmax
    else:
        attrs = {
            'ring_id': ring_id,
            'rank': rank,
            'nranks': nranks,
            'ignore_index': ignore_index,
        }
        helper = LayerHelper('c_softmax_with_cross_entropy', **locals())
        softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
        loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
        helper.append_op(
            type='c_softmax_with_cross_entropy',
            inputs={'Logits': logits, 'Label': label},
            outputs={'Softmax': softmax, 'Loss': loss},
            attrs=attrs,
        )

        if return_softmax:
            return loss, softmax

        return loss


def _c_softmax_with_multi_label_cross_entropy(
    logits,
    label,
    smooth_weight,
    group=None,
    return_softmax=False,
    ignore_index=-100,
    sum_multi_label_loss=True,
):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    global_rank = collective._get_global_env().rank
    rank = global_rank if group is None else group.get_group_rank(global_rank)
    nranks = (
        collective._get_global_env().world_size
        if group is None
        else group.nranks
    )

    input_shape = list(logits.shape)
    label_shape = list(label.shape)
    input_dims = len(input_shape)
    label_dims = len(label_shape)
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f'Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got input_dims{input_dims}, label_dims{label_dims})'
        )
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=-1)
        label_shape = list(label.shape)
    if label_shape[-1] < 1 or label_shape[-1] > input_shape[-1] * nranks:
        raise ValueError(
            f'Expected label_shape[-1] >= 1 and label_shape[-1] <= input_shape[-1] * nranks\
             (got label_shape[-1] = {label_shape[-1]}, input_shape[-1] = {input_shape[-1]})'
        )

    if in_dynamic_mode():
        softmax, loss = _legacy_C_ops.c_softmax_with_multi_label_cross_entropy(
            logits,
            label,
            smooth_weight,
            'ring_id',
            ring_id,
            'rank',
            rank,
            'nranks',
            nranks,
            'ignore_index',
            ignore_index,
            'sum_multi_label_loss',
            sum_multi_label_loss,
        )
        if not return_softmax:
            return loss
        else:
            return loss, softmax
    else:
        attrs = {
            'ring_id': ring_id,
            'rank': rank,
            'nranks': nranks,
            'ignore_index': ignore_index,
            'sum_multi_label_loss': sum_multi_label_loss,
        }
        helper = LayerHelper(
            'c_softmax_with_multi_label_cross_entropy', **locals()
        )
        softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
        loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
        helper.append_op(
            type='c_softmax_with_multi_label_cross_entropy',
            inputs={
                'Logits': logits,
                'Label': label,
                'SmoothWeight': smooth_weight,
            },
            outputs={'Softmax': softmax, 'Loss': loss},
            attrs=attrs,
        )

        if return_softmax:
            return loss, softmax

        return loss


@inplace_apis_in_dygraph_only
def _c_softmax_with_cross_entropy_(
    logits,
    label,
    group=None,
    return_softmax=False,
    ignore_index=-100,
):
    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id
    global_rank = paddle.distributed.get_rank()
    rank = global_rank if group is None else group.get_group_rank(global_rank)
    nranks = (
        paddle.distributed.get_world_size() if group is None else group.nranks
    )

    input_dims = len(logits.shape)
    label_dims = len(label.shape)

    if input_dims not in [label_dims, label_dims + 1]:
        raise ValueError(
            f"Expected input dimensions to be {label_dims} or {label_dims + 1} "
            f"(got {input_dims} dimensions for input)"
        )

    if input_dims == label_dims + 1:
        label = paddle.unsqueeze(label, axis=-1)

    softmax, loss = _legacy_C_ops.c_softmax_with_cross_entropy_(
        logits,
        label,
        'ring_id',
        ring_id,
        'rank',
        rank,
        'nranks',
        nranks,
        'ignore_index',
        ignore_index,
    )

    return (loss, softmax) if return_softmax else loss


def _linear(x, weight, bias=None, name=None):
    """
    Function Linear
    """
    if in_dynamic_mode():
        pre_bias = _create_tensor(dtype=x.dtype)
        _legacy_C_ops.matmul(
            x,
            weight,
            pre_bias,
            'transpose_X',
            False,
            'transpose_Y',
            False,
            "alpha",
            1,
        )
        return dygraph_utils._append_bias_in_dygraph(
            pre_bias, bias, axis=len(x.shape) - 1
        )
    else:
        helper = LayerHelper('linear', **locals())
        dtype = x.dtype
        assert (
            len(x.shape) < 4
        ), "X latitude is not supported greater than 3 now."

        check_variable_and_dtype(
            x, 'x', ['float16', 'float32', 'float64'], 'linear'
        )
        check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'], 'linear')

        inputs = {'X': [x], 'Y': [weight]}
        attrs = {
            'transpose_X': False,
            'transpose_Y': False,
            'alpha': 1,
        }
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='matmul_v2', inputs=inputs, outputs={'Out': tmp}, attrs=attrs
        )
        if bias is not None:
            res = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [tmp], 'Y': [bias]},
                outputs={'Out': [res]},
                attrs={'axis': len(x.shape) - 1},
            )
        else:
            res = tmp
        return res


def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    # NOTE: use current_block and find_var_recursive to support while_loop
    startup_block = paddle.static.default_startup_program().current_block()
    main_block = paddle.static.default_main_program().current_block()
    startup_block._find_var_recursive(var.name).is_distributed = True
    main_block._find_var_recursive(var.name).is_distributed = True


def _parallel_linear(
    x,
    num_rows,
    num_cols,
    axis,
    param_attr,
    bias_attr,
    gather_out,
    inner_rank,
    nranks,
    split_tensor,
    name,
    group=None,
):
    """
    Parallel Linear

    axis the dimension of the parameter of linear layer.
    axis = 0: the row dimension
    axis = 1: the col dimension

    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if axis == 0:
        if split_tensor:
            x = _c_split(x, group=group)
    else:
        x = _c_identity(x, group=group)

    linear = paddle.nn.Linear(
        num_rows,
        num_cols,
        weight_attr=param_attr,
        bias_attr=bias_attr,
        name=name,
    )

    # NOTE: npu linear function use matmul_v2 but linear use matmul
    linear_function = paddle.nn.functional.linear
    linear_out = linear_function(
        x,
        linear.weight,
        # NOTE(wangxi): row split, bias need add after allreduce
        None if axis == 0 else linear.bias,
        linear.name,
    )

    _set_var_distributed(linear.weight)
    # set is_distributed for splited bias
    # if a linear layer is splited by row, each rank would hold a complete bias and they should be the same in each rank.
    # if a linear layer is splited by col, the bias would also be split into each rank as its weight
    if axis == 1 and linear._bias_attr is not False:
        _set_var_distributed(linear.bias)

    if not gather_out:
        return linear_out

    out_shape = list(linear_out.shape)
    out_shape[0] *= 1 if axis == 0 else nranks
    main_block = paddle.static.default_main_program().current_block()
    out = main_block.create_var(
        shape=out_shape,
        dtype=linear_out.dtype,
        type=linear_out.type,
        lod_level=linear_out.lod_level,
        persistable=False,
        is_data=False,
        need_check_feed=linear_out.desc.need_check_feed(),
    )
    if axis == 0:
        main_block.append_op(
            type='mp_allreduce_sum',
            inputs={'X': linear_out},
            outputs={'Out': out},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': True,
            },
        )
        if linear.bias is not None:
            out = out + linear.bias
    else:
        main_block.append_op(
            type='c_concat',
            inputs={'X': linear_out},
            outputs={'Out': out},
            attrs={
                'rank': inner_rank,
                'ring_id': ring_id,
                'nranks': nranks,
                'use_calc_stream': True,
                'use_model_parallel': True,
            },
        )
    return out


def _parallel_embedding(
    x,
    per_part_embeddings,
    origin_size,
    param_attr,
    inner_rank,
    num_partitions,
    name,
    group=None,
):
    """
    Parallel Embedding
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    helper = LayerHelper("_parallel_embedding", **locals())

    per_part_size = per_part_embeddings
    rank = inner_rank

    vocab_start_index = rank * per_part_size
    dtype = helper.get_default_dtype()
    size = [per_part_size, origin_size[1]]

    weight = helper.create_parameter(
        attr=param_attr, shape=size, dtype=dtype, is_bias=False
    )

    if num_partitions == 1:
        return paddle.nn.functional.embedding(
            x, weight=weight, padding_idx=None, sparse=False, name=name
        )

    startup_block = paddle.static.default_startup_program().global_block()
    main_block = paddle.static.default_main_program().global_block()
    startup_block.vars[weight.name].is_distributed = True
    main_block.vars[weight.name].is_distributed = True

    output_parallel = _c_lookup_table(
        weight,
        x,
        start_index=vocab_start_index,
        vocab_size=origin_size[0],
        name=name,
    )
    out = _mp_allreduce(
        output_parallel,
        group=group,
        use_calc_stream=True,
        use_model_parallel=True,
    )
    return out


def split(
    x,
    size,
    operation,
    axis=0,
    num_partitions=1,
    gather_out=True,
    weight_attr=None,
    bias_attr=None,
    name=None,
):
    """

    Split the weight of the specified operation into multiple devices
    and do the computation in parallel.

    Now the following three cases are supported.

    Case 1: Parallel Embedding
        The weight of the embedding operation is a NxM matrix with N rows and M columns.
        With parallel embedding, the weight is split into num_partitions partitions, each
        of which is a matrix with (N/num_partitions + 1) rows and M column where the last
        row as the padding idx.

        Suppose we split the NxM weight into two partitions on device_0 and device_1
        respectively. Then, one each device, the final weight has (N/2 + 1) rows with the
        index range from 0 to N/2. On device_0, all values in the input within [0, N/2 -1]
        keep unchanged and all other values are changed to N/2 which is the padding index and
        are mapped to all zeros after embedding. In the same way, on device_1, the value V in the
        input within [N/2, N-1] will be changed to (V - N/2), and all other values are changed
        to N/2 and are mapped to all zeros after embedding. Finally, the results on the two
        devices are sum-reduced.

        The Embedding put on single card is as shown below:

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/split_embedding_single.png
            :width: 800
            :height: 350
            :alt: single_embedding
            :align: center

        Parallel Embedding is shown as below:

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/split_embedding_split.png
            :width: 800
            :alt: split_embedding
            :align: center

    Case 2: Row Parallel Linear
        The weight of the linear operation is a NxM matrix with N rows and M columns.
        With row parallel linear, the weight is split into num_partitions partitions, each
        of which is a matrix with N/num_partitions rows and M column.

        The linear layer put on single card is shown as below, the input variable is represented by X,
        the weight matrix is represented by W and the output variable is O. The linear layer on single card is
        simple matrix multiplication operation, O = X * W.

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/split_single.png
            :width: 800
            :alt: single_linear
            :align: center

        Row Parallel Linear is shown as below. As the name suggests, Row Parallel Linear splits the weight matrix W into
        [[W_row1], [W_row2]] along the row. And accordingly the input is splitted along the column into [X_col1, X_col2] and multiply their
        respective weight matrices. Finally apply AllReduce on the output from each card to get the final output.

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/split_row.png
            :width: 800
            :alt: split_row
            :align: center

    Case 3: Column Parallel Linear
        The weight of the linear operation is a NxM matrix with N rows and M columns.
        With column parallel linear, the weight is split into num_partitions partitions, each
        of which is a matrix with N rows and M/num_partitions column.

        The linear layer put on single card has been illustrated on case 2 and Column Parallel Linear
        is shown as below. The Column Parallel Linear splits the weight matrix W into [W_col1, W_col2] along the column and
        these splitted matrices respectively multiply the input. Finally apply AllGather on the output from each card to get the final output.

        .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/split_col.png
            :width: 800
            :alt: split_col
            :align: center

    As observed, the column parallel linear and row parallel linear can be combined to skip one ALLGATHER communication
    operator. Furthermore the Attention and MLP can be combined to improve the performance as shown below.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/split_col_row.png
            :width: 800
            :alt: split_col_row
            :align: center

    Args:
        x (Tensor): Input tensor. It's data type should be float16, float32, float64, int32 or int64.
        size (list|tuple): A list or tuple with two elements indicating the shape of the weight.
        operation (str): The name of the operation. The supported operations are 'linear' and 'embedding'.
        axis (int, Optional): Indicate along which axis to split the weight. Default: 0.
        num_partitions (int, Optional): How many parts the weight is partitioned. Default: 1.
        gather_out (bool, Optional): Whether to gather the output after computation. By default, the output
            on each partitions will be gathered after computation. Default: True.
        weight_attr (ParamAttr, Optional): The parameter attribute for the learnable
            weights(Parameter) of the specified operation. Default: None.
        bias_attr (ParamAttr, Optional): The parameter attribute for the bias
            of the specified operation. Default: None.
        name (str, Optional): The default value is None. Normally there is no need for user to set this
            property. Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle
            >>> import paddle.distributed.fleet as fleet

            >>> paddle.enable_static()
            >>> paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            >>> fleet.init(is_collective=True)
            >>> data = paddle.randint(0, 8, shape=[10,4])
            >>> emb_out = paddle.distributed.split(
            ...     data,
            ...     (8, 8),
            ...     operation="embedding",
            ...     num_partitions=2)

    """
    assert isinstance(size, (list, tuple)), (
        "The type of size for "
        "paddle.distributed.split must be list or tuple."
    )
    assert len(size) == 2, (
        "Number of elements in size of " "paddle.distributed.split must be two."
    )
    assert isinstance(operation, str), (
        "The type of operation for " "paddle.distributed.split must be str."
    )
    supported_operations = [
        'linear',
        'embedding',
    ]
    assert operation in supported_operations, (
        "The operation for "
        f"paddle.distributed.split must be one of {supported_operations}."
    )
    if in_dynamic_mode():
        raise ValueError(
            "paddle.distributed.split cannot be used in dynamic "
            "graph mode, please use ParallelEmbedding, ParallelRowLinear, "
            "ParallelColumnLinear instead."
        )
    else:
        from paddle.distributed.fleet import fleet

        assert fleet._role_maker, (
            "To use paddle.distributed.split, "
            "you must call fleet.init() firstly."
        )
        rank = fleet.worker_index()
        nranks = fleet.worker_num()

    # rank within a model parallel group
    inner_rank = rank % num_partitions

    if operation == "embedding":
        assert axis == 0, (
            "We only support to split the weight of embedding "
            "along the first axis now."
        )
        assert size[0] % num_partitions == 0, (
            "The length of the vocabulary must be divisible by num_partitions "
            f"but received vocabulary={size[0]} num_partitions={num_partitions}"
        )

        per_part_size = size[0] // num_partitions
        emb_out = _parallel_embedding(
            x,
            per_part_size,
            size,
            weight_attr,
            inner_rank,
            num_partitions,
            name,
            group=None,
        )
        return emb_out
    else:
        should_split = False
        if axis == 0:
            assert size[0] % num_partitions == 0, (
                f"Number of rows of the weight for linear ({size[0]}) must be"
                f" divisible by num_partitions ({num_partitions})"
            )
            per_part_size = size[0] // num_partitions
            linear_size = (per_part_size, size[1])
            if x.shape[-1] == size[0]:
                should_split = True

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                f"Number of column of the weight for linear ({size[1]}) must be"
                f" divisible by num_partitions ({num_partitions})"
            )
            per_part_size = size[1] // num_partitions
            linear_size = (size[0], per_part_size)
        else:
            raise ValueError(
                "The value of axis must be 0 or 1, but the value "
                f"given is {axis}."
            )

        linear_out = _parallel_linear(
            x,
            linear_size[0],
            linear_size[1],
            axis,
            weight_attr,
            bias_attr,
            gather_out,
            inner_rank,
            num_partitions,
            should_split,
            name=name,
            group=None,
        )
        return linear_out
