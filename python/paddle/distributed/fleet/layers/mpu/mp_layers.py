#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os

import paddle
from paddle.autograd import PyLayer
from paddle.base import core
from paddle.distributed import fleet
from paddle.nn import functional as F

from ....communication.reduce import ReduceOp, _get_reduce_op
from ...base import topology as tp
from ...utils.log_util import logger
from . import mp_ops
from .random import get_rng_state_tracker

__all__ = []

# Follow this paper to achieve the file:
# Shoeybi M, Patwary M, Puri R, et al. Megatron-lm: Training multi-billion parameter
# language models using model parallelism[J]. arXiv preprint arXiv:1909.08053, 2019. (https://arxiv.org/abs/1909.08053)


def is_fused_matmul_bias_supported():
    return hasattr(core.eager.ops.legacy, 'fused_gemm_epilogue')


def is_fused_linear_param_grad_add_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(paddle._C_ops, 'fused_linear_param_grad_add')
    else:
        return False


class VocabParallelEmbedding(paddle.nn.Layer):
    """Embedding mp parallelized in the vocabulary dimension.
    this class is used for splitting embedding in mp group.

    Args:
        num_embeddings(int): One element which indicate the size of the dictionary of embeddings.
        embedding_dim(int): One element which indicate the size of each embedding vector respectively.
        weight_attr(ParamAttr|None): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_paddle_nn_initializer_Assign`
            is used to load custom or pre-trained word vectors. See code example for details.
        mp_group(Group): The tensor parallel group.
        name(str, optional): For detailed information, please refer
               to :ref:`api_guide_Name`. Usually name is no need to set and
               None by default.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distributed import fleet

            >>> class SimpleMPNet(paddle.nn.Layer):
            ...     def __init__(self, vocab_size, hidden_size, inner_size, output_size):
            ...         super().__init__()
            ...         self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            ...             hidden_size,
            ...             inner_size,
            ...             gather_output=False,
            ...             has_bias=True)
            ...         self.linear2 = fleet.meta_parallel.RowParallelLinear(
            ...             inner_size,
            ...             hidden_size,
            ...             input_is_parallel=True,
            ...             has_bias=True)
            ...         self.linear3 = paddle.nn.Linear(hidden_size, output_size)
            ...         self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            ...                         vocab_size,
            ...                         hidden_size)
            ...     def forward(self, x):
            ...         x = self.embedding(x)
            ...         x = self.linear1(x)
            ...         x = self.linear2(x)
            ...         x = self.linear3(x)
            ...         return x

    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        weight_attr=None,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )

        self.origin_num_embeddings = num_embeddings
        self.is_mp = self.world_size > 1

        assert (
            num_embeddings % self.world_size == 0
        ), "The length of the vocabulary must be divisible by the parallelism degree of MP"

        per_part_size = num_embeddings // self.world_size

        self.vocab_start_index = self.rank * per_part_size
        self._dtype = self._helper.get_default_dtype()
        self._size = [per_part_size, embedding_dim]
        self._weight_attr = weight_attr
        self._name = name
        self.num_embeddings = num_embeddings

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    attr=self._weight_attr,
                    shape=self._size,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                attr=self._weight_attr,
                shape=self._size,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False
        if self.weight.is_distributed:
            self.weight.split_axis = 0

    def forward(self, x):
        if self.is_mp:
            output_parallel = mp_ops._c_lookup_table(
                self.weight,
                x,
                start_index=self.vocab_start_index,
                vocab_size=self.num_embeddings,
                name=self._name,
            )
            output = mp_ops._mp_allreduce(
                output_parallel,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
        else:
            output = F.embedding(
                x,
                weight=self.weight,
                padding_idx=None,
                sparse=False,
                name=self._name,
            )
        return output


_raise_cuda_env_unset_warning = True


class InnerOverlapLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        fuse_matmul_bias,
        mp_async_allreduce,
        mp_skip_c_identity,
        mp_fused_linear_param_grad_add,
        model_parallel_group,
    ):
        ctx.save_for_backward(x, weight, bias)
        ctx.model_parallel_group = model_parallel_group
        ctx.mp_fused_linear_param_grad_add = mp_fused_linear_param_grad_add
        if mp_skip_c_identity is False:
            x = paddle._legacy_C_ops.c_identity(
                x,
                'use_calc_stream',
                True,
                'ring_id',
                model_parallel_group.id,
                'use_model_parallel',
                True,
            )
        if not fuse_matmul_bias:
            return paddle._C_ops.linear(x, weight, bias)
        else:
            return paddle._legacy_C_ops.fused_gemm_epilogue(x, weight, bias)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensor()
        if dy.dtype == weight.dtype:
            dx = paddle.matmul(dy, weight, transpose_y=True)
        else:
            dx = paddle.matmul(
                dy, paddle.cast(weight, dtype=dy.dtype), transpose_y=True
            )
        op_type = _get_reduce_op(ReduceOp.SUM, "_c_identity")
        task = ctx.model_parallel_group.process_group.all_reduce(
            dx, op_type, sync_op=False
        )
        # Using small operation to preempt GPU SMs for all_reduce to achieve overlap.
        if int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0")) != 1:
            global _raise_cuda_env_unset_warning
            if _raise_cuda_env_unset_warning:
                logger.warning(
                    "You set mp_async_allreduce=True, but you forget to set environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance "
                    "loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance."
                )
            _raise_cuda_env_unset_warning = False
            tmp = paddle.ones([512])

        if ctx.mp_fused_linear_param_grad_add:
            if not is_fused_linear_param_grad_add_supported():
                raise NotImplementedError(
                    "You set mp_fused_linear_param_grad_add=True, "
                    "however, the paddle you are using not support this operation. "
                    "Please unset fused_linear_param_grad_add or use paddle compiled "
                    "with cuda 11.6 or higher."
                )

            if bias is None:
                if hasattr(weight, "main_grad"):
                    (
                        weight.main_grad,
                        _,
                    ) = paddle._C_ops.fused_linear_param_grad_add(
                        x, dy, weight.main_grad, None, True, False
                    )
                    task.wait()
                    return dx, None
                else:
                    if weight.grad is not None:
                        (
                            weight.grad,
                            _,
                        ) = paddle._C_ops.fused_linear_param_grad_add(
                            x, dy, weight.grad, None, False, False
                        )
                        task.wait()
                        return dx, None
                    else:
                        (
                            dw,
                            _,
                        ) = paddle._C_ops.fused_linear_param_grad_add(
                            x, dy, None, None, False, False
                        )
                        task.wait()
                        return dx, dw

            if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
                (
                    weight.main_grad,
                    bias.main_grad,
                ) = paddle._C_ops.fused_linear_param_grad_add(
                    x,
                    dy,
                    weight.main_grad,
                    bias.main_grad,
                    True,
                    True,
                )
                task.wait()
                return dx, None, None
            else:
                if weight.grad is not None:
                    assert bias.grad is not None
                    (
                        weight.grad,
                        bias.grad,
                    ) = paddle._C_ops.fused_linear_param_grad_add(
                        x, dy, weight.grad, bias.grad, False, True
                    )
                    task.wait()
                    return dx, None, None
                else:
                    # When main_grad is not enabled and gradient_accumulation is used, the grad is not initialized for the first acc step.
                    (
                        dw,
                        dbias,
                    ) = paddle._C_ops.fused_linear_param_grad_add(
                        x, dy, None, None, False, True
                    )
                    task.wait()
                    return dx, dw, dbias
        else:
            dy = dy.reshape([-1, dy.shape[-1]])
            dw = paddle.matmul(
                x.reshape([-1, x.shape[-1]]),
                dy,
                transpose_x=True,
            )
            if bias is None:
                task.wait()
                return dx, dw
            else:
                dbias = paddle.sum(dy, axis=0)
                task.wait()
                return dx, dw, dbias


class ColumnParallelLinear(paddle.nn.Layer):
    """Linear layer with mp parallelized(column).
    this class is used for splitting Linear Layer in mp group, column split the weight of the Linear layer.

    Args:
        in_features(int): The number of input units.
        out_features(int): The number of output units.
        weight_attr(ParamAttr|None): The attribute for the learnable weight of this layer. The default value is None
            and the weight will be initialized to zero. For detailed information, please refer to paddle.ParamAttr.
        has_bias(bool): whether to add bias.
        gather_output(bool): whether to do allgather for the output of each rank.
        fuse_matmul_bias(bool): whether to fuse matmul and bias.
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distributed import fleet

            >>> class SimpleMPNet(paddle.nn.Layer):
            ...     def __init__(self, vocab_size, hidden_size, inner_size, output_size):
            ...         super().__init__()
            ...         self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            ...             hidden_size,
            ...             inner_size,
            ...             gather_output=False,
            ...             has_bias=True)
            ...         self.linear2 = fleet.meta_parallel.RowParallelLinear(
            ...             inner_size,
            ...             hidden_size,
            ...             input_is_parallel=True,
            ...             has_bias=True)
            ...         self.linear3 = paddle.nn.Linear(hidden_size, output_size)
            ...         self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            ...                         vocab_size,
            ...                         hidden_size)
            ...     def forward(self, x):
            ...         x = self.embedding(x)
            ...         x = self.linear1(x)
            ...         x = self.linear2(x)
            ...         x = self.linear3(x)
            ...         return x
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=None,
        gather_output=True,
        fuse_matmul_bias=False,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self._name = name
        self.is_mp = self.world_size > 1

        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            f"Number of column of the weight for linear ({out_features}) must be"
            f" divisible by model parallel size ({self.world_size})"
        )
        self.output_size_per_partition = out_features // self.world_size

        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[in_features, self.output_size_per_partition],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[in_features, self.output_size_per_partition],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False

        if self.weight.is_distributed:
            self.weight.split_axis = 1

        if has_bias:
            # initialize bias to zero like Megatron
            self.bias = self.create_parameter(
                shape=[self.output_size_per_partition],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
            self.bias.is_distributed = True if self.is_mp else False
            if self.bias.is_distributed:
                self.bias.split_axis = 0
        else:
            self.bias = None

        self.linear = F.linear

        self.fuse_matmul_bias = fuse_matmul_bias

        mp_configs = fleet.fleet._user_defined_strategy.hybrid_configs[
            "mp_configs"
        ]
        self.mp_async_allreduce = self.is_mp and mp_configs.mp_async_allreduce
        self.mp_skip_c_identity = (
            self.is_mp
            and mp_configs.mp_async_allreduce
            and mp_configs.mp_skip_c_identity
        )
        self.mp_fused_linear_param_grad_add = (
            self.is_mp
            and mp_configs.mp_async_allreduce
            and mp_configs.mp_fused_linear_param_grad_add
        )
        if (
            self.mp_async_allreduce
            or self.mp_skip_c_identity
            or self.mp_fused_linear_param_grad_add
        ):
            assert (
                paddle.in_dynamic_mode()
            ), "mp_async_allreduce, mp_skip_c_identity and mp_fused_linear_param_grad_add are only available under dygraph mode"
        if self.fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in ColumnParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear

    def forward(self, x):
        # use inner api to process identity

        def _overlap_linear():
            return InnerOverlapLinear.apply(
                x,
                self.weight,
                self.bias,
                self.fuse_matmul_bias,
                self.mp_async_allreduce,
                self.mp_skip_c_identity,
                self.mp_fused_linear_param_grad_add,
                self.model_parallel_group,
            )

        if self.mp_async_allreduce:
            output_parallel = _overlap_linear()
        else:
            if self.is_mp:
                input_parallel = mp_ops._c_identity(
                    x,
                    group=self.model_parallel_group,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            else:
                input_parallel = x

            output_parallel = self.linear(
                input_parallel, self.weight, self.bias, name=self._name
            )

        if self.gather_output and self.is_mp:
            output = mp_ops._c_concat(
                output_parallel, group=self.model_parallel_group
            )
        else:
            output = output_parallel
        return output


class MPScale(PyLayer):
    @staticmethod
    def forward(ctx, x, mp_degree):
        out = paddle.scale(x, 1.0 / mp_degree)
        return out

    @staticmethod
    def backward(ctx, dout):
        return dout


class RowParallelLinear(paddle.nn.Layer):
    """Linear layer with mp parallelized(row).
    this class is used for splitting Linear Layer in mp group, row split the weight of the Linear layer.

    Args:
        in_features(int): The number of input units.
        out_features(int): The number of output units.
        weight_attr(ParamAttr|None): The attribute for the learnable weight of this layer. The default value is None
            and the weight will be initialized to zero. For detailed information, please refer to paddle.ParamAttr.
        has_bias(bool): whether to add bias.
        input_is_parallel(bool): whether the input has already been splitted across the mp group.
        fuse_matmul_bias(bool): whether to fuse matmul and bias.
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distributed import fleet

            >>> class SimpleMPNet(paddle.nn.Layer):
            ...     def __init__(self, vocab_size, hidden_size, inner_size, output_size):
            ...         super().__init__()
            ...         self.linear1 = fleet.meta_parallel.ColumnParallelLinear(
            ...             hidden_size,
            ...             inner_size,
            ...             gather_output=False,
            ...             has_bias=True)
            ...         self.linear2 = fleet.meta_parallel.RowParallelLinear(
            ...             inner_size,
            ...             hidden_size,
            ...             input_is_parallel=True,
            ...             has_bias=True)
            ...         self.linear3 = paddle.nn.Linear(hidden_size, output_size)
            ...         self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            ...                         vocab_size,
            ...                         hidden_size)
            ...     def forward(self, x):
            ...         x = self.embedding(x)
            ...         x = self.linear1(x)
            ...         x = self.linear2(x)
            ...         x = self.linear3(x)
            ...         return x

    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=True,
        input_is_parallel=False,
        fuse_matmul_bias=False,
        mp_group=None,
        name=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self._weight_attr = weight_attr
        self._dtype = self._helper.get_default_dtype()
        self._name = name

        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )

        self.is_mp = self.world_size > 1
        mp_configs = fleet.fleet._user_defined_strategy.hybrid_configs[
            "mp_configs"
        ]
        self.mp_async_allreduce = self.is_mp and mp_configs.mp_async_allreduce
        self.mp_skip_c_identity = (
            self.is_mp
            and mp_configs.mp_async_allreduce
            and mp_configs.mp_skip_c_identity
        )
        self.mp_fused_linear_param_grad_add = (
            self.is_mp
            and mp_configs.mp_async_allreduce
            and mp_configs.mp_fused_linear_param_grad_add
        )
        if (
            self.mp_async_allreduce
            or self.mp_skip_c_identity
            or self.mp_fused_linear_param_grad_add
        ):
            assert (
                paddle.in_dynamic_mode()
            ), "mp_async_allreduce, mp_skip_c_identity and mp_fused_linear_param_grad_add are only available under dygraph mode"
        assert in_features % self.world_size == 0, (
            f"Number of row of the weight for linear ({in_features}) must be"
            f" divisible by model parallel size ({self.world_size})"
        )

        self.input_size_per_partition = in_features // self.world_size

        if self.is_mp and paddle.in_dynamic_mode():
            with get_rng_state_tracker().rng_state():
                self.weight = self.create_parameter(
                    shape=[self.input_size_per_partition, self.out_features],
                    attr=self._weight_attr,
                    dtype=self._dtype,
                    is_bias=False,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.input_size_per_partition, self.out_features],
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )

        self.weight.is_distributed = True if self.is_mp else False
        if self.weight.is_distributed:
            self.weight.split_axis = 0

        if has_bias:
            self.bias = self.create_parameter(
                shape=[self.out_features],
                attr=paddle.nn.initializer.Constant(value=0.0),
                dtype=self._dtype,
                is_bias=True,
            )
        else:
            self.bias = None

        self.linear = F.linear

        if fuse_matmul_bias:
            if not is_fused_matmul_bias_supported():
                raise NotImplementedError(
                    "You set fuse_matmul_bias=True in RowParallelLinear, "
                    "however, the paddle you are using not support this operation. "
                    "Please set fuse_matmul_bias=False or use paddle compiled "
                    "with cuda 11.6 or higher."
                )
            from paddle.incubate.nn.functional import fused_linear

            self.linear = fused_linear
        self.fuse_matmul_bias = fuse_matmul_bias

    def forward(self, x):
        if self.input_is_parallel or (not self.is_mp):
            input_parallel = x
        else:
            # split last dim
            input_parallel = mp_ops._c_split(x, group=self.model_parallel_group)

        if self.is_mp:
            if self.fuse_matmul_bias:
                bias = MPScale.apply(self.bias, self.world_size)
                output_parallel = self.linear(
                    input_parallel, self.weight, bias, name=self._name
                )
                output = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
            else:
                output_parallel = self.linear(
                    input_parallel, self.weight, name=self._name
                )
                output_ = mp_ops._mp_allreduce(
                    output_parallel,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                    skip_c_identity_dynamic=self.mp_skip_c_identity,
                )
                output = (
                    output_ + self.bias if self.bias is not None else output_
                )
        else:
            output = self.linear(
                input_parallel, self.weight, self.bias, name=self._name
            )

        return output


class ParallelCrossEntropy(paddle.nn.Layer):
    """CrossEntropy with mp parallelized.
    this class is used for splitting softmax cross entropy in mp group.

    Args:
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .
        ignore_index (long int, optional):  Specifies a target value that is ignored and
            does not contribute to the loss. A negative value means that no label value
            needs to be ignored. Default is -100 .

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('No img to demonstrate')
            >>> from paddle.distributed.fleet.layers.mpu import ParallelCrossEntropy
            >>> loss_func = ParallelCrossEntropy
            >>> loss = loss_func(img, label)

    """

    def __init__(
        self, mp_group=None, name=None, ignore_index=-100, inplace=False
    ):
        super().__init__()
        self.name = name
        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )
        self.ignore_index = ignore_index
        self.inplace = inplace

    def forward(self, input, label):
        # Determine the appropriate function to use based on the 'inplace' flag.
        func = (
            mp_ops._c_softmax_with_cross_entropy_
            if self.inplace
            else mp_ops._c_softmax_with_cross_entropy
        )

        # Call the selected function with the common arguments.
        loss = func(
            input,
            label,
            group=self.model_parallel_group,
            ignore_index=self.ignore_index,
        )
        return loss


class ParallelMultiLabelCrossEntropy(paddle.nn.Layer):
    """CrossEntropy with mp parallelized.
    this class is used for splitting softmax cross entropy in mp group.

    Args:
        mp_group(Group): The tensor parallel group.
        name(str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .
        ignore_index (long int, optional):  Specifies a target value that is ignored and
            does not contribute to the loss. A negative value means that no label value
            needs to be ignored. Default is -100 .
        sum_multi_label_loss (bool, optional): Whether to sum the loss. Default is True .

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('No img to demonstrate')
            >>> from paddle.distributed.fleet.layers.mpu import ParallelMultiLabelCrossEntropy
            >>> loss_func = ParallelMultiLabelCrossEntropy()
            >>> loss = loss_func(img, label, smooth_weight)

    """

    def __init__(
        self,
        mp_group=None,
        name=None,
        ignore_index=-100,
        sum_multi_label_loss=True,
    ):
        super().__init__()
        self.name = name
        self.model_parallel_group = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_group()
            if mp_group is None
            else mp_group
        )
        self.world_size = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_world_size()
            if mp_group is None
            else mp_group.nranks
        )
        self.rank = (
            tp._HYBRID_PARALLEL_GROUP.get_model_parallel_rank()
            if mp_group is None
            else mp_group.rank
        )
        self.ignore_index = ignore_index
        self.sum_multi_label_loss = sum_multi_label_loss

    def forward(self, input, label, smooth_weight):
        loss = mp_ops._c_softmax_with_multi_label_cross_entropy(
            input,
            label,
            smooth_weight,
            group=self.model_parallel_group,
            ignore_index=self.ignore_index,
            sum_multi_label_loss=self.sum_multi_label_loss,
        )
        return loss
