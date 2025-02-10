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

import functools
import math
import operator
from typing import TYPE_CHECKING, Literal, overload

import paddle
from paddle import _C_ops, base, in_dynamic_mode
from paddle.static.nn.control_flow import Assert
from paddle.utils import deprecated

from ...base.data_feeder import check_type, check_variable_and_dtype
from ...base.framework import (
    _current_expected_place,
    core,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from ...base.layer_helper import LayerHelper
from ...common_ops_import import Variable
from ...tensor.manipulation import reshape

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable, TypeAlias

    from paddle import Tensor

    _ReduceMode: TypeAlias = Literal['mean', 'sum', 'none']
__all__ = []

kIgnoreIndex = -100


def dice_loss(
    input: Tensor,
    label: Tensor,
    epsilon: float = 1e-05,
    name: str | None = None,
) -> Tensor:
    r"""

    Dice loss for comparing the similarity between the input predictions and the label.
    This implementation is for binary classification, where the input is sigmoid
    predictions of each pixel, usually used for segmentation task. The dice loss can
    be defined as the following equation:

    .. math::

        dice\_loss &= 1 - \frac{2 * intersection\_area}{total\_area} \\
                  &= \frac{(total\_area - intersection\_area) - intersection\_area}{total\_area} \\
                  &= \frac{(union\_area - intersection\_area)}{total\_area}


    Parameters:
        input (Tensor): Tensor, rank>=2, shape is :math:`[N_1, N_2, ..., N_k, D]`, where :math:`N_1` is
                          the batch_size, :math:`D` is the number of categories. It is usually the output
                          predictions of sigmoid activation. The data type can be float32 or float64.
        label (Tensor): Tensor, the ground truth with the same rank as input, shape is :math:`[N_1, N_2, ..., N_k, 1]`.
                          where :math:`N_1` is the batch_size. The data type can be int32 or int64.
        epsilon (float): The epsilon will be added to the numerator and denominator.
                         If both input and label are empty, it makes sure dice is 1.
                         Default: 0.00001
        name(str|None, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        0-D Tensor, which shape is [], data type is the same as `input` .

    Example:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> x = paddle.randn((3,224,224,2))
            >>> label = paddle.randint(high=2, shape=(3,224,224,1))
            >>> predictions = F.softmax(x)
            >>> loss = F.dice_loss(input=predictions, label=label)
    """
    assert input.dtype in (paddle.float32, paddle.float64)
    assert label.dtype in (paddle.int32, paddle.int64)
    assert (
        len(input.shape) >= 2
    ), "The rank of input should be greater than or equal to 2."
    assert len(input.shape) == len(label.shape), (
        "The rank of input and label should be equal, "
        f"but received input: {len(input.shape)}, label: {len(label.shape)}."
    )
    assert label.shape[-1] == 1, (
        "The last dimension of label should be 1, "
        f"but received {label.shape[-1]}."
    )
    assert (
        input.shape[:-1] == label.shape[:-1]
    ), "All dimensions should be equal except the last one."
    assert (
        functools.reduce(operator.mul, input.shape) != 0
        and functools.reduce(operator.mul, label.shape) != 0
    ), "Any dimension of input and label cannot be equal to 0."

    label = paddle.squeeze(label, [-1])
    label = paddle.nn.functional.one_hot(label, input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    inse = paddle.sum(input * label, axis=reduce_dim)
    dice_denominator = paddle.sum(input, axis=reduce_dim) + paddle.sum(
        label, axis=reduce_dim
    )
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return paddle.mean(dice_score)


def log_loss(
    input: Tensor,
    label: Tensor,
    epsilon: float = 0.0001,
    name: str | None = None,
) -> Tensor:
    r"""

    **Negative Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    negative log loss.

    .. math::

        Out = -label * \log{(input + \epsilon)}
              - (1 - label) * \log{(1 - input + \epsilon)}

    Args:
        input (Tensor):  A 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator. Data type float32.
        label (Tensor):  The ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
                                Data type float32.
        epsilon (float, optional): A small number for numerical stability. Default 1e-4.
        name(str|None, optional): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Tensor, which shape is [N x 1], data type is float32.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> label = paddle.randn((10,1))
            >>> prob = paddle.randn((10,1))
            >>> cost = F.log_loss(input=prob, label=label)
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.log_loss(input, label, epsilon)

    helper = LayerHelper('log_loss', **locals())
    check_variable_and_dtype(input, 'input', ['float32'], 'log_loss')
    check_variable_and_dtype(label, 'label', ['float32'], 'log_loss')

    loss = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='log_loss',
        inputs={'Predicted': [input], 'Labels': [label]},
        outputs={'Loss': [loss]},
        attrs={'epsilon': epsilon},
    )
    return loss


def base_softmax_with_cross_entropy(
    logits: Tensor,
    label: Tensor,
    soft_label: bool = False,
    ignore_index: int = -100,
    numeric_stable_mode: bool = True,
    return_softmax: bool = False,
    axis: int = -1,
) -> Tensor:
    r"""

    This operator implements the cross entropy loss function with softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators
    expects mutually exclusive hard labels, each sample in a batch is in exactly
    one class with a probability of 1.0. Each sample in the batch will have a
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::
        \\loss_j=-\text{logits}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::
        \\loss_j= -\sum_{i=0}^{K}\text{label}_i\left(\text{logits}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right)\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::
        \\max_j&=\max_{i=0}^{K}{\text{logits}_i} \\
                log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logits_i - max_j)\\
                softmax_j &= \exp(logits_j - max_j - {log\_max\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Tensor): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`,
            Label is a ``Tensor``  in the same shape with :attr:`logits`.
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor``
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpret the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`.
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False`
                                              and GPU is used. When :attr:`soft_label`
                                              is :attr:`True` or CPU is used, the
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        - If `return_softmax` is False, return the cross entropy loss as a ``Tensor``.
          The dtype is the same as the input ``logits``. The shape is consistent with ``logits`` except in dimension :attr:`axis` as 1.
        - If `return_softmax` is True, return a tuple of two ``Tensor``: the cross entropy loss and the softmax result.
          The dtype of the cross entropy loss is the same as the input ``logits``, and the shape is consistent with ``logits``
          except in dimension :attr:`axis` as 1. The dtype and shape of the softmax result are the same as the input ``logits``.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> logits = paddle.to_tensor([0.4, 0.6, 0.9])
            >>> label = paddle.randint(high=2, shape=[1], dtype="int64")

            >>> out = paddle.nn.functional.softmax_with_cross_entropy(logits=logits, label=label)
            >>> print(out)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [1.15328646])
    """
    input_dims = len(list(logits.shape))
    if input_dims == 0:
        raise ValueError('The dimension of input should be larger than zero!')

    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f'Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got input_dims{input_dims}, label_dims{label_dims})'
        )
    if input_dims - 1 == label_dims:
        batch_size = label.shape[0]
        new_shape = [batch_size, logits.shape[1]]
        label = paddle.reshape(label, new_shape)

        # origin function
        # -------
        # label = paddle.unsqueeze(label, axis)
        # ------

        # notice : dim of label [-1,2,10] while logits is [-1,1] ,if use unsquezee
        # logits [-1,1]->[-1,1,1], but input need 1 != 2 so change
        # the modified function make logits [-1,1] -> [-1,2] (uncertain)
        # another possible modify [-1,1] -> [-1,2,1]
    if in_dynamic_or_pir_mode():
        softmax, loss = _C_ops.cross_entropy_with_softmax(
            logits,
            label,
            soft_label,
            True,
            numeric_stable_mode,
            ignore_index,
            axis,
        )
        if not return_softmax:
            return loss
        else:
            return loss, softmax
    else:
        attrs = {
            'soft_label': soft_label,
            'ignore_index': ignore_index,
            'numeric_stable_mode': numeric_stable_mode,
            'axis': axis,
        }
        helper = LayerHelper('softmax_with_cross_entropy', **locals())
        softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
        loss = helper.create_variable_for_type_inference(dtype=logits.dtype)

        outputs = {'Softmax': softmax, 'Loss': loss}
        helper.append_op(
            type='softmax_with_cross_entropy',
            inputs={'Logits': logits, 'Label': label},
            outputs=outputs,
            attrs=attrs,
        )

        if return_softmax:
            return loss, softmax

        return loss


def npair_loss(
    anchor: Tensor, positive: Tensor, labels: Tensor, l2_reg: float = 0.002
) -> Tensor:
    """

    Npair loss requires paired data. Npair loss has two parts: the first part is L2
    regularizer on the embedding vector; the second part is cross entropy loss which
    takes the similarity matrix of anchor and positive as logits.

    For more information, please refer to:
    `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_

    Args:
      anchor(Tensor): embedding vector for the anchor image. shape=[batch_size, embedding_dims],
                        the data type is float32 or float64.
      positive(Tensor): embedding vector for the positive image. shape=[batch_size, embedding_dims],
                        the data type is float32 or float64.
      labels(Tensor): 1-D tensor. shape=[batch_size], the data type is float32 or float64 or int64.
      l2_reg(float, optional): L2 regularization term on embedding vector, default: 0.002.


    Returns:
      A 0-D Tensor representing the npair loss, the data type is the same as anchor, the shape is [].

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from typing import Literal
            >>> paddle.seed(2023)
            >>> dtype: Literal["float32"] = "float32"

            >>> anchor = paddle.rand(shape=(18, 6), dtype=dtype)
            >>> positive = paddle.rand(shape=(18, 6), dtype=dtype)
            >>> labels = paddle.rand(shape=(18,), dtype=dtype)

            >>> npair_loss = paddle.nn.functional.npair_loss(anchor, positive, labels, l2_reg = 0.002)
            >>> print(npair_loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    2.94269347)

    """
    if in_dynamic_mode():
        if anchor.size == 0:
            raise ValueError("The dims of anchor should be greater than 0.")
        if positive.size == 0:
            raise ValueError("The dims of positive should be greater than 0.")
    check_variable_and_dtype(
        anchor, 'anchor', ['float32', 'float64'], 'npair_loss'
    )
    check_variable_and_dtype(
        positive, 'positive', ['float32', 'float64'], 'positive'
    )
    check_variable_and_dtype(
        labels, 'labels', ['float32', 'float64', 'int64'], 'labels'
    )
    Beta = 0.25
    batch_size = labels.shape[0]

    labels = paddle.reshape(labels, shape=[batch_size, 1])
    labels = paddle.tile(labels, repeat_times=[1, batch_size])

    labels = paddle.equal(labels, paddle.transpose(labels, perm=[1, 0])).astype(
        'float32'
    )
    labels = labels / paddle.sum(labels, axis=1, keepdim=True)

    l2loss = paddle.mean(paddle.sum(paddle.square(anchor), 1)) + paddle.mean(
        paddle.sum(paddle.square(positive), 1)
    )
    l2loss = l2loss * Beta * l2_reg

    similarity_matrix = paddle.matmul(
        anchor, positive, transpose_x=False, transpose_y=True
    )
    softmax_ce = base_softmax_with_cross_entropy(
        logits=similarity_matrix, label=labels, soft_label=True
    )
    cross_entropy = paddle.sum(labels * softmax_ce, 0)
    celoss = paddle.mean(cross_entropy)

    return l2loss + celoss


def square_error_cost(input: Tensor, label: Tensor) -> Tensor:
    r"""

    This op accepts input predictions and target label and returns the
    squared error cost.

    For predictions label, and target label, the equation is:

    .. math::

        Out = (input - label)^2

    Parameters:
        input (Tensor): Input tensor, the data type should be float32.
        label (Tensor): Label tensor, the data type should be float32.

    Returns:
        Tensor, The tensor storing the element-wise squared error
        difference between input and label.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> input = paddle.to_tensor([1.1, 1.9])
            >>> label = paddle.to_tensor([1.0, 2.0])
            >>> output = paddle.nn.functional.square_error_cost(input, label)
            >>> print(output)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.01000000, 0.01000000])

    """
    if in_dynamic_or_pir_mode():
        minus_out = _C_ops.subtract(input, label)
        square_out = _C_ops.square(minus_out)
        return square_out
    else:
        check_variable_and_dtype(
            input, "input", ['float32', 'float64'], 'square_error_cost'
        )
        check_variable_and_dtype(
            label, "label", ['float32', 'float64'], 'square_error_cost'
        )
        helper = LayerHelper('square_error_cost', **locals())
        minus_out = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type='elementwise_sub',
            inputs={'X': [input], 'Y': [label]},
            outputs={'Out': [minus_out]},
        )

        square_out = helper.create_variable_for_type_inference(
            dtype=input.dtype
        )
        helper.append_op(
            type='square',
            inputs={'X': [minus_out]},
            outputs={'Out': [square_out]},
        )
        return square_out


def edit_distance(
    input: Tensor,
    label: Tensor,
    normalized: bool = True,
    ignored_tokens: Sequence[int] | None = None,
    input_length: Tensor | None = None,
    label_length: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    This op computes the edit distances, also called Levenshtein distance, between a batch of
    hypothesis strings and their references. It measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into another.
    The operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    So the edit distance between A and B is 3.

    The input is a Tensor, the input_length and label_length should be supported.

    The `batch_size` of labels should be same as `input`.

    The output include the edit distance value between every pair of input and related label, and the number of sequence.
    If Attr(normalized) is true,
    the edit distance value will be divided by the length of label.

    Parameters:
        input(Tensor): The input tensor, its rank should be equal to 2 and its data type should be int64.
        label(Tensor): The label tensor, its rank should be equal to 2 and its data type should be int64.
        normalized(bool, optional): Indicated whether to normalize the edit distance. Default: True.
        ignored_tokens(list|tuple|None, optional): Tokens that will be removed before calculating edit distance. Default: None.
        input_length(Tensor|None, optional): The length for each sequence in `input` if it's of Tensor type, it should have shape `(batch_size, )` and its data type should be int64.
        label_length(Tensor|None, optional): The length for each sequence in `label` if it's of Tensor type, it should have shape `(batch_size, )` and its data type should be int64.
        NOTE: To be avoid unexpected result, the value of every elements in input_length and label_length should be equal to the value of the second dimension of input and label. For example, The input: [[1,2,3,4],[5,6,7,8],[9,10,11,12]], the shape of input is [3,4] and the input_length should be [4,4,4]

    Returns:
        Tuple:
            distance(Tensor): edit distance result, its data type is float32, and its shape is (batch_size, 1).
            sequence_num(Tensor): sequence number, its data type is float32, and its shape is (1,).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input = paddle.to_tensor([[1,2,3],[4,5,6],[4,4,4],[1,1,1]], dtype='int64')
            >>> label = paddle.to_tensor([[1,3,4,1],[4,5,8,1],[7,7,7,1],[1,1,1,1]], dtype='int64')
            >>> input_len = paddle.to_tensor([3,3,3,3], dtype='int64')
            >>> label_len = paddle.to_tensor([4,4,4,4], dtype='int64')

            >>> distance, sequence_num = F.loss.edit_distance(input=input, label=label, input_length=input_len, label_length=label_len, normalized=False)
            >>> print(distance)
            Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
                    [4])
            >>> print(sequence_num)
            Tensor(shape=[4, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[3.],
                     [2.],
                     [4.],
                     [1.]])

            >>> distance, sequence_num = F.loss.edit_distance(input=input, label=label, input_length=input_len, label_length=label_len, normalized=True)
            >>> print(distance)
            Tensor(shape=[1], dtype=int64, place=Place(cpu), stop_gradient=True,
                    [4])
            >>> print(sequence_num)
            Tensor(shape=[4, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[0.75000000],
                     [0.50000000],
                     [1.        ],
                     [0.25000000]])

    """

    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        raise ValueError(
            f'Expected ignored_tokens is None (got {ignored_tokens})'
        )

    if in_dynamic_mode():
        return _C_ops.edit_distance(
            input, label, input_length, label_length, normalized
        )

    check_variable_and_dtype(input, 'input', ['int64'], 'edit_distance')
    check_variable_and_dtype(label, 'label', ['int64'], 'edit_distance')
    this_inputs = {"Hyps": [input], "Refs": [label]}
    if input_length is not None and label_length is not None:
        this_inputs['HypsLength'] = [input_length]
        this_inputs['RefsLength'] = [label_length]

    # edit distance op
    edit_distance_out = helper.create_variable_for_type_inference(dtype="int64")
    sequence_num = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs=this_inputs,
        outputs={"Out": [edit_distance_out], "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized},
    )

    return edit_distance_out, sequence_num


def binary_cross_entropy(
    input: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    """
    Measure the binary_cross_entropy loss between input predictions ``input``
    and target labels ``label`` . The binary_cross_entropy loss can be described as:

    If :attr:`weight` is set, the loss is:

    .. math::
        Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`weight` is None, the loss is:

    .. math::
        Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`reduction` set to ``'none'``, the interface will return the original loss `Out`.

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(Out)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(Out)

    Note that the input predictions ``input`` always be the output of sigmoid, and the target labels ``label``
    should be numbers between 0 and 1.

    Parameters:
        input (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``input``
            should always be the output of sigmod.  Available dtype is float16, float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``input``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float16, float32, float64.
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, has to be a Tensor of size nbatch and the data type
            is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor. If ``reduction`` is ``'none'``, the shape of output is
            same as ``input`` , else the shape of output is scalar.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([0.5, 0.6, 0.7], 'float32')
            >>> label = paddle.to_tensor([1.0, 0.0, 1.0], 'float32')
            >>> output = paddle.nn.functional.binary_cross_entropy(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.65537095)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in binary_cross_entropy should be 'sum', "
            f"'mean' or 'none', but received {reduction}, which is not allowed."
        )

    if in_dynamic_or_pir_mode():
        out = _C_ops.bce_loss(input, label)
        if weight is not None:
            out = _C_ops.multiply(out, weight, 'axis', -1)

        if reduction == 'sum':
            return _C_ops.sum(out, [], None, False)

        elif reduction == 'mean':
            return _C_ops.mean_all(out)
        else:
            return out
    else:
        check_variable_and_dtype(
            input,
            'input',
            ['float16', 'float32', 'float64'],
            'binary_cross_entropy',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['float16', 'float32', 'float64'],
            'binary_cross_entropy',
        )

        sub_name = name if weight is None and reduction == 'none' else None
        helper = LayerHelper("binary_cross_entropy", name=sub_name)
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type='bce_loss',
            inputs={
                'X': [input],
                'Label': [label],
            },
            outputs={'Out': [out]},
        )

        if weight is not None:
            if isinstance(weight, paddle.static.Variable):
                weight_name = name if reduction == 'none' else None
                out = paddle.multiply(out, weight, name=weight_name)
            else:
                raise ValueError(
                    "The weight is not a Tensor, please convert to Tensor."
                )

        if reduction == 'sum':
            return paddle.sum(out, name=name)
        elif reduction == 'mean':
            return paddle.mean(out, name=name)
        else:
            return out


def binary_cross_entropy_with_logits(
    logit: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    reduction: _ReduceMode = 'mean',
    pos_weight: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Combine the sigmoid layer and the :ref:`api_paddle_nn_BCELoss` layer.

    This measures the element-wise probability error in classification tasks
    in which each class is independent.
    This can be thought of as predicting labels for a data-point, where labels
    are not mutually exclusive. For example, a news article can be about
    politics, technology or sports at the same time or none of these.

    Firstly, calculate loss function as follows:

    .. math::
           Out = -Labels * \log(\sigma(Logit)) - (1 - Labels) * \log(1 - \sigma(Logit))

    We know that :math:`\sigma(Logit) = \frac{1}{1 + e^{-Logit}}`. By substituting this we get:

    .. math::
           Out = Logit - Logit * Labels + \log(1 + e^{-Logit})

    For stability and to prevent overflow of :math:`e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:

    .. math::
           Out = \max(Logit, 0) - Logit * Labels + \log(1 + e^{-\|Logit\|})

    Then, if ``weight`` or ``pos_weight`` is not None, then multiply the
    weight tensor on the loss `Out`. The ``weight`` tensor will attach different
    weight on every items in the batch. The ``pos_weight`` will attach different
    weight on the positive label of each class.

    Finally, apply reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target labels ``label`` should be numbers between 0 and 1.

    Args:
        logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``logit``
            is usually the output of Linear layer. Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``logit``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,
            The data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector
            with length equal to the number of classes. The data type is float32, float64.
            Default is ``'None'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor. If ``reduction`` is ``'none'``, the shape of output is
            same as ``logit`` , else the shape of output is scalar.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> logit = paddle.to_tensor([5.0, 1.0, 3.0])
            >>> label = paddle.to_tensor([1.0, 0.0, 1.0])
            >>> output = paddle.nn.functional.binary_cross_entropy_with_logits(logit, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.45618808)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in binary_cross_entropy_with_logits "
            f"should be 'sum', 'mean' or 'none', but received {reduction}, which is not allowed."
        )

    if in_dynamic_or_pir_mode():
        if in_pir_mode():
            check_type(
                logit,
                'logit',
                paddle.pir.Value,
                'binary_cross_entropy_with_logits',
            )
            check_type(
                label,
                'label',
                paddle.pir.Value,
                'binary_cross_entropy_with_logits',
            )
        one = _C_ops.full(
            [1],
            1.0,
            logit.dtype,
            _current_expected_place(),
        )

        if pos_weight is not None:
            pos_weight = _C_ops.add(
                _C_ops.multiply(label, _C_ops.subtract(pos_weight, one)), one
            )
        out = _C_ops.sigmoid_cross_entropy_with_logits(
            logit, label, pos_weight, False, -100
        )

        if weight is not None:
            out = _C_ops.multiply(out, weight)

        if reduction == "sum":
            return _C_ops.sum(out, [], None, False)
        elif reduction == "mean":
            return _C_ops.mean_all(out)
        else:
            return out
    else:
        check_variable_and_dtype(
            logit,
            'logit',
            ['float32', 'float64'],
            'binary_cross_entropy_with_logits',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['float32', 'float64'],
            'binary_cross_entropy_with_logits',
        )
        sigmoid_name = None
        if reduction == 'none' and pos_weight is None and weight is None:
            sigmoid_name = name

        helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

        out = helper.create_variable_for_type_inference(dtype=logit.dtype)

        one = paddle.full(shape=[1], fill_value=1.0, dtype=logit.dtype)
        if pos_weight is not None:
            check_variable_and_dtype(
                pos_weight,
                'pos_weight',
                ['float32', 'float64'],
                'binary_cross_entropy_with_logits',
            )
            pos_weight = paddle.add(
                paddle.multiply(label, paddle.subtract(pos_weight, one)), one
            )

        helper.append_op(
            type="sigmoid_cross_entropy_with_logits",
            inputs={"X": logit, "Label": label, "pos_weight": pos_weight},
            attrs={"ignore_index": kIgnoreIndex, 'normalize': False},
            outputs={"Out": out},
        )

        if weight is not None:
            check_variable_and_dtype(
                weight,
                'weight',
                ['float32', 'float64'],
                'binary_cross_entropy_with_logits',
            )
            weight_name = name if reduction == 'none' else None
            out = paddle.multiply(out, weight, name=weight_name)

        if reduction == "sum":
            return paddle.sum(out, name=name)
        elif reduction == "mean":
            return paddle.mean(out, name=name)
        return out


def hsigmoid_loss(
    input: Tensor,
    label: Tensor,
    num_classes: int,
    weight: Tensor,
    bias: Tensor | None = None,
    path_table: Tensor | None = None,
    path_code: Tensor | None = None,
    is_sparse: bool = False,
    name: str | None = None,
) -> Tensor:
    """
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.

    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.

    Comparing to softmax, hsigmoid can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The API supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_.

    For the custom tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        input (Tensor): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 or float64.
        label (Tensor): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (path_code and path_table is None are None), `num_classes`
            should not be None. If the custom tree is used (path_code and path_table is None are not None),
            `num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        weight (Tensor): A tensor with shape (num_classes - 1, D), with the same data type as `input`.
        bias (Tensor, optional): A tensor with shape (num_classes - 1, 1), with the same data type as `input`.
            If `bias` is None, no bias will be add. Default is None.
        path_table (Tensor, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. If `path_table` and `path_code` are None, the default tree will be used.
            Default is None.
        path_code (Tensor, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. If `path_table` and
            `path_code` are None, the default tree will be used. Default is None.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating. If `is_sparse` is True,
            the gradient of `weight` and `input` will be sparse. Default is False.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as `input`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.set_device('cpu')
            >>> paddle.seed(2023)

            >>> input = paddle.uniform([4, 3])
            >>> print(input)
            Tensor(shape=[4, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[ 0.73167229,  0.04029441, -0.48078126],
                     [ 0.81050646, -0.15199822, -0.18717426],
                     [ 0.94041789,  0.48874724,  0.03570259],
                     [ 0.46585739,  0.95573163, -0.91368192]])
            >>> label = paddle.to_tensor([0, 1, 4, 5])
            >>> num_classes = 5
            >>> weight = paddle.uniform([num_classes - 1, 3])
            >>> print(weight)
            Tensor(shape=[4, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[-0.14721161,  0.43916738, -0.58377075],
                     [-0.60536981, -0.23151302, -0.70793629],
                     [-0.54572451, -0.10784978, -0.56684279],
                     [ 0.35370791, -0.07079649,  0.84765708]])
            >>> out = F.hsigmoid_loss(input, label, num_classes, weight)
            >>> print(out)
            Tensor(shape=[4, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[2.23681736],
                     [1.97140026],
                     [1.66425037],
                     [2.54727197]])

    """
    if num_classes < 2:
        raise ValueError(f'Expected num_classes >= 2 (got {num_classes})')

    if in_dynamic_mode():
        out, _, _ = _C_ops.hsigmoid_loss(
            input,
            label,
            weight,
            bias,
            path_table,
            path_code,
            num_classes,
            is_sparse,
            is_sparse,
        )
        return out

    check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'hsigmoid_loss'
    )
    check_variable_and_dtype(label, 'label', ['int64'], 'hsigmoid_loss')
    check_variable_and_dtype(
        weight, 'weight', ['float32', 'float64'], 'hsigmoid_loss'
    )
    if bias is not None:
        check_variable_and_dtype(
            bias, 'bias', ['float32', 'float64'], 'hsigmoid_loss'
        )
    if path_table is not None:
        check_variable_and_dtype(
            path_table, 'path_table', ['int64'], 'hsigmoid_loss'
        )
    if path_code is not None:
        check_variable_and_dtype(
            path_code, 'path_code', ['int64'], 'hsigmoid_loss'
        )

    if in_pir_mode():
        out, _, _ = _C_ops.hsigmoid_loss(
            input,
            label,
            weight,
            bias,
            path_table,
            path_code,
            num_classes,
            is_sparse,
            is_sparse,
        )
        return out
    else:
        attrs = {
            "num_classes": num_classes,
            "is_sparse": is_sparse,
        }

        inputs = {
            "X": input,
            "W": weight,
            "Bias": bias,
            "PathTable": path_table,
            "PathCode": path_code,
            "Label": label,
        }

        helper = LayerHelper('hsigmoid_loss', **locals())
        out = helper.create_variable_for_type_inference(input.dtype)
        pre_out = helper.create_variable_for_type_inference(input.dtype)
        outputs = {"Out": out, "PreOut": pre_out, "W_Out": weight}

        helper.append_op(
            type="hierarchical_sigmoid",
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
        )
        return out


def smooth_l1_loss(
    input: Tensor,
    label: Tensor,
    reduction: _ReduceMode = 'mean',
    delta: float = 1.0,
    name: str | None = None,
) -> Tensor:
    r"""
    Calculate smooth_l1_loss. Creates a criterion that uses a squared
    term if the absolute element-wise error falls below 1 and an L1 term otherwise.
    In some cases it can prevent exploding gradients and it is more robust and less
    sensitivity to outliers. Also known as the Huber loss:

    .. math::

        loss(x,y) = \frac{1}{n}\sum_{i}z_i


    where :math:`z_i` is given by:

    .. math::

        \mathop{z_i} = \left\{\begin{array}{rcl}
                0.5(x_i - y_i)^2 & & {if |x_i - y_i| < \delta} \\
                \delta * |x_i - y_i| - 0.5 * \delta^2 & & {otherwise}
            \end{array} \right.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is
            (N, C), where C is number of classes, and if shape is more than 2D, this
            is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor, the data type is float32 or float64. The shape of label
            is the same as the shape of input.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        delta (float, optional): Specifies the hyperparameter :math:`\delta` to be used.
            The value determines how large the errors need to be to use L1. Errors
            smaller than delta are minimized with L2. Parameter is ignored for
            negative/zero values. Default = 1.0
        name (str|None, optional): For details, please refer to :ref:`api_guide_Name`. Generally, no setting is required. Default: None.

    Returns:
        Tensor, The tensor variable storing the smooth_l1_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> input = paddle.rand([3, 3]).astype('float32')
            >>> label = paddle.rand([3, 3]).astype('float32')
            >>> output = paddle.nn.functional.smooth_l1_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.08307374)

    """

    if in_dynamic_or_pir_mode():
        out = _C_ops.huber_loss(input, label, delta)
    else:
        check_variable_and_dtype(
            input,
            'input',
            ['float16', 'float32', 'float64', 'uint16'],
            'smooth_l1_loss',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['float16', 'float32', 'float64', 'uint16'],
            'smooth_l1_loss',
        )
        helper = LayerHelper('huber_loss', **locals())
        residual = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        out = helper.create_variable_for_type_inference(
            dtype=helper.input_dtype()
        )
        helper.append_op(
            type='huber_loss',
            inputs={'X': input, 'Y': label},
            outputs={'Out': out, 'Residual': residual},
            attrs={'delta': delta},
        )

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in smooth_l1_loss should be 'sum', 'mean' or"
            f" 'none', but received {reduction}, which is not allowed."
        )
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return paddle.mean(out)
    elif reduction == 'sum':
        return paddle.sum(out)


def margin_ranking_loss(
    input: Tensor,
    other: Tensor,
    label: Tensor,
    margin: float = 0.0,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""

    Calculate the margin rank loss between the input, other and label, use the math function as follows.

    .. math::
        margin\_rank\_loss = max(0, -label * (input - other) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        input(Tensor): the first input tensor, it's data type should be float32, float64.
        other(Tensor): the second input tensor, it's data type should be float32, float64.
        label(Tensor): the label value corresponding to input, it's data type should be float32, float64.
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candidates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
            >>> other = paddle.to_tensor([[2, 1], [2, 4]], dtype='float32')
            >>> label = paddle.to_tensor([[1, -1], [-1, -1]], dtype='float32')
            >>> loss = paddle.nn.functional.margin_ranking_loss(input, other, label)
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.75000000)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in MarginRankingLoss should be 'sum', 'mean' or 'none', but "
            f"received {reduction}, which is not allowed."
        )
    if in_dynamic_or_pir_mode():
        out = _C_ops.subtract(other, input)
        out = _C_ops.multiply(out, label)
        if margin != 0.0:
            margin = paddle.to_tensor([margin], dtype=out.dtype)
            out = _C_ops.add(out, margin)
        out = _C_ops.relu(out)
        if reduction == 'sum':
            return _C_ops.sum(out, [], None, False)
        elif reduction == 'mean':
            return _C_ops.mean_all(out)
        return out
    else:
        helper = LayerHelper("margin_ranking_loss", **locals())
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'margin_rank_loss'
        )
        check_variable_and_dtype(
            other, 'other', ['float32', 'float64'], 'margin_rank_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'margin_rank_loss'
        )

        out = paddle.subtract(input, other)
        neg_label = paddle.neg(label)
        out = paddle.multiply(neg_label, out)

        if margin != 0.0:
            margin_var = out.block.create_var(dtype=out.dtype)
            margin_var = paddle.full(
                shape=[1], fill_value=margin, dtype=out.dtype
            )
            out = paddle.add(out, margin_var)

        result_out = helper.create_variable_for_type_inference(input.dtype)

        if reduction == 'none':
            helper.append_op(
                type="relu", inputs={"X": out}, outputs={"Out": result_out}
            )
            return result_out
        elif reduction == 'sum':
            out = paddle.nn.functional.relu(out)
            attrs = {"dim": [0], "keep_dim": False, "reduce_all": True}
            helper.append_op(
                type="reduce_sum",
                inputs={"X": out},
                outputs={"Out": result_out},
                attrs=attrs,
            )
            return result_out
        elif reduction == 'mean':
            out = paddle.nn.functional.relu(out)
            helper.append_op(
                type="mean",
                inputs={"X": out},
                outputs={"Out": result_out},
                attrs={},
            )
            return result_out


def l1_loss(
    input: Tensor,
    label: Tensor,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""

    Computes the L1 Loss of Tensor ``input`` and ``label`` as follows.

    If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert input - label \rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert input - label \rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert input - label \rvert)


    Parameters:
        input (Tensor): The input tensor. The shapes is [N, `*`], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, `*`], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the L1 Loss of Tensor ``input`` and ``label``.
        If `reduction` is ``'none'``, the shape of output loss is :math:`[N, *]`, the same as ``input`` .
        If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.to_tensor([[1.5, 0.8], [0.2, 1.3]])
            >>> label = paddle.to_tensor([[1.7, 1], [0.4, 0.5]])

            >>> l1_loss = paddle.nn.functional.l1_loss(input, label)
            >>> print(l1_loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.34999999)

            >>> l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='none')
            >>> print(l1_loss)
            Tensor(shape=[2, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[0.20000005, 0.19999999],
                     [0.20000000, 0.79999995]])

            >>> l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='sum')
            >>> print(l1_loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    1.39999998)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
            f"received {reduction}, which is not allowed."
        )

    if in_dynamic_or_pir_mode():
        unreduced = _C_ops.abs(_C_ops.subtract(input, label))

        if reduction == 'mean':
            return _C_ops.mean_all(unreduced)
        elif reduction == 'sum':
            return _C_ops.sum(unreduced, [], None, False)
        else:
            return unreduced
    else:
        check_variable_and_dtype(
            input,
            'input',
            ['float32', 'float64', 'int32', 'int64', 'float16'],
            'l1_loss',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['float32', 'float64', 'int32', 'int64', 'float16'],
            'l1_loss',
        )

        if reduction == 'sum':
            unreduced = paddle.abs(paddle.subtract(x=input, y=label))
            return paddle.sum(unreduced, name=name)
        elif reduction == 'mean':
            unreduced = paddle.abs(paddle.subtract(x=input, y=label))
            return paddle.mean(unreduced, name=name)
        else:
            return paddle.abs(paddle.subtract(x=input, y=label, name=name))


def nll_loss(
    input: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    """
    This api returns negative log likelihood.
    See more detail in :ref:`NLLLoss <api_paddle_nn_NLLLoss>` .


    Parameters:
         input (Tensor): Input tensor, the shape is :math:`[N, C]`, `C` is the number of classes.
             But in K-dimension situation, the shape is :math:`[N, C, d_1, d_2, ..., d_K]`.
             The data type is float32, float64.
         label (Tensor): Label tensor, the shape is :math:`[N,]` or :math:`[N, d_1, d_2, ..., d_K]`.
             The data type is int64.
         weight (Tensor, optional): Weight tensor, a manual rescaling weight given
             to each class. If given, it has to be a 1D Tensor whose size is `[C, ]`. Otherwise,
             it treated as if having all ones. the data type is
             float32, float64, Default is ``'None'``.
         ignore_index (int, optional): Specifies a target value that is ignored
             and does not contribute to the input gradient. Default is -100.
         reduction (str, optional): Indicate how to average the loss,
             the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
             If `reduction` is ``'mean'``, the reduced mean loss is returned;
             if `reduction` is ``'sum'``, the reduced sum loss is returned;
             if `reduction` is ``'none'``, no reduction will be applied.
             Default is ``'mean'``.
         name (str|None, optional): Name for the operation (optional, default is None).
             For more information, please refer to :ref:`api_guide_Name`.

    Returns:
         `Tensor`, the value of negative log likelihood loss.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.nn.functional import nll_loss
            >>> log_softmax = paddle.nn.LogSoftmax(axis=1)

            >>> input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
            ...     [0.53331435, 0.07999352, 0.8549948 ],
            ...     [0.25879037, 0.39530203, 0.698465  ],
            ...     [0.73427284, 0.63575995, 0.18827209],
            ...     [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
            >>> log_out = log_softmax(input)
            >>> label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
            >>> result = nll_loss(log_out, label)
            >>> print(result)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   1.07202101)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
            f"'none', but received {reduction}, which is not allowed."
        )

    input_shape = list(input.shape)
    input_dims = len(input_shape)
    label_shape = list(label.shape)
    label_dims = len(label_shape)

    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f"Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got input_dims{input_dims}, label_dims{label_dims})"
        )

    if input_dims < 2:
        raise ValueError(f'Expected 2 or more dimensions (got {input_dims})')

    if input_shape[1] < 1:
        raise ValueError(
            f"Expected 1 or more classes (got num classes{input_shape[1]})"
        )

    n = input_shape[0]
    c = input_shape[1]
    if in_dynamic_or_pir_mode():
        if input_dims != 2 and input_dims != 4:
            input = _C_ops.reshape(input, [n, c, 1, -1])
            label = _C_ops.reshape(label, [n, 1, -1])
            out_shape = [n] + input_shape[2:]
        out, total_weight = _C_ops.nll_loss(
            input, label, weight, ignore_index, reduction
        )
        if input_dims != 2 and input_dims != 4 and reduction == 'none':
            out = _C_ops.reshape(out, out_shape)
        return out
    else:
        helper = LayerHelper('nll_loss', **locals())

        if input_dims != 2 and input_dims != 4:
            input = reshape(input, shape=[n, c, 1, -1])
            label = reshape(label, shape=[n, 1, -1])
            out_shape = [n] + input_shape[2:]

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'nll_loss'
        )
        check_variable_and_dtype(label, 'label', ['int64'], 'nll_loss')
        inputs = {'X': input, 'Label': label}
        attrs = {'reduction': reduction, 'ignore_index': ignore_index}
        if weight is not None:
            if isinstance(weight, Variable):
                inputs['Weight'] = weight

        out = helper.create_variable_for_type_inference(dtype=input.dtype)
        total_weight = helper.create_variable_for_type_inference(
            dtype=input.dtype
        )
        outputs = {'Out': out, 'Total_weight': total_weight}

        helper.append_op(
            type='nll_loss', inputs=inputs, outputs=outputs, attrs=attrs
        )
        if input_dims != 2 and input_dims != 4 and reduction == 'none':
            out = reshape(out, shape=out_shape)

        return out


def poisson_nll_loss(
    input: Tensor,
    label: Tensor,
    log_input: bool = True,
    full: bool = False,
    epsilon: float = 1e-08,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""Poisson negative log likelihood loss.
    See more detail in :ref:`PoissonNLLLoss <api_paddle_nn_PoissonNLLLoss>` .

    Parameters:
         input (Tensor):
            Input tensor, expectation of underlying Poisson distribution.
            The shape of input tensor should be `(N, *)` or `(*)` where `(*)` denotes any number of extra dimensions.
            It's data type should be float16, bfloat16, float32, float64.
         label (Tensor):
            Label tensor, random sampled from Poisson distribution :math:`label \sim \text{Poisson}(input)`.
            The shape of input tensor should be `(N, *)` or `(*)`, same shape as the input tensor.
            It's data type should be float16, bfloat16, float32, float64.
         log_input (bool, optional):
            Whether to the treat input tensor as log input.
            If ``True`` the loss is computed as, :math:`\exp(\text{input}) - \text{label} * \text{input}` .
            If ``False`` then loss is :math:`\text{input} - \text{label} * \log(\text{input}+\text{epsilon})` .
            Default: ``True``.
         full (bool, optional):
            Whether to compute full loss.
            If ``True``, the Stirling approximation term is added.
            If ``False``, the Stirling approximation is dropped.
            Default: ``False``.
         epsilon (float, optional):
            A small value to avoid evaluation of :math:`\log(0)` when `log_input`\ =\ ``False``. ``epsilon > 0``.
            Default: 1e-8.
         reduction (str, optional):
            Indicate how to reduce the loss, the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be applied.
            Default is ``'mean'``.
         name (str|None, optional):
            Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> paddle.seed(2023)

            >>> input = paddle.randn([5, 2], dtype=paddle.float32)
            >>> label = paddle.randn([5, 2], dtype=paddle.float32)
            >>> loss = F.poisson_nll_loss(input, label, log_input=True, reduction='none')
            >>> print(loss)
            Tensor(shape=[5, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[ 1.09998012,  3.68829036],
                    [ 1.95291090,  0.69603068],
                    [-0.39289063, -2.03713036],
                    [ 4.52518702,  1.28625548],
                    [ 3.94454789,  0.53521496]])
            >>> loss = F.poisson_nll_loss(input, label, reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   1.52983975)

    """
    # check parameter values
    if epsilon <= 0:
        raise ValueError(
            f"The value of `epsilon` in poisson_nll_loss should be positive, but received {epsilon:f}, which is not allowed"
        )

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in poisson_nll_loss should be 'sum', 'mean' or 'none', but "
            f"received {reduction}, which is not allowed."
        )
    # check input dtype and dimension
    check_variable_and_dtype(
        input,
        'input',
        ['float16', 'uint16', 'float32', 'float64'],
        'poisson_nll_loss',
    )
    check_variable_and_dtype(
        label,
        'label',
        ['float16', 'uint16', 'float32', 'float64'],
        'poisson_nll_loss',
    )

    if not (input.shape == label.shape):
        raise ValueError("input's shape must equal to label's shape")

    loss_out = 0
    if log_input:
        loss_out = paddle.exp(input) - label * input
    else:
        loss_out = input - label * paddle.log(input + epsilon)
    if full:
        stirling_approx = (
            label * paddle.log(label)
            - label
            + 0.5 * paddle.log(2 * math.pi * label)
        )
        loss_out += paddle.where(
            label > 1,
            stirling_approx,
            paddle.zeros_like(stirling_approx),
        )
    if reduction == 'mean':
        loss_out = paddle.mean(loss_out)
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out)
    return loss_out


def kl_div(
    input: Tensor,
    label: Tensor,
    reduction: _ReduceMode = 'mean',
    log_target: bool = False,
    name: str | None = None,
) -> Tensor:
    r"""
    Calculate the Kullback-Leibler divergence loss
    between Input(X) and Input(Target). Notes that Input(X) is the
    log-probability and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    If `log_target` is False:

    $$l(x, y) = y * (\log(y) - x)$$

    If `log_target` is True:

    $$l(x, y) = \exp(y) * (y - x)$$

    Here :math:`x` is input and :math:`y` is label.

    If `reduction` is ``'none'``, the output loss is the same shape as the input, and the loss at each point is calculated separately. There is no reduction to the result.

    If `reduction` is ``'mean'``, the output loss is the shape of [], and the output is the average of all losses.

    If `reduction` is ``'sum'``, the output loss is the shape of [], and the output is the sum of all losses.

    If `reduction` is ``'batchmean'``, the output loss is the shape of [N], N is the batch size, and the output is the sum of all losses divided by the batch size.

    Args:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means
            any number of additional dimensions. It's data type should be float32, float64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64.
        reduction (str, optional): Indicate how to average the loss,
            the candidates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            if `reduction` is ``'sum'``, the reduced sum loss is returned;
            if `reduction` is ``'none'``, no reduction will be applied.
            Default is ``'mean'``.
        log_target (bool, optional): Indicate whether `label` is passed in log space. Default is False.
        name(str, optional): Name for the operation (optional, default is None). For more information,
            please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The KL divergence loss. The data type is same as input tensor

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> paddle.seed(2023)

            >>> shape = (5, 20)

            >>> # input(x) should be a distribution in the log space
            >>> x = F.log_softmax(paddle.randn(shape), axis=1).astype('float32')

            >>> target = paddle.uniform(shape, min=-10, max=10).astype('float32')

            >>> # 'batchmean' reduction, loss shape will be [], who is 0-D Tensor
            >>> pred_loss = F.kl_div(x, target, reduction='batchmean')
            >>> print(pred_loss.shape)
            []

            >>> # 'mean' reduction, loss shape will be [], who is 0-D Tensor
            >>> pred_loss = F.kl_div(x, target, reduction='mean')
            >>> print(pred_loss.shape)
            []

            >>> # 'sum' reduction, loss shape will be [], who is 0-D Tensor
            >>> pred_loss = F.kl_div(x, target, reduction='sum')
            >>> print(pred_loss.shape)
            []

            >>> # 'none' reduction, loss shape is same with input shape
            >>> pred_loss = F.kl_div(x, target, reduction='none')
            >>> print(pred_loss.shape)
            [5, 20]

            >>> # if label is in the log space, set log_target = True
            >>> target = paddle.uniform(shape, min=0, max=10).astype('float32')
            >>> log_target = paddle.log(target)
            >>> pred_loss_1 = F.kl_div(x, target, reduction='none')
            >>> pred_loss_2 = F.kl_div(x, log_target, reduction='none', log_target=True)
            >>> print(paddle.allclose(pred_loss_1, pred_loss_2))
            Tensor(shape=[], dtype=bool, place=Place(cpu), stop_gradient=True,
            True)

    """
    # ugly type promotion
    if (
        base.data_feeder.convert_dtype(input.dtype) == 'float32'
        and base.data_feeder.convert_dtype(label.dtype) == 'float64'
    ):
        input = paddle.cast(input, 'float64')
    elif (
        base.data_feeder.convert_dtype(input.dtype) == 'float64'
        and base.data_feeder.convert_dtype(label.dtype) == 'float32'
    ):
        label = paddle.cast(label, 'float64')

    if in_dynamic_or_pir_mode():
        out = _C_ops.kldiv_loss(input, label, 'none', log_target)
        if reduction == 'mean':
            out = paddle.mean(out)
        elif reduction == 'sum':
            out = paddle.sum(out)
        elif reduction == 'batchmean':
            if len(input.shape) > 0:
                batch_size = input.shape[0]
                out = paddle.sum(out) / batch_size
        return out
    else:
        helper = LayerHelper('kl_div', **locals())

        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'kl_div'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'kl_div'
        )
        base.data_feeder.check_type(reduction, 'reduction', str, 'kl_div')

        loss = helper.create_variable_for_type_inference(dtype=input.dtype)
        helper.append_op(
            type='kldiv_loss',
            inputs={'X': input, 'Target': label},
            outputs={'Loss': loss},
            attrs={'reduction': 'none', 'log_target': log_target},
        )

        if reduction == 'mean':
            loss = paddle.mean(loss)
        elif reduction == 'sum':
            loss = paddle.sum(loss)
        elif reduction == 'batchmean':
            batch_size = paddle.shape(input)[0]
            loss = paddle.sum(loss) / batch_size
        return loss


def mse_loss(
    input: Tensor,
    label: Tensor,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
    Accept input predications and label and returns the mean square error.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    Parameters:
        input (Tensor): Input tensor, the data type should be float32 or float64.
        label (Tensor): Label tensor, the data type should be float32 or float64.
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor, The tensor tensor storing the mean square error difference of input and label.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> mse_loss = paddle.nn.loss.MSELoss()
            >>> input = paddle.to_tensor(1.5)
            >>> label = paddle.to_tensor(1.7)
            >>> output = mse_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.04000002)

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'mse_loss' should be 'sum', 'mean' or 'none', "
            f"but received {reduction}."
        )

    if not in_dynamic_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'mse_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'mse_loss'
        )

    if reduction == 'none':
        return paddle.square(paddle.subtract(input, label), name=name)
    elif reduction == 'mean':
        return paddle.mean(
            paddle.square(paddle.subtract(input, label)), name=name
        )
    else:
        return paddle.sum(
            paddle.square(paddle.subtract(input, label)), name=name
        )


def ctc_loss(
    log_probs: Tensor,
    labels: Tensor,
    input_lengths: Tensor,
    label_lengths: Tensor,
    blank: int = 0,
    reduction: _ReduceMode = 'mean',
    norm_by_times: bool = False,
) -> Tensor:
    """

    An operator integrating the open source Warp-CTC library (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation
    is integrated to the Warp-CTC library to normalize values for each row of the input tensor.

    Parameters:
        log_probs (Tensor): The unscaled probability sequence with padding, which is a 3-D Tensor. The tensor shape is [max_logit_length, batch_size, num_classes + 1], where max_logit_length is the longest length of input logit sequence. The data type should be float32 or float64.
        labels (Tensor): The ground truth sequence with padding, which must be a 3-D Tensor. The tensor shape is [batch_size, max_label_length], where max_label_length is the longest length of label sequence. The data type must be int32.
        input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.
        blank (int, optional): The blank label index of Connectionist Temporal Classification (CTC) loss, which is in the half-opened interval [0, num_classes + 1). The data type must be int32. Default: 0.
        reduction (str, optional): Indicate how to average the loss, the candidates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output loss will be divided by the label_lengths, and then return the mean of quotient; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default: ``'mean'``.
        norm_by_times (bool, optional): Whether to normalize the gradients by the number of time-step, which is also the sequence's length. There is no need to normalize the gradients if reduction mode is 'mean'. Default: False.

    Returns:
        Tensor, The Connectionist Temporal Classification (CTC) loss between ``log_probs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is []. Data type is the same as ``log_probs``.

    Examples:

        .. code-block:: python

            >>> # declarative mode
            >>> import paddle.nn.functional as F
            >>> import paddle
            >>> import numpy as np

            >>> # length of the longest logit sequence
            >>> max_seq_length = 4
            >>> #length of the longest label sequence
            >>> max_label_length = 3
            >>> # number of logit sequences
            >>> batch_size = 2
            >>> # class num
            >>> class_num = 3

            >>> log_probs = paddle.to_tensor(np.array([
            ...     [[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
            ...      [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],
            ...     [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
            ...      [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],
            ...     [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
            ...      [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],
            ...     [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
            ...      [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],
            ...     [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
            ...      [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]
            ... ]), dtype="float32")
            >>> labels = paddle.to_tensor([[1, 2, 2],
            ...     [1, 2, 2]], dtype="int32")
            >>> input_lengths = paddle.to_tensor([5, 5], dtype="int64")
            >>> label_lengths = paddle.to_tensor([3, 3], dtype="int64")

            >>> loss = F.ctc_loss(log_probs, labels,
            ...     input_lengths,
            ...     label_lengths,
            ...     blank=0,
            ...     reduction='none')
            >>> print(loss)
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [3.91798496, 2.90765190])

            >>> loss = F.ctc_loss(log_probs, labels,
            ...     input_lengths,
            ...     label_lengths,
            ...     blank=0,
            ...     reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   1.13760614)

    """

    def warpctc(
        input,
        label,
        blank=0,
        norm_by_times=False,
        input_length=None,
        label_length=None,
    ):
        if in_dynamic_or_pir_mode():
            if input_length is None or label_length is None:
                raise ValueError(
                    "input_length and label_length must not be None in dygraph mode!"
                )
            loss_out = _C_ops.warpctc(
                input, label, input_length, label_length, blank, norm_by_times
            )
            return loss_out
        else:
            helper = LayerHelper('warpctc', **locals())
            check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], "warpctc"
            )
            check_variable_and_dtype(label, 'label', ['int32'], "warpctc")
            this_inputs = {'Logits': [input], 'Label': [label]}
            if input_length is not None and label_length is not None:
                check_variable_and_dtype(
                    input_length, 'LogitsLength', ['int64'], "warpctc"
                )
                check_variable_and_dtype(
                    label_length, 'LabelLength', ['int64'], "warpctc"
                )
                this_inputs['LogitsLength'] = [input_length]
                this_inputs['LabelLength'] = [label_length]

            loss_out = helper.create_variable_for_type_inference(
                dtype=input.dtype
            )
            grad_out = helper.create_variable_for_type_inference(
                dtype=input.dtype
            )

            helper.append_op(
                type='warpctc',
                inputs=this_inputs,
                outputs={'WarpCTCGrad': [grad_out], 'Loss': [loss_out]},
                attrs={
                    'blank': blank,
                    'norm_by_times': norm_by_times,
                },
            )
            return loss_out

    loss_out = warpctc(
        log_probs, labels, blank, norm_by_times, input_lengths, label_lengths
    )

    loss_out = paddle.squeeze(loss_out, [-1])
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        loss_out = paddle.mean(loss_out / label_lengths.astype(loss_out.dtype))
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out)
    return loss_out


def rnnt_loss(
    input: Tensor,
    label: Tensor,
    input_lengths: Tensor,
    label_lengths: Tensor,
    blank: int = 0,
    fastemit_lambda: float = 0.001,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    """
    An operator integrating the open source Warp-Transducer library (https://github.com/b-flo/warp-transducer.git)
    to compute Sequence Transduction with Recurrent Neural Networks (RNN-T) loss.

    Parameters:
        input (Tensor): The logprobs sequence with padding, which is a 4-D Tensor. The tensor shape is [B, Tmax, Umax, D], where Tmax is the longest length of input logit sequence. The data type should be float32 or float64.
        label (Tensor): The ground truth sequence with padding, which must be a 2-D Tensor. The tensor shape is [B, Umax], where Umax is the longest length of label sequence. The data type must be int32.
        input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.
        blank (int, optional): The blank label index of RNN-T loss, which is in the half-opened interval [0, B). The data type must be int32. Default is 0.
        fastemit_lambda (float, default 0.001): Regularization parameter for FastEmit (https://arxiv.org/pdf/2010.11148.pdf)
        reduction (string, optional): Indicate how to average the loss, the candidates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output will be sum of loss and be divided by the batch_size; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default is ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, The RNN-T loss between ``logprobs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is []. Data type is the same as ``logprobs``.

    Examples:

        .. code-block:: python

            >>> # declarative mode
            >>> import paddle.nn.functional as F
            >>> import numpy as np
            >>> import paddle
            >>> import functools

            >>> fn = functools.partial(F.rnnt_loss, reduction='sum', fastemit_lambda=0.0, blank=0)

            >>> acts = np.array([[
            ...     [[0.1, 0.6, 0.1, 0.1, 0.1],
            ...      [0.1, 0.1, 0.6, 0.1, 0.1],
            ...      [0.1, 0.1, 0.2, 0.8, 0.1]],
            ...     [[0.1, 0.6, 0.1, 0.1, 0.1],
            ...      [0.1, 0.1, 0.2, 0.1, 0.1],
            ...      [0.7, 0.1, 0.2, 0.1, 0.1]]
            ... ]])
            >>> labels = [[1, 2]]

            >>> acts = paddle.to_tensor(acts, stop_gradient=False)

            >>> lengths = [acts.shape[1]] * acts.shape[0]
            >>> label_lengths = [len(l) for l in labels]
            >>> labels = paddle.to_tensor(labels, paddle.int32)
            >>> lengths = paddle.to_tensor(lengths, paddle.int32)
            >>> label_lengths = paddle.to_tensor(label_lengths, paddle.int32)

            >>> costs = fn(acts, labels, lengths, label_lengths)
            >>> print(costs)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=False,
                   -2.85042444)

    """

    def warprnnt(
        input, label, input_length, label_length, blank=0, fastemit_lambda=0.001
    ):
        if in_dynamic_or_pir_mode():
            loss_out = _C_ops.warprnnt(
                input,
                label,
                input_length,
                label_length,
                blank,
                fastemit_lambda,
            )
            return loss_out
        helper = LayerHelper('warprnnt', **locals())
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], "warprnnt"
        )
        check_variable_and_dtype(label, 'label', ['int32'], "warprnnt")
        check_variable_and_dtype(
            input_length, 'input_lengths', ['int32'], "warprnnt"
        )
        check_variable_and_dtype(
            label_length, 'label_lengths', ['int32'], "warprnnt"
        )
        this_inputs = {
            'input': [input],
            'label': [label],
            'input_lengths': [input_length],
            'label_lengths': [label_length],
        }

        loss_out = helper.create_variable_for_type_inference(dtype=input.dtype)
        grad_out = helper.create_variable_for_type_inference(dtype=input.dtype)

        helper.append_op(
            type='warprnnt',
            inputs=this_inputs,
            outputs={'warprnntgrad': [grad_out], 'loss': [loss_out]},
            attrs={
                'blank': blank,
                'fastemit_lambda': fastemit_lambda,
            },
        )
        return loss_out

    B = input.shape[0]

    # NOTE manually done log_softmax for CPU version,
    # log_softmax is computed within GPU version.

    # (B,)
    loss_out = warprnnt(
        input, label, input_lengths, label_lengths, blank, fastemit_lambda
    )

    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        loss_out = paddle.sum(loss_out, name=name) / B
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out, name=name)
    return loss_out


@overload
def margin_cross_entropy(
    logits: Tensor,
    label: Tensor,
    margin1: float = ...,
    margin2: float = ...,
    margin3: float = ...,
    scale: float = ...,
    group=...,
    return_softmax: Literal[True] = ...,
    reduction: _ReduceMode | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def margin_cross_entropy(
    logits: Tensor,
    label: Tensor,
    margin1: float = ...,
    margin2: float = ...,
    margin3: float = ...,
    scale: float = ...,
    group=...,
    return_softmax: Literal[False] = ...,
    reduction: _ReduceMode | None = ...,
) -> Tensor: ...


@overload
def margin_cross_entropy(
    logits: Tensor,
    label: Tensor,
    margin1: float = ...,
    margin2: float = ...,
    margin3: float = ...,
    scale: float = ...,
    group=...,
    return_softmax: bool = ...,
    reduction: _ReduceMode | None = ...,
) -> Tensor | tuple[Tensor, Tensor]: ...


def margin_cross_entropy(
    logits,
    label,
    margin1=1.0,
    margin2=0.5,
    margin3=0.0,
    scale=64.0,
    group=None,
    return_softmax=False,
    reduction='mean',
):
    r"""
    .. math::

        L=-\frac{1}{N}\sum^N_{i=1}\log\frac{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}}{e^{s(cos(m_{1}\theta_{y_i}+m_{2})-m_{3})}+\sum^n_{j=1,j\neq y_i} e^{scos\theta_{y_i}}}

    where the :math:`\theta_{y_i}` is the angle between the feature :math:`x` and
    the representation of class :math:`i`. The details of ArcFace loss
    could be referred to https://arxiv.org/abs/1801.07698.

    .. hint::
        The API supports single GPU and multi GPU, and don't supports CPU.
        For data parallel mode, set ``group=False``.
        For model parallel mode, set ``group=None`` or the group instance return by paddle.distributed.new_group.
        And logits.shape[-1] can be different at each rank.

    Args:
        logits (Tensor): shape[N, local_num_classes], the output of the normalized X multiply the normalized W.
                The logits is shard_logits when using model parallel.
        label (Tensor): shape[N] or shape[N, 1], the ground truth label.
        margin1 (float, optional): m1 of margin loss, default value is `1.0`.
        margin2 (float, optional): m2 of margin loss, default value is `0.5`.
        margin3 (float, optional): m3 of margin loss, default value is `0.0`.
        scale (float, optional): s of margin loss, default value is `64.0`.
        group (Group, optional): The group instance return by paddle.distributed.new_group
            or ``None`` for global default group or ``False`` for data parallel (do not communication cross ranks).
            Default is ``None``.
        return_softmax (bool, optional): Whether return softmax probability. Default value is `False`.
        reduction (str|None, optional): The candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                    If :attr:`reduction` is ``'mean'``, return the average of loss;
                    If :attr:`reduction` is ``'sum'``, return the sum of loss;
                    If :attr:`reduction` is ``'none'``, no reduction will be applied.
                    Default value is `'mean'`.

    Returns:
        Tensor|tuple[Tensor, Tensor], return the cross entropy loss if
            `return_softmax` is False, otherwise the tuple (loss, softmax),
            softmax is shard_softmax when using model parallel, otherwise
            softmax is in the same shape with input logits. If
            ``reduction == None``, the shape of loss is ``[N, 1]``, otherwise
            the shape is ``[]``.

    Examples:

        .. code-block:: python
            :name: code-example1

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import paddle
            >>> paddle.seed(2023)
            >>> paddle.device.set_device('gpu')
            >>> m1 = 1.0
            >>> m2 = 0.5
            >>> m3 = 0.0
            >>> s = 64.0
            >>> batch_size = 2
            >>> feature_length = 4
            >>> num_classes = 4

            >>> label = paddle.randint(low=0, high=num_classes, shape=[batch_size], dtype='int64')

            >>> X = paddle.randn(
            ...     shape=[batch_size, feature_length],
            ...     dtype='float64')
            >>> X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
            >>> X = paddle.divide(X, X_l2)

            >>> W = paddle.randn(
            ...     shape=[feature_length, num_classes],
            ...     dtype='float64')
            >>> W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
            >>> W = paddle.divide(W, W_l2)

            >>> logits = paddle.matmul(X, W)
            >>> loss, softmax = paddle.nn.functional.margin_cross_entropy(
            ...     logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)
            >>> print(logits)
            Tensor(shape=[2, 4], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                   [[-0.59561850,  0.32797505,  0.80279214,  0.00144975],
                    [-0.16265212,  0.84155098,  0.62008629,  0.79126072]])
            >>> print(label)
            Tensor(shape=[2], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                   [1, 0])
            >>> print(loss)
            Tensor(shape=[2, 1], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                   [[61.94391901],
                    [93.30853839]])
            >>> print(softmax)
            Tensor(shape=[2, 4], dtype=float64, place=Place(gpu:0), stop_gradient=True,
                   [[0.00000000, 0.00000000, 1.        , 0.00000000],
                    [0.00000000, 0.96152676, 0.00000067, 0.03847257]])

        .. code-block:: python
            :name: code-example2

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # Multi GPU, test_margin_cross_entropy.py
            >>> from typing import List
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> paddle.seed(2023)
            >>> strategy = dist.fleet.DistributedStrategy()
            >>> dist.fleet.init(is_collective=True, strategy=strategy)
            >>> rank_id = dist.get_rank()
            >>> m1 = 1.0
            >>> m2 = 0.5
            >>> m3 = 0.0
            >>> s = 64.0
            >>> batch_size = 2
            >>> feature_length = 4
            >>> num_class_per_card = [4, 8]
            >>> num_classes = paddle.sum(paddle.to_tensor(num_class_per_card))

            >>> label = paddle.randint(low=0, high=num_classes.item(), shape=[batch_size], dtype='int64') # type: ignore
            >>> label_list: List[paddle.Tensor] = []
            >>> dist.all_gather(label_list, label)
            >>> label = paddle.concat(label_list, axis=0)

            >>> X = paddle.randn(
            ...     shape=[batch_size, feature_length],
            ...     dtype='float64'
            ... )
            >>> X_list: List[paddle.Tensor] = []
            >>> dist.all_gather(X_list, X)
            >>> X = paddle.concat(X_list, axis=0)
            >>> X_l2 = paddle.sqrt(paddle.sum(paddle.square(X), axis=1, keepdim=True))
            >>> X = paddle.divide(X, X_l2)

            >>> W = paddle.randn(
            ...     shape=[feature_length, num_class_per_card[rank_id]],
            ...     dtype='float64')
            >>> W_l2 = paddle.sqrt(paddle.sum(paddle.square(W), axis=0, keepdim=True))
            >>> W = paddle.divide(W, W_l2)

            >>> logits = paddle.matmul(X, W)
            >>> loss, softmax = paddle.nn.functional.margin_cross_entropy(
            ...     logits, label, margin1=m1, margin2=m2, margin3=m3, scale=s, return_softmax=True, reduction=None)
            >>> print(logits)
            >>> print(label)
            >>> print(loss)
            >>> print(softmax)

            >>> # python -m paddle.distributed.launch --gpus=0,1 --log_dir log test_margin_cross_entropy.py
            >>> # cat log/workerlog.0
            >>> # Tensor(shape=[4, 4], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[-0.59561850,  0.32797505,  0.80279214,  0.00144975],
            >>> #         [-0.16265212,  0.84155098,  0.62008629,  0.79126072],
            >>> #         [-0.59561850,  0.32797505,  0.80279214,  0.00144975],
            >>> #         [-0.16265212,  0.84155098,  0.62008629,  0.79126072]])
            >>> # Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
            >>> #        [5, 4, 5, 4])
            >>> # Tensor(shape=[4, 1], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[104.27437027],
            >>> #         [113.40243782],
            >>> #         [104.27437027],
            >>> #         [113.40243782]])
            >>> # Tensor(shape=[4, 4], dtype=float64, place=Place(gpu:0), stop_gradient=True,
            >>> #        [[0.00000000, 0.00000000, 0.01210039, 0.00000000],
            >>> #         [0.00000000, 0.96152674, 0.00000067, 0.03847257],
            >>> #         [0.00000000, 0.00000000, 0.01210039, 0.00000000],
            >>> #         [0.00000000, 0.96152674, 0.00000067, 0.03847257]])
            >>> # cat log/workerlog.1
            >>> # Tensor(shape=[4, 8], dtype=float64, place=Place(gpu:1), stop_gradient=True,
            >>> #        [[-0.34913275, -0.35180883, -0.53976657, -0.75234331,  0.70534995,
            >>> #           0.87157838,  0.31064437,  0.19537700],
            >>> #         [-0.63941012, -0.05631600, -0.02561853,  0.09363013,  0.56571130,
            >>> #           0.13611246,  0.08849565,  0.39219619],
            >>> #         [-0.34913275, -0.35180883, -0.53976657, -0.75234331,  0.70534995,
            >>> #           0.87157838,  0.31064437,  0.19537700],
            >>> #         [-0.63941012, -0.05631600, -0.02561853,  0.09363013,  0.56571130,
            >>> #           0.13611246,  0.08849565,  0.39219619]])
            >>> # Tensor(shape=[4], dtype=int64, place=Place(gpu:1), stop_gradient=True,
            >>> #        [5, 4, 5, 4])
            >>> # Tensor(shape=[4, 1], dtype=float64, place=Place(gpu:1), stop_gradient=True,
            >>> #        [[104.27437027],
            >>> #         [113.40243782],
            >>> #         [104.27437027],
            >>> #         [113.40243782]])
            >>> # Tensor(shape=[4, 8], dtype=float64, place=Place(gpu:1), stop_gradient=True,
            >>> #        [[0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00002368, 0.98787593,
            >>> #          0.00000000, 0.00000000],
            >>> #         [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000002, 0.00000000,
            >>> #          0.00000000, 0.00000000],
            >>> #         [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00002368, 0.98787593,
            >>> #          0.00000000, 0.00000000],
            >>> #         [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000002, 0.00000000,
            >>> #          0.00000000, 0.00000000]])

    """

    assert reduction in ['mean', 'sum', 'none', None]
    if not (group is False or group is None or hasattr(group, 'is_member')):
        raise ValueError(
            f'Expected group is False, None or instance of paddle.distributed.collective.Group \
             (got group: {group})'
        )
        return

    if hasattr(group, 'is_member') and not group.is_member():
        return

    ring_id = 0
    rank = 0
    nranks = 1
    if group is not False:
        ring_id = 0 if group is None else group.id
        if core.is_compiled_with_dist():
            parallel_env = paddle.distributed.ParallelEnv()
            global_rank = parallel_env.rank
            rank = (
                global_rank
                if group is None
                else group.get_group_rank(global_rank)
            )
            nranks = parallel_env.world_size if group is None else group.nranks

    input_dims = len(list(logits.shape))
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f'Expected input_dims - 1 = label_dims or input_dims == label_dims\
             (got input_dims{input_dims}, label_dims{label_dims})'
        )
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=-1)

    if in_dynamic_or_pir_mode():
        softmax, loss = _C_ops.margin_cross_entropy(
            logits,
            label,
            return_softmax,
            ring_id,
            rank,
            nranks,
            margin1,
            margin2,
            margin3,
            scale,
        )
        if reduction == 'mean':
            loss = paddle.mean(loss)
        elif reduction == 'sum':
            loss = paddle.sum(loss)
        if not return_softmax:
            return loss
        else:
            return loss, softmax
    else:
        op_type = 'margin_cross_entropy'
        helper = LayerHelper(op_type, **locals())
        softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
        loss = helper.create_variable_for_type_inference(dtype=logits.dtype)

        check_variable_and_dtype(
            logits,
            'logits',
            ['float16', 'float32', 'float64'],
            'margin_cross_entropy',
        )
        check_variable_and_dtype(
            label, 'label', ['int32', 'int64'], 'margin_cross_entropy'
        )

        helper.append_op(
            type=op_type,
            inputs={'Logits': logits, 'Label': label},
            outputs={'Softmax': softmax, 'Loss': loss},
            attrs={
                'return_softmax': return_softmax,
                'ring_id': ring_id,
                'rank': rank,
                'nranks': nranks,
                'margin1': margin1,
                'margin2': margin2,
                'margin3': margin3,
                'scale': scale,
            },
        )

        if reduction == 'mean':
            loss = paddle.mean(loss)
        elif reduction == 'sum':
            loss = paddle.sum(loss)

        if not return_softmax:
            return loss
        else:
            return loss, softmax


@overload
def softmax_with_cross_entropy(
    logits: Tensor,
    label: Tensor,
    soft_label: bool = ...,
    ignore_index: int = ...,
    numeric_stable_mode: bool = ...,
    return_softmax: Literal[True] = ...,
    axis: int = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def softmax_with_cross_entropy(
    logits: Tensor,
    label: Tensor,
    soft_label: bool = ...,
    ignore_index: int = ...,
    numeric_stable_mode: bool = ...,
    return_softmax: Literal[False] = ...,
    axis: int = ...,
) -> Tensor: ...


@overload
def softmax_with_cross_entropy(
    logits: Tensor,
    label: Tensor,
    soft_label: bool = ...,
    ignore_index: int = ...,
    numeric_stable_mode: bool = ...,
    return_softmax: bool = ...,
    axis: int = ...,
) -> Tensor | tuple[Tensor, Tensor]: ...


@deprecated(
    since="2.0.0",
    update_to="paddle.nn.functional.cross_entropy",
    level=1,
    reason=(
        'Please notice that behavior of "paddle.nn.functional.softmax_with_cross_entropy" '
        'and "paddle.nn.functional.cross_entropy" is different.'
    ),
)
def softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=-100,
    numeric_stable_mode=True,
    return_softmax=False,
    axis=-1,
):
    r"""
    This operator implements the cross entropy loss function with softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators
    expects mutually exclusive hard labels, each sample in a batch is in exactly
    one class with a probability of 1.0. Each sample in the batch will have a
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::
        \\loss_j=-\text{logits}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::
        \\loss_j= -\sum_{i=0}^{K}\text{label}_i\left(\text{logits}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right)\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::
        \\max_j&=\max_{i=0}^{K}{\text{logits}_i} \\
                log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logits_i - max_j)\\
                softmax_j &= \exp(logits_j - max_j - {log\_max\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Tensor): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`,
            Label is a ``Tensor``  in the same shape with :attr:`logits`.
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor``
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpret the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`.
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False`
                                              and GPU is used. When :attr:`soft_label`
                                              is :attr:`True` or CPU is used, the
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        - If `return_softmax` is False, return the cross entropy loss as a ``Tensor``.
          The dtype is the same as the input ``logits``. The shape is consistent with ``logits`` except in dimension :attr:`axis` as 1.
        - If `return_softmax` is True, return a tuple of two ``Tensor``: the cross entropy loss and the softmax result.
          The dtype of the cross entropy loss is the same as the input ``logits``, and the shape is consistent with ``logits``
          except in dimension :attr:`axis` as 1. The dtype and shape of the softmax result are the same as the input ``logits``.


    Examples:
        .. code-block:: python

            >>> import paddle

            >>> logits = paddle.to_tensor([0.4, 0.6, 0.9], dtype="float32")
            >>> label = paddle.to_tensor([1], dtype="int64")

            >>> out = paddle.nn.functional.softmax_with_cross_entropy(logits=logits, label=label)
            >>> print(out)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [1.15328646])

    """
    return base_softmax_with_cross_entropy(
        logits,
        label,
        soft_label,
        ignore_index,
        numeric_stable_mode,
        return_softmax,
        axis,
    )


def cross_entropy(
    input: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: _ReduceMode = 'mean',
    soft_label: bool = False,
    axis: int = -1,
    use_softmax: bool = True,
    label_smoothing: float = 0.0,
    name: str | None = None,
) -> Tensor:
    r"""

    By default, the cross entropy loss function is implemented using softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable computing.

    Calculate the cross entropy loss function without softmax when use_softmax=False.

    By default, calculate the mean of the result, and you can also affect
    the default behavior by using the reduction parameter. Please refer to the part of
    parameters for details.

    Can be used to calculate the softmax cross entropy loss with soft and hard labels.
    Where, the hard labels mean the actual label value, 0, 1, 2, etc.  And the soft labels
    mean the probability of the actual label, 0.6, 0.8, 0.2, etc.

    The calculation includes the following two steps.

    - **1.softmax cross entropy**

        1. Hard label (each sample can only be assigned into one category)

        1.1. when use_softmax=True

            .. math::
              \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        1.2. when use_softmax=False

            .. math::
              \\loss_j=-\log\left({P}_{label_j}\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).


        2. Soft label (each sample is assigned to multiple categories with a certain probability, and the probability sum is 1).

        2.1. when use_softmax=True

            .. math::
              \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories.

        2.2. when use_softmax=False

            .. math::
              \\loss_j=-\sum_{j=0}^{C}\left({label}_j*\log\left({P}_{label_j}\right)\right) , j = 1,...,N

            where, N is the number of samples and C is the number of categories, P is input(the output of softmax).




    - **2. Weight and reduction processing**

        1. Weight

            If the ``weight`` parameter is ``None`` , go to the next step directly.

            If the ``weight`` parameter is not ``None`` , the cross entropy of each sample is weighted by weight
            according to soft_label = False or True as follows.

            1.1. Hard labels (soft_label = False)

            .. math::
                \\loss_j=loss_j*weight[label_j]


            1.2. Soft labels (soft_label = True)

             .. math::
                \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

        2. reduction

            2.1 if the ``reduction`` parameter is ``none``

                Return the previous result directly

            2.2 if the ``reduction`` parameter is ``sum``

                Return the sum of the previous results

            .. math::
               \\loss=\sum_{j}loss_j

            2.3 if the ``reduction`` parameter is ``mean`` , it will be processed according to
            the ``weight`` parameter as follows.

            2.3.1. If the  ``weight``  parameter is ``None``

                   Return the average value of the previous results

            .. math::
                \\loss=\sum_{j}loss_j/N

                  where, N is the number of samples and C is the number of categories.

            2.3.2. If the 'weight' parameter is not 'None', the weighted average value of the previous result will be returned

            1. Hard labels (soft_label = False)

            .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j]

            2. Soft labels (soft_label = True)

            .. math::
                \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)


    Parameters:
        input (Tensor): the data type is float32, float64. Shape is :math:`[N_1, N_2, ..., N_k, C]`, where C is number of classes, ``k >= 1`` .

            Note:
                1. when use_softmax=True, it expects unscaled logits. This operator should not be used with the output of softmax operator, which will produce incorrect results.
                2. when use_softmax=False, it expects the output of softmax operator.

        label (Tensor):
            1. If soft_label=False, the shape is
            :math:`[N_1, N_2, ..., N_k]` or :math:`[N_1, N_2, ..., N_k, 1]`, k >= 1.
            the data type is int32, int64, float32, float64, where each value is [0, C-1].

            2. If soft_label=True and no label_smoothing, the shape and data type
            should be same with ``input`` , and the sum of the labels for each sample should be 1.

            3. If has label_smoothing, (i.e. label_smoothing > 0.0), no matter what ``soft_label`` is,
            the shape and data type of ``label`` could be either the situation 1 or situation 2.
            In other words, if label_smoothing > 0.0, the format of label could be one-hot label or integer label.

        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size C and the data type is float32, float64.
            Default is ``'None'`` .
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the loss. A negative value means that no label
            value needs to be ignored. Only valid when soft_label = False.
            Default is ``-100`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        soft_label (bool, optional): Indicate whether label is soft. Default is ``False``.
        label_smoothing (float, optional): A float in [0.0, 1.0].
            Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing.
            The targets become  a mixture of the original ground truth and a uniform distribution as
            described in paper 'Rethinking the Inception Architecture for Computer Vision'.
            Default is ``0.0``.
        axis (int, optional):The index of dimension to perform softmax calculations.
            It should be in range :math:`[-1, rank - 1]`, where :math:`rank` is the
            number of dimensions of input :attr:`input`.
            Default is ``-1`` .
        use_softmax (bool, optional): Indicate whether compute softmax before cross_entropy.
            Default is ``True``.
        name (str|None, optional): The name of the operator. Default is ``None`` .
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:

        Tensor. Return the softmax cross_entropy loss of ``input`` and ``label``.
        The data type is the same as input.

        If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the dimension of return value is ``1``.

        If :attr:`reduction` is ``'none'``:

        1. If soft_label = False, the dimension of return value is the same with ``label`` .

        2. if soft_label = True, the dimension of return value is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> # hard labels
            >>> import paddle
            >>> paddle.seed(99999)
            >>> N=100
            >>> C=200
            >>> reduction='mean'
            >>> input =  paddle.rand([N, C], dtype='float64')
            >>> label =  paddle.randint(0, C, shape=[N], dtype='int64')
            >>> weight = paddle.rand([C], dtype='float64')

            >>> cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
            ...     weight=weight, reduction=reduction)
            >>> dy_ret = cross_entropy_loss(
            ...                             input,
            ...                             label)

            >>> print(dy_ret)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
                   5.35419278)

        .. code-block:: python
            :name: code-example2

            >>> # soft labels
            >>> # case1: soft labels without label_smoothing
            >>> import paddle
            >>> from typing import Optional
            >>> paddle.seed(99999)
            >>> axis = -1
            >>> N = 4
            >>> C = 3
            >>> shape = [N, C]
            >>> reduction='mean'
            >>> weight: Optional[paddle.Tensor] = None
            >>> logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> labels /= paddle.sum(labels, axis=axis, keepdim=True)
            >>> paddle_loss_mean = paddle.nn.functional.cross_entropy(
            ...     logits,
            ...     labels,
            ...     soft_label=True,
            ...     axis=axis,
            ...     weight=weight,
            ...     reduction=reduction
            ... )
            >>> print(paddle_loss_mean)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
                   1.12801195)


            >>> # case2: soft labels with label_smoothing
            >>> import paddle
            >>> from typing import Optional
            >>> paddle.seed(99999)
            >>> axis = -1
            >>> N = 4
            >>> C = 3
            >>> shape = [N, C]
            >>> label_smoothing = 0.4
            >>> reduction='mean'
            >>> weight: Optional[paddle.Tensor] = None
            >>> logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
            >>> integer_labels = paddle.randint(low=0, high=C, shape=[N], dtype='int64')
            >>> one_hot_labels = paddle.nn.functional.one_hot(integer_labels, C).astype('float32')

            >>> # integer labels
            >>> paddle_integer_loss_mean = paddle.nn.functional.cross_entropy(
            ...     logits,
            ...     integer_labels,
            ...     axis=axis,
            ...     weight=weight,
            ...     label_smoothing=label_smoothing,
            ...     reduction=reduction
            ... )
            >>> print(paddle_integer_loss_mean)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            1.08317309)

            >>> # one_hot labels
            >>> paddle_one_hot_loss_mean = paddle.nn.functional.cross_entropy(
            ...                                                         logits,
            ...                                                         one_hot_labels,
            ...                                                         axis=axis,
            ...                                                         weight=weight,
            ...                                                         label_smoothing=label_smoothing,
            ...                                                         reduction=reduction)
            >>> print(paddle_one_hot_loss_mean)
            Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
            1.08317309)

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropy"
            f"should be 'sum', 'mean' or 'none', but received {reduction}, which is not allowed."
        )
    if ignore_index > 0 and soft_label:
        raise ValueError(
            "When soft_label == True, the value of 'ignore_index' in softmax_cross_entropy"
            f"should be '-100', but received {ignore_index}, which is not allowed."
        )

    input_dims = len(list(input.shape))
    if input_dims == 0:
        raise ValueError('The dimension of input should be larger than zero!')

    label_dims = len(list(label.shape))
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=axis)

    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            f'Expected nput_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{input_dims}, label_dims{label_dims})'
        )

    if label_smoothing > 0.0:
        soft_label = True
        # converting the label to one-hot encoding
        # for 1d case, converting label's shape from [N] to [N, C]
        # for 2d case, converting label's shape from [N, d_1, ..., d_k] to [N, d_1, ..., d_k, C]
        if input_dims - 1 == label_dims:
            label = paddle.squeeze(label, axis=axis)
            label = paddle.nn.functional.one_hot(label, input.shape[-1])

        label = paddle.nn.functional.label_smooth(
            label, epsilon=label_smoothing
        )
        label = label.astype(input.dtype)
        label_dims = len(list(label.shape))

    if in_dynamic_mode():
        if not soft_label:
            valid_label = (
                paddle.cast(label != ignore_index, dtype=label.dtype) * label
            )
        _, out = _C_ops.cross_entropy_with_softmax(
            input, label, soft_label, use_softmax, True, ignore_index, axis
        )

        if weight is not None:
            # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
            if soft_label:
                # chajchaj:
                # weight's shape is C, where C is class num.
                # for 1d case: label's shape is [N,C], weight_gather's shape is N.
                # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
                weight_gather = paddle.matmul(
                    x=paddle.cast(label, weight.dtype),
                    y=weight,
                    transpose_x=False,
                    transpose_y=True,
                )
                out_shape = list(out.shape)
                weight_gather_reshape = reshape(weight_gather, shape=out_shape)
                out = paddle.cast(out, weight_gather_reshape.dtype)

                out = _C_ops.multiply(out, weight_gather_reshape)
            else:
                if input.shape[axis] != weight.shape[-1]:
                    raise ValueError(
                        f"input's class_dimension({input.shape[axis]}) must equal to "
                        f"weight's class_dimension({weight.shape[-1]}) "
                        "when weight is provided"
                    )

                ignore_weight_mask = paddle.cast(
                    (label != ignore_index), out.dtype
                )
                if (
                    ignore_weight_mask.ndim > 1
                    and ignore_weight_mask.shape[axis] == 1
                ):
                    # TODO: Temporarily use squeeze instead of squeeze_
                    ignore_weight_mask = paddle.squeeze(
                        ignore_weight_mask, axis
                    )
                if axis != -1 and axis != valid_label.ndim - 1:
                    temp_perm = (
                        list(range(axis % valid_label.ndim))
                        + list(
                            range(
                                (axis % valid_label.ndim + 1), valid_label.ndim
                            )
                        )
                        + [axis % valid_label.ndim]
                    )
                    weight_gather = _C_ops.gather_nd(
                        weight, valid_label.transpose(temp_perm)
                    )
                else:
                    weight_gather = _C_ops.gather_nd(weight, valid_label)
                weight_gather = _C_ops.multiply(
                    weight_gather, ignore_weight_mask
                )
                input_shape = list(label.shape)
                weight_gather_reshape = reshape(
                    weight_gather, shape=input_shape
                )
                out = paddle.cast(out, weight_gather_reshape.dtype)
                out = _C_ops.multiply(out, weight_gather_reshape)

        if reduction == "sum":
            #   because of base_softmax_with_cross_entropy op's inner logic,
            #   in the out tensor of this op, the loss of sample with class_index==ignore_index is 0
            #   so, reduce_sum all directly is ok
            return _C_ops.sum(out, [], None, False)
        elif reduction == "mean":
            # 1. if weight==none,
            #     numerator: reduce_sum all loss directly is ok causeof base_softmax_with_cross_entropy's inner logic
            #     denominator: count sample num with class_index!=ignore_index
            # 2. else
            #     numerator: loss's weighted sum
            #     denominator: cal the sum of weight where the sample's class_index!=ignore_index
            if ignore_index >= 0:  # ignore label
                out_sum = _C_ops.sum(out, [], None, False)
                # for each label[i],set 1 or 0, according to ignore_index
                # mask[i]=0, if label[i]==ignore_index
                # mask[i]=1, otherwise
                mask = label != ignore_index
                if weight is None:
                    mask = paddle.cast(mask, dtype=out_sum.dtype)
                    count = _C_ops.sum(mask, [], None, False)
                    ret = out_sum / (count + (count == 0.0).astype(count.dtype))
                else:
                    mask = paddle.cast(mask, weight_gather_reshape.dtype)
                    weight_ignored = _C_ops.multiply(
                        mask, weight_gather_reshape
                    )
                    weight_sum = _C_ops.sum(weight_ignored, [], None, False)
                    ret = out_sum / (
                        weight_sum
                        + (weight_sum == 0.0).astype(weight_sum.dtype)
                    )
                return ret
            elif weight is not None:
                out_sum = _C_ops.sum(out, [], None, False)
                total_weight = _C_ops.sum(
                    weight_gather_reshape, [], None, False
                )
                return out_sum / (
                    total_weight
                    + (total_weight == 0.0).astype(total_weight.dtype)
                )
            else:
                return _C_ops.mean_all(out)

        else:
            if input_dims - 1 == label_dims:
                out = paddle.squeeze(out, axis=axis)
            return out

    else:
        check_variable_and_dtype(
            input,
            'input',
            ['uint16', 'float16', 'float32', 'float64'],
            'softmax_cross_entropy',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
            'softmax_cross_entropy',
        )
        if in_pir_mode():
            softmax, out = _C_ops.cross_entropy_with_softmax(
                input, label, soft_label, use_softmax, True, ignore_index, axis
            )
        else:
            attrs = {
                'soft_label': soft_label,
                'ignore_index': ignore_index,
                'numeric_stable_mode': True,
                'axis': axis,
                'use_softmax': use_softmax,
            }
            helper = LayerHelper('softmax_with_cross_entropy', **locals())
            softmax = helper.create_variable_for_type_inference(
                dtype=input.dtype
            )
            out = helper.create_variable_for_type_inference(dtype=input.dtype)

            outputs = {'Softmax': softmax, 'Loss': out}
            helper.append_op(
                type='softmax_with_cross_entropy',
                inputs={'Logits': input, 'Label': label},
                outputs=outputs,
                attrs=attrs,
            )

        if weight is not None:
            check_variable_and_dtype(
                weight,
                'weight',
                ['float32', 'float64'],
                'softmax_cross_entropy',
            )
            weight_name = name if reduction == 'none' else None
            if soft_label:
                # chajchaj:
                # trans weight from class to sample, shape:N or [N,H,W] for 1d and 2d cases.
                # weight's shape is C, where C is class num.
                # for 1d case: label's shape is [N,C], weight_gather's shape is N.
                # for 2d case: label's shape is [N,H,W,C], weight_gather's shape is [N,H,W].
                weight_gather = paddle.matmul(
                    x=paddle.cast(label, weight.dtype),
                    y=weight,
                    transpose_x=False,
                    transpose_y=True,
                )

                out_shape = list(out.shape)
                weight_gather_reshape = reshape(weight_gather, shape=out_shape)
                out = paddle.cast(out, weight_gather_reshape.dtype)
            else:
                if input.shape[axis] != weight.shape[-1]:
                    raise ValueError(
                        f"input's class_dimension({input.shape[axis]}) must equal to "
                        f"weight's class_dimension({weight.shape[-1]}) "
                        "when weight is provided"
                    )

                valid_label = paddle.multiply(
                    paddle.cast(label != ignore_index, dtype=label.dtype), label
                )
                ignore_weight_mask = paddle.cast(
                    (label != ignore_index), input.dtype
                )
                if (
                    ignore_weight_mask.ndim > 1
                    and ignore_weight_mask.shape[axis] == 1
                ):
                    ignore_weight_mask = paddle.squeeze(
                        ignore_weight_mask, axis
                    )
                if axis != -1 and axis != valid_label.ndim - 1:
                    temp_perm = (
                        list(range(axis % valid_label.ndim))
                        + list(
                            range(
                                (axis % valid_label.ndim + 1), valid_label.ndim
                            )
                        )
                        + [axis % valid_label.ndim]
                    )
                    weight_gather = paddle.gather_nd(
                        weight, paddle.transpose(valid_label, temp_perm)
                    )
                else:
                    weight_gather = paddle.gather_nd(weight, valid_label)
                weight_gather = paddle.multiply(
                    weight_gather, ignore_weight_mask
                )

                input_shape = list(label.shape)
                weight_gather_reshape = reshape(
                    weight_gather, shape=input_shape
                )
            out = paddle.multiply(out, weight_gather_reshape, name=weight_name)

        if reduction == "sum":
            return paddle.sum(out, name=name)
        elif reduction == "mean":
            if ignore_index >= 0:
                out_sum = paddle.sum(out, name=name)
                # for each label[i],set 1 or 0, according to ignore_index
                # mask[i]=0, if label[i]==ignore_index
                # mask[i]=1, otherwise
                mask = label != ignore_index
                if weight is None:
                    mask = paddle.cast(mask, dtype=out_sum.dtype)
                    count = paddle.sum(mask, name=name)
                    ret = out_sum / (count + paddle.equal(count, 0.0))
                else:
                    mask = paddle.cast(mask, weight_gather_reshape.dtype)
                    weight_ignored = paddle.multiply(
                        mask, weight_gather_reshape
                    )
                    weight_sum = paddle.sum(weight_ignored, name=name)
                    ret = out_sum / (weight_sum + paddle.equal(weight_sum, 0.0))
                return ret
            elif weight is not None:
                out_sum = paddle.sum(out, name=name)
                total_weight = paddle.sum(weight_gather_reshape)
                return out_sum / (
                    total_weight + paddle.equal(total_weight, 0.0)
                )
            else:
                return paddle.mean(out, name=name)

        else:
            if input_dims - 1 == label_dims:
                out = paddle.squeeze(out, axis=axis)

            return out


def sigmoid_focal_loss(
    logit: Tensor,
    label: Tensor,
    normalizer: Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: _ReduceMode = 'sum',
    name: str | None = None,
) -> Tensor:
    r"""
    `Focal Loss <https://arxiv.org/abs/1708.02002>`_ is proposed to address the
    foreground-background class imbalance for classification tasks. It down-weights
    easily-classified examples and thus focuses training on hard examples. For example,
    it is used in one-stage object detection where the foreground-background class
    imbalance is extremely high.

    This operator measures focal loss function as follows:

    .. math::
           Out = -Labels * alpha * {(1 - \sigma(Logit))}^{gamma}\log(\sigma(Logit)) - (1 - Labels) * (1 - alpha) * {\sigma(Logit)}^{gamma}\log(1 - \sigma(Logit))

    We know that :math:`\sigma(Logit) = \frac{1}{1 + \exp(-Logit)}`.

    Then, if :attr:`normalizer` is not None, this operator divides the
    normalizer tensor on the loss `Out`:

    .. math::
           Out = \frac{Out}{normalizer}

    Finally, this operator applies reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, the operator will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target ``label`` is 0 for the negative class and is 1 for the positive class.

    Args:
        logit (Tensor): The input logit tensor. The shape is [N, *], where N is batch_size,
            `*` means any number of additional dimensions. The ``logit`` is usually the
            output of a convolution layer. Available dtype is float32, float64.
        label (Tensor): The target label tensor with the same shape as
            ``logit``. The target label whose value should be numbers between 0 and 1.
            Available dtype is float32, float64.
        normalizer (Tensor, optional): The number normalizes the focal loss. It has to be
            a 1-D Tensor with shape `[1, ]` or 0-D Tensor with shape `[]`. The data type
            is float32, float64. For object detection task, it is the number of positive samples.
            If set to None, the focal loss will not be normalized. Default is None.
        alpha(int|float, optional): Hyper-parameter to balance the positive and negative example,
            it should be between 0 and 1.  Default value is set to 0.25.
        gamma(int|float, optional): Hyper-parameter to modulate the easy and hard examples.
            Default value is set to 2.0.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'sum'``.
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[]`, otherwise the shape is the same as ``logit``. The same dtype as ``logit`` tensor.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> logit = paddle.to_tensor([[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]], dtype='float32')
            >>> label = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')
            >>> one = paddle.to_tensor([1.], dtype='float32')
            >>> fg_label = paddle.greater_equal(label, one)
            >>> fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32'))
            >>> output = paddle.nn.functional.sigmoid_focal_loss(logit, label, normalizer=fg_num)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.65782464)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in sigmoid_focal_loss "
            f"should be 'sum', 'mean' or 'none', but received {reduction}, which is not allowed."
        )

    if normalizer is not None:
        check_variable_and_dtype(
            normalizer,
            'normalizer',
            ['float32', 'float64'],
            'sigmoid_focal_loss',
        )
        normalizer_shape = list(normalizer.shape)
        normalizer_dims = len(normalizer_shape)
        if normalizer_dims > 1:
            raise ValueError(
                f"Expected zero or one dimension of normalizer in sigmoid_focal_loss but got {normalizer_dims}."
            )

    if in_dynamic_or_pir_mode():
        place = _current_expected_place()
        one = _C_ops.full(paddle.shape(logit), 1.0, logit.dtype, place)

        loss = _C_ops.sigmoid_cross_entropy_with_logits(
            logit, label, None, False, -100
        )

        pred = _C_ops.sigmoid(logit)

        p_t = _C_ops.add(
            _C_ops.multiply(pred, label),
            _C_ops.multiply(
                _C_ops.subtract(one, pred), _C_ops.subtract(one, label)
            ),
        )

        alpha = paddle.to_tensor(alpha, dtype=loss.dtype)
        alpha_t = _C_ops.add(
            _C_ops.multiply(alpha, label),
            _C_ops.multiply(
                _C_ops.subtract(one, alpha), _C_ops.subtract(one, label)
            ),
        )
        loss = _C_ops.multiply(alpha_t, loss)

        if in_dynamic_mode():
            gamma = paddle.to_tensor(gamma, dtype=loss.dtype)
        gamma_t = _C_ops.pow(_C_ops.subtract(one, p_t), gamma)
        loss = _C_ops.multiply(gamma_t, loss)

        if normalizer is not None:
            loss = _C_ops.divide(loss, normalizer)

        if reduction == "sum":
            return _C_ops.sum(loss, [], None, False)
        elif reduction == "mean":
            return _C_ops.mean_all(loss)

        return loss

    else:
        check_variable_and_dtype(
            logit, 'logit', ['float32', 'float64'], 'sigmoid_focal_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'sigmoid_focal_loss'
        )

        bce_name = None
        if reduction == 'none' and normalizer is None:
            bce_name = name
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit, label, None, reduction='none', name=bce_name
        )

        pred = paddle.nn.functional.sigmoid(logit)
        p_t = pred * label + (1 - pred) * (1 - label)

        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = paddle.multiply(alpha_t, loss)

        gamma_t = paddle.pow((1 - p_t), gamma)
        loss = paddle.multiply(gamma_t, loss)

        if normalizer is not None:
            normalizer_name = name if reduction == 'none' else None
            loss = paddle.divide(loss, normalizer, name=normalizer_name)

        if reduction == 'mean':
            loss = paddle.mean(loss, name=name)
        elif reduction == 'sum':
            loss = paddle.sum(loss, name=name)

        return loss


def multi_label_soft_margin_loss(
    input: Tensor,
    label: Tensor,
    weight: Tensor | None = None,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
    Calculate a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = - \frac{1}{C} * \sum_i y[i] * \log((1 + \exp(-x[i]))^{-1})
            + (1-y[i]) * \log\left(\frac{\exp(-x[i])}{(1 + \exp(-x[i]))}\right)

    where :math:`i \in \left\{0, \; \cdots , \; \text{x.nElement}() - 1\right\}`,
    :math:`y[i] \in \left\{0, \; 1\right\}`.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes, and if shape is more than 2D, this is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor, the data type is float32 or float64. The shape of label is the same as the shape of input.
        weight (Tensor,optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of size C and the data type is float32, float64.
                Default is ``'None'`` .
        reduction (str, optional): Indicate how to average the loss by batch_size,
                the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
                If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
                If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
                If :attr:`reduction` is ``'sum'``, the summed loss is returned.
                Default: ``'mean'``
        name (str|None, optional): Name for the operation (optional, default is None).
                For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means number of classes, available dtype is float32, float64. The sum operation operates over all the elements.
        label: N-D Tensor, same shape as the input.
        weight:N-D Tensor, the shape is [N,1]
        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:
        Tensor, The tensor variable storing the multi_label_soft_margin_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> # label elements in {1., -1.}
            >>> label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)
            >>> loss = F.multi_label_soft_margin_loss(input, label, reduction='none')
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [3.49625897, 0.71111226, 0.43989015])
            >>> loss = F.multi_label_soft_margin_loss(input, label, reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   1.54908717)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'multi_label_soft_margin_loss' should be 'sum', 'mean' or 'none', "
            f"but received {reduction}."
        )

    if not (input.shape == label.shape):
        raise ValueError(
            "The input and label should have same dimension,"
            f"but received {input.shape}!={label.shape}"
        )

    if not in_dynamic_mode():
        check_variable_and_dtype(
            input,
            'input',
            ['float32', 'float64'],
            'multilabel_soft_margin_loss',
        )
        check_variable_and_dtype(
            label,
            'label',
            ['float32', 'float64'],
            'multilabel_soft_margin_loss',
        )

    loss = -(
        label * paddle.nn.functional.log_sigmoid(input)
        + (1 - label) * paddle.nn.functional.log_sigmoid(-input)
    )

    if weight is not None:
        if not in_dynamic_mode():
            check_variable_and_dtype(
                weight,
                'weight',
                ['float32', 'float64'],
                'multilabel_soft_margin_loss',
            )
        loss = loss * weight

    loss = loss.mean(axis=-1)  # only return N loss values

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return paddle.mean(loss)
    elif reduction == "sum":
        return paddle.sum(loss)


def hinge_embedding_loss(
    input: Tensor,
    label: Tensor,
    margin: float = 1.0,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
    Calculates hinge_embedding_loss. Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`(containing 1 or -1).
    This is usually used for measuring whether two inputs are similar or dissimilar, e.g. using the L1 pairwise distance as :math:`x`,
    and is typically used for learning nonlinear embeddings or semi-supervised learning.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64.
            the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.
        label (Tensor): Label tensor containing 1 or -1, the data type is float32 or float64.
            The shape of label is the same as the shape of input.
        margin (float, optional): Specifies the hyperparameter margin to be used.
            The value determines how large the input need to be to calculate in
            hinge_embedding_loss. When label is -1, Input smaller than margin are minimized with hinge_embedding_loss.
            Default = 1.0
        reduction (str, optional): Indicate how to average the loss by batch_size.
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:

        input: N-D Tensor, the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64. The sum operation operates over all the elements.

        label: N-D Tensor, same shape as the input. tensor elements should containing 1 or -1, the data type is float32 or float64.

        output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input.

    Returns:
        Tensor. The tensor variable storing the hinge_embedding_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
            >>> # label elements in {1., -1.}
            >>> label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

            >>> loss = F.hinge_embedding_loss(input, label, margin=1.0, reduction='none')
            >>> print(loss)
            Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[ 0., -2.,  0.],
                    [ 0., -1.,  2.],
                    [ 1.,  1.,  1.]])
            >>> loss = F.hinge_embedding_loss(input, label, margin=1.0, reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.22222222)

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'hinge_embedding_loss' should be 'sum', 'mean' or 'none', "
            f"but received {reduction}."
        )

    if not in_dynamic_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'hinge_embedding_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'hinge_embedding_loss'
        )

    zero_ = paddle.zeros([1], dtype=input.dtype)
    loss = paddle.where(label == 1.0, input, zero_) + paddle.where(
        label == -1.0, paddle.nn.functional.relu(margin - input), zero_
    )

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    label: Tensor,
    margin: float = 0,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
    Compute the cosine embedding loss of Tensor ``input1``, ``input2`` and ``label`` as follows.

    If label = 1, then the loss value can be calculated as follow:

    .. math::
        Out = 1 - cos(input1, input2)

    If label = -1, then the loss value can be calculated as follow:

    .. math::
        Out = max(0, cos(input1, input2)) - margin

    The operator cos can be described as follow:
     .. math::
        cos(x1, x2) = \frac{x1 \cdot{} x2}{\Vert x1 \Vert_2 * \Vert x2 \Vert_2}

    Parameters:
        input1 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        input2 (Tensor): tensor with shape: [N, M] or [M], 'N' means batch size, which can be 0, 'M' means the length of input array.
                         Available dtypes are float32, float64.
        label (Tensor): tensor with shape: [N] or [1], 'N' means the length of input array. The target labels values should be -1 or 1.
                         Available dtypes are int32, int64, float32, float64.
        margin (float, optional): Should be a number from :math:`-1` to :math:`1`,
                         :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the
                         default value is :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
                         ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                         ``'mean'``: the sum of the output will be divided by the number of elements in the output
                         ``'sum'``: the output will be summed.
        name (str, optional): Name for the operation (optional, default is None).
                         For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the cosine embedding Loss of Tensor ``input1`` ``input2`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [].

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input1 = paddle.to_tensor([[1.6, 1.2, -0.5], [3.2, 2.6, -5.8]], 'float32')
            >>> input2 = paddle.to_tensor([[0.5, 0.5, -1.8], [2.3, -1.4, 1.1]], 'float32')
            >>> label = paddle.to_tensor([1, -1], 'int64')

            >>> output = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
            >>> print(output)  # 0.21155193
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.21155193)
            >>> output = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='sum')
            >>> print(output)  # 0.42310387
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.42310387)
            >>> output = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='none')
            >>> print(output)  # [0.42310387, 0.        ]
            Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [0.42310387, 0.        ])

    """
    if len(label.shape) != 1:
        raise ValueError(
            "1D target tensor expected, multi-target not supported"
        )

    if input1.shape != input2.shape:
        raise ValueError(
            "the shape of input tensor 1 should be equal to input tensor 2, but found inputs with "
            "different sizes"
        )

    if len(input1.shape) > 2:
        raise ValueError(
            "1D target tensor expects 1D or 2D input tensors, but found inputs with different sizes"
        )

    if input1.dtype not in [paddle.float32, paddle.float64]:
        raise ValueError(
            "The data type of input Variable must be 'float32' or 'float64'"
        )
    if label.dtype not in [
        paddle.int32,
        paddle.int64,
        paddle.float32,
        paddle.float64,
    ]:
        raise ValueError(
            "The data type of label Variable must be 'int32', 'int64', 'float32', 'float64'"
        )

    prod_sum = (input1 * input2).sum(axis=-1)
    mag_square1 = paddle.square(input1).sum(axis=-1) + 10e-12
    mag_square2 = paddle.square(input2).sum(axis=-1) + 10e-12
    denom = paddle.sqrt(mag_square1 * mag_square2)
    cos = prod_sum / denom
    zeros = paddle.zeros_like(cos)
    pos = 1 - cos
    neg = paddle.clip(cos - margin, min=0)
    out_pos = paddle.where(label == 1, pos, zeros)
    out_neg = paddle.where(label == -1, neg, zeros)
    out = out_pos + out_neg

    if reduction == 'none':
        return out
    if reduction == 'mean':
        return paddle.mean(out, name=name)
    elif reduction == 'sum':
        return paddle.sum(out, name=name)


def triplet_margin_with_distance_loss(
    input: Tensor,
    positive: Tensor,
    negative: Tensor,
    distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
    Measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


    where the default distance function

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    or user can defined their own distance functions. `margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to be 0. If `swap` is true, it will compare distance of (input, negative) with
    distance of (negative, positive) and change it to the smaller one. For more details see http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf.

    Parameters:

        input (Tensor):Input tensor, the data type is float32 or float64.
            the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor):Positive tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        negative (Tensor):Negative tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        distance_function (callable|None, optional): Quantifies the distance between two tensors. if not specified, 2 norm functions will be used.

        margin (float, optional): A nonnegative margin representing the minimum difference
            between the positive and negative distances required for the loss to be 0. Default value is :math:`1`.

        swap (bool, optional):The distance swap changes the negative distance to the swap distance (distance between positive samples
                and negative samples) if swap distance smaller than negative distance. Default: ``False``.

        reduction (str, optional):Indicate how to average the loss by batch_size.
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``
        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output: Tensor. The tensor variable storing the triplet_margin_with_distance_loss of input and positive and negative.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> positive = paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            >>> negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            >>> loss = F.triplet_margin_with_distance_loss(input, positive, negative, margin=1.0, reduction='none')
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [0.        , 0.57496595, 0.        ])

            >>> loss = F.triplet_margin_with_distance_loss(input, positive, negative, margin=1.0, reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.19165532)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'triplet_margin_with_distance_loss' "
            "should be 'sum', 'mean' or 'none', "
            f"but received {reduction}."
        )
    if margin < 0:
        raise ValueError(
            "The margin between positive samples and negative samples should be greater than 0."
        )
    if not in_dynamic_mode():
        check_variable_and_dtype(
            input,
            'input',
            ['float32', 'float64'],
            'triplet_margin_with_distance_loss',
        )
        check_variable_and_dtype(
            positive,
            'positive',
            ['float32', 'float64'],
            'triplet_margin_with_distance_loss',
        )
        check_variable_and_dtype(
            negative,
            'negative',
            ['float32', 'float64'],
            'triplet_margin_with_distance_loss',
        )

    if not (input.shape == positive.shape == negative.shape):
        raise ValueError(
            "input's shape must equal to "
            "positive's shape and  "
            "negative's shape"
        )

    distance_function = (
        distance_function
        if distance_function is not None
        else paddle.nn.PairwiseDistance(2)
    )

    positive_dist = distance_function(input, positive)
    negative_dist = distance_function(input, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = paddle.minimum(negative_dist, swap_dist)

    if (
        not isinstance(positive_dist, paddle.pir.Value)
        and not paddle.all(positive_dist > 0)
    ) or (
        not isinstance(negative_dist, paddle.pir.Value)
        and not paddle.all(negative_dist > 0)
    ):
        raise ValueError(
            "The positive distance or negative distance should be greater than 0, "
            "The distance functions should be checked."
        )

    loss = paddle.clip(positive_dist - negative_dist + margin, min=0.0)

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def triplet_margin_loss(
    input: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    epsilon: float = 1e-06,
    swap: bool = False,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
        Measures the triplet loss given an input
        tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
        This is used for measuring a relative similarity between samples. A triplet
        is composed by `input`, `positive` and `negative` (i.e., `input`, `positive examples` and `negative
        examples` respectively). The shapes of all input tensors should be
        :math:`(N, *)`.

        The loss function for each sample in the mini-batch is:

        .. math::
            L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


        where

        .. math::
            d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64.
            the shape is [N, \*], N is batch size and `\*` means any number of additional dimensions, available dtype is float32, float64.

        positive (Tensor): Positive tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        negative (Tensor): Negative tensor, the data type is float32 or float64.
            The shape of label is the same as the shape of input.

        margin (float, optional): Default: :math:`1`.

        p (float, optional): The norm degree for pairwise distance. Default: :math:`2.0`.

        epsilon (float, optional): Add small value to avoid division by zero,
            default value is 1e-6.

        swap (bool, optional): The distance swap change the negative distance to the distance between
            positive sample and negative sample. For more details, see `Learning shallow convolutional feature descriptors with triplet losses`.
            Default: ``False``.


        reduction (str, optional):Indicate how to average the loss by batch_size.
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``

        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output: Tensor. The tensor variable storing the triplet_margin_loss of input and positive and negative.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> positive = paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
            >>> negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
            >>> loss = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='none')
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [0.        , 0.57496595, 0.        ])

            >>> loss = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.19165532)

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'triplet_margin_loss' should be 'sum', 'mean' or 'none', "
            f"but received {reduction}."
        )
    if margin < 0:
        raise ValueError(
            "The margin between positive samples and negative samples should be greater than 0."
        )
    if not in_dynamic_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'triplet_margin_loss'
        )
        check_variable_and_dtype(
            positive, 'positive', ['float32', 'float64'], 'triplet_margin_loss'
        )
        check_variable_and_dtype(
            negative, 'negative', ['float32', 'float64'], 'triplet_margin_loss'
        )

    if not (input.shape == positive.shape == negative.shape):
        raise ValueError(
            "input's shape must equal to "
            "positive's shape and  "
            "negative's shape"
        )

    distance_function = paddle.nn.PairwiseDistance(p, epsilon=epsilon)
    positive_dist = distance_function(input, positive)
    negative_dist = distance_function(input, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = paddle.minimum(negative_dist, swap_dist)

    loss = paddle.clip(positive_dist - negative_dist + margin, min=0.0)

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def multi_margin_loss(
    input: Tensor,
    label: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Tensor | None = None,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""
        Measures a multi-class classification hinge loss between input :math:`input` and label :math:`label`:

        For i-th mini-batch sample, the loss in terms of the 1D input :math:`input_i` and scalar
        output :math:`label_i` is:

        .. math::
            \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, \text{margin} - input_i[label_i] + input_i[j])^p}{\text{C}}

        where :math:`0 \leq j \leq \text{C}-1`, :math:`0 \leq i \leq \text{N}-1` and :math:`j \neq label_i`.

        Optionally, you can give non-equal weighting on the classes by passing
        a 1D :attr:`weight` tensor into the constructor.

        The loss function for i-th sample then becomes:

        .. math::
            \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, weight[label_i] * (\text{margin} - input_i[label_i] + input_i[j]))^p}{\text{C}}


    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is (N, C), where C is number of classes.

        label (Tensor): Label tensor, the data type is int32 or int64. The shape of label is (N,)

        p (int, optional): The power num. Default: :math:`1`.

        margin (float, optional): Default: :math:`1`.

        weight (Tensor|None, optional): a manual rescaling weight given to each class.
                If given, has to be a Tensor of shape (C,) and the data type is float32, float64.
                Default is ``'None'`` .


        reduction (str, optional):Indicate how to calculate the loss by batch_size.
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default: ``'mean'``

        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Output: Tensor. The tensor variable storing the multi_margin_loss of input and label.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
            >>> label = paddle.to_tensor([1, 2, 1], dtype=paddle.int32)
            >>> loss = F.multi_margin_loss(input, label, margin=1.0, reduction='none')
            >>> print(loss)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [0.        , 0.66666663, 0.        ])

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'multi_margin_loss' should be 'sum', 'mean' or 'none', "
            f"but received {reduction}."
        )

    if not in_dynamic_mode():
        check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'multi_margin_loss'
        )
        check_variable_and_dtype(
            label, 'label', ['int32', 'int64'], 'multi_margin_loss'
        )
    if not (input.shape[0] == label.shape[0]):
        raise ValueError(
            "The label's shape[0] should be equal to input's shape[0], "
            f"but received input's shape[0] {input.shape[0]} and label's shape[0]:{label.shape[0]}. "
        )
    label = label.reshape((-1, 1))
    index_sample = paddle.index_sample(input, label)
    if weight is not None:
        if not in_dynamic_mode():
            check_variable_and_dtype(
                weight, 'weight', ['float32', 'float64'], 'multi_margin_loss'
            )
        if not (input.shape[1] == weight.shape[0]):
            raise ValueError(
                "The weight's shape[0] should be equal to input's shape[1]"
                f"but received weight's shape[0]: {weight.shape[0]} and input's shape[1]: {input.shape[1]}"
            )
        weight = paddle.gather(weight, label, axis=0).reshape((-1, 1))
        loss = paddle.mean(
            paddle.pow(
                paddle.clip(weight * (margin - index_sample + input), min=0.0),
                p,
            ),
            axis=1,
        ) - weight * (margin**p / paddle.shape(input)[1])
    else:
        loss = (
            paddle.mean(
                paddle.pow(
                    paddle.clip(margin - index_sample + input, min=0.0), p
                ),
                axis=1,
            )
            - margin**p / paddle.shape(input)[1]
        )

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def soft_margin_loss(
    input: Tensor,
    label: Tensor,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    """

    The API measures the soft margin loss between input predictions ``input``
    and target labels ``label`` . It can be described as:

    .. math::
        Out = log(1 + exp((-label * input)))

    Parameters:

        input (Tensor): The input predications tensor with shape: ``[N, *]``,
            N is batch_size, `*` means any number of additional dimensions. The ``input`` ranges from -inf to inf.
            Available dtype is float32, float64.

        label (Tensor): The target labels tensor with the same shape as
            ``input``. The target labels which values should be numbers -1 or 1.
            Available dtype is int32, int64, float32, float64.

        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candidates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.

        name (str|None, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:

        Output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``input`` , else the shape of output is [].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> input = paddle.to_tensor([[0.5, 0.6, 0.7],[0.3, 0.5, 0.2]], 'float32')
            >>> label = paddle.to_tensor([[1.0, -1.0, 1.0],[-1.0, 1.0, 1.0]], 'float32')
            >>> output = paddle.nn.functional.soft_margin_loss(input, label)
            >>> print(output)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.64022040)

            >>> input = paddle.uniform(shape=(5, 5), dtype="float32", min=0.1, max=0.8)
            >>> label = paddle.randint(0, 2, shape=(5, 5), dtype="int64")
            >>> label[label==0] = -1

            >>> output = paddle.nn.functional.soft_margin_loss(input, label, reduction='none')
            >>> print(output)
            Tensor(shape=[5, 5], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[1.10725629, 0.48778144, 0.56217247, 1.12581408, 0.51430041],
                    [0.90375793, 0.37761253, 0.43007556, 0.95089805, 0.43288314],
                    [1.16043591, 0.63015938, 0.51362717, 0.43617544, 0.57783306],
                    [0.81927848, 0.52558368, 0.59713912, 0.83100700, 0.50811619],
                    [0.82684207, 1.02064908, 0.50296998, 1.13461733, 0.93222517]])

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in soft_margin_loss should be 'sum', "
            f"'mean' or 'none', but received {reduction}, which is not allowed."
        )

    if not in_dynamic_mode():
        base.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'soft_margin_loss'
        )
        base.data_feeder.check_variable_and_dtype(
            label,
            'label',
            ['int32', 'int64', 'float32', 'float64'],
            'soft_margin_loss',
        )

    if not (input.shape == label.shape):
        raise ValueError("input's shape must equal to " "label's shape")

    label = paddle.cast(label, input.dtype)
    out = paddle.log(1 + paddle.exp(-label * input))

    if reduction == 'sum':
        return paddle.sum(out, name=name)
    elif reduction == 'mean':
        return paddle.mean(out, name=name)
    else:
        return out


def gaussian_nll_loss(
    input: Tensor,
    label: Tensor,
    variance: Tensor,
    full: bool = False,
    epsilon: float = 1e-06,
    reduction: _ReduceMode = 'mean',
    name: str | None = None,
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    Gaussian negative log likelihood loss among ``input``, ``variance`` and
    ``label``. Note that the ``label`` is treated as samples from Gaussian distributions.
    This function is used to train a neural network predicts
    the ``input`` and ``variance`` of a gaussian distribution that ``label`` are supposed to
    be coming from. This means ``input`` and ``variance`` should be functions(the neural network) of some inputs.

    For a ``label`` having Gaussian distribution with ``input`` and ``variance`` predicted by neural network
    the loss is calculated as follows:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{epsilon}\right)\right) + \frac{\left(\text{input} - \text{label}\right)^2}
        {\text{max}\left(\text{var}, \ \text{epsilon}\right)}\right) + \text{const.}

    where :attr:`epsilon` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``variance`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        input (Tensor): input tensor, :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of additional
            dimensions. Expectation of the Gaussian distribution, available dtype is float32, float64.
        label (Tensor): target label tensor, :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input
            but with one dimension equal to 1 (to allow for broadcasting). Sample from the Gaussian distribution, available dtype is float32, float64.
        variance (Tensor): tensor of positive variance(s), :math:`(N, *)` or :math:`(*)`, same shape as the input, or same shape as the input but
            with one dimension equal to 1, or same shape as the input but with one fewer
            dimension (to allow for broadcasting). One for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic), available dtype is float32, float64.
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        epsilon (float, optional): value used to clamp ``variance`` (see note below), for
            stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.
        name (str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:

        output (Tensor): If ``reduction`` is ``'none'``, the shape of output is same as ``input`` , else the shape of output is [].

    Examples::
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F
            >>> paddle.seed(2023)

            >>> input = paddle.randn([5, 2], dtype=paddle.float32)
            >>> label = paddle.randn([5, 2], dtype=paddle.float32)
            >>> variance = paddle.ones([5, 2], dtype=paddle.float32)

            >>> loss = F.gaussian_nll_loss(input, label, variance, reduction='none')
            >>> print(loss)
            Tensor(shape=[5, 2], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0.21808575, 1.43013096],
                    [1.05245590, 0.00394560],
                    [1.20861185, 0.00000062],
                    [0.56946373, 0.73300570],
                    [0.37142906, 0.12038800]])

            >>> loss = F.gaussian_nll_loss(input, label, variance, reduction='mean')
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                   0.57075173)

    Note:
        The clamping of ``variance`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.
    """

    # Check variance shape
    # If variance.shape == input.shape, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if variance.shape != input.shape:
        # If variance is one dimension short of input, but the shape match otherwise, then this is a homoscedastic case.
        # e.g. input.shape = (10, 2, 3), variance.shape = (10, 2)
        # -> unsqueeze variance so that variance.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.shape[:-1] == variance.shape:
            variance = paddle.unsqueeze(variance, -1)
        # This checks if the shape match up to the final dimension, and the final dimension of variance is of shape 1.
        # This is also a homoscedastic case.
        # e.g. input.shape = (10, 2, 3), variance.shape = (10, 2, 1)
        elif (
            input.shape[:-1] == variance.shape[:-1] and variance.shape[-1] == 1
        ):  # Heteroscedastic case
            pass
        # If none of the above pass, then the shape of variance is incorrect.
        else:
            raise ValueError("variance is of incorrect shape")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    check_variable_and_dtype(
        input,
        'Input',
        ['float32', 'float64'],
        'gaussian_nll_loss',
    )
    check_variable_and_dtype(
        label,
        'Label',
        ['float32', 'float64'],
        'gaussian_nll_loss',
    )
    check_variable_and_dtype(
        variance,
        'Variance',
        ['float32', 'float64'],
        'gaussian_nll_loss',
    )
    # Entries of variance must be non-negative
    if not in_dynamic_mode():
        condition = paddle.all(variance > 0)
        Assert(condition, [variance], 6)
    else:
        if input.dtype not in [paddle.float32, paddle.float64]:
            raise ValueError(
                "The data type of input Variable must be 'float32' or 'float64'"
            )
        if label.dtype not in [
            paddle.float32,
            paddle.float64,
        ]:
            raise ValueError(
                "The data type of label Variable must be 'float32', 'float64'"
            )
        if variance.dtype not in [paddle.float32, paddle.float64]:
            raise ValueError(
                "The data type of variance Variable must be 'float32', 'float64'"
            )
        if paddle.any(variance < 0):
            raise ValueError("variance has negative entry/entries")

    # Clamp for stability
    variance = variance.clone()
    with paddle.no_grad():
        variance = paddle.clip(variance, min=epsilon)
    # Calculate the loss
    loss = 0.5 * (
        paddle.log(variance) + paddle.square(input - label) / variance
    )
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return paddle.mean(loss, name=name)
    elif reduction == 'sum':
        return paddle.sum(loss, name=name)
    elif reduction == 'none':
        return loss


def adaptive_log_softmax_with_loss(
    input: Tensor,
    label: Tensor,
    head_weight: Tensor,
    tail_weights: Sequence[Sequence[Tensor]],
    cutoffs: Sequence[int | Tensor],
    head_bias: Tensor | None = None,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Compute adaptive logsoftmax result and negative log likelihood between ``input`` and ``label``.
    Parameter ``head``, ``tail_weights``, ``cutoffs`` are inner members of AdaptiveLogSoftmaxWithLoss
    Please refer to :ref:`api_paddle_nn_AdaptiveLogSoftmaxWithLoss`.

    Args:
        input (Tensor): Input tensor, the data type should be float32 or float64.
        label (Tensor): Label tensor, the data type should be float32 or float64.
        head_weight (Tensor): weight tensor for linear computation, the data type should be float32 or float64, the shape should be ``[input.shape[1], shortlist_size + n_clusters]``, where ``shortlist_size`` is the first element in the cutoffs list, and ``n_clusters`` is the length of the cutoffs list minus 1.
        tail_weights (list|tuple): weight tensor list or tuple for linear computation, the data type should be float32 or float64. The number of elements in the tail_weights depends on the value of the n_clusters, and each element contains the weights of two linear layers, their dimensions are ``[input.shape[1], hsz]`` and ``[hsz, osz]``, where ``hsz`` is the number of input features in_features divided by div_value to the power ``(i + 1)``, where i is the cyclic variable, from ``0`` to ``n_clusters - 1``, and ``osz`` is the ``(i + 1)`` The difference between the cutoff and the ith cutoff.
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets.
        head_bias (Tensor|None, optional): bias tensor for linear computation, the data type should be float32 or float64. Default: ``None``.
        name (str|None, optional): Name for the operation (optional, default is ``None``). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - output (Tensor). The tensor storing adaptive logsoftmax result, the shape of output is ``[N]``
        - loss (Tensor). The tensor variable storing the adaptive_log_softmax_loss of input and label.

    Examples:
        .. code-block:: python

            >>> from typing import List
            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.seed(2024)
            >>> input = paddle.randn([3, 5], dtype=paddle.float32)
            >>> head_weight = paddle.randn([5, 3], dtype=paddle.float32)
            >>> head_bias = paddle.randn([3], dtype=paddle.float32)
            >>> tail_weights: List[List[paddle.Tensor]] = [[]]
            >>> tail_weights[0].append(paddle.randn([5, 2], dtype=paddle.float32))
            >>> tail_weights[0].append(paddle.randn([2, 1], dtype=paddle.float32))
            >>> out, loss = F.adaptive_log_softmax_with_loss(
            ...     input,
            ...     paddle.full([3], 1, dtype='int64'),
            ...     head_weight,
            ...     tail_weights,
            ...     cutoffs=[2],
            ...     head_bias=head_bias
            ... )
            >>> print(out)
            Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [-0.99842924, -2.27753878, -0.16740258])
            >>> print(loss)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            1.14779019)
    """
    target_dim = label.dim()

    if target_dim == 1:
        if input.shape[0] != label.shape[0]:
            raise ValueError(
                'Input and label should have the same size '
                'in the batch dimension.'
            )
        if input.dim() != 2:
            raise ValueError(
                '1D label tensor expects 2D input tensors, '
                f'but found inputs with size {input.shape}'
            )
    elif target_dim == 0:
        if input.dim() != 1:
            raise ValueError(
                '0D label tensor expects 1D input tensors, '
                f'but found inputs with size {input.shape}'
            )
    else:
        raise ValueError(
            '0D or 1D label tensor expected, ' 'multi-label not supported'
        )

    is_batched = target_dim > 0
    input = input if is_batched else input.unsqueeze(0)
    label = label if is_batched else label.unsqueeze(0)

    used_rows = 0
    batch_size = label.shape[0]

    output = paddle.zeros([batch_size], dtype=input.dtype)
    gather_inds = paddle.empty([batch_size], dtype=label.dtype)

    cutoff_values = [0, *cutoffs]
    for i in range(len(cutoff_values) - 1):
        low_idx = cutoff_values[i]
        high_idx = cutoff_values[i + 1]

        label_mask = (label >= low_idx) & (label < high_idx)
        row_indices = label_mask.nonzero().squeeze()

        if row_indices.numel() == 0:
            continue

        if i == 0:
            scatter_output = paddle.scatter_nd(
                row_indices.unsqueeze(1),
                label.masked_select(label_mask),
                gather_inds.shape,
            )
            gather_inds = scatter_output
        else:
            relative_label = label[label_mask] - low_idx
            input_subset = input.index_select(row_indices, axis=0)

            cluster_output = paddle.nn.functional.linear(
                x=input_subset, weight=tail_weights[i - 1][0]
            )
            cluster_output = paddle.nn.functional.linear(
                x=cluster_output, weight=tail_weights[i - 1][1]
            )

            cluster_index = cutoffs[0] + i - 1

            gather_inds = paddle.index_fill(
                gather_inds, row_indices, 0, cluster_index
            )

            cluster_logprob = paddle.nn.functional.log_softmax(
                cluster_output, axis=1
            )

            local_logprob = paddle.take_along_axis(
                cluster_logprob, relative_label.unsqueeze(1), axis=1
            )
            scatter_output = paddle.scatter_nd(
                row_indices.unsqueeze(1), local_logprob.squeeze(1), output.shape
            )
            output = (
                output * (scatter_output == 0).astype('float32')
                + scatter_output
            )

        used_rows += row_indices.numel()

    if used_rows != batch_size:
        raise ValueError(
            f"label values should be in [0, n_classes - 1], "
            f"but values in range [{label.min().item()}, {label.max().item()}] "
            "were found. "
        )

    head_output = paddle.nn.functional.linear(
        x=input, weight=head_weight, bias=head_bias
    )
    head_logprob = paddle.nn.functional.log_softmax(head_output, axis=1)
    output += paddle.take_along_axis(
        head_logprob, gather_inds.unsqueeze(1), axis=1
    ).squeeze()
    loss = (-output).mean()

    if not is_batched:
        output = output.squeeze(0)

    return output, loss
