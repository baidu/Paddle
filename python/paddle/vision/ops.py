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

from typing import TYPE_CHECKING, Callable, Literal, overload

import numpy as np

import paddle
from paddle import _C_ops
from paddle.tensor.math import _add_with_axis
from paddle.utils import convert_to_list

from ..base import core
from ..base.data_feeder import check_type, check_variable_and_dtype
from ..base.framework import (
    Variable,
    convert_np_dtype_to_dtype_,
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
)
from ..base.layer_helper import LayerHelper
from ..framework import _current_expected_place
from ..nn import BatchNorm2D, Conv2D, Layer, ReLU, Sequential
from ..nn.initializer import Normal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle import Tensor, nn
    from paddle._typing import ParamAttrLike, Size2, Size4
    from paddle.nn.functional.common import _PaddingSizeMode

__all__ = [
    'yolo_loss',
    'yolo_box',
    'prior_box',
    'box_coder',
    'deform_conv2d',
    'DeformConv2D',
    'distribute_fpn_proposals',
    'generate_proposals',
    'read_file',
    'decode_jpeg',
    'roi_pool',
    'RoIPool',
    'psroi_pool',
    'PSRoIPool',
    'roi_align',
    'RoIAlign',
    'nms',
    'matrix_nms',
]


def yolo_loss(
    x: Tensor,
    gt_box: Tensor,
    gt_label: Tensor,
    anchors: Sequence[int],
    anchor_mask: Sequence[int],
    class_num: int,
    ignore_thresh: float,
    downsample_ratio: int,
    gt_score: Tensor | None = None,
    use_label_smooth: bool = True,
    name: str | None = None,
    scale_x_y: float = 1.0,
) -> Tensor:
    r"""

    This operator generates YOLOv3 loss based on given predict result and ground
    truth boxes.

    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict
    given number bounding boxes, this given number, which following will be represented as S,
    is specified by the number of anchor clusters in each scale. In the second dimension(the channel
    dimension), C should be equal to S * (class_num + 5), class_num is the object
    category number of source dataset(such as 80 in coco dataset), so in the
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h,
    also includes confidence score of the box and class one-hot key of each anchor box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box predictions
    should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    In the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    As for confidence score, it is the logistic regression value of IoU between
    anchor boxes and ground truth boxes, the score of the anchor box which has
    the max IoU should be 1, and if the anchor box has IoU bigger than ignore
    thresh, the confidence score loss of this anchor box will be ignored.

    Therefore, the YOLOv3 loss consists of three major parts: box location loss,
    objectness loss and classification loss. The L1 loss is used for
    box coordinates (w, h), sigmoid cross entropy loss is used for box
    coordinates (x, y), objectness loss and classification loss.

    Each ground truth box finds a best matching anchor box in all anchors.
    Prediction of this anchor box will incur all three parts of losses, and
    prediction of anchor boxes with no GT box matched will only incur objectness
    loss.

    In order to trade off box coordinate losses between big boxes and small
    boxes, box coordinate losses will be multiplied by scale weight, which is
    calculated as follows.

    $$
    weight_{box} = 2.0 - t_w * t_h
    $$

    Final loss will be represented as follows.

    $$
    loss = (loss_{xy} + loss_{wh}) * weight_{box} + loss_{conf} + loss_{class}
    $$

    While :attr:`use_label_smooth` is set to be :attr:`True`, the classification
    target will be smoothed when calculating classification loss, target of
    positive samples will be smoothed to :math:`1.0 - 1.0 / class\_num` and target of
    negative samples will be smoothed to :math:`1.0 / class\_num`.

    While :attr:`gt_score` is given, which means the mixup score of ground truth
    boxes, all losses incurred by a ground truth box will be multiplied by its
    mixup score.

    Args:
        x (Tensor): The input tensor of YOLOv3 loss operator, This is a 4-D
                      tensor with shape of [N, C, H, W]. H and W should be same,
                      and the second dimension(C) stores box locations, confidence
                      score and classification one-hot keys of each anchor box.
                      The data type is float32 or float64.
        gt_box (Tensor): ground truth boxes, should be in shape of [N, B, 4],
                          in the third dimension, x, y, w, h should be stored.
                          x,y is the center coordinate of boxes, w, h are the
                          width and height, x, y, w, h should be divided by
                          input image height to scale to [0, 1].
                          N is the batch number and B is the max box number in
                          an image.The data type is float32 or float64.
        gt_label (Tensor): class id of ground truth boxes, should be in shape
                            of [N, B].The data type is int32.
        anchors (list|tuple): The anchor width and height, it will be parsed
                              pair by pair.
        anchor_mask (list|tuple): The mask index of anchors used in current
                                  YOLOv3 loss calculation.
        class_num (int): The number of classes.
        ignore_thresh (float): The ignore threshold to ignore confidence loss.
        downsample_ratio (int): The downsample ratio from network input to YOLOv3
                                loss input, so 32, 16, 8 should be set for the
                                first, second, and third YOLOv3 loss operators.
        gt_score (Tensor|None, optional): mixup score of ground truth boxes, should be in shape
                            of [N, B]. Default None.
        use_label_smooth (bool, optional): Whether to use label smooth. Default True.
        name (str|None, optional): The default value is None. Normally there is no need
                       for user to set this property. For more information,
                       please refer to :ref:`api_guide_Name`
        scale_x_y (float, optional): Scale the center point of decoded bounding box.
                           Default 1.0.

    Returns:
        Tensor: A 1-D tensor with shape [N], the value of yolov3 loss

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.rand([2, 14, 8, 8]).astype('float32')
            >>> gt_box = paddle.rand([2, 10, 4]).astype('float32')
            >>> gt_label = paddle.rand([2, 10]).astype('int32')
            >>> loss = paddle.vision.ops.yolo_loss(x,
            ...                                    gt_box=gt_box,
            ...                                    gt_label=gt_label,
            ...                                    anchors=[10, 13, 16, 30],
            ...                                    anchor_mask=[0, 1],
            ...                                    class_num=2,
            ...                                    ignore_thresh=0.7,
            ...                                    downsample_ratio=8,
            ...                                    use_label_smooth=True,
            ...                                    scale_x_y=1.)
    """

    if in_dynamic_or_pir_mode():
        loss = _C_ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            gt_score,
            anchors,
            anchor_mask,
            class_num,
            ignore_thresh,
            downsample_ratio,
            use_label_smooth,
            scale_x_y,
        )
        return loss

    else:
        helper = LayerHelper('yolov3_loss', **locals())

        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'yolo_loss')
        check_variable_and_dtype(
            gt_box, 'gt_box', ['float32', 'float64'], 'yolo_loss'
        )
        check_variable_and_dtype(gt_label, 'gt_label', 'int32', 'yolo_loss')
        check_type(anchors, 'anchors', (list, tuple), 'yolo_loss')
        check_type(anchor_mask, 'anchor_mask', (list, tuple), 'yolo_loss')
        check_type(class_num, 'class_num', int, 'yolo_loss')
        check_type(ignore_thresh, 'ignore_thresh', float, 'yolo_loss')
        check_type(use_label_smooth, 'use_label_smooth', bool, 'yolo_loss')

        loss = helper.create_variable_for_type_inference(dtype=x.dtype)

        objectness_mask = helper.create_variable_for_type_inference(
            dtype='int32'
        )
        gt_match_mask = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {
            "X": x,
            "GTBox": gt_box,
            "GTLabel": gt_label,
        }
        if gt_score is not None:
            inputs["GTScore"] = gt_score

        attrs = {
            "anchors": anchors,
            "anchor_mask": anchor_mask,
            "class_num": class_num,
            "ignore_thresh": ignore_thresh,
            "downsample_ratio": downsample_ratio,
            "use_label_smooth": use_label_smooth,
            "scale_x_y": scale_x_y,
        }

        helper.append_op(
            type='yolov3_loss',
            inputs=inputs,
            outputs={
                'Loss': loss,
                'ObjectnessMask': objectness_mask,
                'GTMatchMask': gt_match_mask,
            },
            attrs=attrs,
        )
        return loss


def yolo_box(
    x: Tensor,
    img_size: Tensor,
    anchors: Sequence[int],
    class_num: int,
    conf_thresh: float,
    downsample_ratio: int,
    clip_bbox: bool = True,
    name: str | None = None,
    scale_x_y: float = 1.0,
    iou_aware: bool = False,
    iou_aware_factor: float = 0.5,
) -> Tensor:
    r"""

    This operator generates YOLO detection boxes from output of YOLOv3 network.

    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict
    given number boxes, this given number, which following will be represented as S,
    is specified by the number of anchors. In the second dimension(the channel
    dimension), C should be equal to S * (5 + class_num) if :attr:`iou_aware` is false,
    otherwise C should be equal to S * (6 + class_num). class_num is the object
    category number of source dataset(such as 80 in coco dataset), so the
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h,
    also includes confidence score of the box and class one-hot key of each anchor
    box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box
    predictions should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    in the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    The logistic regression value of the 5th channel of each anchor prediction boxes
    represents the confidence score of each prediction box, and the logistic
    regression value of the last :attr:`class_num` channels of each anchor prediction
    boxes represents the classification scores. Boxes with confidence scores less than
    :attr:`conf_thresh` should be ignored, and box final scores is the product of
    confidence scores and classification scores.

    $$
    score_{pred} = score_{conf} * score_{class}
    $$


    Args:
        x (Tensor): The input tensor of YoloBox operator is a 4-D tensor with
                      shape of [N, C, H, W]. The second dimension(C) stores box
                      locations, confidence score and classification one-hot keys
                      of each anchor box. Generally, X should be the output of
                      YOLOv3 network. The data type is float32 or float64.
        img_size (Tensor): The image size tensor of YoloBox operator, This is a
                           2-D tensor with shape of [N, 2]. This tensor holds
                           height and width of each input image used for resizing
                           output box in input image scale. The data type is int32.
        anchors (list|tuple): The anchor width and height, it will be parsed pair
                              by pair.
        class_num (int): The number of classes.
        conf_thresh (float): The confidence scores threshold of detection boxes.
                             Boxes with confidence scores under threshold should
                             be ignored.
        downsample_ratio (int): The downsample ratio from network input to
                                :attr:`yolo_box` operator input, so 32, 16, 8
                                should be set for the first, second, and third
                                :attr:`yolo_box` layer.
        clip_bbox (bool, optional): Whether clip output bonding box in :attr:`img_size`
                          boundary. Default true.
        name (str|None, optional): The default value is None. Normally there is no need
                       for user to set this property. For more information,
                       please refer to :ref:`api_guide_Name`.
        scale_x_y (float, optional): Scale the center point of decoded bounding box. Default 1.0
        iou_aware (bool, optional): Whether use iou aware. Default false.
        iou_aware_factor (float, optional): iou aware factor. Default 0.5.

    Returns:
        Tensor: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,
        and a 3-D tensor with shape [N, M, :attr:`class_num`], the classification
        scores of boxes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = paddle.rand([2, 14, 8, 8]).astype('float32')
            >>> img_size = paddle.ones((2, 2)).astype('int32')
            >>> boxes, scores = paddle.vision.ops.yolo_box(x,
            ...                                             img_size=img_size,
            ...                                             anchors=[10, 13, 16, 30],
            ...                                             class_num=2,
            ...                                             conf_thresh=0.01,
            ...                                             downsample_ratio=8,
            ...                                             clip_bbox=True,
            ...                                             scale_x_y=1.)
    """
    if in_dynamic_or_pir_mode():
        boxes, scores = _C_ops.yolo_box(
            x,
            img_size,
            anchors,
            class_num,
            conf_thresh,
            downsample_ratio,
            clip_bbox,
            scale_x_y,
            iou_aware,
            iou_aware_factor,
        )
        return boxes, scores

    else:
        helper = LayerHelper('yolo_box', **locals())

        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'yolo_box')
        check_variable_and_dtype(img_size, 'img_size', 'int32', 'yolo_box')
        check_type(anchors, 'anchors', (list, tuple), 'yolo_box')
        check_type(conf_thresh, 'conf_thresh', float, 'yolo_box')

        boxes = helper.create_variable_for_type_inference(dtype=x.dtype)
        scores = helper.create_variable_for_type_inference(dtype=x.dtype)

        attrs = {
            "anchors": anchors,
            "class_num": class_num,
            "conf_thresh": conf_thresh,
            "downsample_ratio": downsample_ratio,
            "clip_bbox": clip_bbox,
            "scale_x_y": scale_x_y,
            "iou_aware": iou_aware,
            "iou_aware_factor": iou_aware_factor,
        }

        helper.append_op(
            type='yolo_box',
            inputs={
                "X": x,
                "ImgSize": img_size,
            },
            outputs={
                'Boxes': boxes,
                'Scores': scores,
            },
            attrs=attrs,
        )
        return boxes, scores


def prior_box(
    input: Tensor,
    image: Tensor,
    min_sizes: Sequence[float] | float,
    max_sizes: Sequence[float] | float | None = None,
    aspect_ratios: Sequence[float] | float = [1.0],
    variance: Sequence[float] | float = [0.1, 0.1, 0.2, 0.2],
    flip: bool = False,
    clip: bool = False,
    steps: Sequence[float] | float = [0.0, 0.0],
    offset: float = 0.5,
    min_max_aspect_ratios_order: bool = False,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""

    This op generates prior boxes for SSD(Single Shot MultiBox Detector) algorithm.

    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Args:
       input (Tensor): 4-D tensor(NCHW), the data type should be float32 or float64.
       image (Tensor): 4-D tensor(NCHW), the input image data of PriorBoxOp,
            the data type should be float32 or float64.
       min_sizes (list|tuple|float): the min sizes of generated prior boxes.
       max_sizes (list|tuple|float|None, optional): the max sizes of generated prior boxes.
            Default: None, means [] and will not be used.
       aspect_ratios (list|tuple|float, optional): the aspect ratios of generated
            prior boxes. Default: [1.0].
       variance (list|tuple|float, optional): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip (bool): Whether to flip aspect ratios. Default:False.
       clip (bool): Whether to clip out-of-boundary boxes. Default: False.
       steps (list|tuple|float, optional): Prior boxes steps across width and height, If
            steps[0] equals to 0.0 or steps[1] equals to 0.0, the prior boxes steps across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset (float, optional)): Prior boxes center offset. Default: 0.5
       min_max_aspect_ratios_order (bool, optional): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.
       name (str, optional): The default value is None. Normally there is no need for
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: the output prior boxes and the expanded variances of PriorBox.
            The prior boxes is a 4-D tensor, the layout is [H, W, num_priors, 4],
            num_priors is the total box count of each position of input.
            The expanded variances is a 4-D tensor, same shape as the prior boxes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> input = paddle.rand((1, 3, 6, 9), dtype=paddle.float32)
            >>> image = paddle.rand((1, 3, 9, 12), dtype=paddle.float32)
            >>> box, var = paddle.vision.ops.prior_box(
            ...     input=input,
            ...     image=image,
            ...     min_sizes=[2.0, 4.0],
            ...     clip=True,
            ...     flip=True)
            ...
    """

    def _is_list_or_tuple_(data):
        return isinstance(data, (list, tuple))

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not _is_list_or_tuple_(steps):
        steps = [steps]
    if not len(steps) == 2:
        raise ValueError('steps should be (step_w, step_h)')

    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))

    cur_max_sizes = None
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        cur_max_sizes = max_sizes

    if in_dynamic_or_pir_mode():
        step_w, step_h = steps
        if max_sizes is None:
            max_sizes = []
        box, var = _C_ops.prior_box(
            input,
            image,
            min_sizes,
            max_sizes,
            aspect_ratios,
            variance,
            flip,
            clip,
            step_w,
            step_h,
            offset,
            min_max_aspect_ratios_order,
        )
        return box, var

    else:
        helper = LayerHelper("prior_box", **locals())
        dtype = helper.input_dtype()
        check_variable_and_dtype(
            input, 'input', ['uint8', 'int8', 'float32', 'float64'], 'prior_box'
        )
        attrs = {
            'min_sizes': min_sizes,
            'aspect_ratios': aspect_ratios,
            'variances': variance,
            'flip': flip,
            'clip': clip,
            'step_w': steps[0],
            'step_h': steps[1],
            'offset': offset,
            'min_max_aspect_ratios_order': min_max_aspect_ratios_order,
        }
        if cur_max_sizes is not None:
            attrs['max_sizes'] = cur_max_sizes

        box = helper.create_variable_for_type_inference(dtype)
        var = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="prior_box",
            inputs={"Input": input, "Image": image},
            outputs={"Boxes": box, "Variances": var},
            attrs=attrs,
        )
        box.stop_gradient = True
        var.stop_gradient = True
        return box, var


def box_coder(
    prior_box: Tensor,
    prior_box_var: Tensor | Sequence[float],
    target_box: Tensor,
    code_type: Literal[
        "encode_center_size", "decode_center_size"
    ] = "encode_center_size",
    box_normalized: bool = True,
    axis: int = 0,
    name: str | None = None,
) -> Tensor:
    r"""
    Encode/Decode the target bounding box with the priorbox information.

    The Encoding schema described below:

    .. math::

        ox &= (tx - px) / pw / pxv

        oy &= (ty - py) / ph / pyv

        ow &= log(abs(tw / pw)) / pwv

        oh &= log(abs(th / ph)) / phv

    The Decoding schema described below:

    .. math::

        ox &= (pw * pxv * tx * + px) - tw / 2

        oy &= (ph * pyv * ty * + py) - th / 2

        ow &= exp(pwv * tw) * pw + tw / 2

        oh &= exp(phv * th) * ph + th / 2

    where `tx`, `ty`, `tw`, `th` denote the target box's center coordinates,
    width and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote
    the priorbox's (anchor) center coordinates, width and height. `pxv`,
    `pyv`, `pwv`, `phv` denote the variance of the priorbox and `ox`, `oy`,
    `ow`, `oh` denote the encoded/decoded coordinates, width and height.
    During Box Decoding, two modes for broadcast are supported. Say target
    box has shape [N, M, 4], and the shape of prior box can be [N, 4] or
    [M, 4]. Then prior box will broadcast to target box along the
    assigned axis.

    Args:
        prior_box (Tensor): Box list prior_box is a 2-D Tensor with shape
            [M, 4] holds M boxes and data type is float32 or float64. Each box
            is represented as [xmin, ymin, xmax, ymax], [xmin, ymin] is the
            left top coordinate of the anchor box, if the input is image feature
            map, they are close to the origin of the coordinate system.
            [xmax, ymax] is the right bottom coordinate of the anchor box.
        prior_box_var (Tensor|List|tuple|None): prior_box_var supports four types
            of input. One is Tensor with shape [M, 4] which holds M group and
            data type is float32 or float64. The second is list or tuple consist
            of 4 elements shared by all boxes and data type is float32 or float64.
            Other is None and not involved in calculation.
        target_box (Tensor): This input can be a 2-D DenseTensor with shape
            [N, 4] when code_type is 'encode_center_size'. This input also can
            be a 3-D Tensor with shape [N, M, 4] when code_type is
            'decode_center_size'. Each box is represented as
            [xmin, ymin, xmax, ymax]. The data type is float32 or float64.
        code_type (str, optional): The code type used with the target box. It can be
            `encode_center_size` or `decode_center_size`. `encode_center_size`
            by default.
        box_normalized (bool, optional): Whether treat the priorbox as a normalized box.
            Set true by default.
        axis (int, optional): Which axis in PriorBox to broadcast for box decode,
            for example, if axis is 0 and TargetBox has shape [N, M, 4] and
            PriorBox has shape [M, 4], then PriorBox will broadcast to [N, M, 4]
            for decoding. It is only valid when code type is
            `decode_center_size`. Set 0 by default.
        name (str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor: output boxes, when code_type is 'encode_center_size', the
            output tensor of box_coder_op with shape [N, M, 4] representing the
            result of N target boxes encoded with M Prior boxes and variances.
            When code_type is 'decode_center_size', N represents the batch size
            and M represents the number of decoded boxes.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # For encode
            >>> prior_box_encode = paddle.rand((80, 4), dtype=paddle.float32)
            >>> prior_box_var_encode = paddle.rand((80, 4), dtype=paddle.float32)
            >>> target_box_encode = paddle.rand((20, 4), dtype=paddle.float32)
            >>> output_encode = paddle.vision.ops.box_coder(
            ...     prior_box=prior_box_encode,
            ...     prior_box_var=prior_box_var_encode,
            ...     target_box=target_box_encode,
            ...     code_type="encode_center_size")
            ...
            >>> # For decode
            >>> prior_box_decode = paddle.rand((80, 4), dtype=paddle.float32)
            >>> prior_box_var_decode = paddle.rand((80, 4), dtype=paddle.float32)
            >>> target_box_decode = paddle.rand((20, 80, 4), dtype=paddle.float32)
            >>> output_decode = paddle.vision.ops.box_coder(
            ...     prior_box=prior_box_decode,
            ...     prior_box_var=prior_box_var_decode,
            ...     target_box=target_box_decode,
            ...     code_type="decode_center_size",
            ...     box_normalized=False)
            ...
    """
    if in_dynamic_or_pir_mode():
        if isinstance(prior_box_var, (core.eager.Tensor, paddle.pir.Value)):
            output_box = _C_ops.box_coder(
                prior_box,
                prior_box_var,
                target_box,
                code_type,
                box_normalized,
                axis,
                [],
            )
        elif isinstance(prior_box_var, (list, tuple)):
            prior_box_var = list(prior_box_var)
            assert (
                len(prior_box_var) == 4
            ), "Input prior_box_var must be Variable or list|tuple with 4 elements."
            output_box = _C_ops.box_coder(
                prior_box,
                None,
                target_box,
                code_type,
                box_normalized,
                axis,
                prior_box_var,
            )
        else:
            raise TypeError(
                "Input prior_box_var must be Variable or list|tuple"
            )
        return output_box
    else:
        check_variable_and_dtype(
            prior_box, 'prior_box', ['float32', 'float64'], 'box_coder'
        )
        check_variable_and_dtype(
            target_box, 'target_box', ['float32', 'float64'], 'box_coder'
        )
        helper = LayerHelper("box_coder", **locals())

        output_box = helper.create_variable_for_type_inference(
            dtype=prior_box.dtype
        )

        inputs = {"PriorBox": prior_box, "TargetBox": target_box}
        attrs = {
            "code_type": code_type,
            "box_normalized": box_normalized,
            "axis": axis,
        }
        if isinstance(prior_box_var, Variable):
            inputs['PriorBoxVar'] = prior_box_var
        elif isinstance(prior_box_var, (list, tuple)):
            attrs['variance'] = prior_box_var
            assert (
                len(attrs['variance']) == 4
            ), "Input prior_box_var must be Variable or list|tuple with 4 elements."
        else:
            raise TypeError(
                "Input prior_box_var must be Variable or list|tuple"
            )
        helper.append_op(
            type="box_coder",
            inputs=inputs,
            attrs=attrs,
            outputs={"OutputBox": output_box},
        )
        return output_box


def deform_conv2d(
    x: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: Size2 = 1,
    padding: Size2 = 0,
    dilation: Size2 = 1,
    deformable_groups: int = 1,
    groups: int = 1,
    mask: Tensor | None = None,
    name: str | None = None,
) -> Tensor:
    r"""
    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          x shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          weight shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          offset shape: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

          mask shape: :math:`(N, H_f * W_f, H_{out}, W_{out})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, kH, kW], where M is
            the number of output channels, g is the number of groups, kH is the filter's
            height, kW is the filter's width.
        bias (Tensor, optional): The bias with shape [M,]. Default: None.
        stride (int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        padding (int|list|tuple, optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        dilation (int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        deformable_groups (int): The number of deformable group partitions.
            Default: 1.
        groups (int, optional): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        mask (Tensor, optional): The input mask of deformable convolution layer.
            A Tensor with type float32, float64. It should be None when you use
            deformable convolution v1. Default: None.
        name(str|None, optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Tensor: 4-D Tensor storing the deformable convolution result.\
            A Tensor with type float32, float64.

    Examples:
        .. code-block:: python

            >>> #deformable conv v2:

            >>> import paddle
            >>> input = paddle.rand((8, 1, 28, 28))
            >>> kh, kw = 3, 3
            >>> weight = paddle.rand((16, 1, kh, kw))
            >>> # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
            >>> # mask shape should be [bs, hw * hw, out_h, out_w]
            >>> # In this case, for an input of 28, stride of 1
            >>> # and kernel size of 3, without padding, the output size is 26
            >>> offset = paddle.rand((8, 2 * kh * kw, 26, 26))
            >>> mask = paddle.rand((8, kh * kw, 26, 26))
            >>> out = paddle.vision.ops.deform_conv2d(input, offset, weight, mask=mask)
            >>> print(out.shape)
            [8, 16, 26, 26]

            >>> #deformable conv v1:

            >>> import paddle
            >>> input = paddle.rand((8, 1, 28, 28))
            >>> kh, kw = 3, 3
            >>> weight = paddle.rand((16, 1, kh, kw))
            >>> # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
            >>> # In this case, for an input of 28, stride of 1
            >>> # and kernel size of 3, without padding, the output size is 26
            >>> offset = paddle.rand((8, 2 * kh * kw, 26, 26))
            >>> out = paddle.vision.ops.deform_conv2d(input, offset, weight)
            >>> print(out.shape)
            [8, 16, 26, 26]
    """
    stride = convert_to_list(stride, 2, 'stride')
    padding = convert_to_list(padding, 2, 'padding')
    dilation = convert_to_list(dilation, 2, 'dilation')

    use_deform_conv2d_v1 = True if mask is None else False

    if in_dynamic_or_pir_mode():
        pre_bias = _C_ops.deformable_conv(
            x,
            offset,
            weight,
            mask,
            stride,
            padding,
            dilation,
            deformable_groups,
            groups,
            1,
        )
        if bias is not None:
            out = _add_with_axis(pre_bias, bias, axis=1)
        else:
            out = pre_bias
    else:
        check_variable_and_dtype(
            x, "x", ['float32', 'float64'], 'deform_conv2d'
        )
        check_variable_and_dtype(
            offset, "offset", ['float32', 'float64'], 'deform_conv2d'
        )

        num_channels = x.shape[1]

        helper = LayerHelper('deformable_conv', **locals())
        dtype = helper.input_dtype()

        stride = convert_to_list(stride, 2, 'stride')
        padding = convert_to_list(padding, 2, 'padding')
        dilation = convert_to_list(dilation, 2, 'dilation')

        pre_bias = helper.create_variable_for_type_inference(dtype)

        if use_deform_conv2d_v1:
            op_type = 'deformable_conv_v1'
            inputs = {
                'Input': x,
                'Filter': weight,
                'Offset': offset,
            }
        else:
            op_type = 'deformable_conv'
            inputs = {
                'Input': x,
                'Filter': weight,
                'Offset': offset,
                'Mask': mask,
            }

        outputs = {"Output": pre_bias}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'deformable_groups': deformable_groups,
            'im2col_step': 1,
        }
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs
        )

        if bias is not None:
            out = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias], 'Y': [bias]},
                outputs={'Out': [out]},
                attrs={'axis': 1},
            )
        else:
            out = pre_bias
    return out


class DeformConv2D(Layer):
    r"""
    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          x shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          weight shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          offset shape: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

          mask shape: :math:`(N, H_f * W_f, H_{out}, W_{out})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1


    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. The default value is 1.
        padding (int|list|tuple, optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        deformable_groups (int, optional): The number of deformable group partitions.
            Default: deformable_groups = 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        weight_attr(ParamAttr|None, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool|None, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Shape:
        - x: :math:`(N, C_{in}, H_{in}, W_{in})`
        - offset: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`
        - mask: :math:`(N, H_f * W_f, H_{out}, W_{out})`
        - output: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        ..  math::

            H_{out}&= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1 \\
            W_{out}&= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1

    Examples:
        .. code-block:: python

            >>> #deformable conv v2:
            >>> import paddle
            >>> input = paddle.rand((8, 1, 28, 28))
            >>> kh, kw = 3, 3
            >>> # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
            >>> # mask shape should be [bs, hw * hw, out_h, out_w]
            >>> # In this case, for an input of 28, stride of 1
            >>> # and kernel size of 3, without padding, the output size is 26
            >>> offset = paddle.rand((8, 2 * kh * kw, 26, 26))
            >>> mask = paddle.rand((8, kh * kw, 26, 26))
            >>> deform_conv = paddle.vision.ops.DeformConv2D(
            ...     in_channels=1,
            ...     out_channels=16,
            ...     kernel_size=[kh, kw])
            >>> out = deform_conv(input, offset, mask)
            >>> print(out.shape)
            [8, 16, 26, 26]

            >>> #deformable conv v1:
            >>> import paddle
            >>> input = paddle.rand((8, 1, 28, 28))
            >>> kh, kw = 3, 3
            >>> # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
            >>> # mask shape should be [bs, hw * hw, out_h, out_w]
            >>> # In this case, for an input of 28, stride of 1
            >>> # and kernel size of 3, without padding, the output size is 26
            >>> offset = paddle.rand((8, 2 * kh * kw, 26, 26))
            >>> deform_conv = paddle.vision.ops.DeformConv2D(
            ...     in_channels=1,
            ...     out_channels=16,
            ...     kernel_size=[kh, kw])
            >>> out = deform_conv(input, offset)
            >>> print(out.shape)
            [8, 16, 26, 26]
    """

    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2,
        stride: Size2 = 1,
        padding: Size2 = 0,
        dilation: Size2 = 1,
        deformable_groups: int = 1,
        groups: int = 1,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
    ) -> None:
        super().__init__()
        assert (
            weight_attr is not False
        ), "weight_attr should not be False in Conv."
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._deformable_groups = deformable_groups
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channel_dim = 1

        self._stride = convert_to_list(stride, 2, 'stride')
        self._dilation = convert_to_list(dilation, 2, 'dilation')
        self._kernel_size = convert_to_list(kernel_size, 2, 'kernel_size')

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")

        self._padding = convert_to_list(padding, 2, 'padding')

        filter_shape = [out_channels, in_channels // groups, *self._kernel_size]

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num) ** 0.5
            return Normal(0.0, std)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._weight_attr,
            default_initializer=_get_default_param_initializer(),
        )
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True
        )

    def forward(
        self, x: Tensor, offset: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        out = deform_conv2d(
            x=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            deformable_groups=self._deformable_groups,
            groups=self._groups,
            mask=mask,
        )
        return out


@overload
def distribute_fpn_proposals(
    fpn_rois: Tensor,
    min_level: int,
    max_level: int,
    refer_level: int,
    refer_scale: int,
    pixel_offset: bool = ...,
    rois_num: None = ...,
    name: str | None = ...,
) -> tuple[list[Tensor], Tensor, None]: ...


@overload
def distribute_fpn_proposals(
    fpn_rois: Tensor,
    min_level: int,
    max_level: int,
    refer_level: int,
    refer_scale: int,
    pixel_offset: bool = ...,
    rois_num: Tensor = ...,
    name: str | None = ...,
) -> tuple[list[Tensor], Tensor, list[Tensor]]: ...


def distribute_fpn_proposals(
    fpn_rois,
    min_level,
    max_level,
    refer_level,
    refer_scale,
    pixel_offset=False,
    rois_num=None,
    name=None,
):
    r"""

    In Feature Pyramid Networks (FPN) models, it is needed to distribute
    all proposals into different FPN level, with respect to scale of the proposals,
    the referring scale and the referring level. Besides, to restore the order of
    proposals, we return an array which indicates the original index of rois
    in current proposals. To compute FPN level for each roi, the formula is given as follows:

    .. math::
        roi\_scale &= \sqrt{BBoxArea(fpn\_roi)} \\
        level &= floor(\log(\frac{roi\_scale}{refer\_scale}) + refer\_level)

    where BBoxArea is a function to compute the area of each roi.

    Args:
        fpn_rois (Tensor): The input fpn_rois. 2-D Tensor with shape [N, 4] and data type can be
            float32 or float64.
        min_level (int): The lowest level of FPN layer where the proposals come
            from.
        max_level (int): The highest level of FPN layer where the proposals
            come from.
        refer_level (int): The referring level of FPN layer with specified scale.
        refer_scale (int): The referring scale of FPN layer with specified level.
        pixel_offset (bool, optional): Whether there is pixel offset. If True, the offset of
            image shape will be 1. 'False' by default.
        rois_num (Tensor|None, optional): 1-D Tensor contains the number of RoIs in each image.
            The shape is [B] and data type is int32. B is the number of images.
            If rois_num not None, it will return a list of 1-D Tensor. Each element
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name (str|None, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        - multi_rois (List), The proposals in each FPN level. It is a list of 2-D Tensor with shape [M, 4], where M is
          and data type is same as `fpn_rois` . The length is max_level-min_level+1.
        - restore_ind (Tensor), The index used to restore the order of fpn_rois. It is a 2-D Tensor with shape [N, 1]
          , where N is the number of total rois. The data type is int32.
        - rois_num_per_level (List), A list of 1-D Tensor and each Tensor is
          the RoIs' number in each image on the corresponding level. The shape
          is [B] and data type of int32, where B is the number of images.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> fpn_rois = paddle.rand((10, 4))
            >>> rois_num = paddle.to_tensor([3, 1, 4, 2], dtype=paddle.int32)
            >>> multi_rois, restore_ind, rois_num_per_level = paddle.vision.ops.distribute_fpn_proposals(
            ...     fpn_rois=fpn_rois,
            ...     min_level=2,
            ...     max_level=5,
            ...     refer_level=4,
            ...     refer_scale=224,
            ...     rois_num=rois_num)
            ...
    """
    assert (
        max_level > 0 and min_level > 0
    ), "min_level and max_level should be greater than 0"

    num_lvl = max_level - min_level + 1
    assert num_lvl > 1, "max_level should be greater than min_level"
    assert (
        num_lvl < 100
    ), "Only support max to 100 levels, (max_level - min_level + 1 < 100)"

    if in_dynamic_or_pir_mode():
        assert (
            rois_num is not None
        ), "rois_num should not be None in dygraph mode."
        (
            multi_rois,
            rois_num_per_level,
            restore_ind,
        ) = _C_ops.distribute_fpn_proposals(
            fpn_rois,
            rois_num,
            min_level,
            max_level,
            refer_level,
            refer_scale,
            pixel_offset,
        )
        return multi_rois, restore_ind, rois_num_per_level

    else:
        check_variable_and_dtype(
            fpn_rois,
            'fpn_rois',
            ['float32', 'float64'],
            'distribute_fpn_proposals',
        )
        helper = LayerHelper('distribute_fpn_proposals', **locals())
        dtype = helper.input_dtype('fpn_rois')
        multi_rois = [
            helper.create_variable_for_type_inference(dtype)
            for i in range(num_lvl)
        ]

        restore_ind = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {'FpnRois': fpn_rois}
        outputs = {
            'MultiFpnRois': multi_rois,
            'RestoreIndex': restore_ind,
        }

        if rois_num is not None:
            inputs['RoisNum'] = rois_num
            rois_num_per_level = [
                helper.create_variable_for_type_inference(dtype='int32')
                for i in range(num_lvl)
            ]
            outputs['MultiLevelRoIsNum'] = rois_num_per_level
        else:
            rois_num_per_level = None

        helper.append_op(
            type='distribute_fpn_proposals',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'min_level': min_level,
                'max_level': max_level,
                'refer_level': refer_level,
                'refer_scale': refer_scale,
                'pixel_offset': pixel_offset,
            },
        )
        return multi_rois, restore_ind, rois_num_per_level


def read_file(filename: str, name: str | None = None) -> Tensor:
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Args:
        filename (str): Path of the file to be read.
        name (str|None, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        A uint8 tensor.

    Examples:
        .. code-block:: python

            >>> import cv2
            >>> import paddle
            >>> paddle.seed(2023)
            >>> fake_img = (paddle.rand((400, 300, 3)).numpy() * 255).astype('uint8')
            >>> cv2.imwrite('fake.jpg', fake_img)
            >>> img_bytes = paddle.vision.ops.read_file('fake.jpg')
            >>> print(img_bytes.shape)
            [142773]
    """

    attr_dtype = convert_np_dtype_to_dtype_('uint8')
    if in_dynamic_or_pir_mode():
        return _C_ops.read_file(filename, attr_dtype, paddle.CPUPlace())
    else:
        inputs = {}
        attrs = {'filename': filename}

        helper = LayerHelper("read_file", **locals())
        out = helper.create_variable_for_type_inference('uint8')
        helper.append_op(
            type="read_file", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )

        return out


def decode_jpeg(
    x: Tensor,
    mode: Literal["unchanged", "gray", "rgb"] = "unchanged",
    name: str | None = None,
) -> Tensor:
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor or 1 dimensional Gray Tensor.
    Optionally converts the image to the desired format.
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        x (Tensor): A one dimensional uint8 tensor containing the raw bytes
            of the JPEG image.
        mode (str, optional): The read mode used for optionally converting the image. Must be one of
            ["unchanged", "gray", "rgb"]. Default: 'unchanged'.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: A decoded image tensor with shape (image_channels, image_height, image_width)

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:GPU)
            >>> import cv2
            >>> import numpy as np
            >>> import paddle
            >>> paddle.device.set_device('gpu')

            >>> fake_img = (np.random.random(
            ...             (400, 300, 3)) * 255).astype('uint8')
            >>> cv2.imwrite('fake.jpg', fake_img)
            >>> img_bytes = paddle.vision.ops.read_file('fake.jpg')
            >>> img = paddle.vision.ops.decode_jpeg(img_bytes)
            >>> print(img.shape)
            [3, 400, 300]
    """
    if in_dynamic_or_pir_mode():
        return _C_ops.decode_jpeg(x, mode, _current_expected_place())
    else:
        inputs = {'X': x}
        attrs = {"mode": mode}

        helper = LayerHelper("decode_jpeg", **locals())
        out = helper.create_variable_for_type_inference('uint8')
        helper.append_op(
            type="decode_jpeg", inputs=inputs, attrs=attrs, outputs={"Out": out}
        )

        return out


def psroi_pool(
    x: Tensor,
    boxes: Tensor,
    boxes_num: Tensor,
    output_size: Size2,
    spatial_scale: float = 1.0,
    name: str | None = None,
) -> Tensor:
    """
    Position sensitive region of interest pooling (also known as PSROIPooling) is to perform
    position-sensitive average pooling on regions of interest specified by input. It performs
    on inputs of nonuniform sizes to obtain fixed-size feature maps.

    PSROIPooling is proposed by R-FCN. Please refer to https://arxiv.org/abs/1605.06409 for more details.

    Args:
        x (Tensor): Input features with shape (N, C, H, W). The data type can be float32 or float64.
        boxes (Tensor): Box coordinates of ROIs (Regions of Interest) to pool over. It should be
                         a 2-D Tensor with shape (num_rois, 4). Given as [[x1, y1, x2, y2], ...],
                         (x1, y1) is the top left coordinates, and (x2, y2) is the bottom
                         right coordinates.
        boxes_num (Tensor): The number of boxes contained in each picture in the batch.
        output_size (int|Tuple(int, int))  The pooled output size(H, W), data type
                               is int32. If int, H and W are both equal to output_size.
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate ROI coords from their
                               input scale to the scale used when pooling. Default: 1.0
        name(str|None, optional): The default value is None.
                             Normally there is no need for user to set this property.
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        4-D Tensor. The pooled ROIs with shape (num_rois, output_channels, pooled_h, pooled_w).
        The output_channels equal to C / (pooled_h * pooled_w), where C is the channels of input.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> x = paddle.uniform([2, 490, 28, 28], dtype='float32')
            >>> boxes = paddle.to_tensor([[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32')
            >>> boxes_num = paddle.to_tensor([1, 2], dtype='int32')
            >>> pool_out = paddle.vision.ops.psroi_pool(x, boxes, boxes_num, 7, 1.0)
            >>> print(pool_out.shape)
            [3, 10, 7, 7]
    """

    check_type(output_size, 'output_size', (int, tuple, list), 'psroi_pool')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    pooled_height, pooled_width = output_size
    assert len(x.shape) == 4, "Input features with shape should be (N, C, H, W)"
    if pooled_height * pooled_width == 0:
        raise ValueError('output_size should not contain 0.')
    output_channels = int(x.shape[1] / (pooled_height * pooled_width))
    if in_dynamic_or_pir_mode():
        return _C_ops.psroi_pool(
            x,
            boxes,
            boxes_num,
            pooled_height,
            pooled_width,
            output_channels,
            spatial_scale,
        )
    else:
        helper = LayerHelper('psroi_pool', **locals())
        dtype = helper.input_dtype()
        out = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='psroi_pool',
            inputs={'X': x, 'ROIs': boxes},
            outputs={'Out': out},
            attrs={
                'output_channels': output_channels,
                'spatial_scale': spatial_scale,
                'pooled_height': pooled_height,
                'pooled_width': pooled_width,
            },
        )
        return out


class PSRoIPool(Layer):
    """
    This interface is used to construct a callable object of the ``PSRoIPool`` class. Please
    refer to :ref:`api_paddle_vision_ops_psroi_pool`.

    Args:
        output_size (int|Tuple(int, int))  The pooled output size(H, W), data type
                               is int32. If int, H and W are both equal to output_size.
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate ROI coords from their
                               input scale to the scale used when pooling. Default: 1.0.

    Shape:
        - x: 4-D Tensor with shape (N, C, H, W).
        - boxes: 2-D Tensor with shape (num_rois, 4).
        - boxes_num: 1-D Tensor.
        - output: 4-D tensor with shape (num_rois, output_channels, pooled_h, pooled_w).
              The output_channels equal to C / (pooled_h * pooled_w), where C is the channels of input.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> psroi_module = paddle.vision.ops.PSRoIPool(7, 1.0)
            >>> x = paddle.uniform([2, 490, 28, 28], dtype='float32')
            >>> boxes = paddle.to_tensor([[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]], dtype='float32')
            >>> boxes_num = paddle.to_tensor([1, 2], dtype='int32')
            >>> pool_out = psroi_module(x, boxes, boxes_num)
            >>> print(pool_out.shape)
            [3, 10, 7, 7]
    """

    output_size: Size2
    spatial_scale: float

    def __init__(self, output_size: Size2, spatial_scale: float = 1.0) -> None:
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, x: Tensor, boxes: Tensor, boxes_num: Tensor) -> Tensor:
        return psroi_pool(
            x, boxes, boxes_num, self.output_size, self.spatial_scale
        )


def roi_pool(
    x: Tensor,
    boxes: Tensor,
    boxes_num: Tensor,
    output_size: Size2,
    spatial_scale: float = 1.0,
    name: str | None = None,
) -> Tensor:
    """
    This operator implements the roi_pooling layer.
    Region of interest pooling (also known as RoI pooling) is to perform max pooling on inputs of nonuniform sizes to obtain fixed-size feature maps (e.g. 7*7).
    The operator has three steps: 1. Dividing each region proposal into equal-sized sections with output_size(h, w) 2. Finding the largest value in each section 3. Copying these max values to the output buffer
    For more information, please refer to https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn.

    Args:
        x (Tensor): input feature, 4D-Tensor with the shape of [N,C,H,W],
            where N is the batch size, C is the input channel, H is Height, W is weight.
            The data type is float32 or float64.
        boxes (Tensor): boxes (Regions of Interest) to pool over.
            2D-Tensor with the shape of [num_boxes,4].
            Given as [[x1, y1, x2, y2], ...], (x1, y1) is the top left coordinates,
            and (x2, y2) is the bottom right coordinates.
        boxes_num (Tensor): the number of RoIs in each image, data type is int32.
        output_size (int or tuple[int, int]): the pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0.
        name(str, optional): for detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default. Default: None.

    Returns:
        pool_out (Tensor): the pooled feature, 4D-Tensor with the shape of [num_boxes, C, output_size[0], output_size[1]].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.ops import roi_pool

            >>> data = paddle.rand([1, 256, 32, 32])
            >>> boxes = paddle.rand([3, 4])
            >>> boxes[:, 2] += boxes[:, 0] + 3
            >>> boxes[:, 3] += boxes[:, 1] + 4
            >>> boxes_num = paddle.to_tensor([3]).astype('int32')
            >>> pool_out = roi_pool(data, boxes, boxes_num=boxes_num, output_size=3)
            >>> print(pool_out.shape)
            [3, 256, 3, 3]
    """

    check_type(output_size, 'output_size', (int, tuple), 'roi_pool')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dynamic_or_pir_mode():
        assert (
            boxes_num is not None
        ), "boxes_num should not be None in dygraph mode."
        return _C_ops.roi_pool(
            x, boxes, boxes_num, pooled_height, pooled_width, spatial_scale
        )
    else:
        check_variable_and_dtype(x, 'x', ['float32'], 'roi_pool')
        check_variable_and_dtype(boxes, 'boxes', ['float32'], 'roi_pool')
        helper = LayerHelper('roi_pool', **locals())
        dtype = helper.input_dtype()
        pool_out = helper.create_variable_for_type_inference(dtype)
        argmaxes = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {
            "X": x,
            "ROIs": boxes,
        }
        if boxes_num is not None:
            inputs['RoisNum'] = boxes_num
        helper.append_op(
            type="roi_pool",
            inputs=inputs,
            outputs={"Out": pool_out, "Argmax": argmaxes},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale,
            },
        )
        return pool_out


class RoIPool(Layer):
    """
    This interface is used to construct a callable object of the `RoIPool` class. Please
    refer to :ref:`api_paddle_vision_ops_roi_pool`.

    Args:
        output_size (int or tuple[int, int]): the pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0.

    Returns:
        pool_out (Tensor): the pooled feature, 4D-Tensor with the shape of [num_boxes, C, output_size[0], output_size[1]].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.ops import RoIPool

            >>> data = paddle.rand([1, 256, 32, 32])
            >>> boxes = paddle.rand([3, 4])
            >>> boxes[:, 2] += boxes[:, 0] + 3
            >>> boxes[:, 3] += boxes[:, 1] + 4
            >>> boxes_num = paddle.to_tensor([3]).astype('int32')
            >>> roi_pool = RoIPool(output_size=(4, 3))
            >>> pool_out = roi_pool(data, boxes, boxes_num)
            >>> print(pool_out.shape)
            [3, 256, 4, 3]
    """

    def __init__(self, output_size: Size2, spatial_scale: float = 1.0) -> None:
        super().__init__()
        self._output_size = output_size
        self._spatial_scale = spatial_scale

    def forward(self, x: Tensor, boxes: Tensor, boxes_num: Tensor) -> Tensor:
        return roi_pool(
            x=x,
            boxes=boxes,
            boxes_num=boxes_num,
            output_size=self._output_size,
            spatial_scale=self._spatial_scale,
        )

    def extra_repr(self) -> str:
        main_str = 'output_size={_output_size}, spatial_scale={_spatial_scale}'
        return main_str.format(**self.__dict__)


def roi_align(
    x: Tensor,
    boxes: Tensor,
    boxes_num: Tensor,
    output_size: Size2,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = True,
    name: str | None = None,
) -> Tensor:
    """
    Implementing the roi_align layer.
    Region of Interest (RoI) Align operator (also known as RoI Align) is to
    perform bilinear interpolation on inputs of nonuniform sizes to obtain
    fixed-size feature maps (e.g. 7*7), as described in Mask R-CNN.

    Dividing each region proposal into equal-sized sections with the pooled_width
    and pooled_height. Location remains the origin result.

    In each ROI bin, the value of the four regularly sampled locations are
    computed directly through bilinear interpolation. The output is the mean of
    four locations. Thus avoid the misaligned problem.

    Args:
        x (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W],
            where N is the batch size, C is the input channel, H is Height,
            W is weight. The data type is float32 or float64.
        boxes (Tensor): Boxes (RoIs, Regions of Interest) to pool over. It
            should be a 2-D Tensor of shape (num_boxes, 4). The data type is
            float32 or float64. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
            the top left coordinates, and (x2, y2) is the bottom right coordinates.
        boxes_num (Tensor): The number of boxes contained in each picture in
            the batch, the data type is int32.
        output_size (int or Tuple[int, int]): The pooled output size(h, w), data
            type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate
            ROI coords from their input scale to the scale used when pooling.
            Default: 1.0.
        sampling_ratio (int, optional): number of sampling points in the interpolation
            grid used to compute the output value of each pooled output bin.
            If > 0, then exactly ``sampling_ratio x sampling_ratio`` sampling
            points per bin are used.
            If <= 0, then an adaptive number of grid points are used (computed
            as ``ceil(roi_width / output_width)``, and likewise for height).
            Default: -1.
        aligned (bool, optional): If False, use the legacy implementation. If True, pixel
            shift the box coordinates it by -0.5 for a better alignment with the
            two neighboring pixel indices. This version is used in Detectron2.
            Default: True.
        name(str|None, optional): For detailed information, please refer to :
            ref:`api_guide_Name`. Usually name is no need to set and None by
            default. Default: None.

    Returns:
        The output of ROIAlignOp is a 4-D tensor with shape (num_boxes,\
            channels, pooled_h, pooled_w). The data type is float32 or float64.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.ops import roi_align

            >>> data = paddle.rand([1, 256, 32, 32])
            >>> boxes = paddle.rand([3, 4])
            >>> boxes[:, 2] += boxes[:, 0] + 3
            >>> boxes[:, 3] += boxes[:, 1] + 4
            >>> boxes_num = paddle.to_tensor([3]).astype('int32')
            >>> align_out = roi_align(data, boxes, boxes_num, output_size=3)
            >>> print(align_out.shape)
            [3, 256, 3, 3]
    """

    check_type(output_size, 'output_size', (int, tuple), 'roi_align')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dynamic_or_pir_mode():
        assert (
            boxes_num is not None
        ), "boxes_num should not be None in dygraph mode."
        return _C_ops.roi_align(
            x,
            boxes,
            boxes_num,
            pooled_height,
            pooled_width,
            spatial_scale,
            sampling_ratio,
            aligned,
        )
    else:
        check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'roi_align')
        check_variable_and_dtype(
            boxes, 'boxes', ['float32', 'float64'], 'roi_align'
        )
        helper = LayerHelper('roi_align', **locals())
        dtype = helper.input_dtype()
        align_out = helper.create_variable_for_type_inference(dtype)
        inputs = {
            "X": x,
            "ROIs": boxes,
        }
        if boxes_num is not None:
            inputs['RoisNum'] = boxes_num
        helper.append_op(
            type="roi_align",
            inputs=inputs,
            outputs={"Out": align_out},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale,
                "sampling_ratio": sampling_ratio,
                "aligned": aligned,
            },
        )
        return align_out


class RoIAlign(Layer):
    """
    This interface is used to construct a callable object of the `RoIAlign` class.
    Please refer to :ref:`api_paddle_vision_ops_roi_align`.

    Args:
        output_size (int or tuple[int, int]): The pooled output size(h, w),
            data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): Multiplicative spatial scale factor
            to translate ROI coords from their input scale to the scale used
            when pooling. Default: 1.0.

    Returns:
        The output of ROIAlign operator is a 4-D tensor with \
            shape (num_boxes, channels, pooled_h, pooled_w).

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.ops import RoIAlign

            >>> data = paddle.rand([1, 256, 32, 32])
            >>> boxes = paddle.rand([3, 4])
            >>> boxes[:, 2] += boxes[:, 0] + 3
            >>> boxes[:, 3] += boxes[:, 1] + 4
            >>> boxes_num = paddle.to_tensor([3]).astype('int32')
            >>> roi_align = RoIAlign(output_size=(4, 3))
            >>> align_out = roi_align(data, boxes, boxes_num)
            >>> print(align_out.shape)
            [3, 256, 4, 3]
    """

    def __init__(self, output_size: Size2, spatial_scale: float = 1.0) -> None:
        super().__init__()
        self._output_size = output_size
        self._spatial_scale = spatial_scale

    def forward(
        self, x: Tensor, boxes: Tensor, boxes_num: Tensor, aligned: bool = True
    ) -> Tensor:
        return roi_align(
            x=x,
            boxes=boxes,
            boxes_num=boxes_num,
            output_size=self._output_size,
            spatial_scale=self._spatial_scale,
            aligned=aligned,
        )


class ConvNormActivation(Sequential):
    """
    Configurable block used for Convolution-Normalization-Activation blocks.
    This code is based on the torchvision code with modifications.
    You can also see at https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L68
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int|list|tuple, optional): Size of the convolving kernel. Default: 3
        stride (int|list|tuple, optional): Stride of the convolution. Default: 1
        padding (int|str|tuple|list, optional): Padding added to all four sides of the input. Default: None,
            in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., paddle.nn.Layer], optional): Norm layer that will be stacked on top of the convolution layer.
            If ``None`` this layer won't be used. Default: ``paddle.nn.BatchNorm2D``
        activation_layer (Callable[..., paddle.nn.Layer], optional): Activation function which will be stacked on top of the normalization
            layer (if not ``None``), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``paddle.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        bias (bool|None, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2 = 3,
        stride: Size2 = 1,
        padding: _PaddingSizeMode | Size2 | Size4 | str | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nn.Layer] = BatchNorm2D,
        activation_layer: Callable[..., nn.Layer] = ReLU,
        dilation: int = 1,
        bias: bool | None = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


def nms(
    boxes: Tensor,
    iou_threshold: float = 0.3,
    scores: Tensor | None = None,
    category_idxs: Tensor | None = None,
    categories: Sequence[int] | None = None,
    top_k: int | None = None,
) -> Tensor:
    r"""
    This operator implements non-maximum suppression. Non-maximum suppression (NMS)
    is used to select one bounding box out of many overlapping bounding boxes in object detection.
    Boxes with IoU > iou_threshold will be considered as overlapping boxes,
    just one with highest score can be kept. Here IoU is Intersection Over Union,
    which can be computed by:

    ..  math::

        IoU = \frac{intersection\_area(box1, box2)}{union\_area(box1, box2)}

        If scores are provided, input boxes will be sorted by their scores firstly.

    If category_idxs and categories are provided, NMS will be performed with a batched style,
    which means NMS will be applied to each category respectively and results of each category
    will be concatenated and sorted by scores.

    If K is provided, only the first k elements will be returned. Otherwise, all box indices sorted by scores will be returned.

    Args:
        boxes(Tensor): The input boxes data to be computed, it's a 2D-Tensor with
            the shape of [num_boxes, 4]. The data type is float32 or float64.
            Given as [[x1, y1, x2, y2], …],  (x1, y1) is the top left coordinates,
            and (x2, y2) is the bottom right coordinates.
            Their relation should be ``0 <= x1 < x2 && 0 <= y1 < y2``.
        iou_threshold(float, optional): IoU threshold for determine overlapping boxes. Default value: 0.3.
        scores(Tensor|None, optional): Scores corresponding to boxes, it's a 1D-Tensor with
            shape of [num_boxes]. The data type is float32 or float64. Default: None.
        category_idxs(Tensor|None, optional): Category indices corresponding to boxes.
            it's a 1D-Tensor with shape of [num_boxes]. The data type is int64. Default: None.
        categories(list|tuple|None, optional): A list of unique id of all categories. The data type is int64. Default: None.
        top_k(int|None, optional): The top K boxes who has higher score and kept by NMS preds to
            consider. top_k should be smaller equal than num_boxes. Default: None.

    Returns:
        Tensor: 1D-Tensor with the shape of [num_boxes]. Indices of boxes kept by NMS.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> boxes = paddle.rand([4, 4]).astype('float32')
            >>> boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            >>> boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            >>> print(boxes)
            Tensor(shape=[4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.86583614, 0.52014720, 1.12544549, 1.42540050],
             [0.42400089, 0.40641287, 1.39420986, 1.15078652],
             [0.51785129, 0.73292869, 1.49571705, 0.77608776],
             [0.42639419, 0.71958369, 0.63450879, 0.91689879]])

            >>> out = paddle.vision.ops.nms(boxes, 0.1)
            >>> print(out)
            Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 2, 3])

            >>> scores = paddle.to_tensor([0.6, 0.7, 0.4, 0.233])
            >>> categories = [0, 1, 2, 3]
            >>> category_idxs = paddle.to_tensor([2, 0, 0, 3], dtype="int64")
            >>> out = paddle.vision.ops.nms(boxes,
            ...                             0.1,
            ...                             paddle.to_tensor(scores),
            ...                             paddle.to_tensor(category_idxs),
            ...                             categories,
            ...                             4)
            >>> print(out)
            Tensor(shape=[4], dtype=int64, place=Place(cpu), stop_gradient=True,
            [1, 0, 2, 3])
    """

    def _nms(boxes, iou_threshold):
        if in_dynamic_or_pir_mode():
            return _C_ops.nms(boxes, iou_threshold)

        else:
            helper = LayerHelper('nms', **locals())
            out = helper.create_variable_for_type_inference('int64')
            helper.append_op(
                type='nms',
                inputs={'Boxes': boxes},
                outputs={'KeepBoxesIdxs': out},
                attrs={'iou_threshold': iou_threshold},
            )
            return out

    if scores is None:
        return _nms(boxes, iou_threshold)

    import paddle

    if category_idxs is None:
        sorted_global_indices = paddle.argsort(scores, descending=True)
        sorted_keep_boxes_indices = _nms(
            boxes[sorted_global_indices], iou_threshold
        )
        return sorted_global_indices[sorted_keep_boxes_indices]

    if top_k is not None:
        assert (
            top_k <= scores.shape[0]
        ), "top_k should be smaller equal than the number of boxes"
    assert (
        categories is not None
    ), "if category_idxs is given, categories which is a list of unique id of all categories is necessary"

    mask = paddle.zeros_like(scores, dtype='int32')

    for category_id in categories:
        cur_category_boxes_idxs = paddle.where(
            paddle.equal(category_idxs, paddle.to_tensor(category_id))
        )[0]
        shape = cur_category_boxes_idxs.shape[0]
        cur_category_boxes_idxs = paddle.reshape(
            cur_category_boxes_idxs, [shape]
        )
        if shape == 0:
            continue
        elif shape == 1:
            mask[cur_category_boxes_idxs] = 1
            continue
        cur_category_boxes = boxes[cur_category_boxes_idxs]
        cur_category_scores = scores[cur_category_boxes_idxs]
        cur_category_sorted_indices = paddle.argsort(
            cur_category_scores, descending=True
        )
        cur_category_sorted_boxes = cur_category_boxes[
            cur_category_sorted_indices
        ]

        cur_category_keep_boxes_sub_idxs = cur_category_sorted_indices[
            _nms(cur_category_sorted_boxes, iou_threshold)
        ]

        updates = paddle.ones_like(
            cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs],
            dtype='int32',
        )
        mask = paddle.scatter(
            mask,
            cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs],
            updates,
            overwrite=True,
        )
    keep_boxes_idxs = paddle.where(mask)[0]
    shape = keep_boxes_idxs.shape[0]
    keep_boxes_idxs = paddle.reshape(keep_boxes_idxs, [shape])
    sorted_sub_indices = paddle.argsort(
        scores[keep_boxes_idxs], descending=True
    )

    if top_k is None:
        return keep_boxes_idxs[sorted_sub_indices]

    if in_dygraph_mode():
        top_k = min(top_k, shape)
        _, topk_sub_indices = paddle.topk(scores[keep_boxes_idxs], top_k)
        return keep_boxes_idxs[topk_sub_indices]

    return keep_boxes_idxs[sorted_sub_indices][:top_k]


@overload
def generate_proposals(
    scores: Tensor,
    bbox_deltas: Tensor,
    img_size: Tensor,
    anchors: Tensor,
    variances: Tensor,
    pre_nms_top_n: float = ...,
    post_nms_top_n: float = ...,
    nms_thresh: float = ...,
    min_size: float = ...,
    eta: float = ...,
    pixel_offset: bool = ...,
    return_rois_num: Literal[True] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def generate_proposals(
    scores: Tensor,
    bbox_deltas: Tensor,
    img_size: Tensor,
    anchors: Tensor,
    variances: Tensor,
    pre_nms_top_n: float = ...,
    post_nms_top_n: float = ...,
    nms_thresh: float = ...,
    min_size: float = ...,
    eta: float = ...,
    pixel_offset: bool = ...,
    return_rois_num: Literal[False] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, None]: ...


@overload
def generate_proposals(
    scores: Tensor,
    bbox_deltas: Tensor,
    img_size: Tensor,
    anchors: Tensor,
    variances: Tensor,
    pre_nms_top_n: float = ...,
    post_nms_top_n: float = ...,
    nms_thresh: float = ...,
    min_size: float = ...,
    eta: float = ...,
    pixel_offset: bool = ...,
    return_rois_num: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor | None]: ...


def generate_proposals(
    scores,
    bbox_deltas,
    img_size,
    anchors,
    variances,
    pre_nms_top_n=6000,
    post_nms_top_n=1000,
    nms_thresh=0.5,
    min_size=0.1,
    eta=1.0,
    pixel_offset=False,
    return_rois_num=False,
    name=None,
):
    """
    This operation proposes RoIs according to each box with their
    probability to be a foreground object. And
    the proposals of RPN output are  calculated by anchors, bbox_deltas and scores. Final proposals
    could be used to train detection net.

    For generating proposals, this operation performs following steps:

    1. Transpose and resize scores and bbox_deltas in size of
       (H * W * A, 1) and (H * W * A, 4)
    2. Calculate box locations as proposals candidates.
    3. Clip boxes to image
    4. Remove predicted boxes with small area.
    5. Apply non-maximum suppression (NMS) to get final proposals as output.

    Args:
        scores (Tensor): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas (Tensor): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        img_size (Tensor): A 2-D Tensor with shape [N, 2] represents origin
            image shape information for N batch, including height and width of the input sizes.
            The data type can be float32 or float64.
        anchors (Tensor):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances (Tensor): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n (float, optional): Number of total bboxes to be kept per
            image before NMS. `6000` by default.
        post_nms_top_n (float, optional): Number of total bboxes to be kept per
            image after NMS. `1000` by default.
        nms_thresh (float, optional): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size (float, optional): Remove predicted boxes with either height or
            width less than this value. `0.1` by default.
        eta(float, optional): Apply in adaptive NMS, only works if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration. 1.0 by default.
        pixel_offset (bool, optional): Whether there is pixel offset. If True, the offset of `img_size` will be 1. 'False' by default.
        return_rois_num (bool, optional): Whether to return `rpn_rois_num` . When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's
            num of each image in one batch. 'False' by default.
        name(str|None, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        - rpn_rois (Tensor): The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - rpn_roi_probs (Tensor): The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - rpn_rois_num (Tensor): Rois's num of each image in one batch. 1-D Tensor with shape ``[B,]`` while ``B`` is the batch size. And its sum equals to RoIs number ``N`` .

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> scores = paddle.rand((2,4,5,5), dtype=paddle.float32)
            >>> bbox_deltas = paddle.rand((2, 16, 5, 5), dtype=paddle.float32)
            >>> img_size = paddle.to_tensor([[224.0, 224.0], [224.0, 224.0]])
            >>> anchors = paddle.rand((2,5,4,4), dtype=paddle.float32)
            >>> variances = paddle.rand((2,5,10,4), dtype=paddle.float32)
            >>> rois, roi_probs, roi_nums = paddle.vision.ops.generate_proposals(scores, bbox_deltas,
            ...                 img_size, anchors, variances, return_rois_num=True)
            >>> # doctest: +SKIP('random sample')
            >>> print(rois, roi_probs, roi_nums)
            Tensor(shape=[2, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.]])
            Tensor(shape=[2, 1], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0.],
             [0.]])
            Tensor(shape=[2], dtype=int32, place=Place(cpu), stop_gradient=True,
            [1, 1])
    """

    if in_dygraph_mode():
        assert (
            return_rois_num
        ), "return_rois_num should be True in dygraph mode."
        attrs = (
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            min_size,
            eta,
            pixel_offset,
        )
        rpn_rois, rpn_roi_probs, rpn_rois_num = _C_ops.generate_proposals(
            scores, bbox_deltas, img_size, anchors, variances, *attrs
        )

        return rpn_rois, rpn_roi_probs, rpn_rois_num
    elif in_pir_mode():
        assert (
            return_rois_num
        ), "return_rois_num should be True in PaddlePaddle inner op mode."
        rpn_rois, rpn_roi_probs, rpn_rois_num = _C_ops.generate_proposals(
            scores,
            bbox_deltas,
            img_size,
            anchors,
            variances,
            pre_nms_top_n,
            post_nms_top_n,
            nms_thresh,
            min_size,
            eta,
            pixel_offset,
        )
        rpn_rois.stop_gradient = True
        rpn_roi_probs.stop_gradient = True
        rpn_rois_num.stop_gradient = True
        return rpn_rois, rpn_roi_probs, rpn_rois_num
    else:
        helper = LayerHelper('generate_proposals_v2', **locals())

        check_variable_and_dtype(
            scores, 'scores', ['float32'], 'generate_proposals_v2'
        )
        check_variable_and_dtype(
            bbox_deltas, 'bbox_deltas', ['float32'], 'generate_proposals_v2'
        )
        check_variable_and_dtype(
            img_size,
            'img_size',
            ['float32', 'float64'],
            'generate_proposals_v2',
        )
        check_variable_and_dtype(
            anchors, 'anchors', ['float32'], 'generate_proposals_v2'
        )
        check_variable_and_dtype(
            variances, 'variances', ['float32'], 'generate_proposals_v2'
        )

        rpn_rois = helper.create_variable_for_type_inference(
            dtype=bbox_deltas.dtype
        )
        rpn_roi_probs = helper.create_variable_for_type_inference(
            dtype=scores.dtype
        )
        outputs = {
            'RpnRois': rpn_rois,
            'RpnRoiProbs': rpn_roi_probs,
        }
        if return_rois_num:
            rpn_rois_num = helper.create_variable_for_type_inference(
                dtype='int32'
            )
            rpn_rois_num.stop_gradient = True
            outputs['RpnRoisNum'] = rpn_rois_num

        helper.append_op(
            type="generate_proposals_v2",
            inputs={
                'Scores': scores,
                'BboxDeltas': bbox_deltas,
                'ImShape': img_size,
                'Anchors': anchors,
                'Variances': variances,
            },
            attrs={
                'pre_nms_topN': pre_nms_top_n,
                'post_nms_topN': post_nms_top_n,
                'nms_thresh': nms_thresh,
                'min_size': min_size,
                'eta': eta,
                'pixel_offset': pixel_offset,
            },
            outputs=outputs,
        )
        rpn_rois.stop_gradient = True
        rpn_roi_probs.stop_gradient = True
        if not return_rois_num:
            rpn_rois_num = None

        return rpn_rois, rpn_roi_probs, rpn_rois_num


@overload
def matrix_nms(
    bboxes: Tensor,
    scores: Tensor,
    score_threshold: float,
    post_threshold: float,
    nms_top_k: int,
    keep_top_k: int,
    use_gaussian: bool = ...,
    gaussian_sigma: float = ...,
    background_label: int = ...,
    normalized: bool = ...,
    return_index: Literal[True] = ...,
    return_rois_num: Literal[False] = ...,
    name: str | None = None,
) -> tuple[Tensor, None, Tensor]: ...


@overload
def matrix_nms(
    bboxes: Tensor,
    scores: Tensor,
    score_threshold: float,
    post_threshold: float,
    nms_top_k: int,
    keep_top_k: int,
    use_gaussian: bool = ...,
    gaussian_sigma: float = ...,
    background_label: int = ...,
    normalized: bool = ...,
    return_index: Literal[True] = ...,
    return_rois_num: Literal[True] = ...,
    name: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def matrix_nms(
    bboxes: Tensor,
    scores: Tensor,
    score_threshold: float,
    post_threshold: float,
    nms_top_k: int,
    keep_top_k: int,
    use_gaussian: bool = ...,
    gaussian_sigma: float = ...,
    background_label: int = ...,
    normalized: bool = ...,
    return_index: Literal[False] = ...,
    return_rois_num: Literal[True] = ...,
    name: str | None = None,
) -> tuple[Tensor, Tensor, None]: ...


@overload
def matrix_nms(
    bboxes: Tensor,
    scores: Tensor,
    score_threshold: float,
    post_threshold: float,
    nms_top_k: int,
    keep_top_k: int,
    use_gaussian: bool = ...,
    gaussian_sigma: float = ...,
    background_label: int = ...,
    normalized: bool = ...,
    return_index: Literal[False] = ...,
    return_rois_num: Literal[False] = ...,
    name: str | None = None,
) -> tuple[Tensor, None, None]: ...


@overload
def matrix_nms(
    bboxes: Tensor,
    scores: Tensor,
    score_threshold: float,
    post_threshold: float,
    nms_top_k: int,
    keep_top_k: int,
    use_gaussian: bool = ...,
    gaussian_sigma: float = ...,
    background_label: int = ...,
    normalized: bool = ...,
    return_index: bool = ...,
    return_rois_num: bool = ...,
    name: str | None = None,
) -> tuple[Tensor, Tensor | None, Tensor | None]: ...


def matrix_nms(
    bboxes,
    scores,
    score_threshold,
    post_threshold,
    nms_top_k,
    keep_top_k,
    use_gaussian=False,
    gaussian_sigma=2.0,
    background_label=0,
    normalized=True,
    return_index=False,
    return_rois_num=True,
    name=None,
):
    """

    This operator does matrix non maximum suppression (NMS).
    First selects a subset of candidate bounding boxes that have higher scores
    than score_threshold (if provided), then the top k candidate is selected if
    nms_top_k is larger than -1. Score of the remaining candidate are then
    decayed according to the Matrix NMS scheme.
    After NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.

    Args:
        bboxes (Tensor): A 3-D Tensor with shape [N, M, 4] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
        scores (Tensor): A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes. The data type is float32 or float64.
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score.
        post_threshold (float): Threshold to filter out bounding boxes with
                                low confidence score AFTER decaying.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        use_gaussian (bool, optional): Use Gaussian as the decay function. Default: False
        gaussian_sigma (float, optional): Sigma for Gaussian decay function. Default: 2.0
        background_label (int, optional): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        normalized (bool, optional): Whether detections are normalized. Default: True
        return_index(bool, optional): Whether return selected index. Default: False
        return_rois_num(bool, optional): whether return rois_num. Default: True
        name(str|None, optional): Name of the matrix nms op. Default: None.
    Returns:
        - A tuple with three Tensor, (Out, Index, RoisNum) if return_index is True,
          otherwise, a tuple with two Tensor (Out, RoisNum) is returned.
        - Out (Tensor), A 2-D Tensor with shape [No, 6] containing the
          detection results.
          Each row has 6 values, [label, confidence, xmin, ymin, xmax, ymax]
        - Index (Tensor), A 2-D Tensor with shape [No, 1] containing the
          selected indices, which are absolute values cross batches.
        - rois_num (Tensor), A 1-D Tensor with shape [N] containing
          the number of detected boxes in each image.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.vision.ops import matrix_nms

            >>> boxes = paddle.rand([4, 1, 4])
            >>> boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
            >>> boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
            >>> scores = paddle.rand([4, 80, 1])
            >>> out = matrix_nms(bboxes=boxes, scores=scores, background_label=0,
            ...                         score_threshold=0.5, post_threshold=0.1,
            ...                         nms_top_k=400, keep_top_k=200, normalized=False)
    """
    if in_dynamic_or_pir_mode():
        out, index, rois_num = _C_ops.matrix_nms(
            bboxes,
            scores,
            score_threshold,
            nms_top_k,
            keep_top_k,
            post_threshold,
            use_gaussian,
            gaussian_sigma,
            background_label,
            normalized,
        )
        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return out, rois_num, index
    else:
        check_variable_and_dtype(
            bboxes, 'BBoxes', ['float32', 'float64'], 'matrix_nms'
        )
        check_variable_and_dtype(
            scores, 'Scores', ['float32', 'float64'], 'matrix_nms'
        )
        check_type(score_threshold, 'score_threshold', float, 'matrix_nms')
        check_type(post_threshold, 'post_threshold', float, 'matrix_nms')
        check_type(nms_top_k, 'nums_top_k', int, 'matrix_nms')
        check_type(keep_top_k, 'keep_top_k', int, 'matrix_nms')
        check_type(normalized, 'normalized', bool, 'matrix_nms')
        check_type(use_gaussian, 'use_gaussian', bool, 'matrix_nms')
        check_type(gaussian_sigma, 'gaussian_sigma', float, 'matrix_nms')
        check_type(background_label, 'background_label', int, 'matrix_nms')
        helper = LayerHelper('matrix_nms', **locals())
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')
        outputs = {'Out': output, 'Index': index}
        if return_rois_num:
            rois_num = helper.create_variable_for_type_inference(dtype='int32')
            outputs['RoisNum'] = rois_num

        helper.append_op(
            type="matrix_nms",
            inputs={'BBoxes': bboxes, 'Scores': scores},
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'post_threshold': post_threshold,
                'nms_top_k': nms_top_k,
                'gaussian_sigma': gaussian_sigma,
                'use_gaussian': use_gaussian,
                'keep_top_k': keep_top_k,
                'normalized': normalized,
            },
            outputs=outputs,
        )
        output.stop_gradient = True

        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return output, rois_num, index
