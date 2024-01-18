# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# repo: PaddleDetection
# model: configs^sparse_rcnn^sparse_rcnn_r50_fpn_3x_pro100_coco_single_dy2st_train
# method||__getitem__,api||paddle.tensor.creation.full,method||astype,api||paddle.vision.ops.distribute_fpn_proposals,api||paddle.vision.ops.roi_align,api||paddle.vision.ops.roi_align,api||paddle.vision.ops.roi_align,api||paddle.vision.ops.roi_align,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.gather,method||reshape,method||transpose,method||reshape
import unittest

import numpy as np

import paddle


class SIR42(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_562,  # (shape: [1, 256, 120, 184], dtype: paddle.float32, stop_gradient: False)
        var_563,  # (shape: [1, 256, 60, 92], dtype: paddle.float32, stop_gradient: False)
        var_564,  # (shape: [1, 256, 30, 46], dtype: paddle.float32, stop_gradient: False)
        var_565,  # (shape: [1, 256, 15, 23], dtype: paddle.float32, stop_gradient: False)
        var_567,  # (shape: [1, 100, 4], dtype: paddle.float32, stop_gradient: True)
        var_568,  # (shape: [1, 100, 256], dtype: paddle.float32, stop_gradient: False)
    ):
        var_569 = var_567.__getitem__(0)
        var_570 = paddle.tensor.creation.full([1], 100)
        var_571 = var_570.astype('int32')
        out = paddle.vision.ops.distribute_fpn_proposals(
            var_569, 2, 5, 4, 224, rois_num=var_571
        )
        var_572 = out[0][0]
        var_573 = out[0][1]
        var_574 = out[0][2]
        var_575 = out[0][3]
        var_576 = out[1]
        var_577 = out[2][0]
        var_578 = out[2][1]
        var_579 = out[2][2]
        var_580 = out[2][3]
        var_581 = paddle.vision.ops.roi_align(
            x=var_562,
            boxes=var_572,
            boxes_num=var_577,
            output_size=7,
            spatial_scale=0.25,
            sampling_ratio=2,
            aligned=True,
        )
        var_582 = paddle.vision.ops.roi_align(
            x=var_563,
            boxes=var_573,
            boxes_num=var_578,
            output_size=7,
            spatial_scale=0.125,
            sampling_ratio=2,
            aligned=True,
        )
        var_583 = paddle.vision.ops.roi_align(
            x=var_564,
            boxes=var_574,
            boxes_num=var_579,
            output_size=7,
            spatial_scale=0.0625,
            sampling_ratio=2,
            aligned=True,
        )
        var_584 = paddle.vision.ops.roi_align(
            x=var_565,
            boxes=var_575,
            boxes_num=var_580,
            output_size=7,
            spatial_scale=0.03125,
            sampling_ratio=2,
            aligned=True,
        )
        var_585 = paddle.tensor.manipulation.concat(
            [var_581, var_582, var_583, var_584]
        )
        var_586 = paddle.tensor.manipulation.gather(var_585, var_576)
        var_587 = var_586.reshape([100, 256, -1])
        var_588 = var_587.transpose(perm=[2, 0, 1])
        var_589 = var_568.reshape([1, 100, 256])
        return var_589, var_588


class TestSIR42(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 120, 184], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 60, 92], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 30, 46], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 15, 23], dtype=paddle.float32),
            paddle.rand(shape=[1, 100, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 100, 256], dtype=paddle.float32),
        )
        self.net = SIR42()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
