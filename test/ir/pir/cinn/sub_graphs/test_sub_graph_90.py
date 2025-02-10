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

from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.relu = paddle.nn.functional.relu

    def triple_full(self):
        y1 = paddle.full([4], 1)
        y2 = paddle.full([4], 0)
        y3 = paddle.full([4], 0)
        return y1, y2, y3

    def concat_case_1(self):
        y1, y2, y3 = self.triple_full()
        out = paddle.concat([y1, y2, y3])
        return self.relu(out)

    def concat_case_2(self):
        y1, y2, y3 = self.triple_full()
        out = paddle.concat([y2, y1, y3])
        return self.relu(out)

    def concat_case_3(self):
        y1, y2, y3 = self.triple_full()
        out = paddle.concat([y3, y2, y1])
        return self.relu(out)

    def forward(self, x):
        outs = []
        for fn in [self.concat_case_1, self.concat_case_2, self.concat_case_3]:
            # to trigger duplicate subgraph and cache them.
            for i in range(3):
                outs.append(self.relu(fn()))
        outs.append(self.relu(x))
        return outs


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (paddle.rand(shape=[12], dtype=paddle.float32),)
        self.net = LayerCase
        self.with_train = True

    def set_flags(self):
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags(
            {
                "FLAGS_deny_cinn_ops": "relu",
                "FLAGS_prim_forward_blacklist": "pd_op.relu",
                "FLAGS_enable_cinn_compile_cache": False,
            }
        )


if __name__ == '__main__':
    unittest.main()
