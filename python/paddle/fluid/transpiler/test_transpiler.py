#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.proto.framework_pb2 as framework_pb2


class TestTranspiler(unittest.TestCase):
    def net_conf(self):
        x = fluid.layers.data(name='x', shape=[1000], dtype='float32')
        y_predict = fluid.layers.fc(input=x,
                                    size=1000,
                                    act=None,
                                    param_attr=fluid.ParamAttr(name='fc_w'),
                                    bias_attr=fluid.ParamAttr(name='fc_b'))
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
        sgd_optimizer.minimize(avg_cost)
        self.loss_name = avg_cost.name

    def test_transpiler(self):
        main = fluid.Program()
        with fluid.program_guard(main):
            self.net_conf()
        origin_prog = main.clone()

        iro = core.IROptimizer()
        new_desc = iro.optimize(core.OptimizeLevel.NONE, main.desc)

        def _desc2str_(desc):
            protostr = desc.serialize_to_string()
            proto = framework_pb2.ProgramDesc.FromString(str(protostr))
            return proto.__str__()

        print(_desc2str_(new_desc))


if __name__ == "__main__":
    unittest.main()
