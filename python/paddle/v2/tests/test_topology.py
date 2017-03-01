# Copyright PaddlePaddle contributors. All Rights Reserved
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

import paddle.trainer.PyDataProvider2 as pydp2
import paddle.trainer_config_helpers as conf_helps
import paddle.v2.configs as configs
import paddle.v2.topology as topology


class TestTopology(unittest.TestCase):
    def test_data_type(self):
        pixel = configs.layer.data(
            name='pixel', type=configs.data_type.dense_vector(784))
        label = configs.layer.data(
            name='label', type=configs.data_type.integer_value(10))
        hidden = configs.layer.fc(input=pixel,
                                  size=100,
                                  act=conf_helps.SigmoidActivation())
        inference = configs.layer.fc(input=hidden,
                                     size=10,
                                     act=conf_helps.SoftmaxActivation())
        cost = configs.layer.classification_cost(input=inference, label=label)
        topo = topology.Topology(cost)
        data_types = topo.data_type()
        self.assertEqual(len(data_types), 2)
        pixel_data_type = filter(lambda type: type[0] == "pixel", data_types)
        self.assertEqual(len(pixel_data_type), 1)
        pixel_data_type = pixel_data_type[0]
        self.assertEqual(pixel_data_type[1].type, pydp2.DataType.Dense)
        self.assertEqual(pixel_data_type[1].dim, 784)

        label_data_type = filter(lambda type: type[0] == "label", data_types)
        self.assertEqual(len(label_data_type), 1)
        label_data_type = label_data_type[0]
        self.assertEqual(label_data_type[1].type, pydp2.DataType.Index)
        self.assertEqual(label_data_type[1].dim, 10)

    def test_get_layer(self):
        pixel = configs.layer.data(
            name='pixel', type=configs.data_type.dense_vector(784))
        label = configs.layer.data(
            name='label', type=configs.data_type.integer_value(10))
        hidden = configs.layer.fc(input=pixel,
                                  size=100,
                                  act=conf_helps.SigmoidActivation())
        inference = configs.layer.fc(input=hidden,
                                     size=10,
                                     act=conf_helps.SoftmaxActivation())
        cost = configs.layer.classification_cost(input=inference, label=label)
        topo = topology.Topology(cost)
        pixel_layer = topo.get_layer("pixel")
        label_layer = topo.get_layer("label")
        self.assertEqual(pixel_layer, pixel)
        self.assertEqual(label_layer, label)

    def test_parse(self):
        pixel = configs.layer.data(
            name='pixel', type=configs.data_type.dense_vector(784))
        label = configs.layer.data(
            name='label', type=configs.data_type.integer_value(10))
        hidden = configs.layer.fc(input=pixel,
                                  size=100,
                                  act=conf_helps.SigmoidActivation())
        inference = configs.layer.fc(input=hidden,
                                     size=10,
                                     act=conf_helps.SoftmaxActivation())
        maxid = configs.layer.max_id(input=inference)
        cost1 = configs.layer.classification_cost(input=inference, label=label)
        cost2 = configs.layer.cross_entropy_cost(input=inference, label=label)

        topology.Topology(cost2).proto()
        topology.Topology([cost1]).proto()
        topology.Topology([cost1, cost2]).proto()
        topology.Topology([inference, maxid]).proto()


if __name__ == '__main__':
    unittest.main()
