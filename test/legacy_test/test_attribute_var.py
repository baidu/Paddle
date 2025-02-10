# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np

import paddle
import paddle.inference as paddle_infer
from paddle.base.framework import OpProtoHolder, Program, program_guard

paddle.enable_static()


class UnittestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.init_info()

    def tearDwon(self):
        self.temp_dir.cleanup()

    def init_info(self):
        self.shapes = None
        self.save_path = None

    def path_prefix(self):
        return type(self).__name__

    def infer_prog(self):
        if paddle.framework.use_pir_api():
            config = paddle_infer.Config(
                self.save_path + '.json', self.save_path + '.pdiparams'
            )
            config.enable_new_ir()
            config.enable_new_executor()
        else:
            config = paddle_infer.Config(
                self.save_path + '.pdmodel', self.save_path + '.pdiparams'
            )
        config.disable_mkldnn()
        predictor = paddle_infer.create_predictor(config)
        input_names = predictor.get_input_names()
        for i, shape in enumerate(self.shapes):
            input_handle = predictor.get_input_handle(input_names[i])
            self.fake_input = np.random.randn(*shape).astype("float32")
            input_handle.reshape(shape)
            input_handle.copy_from_cpu(self.fake_input)
        predictor.run()
        output_names = predictor.get_output_names()
        res = []
        for out_name in output_names:
            output_handle = predictor.get_output_handle(out_name)
            output_data = output_handle.copy_to_cpu()
            res.append(output_data)

        if len(output_names) == 1:
            res = res[0]

        return res


class TestTileTensorList(UnittestBase):
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'tile_tensors')

    def _test_static(self):
        main_prog = Program()
        startup_prog = Program()
        with program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)
            shape0 = paddle.full([1], 1, dtype='int32')
            shape1 = paddle.full([1], 2, dtype='int32')
            shape = [3, shape1, shape0]
            out = paddle.tile(feat, shape)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue("Vars[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[x, out])
            self.assertEqual(res[1].shape, (6, 6, 10))

            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            # Test for Inference Predictor
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (6, 6, 10))


class TestTileTensor(UnittestBase):
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, 'tile_tensor')

    def _test_static(self):
        main_prog = Program()
        startup_prog = Program()
        with program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)
            # shape is a Variable
            shape = paddle.assign([3, 2, 1])
            out = paddle.tile(feat, shape)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue("Var[" in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[x, out])
            self.assertEqual(res[1].shape, (6, 6, 10))

            paddle.static.save_inference_model(self.save_path, [x], [out], exe)
            # Test for Inference Predictor
            infer_out = self.infer_prog()
            self.assertEqual(infer_out.shape, (6, 6, 10))


class TestRegisterSupportTensorInOpMaker(unittest.TestCase):
    def setUp(self):
        self.all_protos = OpProtoHolder.instance()
        self.support_tensor_attrs = {
            'dropout': ['dropout_prob'],
            'tile': ['repeat_times'],
        }
        # Just add a op example to test not support tensor
        self.not_support_tensor_attrs = {'svd': ['full_matrices']}

    def test_support_tensor(self):
        # All Attribute tagged with .SupportTensor() in OpMaker will return True
        for op_type, attr_names in self.support_tensor_attrs.items():
            for attr_name in attr_names:
                self.assertTrue(self.is_support_tensor_attr(op_type, attr_name))

        # All Attribute not tagged with .SupportTensor() in OpMaker will return False
        for op_type, attr_names in self.not_support_tensor_attrs.items():
            for attr_name in attr_names:
                self.assertFalse(
                    self.is_support_tensor_attr(op_type, attr_name)
                )

    def is_support_tensor_attr(self, op_type, attr_name):
        proto = self.all_protos.get_op_proto(op_type)
        for attr in proto.attrs:
            if attr.name == attr_name:
                return attr.support_tensor
        raise RuntimeError("Not found attribute : ", attr_name)


if __name__ == '__main__':
    unittest.main()
