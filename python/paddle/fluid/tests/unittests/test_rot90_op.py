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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class TestRot90_API(unittest.TestCase):
    """Test rot90 api."""

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            axis = [0]
            input = fluid.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.rot90(input, k=1, dims=[0, 1])
            output = paddle.rot90(output, k=1, dims=[0, 1])
            output = output.rot90(k=1, dims=[0, 1])
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)

            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
            res = exe.run(train_program,
                          feed={'input': img},
                          fetch_list=[output])

            out_np = np.array(res[0])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)

            self.assertTrue(
                (out_np == out_ref).all(),
                msg='rot90 output is wrong, out =' + str(out_np))

    def test_dygraph(self):
        img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
        with fluid.dygraph.guard():
            inputs = fluid.dygraph.to_variable(img)

            ret = paddle.rot90(inputs, k=1, dims=[0, 1])
            ret = ret.rot90(1, dims=[0, 1])
            ret = paddle.rot90(ret, k=1, dims=[0, 1])
            out_ref = np.array([[4, 1], [5, 2], [6, 3]]).astype(np.float32)

            self.assertTrue(
                (ret.numpy() == out_ref).all(),
                msg='rot90 output is wrong, out =' + str(ret.numpy()))


if __name__ == "__main__":
    unittest.main()
