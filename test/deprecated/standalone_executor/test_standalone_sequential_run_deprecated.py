# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle


class TestStandaloneExecutor(unittest.TestCase):
    def build_program(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(name="data", shape=[2, 2], dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.static.nn.fc(a, 2)
            c = t + b

        return main_program, startup_program, [c]

    def run_program(self, sequential_run=False):
        seed = 100
        paddle.seed(seed)
        np.random.seed(seed)
        main, startup, outs = self.build_program()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.sequential_run = sequential_run
        print(build_strategy)
        compiled_program = paddle.static.CompiledProgram(
            main, build_strategy=build_strategy
        )

        exe = paddle.static.Executor()
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            data = np.ones([2, 2], dtype="float32")
            ret = exe.run(
                compiled_program,
                feed={"data": data},
                fetch_list=list(outs),
            )
            return ret

    def test_result(self):
        paddle.enable_static()
        ret1 = self.run_program(True)
        ret2 = self.run_program(False)
        np.testing.assert_array_equal(ret1, ret2)

    def test_str_flag(self):
        paddle.enable_static()
        os.environ['FLAGS_new_executor_sequential_run'] = 'true'
        ret1 = self.run_program(True)
        assert os.environ['FLAGS_new_executor_sequential_run'] == "true"


if __name__ == "__main__":
    unittest.main()
