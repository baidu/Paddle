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

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle.optimizer import Adam
from paddle.pir_utils import IrGuard

paddle.enable_static()
IMAGE_SIZE = 784


class TestSimpleParamSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_params(self, prog):
        scope = paddle.static.global_scope()

        def get_tensor(name):
            t = scope.find_var(name).get_tensor()
            return t

        param_dict = {}
        opt_dict = {}
        for op in prog.global_block().ops:
            if op.name() == "builtin.parameter" and "persistable" in op.attrs():
                if op.attrs()['persistable'] == [True]:
                    name = op.attrs()["parameter_name"]
                    param_dict.update({name: get_tensor(name)})
            elif op.name() == "pd_op.data" and "persistable" in op.attrs():
                if op.attrs()['persistable'] == [True]:
                    name = op.attrs()["name"]
                    opt_dict.update({name: get_tensor(name)})
        return param_dict, opt_dict

    def test_params_python(self):
        with IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(
                main_program, paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="static_x", shape=[None, IMAGE_SIZE], dtype='float32'
                )
                z = paddle.static.nn.fc(x, 10)
                z = paddle.static.nn.fc(z, 10, bias_attr=False)
                loss = paddle.mean(z)
                opt = Adam(learning_rate=1e-3)
                opt.minimize(loss)
                exe = paddle.static.Executor(self.place)
                exe.run(paddle.static.default_startup_program())
                fake_inputs = np.random.randn(2, IMAGE_SIZE).astype('float32')
                exe.run(
                    main_program,
                    feed={'static_x': fake_inputs},
                    fetch_list=[loss],
                )
                scope = paddle.static.global_scope()
                params = main_program.global_block().all_parameters()
                param_dict = {}
                # save parameters
                for v in params:
                    name = v.get_defining_op().attrs()["parameter_name"]
                    param_dict.update({name: scope.var(name).get_tensor()})

                path = os.path.join(self.temp_dir.name, "save_pickle")
                paddle.static.io.save(main_program, path)

                # change the value of parameters
                for v in params:
                    name = v.get_defining_op().attrs()["parameter_name"]
                    tensor = scope.var(name).get_tensor()
                    tensor.set(np.zeros_like(np.array(tensor)), self.place)

                # load parameters
                paddle.static.io.load(main_program, path)
                for v in params:
                    if v.get_defining_op().name() == "builtin.parameter":
                        name = v.get_defining_op().attrs()["parameter_name"]
                        t = scope.find_var(name).get_tensor()
                        np.testing.assert_array_equal(t, param_dict[name])

    def test_params_cpp(self):
        with IrGuard():
            prog = paddle.static.Program()
            with paddle.static.program_guard(prog):
                x = paddle.static.data(
                    name="static_x", shape=[None, IMAGE_SIZE], dtype='float32'
                )
                z = paddle.static.nn.fc(x, 10)
                z = paddle.static.nn.fc(z, 10, bias_attr=False)
                loss = paddle.mean(z)
                opt = Adam(learning_rate=1e-3)
                opt.minimize(loss)
                exe = paddle.static.Executor(self.place)
                exe.run(paddle.static.default_startup_program())
                fake_inputs = np.random.randn(2, IMAGE_SIZE).astype('float32')
                exe.run(prog, feed={'static_x': fake_inputs}, fetch_list=[loss])

                param_dict, opt_dict = self.get_params(prog)
                # test save_func and load_func
                save_dir = os.path.join(self.temp_dir.name, "save_params")

                for k, v in param_dict.items():
                    path = os.path.join(save_dir, k, '.pdparams')
                    # test fp16
                    paddle.base.core.save_func(v, k, path, True, True)
                    tensor = param_dict[k]
                    tensor.set(np.zeros_like(np.array(tensor)), self.place)
                    paddle.base.core.load_func(
                        path,
                        -1,
                        [],
                        False,
                        tensor,
                        paddle.framework._current_expected_place_(),
                    )
                    np.testing.assert_array_equal(tensor, v)

                for k, v in opt_dict.items():
                    path = os.path.join(save_dir, k, '.pdopt')
                    paddle.base.core.save_func(v, k, path, True, False)
                    tensor = opt_dict[k]
                    tensor.set(np.zeros_like(np.array(tensor)), self.place)
                    paddle.base.core.load_func(
                        path,
                        -1,
                        [],
                        False,
                        tensor,
                        paddle.framework._current_expected_place_(),
                    )
                    np.testing.assert_array_equal(tensor, v)

                # test save_combine_func and load_combine_func
                save_dir = os.path.join(
                    self.temp_dir.name, "save_combine_params"
                )
                path = os.path.join(save_dir, 'demo.pdiparams')
                param_vec = list(param_dict.values())
                paddle.base.core.save_combine_func(
                    param_vec, list(param_dict.keys()), path, True, False, False
                )
                param_new = []
                for tensor in param_vec:
                    tensor.set(np.zeros_like(np.array(tensor)), self.place)
                    param_new.append(tensor)
                paddle.base.core.load_combine_func(
                    path,
                    list(param_dict.keys()),
                    param_new,
                    False,
                    paddle.framework._current_expected_place_(),
                )
                np.testing.assert_equal(param_new, param_vec)
                # save to memory
                paddle.base.core.save_combine_func(
                    param_vec, list(param_dict.keys()), path, True, False, True
                )
                # save as fp16
                paddle.base.core.save_combine_func(
                    param_vec, list(param_dict.keys()), path, True, True, False
                )
                # load as fp16
                paddle.base.core.load_combine_func(
                    path,
                    list(param_dict.keys()),
                    param_new,
                    True,
                    paddle.framework._current_expected_place_(),
                )

                # test save_vars
                path_prefix = os.path.join(save_dir, 'new')
                params_path = path_prefix + ".pdiparams"
                if os.path.isdir(params_path):
                    raise ValueError(
                        f"'{params_path}' is an existing directory."
                    )

                save_dirname = os.path.dirname(params_path)
                params_filename = os.path.basename(params_path)
                # test combine
                paddle.static.io.save_vars(
                    executor=exe,
                    dirname=save_dirname,
                    main_program=prog,
                    filename=params_filename,
                )
                # test separate
                paddle.static.io.save_vars(
                    executor=exe,
                    dirname=save_dirname,
                    main_program=prog,
                )
                # test load_vars
                load_dirname = os.path.dirname(params_path)
                load_filename = os.path.basename(params_path)
                # test combine
                paddle.static.io.load_vars(
                    executor=exe,
                    dirname=load_dirname,
                    main_program=prog,
                    filename=load_filename,
                )
                # test separate
                paddle.static.io.load_vars(
                    executor=exe,
                    dirname=load_dirname,
                    main_program=prog,
                )


if __name__ == '__main__':
    unittest.main()
