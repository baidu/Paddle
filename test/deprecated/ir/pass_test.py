#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import random
import unittest
import warnings

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import Block


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = base.Program()
        self.startup_program = base.Program()
        self.feeds = None
        self.fetch_list = None
        self.pass_names = None
        self.pass_attrs = {}
        self.graph_attrs = {}
        self.fused_op_type = None
        self.num_fused_ops = -1

        np.random.seed(123)
        random.seed(124)

    def _get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        return places

    def grad(self, var):
        grad_name = var.name + "@GRAD"
        return self.main_program.global_block().var(grad_name)

    def append_gradients(self, outs):
        with base.program_guard(self.main_program, self.startup_program):
            loss = paddle.mean(outs)
            base.backward.append_backward(loss)

    def check_output(self, startup_on_cpu=False, atol=1e-5):
        '''
        Check whether the fetched outputs of the origin program and the
        optimized program are the same.

        For inference model, the parameters are loaded to CPUPlace first,
        after apply all specified passes, then copy the parameters to GPUPlace.
        We can set startup_on_cpu to True to test inference pass.
        '''
        places = self._get_places()
        for place in places:
            self.check_output_with_place(place, startup_on_cpu, atol)

    def _run_program(self, executor, program):
        outs = executor.run(
            program=program,
            feed=self.feeds,
            fetch_list=self.fetch_list,
            return_numpy=False,
        )
        outs_np = []
        outs_lod = []
        for out in outs:
            outs_np.append(np.array(out))
            outs_lod.append(out.lod())
        return outs_np, outs_lod

    def _apply_ir_passes(self):
        graph = core.Graph(self.main_program.desc)
        graph.set_not_owned("__param_scope__", base.global_scope())
        for attr_name, attr_value in self.graph_attrs.items():
            graph.set(attr_name, attr_value)

        if not isinstance(self.pass_names, list):
            self.pass_names = [self.pass_names]

        pass_builder = core.PassBuilder()
        for name in self.pass_names:
            ir_pass = pass_builder.append_pass(name)
            # Set attr for pass
            if self.pass_attrs.get(name, None) is not None:
                attrs = self.pass_attrs[name]
                for key in attrs:
                    ir_pass.set(key, attrs[key])

        trans_pass = pass_builder.append_pass("graph_to_program_pass")
        opt_program = base.Program()
        trans_pass.set_not_owned("program", opt_program.desc)
        for p in pass_builder.all_passes():
            p.apply(graph)
        opt_program.blocks = [
            Block(opt_program, i) for i in range(opt_program.desc.num_blocks())
        ]
        opt_program._sync_with_cpp()
        return opt_program

    def check_output_with_place(self, place, startup_on_cpu=False, atol=1e-5):
        '''
        Check whether the fetched outputs of the origin program and the
        optimized program are the same.

        For inference model, the parameters are loaded to CPUPlace first,
        after apply all specified passes, then copy the parameters to GPUPlace.
        We can set startup_on_cpu to True to test inference pass.
        '''
        executor = base.Executor(place)
        if startup_on_cpu:
            # Initialize parameters on CPU
            cpu_executor = base.Executor(base.CPUPlace())
            cpu_executor.run(self.startup_program)
            outs, lods = self._run_program(cpu_executor, self.main_program)
        else:
            executor.run(self.startup_program)
            outs, lods = self._run_program(executor, self.main_program)
        self.assertTrue(
            len(self.fetch_list) == len(outs),
            f"Checking the number of fetches failed. Expected: {len(self.fetch_list)}, Received: {len(outs)}",
        )

        # Parameters may be changed in ir passes.
        opt_program = self._apply_ir_passes()
        self.check_program(opt_program)

        if startup_on_cpu and not isinstance(place, base.CPUPlace):
            warnings.warn(
                "Parameters are on CPU, and will be transferred to GPU "
                "automatically by data transform."
            )

        outs_opt, lods_opt = self._run_program(executor, opt_program)
        self.assertTrue(
            len(self.fetch_list) == len(outs_opt),
            f"Checking the number of fetches failed. Expected: {len(self.fetch_list)}, Received: {len(outs_opt)}",
        )
        for i in range(len(self.fetch_list)):
            is_allclose = np.allclose(outs_opt[i], outs[i], atol=atol)
            if not is_allclose:
                a = outs_opt[i]
                b = outs[i]
                diff_mat = np.abs(a - b) / np.abs(a)
                max_diff = np.max(diff_mat)
                offset = np.argmax(diff_mat > atol)
                self.assertTrue(
                    is_allclose,
                    f"Output (name: {self.fetch_list[i].name}, shape: {self.fetch_list[i].shape!s}, dtype: {self.fetch_list[i].dtype}) has diff at {place!s}. "
                    f"The maximum diff is {max_diff:e}, first error element is {offset}, "
                    f"expected {a.flatten()[offset].item():e}, "
                    f"but got {b.flatten()[offset].item():e}",
                )

    def _check_fused_ops(self, program):
        '''
        Check the number of specified fused op is equal to the expected
        number.
        '''
        if self.fused_op_type is None or self.num_fused_ops < 0:
            return

        if program is None or program == self.main_program:
            program = self._apply_ir_passes()

        actual_num_fused_ops = 0
        # Ir passes can only be applied to block 0.
        for op in program.block(0).ops:
            if op.type == self.fused_op_type:
                actual_num_fused_ops += 1
        self.assertTrue(
            self.num_fused_ops == actual_num_fused_ops,
            f"Checking of the number of fused operator < {self.fused_op_type} > failed. "
            f"Expected: {self.num_fused_ops}, Received: {actual_num_fused_ops}",
        )

    def check_program(self, program=None):
        '''
        Check whether the optimized program is different from the origin
        program.
        '''
        if program is None or program == self.main_program:
            program = self._apply_ir_passes()

        self._check_fused_ops(program)

        self.assertTrue(
            self.main_program.desc != program.desc,
            "The optimized program and the origin main_program hold the same "
            "desc.",
        )

        self.assertTrue(
            self.main_program.num_blocks == program.num_blocks,
            "The number of blocks of the origin program and the optimized "
            f"program are different ({self.main_program.num_blocks} vs {program.num_blocks}).",
        )

        is_different = False
        for i in range(program.num_blocks):
            if len(self.main_program.block(i).ops) != len(program.block(i).ops):
                # The number of ops in the block i of the origin program and
                # the optimized program is different.
                is_different = True
                break

            # If there are different ops between the origin and optimized program.
            for op in self.main_program.block(i).ops:
                if not self._find_op(op, program, i):
                    is_different = True
                    break

            if len(self.main_program.block(i).vars) != len(
                program.block(i).vars
            ):
                # The number of vars in the block i of the origin program and
                # the optimized program is different.
                is_different = True
                break

            # If there are different vars between the origin and optimized program.
            for name in self.main_program.block(i).vars:
                var = self.main_program.block(i).var(name)
                if not self._find_var(var, program, i):
                    is_different = True
                    break

        self.assertTrue(
            is_different,
            "The optimized program is logically the same with the origin "
            "program.",
        )

    def _find_op(self, specified_op, program, block_id):
        is_find = False
        for op in program.block(block_id).ops:
            if specified_op.type == op.type:
                for name in op.input_names:
                    if op.input(name) != specified_op.input(name):
                        break
                for name in op.output_names:
                    if op.output(name) != specified_op.output(name):
                        break
                for name in op.attr_names:
                    if op.attr(name) != specified_op.attr(name):
                        break
                is_find = True
                break

        return is_find

    def _find_var(self, specified_var, program, block_id):
        if not program.block(block_id).has_var(specified_var.name):
            return False

        var = program.block(block_id).var(specified_var.name)
        if var.type != specified_var.type:
            return False
        if var.dtype != specified_var.dtype:
            return False
        if var.lod_level != specified_var.lod_level:
            return False
        if var.shape != specified_var.shape:
            return False
        if var.persistable != specified_var.persistable:
            return False

        return True
