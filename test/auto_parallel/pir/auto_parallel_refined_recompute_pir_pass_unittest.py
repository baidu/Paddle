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

from auto_parallel_recompute_pir_pass_unittest import TestRecomputeLlamaAuto

import paddle


class TestRefinedRecomputeLlamaAuto(TestRecomputeLlamaAuto):

    def run_test_cases(self):
        self.config.recompute = True
        self.config.recompute_granularity = "full"

        self.strategy._recompute.enable = True
        losses_0, model_0 = self.run_llama(self.config)

        self.strategy._recompute.refined_ops_patterns = [
            {
                "main_ops": ["matmul"],
                "num": -1,
                "pre_ops": ["multiply"],
                "suf_ops": [],
            }
        ]
        refined_ops_pattern = self.strategy._recompute.refined_ops_patterns[0]
        refined_ops_pattern["num"] = 0
        losses_0, model_0 = self.run_llama(self.config)
        max_mem_allocated_0, max_mem_reserved_0 = self.get_mem_message()

        refined_ops_pattern["num"] = 1
        losses_1, model_1 = self.run_llama(self.config)
        max_mem_allocated_1, max_mem_reserved_1 = self.get_mem_message()

        refined_ops_pattern["num"] = 2
        losses_2, model_2 = self.run_llama(self.config)
        max_mem_allocated_2, max_mem_reserved_2 = self.get_mem_message()

        self.config.recompute = False
        self.strategy._recompute.enable = False
        base_losses, base_model = self.run_llama(self.config)
        max_mem_allocated_base = paddle.device.cuda.max_memory_allocated()
        max_mem_reserved_base = paddle.device.cuda.max_memory_reserved()

        # check loss
        self.check_loss(base_losses, losses_0)
        self.check_loss(base_losses, losses_1)
        self.check_loss(base_losses, losses_2)

        # check program
        base_prog = base_model.dist_main_program()
        prog_0 = model_0.dist_main_program()
        prog_1 = model_1.dist_main_program()
        prog_2 = model_2.dist_main_program()
        segment_num_base, bwd_rc_op_num_base = self.get_recompute_message(
            base_prog
        )

        segment_num_0, bwd_rc_op_num_0 = self.get_recompute_message(prog_0)
        segment_num_1, bwd_rc_op_num_1 = self.get_recompute_message(prog_1)
        segment_num_2, bwd_rc_op_num_2 = self.get_recompute_message(prog_2)

        # check segment number
        assert segment_num_base == 0
        assert segment_num_0 == 4
        assert segment_num_1 == 4
        assert segment_num_2 == 4

        # check recompute op number
        assert bwd_rc_op_num_base == 0
        assert bwd_rc_op_num_0 == 296
        assert bwd_rc_op_num_1 == 292
        assert bwd_rc_op_num_2 == 288

        # memory check
        assert max_mem_reserved_0 < max_mem_reserved_1
        assert max_mem_reserved_1 < max_mem_reserved_2
        assert max_mem_reserved_2 < max_mem_reserved_base

        assert max_mem_allocated_0 < max_mem_allocated_1
        assert max_mem_allocated_1 < max_mem_allocated_2
        assert max_mem_allocated_2 < max_mem_allocated_base


if __name__ == '__main__':
    TestRefinedRecomputeLlamaAuto().run_test_cases()
