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
import sys
import tempfile

import numpy as np

import paddle
from paddle.base import core
from paddle.distributed import fleet
from paddle.inference import Config, PrecisionType, create_predictor


def run(op_type, reduce_type, precision):
    fleet.init(is_collective=True)
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    block = main_program.blocks[0]
    with paddle.static.program_guard(main_program, startup_program):
        data = paddle.static.data(name='data', shape=[3, 4], dtype='float32')
        c_data = block.create_var(
            shape=data.shape,
            dtype=data.dtype,
            type=data.type,
            lod_level=data.lod_level,
            persistable=False,
            is_data=False,
            initializer=paddle.nn.initializer.Constant(value=1.0),
        )
        block.append_op(
            type=op_type,
            inputs={'x': data},
            outputs={'out': c_data},
            attrs={
                'ring_id': 0,
                'reduce_type': reduce_type,
                'use_calc_stream': True,
                'use_model_parallel': True,
            },
        )
        out = paddle.static.nn.fc(
            x=c_data,
            size=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )
        mean = paddle.mean(out)
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_program)

    nranks = 2
    current_endpoint = "127.0.0.1:600" + str(fleet.worker_index())
    trainer_endpoints = ["127.0.0.1:6000", "127.0.0.1:6001"]

    dist_config = core.DistConfig()
    dist_config.set_carrier_id("inference")
    dist_config.set_endpoints(trainer_endpoints, current_endpoint)
    dist_config.set_ranks(nranks, fleet.worker_index())
    dist_config.enable_dist_model(True)

    with tempfile.TemporaryDirectory(prefix="allreduce_") as tmpdir:
        paddle.static.save_inference_model(
            os.path.join(tmpdir, "model"),
            [data],
            [mean],
            exe,
            program=main_program,
        )
        config = Config(
            os.path.join(tmpdir, "model.pdmodel"),
            os.path.join(tmpdir, "model.pdiparams"),
        )
        config.enable_memory_optim()
        config.enable_use_gpu(1000, fleet.worker_index())
        config.set_dist_config(dist_config)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=1,
            precision_mode=PrecisionType.Half
            if precision == "fp16"
            else PrecisionType.Int8,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {"data": [3, 4]}, {"data": [3, 4]}, {"data": [3, 4]}
        )
        predictor = create_predictor(config)
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle("data")
        input_tensor.reshape([3, 4])
        input_tensor.copy_from_cpu(np.ones([3, 4]).astype(np.float32))
        predictor.run()
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # This script just be called by test_trt_convert_all_reduce.py
        sys.exit(0)
    op_type = sys.argv[1]
    reduce_type = sys.argv[2]
    precision = sys.argv[3]
    run(op_type, reduce_type, precision)
