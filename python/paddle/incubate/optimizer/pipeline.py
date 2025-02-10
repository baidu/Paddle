# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import warnings
from collections import defaultdict
from functools import cmp_to_key, reduce

import numpy as np

import paddle
from paddle.base import core, unique_name
from paddle.base.framework import (
    Parameter,
    Program,
    default_startup_program,
    in_dygraph_mode,
)

__all__ = []


class PipelineOptimizer:
    """
        :api_attr: Static Graph

    Pipeline Optimizer: Make a program to run as pipeline, that is splitting a
    program into multiple sections (sub-programs) and each section run on a
    device to enable the training of large scale models and the use of
    heterogeneous devices. Meanwhile, all sections run in the stype of pipeline.

    Args:
        optimizer (Optimizer): The optimizer to use, such as SGD.
        num_microbatches (int): Number of microbatches. [Optional. Default:1].
        start_cpu_core_id (int): The first cpu core id to use. [Optional. Default:0].

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.base as base
            >>> import paddle.base.layers as layers
            >>> import numpy as np

            >>> paddle.enable_static()
            >>> with base.device_guard("gpu:0"):
            ...     x = paddle.static.data(name='x', shape=[-1, 1], dtype='int64')
            ...     y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
            ...     data_loader = base.io.DataLoader.from_generator(
            ...         feed_list=[x, y],
            ...         capacity=64,
            ...         use_double_buffer=True,
            ...         iterable=False)

            ...     emb_x = layers.embedding(input=x, param_attr=base.ParamAttr(name="embx"), size=[10,2], is_sparse=False)
            ...     emb_y = layers.embedding(input=y, param_attr=base.ParamAttr(name="emby",learning_rate=0.9), size=[10,2], is_sparse=False)

            >>> with base.device_guard("gpu:1"):
            ...     concat = layers.concat([emb_x, emb_y], axis=1)
            ...     fc = paddle.static.nn.fc(x=concat, name="fc", size=1, num_flatten_dims=1, bias_attr=False)
            ...     loss = paddle.mean(fc)
            >>> optimizer = paddle.optimizer.SGD(learning_rate=0.5)
            >>> optimizer = paddle.incubate.optimizer.PipelineOptimizer(optimizer)
            >>> optimizer.minimize(loss)

            >>> def train_reader():
            ...     for _ in range(4):
            ...         x = np.random.random(size=[1]).astype('int64')
            ...         y = np.random.random(size=[1]).astype('int64')
            ...         yield x, y
            >>> data_loader.set_sample_generator(train_reader, batch_size=1)

            >>> place = paddle.CUDAPlace(0)
            >>> exe = paddle.static.Executor(place)
            >>> exe.run(paddle.static.default_startup_program())
            >>> batch_size = 1
            >>> data_loader.start()
            >>> exe.train_from_dataset(
            ...         paddle.static.default_main_program())
            >>> data_loader.reset()
    """

    def __init__(self, optimizer, num_microbatches=1, start_cpu_core_id=0):
        self._device = 'cpu'
        if core.is_compiled_with_cuda():
            self._device = "gpu"
        if in_dygraph_mode():
            raise Exception("In dygraph, don't support PipelineOptimizer.")
        valid_optimizers = (
            paddle.optimizer.Optimizer,
            paddle.static.amp.decorator.OptimizerWithMixedPrecision,
        )
        if not isinstance(optimizer, valid_optimizers):
            raise ValueError(
                "The 'optimizer' parameter for "
                "PipelineOptimizer must be an instance of "
                f"{valid_optimizers}, but the given type is {type(optimizer)}."
            )
        self._optimizer = optimizer

        # Get the original optimizer defined by users, such as SGD
        self._origin_optimizer = self._optimizer
        while hasattr(self._origin_optimizer, "inner_opt"):
            self._origin_optimizer = self._origin_optimizer.inner_opt

        assert (
            num_microbatches >= 1
        ), "num_microbatches must be a positive value."
        self._num_microbatches = num_microbatches
        assert (
            start_cpu_core_id >= 0
        ), "start_cpu_core_id must be a non-negative integer."
        self._start_cpu_core_id = start_cpu_core_id
        self._place_list = None
        op_maker = core.op_proto_and_checker_maker
        self._op_role = op_maker.OpRole
        self._op_role_key = op_maker.kOpRoleAttrName()
        self._op_role_var_key = op_maker.kOpRoleVarAttrName()
        self._op_device_key = op_maker.kOpDeviceAttrName()
        self._param_device_map = None
        self._pipeline_pair = []
        self._pp_ring_map = {}
        self.output_var_to_op = None
        self.input_var_to_op = None

    # insert allreduce op to sync global information for global
    # gradient clip and amp
    def _insert_allreduce_op(self, op_idx, block):
        """
        Insert allreduce op to sync global information for global
        gradient clip and amp.
        """
        op = block.ops[op_idx]
        out_name = op.desc.output_arg_names()[0]
        out_var = block.var(out_name)
        offset = 0
        if op.type == "reduce_any":
            # cast the bool var to int32 to use allreduce_max op
            temp_var_name = unique_name.generate(out_name + "_cast_int32")
            temp_var = block.create_var(
                name=temp_var_name, shape=[1], dtype="int32"
            )
            block._insert_op(
                op_idx + 1 + offset,
                type='cast',
                inputs={'X': out_var},
                outputs={'Out': temp_var},
                attrs={
                    'in_dtype': out_var.dtype,
                    'out_dtype': temp_var.dtype,
                    self._op_role_key: self._op_role.Optimize,
                },
            )
            offset += 1
        block._insert_op(
            op_idx + 1 + offset,
            type='all_reduce',
            inputs={'x': temp_var if op.type == "reduce_any" else out_var},
            outputs={'out': temp_var if op.type == "reduce_any" else out_var},
            attrs={
                'ring_id': self.global_ring_id,
                self._op_role_key: self._op_role.Optimize,
                'reduce_type': (
                    paddle.distributed.ReduceOp.MAX
                    if op.type == "reduce_any"
                    else paddle.distributed.ReduceOp.SUM
                ),
            },
        )
        offset += 1
        if op.type == "reduce_any":
            block._insert_op(
                op_idx + 1 + offset,
                type='cast',
                inputs={'X': temp_var},
                outputs={'Out': out_var},
                attrs={
                    'in_dtype': temp_var.dtype,
                    'out_dtype': out_var.dtype,
                    self._op_role_key: self._op_role.Optimize,
                },
            )
            offset += 1
        return offset

    def _create_vars(self, block, ori_block):
        # Create vars for block, copied from ori_block
        used_var_set = set()
        added_op_num = 0
        op_idx = 0
        op_size = block.desc.op_size()
        while op_idx < op_size + added_op_num:
            # Whether to insert allreduce_sum or allreduce_max op.
            # For amp and global gradient clip strategies, we should
            # get the global information, so allreduce op is needed.
            should_insert = False
            op = block.ops[op_idx]
            # For op process vars on all devices, remove its input
            # vars not in this block
            reserved_x = []
            if op.type == 'reduce_any' and self._is_optimize_op(op):
                should_insert = True
            elif op.type == 'concat' and self._is_optimize_op(op):
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
            elif op.type == 'update_loss_scaling':
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
                op.desc.set_output('Out', reserved_x)
            elif op.type == 'check_finite_and_unscale':
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
                op.desc.set_output('Out', reserved_x)
                if len(reserved_x) == 0:
                    block._remove_op(op_idx)
                    op_size -= 1
                    continue
            elif op.type == 'sum' and self._is_gradient_clip_op(op):
                for input_name in op.desc.input("X"):
                    if block._find_var_recursive(input_name):
                        reserved_x.append(input_name)
                op.desc.set_input('X', reserved_x)
                should_insert = True

            vars = op.desc.input_arg_names() + op.desc.output_arg_names()
            for var in vars:
                # a var whose name contains "blocking_queue"
                # only exists in startup program
                if var in used_var_set or "_blocking_queue" in var:
                    continue
                used_var_set.add(var)
                if block._find_var_recursive(str(var)):
                    continue
                source_var = ori_block._var_recursive(str(var))
                if source_var.type == core.VarDesc.VarType.READER:
                    dest_var = block.create_var(
                        name=var,
                        type=core.VarDesc.VarType.READER,
                        persistable=source_var.persistable,
                    )
                elif isinstance(source_var, Parameter):
                    dest_var = block.create_parameter(
                        name=source_var.name,
                        shape=source_var.shape,
                        dtype=source_var.dtype,
                        type=source_var.type,
                        lod_level=source_var.lod_level,
                        stop_gradient=source_var.stop_gradient,
                        trainable=source_var.trainable,
                        optimize_attr=source_var.optimize_attr,
                        regularizer=source_var.regularizer,
                        error_clip=source_var.error_clip,
                    )
                else:
                    dest_var = block._clone_variable(source_var, False)
                self._clone_var_attr(dest_var, source_var)
            # When use with sharding, allreduce_sum and allreduce_max
            # used for global gradient clip and amp will be added by sharding.
            op_idx += 1
            if self.use_sharding or not should_insert:
                continue
            inserted_ops = self._insert_allreduce_op(op_idx - 1, block)
            added_op_num += inserted_ops
            op_idx += inserted_ops
        block._sync_with_cpp()

    def _is_loss_grad_op(self, op):
        assert self._op_role_key in op.attr_names
        op_role = int(op.attr(self._op_role_key))
        return op_role & int(self._op_role.Backward) and op_role & int(
            self._op_role.Loss
        )

    def _is_forward_op(self, op):
        return self._op_role_key in op.attr_names and (
            int(op.attr(self._op_role_key)) == int(self._op_role.Forward)
        )

    def _is_backward_op(self, op):
        return self._op_role_key in op.attr_names and (
            int(op.attr(self._op_role_key)) & int(self._op_role.Backward)
        )

    def _is_loss_op(self, op):
        assert self._op_role_key in op.attr_names
        return int(op.attr(self._op_role_key)) == int(self._op_role.Loss)

    def _is_optimize_op(self, op):
        return self._op_role_key in op.attr_names and (
            int(op.attr(self._op_role_key)) & int(self._op_role.Optimize)
        )

    def _is_update_op(self, op):
        return (
            'Param' in op.input_names
            and 'Grad' in op.input_names
            and ("LearningRate" in op.input_names)
        )

    def _split_program(self, main_program, devices):
        """
        Split a program into sections according to devices that ops run on.
        The op whose op_device attr is "gpu:all" is copied to all sections.

        Args:
            main_program (Program): the main program
            devices: all used devices
        """
        # Map from device to its corresponding section program info
        device_program_map = defaultdict(Program)

        block = main_program.block(0)
        for op in block.ops:
            device = op.attr(self._op_device_key)
            # Copy ops whose op_device set to "gpu:all" to all sections.
            if device == f"{self._device}:all":
                for device in devices:
                    program = device_program_map[device]
                    op_desc = op.desc
                    ap_op = program.global_block().desc.append_op()
                    ap_op.copy_from(op_desc)
                    ap_op._set_attr(self._op_device_key, "")
            else:
                program = device_program_map[device]
                op_desc = op.desc
                ap_op = program.global_block().desc.append_op()
                ap_op.copy_from(op_desc)
                ap_op._set_attr(self._op_device_key, "")

        program_list = []
        for key in devices:
            program = device_program_map[key]
            program._sync_with_cpp()
            program_list.append(program)

        return program_list

    def _get_op_device_for_startup_program(self, var_name):
        """
        For adam optimizer, it will add accumulators and initialize them
        with fill_constant, and force the op device to cpu. Hence, we should
        get the real op_device attribute of the fill_constant as the device
        where the corresponding parameters on.
        """
        assert "beta1_pow_acc" in var_name or "beta2_pow_acc" in var_name, (
            'For accumulators for Adam, the name must contain beta1_pow_acc '
            'or beta2_pow_acc.'
        )
        param_name = var_name[0 : var_name.index('_beta')]
        device = self._param_device_map[param_name]
        return device

    def _split_startup_program(self, startup_program, device_id):
        block = startup_program.global_block()
        new_startup_program = Program()
        for op in block.ops:
            device = op.attr(self._op_device_key)
            if device == "cpu":
                assert op.type == "fill_constant", (
                    "For ops in startup program with the op_device attribute "
                    "of cpu, they must be of type fill_constant."
                )
                output_var = op.output_arg_names[0]
                device = self._get_op_device_for_startup_program(output_var)

            if device:
                device_index = int(device.split(':')[1])
            else:
                # LR related ops
                device = None
            if device and device_index != device_id:
                continue
            op_desc = op.desc
            ap_op = new_startup_program.global_block().desc.append_op()
            ap_op.copy_from(op_desc)
            ap_op._set_attr(self._op_device_key, "")
        new_startup_program._sync_with_cpp()
        self._create_vars(new_startup_program.global_block(), block)
        return new_startup_program

    def _find_post_op(self, index, var_name):
        """
        Find the post op that has variable named var_name as input.
        """
        # bugfix for uniform hybrid parallelism
        if '.cast_fp32' in var_name:
            var_name = var_name.replace('.cast_fp32', '')
        if '.cast_fp16' in var_name:
            var_name = var_name.replace('.cast_fp16', '')

        post_ops = self.input_var_to_op[var_name]
        if post_ops is None:
            return None
        result_op = None
        for post_op, post_idx in reversed(post_ops):
            if post_idx > index:
                result_op = post_op
                break
        return result_op

    def _find_prev_op(self, index, var_name):
        """
        Find the previous op of op with index that outputs
        variable named var_name.
        """
        prev_ops = self.output_var_to_op[var_name]
        if prev_ops is None:
            return None
        result_op = None
        for prev_op, prev_idx in reversed(prev_ops):
            if prev_idx < index:
                result_op = prev_op
                break
        return result_op

    def _rename_arg(self, op, old_name, new_name):
        op._rename_input(old_name, new_name)
        op._rename_output(old_name, new_name)

    def _create_var(self, block, ref_var, name, dtype=None):
        """
        Create a new var for block, which has the same type,
        shape and dtype as ref_var, then rename it with the
        name `name`.
        """
        new_var = block.create_var(
            name=name,
            shape=ref_var.shape,
            dtype=ref_var.dtype if dtype is None else dtype,
            type=ref_var.type,
            lod_level=ref_var.lod_level,
            persistable=ref_var.persistable,
            is_data=ref_var.is_data,
            need_check_feed=ref_var.desc.need_check_feed(),
        )
        self._clone_var_attr(new_var, ref_var)
        return new_var

    def _clone_var_attr(self, dest, src):
        dest.stop_gradient = src.stop_gradient
        if hasattr(src, 'is_distributed'):
            dest.is_distributed = src.is_distributed

    def _strip_grad_suffix(self, name):
        """
        Strip the grad suffix from the given variable name
        """
        pos = name.find(core.grad_var_suffix())
        return name[:pos] if pos != -1 else name

    def _append_grad_suffix(self, name):
        """
        Append grad suffix to the given variable name
        """
        return name + core.grad_var_suffix()

    def _get_op_device_attr(self, op):
        """
        Get the op_device attribute of a op.
        """
        device = (
            op.attr(self._op_device_key)
            if op.has_attr(self._op_device_key)
            else None
        )
        if device:
            assert device[0:3] == 'gpu', (
                "Now, only gpu devices are "
                "supported in pipeline parallelism."
            )
        return device

    def _add_op_device_attr_for_op(self, op, idx, block):
        """
        Add op_device attribute for ops that have not that attribute set.
        We use "gpu:all" to represent the op should be put on all
        sub-programs, such as lr-related ops. Note that: "gpu:all"
        is only used by pipeline as an indicator.
        """
        lrsched_role = int(self._op_role.LRSched)
        if op.attr(self._op_role_key) == lrsched_role:
            # For LRSched ops, we should put them on all sub-programs to
            # make sure each sub-program update the lr correctly
            op._set_attr(self._op_device_key, f"{self._device}:all")
        # bugfix in hybrid parallelism
        elif op.type == "sum" and self._is_backward_op(op):
            # For sum ops that compute the sum of @RENAMED@ vars
            for name in op.desc.input_arg_names():
                assert (
                    '@RENAME@' in name
                ), "The op must be sum used to accumulate renamed vars."
            assert len(op.desc.output_arg_names()) == 1
            out_name = op.desc.output_arg_names()[0]
            post_op = self._find_post_op(idx, out_name)
            assert post_op.has_attr(
                'op_device'
            ), f"{post_op.type} has no op_device attr for var {out_name}"
            device = post_op.attr(self._op_device_key)
            assert device, "The post op must have op_device set."
            op._set_attr(self._op_device_key, device)
        elif (op.type == "cast" or op.type == "scale") and (
            self._is_backward_op(op) or self._is_forward_op(op)
        ):
            prev_op = self._find_prev_op(idx, op.desc.input("X")[0])
            op._set_attr(self._op_device_key, prev_op.attr(self._op_device_key))
        elif op.type == "memcpy" and not self._is_optimize_op(op):
            # for checkpoint offloading
            assert (
                len(op.input_arg_names) == 1 and len(op.output_arg_names) == 1
            )
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            if '@Fetch' in output_name:
                post_op = self._find_post_op(idx, output_name)
                op._set_attr(
                    self._op_device_key, post_op.attr(self._op_device_key)
                )
            else:
                prev_op = self._find_prev_op(idx, op.desc.input("X")[0])
                op._set_attr(
                    self._op_device_key, prev_op.attr(self._op_device_key)
                )
        elif self._is_loss_op(op):
            # For loss * loss_scaling op added by AMP
            offset = 1
            while not block.ops[idx + offset].has_attr(
                self._op_device_key
            ) or not block.ops[idx + offset].attr(self._op_device_key):
                offset += 1
            device = block.ops[idx + offset].attr(self._op_device_key)
            assert device, "Please put you program within device_guard scope."
            for i in range(offset):
                block.ops[idx + i]._set_attr(self._op_device_key, device)
        elif self._is_optimize_op(op) and op.type == "cast":
            # For fp16-->fp32 cast added by AMP
            grad_name = op.output('Out')
            assert len(grad_name) == 1
            param_name = self._strip_grad_suffix(grad_name[0])
            device = self._param_device_map[param_name]
            op._set_attr(self._op_device_key, device)
        elif self._is_gradient_clip_op(op) or self._is_regularization_op(op):
            # For gradient clip and regularization ops, we set their op_device
            # attribute to the device where their corresponding parameters on.
            assert self._op_role_var_key in op.attr_names, (
                "gradient_clip "
                "and regularization ops must have op_role_var attribute."
            )
            op_role_var = op.attr(self._op_role_var_key)
            assert len(op_role_var) == 2, (
                "op_role_var for gradient_clip "
                "regularization ops must have two elements."
            )
            param_name = op_role_var[0]
            device = self._param_device_map[param_name]
            # For sum op added by global gradient clip, it must be
            # put on all devices
            if (
                op.type == 'sum'
                or op.type == 'sqrt'
                or op.type == 'fill_constant'
                or op.type == 'elementwise_max'
                or op.type == 'elementwise_div'
            ):
                device = f"{self._device}:all"
            op._set_attr(self._op_device_key, device)
        elif op.type == "alloc_float_status" or op.type == "clear_float_status":
            op._set_attr(self._op_device_key, f"{self._device}:all")
            # NOTE(wangxi): NPU should only clear the float status
            # once at each batch step
            op._set_attr(self._op_role_key, self._op_role.LRSched)

            float_status_name = op.output_arg_names[0]
            float_status_var = block.var(float_status_name)
            # FIXME(wangxi): pipeline lr schedule will exec on sub_scope(0)
            # while update will exec on sub_scope(last_micro_step), should
            # set persistable to use global scope
            float_status_var.persistable = True
        else:
            other_known_ops = [
                'update_loss_scaling',
                'reduce_any',
                'concat',
                'sum',
                'check_finite_and_unscale',
                'memcpy',
            ]
            assert op.type in other_known_ops, (
                "For other ops without "
                f"op_device set, they must be one of {other_known_ops}, but it "
                f"is {op.type}"
            )
            assert self._is_optimize_op(op)
            op._set_attr(self._op_device_key, f"{self._device}:all")

    def _add_op_device_attr(self, block):
        """
        Add op_device attribute for ops in block that have
        not that attribute set.
        """
        for idx, op in enumerate(list(block.ops)):
            if (
                op.type == "create_py_reader"
                or op.type == "read"
                or op.type == "create_double_buffer_reader"
            ):
                # Copy read related ops to all section to make them exit
                # after each epoch.
                # We use "gpu:all" to represent the op should be put on all
                # sub-programs, such as lr-related ops. Note that: "gpu:all"
                # is only used by pipeline as an indicator.
                op._set_attr(self._op_device_key, f"{self._device}:all")
                continue
            # op_device attribute has been set
            if self._get_op_device_attr(op):
                continue
            self._add_op_device_attr_for_op(op, idx, block)

    def _check_validation(self, block):
        """
        Check whether ops in a block have both the op_device and the
        op_role attributes set.
        Then, return all devices in order.
        """
        device_list = []
        # Section worker only supports the following op_role
        valid_op_role_value = [
            int(self._op_role.LRSched),
            int(self._op_role.Forward),
            int(self._op_role.Backward),
            int(self._op_role.Loss),
            int(self._op_role.Optimize),
            int(self._op_role.Backward) | int(self._op_role.Loss),
        ]
        for op in block.ops:
            if not op._has_kernel(op.type):
                assert op.type == "conditional_block" and (
                    op.attr(self._op_role_key) == int(self._op_role.LRSched)
                ), (
                    "Now, the only supported op without kernel is "
                    "conditional_block, and its op role must be LRSched."
                )
            assert op.has_attr(
                self._op_role_key
            ), f"op ({op.type}) has no {self._op_role_key} attribute."
            op_role = op.attr(self._op_role_key)
            assert (
                int(op_role) in valid_op_role_value
            ), f"op_role {op_role} for op {op.type} must be one of {valid_op_role_value}"

            assert op.has_attr(
                self._op_device_key
            ), f"op ({op.type}) has no {self._op_device_key} attribute."

            device = op.attr(self._op_device_key)
            assert device, (
                "op_device attribute for op " f"{op.type} has not been set."
            )
            if device == f"{self._device}:all":
                continue

            dev_type = device.split(':')[0]
            assert dev_type == "gpu", (
                "Now only gpu devices are supported "
                "for pipeline parallelism."
            )

            if device not in device_list:
                device_list.append(device)

        return device_list

    def _insert_sendrecv_ops_for_boundaries(self, block):
        """
        Insert a pair of send and recv ops for every two
        consecutive ops on different devices.
        """
        # A map from var to device where op takes it as input,
        # avoiding multiple send and recv ops.
        input_var_to_device = {}
        # bugfix hybrid parallelism
        first_optimize_index = None
        for index, op in enumerate(list(block.ops)):
            if self._is_optimize_op(op):
                first_optimize_index = index
                break
        extra_index_info = {
            'index': 0,
            'first_optimize_index': first_optimize_index,
        }

        for index, op in enumerate(list(block.ops)):
            cur_device = op.attr(self._op_device_key)
            if cur_device == f"{self._device}:all":
                continue
            for var_name in op.input_arg_names:
                var = block.var(var_name)
                # skip data var
                if var.is_data:
                    continue
                prev_device = None

                prev_op = self._find_prev_op(index, var_name)
                if prev_op is None:
                    if var_name not in self._param_device_map:
                        continue
                    prev_device = self._param_device_map[var_name]

                if not prev_device:
                    prev_device = (
                        prev_op.attr(self._op_device_key) if prev_op else None
                    )

                if prev_device is None or prev_device == f"{self._device}:all":
                    continue

                if prev_device == cur_device:
                    continue

                if var_name not in input_var_to_device:
                    input_var_to_device[var_name] = []
                if (cur_device, prev_device) in input_var_to_device[var_name]:
                    continue

                device_type = cur_device.split(':')[0] + ':'

                def _check_stage(cur_id, prev_id):
                    # check send/recv stage valid
                    is_forward = self._is_forward_op(op)
                    is_backward = self._is_backward_op(op)
                    assert is_forward or is_backward, (
                        'send/recv in pipeline should only be inserted in forward or backward,'
                        f'please check the op_role of op={op}'
                    )

                    if is_forward:
                        assert prev_id < cur_id, (
                            "In forward, send/recv can only be passed forward, but now "
                            f"prev_stage={prev_id} great than cur_stage={cur_id}, please check op_device of op={op}"
                        )
                    elif is_backward:
                        assert prev_id > cur_id, (
                            "In backward, send/recv can only be passed backward, but now "
                            f"prev_stage={prev_id} less than cur_stage={cur_id}, please check op_device of op={op}"
                        )

                def _insert_send_recv(cur_id, prev_id):
                    cur_dev = device_type + str(cur_id)
                    prev_dev = device_type + str(prev_id)
                    if (cur_dev, prev_dev) in input_var_to_device[var_name]:
                        return

                    if cur_id - prev_id > 1:
                        _insert_send_recv(cur_id - 1, prev_id)
                        _insert_send_recv(cur_id, cur_id - 1)
                        input_var_to_device[var_name].append(
                            (cur_dev, prev_dev)
                        )
                        return
                    elif cur_id - prev_id < -1:
                        _insert_send_recv(cur_id + 1, prev_id)
                        _insert_send_recv(cur_id, cur_id + 1)
                        input_var_to_device[var_name].append(
                            (cur_dev, prev_dev)
                        )
                        return

                    assert abs(cur_id - prev_id) == 1
                    input_var_to_device[var_name].append((cur_dev, prev_dev))

                    op_role = op.attr(self._op_role_key)
                    var = block.vars[var_name]
                    pair = (prev_id, cur_id)
                    # 1000 is just a magic number
                    pair_key = prev_id * 1000 + cur_id
                    if pair not in self._pipeline_pair:
                        self._pipeline_pair.append(pair)
                        self._pp_ring_map[pair_key] = self.ring_id
                        ring_id = self.ring_id
                        self.ring_id += 1
                    else:
                        ring_id = self._pp_ring_map[pair_key]

                    if self.schedule_mode == 'F-then-B':  # F-then-B
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='send_v2',
                            inputs={'X': var},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': True,
                                'peer': 1,
                                'ring_id': ring_id,
                            },
                        )
                        extra_index_info['index'] += 1
                        var_shape = list(var.shape)
                        var_shape[0] = (
                            self.micro_batch_size
                            if var_shape[0] < 0
                            else var_shape[0]
                        )
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='recv_v2',
                            outputs={'Out': [var]},
                            attrs={
                                'out_shape': var_shape,
                                'dtype': var.dtype,
                                self._op_device_key: cur_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': True,
                                'peer': 0,
                                'ring_id': ring_id,
                            },
                        )
                        extra_index_info['index'] += 1
                    elif self.schedule_mode == '1F1B':  # 1F1B
                        var_shape = list(var.shape)
                        var_shape[0] = (
                            self.micro_batch_size
                            if var_shape[0] < 0
                            else var_shape[0]
                        )

                        numel = np.prod(var_shape)
                        use_mp = (self.mp_degree > 1) and (
                            numel % self.mp_degree == 0
                        )

                        if 'subprog' in var.name:
                            # For recompute, if the checkpoints var is layer_norm_6.tmp_2
                            # this var will be sent twice, layer_norm_6.tmp_2 for forward pass,
                            # layer_norm_6.tmp_2.subprog_* for recompute pass.
                            # We can store the first sent var and copy the value to the
                            # second one to reduce one send/recv op.
                            # The origin_ckpt_name is layer_norm_6.tmp_2, which will be used
                            # to find the stored var for the forward pass.
                            origin_name = var.name.split('subprog')[0][0:-1]
                            associate_var = block.var(origin_name)
                            block._insert_op_without_sync(
                                index=index + extra_index_info['index'],
                                type='assign',
                                inputs={'X': [associate_var]},
                                outputs={'Out': [var]},
                                attrs={
                                    'out_shape': var_shape,
                                    'dtype': var.dtype,
                                    self._op_device_key: cur_dev,
                                    self._op_role_key: op_role,
                                    'use_calc_stream': True,
                                },
                            )
                            extra_index_info['index'] += 1
                            return

                        _check_stage(cur_id, prev_id)

                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type='c_sync_calc_stream',
                            inputs={'X': [var]},
                            outputs={'Out': [var]},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: op_role,
                            },
                        )
                        extra_index_info['index'] += 1
                        prefix_name = var.name.split('@')[0]
                        prefix_var = block.var(prefix_name)
                        is_param = (
                            True if isinstance(prefix_var, Parameter) else False
                        )
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type=(
                                'send_v2'
                                if not use_mp or is_param
                                else 'partial_send'
                            ),
                            inputs={'X': var},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': False,
                                'ring_id': ring_id,
                                'peer': 1,
                                # if send_v2, num&id attr is not in op_attrs, will not insert
                                'num': self.mp_degree,
                                'id': self.mp_rank,
                            },
                        )
                        extra_index_info['index'] += 1
                        insert_index = None
                        if int(op_role) == int(self._op_role.Backward):
                            insert_index = extra_index_info[
                                'first_optimize_index'
                            ]
                            new_op_role = self._op_role.Optimize
                        else:
                            insert_index = index
                            new_op_role = self._op_role.Backward
                        sync_comm_op = block._insert_op_without_sync(
                            index=insert_index + extra_index_info['index'],
                            type='c_sync_comm_stream',
                            inputs={'X': [var]},
                            outputs={'Out': [var]},
                            attrs={
                                self._op_device_key: prev_dev,
                                self._op_role_key: new_op_role,
                                'ring_id': ring_id,
                            },
                        )
                        if int(op_role) == int(self._op_role.Forward):
                            sync_comm_op._set_attr('pipeline_flag', '')
                            extra_index_info['index'] += 1
                        block._insert_op_without_sync(
                            index=index + extra_index_info['index'],
                            type=(
                                'recv_v2'
                                if not use_mp or is_param
                                else 'partial_recv'
                            ),
                            outputs={'Out': [var]},
                            attrs={
                                'out_shape': var_shape,
                                'dtype': var.dtype,
                                self._op_device_key: cur_dev,
                                self._op_role_key: op_role,
                                'use_calc_stream': True,
                                'peer': 0,
                                'ring_id': ring_id,
                                # if recv_v2, num&id attr is not in op_attrs, will not insert
                                'num': self.mp_degree,
                                'id': self.mp_rank,
                            },
                        )
                        extra_index_info['index'] += 1
                        if use_mp and not is_param:
                            block._insert_op_without_sync(
                                index=index + extra_index_info['index'],
                                type='partial_allgather',
                                inputs={'X': [var]},
                                outputs={'Out': [var]},
                                attrs={
                                    self._op_device_key: cur_dev,
                                    self._op_role_key: op_role,
                                    'use_calc_stream': True,
                                    'ring_id': 0,
                                    # if recv_v2, num&id attr is not in op_attrs, will not insert
                                    'nranks': self.mp_degree,
                                    'rank': self.mp_rank,
                                },
                            )
                            extra_index_info['index'] += 1
                    else:
                        raise ValueError(
                            "Now only 'F-then-B' and '1F1B' are supported."
                            f"The given value is {self.schedule_mode}."
                        )

                _insert_send_recv(
                    int(cur_device.split(':')[1]),
                    int(prev_device.split(':')[1]),
                )
        block._sync_with_cpp()

    def _insert_loss_scale(self, block):
        """
        Scale the loss corresponding to number of micro-batches.
        """
        if self._num_microbatches == 1:
            return
        for index, op in reversed(tuple(enumerate(list(block.ops)))):
            if self._is_loss_grad_op(op):
                assert op.type == 'fill_constant', (
                    "loss_grad_op must be fill_constant op, "
                    f"but this op is {op.type}"
                )
                assert op.has_attr('value')
                loss_scale = float(op.attr('value'))
                loss_scale = loss_scale / self._num_microbatches
                op._set_attr('value', loss_scale)
                break

    def _rename_gradient_var_name(self, block):
        for index, op in enumerate(block.ops):
            if not self._is_optimize_op(op):
                continue
            input_names = op.input_arg_names
            output_names = op.output_arg_names
            in_out_names = input_names + output_names
            if op.type == 'cast' or op.type == "c_sync_comm_stream":
                continue
            # append "MERGED" to the names of parameter gradients,
            # and modify the op_role_var attribute (by rename_arg func).
            for name in in_out_names:
                if core.grad_var_suffix() not in name:
                    continue
                param_name = name.strip(core.grad_var_suffix())
                new_grad_name = name + "@MERGED"
                self._rename_arg(op, name, new_grad_name)

    def _accumulate_gradients(
        self, block, pp_allreduce_in_optimize=False, strategy=None, shard=None
    ):
        """
        Create a new merged gradient for each parameter and accumulate the
        corresponding gradient to it.
        """
        fp16_allreduce = strategy.fp16_allreduce if strategy else False
        if strategy and strategy.fuse_grad_merge:
            fused_gradient_names = self._accumulate_gradients_with_fuse(
                block, fp16_allreduce, strategy.fuse_grad_size_in_MB, shard
            )
            return fused_gradient_names

        merged_gradient_names = []
        first_opt_op_idx = None

        merged_suffix = '@MERGED@FP16' if fp16_allreduce else '@MERGED'
        dtype = paddle.float16 if fp16_allreduce else None

        for index, op in reversed(tuple(enumerate(list(block.ops)))):
            # remove the cast op of fp16 grad to fp32 grad
            if self._is_optimize_op(op) and op.type == 'cast':
                in_name = op.input_arg_names[0]
                out_name = op.output_arg_names[0]
                if out_name.strip('@GRAD') in self._param_device_map:
                    assert in_name.replace('.cast_fp16', '') == out_name
                    block._remove_op(index)
                    continue

            if self._is_backward_op(op) and first_opt_op_idx is None:
                first_opt_op_idx = index + 1
                # maybe have no optimize
                # if first_opt_op_idx == len(block.ops): return

            if self._is_backward_op(op) and (
                self._op_role_var_key in op.attr_names
            ):
                op_role_var = op.attr(self._op_role_var_key)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                for i in range(0, len(op_role_var), 2):
                    offset = 0
                    param_name = op_role_var[i]
                    if not block.has_var(param_name):
                        continue
                    if '@BroadCast' in param_name:
                        continue

                    param_grad_name = param_name + core.grad_var_suffix()
                    merged_param_grad_name = param_grad_name + merged_suffix
                    if not block.has_var(merged_param_grad_name):
                        self._create_var(
                            block,
                            block.vars[param_name],
                            merged_param_grad_name,
                            dtype,
                        )
                    assert block.has_var(merged_param_grad_name)

                    param_grad_var = block.var(param_grad_name)
                    merged_param_grad_var = block.var(merged_param_grad_name)
                    merged_param_grad_var.persistable = True
                    block._insert_op(
                        index=first_opt_op_idx + offset,
                        type='fill_constant',
                        inputs={},
                        outputs={'Out': [merged_param_grad_var]},
                        attrs={
                            'shape': merged_param_grad_var.shape,
                            'dtype': merged_param_grad_var.dtype,
                            'value': float(0),
                            # a trick to run this op once per mini-batch
                            self._op_role_key: self._op_role.Optimize.LRSched,
                        },
                    )
                    offset += 1
                    grad_name = op_role_var[i + 1]
                    grad_var = block.vars[grad_name]

                    is_fp16_grad = 'cast_fp16' in grad_name
                    need_cast = is_fp16_grad is not fp16_allreduce

                    if need_cast:
                        # if fp16_allreduce:
                        #     cast grad to fp16 to accumulate to merged gradient
                        # else:
                        #     cast grad to fp32 to accumulate to merged gradient
                        cast_grad_var_name = param_grad_name + '@TMP'
                        cast_grad_var = self._create_var(
                            block, param_grad_var, cast_grad_var_name, dtype
                        )
                        cast_grad_var.persistable = False
                        block._insert_op(
                            index=first_opt_op_idx + offset,
                            type='cast',
                            inputs={'X': grad_var},
                            outputs={'Out': cast_grad_var},
                            attrs={
                                'in_dtype': grad_var.dtype,
                                'out_dtype': cast_grad_var.dtype,
                                self._op_role_key: self._op_role.Backward,
                            },
                        )
                        offset += 1
                        grad_var = cast_grad_var

                    block._insert_op(
                        index=first_opt_op_idx + offset,
                        type='sum',
                        inputs={'X': [merged_param_grad_var, grad_var]},
                        outputs={'Out': merged_param_grad_var},
                        attrs={
                            self._op_role_key: self._op_role.Backward,
                        },
                    )
                    offset += 1
                    merged_gradient_names.append(merged_param_grad_name)

        if not fp16_allreduce:
            return merged_gradient_names

        first_opt_op_idx = None
        for index, op in reversed(tuple(enumerate(list(block.ops)))):
            if self._is_backward_op(op) and first_opt_op_idx is None:
                first_opt_op_idx = index + 1
                break
        assert first_opt_op_idx is not None

        # insert cast op from fp16->fp32
        # FIXME(wangxi): maybe put in sharding is better, for some grad
        #                is not in sharding device.
        for fp16_grad_name in merged_gradient_names:
            grad_name = fp16_grad_name.replace('@FP16', '')
            param_name = fp16_grad_name.replace('@GRAD@MERGED@FP16', '')

            if not block.has_var(grad_name):
                self._create_var(block, block.vars[param_name], grad_name)
            assert block.has_var(grad_name)

            fp16_grad_var = block.var(fp16_grad_name)
            grad_var = block.var(grad_name)
            grad_var.persistable = False

            block._insert_op(
                index=first_opt_op_idx,
                type='cast',
                inputs={'X': fp16_grad_var},
                outputs={'Out': grad_var},
                attrs={
                    'in_dtype': fp16_grad_var.dtype,
                    'out_dtype': grad_var.dtype,
                    self._op_role_key: self._op_role.Optimize,
                },
            )

        return merged_gradient_names

    def _insert_accumulate_gradients_with_fuse(
        self, main_block, fp16, fused_size, grad_param_pairs, first_opt_op_idx
    ):
        grad_param_pairs = self._sort_grad_param_by_dtype(
            main_block, grad_param_pairs
        )

        grad_param_segments = []
        merged_suffix = '@MERGED@FP16' if fp16 else '@MERGED'
        dtype = paddle.float16 if fp16 else paddle.float32
        cur_size = 0.0
        last_dtype = None
        # split the grad based on dtype and fused size
        for grad, param in grad_param_pairs:
            real_grad = main_block.var(grad)
            # create the gradient merged var for each grad
            merged_grad_var = main_block.create_var(
                name=param + core.grad_var_suffix() + merged_suffix,
                dtype=dtype,
                shape=real_grad.shape,
                persistable=True,
                stop_gradient=False,
            )
            real_param = main_block.var(param)
            if hasattr(real_param, 'is_distributed'):
                merged_grad_var.is_distributed = real_param.is_distributed
            tmp_size = self._get_var_size(real_grad)
            # two strategies for splitting the grad
            # 1. the current segment's size reach the user defined grad_size_in_MB
            # 2. the upcoming grad holds different dtype compared with grads in current segment
            if (
                len(grad_param_segments) == 0
                or cur_size + tmp_size > fused_size
                or real_grad.dtype != last_dtype
            ):
                grad_param_segments.append(
                    ([real_grad], [real_param], [merged_grad_var])
                )
                last_dtype = real_grad.dtype
                cur_size = 0.0
            else:
                grad_param_segments[-1][0].append(real_grad)
                grad_param_segments[-1][1].append(real_param)
                grad_param_segments[-1][2].append(merged_grad_var)
                cur_size += tmp_size

        fused_gradients = []
        fused_merged_gradients = []
        # create fused vars for grad and param
        for grad_param_segment in grad_param_segments:
            grad_segment = grad_param_segment[0]
            merged_grad_segment = grad_param_segment[2]
            fused_grad = main_block.create_var(
                name=f'FusedGrad_{grad_segment[0].name}',
                dtype=grad_segment[0].dtype,
                persistable=False,
                stop_gradient=False,
            )
            # keep the '.cast_fp16' info in the fuse var name
            fused_merged_grad_name_prefix = (
                'FusedMergedGrad.cast_fp16.'
                if merged_grad_segment[0].dtype == paddle.float16
                else 'FusedMergedGrad'
            )
            fused_merged_grad_name = (
                fused_merged_grad_name_prefix
                + f'_{merged_grad_segment[0].name}'
            )
            fused_merged_grad = main_block.create_var(
                name=fused_merged_grad_name,
                dtype=merged_grad_segment[0].dtype,
                persistable=True,
                stop_gradient=False,
            )
            fused_gradients.append(fused_grad)
            fused_merged_gradients.append(fused_merged_grad)

        assert len(fused_gradients) == len(grad_param_segments)
        assert len(fused_merged_gradients) == len(grad_param_segments)

        # insert coalesce op at the start of the backward pass
        # use param as the coalesce input to make sure the two Fused vars are in same shape
        first_back_op_idx = None
        for index, op in enumerate(main_block.ops):
            if self._is_backward_op(op) and first_back_op_idx is None:
                first_back_op_idx = index
                break
        assert first_back_op_idx is not None
        offset = 0
        for i in range(len(grad_param_segments)):
            fused_grad = fused_gradients[i]
            fused_merged_grad = fused_merged_gradients[i]
            grads = grad_param_segments[i][0]
            params = grad_param_segments[i][1]
            merged_grads = grad_param_segments[i][2]
            main_block._insert_op_without_sync(
                first_back_op_idx + offset,
                type="coalesce_tensor",
                inputs={"Input": params},
                outputs={"Output": grads, "FusedOutput": fused_grad},
                attrs={
                    # Explanation of user_defined_size_of_dtype:
                    # In coalesce op, the align size is 256 bytes
                    # the float takes 4 bytes while fp16 takes 2 bytes.
                    # To meet the requirement, 128 fp16 or 64 float will be aligned
                    # Think the total shape of the input tensors if [64],
                    # if the dtype is float, then the shape of the fuse var is [64]
                    # however if the dtype if fp16, the shape of the fuse var is [128],
                    # which will cause the fused vars' shape vary between each other.
                    # To make sure the shape of the fused vars are identical,
                    # we set the dtype of float and fp16 both to 2.
                    # Under this way, the fused vars' shape for float and fp16 are all [128]
                    "user_defined_size_of_dtype": 2,
                    "copy_data": False,
                    "use_align": True,
                    "dtype": grads[0].dtype,
                    self._op_role_key: self._op_role.Backward,
                    # On npu, the nan/inf check login is different with gpu.
                    # If there are some not initialized sections in the fused var,
                    # and the value in those sections are nan/inf, it will trigger the nan/inf check.
                    # To avoid these problematic triggers, set constant is needed for npu
                    "set_constant": core.is_compiled_with_custom_device('npu'),
                    "constant": 0.0,
                },
            )
            offset += 1
            # For the gradient_merged_fused_var, given a init value during the coalesce op
            # this will remove a problematic fill_constant op. This op role of this coalesce
            # is set to be LRSched to make this coalesce (with init) only run once
            main_block._insert_op_without_sync(
                first_back_op_idx + offset,
                type="coalesce_tensor",
                inputs={"Input": params},
                outputs={
                    "Output": merged_grads,
                    "FusedOutput": fused_merged_grad,
                },
                attrs={
                    "user_defined_size_of_dtype": 2,
                    "set_constant": True,
                    "constant": 0.0,
                    "copy_data": False,
                    "use_align": True,
                    "dtype": merged_grads[0].dtype,
                    self._op_role_key: self._op_role.Optimize.LRSched,
                },
            )
            offset += 1

        # insert gradient merge relating ops
        first_opt_op_idx += offset
        offset = 0
        for i in range(len(fused_gradients)):
            fused_grad = fused_gradients[i]
            fused_merged_grad = fused_merged_gradients[i]
            is_fp16_grad = 'cast_fp16' in fused_grad.name
            need_cast = is_fp16_grad is not fp16
            if need_cast:
                # for fp16 allreduce, cast fp32 grad to fp16
                # for fp32 allreduce, cast fp16 grad to fp32
                cast_grad_var_name = fused_grad.name + '@TMP'
                cast_grad_var = main_block.create_var(
                    name=cast_grad_var_name,
                    dtype=dtype,
                    persistable=False,
                    stop_gradient=False,
                )
                main_block._insert_op(
                    index=first_opt_op_idx + offset,
                    type='cast',
                    inputs={'X': fused_grad},
                    outputs={'Out': cast_grad_var},
                    attrs={
                        'in_dtype': fused_grad.dtype,
                        'out_dtype': cast_grad_var.dtype,
                        self._op_role_key: self._op_role.Backward,
                    },
                )
                offset += 1
                fused_grad = cast_grad_var
            main_block._insert_op(
                index=first_opt_op_idx + offset,
                type='sum',
                inputs={'X': [fused_merged_grad, fused_grad]},
                outputs={'Out': fused_merged_grad},
                attrs={self._op_role_key: self._op_role.Backward},
            )
            offset += 1

        if fp16:
            # if using fp16 allreduce, the optimizer needs fp32 grads, cast them back to fp32
            for grad, param in grad_param_pairs:
                real_grad = main_block.var(grad)
                fp16_grad_name = param + core.grad_var_suffix() + '@MERGED@FP16'
                assert main_block.has_var(fp16_grad_name)
                fp16_grad = main_block.var(fp16_grad_name)
                fp32_grad_name = param + core.grad_var_suffix() + '@MERGED'
                fp32_grad = main_block.create_var(
                    name=fp32_grad_name,
                    dtype=paddle.float32,
                    shape=real_grad.shape,
                    persistable=False,
                    stop_gradient=False,
                )
                main_block._insert_op(
                    index=first_opt_op_idx + offset,
                    type='cast',
                    inputs={'X': fp16_grad},
                    outputs={'Out': fp32_grad},
                    attrs={
                        'in_dtype': paddle.float16,
                        'out_dtype': paddle.float32,
                        self._op_role_key: self._op_role.Optimize,
                    },
                )
                offset += 1

        # replace the var with it's name, which will be used for inserting allreduce
        for i in range(len(fused_merged_gradients)):
            fused_merged_gradients[i] = fused_merged_gradients[i].name

        return fused_merged_gradients, first_opt_op_idx

    def _accumulate_gradients_with_fuse(
        self, main_block, fp16, fused_size, shard=None
    ):
        first_opt_op_idx = None
        grad_param_pairs = []
        # obtain all param/grad pairs that needed to be fused
        for index, op in reversed(tuple(enumerate(list(main_block.ops)))):
            # remove the cast op of fp16 grad to fp32 grad
            if self._is_optimize_op(op) and op.type == 'cast':
                in_name = op.input_arg_names[0]
                out_name = op.output_arg_names[0]
                if out_name.strip('@GRAD') in self._param_device_map:
                    assert in_name.replace('.cast_fp16', '') == out_name
                    main_block._remove_op(index)
                    continue

            if self._is_backward_op(op) and first_opt_op_idx is None:
                first_opt_op_idx = index + 1
                # no optimize phase
                if first_opt_op_idx == len(main_block.ops):
                    return

            if self._is_backward_op(op) and (
                self._op_role_var_key in op.attr_names
            ):
                op_role_var = op.attr(self._op_role_var_key)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    if not main_block.has_var(param_name):
                        continue
                    if '@BroadCast' in param_name:
                        continue
                    grad_param_pairs.append(
                        (op_role_var[i + 1], op_role_var[i])
                    )

        if len(grad_param_pairs) == 0:
            return

        nranks = shard.worker_num if shard else 1
        device_to_pairs = [[] for _ in range(nranks)]
        for pair in grad_param_pairs:
            root_id = shard.device(pair[1]) if shard else 0
            assert 0 <= root_id < nranks
            device_to_pairs[root_id].append(pair)

        all_fused_merged_gradients = []
        for pairs in device_to_pairs:
            (
                fused_merged_gradients,
                first_opt_op_idx,
            ) = self._insert_accumulate_gradients_with_fuse(
                main_block, fp16, fused_size, pairs, first_opt_op_idx
            )
            all_fused_merged_gradients += fused_merged_gradients

        main_block._sync_with_cpp()
        return all_fused_merged_gradients

    def _sort_grad_param_by_dtype(self, main_block, grad_param_pairs):
        # sort the grad param paris by the dtype
        fp16_pairs = []
        fp32_pairs = []
        other_pairs = []
        for pairs in grad_param_pairs:
            dtype = main_block.var(pairs[0]).dtype
            if dtype == paddle.float32:
                fp32_pairs.append(pairs)
            elif dtype == paddle.float16:
                fp16_pairs.append(pairs)
            else:
                other_pairs.append(pairs)
        sorted_pairs = fp16_pairs
        sorted_pairs.extend(fp32_pairs)
        sorted_pairs.extend(other_pairs)
        return sorted_pairs

    def _get_var_size(self, var):
        dtype_to_size = {
            core.VarDesc.VarType.FP16: 2,
            core.VarDesc.VarType.BF16: 2,
            core.VarDesc.VarType.FP32: 4,
            core.VarDesc.VarType.FP64: 8,
            core.VarDesc.VarType.INT16: 2,
            core.VarDesc.VarType.INT32: 4,
            core.VarDesc.VarType.INT64: 8,
            core.VarDesc.VarType.BOOL: 1,
            core.VarDesc.VarType.UINT8: 1,
        }
        assert -1 not in var.shape
        return (
            reduce(lambda x, y: x * y, var.shape, 1)
            * dtype_to_size[var.dtype]
            / 1024.0
            / 1024.0
        )

    def _add_sub_blocks(self, main_block, program_list):
        main_program = main_block.program
        for prog in program_list:
            for op in prog.block(0).ops:
                if not op.has_attr('sub_block'):
                    continue
                origin_sub_block_id = op.attr('sub_block').id
                origin_sub_block = main_program.block(origin_sub_block_id)
                new_sub_block = prog._create_block(parent_idx=0)
                for sub_op in origin_sub_block.ops:
                    op_desc = sub_op.desc
                    ap_op = new_sub_block.desc.append_op()
                    ap_op.copy_from(op_desc)
                new_sub_block._sync_with_cpp()
                self._create_vars(new_sub_block, origin_sub_block)
                op._set_attr('sub_block', new_sub_block)

    def _get_device_info(self, block):
        for op in block.ops:
            if not op._has_kernel(op.type):
                continue
            op_device = op.attr(self._op_device_key)
            return op_device

    def _process_persistable_vars_in_multi_sections(
        self, main_program, startup_prog, program_list
    ):
        """
        Special Case: process persistable vars that exist in
        multiple sections, e.g., shared weight
        """
        # var_info = {var_name: [program1, program2...]},
        # persistable var only
        var_info = {}
        for prog in program_list:
            block = prog.block(0)
            for var_name in block.vars:
                if var_name == "double_buffer_0":
                    continue
                var = block.var(var_name)
                if not var.persistable:
                    continue
                if var_name not in var_info:
                    var_info[var_name] = []
                if prog not in var_info[var_name]:
                    var_info[var_name].append(prog)
        for var_name in list(var_info.keys()):
            if len(var_info[var_name]) == 1:
                var_info.pop(var_name)

        # write_info = {var_name: program}, where program is the only program
        # in which the var named var_name is written.
        write_info = {}
        for var_name in var_info.keys():
            for prog in var_info[var_name]:
                block = prog.block(0)
                for op in block.ops:
                    if (
                        op.type == "recv_v2"
                        or op.type == "create_py_reader"
                        or op.type == "read"
                        or op.type == "update_loss_scaling"
                    ):
                        continue
                    # We have processed lr related vars
                    if op.attr(self._op_role_key) == int(
                        self._op_role.Optimize.LRSched
                    ):
                        continue
                    if var_name in op.desc.output_arg_names():
                        assert var_name not in write_info, (
                            f"two sections write the same var({var_name}): second "
                            f"op {op}."
                        )
                        write_info[var_name] = prog
                        break

        for var_name in var_info.keys():
            # Case 1: read only variables, no special process
            if var_name not in write_info:
                continue

            # Case 2: one write multiple reads
            write_prog = write_info[var_name]
            write_block = write_prog.block(0)
            write_device = self._get_device_info(write_block)
            write_dev_index = int(write_device.split(':')[1])
            all_progs = var_info[var_name]
            for prog in all_progs:
                if prog == write_prog:
                    continue
                read_block = prog.block(0)
                read_device = self._get_device_info(read_block)
                read_dev_index = int(read_device.split(':')[1])
                pair = (write_dev_index, read_dev_index)
                pair_key = write_dev_index * 1000 + read_dev_index
                if pair not in self._pipeline_pair:
                    self._pipeline_pair.append(pair)
                    self._pp_ring_map[pair_key] = self.ring_id
                    ring_id = self.ring_id
                    self.ring_id += 1
                else:
                    ring_id = self._pp_ring_map[pair_key]

                write_block._insert_op(
                    index=0,
                    type='send_v2',
                    inputs={
                        'X': write_block.var(var_name),
                    },
                    attrs={
                        self._op_device_key: write_device,
                        'use_calc_stream': False,
                        # A trick to make the role LRSched to avoid copy every
                        # microbatch
                        self._op_role_key: self._op_role.LRSched,
                        'peer': read_dev_index,
                        'ring_id': ring_id,
                    },
                )
                read_block._insert_op(
                    index=0,
                    type='recv_v2',
                    outputs={'Out': [read_block.var(var_name)]},
                    attrs={
                        'out_shape': read_block.var(var_name).shape,
                        'dtype': read_block.var(var_name).dtype,
                        self._op_device_key: read_device,
                        'use_calc_stream': False,
                        # A trick to make the role LRSched to avoid copy every
                        # microbatch
                        self._op_role_key: self._op_role.LRSched,
                        'peer': write_dev_index,
                        'ring_id': ring_id,
                    },
                )
                read_block._insert_op(
                    index=1,
                    type='c_sync_comm_stream',
                    inputs={'X': [read_block.var(var_name)]},
                    outputs={'Out': [read_block.var(var_name)]},
                    attrs={
                        self._op_device_key: read_device,
                        # A trick to make the role LRSched to avoid copy every
                        # microbatch
                        self._op_role_key: self._op_role.LRSched,
                        'ring_id': ring_id,
                    },
                )

    def _is_gradient_clip_op(self, op):
        return op.desc.has_attr("op_namescope") and op.desc.attr(
            "op_namescope"
        ).startswith("/gradient_clip")

    def _is_regularization_op(self, op):
        return op.desc.has_attr("op_namescope") and op.desc.attr(
            "op_namescope"
        ).startswith("/regularization")

    def _is_weight_decay_op(self, op):
        # in AdamW namescope is /optimizer_*/weight decay/
        return op.desc.has_attr(
            "op_namescope"
        ) and 'weight decay' in op.desc.attr("op_namescope")

    def _get_input_output_info(self, block):
        '''
        Get info of op input and output.
        '''
        # A map from output var to op which generate it.
        output_var_to_op = defaultdict(list)
        # A map from var to op which takes it as input.
        input_var_to_op = defaultdict(list)

        for index, op in enumerate(block.ops):
            for var_name in op.input_arg_names:
                input_var_to_op[var_name].append([op, index])
            for var_name in op.output_arg_names:
                output_var_to_op[var_name].append([op, index])

        return output_var_to_op, input_var_to_op

    def _optimize_forward_send_sync(self, program):
        """
        optimize forward send's sync_comm_stream schedule
        """
        if self.schedule_mode != '1F1B':
            return

        block = program.block(0)

        recv_type = 'recv_v2' if self.mp_degree == 1 else 'partial_recv'
        backward_recv_index = None
        for index, op in enumerate(block.ops):
            if op.type == recv_type and self._is_backward_op(op):
                backward_recv_index = index
                break

        # last pipeline stage
        if backward_recv_index is None:
            return

        offset = 0
        for index, op in enumerate(list(block.ops)):
            if index >= backward_recv_index:
                break
            if op.type == 'c_sync_comm_stream' and op.has_attr('pipeline_flag'):
                var_name = op.input_arg_names[0]
                var = block.var(var_name)
                block._remove_op(index + offset, sync=False)
                offset -= 1
                # NOTE:
                # 1. When the backward recv is completed, it indicates
                # that the forward send is completed too. So we only need
                # to use the NOP op to prevent memory release.
                # 2. Because we removed sync_comm_op,
                # we will insert NOP after recv_op.
                block._insert_op_without_sync(
                    index=backward_recv_index,
                    type='nop',
                    inputs={'X': [var]},
                    outputs={'Out': [var]},
                    attrs={self._op_role_key: self._op_role.Backward},
                )
        block._sync_with_cpp()

    def _mv_head_recv(self, program):
        """
        A pass to move the recv op to the beginning of
        the forward/backward phase
        """
        forward_insert_index = 0
        backward_insert_index = None
        block = program.global_block()
        num_ops = len(program.global_block().ops)
        for i in range(num_ops):
            insert_index = None
            op = program.global_block().ops[i]
            op_role = int(op.attr(self._op_role_key))
            if (
                op_role == int(self._op_role.Backward)
                and backward_insert_index is None
            ):
                backward_insert_index = i
            if (
                op.type != "partial_recv"
                and op.type != "partial_allgather"
                and op.type != "nop"
                and op.type != "recv_v2"
            ):
                continue
            if op_role == int(self._op_role.Forward):
                if i == forward_insert_index:
                    forward_insert_index += 1
                    continue
                insert_index = forward_insert_index
            elif op_role == int(self._op_role.Backward):
                if i == backward_insert_index:
                    backward_insert_index += 1
                    continue
                insert_index = backward_insert_index
            else:
                raise ValueError(f"Unknown op_role: {op_role}")
            op_inputs = {}
            for name in op.input_names:
                op_inputs[name] = op.input(name)
            op_outputs = {}
            for name in op.output_names:
                op_outputs[name] = op.output(name)
            block._insert_op_without_sync(
                index=insert_index,
                type=op.type,
                inputs=op_inputs,
                outputs=op_outputs,
                attrs=op.all_attrs(),
            )
            block._remove_op(i + 1)
            if op_role == int(self._op_role.Forward):
                forward_insert_index += 1
            elif op_role == int(self._op_role.Backward):
                backward_insert_index += 1
        block._sync_with_cpp()

    def _check_pipeline_persist_var(self, program):
        """
        Pipeline may need multiple forward before
        """
        block = program.global_block()

        persist_output = set()
        used_in_backward = set()
        for op in block.ops:
            if self._is_forward_op(op):
                for var_name in op.output_arg_names:
                    var = block.vars[var_name]
                    if var.persistable:
                        persist_output.add(var_name)
            elif self._is_backward_op(op):
                for var_name in op.input_arg_names:
                    if var_name in persist_output:
                        used_in_backward.add(var_name)
        if len(used_in_backward) == 0:
            return
        warnings.warn(
            "The pipeline requires multiple forward calculations before backward, "
            "so when the persistable var is changed in the forward, it may cause "
            "errors in the backward calculation who using this persistable var. "
            "However, some backward op don't need this var(NoNeedBufferVars), "
            "there will be no error at this time.\n"
            "So please check these persistable vars which changed in "
            f"forward and used in backward:\n{used_in_backward}"
        )

    def minimize(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        main_block = loss.block
        self.origin_main_block = main_block
        main_program = main_block.program
        if startup_program is None:
            startup_program = default_startup_program()

        pipeline_opt = main_program._pipeline_opt
        assert pipeline_opt, 'Please use pipeline with fleet.'
        required_keys = [
            'local_rank',
            'schedule_mode',
            'micro_batch_size',
            'ring_id',
            'global_ring_id',
            'use_sharding',
            'mp_degree',
            'mp_rank',
        ]
        for key in required_keys:
            assert (
                key in pipeline_opt
            ), f'Please use pipeline with fleet to use {key}.'
        self.local_rank = pipeline_opt['local_rank']
        self.schedule_mode = pipeline_opt['schedule_mode']
        self.micro_batch_size = pipeline_opt['micro_batch_size']
        self.use_sharding = pipeline_opt['use_sharding']
        self.ring_id = pipeline_opt['ring_id']
        self.global_ring_id = pipeline_opt['global_ring_id']
        self.mp_degree = pipeline_opt['mp_degree']
        self.mp_rank = pipeline_opt['mp_rank']
        self.scale_gradient = pipeline_opt.get('scale_gradient', False)
        assert self.mp_degree >= 1
        assert 0 <= self.mp_rank < self.mp_degree

        optimize_ops, params_grads = self._optimizer.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )
        self._param_device_map = self._origin_optimizer._param_device_map

        (
            self.output_var_to_op,
            self.input_var_to_op,
        ) = self._get_input_output_info(main_block)
        # Step1: add default op_device attribute for ops.
        self._add_op_device_attr(main_block)
        device_list = self._check_validation(main_block)

        def device_cmp(device1, device2):
            dev1_id = int(device1.split(':')[1])
            dev2_id = int(device2.split(':')[1])
            if dev1_id < dev2_id:
                return -1
            elif dev1_id > dev2_id:
                return 1
            else:
                return 0

        sorted_device_list = sorted(device_list, key=cmp_to_key(device_cmp))
        assert sorted_device_list == device_list, (
            "With pipeline parallelism, you must use gpu devices one after "
            "another in the order of their ids."
        )
        # Step2: add send and recv ops between section boundaries
        self._insert_sendrecv_ops_for_boundaries(main_block)

        # Step3: split program into sections and add pairs of
        # send and recv ops for data var.
        main_program = main_block.program
        program_list = self._split_program(main_program, device_list)
        for p in program_list:
            self._create_vars(p.global_block(), main_block)

        if os.getenv("PADDLE_MANUAL_PIPELINE_STAGE", None):
            self.local_rank = int(os.getenv("PADDLE_MANUAL_PIPELINE_STAGE"))
            assert self.local_rank < len(device_list), (
                "Manually specified "
                "pipeline stage must be less than total number of pipeline "
                "stages."
            )
        else:
            self.local_rank %= len(device_list)
        # Step3.5: optimize forward send sync_comm to overlap send and recv
        self._optimize_forward_send_sync(program_list[self.local_rank])

        # Step4: Special Case: process persistable vars that exist in
        # multiple sections
        # FIXME
        # self._process_persistable_vars_in_multi_sections(
        #     main_program, startup_program, program_list)

        # Step5: Add sub blocks for section programs
        self._add_sub_blocks(main_block, program_list)

        place_list = []
        for dev in device_list:
            dev_index = int(dev.split(":")[1])
            if core.is_compiled_with_cuda():
                place_list.append(core.CUDAPlace(dev_index % 1))

        # Step6: Split startup program
        new_startup_program = self._split_startup_program(
            startup_program, self.local_rank
        )

        startup_program._pipeline_opt = {
            "startup_program": new_startup_program,
        }
        real_block = program_list[self.local_rank].global_block()
        if not self.scale_gradient:
            self._insert_loss_scale(real_block)
        if not self.use_sharding:
            # Step7: clear gradients before each mini-batch and
            # accumulate gradients during backward
            self._rename_gradient_var_name(real_block)
            real_block._sync_with_cpp()
            self._accumulate_gradients(real_block)
            real_block._sync_with_cpp()

        if core.is_compiled_with_cuda():
            place_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        # A pass to move the recv op to the beginning of
        # the forward/backward phase
        self._mv_head_recv(program_list[self.local_rank])

        # A pass to check pipeline persist var which changed in
        # forward and used in backward
        self._check_pipeline_persist_var(program_list[self.local_rank])

        main_program._pipeline_opt = {
            "trainer": "PipelineTrainer",
            "device_worker": "Section",
            "pipeline_stage": self.local_rank,
            "num_pipeline_stages": len(device_list),
            "schedule_mode": self.schedule_mode,
            "inner_parallelism": len(device_list),
            "section_program": program_list[self.local_rank],
            "place": place_list[self.local_rank],
            "place_id": place_id,
            "sync_steps": -1,
            "num_microbatches": self._num_microbatches,
            "start_cpu_core_id": self._start_cpu_core_id,
        }
        return (
            optimize_ops,
            params_grads,
            program_list,
            self._pipeline_pair,
            self._pp_ring_map,
        )
