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
from __future__ import annotations

import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from typing import cast

import numpy as np

sys.path.append("../../legacy_test")
from prim_op_test import OpTestUtils, _as_list, convert_uint16_to_float, flatten
from utils import dygraph_guard

import paddle
import paddle.distributed as dist

IMPORT_PACKAGE_TEMPLATE = """

import pathlib
import pickle
import sys
"""

IMPORT_FORWARD_TEST_CLASS_TEMPLATE = """

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[0] / 'test/legacy_test')
)
from auto_parallel_op_test import AutoParallelForwardChecker, convert_input_dims_map_to_placements
"""

IMPORT_GRAD_TEST_CLASS_TEMPLATE = """

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[0] / 'test/legacy_test')
)
from auto_parallel_op_test import AutoParallelGradChecker, convert_input_dims_map_to_placements
"""

LOAD_TEST_INFO_TEMPLATE = """

def load_test_info(test_info_path):
    with open(test_info_path, "rb") as f:
        test_info = pickle.load(f)
    return test_info
"""

FORWARD_TEST_FUNCTION_TEMPLATE = """

def run_forward_check(test_info):
    auto_parallel_forward_checker = AutoParallelForwardChecker(
        test_info["op_type"],
        python_api,
        test_info["dtype"],
        convert_input_dims_map_to_placements(test_info["dims_map"], test_info["inputs"], 1),
        test_info["inputs"],
        test_info["attrs"],
        test_info["outputs"],
        test_info["place"],
        test_info["eager_auto_parallel_threshold"],
        test_info["python_out_sig"],
    )
    auto_parallel_forward_checker.check()
"""

GRAD_TEST_FUNCTION_TEMPLATE = """

def run_grad_check(test_info):
    auto_parallel_forward_checker = AutoParallelGradChecker(
        test_info["op_type"],
        python_api,
        test_info["dtype"],
        convert_input_dims_map_to_placements(test_info["dims_map"], test_info["inputs"], 1),
        test_info["inputs"],
        test_info["attrs"],
        test_info["outputs"],
        test_info["place"],
        test_info["inputs_to_check"],
        test_info["output_names"],
        test_info["no_grad_set"],
        test_info["user_defined_grad_outputs"],
        test_info["eager_auto_parallel_threshold"],
        test_info["python_out_sig"],
    )
    auto_parallel_forward_checker.check()
"""

LOAD_PYTHON_API_TEMPLATE = """
    from {module} import {function}
    python_api = {function}
"""

TEST_BODY_TEMPLATE = """

if __name__ == "__main__":
    test_info = load_test_info(r'{test_info_path}')
    {load_python_api}
    {run_test}
"""


def is_ban_auto_parallel_test(place):
    if (
        isinstance(place, paddle.base.libpaddle.CUDAPlace)
        and paddle.device.cuda.device_count() < 2
        or not paddle.is_compiled_with_distribute()
        or (
            os.environ.get("WITH_COVERAGE") == "ON"
            and os.environ.get("FLAGS_COVERAGE_RUN_AUTO_PARALLEL_IN_OP_TEST")
            != "1"
        )
    ):
        return True
    else:
        return False


def gen_import_packages(check_grad):
    import_code = ''
    import_code += IMPORT_PACKAGE_TEMPLATE
    import_code += (
        IMPORT_FORWARD_TEST_CLASS_TEMPLATE
        if not check_grad
        else IMPORT_GRAD_TEST_CLASS_TEMPLATE
    )
    return import_code


def gen_auto_parallel_test_file(
    check_grad, test_info_path, test_file_path, python_api_info
):
    test_code = ''
    test_code += gen_import_packages(check_grad)
    test_code += LOAD_TEST_INFO_TEMPLATE.format(test_info_path=test_info_path)
    test_code += (
        GRAD_TEST_FUNCTION_TEMPLATE
        if check_grad
        else FORWARD_TEST_FUNCTION_TEMPLATE
    )
    run_test_str = (
        "run_grad_check(test_info)"
        if check_grad
        else "run_forward_check(test_info)"
    )
    load_python_api_str = LOAD_PYTHON_API_TEMPLATE.format(
        module=python_api_info["api_module"],
        function=python_api_info["api_name"],
    )
    test_code += TEST_BODY_TEMPLATE.format(
        test_info_path=test_info_path,
        load_python_api=load_python_api_str,
        run_test=run_test_str,
    )
    with open(test_file_path, "w") as f:
        f.write(test_code)


def get_test_info_and_generated_test_path(
    test_class_name, op_type, backward=False
):
    suffixes = str(uuid.uuid4())
    current_path = pathlib.Path(__file__).resolve().parents[0]
    forward_or_backward = "forward" if not backward else "backward"
    test_info_path = (
        current_path
        / f"{test_class_name}_{op_type}_{forward_or_backward}_info_{suffixes}.pkl"
    )
    generated_test_path = (
        current_path
        / f"{test_class_name}_{op_type}_{forward_or_backward}_test_{suffixes}.py"
    )

    return str(test_info_path), str(generated_test_path)


def check_auto_parallel_info(op_test):
    assert hasattr(
        op_test, 'python_api'
    ), "If you want to check auto parallel, please set python_api in setUp function."
    assert hasattr(
        op_test, 'placements'
    ), "If you want to check auto parallel, please set placements in setUp function."


def dump_test_info(
    op_test,
    place,
    test_info_path,
    backward=False,
    backward_extra_test_info=None,
):
    check_auto_parallel_info(op_test)
    test_info = {}
    with open(test_info_path, "wb") as f:
        test_info["op_type"] = op_test.op_type
        test_info["dtype"] = op_test.dtype
        test_info["dims_map"] = convert_input_placements_to_dims_map(
            op_test.placements, op_test.inputs
        )
        test_info["inputs"] = op_test.inputs
        test_info["attrs"] = op_test.attrs if hasattr(op_test, "attrs") else {}
        test_info["outputs"] = op_test.outputs
        if isinstance(place, paddle.base.libpaddle.CPUPlace):
            test_info["place"] = "cpu"
        if isinstance(place, paddle.base.libpaddle.CUDAPlace):
            test_info["place"] = "gpu"
        eager_auto_parallel_threshold = {
            "atol": (
                op_test.eager_auto_parallel_atol
                if hasattr(op_test, "eager_auto_parallel_atol")
                else None
            ),
            "rtol": (
                op_test.eager_auto_parallel_atol
                if hasattr(op_test, "eager_auto_parallel_atol")
                else None
            ),
        }
        test_info["eager_auto_parallel_threshold"] = (
            eager_auto_parallel_threshold
        )
        test_info["python_out_sig"] = (
            op_test.python_out_sig
            if hasattr(op_test, "python_out_sig")
            else None
        )
        if backward:
            test_info["inputs_to_check"] = backward_extra_test_info[
                "inputs_to_check"
            ]
            test_info["output_names"] = backward_extra_test_info["output_names"]
            test_info["no_grad_set"] = backward_extra_test_info["no_grad_set"]
            test_info["user_defined_grad_outputs"] = backward_extra_test_info[
                "user_defined_grad_outputs"
            ]
        try:
            pickle.dump(test_info, f)
        except Exception as e:
            raise Exception(
                "Dump test info failed, please check your test info."
            )


def get_subprocess_runtime_envs(place):
    runtime_envs = os.environ
    if (
        "CUDA_VISIBLE_DEVICES" not in runtime_envs
        or len(runtime_envs["CUDA_VISIBLE_DEVICES"].split(",")) < 2
    ):
        runtime_envs.update({"CUDA_VISIBLE_DEVICES": "0,1"})
        if isinstance(place, paddle.base.libpaddle.CPUPlace):
            runtime_envs.update({"backend": "cpu"})
        if isinstance(place, paddle.base.libpaddle.CUDAPlace):
            runtime_envs.update({"backend": "gpu"})
    return runtime_envs


def get_subprocess_command(devices, test_file_path, log_dir=None):
    if log_dir:
        if os.path.isabs(log_dir):
            abs_log_dir = log_dir
        else:
            abs_log_dir = os.path.abspath(log_dir)
    else:
        abs_log_dir = tempfile.TemporaryDirectory().name
    start_command = f"{sys.executable} -m paddle.distributed.launch --devices {devices} --log_dir {abs_log_dir}  {test_file_path}"
    return start_command


def run_subprocess(start_command, env, timeout):
    start_command_list = start_command.strip().split()
    try:
        _launcher = subprocess.run(
            start_command_list,
            env=env,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as err:
        raise TimeoutError(
            f"Timeout while running command {err.cmd}, try to set a longer period, {err.timeout} is not enough."
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error occurs when running this test case. The return code of command {err.cmd} is {err.returncode}"
        )


def convert_input_placements_to_dims_map(placements: dict, inputs: dict):
    all_dims_map = {}
    for name, item in inputs.items():
        if name not in placements:
            continue
        # such as inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        # placements = {"X": [("x0", [Shard(0)]), ("x1", [Shard(0)]), ("x2", [Shard(0)])]}
        if isinstance(item, list):
            all_dims_map[name] = []
            for i in range(len(item)):
                dims_map = placements_to_dims_map(
                    placements[name][i][1], inputs[name][i][1].ndim
                )
                all_dims_map[name].append((item[i][0], dims_map))
        # inputs like this : inputs = {'X': x}
        # placements = {"X": [Shard(0)]}
        else:
            dims_map = placements_to_dims_map(
                placements[name], inputs[name].ndim
            )
            all_dims_map[name] = dims_map
    return all_dims_map


def convert_input_dims_map_to_placements(
    dims_map: dict, inputs: dict, mesh_ndim: int
):
    placements_map = {}
    for name, item in inputs.items():
        if name not in dims_map:
            continue
        # such as inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        # dims_map = {"X": [("x0", [-1, 0]), ("x1", [-1, 0]), ("x2", [-1, 0]}
        if isinstance(item, list):
            placements_map[name] = []
            for i in range(len(item)):
                placements = dims_map_to_placements(
                    dims_map[name][i][1], mesh_ndim
                )
                placements_map[name].append((item[i][0], placements))
        # inputs like this : inputs = {'X': x}
        # placements = {"X": [Shard(0)]}
        else:
            placements = dims_map_to_placements(dims_map[name], mesh_ndim)
            placements_map[name] = placements
    return placements_map


# TODO: This method has been implemented in
# paddle/phi/core/distributed/auto_parallel/placement_types.h, bind it
# python and it's logic.
def placements_to_dims_map(placements: list, tensor_ndim: int) -> tuple[int]:
    r = [-1] * tensor_ndim
    for i, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast(dist.Shard, placement).get_dim()
            if r[shard_dim] > -1:
                raise ValueError(
                    f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                    " DTensor operator implementation does not support things like hybrid"
                    " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                )
            r[shard_dim] = i
    return r


# TODO: Add this method to
# paddle/phi/core/distributed/auto_parallel/placement_types.h, and bind it to
# python
def dims_map_to_placements(
    dim_map: tuple[int], mesh_ndim: int, sums: tuple[int] = ()
) -> tuple[dist.Placement]:
    """
    Construct a placements from dim_map list and pending sum.

    Args:
        dim_map (tuple[int]): a list of integer that represents sharding on each
            tensor dimension, see `dim_map` property doc for details
        mesh_ndim (int): the ndim of Process mesh.
        sums (tuple[int]): a list of integer that represents the dist tensor have
            pending sum on which device mesh dimension.

    Return:
        a placement sequence.
    """
    # by default replicate on device mesh dims
    placements: list[dist.Placement] = [
        dist.Replicate() for _ in range(mesh_ndim)
    ]

    # find all mesh dims that need pending reductions
    for s in sums:
        placements[s] = dist.Partial()

    for i, m in enumerate(dim_map):
        if m >= 0:
            placement = placements[m]
            if placement.is_shard():
                placement = cast(dist.Shard, placement)
                raise RuntimeError(
                    f"DeviceMesh dimension can't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                )
            elif placement.is_partial():
                raise RuntimeError(
                    f"DeviceMesh dimension {m} cannot be both shard and partial!"
                )
            placements[m] = dist.Shard(i)

    return tuple(placements)


TOLERANCE = {
    np.dtype('float64'): {"rtol": 1e-15, "atol": 0},
    np.dtype('float32'): {"rtol": 1e-6, "atol": 0},
    np.dtype('float16'): {"rtol": 1e-3, "atol": 0},
    np.dtype('uint16'): {"rtol": 1e-2, "atol": 0},
    np.dtype('int32'): {"rtol": 0, "atol": 0},
}


class AutoParallelForwardChecker:
    def __init__(
        self,
        op_type,
        python_api,
        dtype,
        placements_map,
        inputs,
        attrs,
        outputs,
        place,
        eager_auto_parallel_threshold,
        python_out_sig=None,
    ):
        self.checker_name = "AutoParallelForwardChecker"
        self.init_checker(
            op_type,
            python_api,
            dtype,
            placements_map,
            inputs,
            attrs,
            outputs,
            place,
            eager_auto_parallel_threshold,
            python_out_sig,
        )

    def init_checker(
        self,
        op_type,
        python_api,
        dtype,
        placements_map,
        inputs,
        attrs,
        outputs,
        place,
        eager_auto_parallel_threshold,
        python_out_sig=None,
    ):
        self.op_type = op_type
        self.public_python_api = python_api
        self.dtype = np.dtype(dtype)
        self.placements_map = placements_map
        self.inputs = inputs
        self.attrs = attrs
        self.outputs = outputs
        self.place = place
        if self.place == "cpu":
            paddle.device.set_device("cpu")
        if self.place == "gpu":
            paddle.device.set_device("gpu:" + str(dist.get_rank()))
        self.python_out_sig = python_out_sig
        self.attrs = attrs
        self.outputs = outputs
        self.init_checker_threshold(
            eager_auto_parallel_threshold["atol"],
            eager_auto_parallel_threshold["rtol"],
        )
        self.kernel_sig = self.get_kernel_sig()
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def init_checker_threshold(self, atol=None, rtol=None):
        self.atol = atol if atol else TOLERANCE[self.dtype]["atol"]
        self.rtol = rtol if rtol else TOLERANCE[self.dtype]["rtol"]

    def check(self):
        self.eager_forward_desire = self.get_eager_desire()
        self.check_eager_auto_parallel()

    def check_eager_auto_parallel(self):
        with dygraph_guard():
            actual_ret = self.get_eager_desire(dist_mode=True)
            # check eager auto parallel forward
            if len(actual_ret) != len(self.eager_forward_desire):
                msg = (
                    f"The eager auto parallel out tensor nums is different with eager out tensor nums on {self.place}."
                    f'eager auto parallel out tensor nums = {len(actual_ret)}, eager out tensor nums = {len(self.eager_forward_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(actual_ret)):
                np.testing.assert_allclose(
                    actual_ret[i],
                    self.eager_forward_desire[i],
                    rtol=self.atol,
                    atol=self.rtol,
                    err_msg=(
                        f"Check eager auto parallel failed. Mismatch between eager auto parallel outputs "
                        f"and eager outputs on {self.place!s}. The eager forward output tensor's index is : {i} \n"
                        f"eager auto parallel output tensor:\n{actual_ret[i]}\n eager output tensor:\n{self.eager_forward_desire[i]}\n"
                    ),
                )

    def get_kernel_sig(self):
        with dygraph_guard():
            (
                eager_tensor_inputs,
                attrs_outputs,
                _,
            ) = self.get_eager_input_attr_and_inputdict(stop_gradient=True)
            eager_tensor_outputs = self.get_eager_empty_output(
                stop_gradient=True
            )
            kernel_sig = OpTestUtils._get_kernel_signature(
                self.op_type,
                eager_tensor_inputs,
                eager_tensor_outputs,
                attrs_outputs,
            )
        return kernel_sig

    def get_eager_desire(self, dist_mode=False):
        with dygraph_guard():
            if dist_mode:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    _,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=True, dist_mode=True
                )
            else:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    _,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=True, dist_mode=False
                )
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=paddle.core.VarDesc.VarType,
            )
            inputs_sig, _, _ = self.kernel_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )
            ret = flatten(_as_list(self.public_python_api(*args)))
            ret = paddle.utils.map_structure(lambda x: x.numpy(), ret)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                ret = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), ret
                )
        return ret

    def get_eager_input_attr_and_inputdict(
        self, stop_gradient, dist_mode=False
    ):
        attrs_outputs = {}
        for attrs_name in self.attrs:
            if self.attrs[attrs_name] is not None:
                attrs_outputs[attrs_name] = self.attrs[attrs_name]
        input_dict = {}
        eager_inputs = defaultdict(list)
        for name, item in self.inputs.items():
            # such as inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
            #  placements = {"X": [("x0", [Shard(0)]), ("x1", [Shard(0)]), ("x2", [Shard(0)])]}
            if isinstance(item, list):
                for i in range(len(item)):
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(item[i][1].dtype)
                        else item[i][1].dtype
                    )
                    x = paddle.to_tensor(
                        data=item[i][1],
                        stop_gradient=stop_gradient,
                        dtype=dtype,
                    )
                    if not dist_mode or name not in self.placements_map:
                        eager_inputs[name].append(x)
                        input_dict.update({str(item[i][0]): x})
                    else:
                        dist_x = dist.shard_tensor(
                            x, self._mesh, self.placements_map[name][i][1]
                        )
                        dist_x.stop_gradient = stop_gradient
                        eager_inputs[name].append(dist_x)
                        input_dict.update({str(item[i][0]): dist_x})
            # inputs like this : inputs = {'X': x}
            # placements = {"X": [Shard(0)]}
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=item,
                    stop_gradient=stop_gradient,
                    dtype=dtype,
                )
                if not dist_mode or name not in self.placements_map:
                    eager_inputs[name].append(x)
                    input_dict.update({name: x})
                else:
                    dist_x = dist.shard_tensor(
                        x, self._mesh, self.placements_map[name]
                    )
                    dist_x.stop_gradient = stop_gradient
                    eager_inputs[name].append(dist_x)
                    input_dict.update({name: dist_x})
        return eager_inputs, attrs_outputs, input_dict

    def get_eager_empty_output(self, stop_gradient):
        eager_outputs = defaultdict(list)
        for name, item in self.outputs.items():
            if isinstance(item, list):
                for tup in item:
                    dtype = (
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(tup[1].dtype)
                        else tup[1].dtype
                    )
                    x = paddle.to_tensor(
                        data=[],
                        stop_gradient=stop_gradient,
                        dtype=dtype,
                    )
                    eager_outputs[name].append(x)
            else:
                dtype = (
                    "bfloat16"
                    if OpTestUtils.is_bfloat16_type(item.dtype)
                    else item.dtype
                )
                x = paddle.to_tensor(
                    data=[],
                    stop_gradient=stop_gradient,
                    dtype=dtype,
                )
                eager_outputs[name].append(x)
        return eager_outputs


class AutoParallelGradChecker(AutoParallelForwardChecker):
    def __init__(
        self,
        op_type,
        python_api,
        dtype,
        placements_map,
        inputs,
        attrs,
        outputs,
        place,
        inputs_to_check,
        output_names,
        no_grad_set,
        grad_outputs,
        eager_auto_parallel_threshold,
        python_out_sig=None,
    ):
        super().__init__(
            op_type,
            python_api,
            dtype,
            placements_map,
            inputs,
            attrs,
            outputs,
            place,
            eager_auto_parallel_threshold,
            python_out_sig,
        )
        self.checker_name = "AutoParallelGradChecker"
        self.inputs_to_check = inputs_to_check
        self.output_names = output_names
        self.no_grad_set = no_grad_set
        self.grad_outputs = grad_outputs

    def check(self):
        (
            self.eager_forward_desire,
            self.eager_grad_desire,
        ) = self.get_eager_desire()
        self.check_eager_auto_parallel()

    def check_eager_auto_parallel(self):
        with dygraph_guard():
            actual_forward_res, actual_grad_res = self.get_eager_desire(
                dist_mode=True
            )
            # check eager auto parallel forward
            if len(actual_forward_res) != len(self.eager_forward_desire):
                msg = (
                    f"The eager auto parallel out tensor nums is different with eager out tensor nums on {self.place}."
                    f'eager auto parallel out tensor nums = {len(actual_forward_res)}, eager out tensor nums = {len(self.eager_forward_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(actual_forward_res)):
                np.testing.assert_allclose(
                    actual_forward_res[i],
                    self.eager_forward_desire[i],
                    rtol=self.atol,
                    atol=self.rtol,
                    err_msg=(
                        'Check eager auto parallel failed. Mismatch between eager auto parallel outputs '
                        f'and eager outputs on {self.place}, the eager forward output tensor\'s index is : {i} \n'
                        f'eager auto parallel output tensor:\n{actual_forward_res[i]}\n eager output tensor:\n{self.eager_forward_desire[i]}\n'
                    ),
                )

            # check eager auto parallel grad
            if len(actual_grad_res) != len(self.eager_grad_desire):
                msg = (
                    f"The eager auto parallel grad out tensor nums is different with eager grad out tensor nums on {self.place}."
                    f'eager auto parallel grad out tensor nums = {len(actual_grad_res)}, eager grad out tensor nums = {len(self.eager_grad_desire)}. \n'
                )
                raise RuntimeError(msg)
            for i in range(len(actual_grad_res)):
                np.testing.assert_allclose(
                    actual_grad_res[i],
                    self.eager_grad_desire[i],
                    rtol=self.atol,
                    atol=self.rtol,
                    err_msg=(
                        'Check eager auto parallel backward failed. Mismatch between eager auto parallel grad outputs '
                        f'and eager grad outputs on {self.place}, the eager grad output tensor\'s index is : {i} \n'
                        f'eager auto parallel grad output tensor:\n{actual_grad_res[i]}\n eager grad output tensor:\n{self.eager_grad_desire[i]}\n'
                    ),
                )

    def gen_eager_grad_outputs(self):
        if self.grad_outputs is None:
            return None
        eager_vs = []
        for np_v in self.grad_outputs:
            eager_vs.append(
                paddle.to_tensor(
                    data=np_v,
                    place=self.place,
                    dtype=(
                        "bfloat16"
                        if OpTestUtils.is_bfloat16_type(np_v.dtype)
                        else np_v.dtype
                    ),
                )
            )
        return eager_vs

    def get_output_dict(self, np_outputs, api_outputs, outputs_sig):
        assert len(api_outputs) <= len(
            outputs_sig
        ), f"forward api outputs length must be the less than or equal to KernelSignature outputs,but receive {len(api_outputs)} and {len(outputs_sig)}"
        output_dict = {}
        for i in range(len(api_outputs)):
            output_name = outputs_sig[i]
            if output_name in np_outputs and isinstance(
                np_outputs[output_name], list
            ):
                for j, tup in enumerate(np_outputs[output_name]):
                    output_dict.update({tup[0]: api_outputs[i][j]})
            else:
                output_dict.update({output_name: api_outputs[i]})
        return output_dict

    def gen_no_grad_set(self, var_dict):
        if self.no_grad_set is None:
            return None
        no_grad_set = set()
        for name in self.no_grad_set:
            if name in var_dict:
                no_grad_set.add(var_dict[name])
        return no_grad_set

    def get_eager_desire(self, dist_mode=False):
        with dygraph_guard():
            if dist_mode:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    inputs_dict,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=False, dist_mode=True
                )
            else:
                (
                    eager_tensor_inputs,
                    attrs_outputs,
                    inputs_dict,
                ) = self.get_eager_input_attr_and_inputdict(
                    stop_gradient=False, dist_mode=False
                )
            args = OpTestUtils.prepare_python_api_arguments(
                self.public_python_api,
                eager_tensor_inputs,
                attrs_outputs,
                self.kernel_sig,
                target_dtype=paddle.core.VarDesc.VarType,
            )
            inputs_sig, _, outputs_sig = self.kernel_sig
            if self.python_out_sig is not None:
                outputs_sig = self.python_out_sig
            args = OpTestUtils.assumption_assert_and_transform(
                args, len(inputs_sig)
            )

            forward_res = _as_list(self.public_python_api(*args))
            outputs_dict = self.get_output_dict(
                self.outputs, forward_res, outputs_sig
            )
            ys = []
            if isinstance(self.output_names, list):
                for output_name in self.output_names:
                    ys.append(outputs_dict[output_name])
            else:
                ys.append(outputs_dict[self.output_names])
            xs = []
            if isinstance(self.inputs_to_check, list):
                for input_name in self.inputs_to_check:
                    xs.append(inputs_dict[input_name])
            else:
                xs.append(inputs_dict[self.inputs_to_check])
            vs = self.gen_eager_grad_outputs()
            no_grad_vars = self.gen_no_grad_set(
                var_dict={**inputs_dict, **outputs_dict}
            )
            grad_res = paddle.grad(
                ys, xs, vs, allow_unused=True, no_grad_vars=no_grad_vars
            )
            forward_res = paddle.utils.map_structure(
                lambda x: x.numpy(), forward_res
            )
            grad_res = paddle.utils.map_structure(lambda x: x.numpy(), grad_res)
            if OpTestUtils.is_bfloat16_type(self.dtype):
                forward_res = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), forward_res
                )
                grad_res = paddle.utils.map_structure(
                    lambda x: convert_uint16_to_float(x), grad_res
                )

        return forward_res, grad_res
