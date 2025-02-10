#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import atexit
import os
import platform
import sys

# The legacy core need to be removed before "import core",
# in case of users installing paddlepaddle without -U option
core_suffix = 'so'
if os.name == 'nt':
    core_suffix = 'pyd'

legacy_core = (
    os.path.abspath(os.path.dirname(__file__)) + os.sep + 'core.' + core_suffix
)
if os.path.exists(legacy_core):
    sys.stderr.write('Deleting legacy file ' + legacy_core + '\n')
    try:
        os.remove(legacy_core)
    except Exception as e:
        raise e

# import all class inside framework into base module
# import all class inside executor into base module
from . import (  # noqa: F401
    backward,
    compiler,
    core,
    data_feed_desc,
    dataset,
    dygraph,
    executor,
    framework,
    incubate,
    initializer,
    io,
    layers,
    trainer_desc,
    unique_name,
)
from .backward import (  # noqa: F401
    append_backward,
    gradients,
)
from .compiler import (  # noqa: F401
    BuildStrategy,
    CompiledProgram,
    IpuCompiledProgram,
    IpuStrategy,
)
from .core import (  # noqa: F401
    CPUPlace,
    CUDAPinnedPlace,
    CUDAPlace,
    CustomPlace,
    DenseTensor,
    DenseTensorArray,
    IPUPlace,
    Scope,
    XPUPlace,
    _cuda_synchronize,
    _Scope,
    _set_warmup,
)
from .data_feed_desc import DataFeedDesc  # noqa: F401
from .data_feeder import DataFeeder  # noqa: F401
from .dataset import (  # noqa: F401
    DatasetFactory,
    InMemoryDataset,
)
from .dygraph.base import disable_dygraph, enable_dygraph
from .dygraph.tensor_patch_methods import monkey_patch_tensor
from .executor import (  # noqa: F401
    Executor,
    global_scope,
    scope_guard,
)
from .framework import (  # noqa: F401
    Program,
    Variable,
    cpu_places,
    cuda_pinned_places,
    cuda_places,
    default_main_program,
    default_startup_program,
    device_guard,
    get_flags,
    in_dygraph_mode,
    in_dynamic_or_pir_mode,
    in_pir_mode,
    ipu_shard_guard,
    is_compiled_with_cinn,
    is_compiled_with_cuda,
    is_compiled_with_rocm,
    is_compiled_with_xpu,
    name_scope,
    process_type_promotion,
    program_guard,
    require_version,
    set_flags,
    set_ipu_shard,
    xpu_places,
)
from .initializer import set_global_initializer  # noqa: F401
from .layers.math_op_patch import monkey_patch_variable
from .lod_tensor import (  # noqa: F401
    create_lod_tensor,
)
from .param_attr import ParamAttr, WeightNormParamAttr  # noqa: F401
from .trainer_desc import (  # noqa: F401
    DistMultiTrainer,
    HeterPipelineTrainer,
    HeterXpuTrainer,
    MultiTrainer,
    PipelineTrainer,
    TrainerDesc,
)

Tensor = DenseTensor
enable_imperative = enable_dygraph
disable_imperative = disable_dygraph

__all__ = []


def __bootstrap__():
    """
    Enable reading gflags from environment variables.

    Returns:
        None
    """
    in_test = 'unittest' in sys.modules

    try:
        num_threads = int(os.getenv('OMP_NUM_THREADS', '1'))
    except ValueError:
        num_threads = 1

    if num_threads > 1:
        print(
            f'WARNING: OMP_NUM_THREADS set to {num_threads}, not 1. The computation '
            'speed will not be optimized if you use data parallel. It will '
            'fail if this PaddlePaddle binary is compiled with OpenBlas since'
            ' OpenBlas does not support multi-threads.',
            file=sys.stderr,
        )
        print('PLEASE USE OMP_NUM_THREADS WISELY.', file=sys.stderr)

    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    flag_prefix = "FLAGS_"
    read_env_flags = [
        key[len(flag_prefix) :]
        for key in core.globals().keys()
        if key.startswith(flag_prefix)
    ]

    def remove_flag_if_exists(name):
        if name in read_env_flags:
            read_env_flags.remove(name)

    sysstr = platform.system()
    if 'Darwin' in sysstr:
        remove_flag_if_exists('use_pinned_memory')

    if core.is_compiled_with_ipu():
        # Currently we request all ipu available for training and testing
        #   finer control of pod of IPUs will be added later
        read_env_flags += []

    core.init_gflags(["--tryfromenv=" + ",".join(read_env_flags)])
    # Note(zhouwei25): sys may not have argv in some cases,
    # Such as: use Python/C API to call Python from C++
    try:
        core.init_glog(sys.argv[0])
    except Exception:
        sys.argv = [""]
        core.init_glog(sys.argv[0])
    # don't init_p2p when in unittest to save time.
    core.init_memory_method()
    core.init_devices()
    core.init_tensor_operants()
    core.init_default_kernel_signatures()


# TODO(panyx0718): Avoid doing complex initialization logic in __init__.py.
# Consider paddle.init(args) or paddle.main(args)
monkey_patch_variable()
__bootstrap__()
monkey_patch_tensor()

# NOTE(Aurelius84): clean up ExecutorCacheInfo in advance manually.
atexit.register(core.clear_executor_cache)
atexit.register(core.pir.clear_cinn_compilation_cache)

# NOTE(Aganlengzi): clean up KernelFactory in advance manually.
# NOTE(wangran16): clean up DeviceManager in advance manually.
# Keep clear_kernel_factory running before clear_device_manager
atexit.register(core.clear_device_manager)
atexit.register(core.clear_kernel_factory)
atexit.register(core.ProcessGroupIdMap.destroy)
