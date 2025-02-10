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

from __future__ import annotations

import ctypes
import errno
import fnmatch
import glob
import multiprocessing
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from subprocess import CalledProcessError

from setuptools import Command, Extension, setup
from setuptools.command.develop import develop as DevelopCommandBase
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install as InstallCommandBase
from setuptools.command.install_lib import install_lib
from setuptools.dist import Distribution

python_version = platform.python_version()
version_detail = sys.version_info
version = str(version_detail[0]) + '.' + str(version_detail[1])
env_version = os.getenv("PY_VERSION", None)

if version_detail < (3, 8):
    raise RuntimeError(
        f"Paddle only supports Python version >= 3.8 now,"
        f"you are using Python {python_version}"
    )
elif env_version is None:
    print(f"export PY_VERSION = { version }")
    os.environ["PY_VERSION"] = python_version

elif env_version != version:
    raise ValueError(
        f"You have set the PY_VERSION environment variable to {env_version}, but "
        f"your current Python version is {version}, "
        f"Please keep them consistent."
    )


# check cmake
CMAKE = shutil.which('cmake3') or shutil.which('cmake')
assert (
    CMAKE
), 'The "cmake" executable is not found. Please check if Cmake is installed.'


TOP_DIR = os.path.dirname(os.path.realpath(__file__))

IS_WINDOWS = os.name == 'nt'


def filter_setup_args(input_args):
    cmake_and_build = True
    only_cmake = False
    rerun_cmake = False
    filter_args_list = []
    for arg in input_args:
        if arg == 'rerun-cmake':
            rerun_cmake = True  # delete CMakeCache.txt and rerun cmake
            continue
        if arg == 'only-cmake':
            only_cmake = True  # only cmake and do not make, leave a chance for users to adjust build options
            continue
        if arg in ['clean', 'egg_info', 'sdist']:
            cmake_and_build = False
        filter_args_list.append(arg)
    return cmake_and_build, only_cmake, rerun_cmake, filter_args_list


cmake_and_build, only_cmake, rerun_cmake, filter_args_list = filter_setup_args(
    sys.argv
)


def parse_input_command(input_parameters):
    dist = Distribution()
    # get script name :setup.py
    sys.argv = input_parameters
    dist.script_name = os.path.basename(sys.argv[0])
    # get args of setup.py
    dist.script_args = sys.argv[1:]
    print(
        "Start executing python {} {}".format(
            dist.script_name, "".join(dist.script_args)
        )
    )
    try:
        dist.parse_command_line()
    except:
        print(
            f"An error occurred while parsing"
            f"the parameters, {dist.script_args}"
        )
        sys.exit(1)


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


RC = 0
ext_suffix = (
    '.dll'
    if os.name == 'nt'
    else ('.dylib' if sys.platform == 'darwin' else '.so')
)


def get_header_install_dir(header):
    if 'pb.h' in header:
        install_dir = re.sub(
            env_dict.get("PADDLE_BINARY_DIR") + '/', '', header
        )
    elif 'third_party' not in header:
        # paddle headers
        install_dir = re.sub(
            env_dict.get("PADDLE_SOURCE_DIR") + '/', '', header
        )
        if 'fluid/jit' in install_dir:
            install_dir = re.sub('fluid/jit', 'jit', install_dir)
    else:
        # third_party
        install_dir = re.sub(
            env_dict.get("THIRD_PARTY_PATH"), 'third_party', header
        )
        patterns = [
            'install/mkldnn/include/',
            'pybind/src/extern_pybind/include/',
            'third_party/xpu/src/extern_xpu/xpu/include/',
        ]
        for pattern in patterns:
            install_dir = re.sub(pattern, '', install_dir)
    return install_dir


class InstallHeaders(Command):
    """Override how headers are copied."""

    description = 'install C/C++ header files'

    user_options = [
        ('install-dir=', 'd', 'directory to install header files to'),
        ('force', 'f', 'force installation (overwrite existing files)'),
    ]

    boolean_options = ['force']

    def initialize_options(self):
        self.install_dir = None
        self.force = 0
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options(
            'install', ('install_headers', 'install_dir'), ('force', 'force')
        )

    def run(self):
        hdrs = self.distribution.headers
        if not hdrs:
            return
        self.mkpath(self.install_dir)
        for header in hdrs:
            install_dir = get_header_install_dir(header)
            install_dir = os.path.join(
                self.install_dir, os.path.dirname(install_dir)
            )
            if not os.path.exists(install_dir):
                self.mkpath(install_dir)
            (out, _) = self.copy_file(header, install_dir)
            self.outfiles.append(out)
            # (out, _) = self.mkdir_and_copy_file(header)
            # self.outfiles.append(out)

    def get_inputs(self):
        return self.distribution.headers or []

    def get_outputs(self):
        return self.outfiles


class InstallCommand(InstallCommandBase):
    def finalize_options(self):
        ret = InstallCommandBase.finalize_options(self)
        self.install_lib = self.install_platlib

        self.install_headers = os.path.join(
            self.install_platlib, 'paddle', 'include'
        )
        return ret


class DevelopCommand(DevelopCommandBase):
    def run(self):
        # copy proto and .so to python_source_dir
        fluid_proto_binary_path = (
            paddle_binary_dir + '/python/paddle/base/proto/'
        )
        fluid_proto_source_path = (
            paddle_source_dir + '/python/paddle/base/proto/'
        )
        distributed_proto_binary_path = (
            paddle_binary_dir + '/python/paddle/distributed/fleet/proto/'
        )
        distributed_proto_source_path = (
            paddle_source_dir + '/python/paddle/distributed/fleet/proto/'
        )
        os.system(f"rm -rf {fluid_proto_source_path}")
        shutil.copytree(fluid_proto_binary_path, fluid_proto_source_path)
        os.system(f"rm -rf {distributed_proto_source_path}")
        shutil.copytree(
            distributed_proto_binary_path, distributed_proto_source_path
        )
        shutil.copy(
            paddle_binary_dir + '/python/paddle/base/libpaddle.so',
            paddle_source_dir + '/python/paddle/base/',
        )
        dynamic_library_binary_path = paddle_binary_dir + '/python/paddle/libs/'
        dynamic_library_source_path = paddle_source_dir + '/python/paddle/libs/'
        for lib_so in os.listdir(dynamic_library_binary_path):
            shutil.copy(
                dynamic_library_binary_path + lib_so,
                dynamic_library_source_path,
            )
        # write version.py and cuda_env_config_py to python_source_dir
        write_version_py(
            filename=f'{paddle_source_dir}/python/paddle/version/__init__.py'
        )
        write_cuda_env_config_py(
            filename=f'{paddle_source_dir}/python/paddle/cuda_env.py'
        )
        write_parameter_server_version_py(
            filename=f'{paddle_source_dir}/python/paddle/incubate/distributed/fleet/parameter_server/version.py'
        )
        DevelopCommandBase.run(self)


class EggInfo(egg_info):
    """Copy license file into `.dist-info` folder."""

    def run(self):
        # don't duplicate license into `.dist-info` when building a distribution
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file(
                env_dict.get("PADDLE_SOURCE_DIR") + "/LICENSE", self.egg_info
            )

        egg_info.run(self)


# class Installlib is rewritten to add header files to .egg/paddle
class InstallLib(install_lib):
    def run(self):
        self.build()
        outfiles = self.install()
        hrds = self.distribution.headers
        if not hrds:
            return
        for header in hrds:
            install_dir = get_header_install_dir(header)
            install_dir = os.path.join(
                self.install_dir, 'paddle/include', os.path.dirname(install_dir)
            )
            if not os.path.exists(install_dir):
                self.mkpath(install_dir)
            self.copy_file(header, install_dir)
        if outfiles is not None:
            # always compile, in case we have any extension stubs to deal with
            self.byte_compile(outfiles)


def git_commit() -> str:
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        git_commit = (
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                cwd=env_dict.get("PADDLE_SOURCE_DIR"),
            )
            .communicate()[0]
            .strip()
        )
    except:
        git_commit = 'Unknown'
    git_commit = git_commit.decode('utf-8')
    return str(git_commit)


def _get_version_detail(idx):
    assert (
        idx < 3
    ), "version info consists of %(major)d.%(minor)d.%(patch)d, \
        so detail index must less than 3"
    tag_version_regex = env_dict.get("TAG_VERSION_REGEX")
    paddle_version = env_dict.get("PADDLE_VERSION")
    if re.match(tag_version_regex, paddle_version):
        version_details = paddle_version.split('.')
        if len(version_details) >= 3:
            return version_details[idx]
    return 0


def _mkdir_p(dir_str):
    try:
        os.makedirs(dir_str)
    except OSError as e:
        raise RuntimeError("Failed to create build folder")


def get_major() -> int:
    return int(_get_version_detail(0))


def get_minor() -> int:
    return int(_get_version_detail(1))


def get_patch() -> int:
    return str(_get_version_detail(2))


def get_nccl_version() -> int:
    if env_dict.get("WITH_NCCL") == 'ON':
        return int(env_dict.get("NCCL_VERSION"))
    return 0


def get_cuda_version() -> str:
    with_gpu = env_dict.get("WITH_GPU")
    if with_gpu == 'ON':
        return env_dict.get("CUDA_VERSION")
    else:
        return 'False'


def get_cudnn_version() -> str:
    with_gpu = env_dict.get("WITH_GPU")
    if with_gpu == 'ON':
        temp_cudnn_version = ''
        cudnn_major_version = env_dict.get("CUDNN_MAJOR_VERSION")
        if cudnn_major_version:
            temp_cudnn_version += cudnn_major_version
            cudnn_minor_version = env_dict.get("CUDNN_MINOR_VERSION")
            if cudnn_minor_version:
                temp_cudnn_version = (
                    temp_cudnn_version + '.' + cudnn_minor_version
                )
                cudnn_patchlevel_version = env_dict.get(
                    "CUDNN_PATCHLEVEL_VERSION"
                )
                if cudnn_patchlevel_version:
                    temp_cudnn_version = (
                        temp_cudnn_version + '.' + cudnn_patchlevel_version
                    )
        return temp_cudnn_version
    else:
        return 'False'


def get_xpu_xre_version() -> str:
    with_xpu = env_dict.get("WITH_XPU")
    if with_xpu == 'ON':
        return env_dict.get("XPU_XRE_BASE_VERSION")
    else:
        return 'False'


def get_xpu_xccl_version() -> str:
    with_xpu_xccl = env_dict.get("WITH_XPU_BKCL")
    if with_xpu_xccl == 'ON':
        return env_dict.get("XPU_XCCL_BASE_VERSION")
    else:
        return 'False'


def get_xpu_xhpc_version() -> str:
    with_xpu_xhpc = env_dict.get("WITH_XPU")
    if with_xpu_xhpc == 'ON':
        return env_dict.get("XPU_XHPC_BASE_DATE")
    else:
        return 'False'


def is_tagged() -> bool:
    try:
        cmd = [
            'git',
            'describe',
            '--exact-match',
            '--tags',
            'HEAD',
            '2>/dev/null',
        ]
        git_tag = (
            subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                cwd=env_dict.get("PADDLE_SOURCE_DIR"),
            )
            .communicate()[0]
            .strip()
        )
        git_tag = git_tag.decode()
    except:
        return False
    if str(git_tag).replace('v', '') == env_dict.get("PADDLE_VERSION"):
        return True
    else:
        return False


def get_cinn_version() -> str:
    if env_dict.get("WITH_CINN") != 'ON':
        return "False"
    return "0.3.0"


def get_cuda_archs() -> list[int]:
    compiled_cuda_archs = env_dict.get("COMPILED_CUDA_ARCHS")
    if isinstance(compiled_cuda_archs, str):
        compiled_cuda_archs = re.findall(r'\d+', compiled_cuda_archs)
        return [int(arch) for arch in compiled_cuda_archs]
    else:
        return []


def get_tensorrt_version() -> str:

    def find_libnvinfer():
        """Search for libnvinfer.so file in LD_LIBRARY_PATH."""

        trt_infer_rt_path = env_dict.get("TR_INFER_RT")
        tensorrt_library_path = env_dict.get("TENSORRT_LIBRARY_DIR")

        libnvinfer_file = os.path.join(tensorrt_library_path, trt_infer_rt_path)

        if os.path.exists(libnvinfer_file):
            return libnvinfer_file
        else:
            print(f"{libnvinfer_file} not found.")
        return None

    try:
        libnvinfer_path = find_libnvinfer()
        if not libnvinfer_path:
            return None

        trt = ctypes.CDLL(libnvinfer_path)
        get_version = trt.getInferLibVersion
        get_version.restype = ctypes.c_int
        version = get_version()
        version_str = str(version)
        major = version_str[:1] if len(version_str) > 1 else version_str
        minor = version_str[1:2] if len(version_str) > 3 else version_str[1:]
        patch = version_str[3:] if len(version_str) > 3 else ''

        minor = minor if minor else '0'
        patch = patch if patch else '0'
        version_str = f"{major}.{minor}.{patch}"

        return version_str

    except Exception as e:
        print(f"Error while getting TensorRT version: {e}")
        return None


def write_version_py(filename='paddle/version/__init__.py'):
    cnt = '''# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
full_version     = '%(major)d.%(minor)d.%(patch)s'
major            = '%(major)d'
minor            = '%(minor)d'
patch            = '%(patch)s'
nccl_version     = '%(nccl)d'
rc               = '%(rc)d'
cuda_version     = '%(cuda)s'
cudnn_version    = '%(cudnn)s'
xpu_xre_version  = '%(xpu_xre)s'
xpu_xccl_version = '%(xpu_xccl)s'
xpu_xhpc_version = '%(xpu_xhpc)s'
is_tagged        = %(is_tagged)s
commit           = '%(commit)s'
with_mkl         = '%(with_mkl)s'
cinn_version     = '%(cinn)s'
tensorrt_version = '%(tensorrt)s'
with_pip_cuda_libraries = '%(with_pip_cuda_libraries)s'
with_pip_tensorrt       = '%(with_pip_tensorrt)s'
compiled_cuda_archs     = %(compiled_cuda_archs)s

__all__ = ['cuda', 'cudnn', 'nccl', 'show', 'xpu', 'xpu_xre', 'xpu_xccl', 'xpu_xhpc', 'tensorrt', 'cuda_archs']

def show() -> None:
    """Get the version of paddle if `paddle` package if tagged. Otherwise, output the corresponding commit id.

    Returns:
        If paddle package is not tagged, the commit-id of paddle will be output.
        Otherwise, the following information will be output.

        full_version: version of paddle

        major: the major version of paddle

        minor: the minor version of paddle

        patch: the patch level version of paddle

        rc: whether it's rc version

        cuda: the cuda version of package. It will return `False` if CPU version paddle package is installed

        cudnn: the cudnn version of package. It will return `False` if CPU version paddle package is installed

        xpu_xre: the xpu xre version of package. It will return `False` if non-XPU version paddle package is installed

        xpu_xccl: the xpu xccl version of package. It will return `False` if non-XPU version paddle package is installed

        xpu_xhpc: the xpu xhpc version of package. It will return `False` if non-XPU version paddle package is installed

        cinn: the cinn version of package. It will return `False` if paddle package is not compiled with CINN

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> # Case 1: paddle is tagged with 2.2.0
            >>> paddle.version.show()
            >>> # doctest: +SKIP('Different environments yield different output.')
            full_version: 2.2.0
            major: 2
            minor: 2
            patch: 0
            rc: 0
            cuda: '10.2'
            cudnn: '7.6.5'
            xpu_xre: '4.32.0.1'
            xpu_xccl: '1.0.7'
            xpu_xhpc: '20231208'
            cinn: False
            >>> # doctest: -SKIP

            >>> # Case 2: paddle is not tagged
            >>> paddle.version.show()
            >>> # doctest: +SKIP('Different environments yield different output.')
            commit: cfa357e984bfd2ffa16820e354020529df434f7d
            cuda: '10.2'
            cudnn: '7.6.5'
            xpu_xre: '4.32.0.1'
            xpu_xccl: '1.0.7'
            xpu_xhpc: '20231208'
            cinn: False
            >>> # doctest: -SKIP
    """
    if is_tagged:
        print('full_version:', full_version)
        print('major:', major)
        print('minor:', minor)
        print('patch:', patch)
        print('rc:', rc)
    else:
        print('commit:', commit)
    print('cuda:', cuda_version)
    print('cudnn:', cudnn_version)
    print('nccl:', nccl_version)
    print('xpu_xre:', xpu_xre_version)
    print('xpu_xccl:', xpu_xccl_version)
    print('xpu_xhpc:', xpu_xhpc_version)
    print('cinn:', cinn_version)
    print('tensorrt_version:', tensorrt_version)
    print('cuda_archs:', compiled_cuda_archs)

def mkl() -> str:
    return with_mkl

def nccl() -> str:
    """Get nccl version of paddle package.

    Returns:
        string: Return the version information of cuda nccl. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.nccl()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '2804'

    """
    return nccl_version

def cuda() -> str:
    """Get cuda version of paddle package.

    Returns:
        string: Return the version information of cuda. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cuda()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '10.2'

    """
    return cuda_version

def cudnn() -> str:
    """Get cudnn version of paddle package.

    Returns:
        string: Return the version information of cudnn. If paddle package is CPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cudnn()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '7.6.5'

    """
    return cudnn_version

def xpu() -> str:
    """Get xpu version of paddle package. The API is deprecated now, please use xpu_xhpc() instead.

    Returns:
        string: Return the version information of xpu. If paddle package is non-XPU version, it will return False.
    Examples:
        .. code-block:: python
            >>> import paddle
            >>> paddle.version.xpu()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '20230114'
    """
    return xpu_xhpc_version

def xpu_xre() -> str:
    """Get xpu xre version of paddle package.

    Returns:
        string: Return the version information of xpu. If paddle package is non-XPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.xpu_xre()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '4.32.0.1'

    """
    return xpu_xre_version

def xpu_xccl() -> str:
    """Get xpu xccl version of paddle package.

    Returns:
        string: Return the version information of xpu xccl. If paddle package is non-XPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.xpu_xccl()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '1.0.7'

    """
    return xpu_xccl_version

def xpu_xhpc() -> str:
    """Get xpu xhpc version of paddle package.

    Returns:
        string: Return the version information of xpu xhpc. If paddle package is non-XPU version, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.xpu_xhpc()
            >>> # doctest: +SKIP('Different environments yield different output.')
            '20231208'

    """
    return xpu_xhpc_version

def cinn() -> str:
    """Get CINN version of paddle package.

    Returns:
        string: Return the version information of CINN. If paddle package is not compiled with CINN, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cinn()
            >>> # doctest: +SKIP('Different environments yield different output.')
            False

    """
    return cinn_version

def tensorrt() -> str:
    """Get TensorRT version of paddle package.

    Returns:
        string: Return the version information of TensorRT. If paddle package is not compiled with TensorRT, it will return False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.tensorrt()
            >>> # doctest: +SKIP('Different environments yield different output.')
            False

    """
    return tensorrt_version

def cuda_archs():
    """Get compiled cuda archs of paddle package.

    Returns:
        list[int]: Return the compiled cuda archs if with gpu. If paddle package is not compiled with gpu, it will return "".

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.version.cuda_archs()
            >>> # doctest: +SKIP('Different environments yield different output.')
            [86]

    """
    return compiled_cuda_archs
'''
    commit = git_commit()

    dirname = os.path.dirname(filename)

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    with open(filename, 'w') as f:
        f.write(
            cnt
            % {
                'major': get_major(),
                'minor': get_minor(),
                'patch': get_patch(),
                'nccl': get_nccl_version(),
                'rc': RC,
                'version': env_dict.get("PADDLE_VERSION"),
                'cuda': get_cuda_version(),
                'cudnn': get_cudnn_version(),
                'xpu_xre': get_xpu_xre_version(),
                'xpu_xccl': get_xpu_xccl_version(),
                'xpu_xhpc': get_xpu_xhpc_version(),
                'commit': commit,
                'is_tagged': is_tagged(),
                'with_mkl': env_dict.get("WITH_MKL"),
                'cinn': get_cinn_version(),
                'tensorrt': get_tensorrt_version(),
                'with_pip_cuda_libraries': env_dict.get(
                    "WITH_PIP_CUDA_LIBRARIES"
                ),
                'with_pip_tensorrt': env_dict.get("WITH_PIP_TENSORRT"),
                'compiled_cuda_archs': get_cuda_archs(),
            }
        )


def write_cuda_env_config_py(filename='paddle/cuda_env.py'):
    cnt = ""
    if env_dict.get("JIT_RELEASE_WHL") == 'ON':
        cnt = '''# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY
#
import os
os.environ['CUDA_CACHE_MAXSIZE'] = '805306368'
'''

    with open(filename, 'w') as f:
        f.write(cnt)


def write_parameter_server_version_py(
    filename='paddle/incubate/distributed/fleet/parameter_server/version.py',
):
    cnt = '''

# THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

from paddle.incubate.distributed.fleet.base import Mode

BUILD_MODE=Mode.%(mode)s

def is_transpiler():
    return Mode.TRANSPILER == BUILD_MODE

'''

    dirname = os.path.dirname(filename)

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(filename, 'w') as f:
        f.write(
            cnt
            % {
                'mode': (
                    'PSLIB'
                    if env_dict.get("WITH_PSLIB") == 'ON'
                    else 'TRANSPILER'
                )
            }
        )


def find_files(pattern, root, recursive=False):
    for dirpath, _, files in os.walk(root):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(dirpath, filename)
        if not recursive:
            break


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError(f'Can only cd to absolute path, got: {path}')
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def options_process(args, build_options):
    for key, value in sorted(build_options.items()):
        if value is not None:
            args.append(f"-D{key}={value}")


def get_cmake_generator():
    if os.getenv("GENERATOR"):
        cmake_generator = os.getenv("GENERATOR")
        if os.system('ninja --version') == 0:
            print("Ninja has been installed,use ninja to compile Paddle now.")
        else:
            print("Ninja has not been installed,install it now.")
            os.system('python -m pip install ninja')
    else:
        cmake_generator = "Unix Makefiles"
    return cmake_generator


def cmake_run(build_path):
    args = []
    env_var = os.environ.copy()  # get env variables
    paddle_build_options = {}
    other_options = {}
    other_options.update(
        {
            option: option
            for option in (
                "PYTHON_LIBRARY",
                "INFERENCE_DEMO_INSTALL_DIR",
                "ON_INFER",
                "PYTHON_EXECUTABLE",
                "TENSORRT_ROOT",
                "CUDA_ARCH_NAME",
                "CUDA_ARCH_BIN",
                "PYTHON_INCLUDE_DIR",
                "PYTHON_LIBRARIES",
                "PY_VERSION",
                "CUB_PATH",
                "NEW_RELEASE_PYPI",
                "CUDNN_ROOT",
                "THIRD_PARTY_PATH",
                "NOAVX_CORE_FILE",
                "LITE_GIT_TAG",
                "CUDA_TOOLKIT_ROOT_DIR",
                "NEW_RELEASE_JIT",
                "XPU_SDK_ROOT",
                "MSVC_STATIC_CRT",
                "NEW_RELEASE_ALL",
                "GENERATOR",
            )
        }
    )
    # if environment variables which start with "WITH_" or "CMAKE_",put it into build_options
    for option_key, option_value in env_var.items():
        if option_key.startswith(("CMAKE_", "WITH_")):
            paddle_build_options[option_key] = option_value
        if option_key in other_options:
            if (
                option_key == 'PYTHON_EXECUTABLE'
                or option_key == 'PYTHON_LIBRARY'
                or option_key == 'PYTHON_LIBRARIES'
            ):
                key = option_key + ":FILEPATH"
            elif option_key == 'PYTHON_INCLUDE_DIR':
                key = option_key + ':PATH'
            elif option_key == 'GENERATOR':
                key = 'CMAKE_' + option_key
            else:
                key = other_options[option_key]
            if key not in paddle_build_options:
                paddle_build_options[key] = option_value

    options_process(args, paddle_build_options)
    with cd(build_path):
        cmake_args = []
        cmake_args.append(CMAKE)
        cmake_args += args
        cmake_args.append('-DWITH_SETUP_INSTALL=ON')
        cmake_args.append(TOP_DIR)
        subprocess.check_call(cmake_args)


def build_run(args, build_path, environ_var):
    with cd(build_path):
        build_args = []
        build_args.append(CMAKE)
        build_args += args
        try:
            subprocess.check_call(build_args, cwd=build_path, env=environ_var)
        except (CalledProcessError, KeyboardInterrupt) as e:
            sys.exit(1)


def run_cmake_build(build_path):
    build_type = (
        os.getenv("CMAKE_BUILD_TYPE")
        if os.getenv("CMAKE_BUILD_TYPE") is not None
        else "release"
    )
    build_args = ["--build", ".", "--target", "install", "--config", build_type]
    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs is not None:
        max_jobs = max_jobs or str(multiprocessing.cpu_count())

        build_args += ["--"]
        if IS_WINDOWS:
            build_args += [f"/p:CL_MPCount={max_jobs}"]
        else:
            build_args += ["-j", max_jobs]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]
    environ_var = os.environ.copy()
    build_run(build_args, build_path, environ_var)


def build_steps():
    print('------- Building start ------')
    build_dir = os.getenv("BUILD_DIR")
    if build_dir is not None:
        build_dir = TOP_DIR + '/' + build_dir
    else:
        build_dir = TOP_DIR + '/build'
    if not os.path.exists(build_dir):
        _mkdir_p(build_dir)
    build_path = build_dir
    print("build_dir:", build_dir)
    # run cmake to generate native build files
    cmake_cache_file_path = os.path.join(build_path, "CMakeCache.txt")
    # if rerun_cmake is True,remove CMakeCache.txt and rerun cmake
    if os.path.isfile(cmake_cache_file_path) and rerun_cmake is True:
        os.remove(cmake_cache_file_path)

    CMAKE_GENERATOR = get_cmake_generator()
    bool_ninja = CMAKE_GENERATOR == "Ninja"
    build_ninja_file_path = os.path.join(build_path, "build.ninja")
    if os.path.exists(cmake_cache_file_path) and not (
        bool_ninja and not os.path.exists(build_ninja_file_path)
    ):
        print("Do not need rerun cmake, everything is ready, run build now")
    else:
        cmake_run(build_path)
    # make
    if only_cmake:
        print(
            "You have finished running cmake, the program exited,run 'cmake build' to adjust build options and 'python setup.py install to build'"
        )
        sys.exit()
    run_cmake_build(build_path)


def get_setup_requires():
    with open(
        env_dict.get("PADDLE_SOURCE_DIR") + '/python/requirements.txt'
    ) as f:
        setup_requires = (
            f.read().splitlines()
        )  # Specify the dependencies to install
    if sys.version_info >= (3, 8):
        setup_requires_tmp = []
        for setup_requires_i in setup_requires:
            if (
                '<"3.6"' in setup_requires_i
                or '<="3.6"' in setup_requires_i
                or '<"3.5"' in setup_requires_i
                or '<="3.5"' in setup_requires_i
                or '<"3.7"' in setup_requires_i
                or '<="3.7"' in setup_requires_i
                or '<"3.8"' in setup_requires_i
                or setup_requires_i.strip().endswith('[build]')
            ):
                continue
            setup_requires_tmp += [setup_requires_i]
        setup_requires = setup_requires_tmp

        return setup_requires
    else:
        raise RuntimeError(
            "please check your python version,Paddle only support Python version>=3.8 now"
        )


def get_paddle_extra_install_requirements():
    paddle_cuda_requires = []
    paddle_tensorrt_requires = []
    # (Note risemeup1): Paddle will install the pypi cuda package provided by Nvidia, which includes the cuda runtime, cudnn, and cublas, thereby making the operation of 'pip install paddle' no longer dependent on the installation of cuda and cudnn.
    if env_dict.get("WITH_PIP_CUDA_LIBRARIES") == "ON":
        if platform.system() == 'Linux':
            PADDLE_CUDA_INSTALL_REQUIREMENTS = {
                "V11": (
                    "nvidia-cuda-runtime-cu11==11.8.89; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cuda-cupti-cu11==11.8.87; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cudnn-cu11==8.9.6.50; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cublas-cu11==11.11.3.6; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cufft-cu11==10.9.0.58; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-curand-cu11==10.3.0.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cusolver-cu11==11.4.1.48; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cusparse-cu11==11.7.5.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-nccl-cu11==2.19.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-nvtx-cu11==11.8.86; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cuda-nvrtc-cu11==11.8.89; platform_system == 'Linux' and platform_machine == 'x86_64'"
                ),
                "V12": (
                    "nvidia-cuda-runtime-cu12==12.3.101; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cuda-cupti-cu12==12.3.101; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cudnn-cu12==9.1.1.17; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cublas-cu12==12.3.4.1; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cufft-cu12==11.2.1.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-curand-cu12==10.3.5.147; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cusolver-cu12==11.6.1.9; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cusparse-cu12==12.3.1.170; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-nccl-cu12==2.19.3; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-nvtx-cu12==12.4.127; platform_system == 'Linux' and platform_machine == 'x86_64' | "
                    "nvidia-cuda-nvrtc-cu12==12.3.107; platform_system == 'Linux' and platform_machine == 'x86_64'"
                ),
            }
        elif platform.system() == 'Windows':
            PADDLE_CUDA_INSTALL_REQUIREMENTS = {
                "V11": (
                    "nvidia-cuda-runtime-cu11==11.8.89 | "
                    "nvidia-cudnn-cu11==8.9.4.19 | "
                    "nvidia-cublas-cu11==11.11.3.6 | "
                    "nvidia-cufft-cu11==10.9.0.58 | "
                    "nvidia-curand-cu11==10.3.0.86 | "
                    "nvidia-cusolver-cu11==11.4.1.48 | "
                    "nvidia-cusparse-cu11==11.7.5.86 "
                ),
                "V12": (
                    "nvidia-cuda-runtime-cu12==12.3.101 | "
                    "nvidia-cudnn-cu12==9.1.1.17 | "
                    "nvidia-cublas-cu12==12.3.4.1 | "
                    "nvidia-cufft-cu12==11.2.1.3 | "
                    "nvidia-curand-cu12==10.3.5.147 | "
                    "nvidia-cusolver-cu12==11.6.1.9 | "
                    "nvidia-cusparse-cu12==12.3.1.170 "
                ),
            }
        try:
            output = subprocess.check_output(['nvcc', '--version']).decode(
                'utf-8'
            )
            version_line = next(
                line for line in output.split('\n') if 'release' in line
            )
            version = version_line.split(' ')[-1].split(',')[0]
            cuda_major_version = version.split('.')[0]
        except Exception as e:
            raise ValueError("CUDA not found")

        paddle_cuda_requires = PADDLE_CUDA_INSTALL_REQUIREMENTS[
            cuda_major_version
        ].split("|")

    if env_dict.get("WITH_PIP_TENSORRT") == "ON":
        version_str = get_tensorrt_version()
        version_default = int(version_str.split(".")[0])
        if platform.system() == 'Linux' or (
            platform.system() == 'Windows' and version_default >= 10
        ):

            PADDLE_TENSORRT_INSTALL_REQUIREMENTS = [
                "tensorrt==8.5.3.1",
                "tensorrt==8.6.0",
                "tensorrt==8.6.1.post1",
            ]

            if not version_str:
                return paddle_cuda_requires, []

            version_main = ".".join(version_str.split(".")[:3])

            matched_package = None
            for (
                paddle_tensorrt_requires
            ) in PADDLE_TENSORRT_INSTALL_REQUIREMENTS:
                paddle_tensorrt_version = paddle_tensorrt_requires.split("==")[
                    1
                ]
                paddle_tensorrt_main = ".".join(
                    paddle_tensorrt_version.split(".")[:3]
                )

                if version_main == paddle_tensorrt_main:
                    matched_package = paddle_tensorrt_requires
                    break

            if matched_package:
                paddle_tensorrt_requires = [matched_package]
            else:
                print(
                    f"No exact match found for TensorRT Version: {version_str}. We currently support TensorRT versions 8.5.3.1, 8.6.0, and 8.6.1."
                )
                return paddle_cuda_requires, []

    return paddle_cuda_requires, paddle_tensorrt_requires


def get_cinn_config_jsons():
    from pathlib import Path

    src_cinn_config_path = (
        env_dict.get("PADDLE_SOURCE_DIR") + '/python/paddle/cinn_config'
    )
    prefix_len = len(src_cinn_config_path) + 1
    p = Path(src_cinn_config_path)
    json_list = list(p.glob('**/*.json'))
    json_path_list = []
    for json in json_list:
        json = str(json)
        json = json[prefix_len:]
        json_path_list += [json]
    return json_path_list


def get_typing_libs_packages(paddle_binary_dir):
    """get all libpaddle sub modules from 'python/paddle/_typing/libs/libpaddle'
    e.g.
        'paddle._typing.libs.libpaddle.cinn'
        'paddle._typing.libs.libpaddle.pir'
        'paddle._typing.libs.libpaddle.eager'
        'paddle._typing.libs.libpaddle.eager.ops'
    """
    base_dir = Path(paddle_binary_dir) / 'python'
    libs_dir = base_dir / 'paddle' / '_typing' / 'libs' / 'libpaddle'
    return [
        '.'.join(str(Path(root).relative_to(base_dir)).split(os.sep))
        for root, _, _ in os.walk(libs_dir)
    ]


def extend_type_hints_package_data(packages, package_data, paddle_binary_dir):
    typing_libs_packages = get_typing_libs_packages(paddle_binary_dir)

    # update packages
    packages += typing_libs_packages

    # update package_data
    type_hints_files = {
        'paddle': ['py.typed', '*.pyi'],
        'paddle.framework': ['*.pyi'],
        'paddle.base': ['*.pyi'],
        'paddle.tensor': ['tensor.pyi'],
        'paddle._typing': ['*.pyi'],
        'paddle._typing.libs': ['*.pyi', '*.md'],
    }

    for libpaddle_module in typing_libs_packages:
        type_hints_files[libpaddle_module] = ['*.pyi']

    for pkg, files in type_hints_files.items():
        if pkg not in package_data:
            package_data[pkg] = []
        package_data[pkg] += files

    return packages, package_data


def get_package_data_and_package_dir():
    if os.name != 'nt':
        package_data = {
            'paddle.base': [env_dict.get("FLUID_CORE_NAME") + '.so']
        }
    else:
        package_data = {
            'paddle.base': [
                env_dict.get("FLUID_CORE_NAME") + '.pyd',
                env_dict.get("FLUID_CORE_NAME") + '.lib',
            ]
        }
    package_data['paddle.base'] += [
        paddle_binary_dir + '/python/paddle/cost_model/static_op_benchmark.json'
    ]

    whl_cinn_config_path = paddle_binary_dir + '/python/paddle/cinn_config'
    src_cinn_config_path = (
        env_dict.get("PADDLE_SOURCE_DIR") + '/python/paddle/cinn_config'
    )
    package_data['paddle.cinn_config'] = []
    if os.path.exists(whl_cinn_config_path):
        shutil.rmtree(whl_cinn_config_path)
    shutil.copytree(src_cinn_config_path, whl_cinn_config_path)
    json_path_list = get_cinn_config_jsons()
    for json in json_path_list:
        package_data['paddle.cinn_config'] += [json]

    if 'develop' in sys.argv:
        package_dir = {'': 'python'}
    else:
        package_dir = {
            '': env_dict.get("PADDLE_BINARY_DIR") + '/python',
            'paddle.base.proto.profiler': env_dict.get("PADDLE_BINARY_DIR")
            + '/paddle/fluid/platform',
            'paddle.base.proto': env_dict.get("PADDLE_BINARY_DIR")
            + '/paddle/fluid/framework',
            'paddle.base': env_dict.get("PADDLE_BINARY_DIR")
            + '/python/paddle/base',
        }
    # put all thirdparty libraries in paddle.libs
    libs_path = paddle_binary_dir + '/python/paddle/libs'
    package_data['paddle.libs'] = []
    if env_dict.get("WITH_SHARED_PHI") == "ON":
        package_data['paddle.libs'] += [
            ('libphi' if os.name != 'nt' else 'phi') + ext_suffix
        ]
        shutil.copy(env_dict.get("PHI_LIB"), libs_path)
        package_data['paddle.libs'] += [
            ('libphi_core' if os.name != 'nt' else 'phi_core') + ext_suffix
        ]
        shutil.copy(env_dict.get("PHI_CORE_LIB"), libs_path)
        if (
            env_dict.get("WITH_GPU") == "ON"
            or env_dict.get("WITH_ROCM") == "ON"
        ):
            package_data['paddle.libs'] += [
                ('libphi_gpu' if os.name != 'nt' else 'phi_gpu') + ext_suffix
            ]
            shutil.copy(env_dict.get("PHI_GPU_LIB"), libs_path)

    if env_dict.get("WITH_SHARED_IR") == "ON":
        package_data['paddle.libs'] += [
            ('libpir' if os.name != 'nt' else 'pir') + ext_suffix
        ]
        shutil.copy(env_dict.get("IR_LIB"), libs_path)

    package_data['paddle.libs'] += [
        ('libwarpctc' if os.name != 'nt' else 'warpctc') + ext_suffix,
        ('libwarprnnt' if os.name != 'nt' else 'warprnnt') + ext_suffix,
    ]
    package_data['paddle.libs'] += [
        ('libcommon' if os.name != 'nt' else 'common') + ext_suffix,
    ]
    shutil.copy(env_dict.get("COMMON_LIB"), libs_path)
    shutil.copy(env_dict.get("WARPCTC_LIBRARIES"), libs_path)
    shutil.copy(env_dict.get("WARPRNNT_LIBRARIES"), libs_path)
    package_data['paddle.libs'] += [
        os.path.basename(env_dict.get("LAPACK_LIB")),
        os.path.basename(env_dict.get("BLAS_LIB")),
        os.path.basename(env_dict.get("GFORTRAN_LIB")),
        os.path.basename(env_dict.get("GNU_RT_LIB_1")),
    ]
    shutil.copy(env_dict.get("BLAS_LIB"), libs_path)
    shutil.copy(env_dict.get("LAPACK_LIB"), libs_path)
    shutil.copy(env_dict.get("GFORTRAN_LIB"), libs_path)
    shutil.copy(env_dict.get("GNU_RT_LIB_1"), libs_path)

    if not sys.platform.startswith("linux"):
        package_data['paddle.libs'] += [
            os.path.basename(env_dict.get("GNU_RT_LIB_2"))
        ]
        shutil.copy(env_dict.get("GNU_RT_LIB_2"), libs_path)
    if env_dict.get("WITH_MKL") == 'ON':
        shutil.copy(env_dict.get("MKLML_SHARED_LIB"), libs_path)
        shutil.copy(env_dict.get("MKLML_SHARED_IOMP_LIB"), libs_path)
        package_data['paddle.libs'] += [
            ('libmklml_intel' if os.name != 'nt' else 'mklml') + ext_suffix,
            ('libiomp5' if os.name != 'nt' else 'libiomp5md') + ext_suffix,
        ]
    else:
        if os.name == 'nt':
            # copy the openblas.dll
            shutil.copy(env_dict.get("OPENBLAS_SHARED_LIB"), libs_path)
            package_data['paddle.libs'] += ['openblas' + ext_suffix]
        elif (
            os.name == 'posix'
            and platform.machine() == 'aarch64'
            and env_dict.get("OPENBLAS_LIB").endswith('so')
        ):
            # copy the libopenblas.so on linux+aarch64
            # special: libpaddle.so without avx depends on 'libopenblas.so.0', not 'libopenblas.so'
            if os.path.exists(env_dict.get("OPENBLAS_LIB") + '.0'):
                shutil.copy(env_dict.get("OPENBLAS_LIB") + '.0', libs_path)
                package_data['paddle.libs'] += ['libopenblas.so.0']

    if env_dict.get("WITH_GPU") == 'ON' or env_dict.get("WITH_ROCM") == 'ON':
        if len(env_dict.get("FLASHATTN_LIBRARIES", "")) > 1:
            package_data['paddle.libs'] += [
                os.path.basename(env_dict.get("FLASHATTN_LIBRARIES"))
            ]
            shutil.copy(env_dict.get("FLASHATTN_LIBRARIES"), libs_path)
        if len(env_dict.get("FLASHATTN_V3_LIBRARIES", "")) > 1:
            package_data['paddle.libs'] += [
                os.path.basename(env_dict.get("FLASHATTN_V3_LIBRARIES"))
            ]
            shutil.copy(env_dict.get("FLASHATTN_V3_LIBRARIES"), libs_path)
    if env_dict.get("WITH_CINN") == 'ON':
        shutil.copy(
            env_dict.get("CINN_LIB_LOCATION")
            + '/'
            + env_dict.get("CINN_LIB_NAME"),
            libs_path,
        )
        shutil.copy(
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/cuda/cinn_cuda_runtime_source.cuh',
            libs_path,
        )
        shutil.copy(
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/hip/cinn_hip_runtime_source.h',
            libs_path,
        )
        package_data['paddle.libs'] += ['libcinnapi.so']
        package_data['paddle.libs'] += ['cinn_cuda_runtime_source.cuh']
        package_data['paddle.libs'] += ['cinn_hip_runtime_source.h']

        cinn_fp16_file = (
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/cuda/float16.h'
        )
        if os.path.exists(cinn_fp16_file):
            shutil.copy(cinn_fp16_file, libs_path)
            package_data['paddle.libs'] += ['float16.h']
        cinn_bf16_file = (
            env_dict.get("CINN_INCLUDE_DIR")
            + '/paddle/cinn/runtime/cuda/bfloat16.h'
        )
        if os.path.exists(cinn_bf16_file):
            shutil.copy(cinn_bf16_file, libs_path)
            package_data['paddle.libs'] += ['bfloat16.h']

        if env_dict.get("CMAKE_BUILD_TYPE") == 'Release' and os.name != 'nt':
            command = (
                f"patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_nvrtc/lib/:$ORIGIN/../../nvidia/cuda_runtime/lib/:$ORIGIN/../../nvidia/cublas/lib/:$ORIGIN/../../nvidia/cudnn/lib/:$ORIGIN/../../nvidia/curand/lib/:$ORIGIN/../../nvidia/cusolver/lib/:$ORIGIN/../../nvidia/nvtx/lib/:$ORIGIN/' {libs_path}/"
                + env_dict.get("CINN_LIB_NAME")
            )
            if os.system(command) != 0:
                raise Exception(
                    'patch '
                    + libs_path
                    + '/'
                    + env_dict.get("CINN_LIB_NAME")
                    + ' failed',
                    f'command: {command}',
                )
    if env_dict.get("WITH_PSLIB") == 'ON':
        shutil.copy(env_dict.get("PSLIB_LIB"), libs_path)
        shutil.copy(env_dict.get("JVM_LIB"), libs_path)
        if os.path.exists(env_dict.get("PSLIB_VERSION_PY")):
            shutil.copy(
                env_dict.get("PSLIB_VERSION_PY"),
                paddle_binary_dir
                + '/python/paddle/incubate/distributed/fleet/parameter_server/pslib/',
            )
        package_data['paddle.libs'] += ['libps' + ext_suffix]
        package_data['paddle.libs'] += ['libjvm' + ext_suffix]
    if env_dict.get("WITH_ONEDNN") == 'ON':
        if env_dict.get("CMAKE_BUILD_TYPE") == 'Release' and os.name != 'nt':
            # only change rpath in Release mode.
            # TODO(typhoonzero): use install_name_tool to patch mkl libs once
            # we can support mkl on mac.
            #
            # change rpath of libdnnl.so.1, add $ORIGIN/ to it.
            # The reason is that all thirdparty libraries in the same directory,
            # thus, libdnnl.so.1 will find libmklml_intel.so and libiomp5.so.
            command = "patchelf --set-rpath '$ORIGIN/' " + env_dict.get(
                "ONEDNN_SHARED_LIB"
            )
            if os.system(command) != 0:
                raise Exception(f"patch libdnnl.so failed, command: {command}")
        shutil.copy(env_dict.get("ONEDNN_SHARED_LIB"), libs_path)
        if os.name != 'nt':
            package_data['paddle.libs'] += ['libdnnl.so.3']
        else:
            package_data['paddle.libs'] += ['mkldnn.dll']

    if env_dict.get("WITH_ONNXRUNTIME") == 'ON':
        shutil.copy(env_dict.get("ONNXRUNTIME_SHARED_LIB"), libs_path)
        shutil.copy(env_dict.get("PADDLE2ONNX_LIB"), libs_path)
        if os.name == 'nt':
            package_data['paddle.libs'] += [
                'paddle2onnx.dll',
                'onnxruntime.dll',
            ]
        else:
            package_data['paddle.libs'] += [
                env_dict.get("PADDLE2ONNX_LIB_NAME"),
                env_dict.get("ONNXRUNTIME_LIB_NAME"),
            ]

    if env_dict.get("WITH_OPENVINO") == 'ON':
        shutil.copy(env_dict.get("OPENVINO_LIB"), libs_path)
        shutil.copy(env_dict.get("TBB_LIB"), libs_path)
        shutil.copy(env_dict.get("OPENVINO_PADDLE_LIB"), libs_path)
        shutil.copy(env_dict.get("OPENVINO_CPU_PLUGIN_LIB"), libs_path)
        if os.name != 'nt':
            package_data['paddle.libs'] += [
                'libopenvino.so.2500',
                'libtbb.so.12',
                'libopenvino_paddle_frontend.so.2500',
                'libopenvino_intel_cpu_plugin.so',
            ]
        else:
            package_data['paddle.libs'] += [
                'openvino.dll',
                'tbb.dll',
                'openvino_paddle_frontend.dll',
                'openvino_intel_cpu_plugin.dll',
            ]

    if env_dict.get("WITH_XPU") == 'ON':
        shutil.copy(env_dict.get("XPU_API_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_API_LIB_NAME")]
        xpu_rt_lib_list = glob.glob(env_dict.get("XPU_RT_LIB") + '*')
        for xpu_rt_lib_file in xpu_rt_lib_list:
            shutil.copy(xpu_rt_lib_file, libs_path)
            package_data['paddle.libs'] += [os.path.basename(xpu_rt_lib_file)]
        xpu_cuda_lib_list = glob.glob(env_dict.get("XPU_CUDA_LIB") + '*')
        for xpu_cuda_lib_file in xpu_cuda_lib_list:
            shutil.copy(xpu_cuda_lib_file, libs_path)
            package_data['paddle.libs'] += [os.path.basename(xpu_cuda_lib_file)]
        if env_dict.get("WITH_XPU_XRE5") == 'ON':
            xpu_cuda_rt_lib_list = glob.glob(
                env_dict.get("XPU_CUDA_RT_LIB") + '*'
            )
            for xpu_cuda_rt_lib_file in xpu_cuda_rt_lib_list:
                shutil.copy(xpu_cuda_rt_lib_file, libs_path)
                package_data['paddle.libs'] += [
                    os.path.basename(xpu_cuda_rt_lib_file)
                ]
            xpu_ml_lib_list = glob.glob(env_dict.get("XPU_ML_LIB") + '*')
            for xpu_ml_lib_file in xpu_ml_lib_list:
                shutil.copy(xpu_ml_lib_file, libs_path)
                package_data['paddle.libs'] += [
                    os.path.basename(xpu_ml_lib_file)
                ]
            shutil.copy(env_dict.get("XPU_XBLAS_LIB"), libs_path)
            package_data['paddle.libs'] += [env_dict.get("XPU_XBLAS_LIB_NAME")]
            shutil.copy(env_dict.get("XPU_XFA_LIB"), libs_path)
            package_data['paddle.libs'] += [env_dict.get("XPU_XFA_LIB_NAME")]
            shutil.copy(env_dict.get("XPU_XPUDNN_LIB"), libs_path)
            package_data['paddle.libs'] += [env_dict.get("XPU_XPUDNN_LIB_NAME")]

    if env_dict.get("WITH_XPU_BKCL") == 'ON':
        shutil.copy(env_dict.get("XPU_BKCL_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_BKCL_LIB_NAME")]

    if env_dict.get("WITH_XPU_XFT") == 'ON':
        shutil.copy(env_dict.get("XPU_XFT_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_XFT_LIB_NAME")]

    if env_dict.get("WITH_XPTI") == 'ON':
        shutil.copy(env_dict.get("XPU_XPTI_LIB"), libs_path)
        package_data['paddle.libs'] += [env_dict.get("XPU_XPTI_LIB_NAME")]

    # remove unused paddle/libs/__init__.py
    if os.path.isfile(libs_path + '/__init__.py'):
        os.remove(libs_path + '/__init__.py')
    package_dir['paddle.libs'] = libs_path

    # change rpath of ${FLUID_CORE_NAME}.ext, add $ORIGIN/../libs/ to it.
    # The reason is that libwarpctc.ext, libwarprnnt.ext, libiomp5.ext etc are in paddle.libs, and
    # ${FLUID_CORE_NAME}.ext is in paddle.base, thus paddle/fluid/../libs will pointer to above libraries.
    # This operation will fix https://github.com/PaddlePaddle/Paddle/issues/3213
    if env_dict.get("CMAKE_BUILD_TYPE") == 'Release':
        if os.name != 'nt':
            # only change rpath in Release mode, since in Debug mode, ${FLUID_CORE_NAME}.xx is too large to be changed.
            if env_dict.get("APPLE") == "1":
                commands = [
                    "install_name_tool -id '@loader_path/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/base/'
                    + env_dict.get("FLUID_CORE_NAME")
                    + '.so'
                ]
                commands.append(
                    "install_name_tool -add_rpath '@loader_path/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/base/'
                    + env_dict.get("FLUID_CORE_NAME")
                    + '.so'
                )
                commands.append(
                    "install_name_tool -add_rpath '@loader_path/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/libs/'
                    + env_dict.get("COMMON_NAME")
                )
                if env_dict.get("WITH_SHARED_PHI") == "ON":
                    commands.append(
                        "install_name_tool -add_rpath '@loader_path' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("PHI_NAME")
                    )
                    commands.append(
                        "install_name_tool -add_rpath '@loader_path' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("PHI_CORE_NAME")
                    )
                    if (
                        env_dict.get("WITH_GPU") == "ON"
                        or env_dict.get("WITH_ROCM") == "ON"
                    ):
                        commands.append(
                            "install_name_tool -add_rpath '@loader_path' "
                            + env_dict.get("PADDLE_BINARY_DIR")
                            + '/python/paddle/libs/'
                            + env_dict.get("PHI_GPU_NAME")
                        )
                if env_dict.get("WITH_SHARED_IR") == "ON":
                    commands.append(
                        "install_name_tool -add_rpath '@loader_path' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("IR_NAME")
                    )
            else:
                commands = [
                    "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cuda_nvrtc/lib:$ORIGIN/../../nvidia/cublas/lib:$ORIGIN/../../nvidia/cudnn/lib:$ORIGIN/../../nvidia/curand/lib:$ORIGIN/../../nvidia/cusparse/lib:$ORIGIN/../../nvidia/nvjitlink/lib:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cufft/lib:$ORIGIN/../../nvidia/cufft/lib:$ORIGIN/../../nvidia/cusolver/lib:$ORIGIN/../../nvidia/nccl/lib:$ORIGIN/../../nvidia/nvtx/lib:$ORIGIN/../libs/' "
                    + env_dict.get("PADDLE_BINARY_DIR")
                    + '/python/paddle/base/'
                    + env_dict.get("FLUID_CORE_NAME")
                    + '.so'
                ]
                if env_dict.get("WITH_SHARED_PHI") == "ON":
                    commands.append(
                        "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN:$ORIGIN/../libs' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("PHI_NAME")
                    )
                    commands.append(
                        "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN:$ORIGIN/../libs' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("PHI_CORE_NAME")
                    )
                    if (
                        env_dict.get("WITH_GPU") == "ON"
                        or env_dict.get("WITH_ROCM") == "ON"
                    ):
                        commands.append(
                            "patchelf --set-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN:$ORIGIN/../libs' "
                            + env_dict.get("PADDLE_BINARY_DIR")
                            + '/python/paddle/libs/'
                            + env_dict.get("PHI_GPU_NAME")
                        )

                if env_dict.get("WITH_SHARED_IR") == "ON":
                    commands.append(
                        "patchelf --set-rpath '$ORIGIN:$ORIGIN/../libs' "
                        + env_dict.get("PADDLE_BINARY_DIR")
                        + '/python/paddle/libs/'
                        + env_dict.get("IR_NAME")
                    )
            # The sw_64 not support patchelf, so we just disable that.
            if platform.machine() != 'sw_64' and platform.machine() != 'mips64':
                for command in commands:
                    if os.system(command) != 0:
                        raise Exception(
                            'patch '
                            + env_dict.get("FLUID_CORE_NAME")
                            + f'{ext_suffix} failed',
                            f'command: {command}',
                        )
    # A list of extensions that specify c++ -written modules that compile source code into dynamically linked libraries
    ext_modules = [Extension('_foo', [paddle_binary_dir + '/python/stub.cc'])]
    if os.name == 'nt':
        # fix the path separator under windows
        fix_package_dir = {}
        for k, v in package_dir.items():
            fix_package_dir[k] = v.replace('/', '\\')
        package_dir = fix_package_dir
        ext_modules = []
    elif sys.platform == 'darwin':
        ext_modules = []

    return package_data, package_dir, ext_modules


def get_headers():
    headers = (
        # paddle level api headers (high level api, for both training and inference)
        list(find_files('*.h', paddle_source_dir + '/paddle'))
        + list(find_files('*.h', paddle_source_dir + '/paddle/phi/api'))
        + list(  # phi unify api header
            find_files('*.h', paddle_source_dir + '/paddle/phi/api/ext')
        )
        + list(  # custom op api
            find_files('*.h', paddle_source_dir + '/paddle/phi/api/include')
        )
        + list(  # phi api
            find_files('*.h', paddle_source_dir + '/paddle/phi/common')
        )
        + list(  # common api
            find_files('*.h', paddle_source_dir + '/paddle/common')
        )
        # phi level api headers (low level api, for training only)
        + list(  # phi extension header
            find_files('*.h', paddle_source_dir + '/paddle/phi')
        )
        + list(  # phi include header
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/include', recursive=True
            )
        )
        + list(  # phi backends headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/phi/backends',
                recursive=True,
            )
        )
        + list(  # phi core headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/core', recursive=True
            )
        )
        + list(  # phi infermeta headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/phi/infermeta',
                recursive=True,
            )
        )
        + list(  # phi kernel headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/phi/kernels',
                recursive=True,
            )
        )
        # phi capi headers
        + list(
            find_files(
                '*.h', paddle_source_dir + '/paddle/phi/capi', recursive=True
            )
        )
        + list(  # utils api headers
            find_files(
                '*.h', paddle_source_dir + '/paddle/utils', recursive=True
            )
        )
        + list(  # phi profiler headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/phi/api/profiler',
                recursive=True,
            )
        )
        + list(  # phi init headers
            find_files(
                'init_phi.h',
                paddle_source_dir + '/paddle/fluid/platform',
                recursive=True,
            )
        )
        + list(  # pir init headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/pir/include',
                recursive=True,
            )
        )
        + list(  # drr init headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/fluid/pir/drr/include',
                recursive=True,
            )
        )
        + list(  # drr init headers
            find_files(
                '*.h',
                paddle_source_dir
                + '/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape',
                recursive=True,
            )
        )
        + list(  # operator init headers
            find_files(
                '*.h',
                paddle_source_dir + '/paddle/fluid/pir/dialect/operator/ir',
            )
        )
        + list(  # pass utils init headers
            find_files(
                'general_functions.h',
                paddle_source_dir + '/paddle/fluid/pir/utils',
            )
        )
        + list(  # serialize and deserialize interface headers
            find_files(
                'interface.h',
                paddle_source_dir
                + '/paddle/fluid/pir/serialize_deserialize/include',
            )
        )
        + list(  # serialize and deserialize interface headers
            find_files(
                'dense_tensor.inl',
                paddle_source_dir + '/paddle/phi/core',
            )
        )
        + list(  # serialize and deserialize interface headers
            find_files(
                'op_yaml_info.h',
                paddle_source_dir
                + '/paddle/fluid/pir/dialect/operator/interface',
            )
        )
        + list(  # serialize and deserialize interface headers
            find_files(
                'op_yaml_info_util.h',
                paddle_source_dir + '/paddle/fluid/pir/dialect/operator/utils',
            )
        )
        + list(  # op yaml parser interface headers
            find_files(
                'op_yaml_info_parser.h',
                paddle_source_dir + '/paddle/fluid/pir/dialect/operator/utils',
            )
        )
        + list(  # pir op utils interface headers
            find_files(
                'utils.h',
                paddle_source_dir + '/paddle/fluid/pir/dialect/operator/utils',
            )
        )
        + list(  # ir translator interface headers
            find_files(
                'op_compat_info.h',
                paddle_source_dir + '/paddle/fluid/ir_adaptor/translator/',
            )
        )
        + list(
            find_files(
                'infer_symbolic_shape.h',
                paddle_source_dir
                + '/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape',
            )
        )
        + list(
            find_files(
                'vjp.h',
                paddle_source_dir
                + '/paddle/fluid/pir/dialect/operator/interface',
            )
        )
        + list(
            find_files(
                'lexer.h',
                paddle_source_dir + '/paddle/pir/src/core/parser',
            )
        )
        + list(
            find_files(
                'token.h',
                paddle_source_dir + '/paddle/pir/src/core/parser',
            )
        )
    )

    jit_layer_headers = [
        'layer.h',
        'serializer.h',
        'serializer_utils.h',
        'all.h',
        'function.h',
    ]

    for f in jit_layer_headers:
        headers += list(
            find_files(
                f, paddle_source_dir + '/paddle/fluid/jit', recursive=True
            )
        )

    if env_dict.get("WITH_ONEDNN") == 'ON':
        headers += list(
            find_files('*', env_dict.get("ONEDNN_INSTALL_DIR") + '/include')
        )  # mkldnn

    if env_dict.get("WITH_OPENVINO") == 'ON':
        headers += list(
            find_files('*', env_dict.get("OPENVINO_INC_DIR"))
        )  # openvino
        headers += list(
            find_files('*', env_dict.get("TBB_INC_DIR"))
        )  # openvino

    if env_dict.get("WITH_GPU") == 'ON' or env_dict.get("WITH_ROCM") == 'ON':
        # externalErrorMsg.pb for External Error message
        headers += list(
            find_files('*.pb', env_dict.get("externalError_INCLUDE_DIR"))
        )

    if env_dict.get("WITH_XPU") == 'ON':
        headers += list(
            find_files(
                '*.h',
                paddle_binary_dir + '/third_party/xpu/src/extern_xpu/xpu',
                recursive=True,
            )
        )  # xdnn api headers
        headers += list(
            find_files(
                '*.hpp',
                paddle_binary_dir + '/third_party/xpu/src/extern_xpu/xpu',
                recursive=True,
            )
        )  # xre headers with .hpp extension

    # pybind headers
    headers += list(find_files('*.h', env_dict.get("PYBIND_INCLUDE_DIR"), True))
    return headers


def get_setup_parameters():
    # get setup_requires
    setup_requires = get_setup_requires()
    if (
        env_dict.get("WITH_GPU") == 'ON'
        and platform.system() in ('Linux', 'Windows')
        and platform.machine()
        in (
            'x86_64',
            'AMD64',
        )
    ):
        paddle_cuda_requires, paddle_tensorrt_requires = (
            get_paddle_extra_install_requirements()
        )
        setup_requires += paddle_cuda_requires
        setup_requires += paddle_tensorrt_requires

    packages = [
        'paddle',
        'paddle.libs',
        'paddle.utils',
        'paddle.utils.gast',
        'paddle.utils.cpp_extension',
        'paddle.dataset',
        'paddle.reader',
        'paddle.distributed',
        'paddle.distributed.checkpoint',
        'paddle.distributed.communication',
        'paddle.distributed.communication.stream',
        'paddle.distributed.metric',
        'paddle.distributed.ps',
        'paddle.distributed.ps.utils',
        'paddle.incubate',
        'paddle.incubate.autograd',
        'paddle.incubate.optimizer',
        'paddle.incubate.checkpoint',
        'paddle.incubate.operators',
        'paddle.incubate.tensor',
        'paddle.incubate.multiprocessing',
        'paddle.incubate.nn',
        'paddle.incubate.jit',
        'paddle.incubate.asp',
        'paddle.incubate.passes',
        'paddle.incubate.framework',
        'paddle.distribution',
        'paddle.distributed.utils',
        'paddle.distributed.sharding',
        'paddle.distributed.fleet',
        'paddle.distributed.auto_tuner',
        'paddle.distributed.launch',
        'paddle.distributed.launch.context',
        'paddle.distributed.launch.controllers',
        'paddle.distributed.launch.job',
        'paddle.distributed.launch.plugins',
        'paddle.distributed.launch.utils',
        'paddle.distributed.fleet.base',
        'paddle.distributed.fleet.recompute',
        'paddle.distributed.fleet.elastic',
        'paddle.distributed.fleet.meta_optimizers',
        'paddle.distributed.fleet.meta_optimizers.sharding',
        'paddle.distributed.fleet.meta_optimizers.dygraph_optimizer',
        'paddle.distributed.fleet.runtime',
        'paddle.distributed.rpc',
        'paddle.distributed.fleet.dataset',
        'paddle.distributed.fleet.data_generator',
        'paddle.distributed.fleet.metrics',
        'paddle.distributed.fleet.proto',
        'paddle.distributed.fleet.utils',
        'paddle.distributed.fleet.layers',
        'paddle.distributed.fleet.layers.mpu',
        'paddle.distributed.fleet.meta_parallel',
        'paddle.distributed.fleet.meta_parallel.pp_utils',
        'paddle.distributed.fleet.meta_parallel.sharding',
        'paddle.distributed.fleet.meta_parallel.parallel_layers',
        'paddle.distributed.auto_parallel',
        'paddle.distributed.auto_parallel.intermediate',
        'paddle.distributed.auto_parallel.dygraph',
        'paddle.distributed.auto_parallel.static',
        'paddle.distributed.auto_parallel.static.operators',
        'paddle.distributed.auto_parallel.static.tuner',
        'paddle.distributed.auto_parallel.static.cost',
        'paddle.distributed.auto_parallel.static.reshard_funcs',
        'paddle.distributed.passes',
        'paddle.distributed.passes.pipeline_scheduler_pass',
        'paddle.distributed.models',
        'paddle.distributed.models.moe',
        'paddle.distributed.transpiler',
        'paddle.distributed.transpiler.details',
        'paddle.framework',
        'paddle.jit',
        'paddle.jit.dy2static',
        'paddle.jit.dy2static.transformers',
        'paddle.jit.pir_dy2static',
        'paddle.jit.sot',
        'paddle.jit.sot.opcode_translator',
        'paddle.jit.sot.opcode_translator.executor',
        'paddle.jit.sot.opcode_translator.executor.variables',
        'paddle.jit.sot.opcode_translator.instruction_utils',
        'paddle.jit.sot.profiler',
        'paddle.jit.sot.symbolic',
        'paddle.jit.sot.utils',
        'paddle.inference',
        'paddle.inference.contrib',
        'paddle.inference.contrib.utils',
        'paddle.base',
        'paddle.base.dygraph',
        'paddle.base.proto',
        'paddle.base.proto.profiler',
        'paddle.base.layers',
        'paddle.base.incubate',
        'paddle.incubate.distributed.fleet',
        'paddle.base.incubate.checkpoint',
        'paddle.amp',
        'paddle.cost_model',
        'paddle.cinn_config',
        'paddle.hapi',
        'paddle.vision',
        'paddle.vision.models',
        'paddle.vision.transforms',
        'paddle.vision.datasets',
        'paddle.audio',
        'paddle.audio.functional',
        'paddle.audio.features',
        'paddle.audio.datasets',
        'paddle.audio.backends',
        'paddle.text',
        'paddle.text.datasets',
        'paddle.incubate',
        'paddle.incubate.nn',
        'paddle.incubate.jit',
        'paddle.incubate.nn.functional',
        'paddle.incubate.nn.layer',
        'paddle.incubate.optimizer.functional',
        'paddle.incubate.autograd',
        'paddle.incubate.distributed',
        'paddle.incubate.distributed.utils',
        'paddle.incubate.distributed.utils.io',
        'paddle.incubate.distributed.fleet',
        'paddle.incubate.distributed.models',
        'paddle.incubate.distributed.models.moe',
        'paddle.incubate.distributed.models.moe.gate',
        'paddle.incubate.distributed.fleet.parameter_server',
        'paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler',
        'paddle.incubate.distributed.fleet.parameter_server.ir',
        'paddle.incubate.distributed.fleet.parameter_server.pslib',
        'paddle.incubate.layers',
        'paddle.quantization',
        'paddle.quantization.quanters',
        'paddle.quantization.observers',
        'paddle.sparse',
        'paddle.sparse.nn',
        'paddle.sparse.nn.layer',
        'paddle.sparse.nn.functional',
        'paddle.incubate.xpu',
        'paddle.io',
        'paddle.io.dataloader',
        'paddle.optimizer',
        'paddle.nn',
        'paddle.nn.functional',
        'paddle.nn.layer',
        'paddle.nn.quant',
        'paddle.nn.quant.qat',
        'paddle.nn.initializer',
        'paddle.nn.utils',
        'paddle.metric',
        'paddle.static',
        'paddle.static.nn',
        'paddle.static.amp',
        'paddle.static.amp.bf16',
        'paddle.static.quantization',
        'paddle.quantization',
        'paddle.quantization.imperative',
        'paddle.tensor',
        'paddle.onnx',
        'paddle.autograd',
        'paddle.device',
        'paddle.device.cuda',
        'paddle.device.xpu',
        'paddle.version',
        'paddle.profiler',
        'paddle.geometric',
        'paddle.geometric.message_passing',
        'paddle.geometric.sampling',
        'paddle.pir',
        'paddle.decomposition',
        'paddle._typing',
        'paddle._typing.libs',
        'paddle.api_tracer',
    ]

    if env_dict.get("WITH_TENSORRT") == 'ON':
        packages.extend(
            [
                'paddle.tensorrt',
                'paddle.tensorrt.impls',
            ]
        )
    paddle_bins = ''
    if not env_dict.get("WIN32"):
        paddle_bins = [
            env_dict.get("PADDLE_BINARY_DIR") + '/paddle/scripts/paddle'
        ]
    package_data, package_dir, ext_modules = get_package_data_and_package_dir()
    headers = get_headers()
    return (
        setup_requires,
        packages,
        paddle_bins,
        package_data,
        package_dir,
        ext_modules,
        headers,
    )


def check_build_dependency():
    missing_modules = '''Missing build dependency: {dependency}
Please run 'pip install -r python/requirements.txt' to make sure you have all the dependencies installed.
'''.strip()

    with open(TOP_DIR + '/python/requirements.txt') as f:
        build_dependencies = (
            f.read().splitlines()
        )  # Specify the dependencies to install

    python_dependencies_module = []
    installed_packages = []

    for dependency in build_dependencies:
        python_dependencies_module.append(
            re.sub("_|-", '', re.sub(r"==.*|>=.*|<=.*", '', dependency))
        )
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])

    for r in reqs.split():
        installed_packages.append(
            re.sub("_|-", '', r.decode().split('==')[0]).lower()
        )

    for dependency in python_dependencies_module:
        if dependency.lower() not in installed_packages:
            raise RuntimeError(missing_modules.format(dependency=dependency))


def install_cpp_dist_and_build_test(install_dir, lib_test_dir, headers, libs):
    """install cpp distribution and build test target

    TODO(huangjiyi):
    1. This function will be moved when separating C++ distribution
    installation from python package installation.
    2. Reduce the header and library files to be installed.
    """
    if env_dict.get("CMAKE_BUILD_TYPE") != 'Release':
        return
    os.makedirs(install_dir, exist_ok=True)
    # install C++ header files
    for header in headers:
        header_install_dir = get_header_install_dir(header)
        header_install_dir = os.path.join(
            install_dir, 'include', os.path.dirname(header_install_dir)
        )
        os.makedirs(header_install_dir, exist_ok=True)
        shutil.copy(header, header_install_dir)

    # install C++ shared libraries
    lib_install_dir = os.path.join(install_dir, 'lib')
    os.makedirs(lib_install_dir, exist_ok=True)
    # install libpaddle.ext
    paddle_libs = glob.glob(
        paddle_binary_dir
        + '/paddle/fluid/pybind/'
        + env_dict.get("FLUID_CORE_NAME")
        + '.*'
    )
    for lib in paddle_libs:
        shutil.copy(lib, lib_install_dir)
    # install dependent libraries
    libs_path = paddle_binary_dir + '/python/paddle/libs'
    for lib in libs:
        lib_path = os.path.join(libs_path, lib)
        shutil.copy(lib_path, lib_install_dir)

    # build test target
    cmake_args = [CMAKE, lib_test_dir, "-B", lib_test_dir]
    if os.getenv("GENERATOR") == "Ninja":
        cmake_args.append("-GNinja")
    subprocess.check_call(cmake_args)
    subprocess.check_call([CMAKE, "--build", lib_test_dir])


def check_submodules():
    def get_submodule_folder():
        git_submodules_path = os.path.join(TOP_DIR, ".gitmodules")
        with open(git_submodules_path) as f:
            return [
                os.path.join(TOP_DIR, line.split("=", 1)[1].strip())
                for line in f
                if line.strip().startswith("path")
            ]

    def submodules_not_exists_or_empty(folder):
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    submodule_folders = get_submodule_folder()
    # f none of the submodule folders exists, try to initialize them
    if any(
        submodules_not_exists_or_empty(folder) for folder in submodule_folders
    ):
        try:
            print(' --- Trying to initialize submodules')
            start = time.time()
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=TOP_DIR,
            )
            end = time.time()
            print(f' --- Submodule initialization took {end - start:.2f} sec')
        except Exception:
            print(' --- Submodule initialization failed')
            print('Please run:\n\tgit submodule update --init --recursive')
            sys.exit(1)


def generate_stub_files(paddle_binary_dir, paddle_source_dir):
    script_path = paddle_source_dir + '/tools/'
    sys.path.append(script_path)

    print('-' * 2, 'Generate stub file tensor.pyi ... ')
    import gen_tensor_stub

    gen_tensor_stub.generate_stub_file(
        input_file=paddle_source_dir
        + '/python/paddle/tensor/tensor.prototype.pyi',
        output_file=paddle_binary_dir + '/python/paddle/tensor/tensor.pyi',
    )

    shutil.copy(
        paddle_binary_dir + '/python/paddle/tensor/tensor.pyi',
        paddle_source_dir + '/python/paddle/tensor/tensor.pyi',
    )
    print('-' * 2, 'End Generate stub file tensor.pyi ... ')

    print('-' * 2, 'Generate stub file for python binding APIs ... ')
    import gen_pybind11_stub

    gen_pybind11_stub.generate_stub_file(
        output_dir=str(Path(paddle_binary_dir) / 'python/paddle/_typing/libs/'),
        module_name='paddle.base.libpaddle',
        ignore_all_errors=True,
        ops_yaml=[
            paddle_source_dir
            + "/paddle/phi/ops/yaml/ops.yaml;paddle.base.libpaddle.eager.ops",
            paddle_source_dir
            + "/paddle/phi/ops/yaml/ops.yaml;paddle.base.libpaddle.pir.ops",
            paddle_source_dir
            + "/paddle/phi/ops/yaml/sparse_ops.yaml;paddle.base.libpaddle.eager.ops;sparse",
            paddle_source_dir
            + "/paddle/phi/ops/yaml/sparse_ops.yaml;paddle.base.libpaddle.pir.ops;sparse",
            paddle_source_dir
            + "/paddle/phi/ops/yaml/strings_ops.yaml;paddle.base.libpaddle.eager.ops;strings",
            paddle_source_dir
            + "/paddle/phi/ops/yaml/strings_ops.yaml;paddle.base.libpaddle.pir.ops;strings",
        ],
    )

    libpaddle_dst = paddle_source_dir + '/python/paddle/_typing/libs/libpaddle'
    if Path(libpaddle_dst).exists():
        shutil.rmtree(libpaddle_dst)

    shutil.copytree(
        paddle_binary_dir + '/python/paddle/_typing/libs/libpaddle',
        libpaddle_dst,
    )

    print('-' * 2, 'End Generate stub for python binding APIs ... ')


def main():
    # Parse the command line and check arguments before we proceed with building steps and setup
    parse_input_command(filter_args_list)

    # check build dependency
    check_build_dependency()
    check_submodules()
    # Execute the build process,cmake and make
    if cmake_and_build:
        build_steps()

    if os.getenv("WITH_PYTHON") == "OFF":
        print("only compile, not package")
        return

    build_dir = os.getenv("BUILD_DIR")
    if build_dir is not None:
        env_dict_path = TOP_DIR + '/' + build_dir + '/python'
    else:
        env_dict_path = TOP_DIR + "/build/python/"
    sys.path.insert(1, env_dict_path)
    from env_dict import env_dict

    global env_dict  # noqa: F811
    global paddle_binary_dir, paddle_source_dir

    paddle_binary_dir = env_dict.get("PADDLE_BINARY_DIR")
    paddle_source_dir = env_dict.get("PADDLE_SOURCE_DIR")

    # preparing parameters for setup()
    paddle_version = env_dict.get("PADDLE_VERSION")
    package_name = env_dict.get("PACKAGE_NAME")

    write_version_py(
        filename=f'{paddle_binary_dir}/python/paddle/version/__init__.py'
    )
    write_cuda_env_config_py(
        filename=f'{paddle_binary_dir}/python/paddle/cuda_env.py'
    )
    write_parameter_server_version_py(
        filename=f'{paddle_binary_dir}/python/paddle/incubate/distributed/fleet/parameter_server/version.py'
    )
    (
        setup_requires,
        packages,
        scripts,
        package_data,
        package_dir,
        ext_modules,
        headers,
    ) = get_setup_parameters()

    # Log for PYPI, get long_description of setup()
    with open(
        paddle_source_dir + '/python/paddle/README.md', "r", encoding='UTF-8'
    ) as f:
        long_description = f.read()

    # strip *.so to reduce package size
    if env_dict.get("WITH_STRIP") == 'ON':
        command = (
            'find '
            + shlex.quote(paddle_binary_dir)
            + '/python/paddle -name "*.so" | xargs -i strip {}'
        )
        if os.system(command) != 0:
            raise Exception(f"strip *.so failed, command: {command}")

    # install cpp distribution
    if env_dict.get("WITH_CPP_DIST") == 'ON':
        paddle_install_dir = env_dict.get("PADDLE_INSTALL_DIR")
        paddle_lib_test_dir = env_dict.get("PADDLE_LIB_TEST_DIR")
        install_cpp_dist_and_build_test(
            paddle_install_dir,
            paddle_lib_test_dir,
            headers,
            package_data['paddle.libs'],
        )

    # generate stub file `tensor.pyi`
    if os.getenv("SKIP_STUB_GEN", '').lower() not in [
        'y',
        'yes',
        't',
        'true',
        'on',
        '1',
    ]:
        generate_stub_files(paddle_binary_dir, paddle_source_dir)
    # package stub files
    packages, package_data = extend_type_hints_package_data(
        packages, package_data, paddle_binary_dir
    )

    setup(
        name=package_name,
        version=paddle_version,
        description='Parallel Distributed Deep Learning',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author_email="Paddle-better@baidu.com",
        maintainer="PaddlePaddle",
        maintainer_email="Paddle-better@baidu.com",
        url='https://www.paddlepaddle.org.cn/',
        download_url='https://github.com/paddlepaddle/paddle',
        license='Apache Software License',
        packages=packages,
        install_requires=setup_requires,
        ext_modules=ext_modules,
        package_data=package_data,
        package_dir=package_dir,
        scripts=scripts,
        distclass=BinaryDistribution,
        headers=headers,
        cmdclass={
            'install_headers': InstallHeaders,
            'install': InstallCommand,
            'egg_info': EggInfo,
            'install_lib': InstallLib,
            'develop': DevelopCommand,
        },
        entry_points={
            'console_scripts': [
                'fleetrun = paddle.distributed.launch.main:launch'
            ]
        },
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Typing :: Typed',
        ],
    )


if __name__ == '__main__':
    main()
