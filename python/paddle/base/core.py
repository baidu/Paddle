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
import platform
import site
import sys
import warnings

has_paddle_dy_lib = False

dy_lib_name = 'libpaddle'
dy_lib_suffix = 'so'
if os.name == 'nt':
    dy_lib_suffix = 'pyd'

current_path = os.path.abspath(os.path.dirname(__file__))
if os.path.exists(current_path + os.sep + dy_lib_name + '.' + dy_lib_suffix):
    has_paddle_dy_lib = True


try:
    if os.name == 'nt':
        third_lib_path = current_path + os.sep + '..' + os.sep + 'libs'
        # Will load shared library from 'path' on windows
        os.environ['path'] = (
            current_path + ';' + third_lib_path + ';' + os.environ['path']
        )
        sys.path.insert(0, third_lib_path)
        # Note: from python3.8, PATH will not take effect
        # https://github.com/python/cpython/pull/12302
        # Use add_dll_directory to specify dll resolution path
        os.add_dll_directory(third_lib_path)

except ImportError as e:
    if os.name == 'nt':
        executable_path = os.path.abspath(os.path.dirname(sys.executable))
        raise ImportError(
            f"""NOTE: You may need to run \"set PATH={executable_path};%PATH%\"
        if you encounters \"DLL load failed\" errors. If you have python
        installed in other directory, replace \"{executable_path}\" with your own
        directory. The original error is: \n {e}"""
        )
    else:
        raise ImportError(
            """NOTE: You may need to run \"export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\"
        if you encounters \"libmkldnn.so not found\" errors. If you have python
        installed in other directory, replace \"/usr/local/lib\" with your own
        directory. The original error is: \n"""
            + str(e)
        )
except Exception as e:
    raise e


def avx_supported():
    """
    Whether current system(Linux, MacOS, Windows) is supported with AVX.
    """
    sysstr = platform.system().lower()
    has_avx = False
    if sysstr == 'linux':
        try:
            pipe = os.popen('cat /proc/cpuinfo | grep -i avx')
            has_avx = pipe.read() != ''
            pipe.close()
        except Exception as e:
            sys.stderr.write(
                'Can not get the AVX flag from /proc/cpuinfo.\n'
                f'The original error is: {e}\n'
            )
        return has_avx
    elif sysstr == 'darwin':
        try:
            pipe = os.popen('sysctl machdep.cpu.features | grep -i avx')
            has_avx = pipe.read() != ''
            pipe.close()
        except Exception as e:
            sys.stderr.write(
                'Can not get the AVX flag from machdep.cpu.features.\n'
                f'The original error is: {e}\n'
            )
        if not has_avx:
            import subprocess

            pipe = subprocess.Popen(
                'sysctl machdep.cpu.leaf7_features | grep -i avx',
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _ = pipe.communicate()
            has_avx = True if pipe.returncode == 0 else False
        return has_avx
    elif sysstr == 'windows':
        import ctypes

        ONE_PAGE = ctypes.c_size_t(0x1000)

        def asm_func(code_str, restype=ctypes.c_uint32, argtypes=()):
            # Call the code_str as a function
            # Alloc 1 page to ensure the protection
            pfnVirtualAlloc = ctypes.windll.kernel32.VirtualAlloc
            pfnVirtualAlloc.restype = ctypes.c_void_p
            MEM_COMMIT = ctypes.c_ulong(0x1000)
            PAGE_READWRITE = ctypes.c_ulong(0x4)
            address = pfnVirtualAlloc(
                None, ONE_PAGE, MEM_COMMIT, PAGE_READWRITE
            )
            if not address:
                raise Exception("Failed to VirtualAlloc")

            # Copy the code into the memory segment
            memmove = ctypes.CFUNCTYPE(
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            )(ctypes._memmove_addr)
            if memmove(address, code_str, len(code_str)) < 0:
                raise Exception("Failed to memmove")

            # Enable execute permissions
            PAGE_EXECUTE = ctypes.c_ulong(0x10)
            pfnVirtualProtect = ctypes.windll.kernel32.VirtualProtect
            res = pfnVirtualProtect(
                ctypes.c_void_p(address),
                ONE_PAGE,
                PAGE_EXECUTE,
                ctypes.byref(ctypes.c_ulong(0)),
            )
            if not res:
                raise Exception("Failed VirtualProtect")

            # Flush instruction cache
            pfnGetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            pfnGetCurrentProcess.restype = ctypes.c_void_p
            prochandle = ctypes.c_void_p(pfnGetCurrentProcess())
            res = ctypes.windll.kernel32.FlushInstructionCache(
                prochandle, ctypes.c_void_p(address), ONE_PAGE
            )
            if not res:
                raise Exception("Failed FlushInstructionCache")

            # Cast the memory to function
            functype = ctypes.CFUNCTYPE(restype, *argtypes)
            func = functype(address)
            return func, address

        # http://en.wikipedia.org/wiki/CPUID#EAX.3D1:_Processor_Info_and_Feature_Bits
        # mov eax,0x1; cpuid; mov cx, ax; ret
        code_str = b"\xB8\x01\x00\x00\x00\x0f\xa2\x89\xC8\xC3"
        avx_bit = 28
        retval = 0
        try:
            # Convert the code_str into a function that returns uint
            func, address = asm_func(code_str)
            retval = func()
            ctypes.windll.kernel32.VirtualFree(
                ctypes.c_void_p(address), ctypes.c_size_t(0), ONE_PAGE
            )
        except Exception as e:
            sys.stderr.write(
                'Failed getting the AVX flag on Windows.\n'
                f'The original error is: {e}\n'
            )
        return (retval & (1 << avx_bit)) > 0
    else:
        sys.stderr.write(f'Do not get AVX flag on {sysstr}\n')
        return False


def run_shell_command(cmd):
    import subprocess

    out, err = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    ).communicate()
    if err:
        return None
    else:
        return out.decode('utf-8').strip()


def get_dso_path(core_so, dso_name):
    if core_so and dso_name:
        return run_shell_command(
            f"ldd {core_so}|grep {dso_name}|awk '{{print $3}}'"
        )
    else:
        return None


def load_dso(dso_absolute_path):
    if dso_absolute_path:
        try:
            from ctypes import cdll

            cdll.LoadLibrary(dso_absolute_path)
        except:
            warnings.warn(f"Load {dso_absolute_path} failed")


def pre_load(dso_name):
    if has_paddle_dy_lib:
        core_so = current_path + os.sep + dy_lib_name + '.' + dy_lib_suffix
    else:
        core_so = None
    dso_path = get_dso_path(core_so, dso_name)
    load_dso(dso_path)


def get_libc_ver():
    ldd_glibc = run_shell_command("ldd --version | awk '/ldd/{print $NF}'")
    if ldd_glibc is not None:
        return ("glibc", ldd_glibc)

    ldd_musl = run_shell_command("ldd 2>&1 | awk '/Version/{print $NF}'")
    if ldd_musl is not None:
        return ("musl", ldd_musl)
    return (None, None)


def less_than_ver(a, b):
    if a is None or b is None:
        return False

    import operator
    import re

    def to_list(s):
        s = re.sub(r'(\.0+)+$', '', s)
        return [int(x) for x in s.split('.')]

    return operator.lt(to_list(a), to_list(b))


# NOTE(zhiqiu): An error may occurs when import paddle in linux platform with glibc < 2.22,
# the error message of which is "dlopen: cannot load any more object with static TLS".
# This happens when:
# (1) the number of dynamic shared libraries (DSO) loaded > 14,
# (2) after that, load a dynamic shared library (DSO) with static TLS.
# For paddle, the problem is that 'libgomp' is a DSO with static TLS, and it is loaded after 14 DSOs.
# So, here is a tricky way to solve the problem by pre load 'libgomp' before 'libpaddle.so'.
# The final solution is to upgrade glibc to > 2.22 on the target system.
if platform.system().lower() == 'linux':
    libc_type, libc_ver = get_libc_ver()
    if libc_type == 'glibc' and less_than_ver(libc_ver, '2.23'):
        try:
            pre_load('libgomp')
        except Exception as e:
            # NOTE(zhiqiu): do not abort if failed, since it may success when import libpaddle.so
            sys.stderr.write('Error: Can not preload libgomp.so')

try:
    from . import libpaddle

    if avx_supported() and not libpaddle.is_compiled_with_avx():
        sys.stderr.write(
            "Hint: Your machine support AVX, but the installed paddlepaddle doesn't have avx core. "
            "Hence, no-avx core with worse performance will be imported.\nIf you like, you could "
            "reinstall paddlepaddle by 'python -m pip install --force-reinstall paddlepaddle-gpu[==version]' "
            "to get better performance.\n"
        )

    # assign tensor alias
    libpaddle.LoDTensor = libpaddle.DenseTensor
    libpaddle.Tensor = libpaddle.DenseTensor
    libpaddle.VarDesc.VarType.LOD_TENSOR = (
        libpaddle.VarDesc.VarType.DENSE_TENSOR
    )
    libpaddle.VarDesc.VarType.LOD_TENSOR_ARRAY = (
        libpaddle.VarDesc.VarType.DENSE_TENSOR_ARRAY
    )

    from .libpaddle import *  # noqa: F403
    from .libpaddle import (  # noqa: F401
        __doc__,
        __file__,
        __name__,
        __package__,
        __unittest_throw_exception__,
        _append_python_callable_object_and_return_id,
        _cleanup,
        _create_loaded_parameter,
        _cuda_synchronize,
        _device_synchronize,
        _dygraph_debug_level,
        _get_all_register_op_kernels,
        _get_amp_attrs,
        _get_amp_op_list,
        _get_current_stream,
        _get_eager_deletion_vars,
        _get_phi_kernel_name,
        _get_registered_phi_kernels,
        _get_use_default_grad_op_desc_maker_ops,
        _has_grad,
        _is_compiled_with_heterps,
        _is_dygraph_debug_enabled,
        _is_program_version_supported,
        _Profiler,
        _ProfilerResult,
        _promote_types_if_complex_exists,
        _RecordEvent,
        _Scope,
        _set_amp_op_list,
        _set_current_stream,
        _set_eager_deletion_mode,
        _set_fuse_parameter_group_size,
        _set_fuse_parameter_memory_size,
        _set_has_grad,
        _set_paddle_lib_path,
        _set_warmup,
        _switch_tracer,
        _test_enforce_gpu_success,
        _xpu_device_synchronize,
    )

    # isort: off

    # custom device
    from .libpaddle import (  # noqa: F401
        CustomDeviceEvent,
        CustomDeviceStream,
        _get_current_custom_device_stream,
        _set_current_custom_device_stream,
        _synchronize_custom_device,
    )

    # prim controller flags
    from .libpaddle import (  # noqa: F401
        __set_all_prim_enabled,
        __set_bwd_prim_enabled,
        __set_eager_prim_enabled,
        __set_fwd_prim_enabled,
        _add_skip_comp_ops,
        _is_bwd_prim_enabled,
        _is_eager_prim_enabled,
        _is_fwd_prim_enabled,
        _remove_skip_comp_ops,
        _set_bwd_prim_blacklist,
        _set_prim_target_grad_name,
        _enter_dual_level,
        _exit_dual_level,
    )

    # type promotion

    # isort: on
    if sys.platform != 'win32':
        from .libpaddle import (  # noqa: F401
            _array_to_share_memory_tensor,
            _cleanup_mmap_fds,
            _convert_to_tensor_list,
            _erase_process_pids,
            _remove_tensor_list_mmap_fds,
            _set_max_memory_map_allocation_pool_size,
            _set_process_pids,
            _set_process_signal_handler,
            _throw_error_if_process_failed,
        )

except Exception as e:
    if has_paddle_dy_lib:
        sys.stderr.write(
            'Error: Can not import paddle core while this file exists: '
            + current_path
            + os.sep
            + 'libpaddle.'
            + dy_lib_suffix
            + '\n'
        )
    if not avx_supported() and libpaddle.is_compiled_with_avx():
        sys.stderr.write(
            "Error: Your machine doesn't support AVX, but the installed PaddlePaddle is avx core, "
            "you should reinstall paddlepaddle with no-avx core.\n"
        )
    raise e


def set_paddle_custom_device_lib_path(lib_path):
    if os.environ.get('CUSTOM_DEVICE_ROOT', None) is not None:
        # use setted environment value
        return
    if os.path.exists(lib_path):
        # set CUSTOM_DEVICE_ROOT default path
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.normpath(lib_path)
    else:
        os.environ['CUSTOM_DEVICE_ROOT'] = ''


# set paddle lib path
def set_paddle_lib_path():
    site_dirs = (
        site.getsitepackages()
        if hasattr(site, 'getsitepackages')
        else [x for x in sys.path if 'site-packages' in x]
    )
    for site_dir in site_dirs:
        lib_dir = os.path.sep.join([site_dir, 'paddle', 'libs'])
        if os.path.exists(lib_dir):
            _set_paddle_lib_path(lib_dir)
            set_paddle_custom_device_lib_path(
                os.path.sep.join([lib_dir, '..', '..', 'paddle_custom_device'])
            )
            return
    if hasattr(site, 'USER_SITE'):
        lib_dir = os.path.sep.join([site.USER_SITE, 'paddle', 'libs'])
        if os.path.exists(lib_dir):
            _set_paddle_lib_path(lib_dir)
            set_paddle_custom_device_lib_path(
                os.path.sep.join([lib_dir, '..', '..', 'paddle_custom_device'])
            )


set_paddle_lib_path()


# This api is used for check of model output.
# In some cases, model does not straightly return data which can be used for check.
# When this flag is set true, required data should be returned in model.
def _model_return_data():
    flag = os.getenv("FLAGS_model_return_data")
    if flag and flag.lower() in ("1", "true"):
        return True
    else:
        return False


# This api is used for check whether prim is on
def _prim_return_log():
    flag = os.getenv("FLAGS_prim_log")
    if flag and flag.lower() in ("1", "true"):
        return True
    else:
        return False


# We have 3 FLAGS to judge whether prim is enabled
# FLAGS_prim_forward: Open or close forward prim strategy
# FLAGS_prim_backward: Open or close backward prim strategy
# FLAGS_prim_all: Open or close all prim strategy
#
#
# Priorities:
# if With CINN and Dy2St:
# # # _set_prim_all_enabled > FLAGS_prim_all > check_and_set_prim_all_enabled == _set_prim_backward_enabled == _set_prim_backward_enabled > FLAGS_prim_forward == FLAGS_prim_backward
# else:
# # # _set_prim_all_enabled > FLAGS_prim_all == check_and_set_prim_all_enabled == _set_prim_backward_enabled == _set_prim_backward_enabled > FLAGS_prim_forward == FLAGS_prim_backward
def __sync_stat_with_flag(flag, print_flag=True):
    if flag == "FLAGS_prim_forward":
        flag_value = os.getenv("FLAGS_prim_forward")
        assert flag_value is not None
        flag_value = flag_value.lower()
        if flag_value in ("false", "0"):
            __set_fwd_prim_enabled(False)
        elif flag_value in ("true", "1"):
            __set_fwd_prim_enabled(True)
        else:
            raise TypeError(f"flag {flag} should be true or false.")
        if print_flag:
            print("forward prim enabled: ", bool(_is_fwd_prim_enabled()))
    elif flag == "FLAGS_prim_backward":
        flag_value = os.getenv("FLAGS_prim_backward")
        assert flag_value is not None
        flag_value = flag_value.lower()
        if flag_value in ("false", "0"):
            __set_bwd_prim_enabled(False)
        elif flag_value in ("true", "1"):
            __set_bwd_prim_enabled(True)
        else:
            raise TypeError(f"flag {flag} should be true or false.")
        if print_flag:
            print("backward prim enabled: ", bool(_is_bwd_prim_enabled()))
    elif flag == "FLAGS_prim_all":
        flag_value = os.getenv("FLAGS_prim_all")
        assert flag_value is not None
        flag_value = flag_value.lower()
        if flag_value in ("false", "0"):
            __set_all_prim_enabled(False)
        elif flag_value in ("true", "1"):
            __set_all_prim_enabled(True)
        else:
            raise TypeError(f"flag {flag} should be true or false.")
        if print_flag:
            print(
                "all prim enabled: ",
                bool(_is_fwd_prim_enabled() and _is_bwd_prim_enabled()),
            )
    else:
        raise TypeError(
            f"We only support FLAGS_prim_forward/FLAGS_prim_backward/FLAGS_prim_all but we got {flag}."
        )


def _is_all_prim_enabled():
    return _is_fwd_prim_enabled() and _is_bwd_prim_enabled()


# Alert!!! This method is only for test coverage, user should never use it directly, this may cause serious system errors.
def _test_use_sync(value):
    __sync_stat_with_flag(value)


# ops in forward_blacklist will not be replaced by composite ops.
prim_config = {
    "forward_blacklist": set(),
    "composite_ops_record": set(),
    "backward_blacklist": set(),
}


def _get_batch_norm_none_var(op):
    """Some outputs of batch_norm's replaced composite rule are not needed and will be removed."""
    use_run_stat = (
        op.attr("is_test") and (not op.attr("trainable_statistics"))
    ) or op.attr("use_global_stats")
    if use_run_stat:
        return ["ReserveSpace", "SavedMean", "SavedVariance"]
    else:
        return ["ReserveSpace"]


# In some case, inputs and outputs of composite op or its replaced composite rule might be None.
# It means such arg will be no longer required in processed program by composite mechanism.
# Therefore, such special ops should be recorded in advance and be released in args check.
ops_contain_none = {
    "batch_norm": _get_batch_norm_none_var,
    "flatten_contiguous_range": ["XShape"],
    "squeeze2": ["XShape"],
    "unsqueeze2": ["XShape"],
}


# some intermediate outputs like xshape will no longer used after decomp, but return none to keep output num the same as origin op
# key is the name of op, and value is the index of output in op.outputs
decomp_ops_contain_unused_output = {
    "pd_op.squeeze": [1],
    "pd_op.unsqueeze": [1],
    "pd_op.batch_norm": [5],
}


# This api is used for development for dynamic shape in prim, and will be removed in future.
def _enable_prim_skip_dynamic_shape():
    flag = os.getenv("FLAGS_prim_skip_dynamic", "1")
    if flag and flag.lower() in ("1", "true"):
        return True
    else:
        return False


def _enable_prim_dynamic_shape():
    flag = os.getenv("FLAGS_prim_enable_dynamic")
    if flag and flag.lower() in ("1", "true"):
        return True
    else:
        return False


def _enable_dist_prim_all():
    flag = os.getenv("FLAGS_dist_prim_all")
    if flag and flag.lower() in ("1", "true"):
        return True
    else:
        return False


def _enable_auto_recompute():
    flag = os.getenv("FLAGS_enable_auto_recompute")

    # NOTE(chenxi67): open recompute when cinn is enabled
    from paddle.base.framework import in_cinn_mode

    if in_cinn_mode():
        if flag and flag.lower() in ("0", "false"):
            return False
        else:
            return True
    if flag and flag.lower() in ("1", "true"):
        return True
    else:
        return False


def _set_prim_forward_blacklist(*args):
    for item in args:
        if not isinstance(item, str):
            raise TypeError("ops set in forward_blacklist must belong to str")
        else:
            prim_config["forward_blacklist"].add(item)


# Currently, this function is not utilized anywhere in the codebase.
# It may be intended for future use or could be removed if unnecessary.
# def _reset_prim_forward_blacklist():
#     prim_config["forward_blacklist"] = set()


def _set_prim_backward_blacklist(*args):
    ops = set(args)
    new_ops = set()
    for item in ops:
        if not isinstance(item, str):
            raise TypeError("All items in set must be strings.")
        if item.startswith("pd_op."):
            item = item[6:]
        prim_config["backward_blacklist"].add(item)
        new_ops.add(item)
    _set_bwd_prim_blacklist(new_ops)


def _set_prim_backward_enabled(value):
    __set_bwd_prim_enabled(bool(value))
    if _prim_return_log():
        print("backward prim enabled: ", bool(_is_bwd_prim_enabled()))


def _set_prim_forward_enabled(value):
    __set_fwd_prim_enabled(bool(value))
    if _prim_return_log():
        print("forward prim enabled: ", bool(_is_fwd_prim_enabled()))


def set_prim_eager_enabled(value):
    __set_eager_prim_enabled(bool(value))
    if _prim_return_log():
        print("eager prim enabled: ", bool(_is_eager_prim_enabled()))


def _set_prim_all_enabled(value):
    __set_all_prim_enabled(bool(value))
    if _prim_return_log():
        print(
            "all prim enabled: ",
            bool(_is_fwd_prim_enabled() and _is_bwd_prim_enabled()),
        )


def __sync_prim_backward_status():
    flag_value = os.getenv("FLAGS_prim_backward")
    if flag_value is None:
        if _prim_return_log():
            print("backward prim enabled: ", bool(_is_bwd_prim_enabled()))
    else:
        __sync_stat_with_flag("FLAGS_prim_backward")


def __sync_prim_forward_status():
    flag_value = os.getenv("FLAGS_prim_forward")
    if flag_value is None:
        if _prim_return_log():
            print("forward prim enabled: ", bool(_is_fwd_prim_enabled()))
    else:
        __sync_stat_with_flag("FLAGS_prim_forward")


def check_and_set_prim_all_enabled(print_flag=False):
    flag_value = os.getenv("FLAGS_prim_all")
    if flag_value is None:
        __sync_prim_backward_status()
        __sync_prim_forward_status()
    else:
        __sync_stat_with_flag("FLAGS_prim_all", print_flag)


check_and_set_prim_all_enabled(True)


SKIPPED_PRIM_VJP_DEFAULT_OPS = ["matmul_grad"]


def _clear_prim_vjp_skip_default_ops():
    for item in SKIPPED_PRIM_VJP_DEFAULT_OPS:
        _remove_skip_comp_ops(item)


# Since some decomposition of special ops like matmul_grad will reduce performance and is difficult to optimize currently by CINN.
# This api is used for development for in prim and cinn, and will be removed in future.
def _check_and_set_prim_vjp_skip_default_ops():
    flag = os.getenv("FLAGS_prim_vjp_skip_default_ops", "1")
    if flag and flag.lower() in ("1", "true"):
        _set_prim_backward_blacklist(*SKIPPED_PRIM_VJP_DEFAULT_OPS)
        return True
    else:
        _clear_prim_vjp_skip_default_ops()
        return False


_check_and_set_prim_vjp_skip_default_ops()


def _check_prim_vjp_ops():
    ops_org = os.getenv("FLAGS_prim_backward_blacklist", "")
    if ops_org:
        ops = []
        for item in ops_org.split(";"):
            ops.append(item.strip())
        _set_prim_backward_blacklist(*ops)


_check_prim_vjp_ops()
