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
import re
import sys

import triton

import paddle


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def find_so_path(generated_dir, python_package_name):
    """
    find the specified so in generated_dir, if not found it will return None.
    """

    so_path = []
    for root, dirs, files in os.walk(generated_dir):
        for file in files:
            if file.endswith(python_package_name + ".so"):
                so_path.append(os.path.join(root, file))
    if len(so_path) == 0:
        return None
    else:
        assert len(so_path) == 1
        return so_path[0]


def multi_process_do(commands):
    THREADS = 40
    import multiprocessing

    process = []

    def one_process_work(commands, thread_id):
        i = thread_id
        while i < len(commands):
            re = os.system(commands[i])
            assert re == 0
            i += THREADS

    for i in range(THREADS):
        p = multiprocessing.Process(target=one_process_work, args=(commands, i))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()


def extract_triton_kernel(kernel, file_name):
    """
    Extract the triton kernel and write it to the specified file_name.

    Args:
        kernel: the triton kernel name.
        file_name: the file name you want to write.
    """

    import inspect
    import re
    import textwrap

    if type(kernel) == triton.runtime.jit.JITFunction:
        fn = kernel.fn
    elif type(kernel) == triton.runtime.autotuner.Autotuner:
        fn = kernel.fn.fn
    else:
        AssertionError("error occures")
    py_script = textwrap.dedent(inspect.getsource(fn))

    # @triton.jit must only appear once
    assert len(re.findall("@triton.jit", py_script)) == 1

    py_script = py_script[py_script.find("@triton.jit") :]
    py_script = "import triton\nimport triton.language as tl\n" + py_script

    py_script = py_script.replace("if bias_ptr is not None", "if bias_ptr")

    with open(file_name, "a+") as f:
        f.write(py_script)
        f.close()


template_install = """

import os
generated_cu = []
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".c") or file.endswith(".cu"):
            generated_cu.append(os.path.join(root, file))


import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{{0}},code=sm_{{0}}".format(cc)]


gencode_flags = get_gencode_flags()



setup(
    name="{python_package_name}",
    ext_modules=CUDAExtension(
        sources = generated_cu,
        extra_compile_args={{
            "cc": ["-lcuda"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        }},
        extra_link_args = ["-lcuda"]
    ),
)
"""


def get_value_hint(x):
    if x % 16 == 0:
        return "i32:16"
    elif x % 8 == 0:
        return "i32"
    elif x % 4 == 0:
        return "i32"
    elif x % 2 == 0:
        return "i32"
    elif x == 1:
        return "i32:1"
    else:
        return "i32"


def build_package(generated_dir, python_package_name):
    """
    Build the package, not install it.

    Args:
        generated_dir: the source cu file dir.
        python_package_name: the python package name.
    """
    setup_file_path = generated_dir + "/setup_cuda.py"
    python_path = sys.executable
    with open(setup_file_path, "w") as f:
        f.write(
            template_install.format(python_package_name=python_package_name)
        )
        f.close()
    install_command = f"cd {generated_dir} && {python_path} setup_cuda.py build"
    re = os.system(install_command)
    assert re == 0


def rename_c_to_cu(generated_dir):
    """
    Rename the .c files int generated_dir to .cu file, becuase the triton aot tool generate the .c files.
    """
    # rename the .c file to .cu
    for filename in os.listdir(generated_dir):
        if filename.endswith(".c"):
            old_path = os.path.join(generated_dir, filename)
            new_path = os.path.join(generated_dir, filename + "u")
            os.rename(old_path, new_path)


def get_pointer_hint(tensor):
    if tensor.dtype == paddle.float16:
        return "*fp16:16"
    elif tensor.dtype == paddle.uint8:
        return "*u8:16"
    elif tensor.dtype == paddle.int8:
        return "*i8:16"
    elif tensor.dtype == paddle.float32:
        return "*fp32:16"


paddle_custom_op_head_part = """ #include <vector>
#include "${op_name}_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_${op_name};

CUdeviceptr get_tensor_ptr(const paddle::Tensor& input){
  if (input.type() == paddle::DataType::FLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int>());
  } else if (input.type() == paddle::DataType::FLOAT32) {
    return (CUdeviceptr)(input.data<float>());
  } else if (input.type() == paddle::DataType::UINT8) {
    return (CUdeviceptr)(input.data<uint8_t>());
  } else if (input.type() == paddle::DataType::INT8) {
    return (CUdeviceptr)(input.data<int8_t>());
  } else {
    assert(false);
    return (CUdeviceptr)(nullptr);
  }
} """


tune_and_invoke_part = """
if (!map_problem_${op_name}.count(problem_size)) {
    std::cout << "we are tuning for ${op_name} which key is: ";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < ${op_name}_kernel_get_num_algos(); ++algo_id) {
        cudaEvent_t beg[REPEAT];
        cudaEvent_t end[REPEAT];
        float elapsed_times[REPEAT];

        auto status = CUDA_SUCCESS;

        for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
        int repeat_id = ii - WARMUP;

        if (repeat_id >= 0) {
            (cudaEventCreate(beg + repeat_id));
            (cudaEventCreate(end + repeat_id));
            (cudaEventRecord(beg[repeat_id]));
        }

        auto flush_l2_cache = paddle::full(
            {10 * 1024 * 1024}, 0, paddle::DataType::INT32, x.place());
        // std::cout << &flush_l2_cache  << std::endl;
        // this is used when out is need to be reset to zero, such as split-k gemm.
        ${reset_zero_when_tune};

        status = status = run_triton_kernel(algo_id);
        // assert(status == CUDA_SUCCESS);

        if (repeat_id >= 0) {
            (cudaEventRecord(end[repeat_id]));
            (cudaEventSynchronize(end[repeat_id]));
            (cudaEventElapsedTime(
                elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
        }
        }

        float avg_elapsed_time = 0.f;
        for (int ii = 0; ii < REPEAT; ++ii) {
        avg_elapsed_time += elapsed_times[ii];
        }
        if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
        min_time = avg_elapsed_time;
        select_id = algo_id;
        }
    }

    map_problem_${op_name}[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ${reset_zero_when_tune};
}

  auto status = CUDA_SUCCESS;
  if (map_problem_${op_name}.count(problem_size)) {
    int algo_id = map_problem_${op_name}[problem_size];
    status = run_triton_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }
"""





#####################################################

paddle_custom_op_head_part_with_two_kernels = """ #include <vector>
#include "${first_kernel_name}_kernel.h"
#include "${second_kernel_name}_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_${first_kernel_name};
std::map<std::vector<int>, int> map_problem_${second_kernel_name};

CUdeviceptr get_tensor_ptr(const paddle::Tensor& input){
  if (input.type() == paddle::DataType::FLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int>());
  } else if (input.type() == paddle::DataType::FLOAT32) {
    return (CUdeviceptr)(input.data<float>());
  } else if (input.type() == paddle::DataType::UINT8) {
    return (CUdeviceptr)(input.data<uint8_t>());
  } else if (input.type() == paddle::DataType::INT8) {
    return (CUdeviceptr)(input.data<int8_t>());
  } else {
    assert(false);
    return (CUdeviceptr)(nullptr);
  }
} """


tune_and_invoke_part_with_two_kernels = """
if (!map_problem_${first_kernel_name}.count(problem_size)) {
    std::cout << "we are tuning for ${first_kernel_name} which key is: ";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < ${first_kernel_name}_kernel_get_num_algos(); ++algo_id) {
        cudaEvent_t beg[REPEAT];
        cudaEvent_t end[REPEAT];
        float elapsed_times[REPEAT];

        auto status = CUDA_SUCCESS;

        for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
        int repeat_id = ii - WARMUP;

        if (repeat_id >= 0) {
            (cudaEventCreate(beg + repeat_id));
            (cudaEventCreate(end + repeat_id));
            (cudaEventRecord(beg[repeat_id]));
        }
        // 这个是不是不能放在模板当中????????????除非强制要求第一个变量量叫做x, 这个部分抽象成一个class是不是会更好?a
        auto flush_l2_cache = paddle::full(
            {10 * 1024 * 1024}, 0, paddle::DataType::INT32, x.place());
        // std::cout << &flush_l2_cache  << std::endl;
        // this is used when out is need to be reset to zero, such as split-k gemm.
        ${reset_zero_when_tune};

        status = status = run_triton_first_kernel(algo_id);
        // assert(status == CUDA_SUCCESS);

        if (repeat_id >= 0) {
            (cudaEventRecord(end[repeat_id]));
            (cudaEventSynchronize(end[repeat_id]));
            (cudaEventElapsedTime(
                elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
        }
        }

        float avg_elapsed_time = 0.f;
        for (int ii = 0; ii < REPEAT; ++ii) {
        avg_elapsed_time += elapsed_times[ii];
        }
        if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
        min_time = avg_elapsed_time;
        select_id = algo_id;
        }
    }

    map_problem_${first_kernel_name}[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ${reset_zero_when_tune};
}

  auto status = CUDA_SUCCESS;
  if (map_problem_${first_kernel_name}.count(problem_size)) {
    int algo_id = map_problem_${op_name}[problem_size];
    status = run_triton_first_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }






if (!map_problem_${second_kernel_name}.count(problem_size)) {
    std::cout << "we are tuning for ${second_kernel_name} which key is: ";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < ${second_kernel_name}_kernel_get_num_algos(); ++algo_id) {
        cudaEvent_t beg[REPEAT];
        cudaEvent_t end[REPEAT];
        float elapsed_times[REPEAT];

        auto status = CUDA_SUCCESS;

        for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
        int repeat_id = ii - WARMUP;

        if (repeat_id >= 0) {
            (cudaEventCreate(beg + repeat_id));
            (cudaEventCreate(end + repeat_id));
            (cudaEventRecord(beg[repeat_id]));
        }
        auto flush_l2_cache = paddle::full(
            {10 * 1024 * 1024}, 0, paddle::DataType::INT32, x.place());
        // std::cout << &flush_l2_cache  << std::endl;
        // this is used when out is need to be reset to zero, such as split-k gemm.
        ${reset_zero_when_tune};

        status = status = run_triton_second_kernel(algo_id);
        // assert(status == CUDA_SUCCESS);

        if (repeat_id >= 0) {
            (cudaEventRecord(end[repeat_id]));
            (cudaEventSynchronize(end[repeat_id]));
            (cudaEventElapsedTime(
                elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
        }
        }

        float avg_elapsed_time = 0.f;
        for (int ii = 0; ii < REPEAT; ++ii) {
        avg_elapsed_time += elapsed_times[ii];
        }
        if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
            min_time = avg_elapsed_time;
            select_id = algo_id;
        }
    }

    map_problem_${second_kernel_name}[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ${reset_zero_when_tune};
}

  auto status = CUDA_SUCCESS;
  if (map_problem_${second_kernel_name}.count(problem_size)) {
    int algo_id = map_problem_${second_kernel_name}[problem_size];
    status = run_triton_second_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }
"""