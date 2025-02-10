


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

import subprocess
a = subprocess.run(["find", "./generated", "-name", "*.cu"], stdout=subprocess.PIPE)
a = a.stdout.decode("utf-8").split("\n")

generated_cu = []

for i in range(len(a)):
    print(a[i])
    if a[i] != "":
        generated_cu += [a[i]]

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def get_gencode_flags():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]


gencode_flags = get_gencode_flags()



setup(
    name="triton_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./matmul.cu",
            # "./fmha_triton.cu",
            # "./fmha2_triton.cu",
            # "./fmha3_triton.cu",
            # "./FcRelu_triton.cu",
            # "./Fc_triton.cu",
            # "./Fc_triton_2.cu",
        ] + generated_cu,
        extra_compile_args={
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
        },
        extra_link_args = ["-lcuda"]
    ),
)
