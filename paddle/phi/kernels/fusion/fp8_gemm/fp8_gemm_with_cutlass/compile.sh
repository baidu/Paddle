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
set -e
cd cutlass_kernels
cutlass_repo_directory="cutlass"
if [ ! -d "$cutlass_repo_directory" ]; then

    git clone --depth=1 https://github.com/NVIDIA/cutlass.git
fi

build_directory="build"
if [ ! -d "$build_directory" ]; then
    mkdir $build_directory
fi

# set environment variables
python_exe_path="/usr/bin/python"
cuda_root_path="/usr/local/cuda"
gpu_cc="89"

cd $build_directory
cmake .. -DPYTHON_EXECUTABLE=$python_exe_path -DCUDA_TOOLKIT_ROOT_DIR=$cuda_root_path -DCOMPUTE_CAPABILITY=$gpu_cc
make -j32
cd -
