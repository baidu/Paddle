#!/bin/bash

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

function base_image(){
  if [[ ${ref_CUDA_MAJOR} == "11.8" ]];then
    dockerfile_name="Dockerfile-118"
    sed "s#<baseimg>#nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8616#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn897#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.6.0#g' ${dockerfile_name}
  else
    echo "Dockerfile ERROR!!!"
    exit 1
  fi

}

export ref_CUDA_MAJOR=11.8
base_image
