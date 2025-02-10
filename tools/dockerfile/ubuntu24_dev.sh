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
   if [[ ${ref_CUDA_MAJOR} == "cpu" ]];then
    dockerfile_name="Dockerfile-cpu"
    sed "s#<baseimg>#ubuntu:24.04#g" ./Dockerfile.ubuntu24 >${dockerfile_name}
    sed -i "s#<setcuda>##g" ${dockerfile_name}
    sed -i "s#WITH_GPU:-ON#WITH_GPU:-OFF#g" ${dockerfile_name}
    sed -i 's#RUN mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.bak##g' ${dockerfile_name}
    sed -i 's#RUN curl .*3bf863cc.pub | gpg --dearmor | tee /usr/share/keyrings/cuda-archive-keyring.gpg##g' ${dockerfile_name}
    sed -i 's#RUN mv /etc/apt/sources.list.d/cuda.list.bak /etc/apt/sources.list.d/cuda.list##g' ${dockerfile_name}
    sed -i 's#RUN sed -i .*signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg.*cuda.list##g' ${dockerfile_name}
    sed -i 's#<install_cpu_package>#RUN apt-get install -y gcc g++ make#g' ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh##g' ${dockerfile_name}
    sed -i 's#ENV CUDNN_VERSION=9.5.0##g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.6" ]];then
    dockerfile_name="Dockerfile-126"
    sed "s#<baseimg>#nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04#g" ./Dockerfile.ubuntu24 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt105018#g' ${dockerfile_name}
  else
    echo "Dockerfile ERROR!!!"
    exit 1
  fi

}

export ref_CUDA_MAJOR=cpu
base_image
export ref_CUDA_MAJOR=12.6
base_image 