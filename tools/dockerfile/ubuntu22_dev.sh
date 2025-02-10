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
  if [[ ${ref_CUDA_MAJOR} == "12.3" ]];then
    dockerfile_name="Dockerfile-123"
    sed "s#<baseimg>#nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04#g" ./Dockerfile.ubuntu22 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i 's#<install_cudnn>#RUN bash /build_scripts/install_cudnn.sh cudnn911#g' ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt105018#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.4" ]];then
    dockerfile_name="Dockerfile-124"
    sed "s#<baseimg>#nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04#g" ./Dockerfile.ubuntu22 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i 's#<install_cudnn>##g' ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt105018#g' ${dockerfile_name}
  else
    echo "Dockerfile ERROR!!!"
    exit 1
  fi

}

export ref_CUDA_MAJOR=12.3
base_image 
export ref_CUDA_MAJOR=12.4
base_image 