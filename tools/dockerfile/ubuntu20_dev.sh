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
    sed "s#<baseimg>#ubuntu:20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>##g" ${dockerfile_name}
    sed -i "s#WITH_GPU:-ON#WITH_GPU:-OFF#g" ${dockerfile_name}
    sed -i "s#RUN apt-key del 7fa2af80##g" ${dockerfile_name}
    sed -i 's#RUN rm /etc/apt/sources.list.d/\*##g' ${dockerfile_name}
    sed -i "s#RUN apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub##g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>#RUN apt-get install -y gcc g++ make#g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh##g' ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_cudnn.sh cudnn841##g' ${dockerfile_name}
    sed -i 's#ENV CUDNN_VERSION=8.4.1##g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.2" ]];then
    dockerfile_name="Dockerfile-112"
    sed "s#<baseimg>#nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.2/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8034#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn821#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.2.1#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.6" ]];then
    dockerfile_name="Dockerfile-116"
    sed "s#<baseimg>#nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8406#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.7" ]];then
    dockerfile_name="Dockerfile-117"
    sed "s#<baseimg>#nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8424#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "11.8" ]];then
    dockerfile_name="Dockerfile-118"
    sed "s#<baseimg>#nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-8.2/bin:\$PATH #g" ${dockerfile_name}
    sed -i "s#gcc121#gcc82#g" ${dockerfile_name}
    sed -i "s#gcc-12.1#gcc-8.2#g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8531#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn860#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.6.0#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.0" ]];then
    dockerfile_name="Dockerfile-120"
    sed "s#<baseimg>#nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-12.0/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8616#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn891#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=8.9.1#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.3" ]];then
    dockerfile_name="Dockerfile-123"
    sed "s#<baseimg>#nvidia/cuda:12.3.1-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.3/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-12.0/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8616#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn900#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=9.0.0#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.4" ]];then
    dockerfile_name="Dockerfile-124"
    sed "s#<baseimg>#nvidia/cuda:12.4.1-devel-ubuntu20.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-12.0/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8616#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn911#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=9.1.1#g' ${dockerfile_name}
  elif [[ ${ref_CUDA_MAJOR} == "12.6" ]];then
    dockerfile_name="Dockerfile-126"
    sed "s#<baseimg>#nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04#g" ./Dockerfile.ubuntu20 >${dockerfile_name}
    sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
    sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
    sed -i "s#<install_gcc>#WORKDIR /usr/bin ENV PATH=/usr/local/gcc-12.0/bin:\$PATH #g" ${dockerfile_name}
    sed -i 's#RUN bash /build_scripts/install_trt.sh#RUN bash /build_scripts/install_trt.sh trt8616#g' ${dockerfile_name}
    sed -i 's#cudnn841#cudnn911#g' ${dockerfile_name}
    sed -i 's#CUDNN_VERSION=8.4.1#CUDNN_VERSION=9.1.1#g' ${dockerfile_name}
  else
    echo "Dockerfile ERROR!!!"
    exit 1
  fi

}


export ref_CUDA_MAJOR=cpu
base_image
export ref_CUDA_MAJOR=11.2
base_image
export ref_CUDA_MAJOR=11.6
base_image
export ref_CUDA_MAJOR=11.7
base_image
export ref_CUDA_MAJOR=11.8
base_image
export ref_CUDA_MAJOR=12.0
base_image
export ref_CUDA_MAJOR=12.3
base_image
export ref_CUDA_MAJOR=12.4
base_image
export ref_CUDA_MAJOR=12.6
base_image
