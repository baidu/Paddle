#!/bin/bash

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

set -xe

REPO="${REPO:-paddledocker}"

function make_cuda112cudnn821trt8034() {
  sed 's/<baseimg>/11.2.2-cudnn8-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN yum remove -y libcudnn8-devel.x86_64 libcudnn8.x86_64 \nRUN bash build_scripts/install_cudnn.sh cudnn821 \nENV CUDNN_VERSION=8.2.1 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8034#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda116cudnn840trt8406() {
  sed 's/<baseimg>/11.6.2-cudnn8-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8406#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda117cudnn841trt8424() {
  sed 's/<baseimg>/11.7.1-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/install_cudnn.sh cudnn841 \nENV CUDNN_VERSION=8.4.1 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8424#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda118cudnn860trt8531() {
  sed 's/<baseimg>/11.8.0-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc82 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-8.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-8.2/bin:\$PATH \nRUN bash build_scripts/install_cudnn.sh cudnn860 \nENV CUDNN_VERSION=8.6.0 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8531#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda120cudnn891trt8616() {
  sed 's/<baseimg>/12.0.1-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc122 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-12.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-12.2/bin:\$PATH \nRUN bash build_scripts/install_cudnn.sh cudnn891 \nENV CUDNN_VERSION=8.9.1 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8616#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda123cudnn900trt8616() {
  sed 's/<baseimg>/12.3.1-devel-centos7/g' Dockerfile.centos >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc122 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-12.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-12.2/bin:\$PATH \nRUN bash build_scripts/install_cudnn.sh cudnn900 \nENV CUDNN_VERSION=9.0.0 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8616#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda124cudnn911trt8616() {
  sed 's/<baseimg>/12.4.1-cudnn-devel-rockylinux8/g' Dockerfile.rockylinux8 >Dockerfile.tmp
  sed -i "s#<install_gcc>#RUN dnf install -y gcc-toolset-12-gcc* \&\& source /opt/rh/gcc-toolset-12/enable \&\& echo 'source /opt/rh/gcc-toolset-12/enable' >>~/.bashrc \nENV PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/share/Modules/bin:\$PATH \nENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/opt/rh/gcc-toolset-12/root/usr/lib:\$LD_LIBRARY_PATH #g" Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_cudnn.sh cudnn911 \nENV CUDNN_VERSION=9.1.1 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8616#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}

function make_cuda125cudnn911trt8616() {
  sed 's/<baseimg>/12.5.1-cudnn-devel-rockylinux8/g' Dockerfile.rockylinux8 >Dockerfile.tmp
  sed -i "s#RUN bash build_scripts/build.sh#RUN bash build_scripts/install_gcc.sh gcc122 \nRUN mv /usr/bin/cc /usr/bin/cc.bak \&\& ln -s /usr/local/gcc-12.2/bin/gcc /usr/bin/cc \nENV PATH=/usr/local/gcc-12.2/bin:\$PATH \nRUN bash build_scripts/install_cudnn.sh cudnn911 \nENV CUDNN_VERSION=9.1.1 \nRUN bash build_scripts/build.sh#g" Dockerfile.tmp
  sed -i "s#build_scripts/install_trt.sh#build_scripts/install_trt.sh trt8616#g" Dockerfile.tmp
  sed -i '/CMD/iRUN ldconfig' Dockerfile.tmp
}


function main() {
  local CMD=$1
  case $CMD in
    cuda112cudnn821trt8034)
      make_cuda112cudnn821trt8034
     ;;
    cuda116cudnn840trt8406)
      make_cuda116cudnn840trt8406
     ;;
    cuda117cudnn841trt8424)
      make_cuda117cudnn841trt8424
     ;;
    cuda118cudnn860trt8531)
      make_cuda118cudnn860trt8531
     ;;
    cuda120cudnn891trt8616)
      make_cuda120cudnn891trt8616
     ;;
    cuda123cudnn900trt8616)
     make_cuda123cudnn900trt8616
     ;;
    cuda124cudnn911trt8616)
     make_cuda124cudnn911trt8616
     ;;
    cuda125cudnn911trt8616)
     make_cuda125cudnn911trt8616
     ;;
    *)
      echo "Make dockerfile error, Without this parameter."
      exit 1
      ;;
  esac
}

main "$@"
