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

VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")

CUDNN_MAJOR=$(cat /usr/include/cudnn.h | grep -v CUDNN_VERSION | grep CUDNN_MAJOR | cut -d' ' -f3)
CUDNN_MINOR=$(cat /usr/include/cudnn.h | grep -v CUDNN_VERSION | grep CUDNN_MINOR | cut -d' ' -f3)
CUDNN_PATCHLEVEL=$(cat /usr/include/cudnn.h | grep -v CUDNN_VERSION | grep CUDNN_PATCHLEVEL | cut -d' ' -f3)
if [[ -z "${CUDNN_MAJOR}" ]]; then
  CUDNN_MAJOR=$(cat /usr/include/cudnn_version.h | grep -v CUDNN_VERSION | grep CUDNN_MAJOR | cut -d' ' -f3)
  CUDNN_MINOR=$(cat /usr/include/cudnn_version.h | grep -v CUDNN_VERSION | grep CUDNN_MINOR | cut -d' ' -f3)
  CUDNN_PATCHLEVEL=$(cat /usr/include/cudnn_version.h | grep -v CUDNN_VERSION | grep CUDNN_PATCHLEVEL | cut -d' ' -f3)
fi
CUDNN_VERSION="${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}"

if [[ "$1" == "trt8034" && "$VERSION" == "11.2" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz --no-check-certificate
  tar -zxf TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-8.0.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.0.3.4/lib/* /usr/lib/
  rm TensorRT-8.0.3.4.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz
elif [[ "$1" == "trt8424" ]];then
   wget https://paddle-qa.bj.bcebos.com/nvidia/trt/TensorRT-8.4.2.4.tgz --no-check-certificate
   tar -zxf TensorRT-8.4.2.4.tgz -C /usr/local
   cp -rf /usr/local/TensorRT-8.4.2.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.4.2.4/lib/* /usr/lib/
   rm -f TensorRT-8.4.2.4.tgz
elif [[ "$1" == "trt8616" && "$VERSION" == "11.8" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz --no-check-certificate
  tar -zxf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-8.6.1.6/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.6.1.6/lib/* /usr/lib/
  rm -f TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
elif [[ "$1" == "trt8616" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz --no-check-certificate
  tar -zxf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-8.6.1.6/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.6.1.6/lib/* /usr/lib/
  rm -f TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
elif [[ "$1" == "trt8406" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.6.cudnn8.3.tar.gz --no-check-certificate
  tar -zxf TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.6.cudnn8.3.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-8.4.0.6/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.4.0.6/lib/* /usr/lib/
  rm -f TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.6.cudnn8.3.tar.gz
elif [[ "$1" == "trt8431" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz --no-check-certificate --no-proxy
  tar -zxf TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-8.4.3.1/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.4.3.1/lib/* /usr/lib/
  rm -f TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
elif [[ "$1" == "trt8531" ]];then
  wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz --no-check-certificate --no-proxy
  tar -zxf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-8.5.3.1/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.5.3.1/lib/* /usr/lib/
  rm -f TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
elif [[ "$1" == "trt105018" ]];then
  wget -q https://paddle-qa.bj.bcebos.com/nvidia/trt/TensorRT-10.5.0.18-cuda12.tgz --no-check-certificate --no-proxy
  tar -zxf TensorRT-10.5.0.18-cuda12.tgz -C /usr/local
  cp -rf /usr/local/TensorRT-10.5.0.18/include/* /usr/include/ && cp -rf /usr/local/TensorRT-10.5.0.18/lib/* /usr/lib/
  rm -f TensorRT-10.5.0.18-cuda12.tgz
elif [[ "$VERSION" == "11.2" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT7-cuda11.1-cudnn8.1.tar.gz --no-check-certificate
  tar -zxf TensorRT7-cuda11.1-cudnn8.1.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.2.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.2.3.4/lib/* /usr/lib/
  rm TensorRT7-cuda11.1-cudnn8.1.tar.gz
elif [[ "$VERSION" == "11.1" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-7.2.3.4.CentOS-7.9.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz --no-check-certificate
  tar -zxf TensorRT-7.2.3.4.CentOS-7.9.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.2.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.2.3.4/lib/* /usr/lib/
  rm -f TensorRT-7.2.3.4.CentOS-7.9.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz
elif [[ "$VERSION" == "11.0" ]];then
  wget -q --no-proxy https://paddle-ci.cdn.bcebos.com/TRT/TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz --no-check-certificate
  tar -zxf TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.1.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.1.3.4/lib/* /usr/lib/
  rm TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz
elif [[ "$VERSION" == "10.2" && "$CUDNN_VERSION" == "7.6.5" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-6.0.1.8.CentOS-7.6.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz --no-check-certificate
  tar -zxf TensorRT-6.0.1.8.CentOS-7.6.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz -C /usr/local
  # trt6.0.1.8 should hack some code.
  sed -i "s/virtual int getConstantValue() const = 0;/&\nprotected:\nvirtual ~IDimensionExpr() {};/g" /usr/local/TensorRT-6.0.1.8/include/NvInferRuntime.h
  sed -i "s/virtual IPlugin\* createPlugin(const char\* layerName, const void\* serialData, size_t serialLength) TRTNOEXCEPT = 0;/&\nprotected:\nvirtual ~IPluginFactory() {}/g" /usr/local/TensorRT-6.0.1.8/include/NvInferRuntime.h
  cp -rf /usr/local/TensorRT-6.0.1.8/include/* /usr/include/ && cp -rf /usr/local/TensorRT-6.0.1.8/lib/* /usr/lib/
  rm -f TensorRT-6.0.1.8.CentOS-7.6.x86_64-gnu.cuda-10.2.cudnn7.6.tar.gz
elif [[ "$VERSION" == "10.2" && "$CUDNN_VERSION" == "8.1.1" ]];then
  wget -q --no-proxy https://paddle-ci.gz.bcebos.com/TRT/TensorRT-7.2.3.4.CentOS-7.9.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz --no-check-certificate
  tar -zxf TensorRT-7.2.3.4.CentOS-7.9.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.2.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.2.3.4/lib/* /usr/lib/
  rm TensorRT-7.2.3.4.CentOS-7.9.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz
elif [[ "$1" == "ubuntu1604-7234" && "$VERSION" == "10.2" ]];then
  wget -q --no-proxy https://sys-p0.bj.bcebos.com/paddle-qa/fullchain_ce_test/trt/TensorRT-7.2.3.4.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz --no-check-certificate
  tar -zxf TensorRT-7.2.3.4.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz -C /usr/local
  cp -rf /usr/local/TensorRT-7.2.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-7.2.3.4/lib/* /usr/lib/
  rm TensorRT-7.2.3.4.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.1.tar.gz
fi
