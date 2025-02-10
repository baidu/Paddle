#!/bin/bash

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# Top-level build script called from Dockerfile

# Stop at any error, show all commands
set -ex

VERSION=$(nvcc --version | grep release | grep -oEi "release ([0-9]+)\.([0-9])"| sed "s/release //")

if [[ "$1" == "cudnn811" && "$VERSION" == "11.1" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-11.2-linux-x64-v8.1.1.33.tgz --no-check-certificate
  tar -xzf cudnn-11.2-linux-x64-v8.1.1.33.tgz && \
  cd cuda && \
  cp -r include /usr && \
  cp -r lib64 /usr && cd ../ && \
  rm -f cudnn-11.2-linux-x64-v8.1.1.33.tgz && \
  rm -rf cuda
elif [[ "$1" == "cudnn811" && "$VERSION" == "10.2" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-10.2-linux-x64-v8.1.1.33.tgz --no-check-certificate
  tar -xzf cudnn-10.2-linux-x64-v8.1.1.33.tgz && \
  cd cuda && \
  cp -r include /usr && \
  cp -r lib64 /usr && cd ../ && \
  rm -f cudnn-10.2-linux-x64-v8.1.1.33.tgz && \
  rm -rf cuda
elif [[ "$1" == "cudnn821" && "$VERSION" == "11.2" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-11.3-linux-x64-v8.2.1.32.tgz --no-check-certificate
  tar -xzf cudnn-11.3-linux-x64-v8.2.1.32.tgz && \
  cd cuda && \
  cp -r include /usr && \
  cp -r lib64 /usr && cd ../ && \
  rm -f cudnn-11.3-linux-x64-v8.2.1.32.tgz && \
  rm -rf cuda
elif [[ "$1" == "cudnn841" && "$VERSION" == "11.6" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz && \
  cd cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive && \
  cp -r include /usr && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive
elif [[ "$1" == "cudnn841" && "$VERSION" == "11.7" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz && \
  cd cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive && \
  cp -r include /usr && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive
elif [[ "$1" == "cudnn860" && "$VERSION" == "11.8" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
  cd cudnn-linux-x86_64-8.6.0.163_cuda11-archive
  cp -r include /usr
  cp -r lib /usr && cd ../
  rm -f cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
  rm -rf cudnn-linux-x86_64-8.6.0.163_cuda11-archive
elif [[ "$1" == "cudnn891" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz && \
  cd cudnn-linux-x86_64-8.9.1.23_cuda12-archive && \
  cp -r include /usr && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-8.9.1.23_cuda12-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-8.9.1.23_cuda12-archive
elif [[ "$1" == "cudnn896" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz && \
  cd cudnn-linux-x86_64-8.9.6.50_cuda12-archive && \
  cp -r include /usr && \
  cp -r lib/libcudnn* /usr/lib/x86_64-linux-gnu && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-8.9.6.50_cuda12-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-8.9.6.50_cuda12-archive
elif [[ "$1" == "cudnn897" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz && \
  cd cudnn-linux-x86_64-8.9.7.29_cuda11-archive && \
  cp -r include /usr && \
  cp -r lib/libcudnn* /usr/lib/x86_64-linux-gnu && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-8.9.7.29_cuda11-archive
elif [[ "$1" == "cudnn900" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz && \
  cd cudnn-linux-x86_64-9.0.0.312_cuda12-archive && \
  cp -r include /usr && \
  mkdir -p /usr/lib/x86_64-linux-gnu && \
  cp -r lib/libcudnn* /usr/lib/x86_64-linux-gnu && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-9.0.0.312_cuda12-archive
elif [[ "$1" == "cudnn911" ]]; then
  wget -q https://paddle-ci.gz.bcebos.com/cudnn/cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz --no-check-certificate
  tar xJvf cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz && \
  cd cudnn-linux-x86_64-9.1.1.17_cuda12-archive && \
  cp -r include /usr && \
  mkdir -p /usr/lib/x86_64-linux-gnu && \
  cp -r lib/libcudnn* /usr/lib/x86_64-linux-gnu && \
  cp -r lib /usr && cd ../ && \
  rm -f cudnn-linux-x86_64-9.1.1.17_cuda12-archive.tar.xz && \
  rm -rf cudnn-linux-x86_64-9.1.1.17_cuda12-archive
fi
