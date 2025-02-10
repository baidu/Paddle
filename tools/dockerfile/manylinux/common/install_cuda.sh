#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set -ex

NCCL_VERSION=v2.21.5-1
CUDNN_VERSION=9.5.1.17

function install_cusparselt_040 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.4.0.7-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.4.0.7-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.4.0.7-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.4.0.7-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_cusparselt_052 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.5.2.1-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.5.2.1-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.5.2.1-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.5.2.1-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_cusparselt_062 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.2.3-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.6.2.3-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.6.2.3-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.6.2.3-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_cusparselt_063 {
    # cuSparseLt license: https://docs.nvidia.com/cuda/cusparselt/license.html
    mkdir tmp_cusparselt && pushd tmp_cusparselt
    wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz
    tar xf libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz
    cp -a libcusparse_lt-linux-x86_64-0.6.3.2-archive/include/* /usr/local/cuda/include/
    cp -a libcusparse_lt-linux-x86_64-0.6.3.2-archive/lib/* /usr/local/cuda/lib64/
    popd
    rm -rf tmp_cusparselt
}

function install_nccl_2162 {
    wget -q https://nccl2-deb.cdn.bcebos.com/nccl_2.16.2-1+cuda11.8_x86_64.txz --no-check-certificate --no-proxy
    tar xf nccl_2.16.2-1+cuda11.8_x86_64.txz
    cp -a nccl_2.16.2-1+cuda11.8_x86_64/include/* /usr/include/
    cp -a nccl_2.16.2-1+cuda11.8_x86_64/lib/* /usr/lib64
    rm -rf nccl_2.16.2-1+cuda11.8_x86_64 nccl_2.16.2-1+cuda11.8_x86_64.txz
}

function install_nccl_2203 {
    wget -q https://nccl2-deb.cdn.bcebos.com/nccl_2.20.3-1+cuda12.3_x86_64.txz --no-check-certificate --no-proxy
    tar xf nccl_2.20.3-1+cuda12.3_x86_64.txz
    cp -a nccl_2.20.3-1+cuda12.3_x86_64/include/* /usr/include/
    cp -a nccl_2.20.3-1+cuda12.3_x86_64/lib/* /usr/lib64
    rm -rf nccl_2.20.3-1+cuda12.3_x86_64 nccl_2.20.3-1+cuda12.3_x86_64.txz
}

function install_nccl_2215 {
    wget -q https://nccl2-deb.cdn.bcebos.com/nccl_2.21.5-1+cuda12.4_x86_64.txz --no-check-certificate --no-proxy
    tar xf nccl_2.21.5-1+cuda12.4_x86_64.txz
    cp -a nccl_2.21.5-1+cuda12.4_x86_64/include/* /usr/include/
    cp -a nccl_2.21.5-1+cuda12.4_x86_64/lib/* /usr/lib64
    rm -rf nccl_2.21.5-1+cuda12.4_x86_64 nccl_2.21.5-1+cuda12.4_x86_64.txz
}

function install_nccl_2234 {
    wget -q https://nccl2-deb.cdn.bcebos.com/nccl_2.23.4-1+cuda12.6_x86_64.txz --no-check-certificate --no-proxy
    tar xf nccl_2.23.4-1+cuda12.6_x86_64.txz
    cp -a nccl_2.23.4-1+cuda12.6_x86_64/include/* /usr/include/
    cp -a nccl_2.23.4-1+cuda12.6_x86_64/lib/* /usr/lib64
    rm -rf nccl_2.23.4-1+cuda12.6_x86_64 nccl_2.23.4-1+cuda12.6_x86_64.txz
}

function install_trt_8616 {
    wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz --no-check-certificate --no-proxy
    tar -zxf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C /usr/local
    cp -rf /usr/local/TensorRT-8.6.1.6/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.6.1.6/lib/* /usr/lib/
    rm -f TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
}

function install_trt_105018 {
    wget -q https://paddle-ci.gz.bcebos.com/TRT/TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz --no-check-certificate --no-proxy
    tar -zxf TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz -C /usr/local
    cp -rf /usr/local/TensorRT-10.5.0.18/include/* /usr/include/ && cp -rf /usr/local/TensorRT-10.5.0.18/lib/* /usr/lib/
    rm -f TensorRT-10.5.0.18.Linux.x86_64-gnu.cuda-12.6.tar.gz
}

function install_118 {
    CUDNN_VERSION=8.9.7.29
    NCCL_VERSION=2.16.5
    TensorRT_VERSION=8.6.1.6
    echo "Installing CUDA 11.8 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and TensorRT ${TensorRT_VERSION} and cuSparseLt-0.4.0"
    rm -rf /usr/local/cuda-11.8 /usr/local/cuda
    # install CUDA 11.8.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    chmod +x cuda_11.8.0_520.61.05_linux.run
    ./cuda_11.8.0_520.61.05_linux.run --toolkit --driver --silent --no-drm --kernel-source-path=/usr/src/kernels/4.18.0-553.34.1.el8_10.x86_64
    rm -f cuda_11.8.0_520.61.05_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.8 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda11-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2162
    install_trt_8616
    install_cusparselt_040

    ldconfig
}

function install_123 {
    CUDNN_VERSION=9.1.1.17
    NCCL_VERSION=2.20.3
    TensorRT_VERSION=10.5
    echo "Installing CUDA 12.3 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and TensorRT ${TensorRT_VERSION} and cuSparseLt-0.5.2"
    rm -rf /usr/local/cuda-12.3 /usr/local/cuda
    # install CUDA 12.3.0 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
    chmod +x cuda_12.3.2_545.23.08_linux.run
    ./cuda_12.3.2_545.23.08_linux.run --toolkit --driver --silent --kernel-source-path=/usr/src/kernels/4.18.0-553.34.1.el8_10.x86_64
    rm -f cuda_12.3.2_545.23.08_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.3 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2203
    install_trt_105018
    install_cusparselt_052

    ldconfig
}

function install_124 {
    CUDNN_VERSION=9.1.1.17
    NCCL=2.21.5
    TensorRT_VERSION=10.5
    echo "Installing CUDA 12.4.1 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and TensorRT ${TensorRT_VERSION} and cuSparseLt-0.6.2"
    rm -rf /usr/local/cuda-12.4 /usr/local/cuda
    # install CUDA 12.4.1 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
    chmod +x cuda_12.4.1_550.54.15_linux.run
    ./cuda_12.4.1_550.54.15_linux.run --toolkit --driver --silent --kernel-source-path=/usr/src/kernels/4.18.0-553.34.1.el8_10.x86_64
    rm -f cuda_12.4.1_550.54.15_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.4 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2215
    install_trt_105018
    install_cusparselt_062

    ldconfig
}

function install_126 {
    CUDNN_VERSION=9.5.1.17
    NCCL_VERSION=2.23.4
    TensorRT_VERSION=10.5
    echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and TensorRT ${TensorRT_VERSION} and cuSparseLt-0.6.3"
    rm -rf /usr/local/cuda-12.6 /usr/local/cuda
    # install CUDA 12.6.3 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
    chmod +x cuda_12.6.3_560.35.05_linux.run
    ./cuda_12.6.3_560.35.05_linux.run --toolkit --driver --silent --kernel-source-path=/usr/src/kernels/4.18.0-553.34.1.el8_10.x86_64
    rm -f cuda_12.6.3_560.35.05_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.6 /usr/local/cuda
    rm -rf /usr/bin/nvidia-smi

    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz -O cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2234
    install_trt_105018
    install_cusparselt_063

    ldconfig
}

function prune_118 {
    echo "Pruning CUDA 11.8 and cuDNN"
    #####################################################################################
    # CUDA 11.8 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-11.8/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-11.8/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 11.8 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-11.8/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2022.3.0 $CUDA_BASE/nsight-systems-2022.4.2/
}

function prune_123 {
  echo "Pruning CUDA 12.3"
  #####################################################################################
  # CUDA 12.3 prune static libs
  #####################################################################################
    export NVPRUNE="/usr/local/cuda-12.3/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-12.3/lib64"

    export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
    export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 12.3 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-12.3/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2023.1.0 $CUDA_BASE/nsight-systems-2023.1.2/
}

function prune_124 {
  echo "Pruning CUDA 12.4"
  #####################################################################################
  # CUDA 12.4 prune static libs
  #####################################################################################
  export NVPRUNE="/usr/local/cuda-12.4/bin/nvprune"
  export CUDA_LIB_DIR="/usr/local/cuda-12.4/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
      export GENCODE=$OVERRIDE_GENCODE
  fi
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" ]]; then
      export GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.4 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.4/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2024.1.0 $CUDA_BASE/nsight-systems-2023.4.4/
}

function prune_126 {
  echo "Pruning CUDA 12.6"
  #####################################################################################
  # CUDA 12.6 prune static libs
  #####################################################################################
  export NVPRUNE="/usr/local/cuda-12.6/bin/nvprune"
  export CUDA_LIB_DIR="/usr/local/cuda-12.6/lib64"

  export GENCODE="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
  export GENCODE_CUDNN="-gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"

  if [[ -n "$OVERRIDE_GENCODE" ]]; then
      export GENCODE=$OVERRIDE_GENCODE
  fi
  if [[ -n "$OVERRIDE_GENCODE_CUDNN" ]]; then
      export GENCODE_CUDNN=$OVERRIDE_GENCODE_CUDNN
  fi

  # all CUDA libs except CuDNN and CuBLAS
  ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
                "echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

  # prune CuDNN and CuBLAS
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
  $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

  #####################################################################################
  # CUDA 12.6 prune visual tools
  #####################################################################################
  export CUDA_BASE="/usr/local/cuda-12.6/"
  rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2024.3.2 $CUDA_BASE/nsight-systems-2024.5.1/
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    11.8) install_118; prune_118
        ;;
    12.3) install_123; prune_123
        ;;
    12.4) install_124; prune_124
        ;;
    12.6) install_126; prune_126
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
