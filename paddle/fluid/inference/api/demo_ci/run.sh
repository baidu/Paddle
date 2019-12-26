#!/bin/bash
set -x
PADDLE_ROOT=$1
TURN_ON_MKL=$2 # use MKL or Openblas
TEST_GPU_CPU=$3 # test both GPU/CPU mode or only CPU mode
DATA_DIR=$4 # dataset
TENSORRT_INCLUDE_DIR=$5 # TensorRT header file dir, default to /usr/local/TensorRT/include
TENSORRT_LIB_DIR=$6 # TensorRT lib file dir, default to /usr/local/TensorRT/lib
inference_install_dir=${PADDLE_ROOT}/build/fluid_inference_install_dir

cd `dirname $0`
current_dir=`pwd`
if [ $2 == ON ]; then
  # You can export yourself if move the install path
  MKL_LIB=${inference_install_dir}/third_party/install/mklml/lib
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKL_LIB}
fi
if [ $3 == ON ]; then
  use_gpu_list='true false'
else
  use_gpu_list='false'
fi

with_static_lib_list="ON OFF"

USE_TENSORRT=OFF
if [ -d "$TENSORRT_INCLUDE_DIR" -a -d "$TENSORRT_LIB_DIR" ]; then
  USE_TENSORRT=ON
fi

PREFIX=inference-vis-demos%2F
URL_ROOT=http://paddlemodels.bj.bcebos.com/${PREFIX}

# download vis_demo data
function download() {
  dir_name=$1
  mkdir -p $dir_name
  cd $dir_name
  if [[ -e "${PREFIX}${dir_name}.tar.gz" ]]; then
    echo "${PREFIX}{dir_name}.tar.gz has been downloaded."
  else
      wget -q ${URL_ROOT}$dir_name.tar.gz
      tar xzf *.tar.gz
  fi
  cd ..
}
mkdir -p $DATA_DIR
cd $DATA_DIR
vis_demo_list='se_resnext50 ocr mobilenet'
for vis_demo_name in $vis_demo_list; do
  download $vis_demo_name
done

# compile and test the demo
cd $current_dir
mkdir -p build
cd build

function run() {
# TODO(Superjomn) reopen this
# something wrong with the TensorArray reset.
  # ---------vis_demo---------
  rm -rf *
  cmake .. -DPADDLE_LIB=${inference_install_dir} \
    -DWITH_MKL=$TURN_ON_MKL \
    -DDEMO_NAME=vis_demo \
    -DWITH_GPU=$TEST_GPU_CPU \
    -DWITH_STATIC_LIB=$WITH_STATIC_LIB
  make -j
  for use_gpu in $use_gpu_list; do
    for vis_demo_name in $vis_demo_list; do
      echo $WITH_STATIC_LIB $use_gpu $vis_demo_name $DATA_DIR
      ./vis_demo \
        --modeldir=$DATA_DIR/$vis_demo_name/model \
        --data=$DATA_DIR/$vis_demo_name/data.txt \
        --refer=$DATA_DIR/$vis_demo_name/result.txt \
        --use_gpu=$use_gpu
      if [ $? -ne 0 ]; then
        echo "vis demo $vis_demo_name runs fail."
        exit 1
      fi
      echo ok
    done
  done

  # --------tensorrt mobilenet------
  if [ $USE_TENSORRT == ON -a $TEST_GPU_CPU == ON ]; then
    rm -rf *
    cmake .. -DPADDLE_LIB=${inference_install_dir} \
      -DWITH_MKL=$TURN_ON_MKL \
      -DDEMO_NAME=trt_mobilenet_demo \
      -DWITH_GPU=$TEST_GPU_CPU \
      -DWITH_STATIC_LIB=$WITH_STATIC_LIB \
      -DUSE_TENSORRT=$USE_TENSORRT \
      -DTENSORRT_INCLUDE_DIR=$TENSORRT_INCLUDE_DIR \
      -DTENSORRT_LIB_DIR=$TENSORRT_LIB_DIR
    make -j
    ./trt_mobilenet_demo \
      --modeldir=$DATA_DIR/mobilenet/model \
      --data=$DATA_DIR/mobilenet/data.txt \
      --refer=$DATA_DIR/mobilenet/result.txt 
    if [ $? -ne 0 ]; then
      echo "trt demo trt_mobilenet_demo runs fail."
      exit 1
    fi
  fi
}
WITH_STATIC_LIB=OFF
run

WITH_STATIC_LIB=ON
run
set +x
