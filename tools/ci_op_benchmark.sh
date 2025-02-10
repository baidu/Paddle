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

set +ex

[ -z "$PADDLE_ROOT" ] && PADDLE_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)

# PR modify op source files
CHANGE_OP_FILES=()

# ops that will run benchmark test
declare -A CHANGE_OP_MAP

# ops that benchmark repo has
declare -A BENCHMARK_OP_MAP

# searched header files
declare -A INCLUDE_SEARCH_MAP

function LOG {
  echo "[$0:${BASH_LINENO[0]}] $*" >&2
}

function collect_kernel_registry_info {
  LOG "[INFO] Collect kernel registry info ..."
  python ${PADDLE_ROOT}/tools/parse_kernel_info.py
  [ $? -ne 0 ] && LOG "[FATAL] Collect kernel registry info fail."
}

# Limit cu file directory
function match_cu_file_directory {
  LOG "[INFO] run function match_cu_file_directory"
  local sub_dir cu_file_dir
  cu_file_dir=$(dirname ${1})
  # the operators under paddle/fluid/operators directory
  [ "${cu_file_dir}" == "paddle/fluid/operators" ] && return 0
  # the operators under paddle/phi/kernels directory
  for sub_dir in "" "/gpu" "/gpudnn" "/sparse/gpu"
  do
    [ "${cu_file_dir}" == "paddle/phi/kernels${sub_dir}" ] && return 0
  done
  return 1
}

# Limit h file directory
function match_h_file_directory {
  LOG "[INFO] run function match_h_file_directory"
  local sub_dir h_file_dir
  h_file_dir=$(dirname ${1})
  # '.h' file should not in directory below
  for sub_dir in "" "/cpu"
  do
    [ "${h_file_dir}" == "paddle/phi/kernels${sub_dir}" ] && return 1
  done
  return 0
}

# Load op files by header file
function load_CHANGE_OP_FILES_by_header_file {
  LOG "[INFO] run function load_CHANGE_OP_FILES_by_header_file"
  local change_file
  for change_file in $(grep -rl "${1}" paddle/fluid/operators paddle/phi/kernels/)
  do
    if [[ "$change_file" =~ "_op.cu" || "$change_file" =~ "_kernel.cu" ||  "$change_file" =~ "_kernel_gpudnn.cu" ]]
    then
      # match cu file directory limit
      match_cu_file_directory $change_file || continue
      LOG "[INFO] Found \"${1}\" include by \"${change_file}\"."
      CHANGE_OP_FILES[${#CHANGE_OP_FILES[@]}]="$change_file"
    elif [[ "$change_file" =~ ".h" ]]
    then
      match_h_file_directory $change_file || continue
      [ -n "${INCLUDE_SEARCH_MAP[$change_file]}" ] && continue
      LOG "[INFO] Found \"${1}\" include by \"${change_file}\", keep searching."
      INCLUDE_SEARCH_MAP[$change_file]="searched"
      load_CHANGE_OP_FILES_by_header_file $change_file
    fi
  done
}

# Load op files that PR changes
function load_CHANGE_OP_FILES {
  LOG "[INFO] run function load_CHANGE_OP_FILES"
  local sub_dir change_file
  # TODO(Avin0323): Need to filter the files added by the new OP.
  for change_file in $(git diff --name-only develop)
  do
    # match directory limit
    [[ "$change_file" =~ "paddle/fluid/operators/" ]] || [[ "$change_file" =~ "paddle/phi/kernels/" ]]  || continue
    # match file name limit
    if [[ "$change_file" =~ "_op.cu" || "$change_file" =~ "_kernel.cu" || "$change_file" =~ "_kernel_gpudnn.cu" ]]
    then
      # match cu file directory limit
      match_cu_file_directory $change_file || continue
      LOG "[INFO] Found \"${change_file}\" changed."
      CHANGE_OP_FILES[${#CHANGE_OP_FILES[@]}]="$change_file"
    elif [[ "$change_file" =~ ".h" ]]
    then
      match_h_file_directory $change_file || continue
      LOG "[INFO] Found \"${change_file}\" changed, keep searching."
      INCLUDE_SEARCH_MAP[${change_file}]="searched"
      load_CHANGE_OP_FILES_by_header_file $change_file
    fi
  done
  if [ ${#CHANGE_OP_FILES[@]} -eq 0 ]; then
    LOG "[INFO] Uninstall PaddlePaddle ..."
    pip uninstall -y paddlepaddle paddlepaddle_gpu
    LOG "[INFO] Install PaddlePaddle ..."
    pip install build/pr_whl/*.whl
    collect_kernel_registry_info
    LOG "[INFO] No op to test, skip this ci." && exit 0
  fi
}

# Clone benchmark repo
function clone_and_collect_op_info {
  LOG "[INFO] Clone benchmark repo ..."
  git clone https://github.com/PaddlePaddle/benchmark.git
  [ $? -ne 0 ] && LOG "[FATAL] Clone benchmark repo fail." && exit -1
  LOG "[INFO] Collect api info ..."
  python benchmark/api/deploy/collect_api_info.py \
      --test_module_name tests                    \
      --info_file api_info.txt >& 2
  [ $? -ne 0 ] && LOG "[FATAL] Collect api info fail." && exit -1
  [ ! -f benchmark/ci/scripts/op_benchmark.config ] && LOG "[FATAL] Missing op_benchmark.config!" && exit -1
}

# Load unique op name from CHANGE_OP_FILES
function load_CHANGE_OP_MAP {
  LOG "[INFO] run function load_CHANGE_OP_MAP"
  local op_name change_file change_file_name
  source benchmark/ci/scripts/op_benchmark.config
  for change_file in ${CHANGE_OP_FILES[@]}
  do
    change_file_name=${change_file#*paddle/fluid/operators/}
    change_file_name=${change_file_name#*paddle/phi/kernels/gpu/}
    change_file_name=${change_file_name#*paddle/phi/kernels/gpudnn/}
    if [ -n "${PADDLE_FILENAME_OP_MAP[$change_file_name]}" ]
    then
      for op_name in ${PADDLE_FILENAME_OP_MAP[$change_file_name]}
      do
        LOG "[INFO] Load op: \"${op_name}\"."
        CHANGE_OP_MAP[${op_name}]="$change_file"
      done
    else
      op_name=${change_file_name##*/}
      op_name=${op_name%_cudnn_op*}
      op_name=${op_name%_op*}
      op_name=${op_name%_grad_kernel*}
      op_name=${op_name%_kernel*}
      [ -n "${SKIP_OP_MAP[$op_name]}" ] && continue
      LOG "[INFO] Load op: \"${op_name}\"."
      CHANGE_OP_MAP[${op_name}]="$change_file"
    fi
  done
}

# Load ops that will run benchmark test
function load_BENCHMARK_OP_MAP {
  LOG "[INFO] run function load_BENCHMARK_OP_MAP"
  local line op_name api_name
  source benchmark/ci/scripts/op_benchmark.config
  for line in $(cat api_info.txt)
  do
    api_name=${line%%,*}
    if [ -n "${BENCHMARK_APINAME_OP_MAP[$api_name]}" ]
    then
      op_name=${BENCHMARK_APINAME_OP_MAP[$api_name]}
    else
      op_name=$api_name
    fi
    if [ -n "${CHANGE_OP_MAP[$op_name]}" ]
    then
      LOG "[INFO] Load benchmark settings with op \"${op_name}\"."
      BENCHMARK_OP_MAP[$op_name]=$line
    fi
  done
}


# run op benchmark test
function run_op_benchmark_test {
  LOG "[INFO] run function run_op_benchmark_test"
  [ ${#BENCHMARK_OP_MAP[*]} -eq 0 ] && return
  local logs_dir op_name branch_name api_info_file
  [ -z "$VISIBLE_DEVICES" ] && export VISIBLE_DEVICES=0
  [ "$BENCHMARK_PRINT_FAIL_LOG" != "1" ] && export BENCHMARK_PRINT_FAIL_LOG=1
  api_info_file="$(pwd)/api_info.txt"
  [ -f "$api_info_file" ] && rm -f $api_info_file
  for api_info in ${BENCHMARK_OP_MAP[*]}
  do
    echo "$api_info" >> $api_info_file
  done
  # install tensorflow for testing accuracy
  # pip install tensorflow==2.3.0 tensorflow-probability
  for branch_name in "dev_whl" "pr_whl"
  do
    LOG "[INFO] Uninstall PaddlePaddle ..."
    pip uninstall -y paddlepaddle paddlepaddle_gpu
    LOG "[INFO] Install PaddlePaddle ..."
    pip install build/${branch_name}/*.whl
    logs_dir="$(pwd)/logs-${branch_name}"
    [ -d $logs_dir ] && rm -rf $logs_dir/* || mkdir -p $logs_dir
    pushd benchmark/api > /dev/null
    bash deploy/main_control.sh tests \
                                tests_v2/configs \
                                $logs_dir \
                                $VISIBLE_DEVICES \
                                "gpu" \
                                "speed" \
                                $api_info_file \
                                "paddle"
    popd > /dev/null
  done
}

# check benchmark result
function check_op_benchmark_result {
  LOG "[INFO] run function check_op_benchmark_result"
  local logs_dir api_info_file check_status_code
  # default 3 times
  [ -z "${RETRY_TIMES}" ] && RETRY_TIMES=3
  logs_dir=$(pwd)/logs-test_pr
  api_info_file=$(pwd)/api_info.txt
  for retry_time in $(seq 0 ${RETRY_TIMES})
  do
    if [ $retry_time -gt 0 ]; then
      # run op benchmark speed test
      # there is no need to recompile and install paddle
      LOG "[INFO] retry ${retry_time} times ..."
      pushd benchmark/api > /dev/null
      bash deploy/main_control.sh tests \
                                  tests_v2/configs \
                                  ${logs_dir} \
                                  $VISIBLE_DEVICES \
                                  "gpu" \
                                  "speed" \
                                  ${api_info_file} \
                                  "paddle"
      popd > /dev/null
    fi
    # check current result and update the file to benchmark test
    python ${PADDLE_ROOT}/tools/check_op_benchmark_result.py \
        --develop_logs_dir $(pwd)/logs-dev_whl \
        --pr_logs_dir $(pwd)/logs-test_pr \
        --api_info_file ${api_info_file}
    check_status_code=$?
    # TODO(Avin0323): retry only if the performance check fails
    [ $check_status_code -eq 0 ] && break
  done
  return $check_status_code
}

function check_CHANGE_OP_MAP {
  LOG "[INFO] run function check_CHANGE_OP_MAP"
  for op_name in ${!CHANGE_OP_MAP[@]}
  do
    if [ -z "${BENCHMARK_OP_MAP[$op_name]}" ]
    then
      # Disable the check of missing op benchmark script temporarily.
      # exit_code=8
      LOG "[WARNING] Missing test script of \"${op_name}\"(${CHANGE_OP_MAP[$op_name]}) in benchmark."
    fi
  done
  if [ $exit_code -ne 0 ]; then
    LOG "[INFO] See https://github.com/PaddlePaddle/Paddle/wiki/PR-CI-OP-benchmark-Manual for details."
    LOG "[INFO] Or you can apply for one RD (zhangting2020(Recommend), Xreki) approval to pass this PR."
    exit ${exit_code}
  fi
}

# diff benchmark result and miss op
function summary_problems {
  LOG "[INFO]  run function summary_problems"
  local op_name exit_code
  exit_code=0
  if [ ${#BENCHMARK_OP_MAP[*]} -ne 0 ]
  then
    check_op_benchmark_result
    exit_code=$?
  fi
  check_CHANGE_OP_MAP
}


function prepare_env {
  LOG "[INFO] Start preparing op benchmark environment ..."
  load_CHANGE_OP_FILES
  clone_and_collect_op_info
  load_CHANGE_OP_MAP
  load_BENCHMARK_OP_MAP
  LOG "[INFO] Op benchmark environment is prepared success!"
}


function gpu_op_benchmark {
  LOG "[INFO] Start run op benchmark gpu test ..."
  run_op_benchmark_test
  summary_problems
  LOG "[INFO] Op benchmark run success and no error!"
  collect_kernel_registry_info
  exit 0
}


# The PR will pass quickly when get approval from specific person.
set +x
approval_line=$(curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000)
if [ -n "${approval_line}" ]; then
  APPROVALS=$(echo ${approval_line} | python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 Xreki zhangting2020)
  LOG "[INFO] current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
  if [ "${APPROVALS}" == "TRUE" ]; then
    LOG "[INFO] ==================================="
    LOG "[INFO] current pr ${GIT_PR_ID} has got approvals. So, Pass CI directly!"
    LOG "[INFO] ==================================="
    exit 0
  fi
fi

case $1 in
  run_op_benchmark)
    prepare_env
    gpu_op_benchmark
  ;;
esac
