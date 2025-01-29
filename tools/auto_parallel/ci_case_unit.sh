#!/usr/bin/env bash
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

set -e

export log_path=/workspace/case_logs
export auto_case_path=/workspace/Paddle/test/auto_parallel/hybrid_strategy
export dygraph_case_path=/workspace/Paddle/test/collective/hybrid_strategy

function case_list_unit() {
    if [ ! -f "testslist.csv" ]; then
        echo "文件 testslist.csv 不存在"
        exit -1
    fi
    if [ ! -f "${log_path}/blacklist.csv" ]; then
        wget -P ${log_path}/ https://paddle-qa.bj.bcebos.com/Auto-Parallel/blacklist.csv --no-proxy || exit 101
        echo "\033 ---- wget blacklist.csv \033"
    fi
    blacklist_file=${log_path}/blacklist.csv
    mapfile -t blacklist < "$blacklist_file"

    target_key=${1:-"all"}
    for ((i=2; i<=`awk -F, 'END {print NR}' testslist.csv`; i++)); do
        item=`awk -F, 'NR=='$i' {print}' testslist.csv`
        case_name=`awk -F, 'NR=='$i' {print $1}' testslist.csv`
        if [[ ${target_key} != "all" ]] && [[ ! ${case_name} =~ ${target_key} ]]; then
            echo "=========== skip $case_name run  ==========="
            continue
        elif [[ " ${blacklist[@]} " == *" ${case_name} "* ]]; then
            echo "======= skip blacklist case: $case_name run  ======="
            continue
        else
            echo "=========== $case_name run  begin ==========="
        fi
        if [[ $item =~ PYTHONPATH=([^,;]*)([,;]|$) ]]; then
            substring="${BASH_REMATCH[1]}"
            echo "PYTHONPATH=$substring"
            export PYTHONPATH=$substring:$PYTHNPATH
        fi
        time_out=`awk -F, 'NR=='$i' {print $4}' testslist.csv`
        if [ -n "$(echo $time_out | sed -n "/^[0-9]\+$/p")" ];then
            echo "Given timeout is a valid number : $time_out"
        else
            echo "FAILED! Given timeout is not a valid number : $time_out"
            exit -1
        fi

        timeout $time_out python $case_name.py >>${log_path}/$case_name 2>&1
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            tail -n 10 ${log_path}/$case_name
        elif [ $exit_code -eq 124 ]; then
            echo "TIMEOUT!"
            exit -1
        else
            echo "FAIL with exit_code : $exit_code"
            exit -1
        fi
        echo "=========== $case_name run  end ==========="
    done
}

main() {
    export exec_case=$1
    echo -e "\033[31m ---- Start executing $exec_case case \033[0m"

    if [[ $exec_case == "auto_unit_test" ]];then
        cd ${auto_case_path}
        case_list_unit
    elif [[ $exec_case == "dygraph_unit_test" ]];then
        cd ${dygraph_case_path}
        case_list_unit
    elif [[ $exec_case == "llama_auto_unit_test" ]];then
        cd ${auto_case_path}
        case_list_unit llama
        case_list_unit pir_reshard
    else
        echo -e "\033[31m ---- Invalid exec_case $exec_case \033[0m"
    fi
}

main $@
