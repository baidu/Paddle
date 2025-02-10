#!/bin/bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

pass_count=0
fail_count=0
timeout_count=0
not_found_count=0
failed_tests=()
timeout_tests=()
not_found_tests=()
timeout_limit=300


GREEN="\e[32m"
RED="\e[31m"
BLUE="\e[34m"
RESET="\e[0m"

# 测试用例
c_tests=(
    # arange
    "test_arange"
    # bitwise_and, bitwise_or
    "test_bitwise_op"
    # cast
    "test_cast_op"
    # concat, concat_grad
    "test_concat_op"
    # cross_entropy_with_softmax, cross_entropy_with_softmax_grad
    "test_softmax_with_cross_entropy_op"
    # einsum
    "^test_einsum$"
    # embedding, embedding_grad
    "test_embedding_deterministic"
    # empty
    "test_empty_op"
    # equal, greater_than, not_equal
    "test_compare_op"
    # expand
    "^test_expand_v2_op$"
    # fill
    "test_fill"
    # flatten
    "test_flatten_contiguous_range_op"
    # full
    "test_full_op"
    # full_
    "^test_full_$"
    # full_like
    "test_full_like_op"
    # gather_nd, gather_nd_grad
    "test_gather_nd_op"
    # gaussian
    "test_gaussian_random_op"
    # maximum
    "test_maximum_op"
    # nonzero
    "test_nonzero_api"
    # pow, pow_grad
    "^test_pow$"
    # reshape
    "test_reshape_op"
    # slice, slice_grad
    "test_slice_op"
    # squared_l2_norm
    "test_squared_l2_norm_op"
    # squeeze
    "test_squeeze2_op"
    # stack
    "test_stack_op"
    # subtract
    "test_subtract_op"
    # transpose, transpose_grad
    "test_transpose_op"
    # tril
    "test_tril_triu"
    # uniform
    "test_uniform_random_op"
    # unsqueeze
    "test_unsqueeze2_op"
    # where
    "test_where_op"
)

py_tests=(
    # add, add_grad, any, cos, divide, elementwise_pow, matmul, matmul_grad
    # mean, mean_grad, multiply, multiply_grad, rsqrt, rsqrt_grad, scale, silu, silu_grad
    # sin, sqrt, sum
    "hygon_llama_ops.py"
    # flash_attn, flash_attn_grad
    "flash_attention_hip.py"
    )

run_ctest() {
    local test_name=$1
    echo -e "${BLUE}Running ctest: $test_name${RESET}"
    output=$(ctest -R "$test_name" --timeout $timeout_limit --output-on-failure 2>&1)
    local test_result=$?

    if echo "$output" | grep -q "No tests were found!!!"; then
        echo -e "${RED}Test $test_name NOT FOUND${RESET}"
        not_found_count=$((not_found_count + 1))
        not_found_tests+=("$test_name")
    elif [ $test_result -eq 0 ]; then
        echo -e "${GREEN}Test $test_name PASSED${RESET}"
        pass_count=$((pass_count + 1))
    elif [ $test_result -eq 124 ]; then
        echo -e "${RED}Test $test_name TIMEOUT${RESET}"
        timeout_count=$((timeout_count + 1))
        timeout_tests+=("$test_name")
    else
        echo -e "${RED}Test $test_name FAILED${RESET}"
        fail_count=$((fail_count + 1))
        failed_tests+=("$test_name")
    fi

    echo
}

run_pytest(){
    for test_name in "${py_tests[@]}"; do
        if [ -f "$test_name" ]; then
            echo -e "${BLUE}Running pytest: $test_name${RESET}"

            PYTHONPATH=.. timeout $timeout_limit python -m unittest "$test_name" > /dev/null 2>&1
            local test_result=$?

            if [ $test_result -eq 0 ]; then
                echo -e "${GREEN}Test $test_name PASSED${RESET}"
                pass_count=$((pass_count + 1))
            elif [ $test_result -eq 124 ]; then
                echo -e "${RED}Test $test_name TIMEOUT${RESET}"
                timeout_count=$((timeout_count + 1))
                timeout_tests+=("$test_name")
            else
                echo -e "${RED}Test $test_name FAILED${RESET}"
                fail_count=$((fail_count + 1))
                failed_tests+=("$test_name")
            fi
        else
            echo -e "${RED}Test $test_name NOT FOUND${RESET}"
            not_found_count=$((not_found_count + 1))
            not_found_tests+=("$test_name")
        fi
        echo
    done
}

run_pytest

cd ..
for test in "${c_tests[@]}"; do
    run_ctest "$test"
done

echo
echo "==============================="
echo "Test Summary"
echo "==============================="
echo "Passed: $pass_count"
echo "Failed: $fail_count"
echo "Timeout: $timeout_count"
echo "Not Found: $not_found_count"
echo "Total: $((pass_count + fail_count + timeout_count + not_found_count))"

if [ $not_found_count -gt 0 ]; then
    echo
    echo -e "${RED}Not Found Test Cases:${RESET}"
    for not_found_test in "${not_found_tests[@]}"; do
        echo -e "${RED} - $not_found_test${RESET}"
    done
fi

if [ $fail_count -gt 0 ]; then
    echo
    echo -e "${RED}Failed Test Cases:${RESET}"
    for failed_test in "${failed_tests[@]}"; do
        echo -e "${RED} - $failed_test${RESET}"
    done
fi

if [ $timeout_count -gt 0 ]; then
    echo
    echo -e "${RED}Timeout Test Cases (>$timeout_limit seconds):${RESET}"
    for timeout_test in "${timeout_tests[@]}"; do
        echo -e "${RED} - $timeout_test${RESET}"
    done
fi
