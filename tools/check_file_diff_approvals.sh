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

if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi


PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
# If you want to add monitoring file modifications, please perform the. github/CODEOWNERS operation
API_FILES=(
    "tools/print_signatures.py"
    "tools/sampcd_processor.py"
    "tools/check_pr_approval.py"
    "tools/checkout_api_compatible.py"
)

approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
git_files=`git diff --numstat upstream/$BRANCH| wc -l`
git_count=`git diff --numstat upstream/$BRANCH| awk '{sum+=$1}END{print sum}'`
failed_num=0
echo_list=()


function check_approval(){
    person_num=`echo $@|awk '{for (i=2;i<=NF;i++)print $i}'`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py $1 $person_num`
    if [[ "${APPROVALS}" == "FALSE" && "${echo_line}" != "" ]]; then
        add_failed "${failed_num}. ${echo_line}"
    fi
}


function add_failed(){
    failed_num=`expr $failed_num + 1`
    echo_list="${echo_list[@]}$1"
}

function run_tools_test() {
    CUR_PWD=$(pwd)
    cd ${PADDLE_ROOT}/tools
    python $1
    cd ${CUR_PWD}
}

changed_env_var_count=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/paddle | grep 'DEFINE_EXPORTED' | grep -v '@@' | wc -l`
if [[ $changed_env_var_count -gt 0 ]]; then
    echo_line="You must have one RD (phlrain or luotao1) approval for changing the FLAGS, which manages the environment variables.\n"
    check_approval 1 phlrain luotao1
fi

changed_deprecated_tests_count=$(expr $(git ls-tree -r --name-only HEAD ${PADDLE_ROOT}/test/deprecated | grep '^test' | wc -l) - $(git ls-tree -r --name-only upstream/$BRANCH ${PADDLE_ROOT}/test/deprecated | grep '^tes' | wc -l))
if [[ $changed_deprecated_tests_count -gt 0 ]]; then
    echo_line="You must have one RD (wanghuancoder (Recommend)) approval for add new test in test/deprecated directory.\n"
    check_approval 1 wanghuancoder
fi

if [[ $git_files -gt 19 || $git_count -gt 999 ]];then
    echo_line="You must have raindrops2sea or XiaoguangHu01 approval for change 20+ files or add than 1000+ lines of content.\n"
    check_approval 1 raindrops2sea XiaoguangHu01
fi

for API_FILE in ${API_FILES[*]}; do
  API_CHANGE=`git diff --name-only upstream/$BRANCH | grep -F "${API_FILE}" | grep -v "/CMakeLists.txt" || true`
  if [ "${API_CHANGE}" ] && [ "${GIT_PR_ID}" != "" ]; then
      # NOTE: per_page=10000 should be ok for all cases, a PR review > 10000 is not human readable.
      if [ "${API_FILE}" == "tools/sampcd_processor.py" ];then
          echo_line="test_sampcd_processor.py will be executed for changed sampcd_processor.py.\n"
          run_tools_test test_sampcd_processor.py
      elif [ "${API_FILE}" == "tools/print_signatures.py" ];then
          echo_line="test_print_signatures.py will be executed for changed print_signatures.py.\n"
          run_tools_test test_print_signatures.py
      elif [ "${API_FILE}" == "tools/checkout_pr_approval.py" ];then
          echo_line="test_checkout_pr_approval.py will be executed for changed checkout_pr_approval.py.\n"
          run_tools_test test_checkout_pr_approval.py
      elif [ "${API_FILE}" == "tools/checkout_api_compatible.py" ];then
          echo_line="test_checkout_api_compatible.py will be executed for changed checkout_api_compatible.py.\n"
          run_tools_test test_checkout_api_compatible.py
      fi
  fi
done

DEPS_PHI_IN_IR=`git diff --name-only upstream/$BRANCH | grep -E "paddle/pir/" | grep "CMakeList" |xargs -r git diff -U0 upstream/$BRANCH --| grep "^\+" | grep "phi" || true`
echo "DEPS_PHI_IN_IR:${DEPS_PHI_IN_IR}"
if [ "${DEPS_PHI_IN_IR}" ] && [ "${DEPS_PHI_IN_IR}" != "" ]; then
    echo_line="You must have one RD (phlrain, zhangbo9674) approval for the CMakeLists.txt with DEPS phi* in paddle/pir directory.\n"
    check_approval 1 phlrain zhangbo9674
fi
FILTER=`git diff --name-only upstream/develop | grep -v "tools/"`
HAS_CONST_CAST=`git diff -U0 upstream/$BRANCH $FILTER | grep '^\+' | grep -o -m 1 "const_cast" || true`
if [ ${HAS_CONST_CAST} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (XiaoguangHu01, zhiqiu, Xreki, zhangbo9674, zyfncg, phlrain) approval for the usage of const_cast.\n"
    check_approval 1 XiaoguangHu01 zhiqiu Xreki zhangbo9674 zyfncg phlrain
fi

HAS_PADDLE_GET=`git diff -U0 upstream/$BRANCH $FILTER |grep "^+" |grep -o -m 1 "paddle::get" || true`
if [ ${HAS_PADDLE_GET} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="paddle::get is not recommended for direct use, because it may throw an bad_variant_access exception without any stack information, so please use PADDLE_GET(_**)(dtype, value) series macros here. If these macros cannot meet your needs, please use try-catch to handle paddle::get and request luotao1 or zhangbo9674 or phlrain review and approve.\n"
    check_approval 1 luotao1 zhangbo9674 phlrain
fi

FILTER=`git diff --name-only upstream/develop | grep -v "tools/"`
HAS_LEGACY_KERNEL_REGISTRATION=`git diff -U0 upstream/$BRANCH $FILTER | grep '^\+' | grep -oE -m 1 "REGISTER_OP[A-Z_]{1,9}KERNEL[_FUNCTOR|_WITH_CUSTOM_TYPE|_EX]*" || true`
if [ ${HAS_LEGACY_KERNEL_REGISTRATION} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="In principle, adding an OpKernel needs to be in the phi/kernels directory. If you must add an OpKernel in the fluid/operators directory, please request one of the RD (zyfncg, YuanRisheng, phlrain) review and approve.\n"
    check_approval 1 zyfncg YuanRisheng phlrain
fi

PYTHON_FILE_ADDED_LINES=$(git diff -U0 upstream/$BRANCH -- 'python/*.py' |grep "^+")
IF_USE_SUBPROCESS=`echo $PYTHON_FILE_ADDED_LINES | grep -B5 --no-group-separator "subprocess\." || true`
if [[ ${IF_USE_SUBPROCESS} ]]; then
    echo_line="You must have one RD (wanghuancoder(Recommend), SigureMo) approval for using subprocess, which may cause security problem.\n"
    check_approval 1 wanghuancoder SigureMo
fi
IF_USE_EVAL=`echo $PYTHON_FILE_ADDED_LINES | grep -B5 --no-group-separator "[^\w\d_]eval([^()]*[a-zA-Z0-9_])" || true`
if [[ ${IF_USE_EVAL} ]]; then
    echo_line="You must have one RD (wanghuancoder(Recommend), SigureMo) approval for using eval, which may cause security problem.\n"
    check_approval 1 wanghuancoder SigureMo
fi

CPP_FILE_ADDED_LINES=$(git diff -U0 upstream/$BRANCH -- 'paddle/' |grep "^+")
IF_USE_FESETROUND=`echo $CPP_FILE_ADDED_LINES | grep -B5 --no-group-separator "fesetround" || true`
if [[ ${IF_USE_FESETROUND} ]]; then
    echo_line="You must have one RD (zyfncg(Recommend), SigureMo, phlrain) approval for using fesetround, which may affect all floating-point precision calculations in the same process.\n"
    check_approval 1 zyfncg SigureMo phlrain
fi

HAS_DEFINE_FLAG=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "DEFINE_int32" |grep -o -m 1 "DEFINE_bool" | grep -o -m 1 "DEFINE_string" || true`
if [ ${HAS_DEFINE_FLAG} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD zyfncg or zhangbo9674 or phlrain approval for the usage (either add or delete) of DEFINE_int32/DEFINE_bool/DEFINE_string flag.\n"
    check_approval 1 zyfncg zhangbo9674 phlrain
fi

NO_NPU_FILE=`git diff --name-only upstream/$BRANCH | grep -v "_npu.py"`
HAS_UNITTEST_SKIP=`git diff -U0 upstream/$BRANCH ${NO_NPU_FILE} | grep "^+[[:space:]]\{0,\}@unittest.skip" || true`
if [ "${HAS_UNITTEST_SKIP}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Unittest is not allowed to be disabled.\nYou must have one RD (kolinwei(Recommend), wanghuancoder, luotao1, QingshuChen) approval for the usage of @unittest.skip or @unittest.skipIf.\n${HAS_UNITTEST_SKIP}\n"
    check_approval 1 kolinwei wanghuancoder luotao1 QingshuChen
fi

HAS_MODIFIED_DEMO_CMAKE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/inference/api/demo_ci/CMakeLists.txt" || true`
if [ "${HAS_MODIFIED_DEMO_CMAKE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (yuanlehome (Recommend), vivienfanghuagood) approval for paddle/fluid/inference/api/demo_ci/CMakeLists.txt.\nwhich manages the compilation parameter of inference demo\n"
    check_approval 1 yuanlehome vivienfanghuagood
fi

HAS_MODIFIED_DECLARATIONS=`git diff -U0 upstream/$BRANCH |grep "^+" |grep "paddle/phi/kernels/declarations.h" || true`
if [ "${HAS_MODIFIED_DECLARATIONS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by zyfncg or zhangbo9674 or phlrain for paddle/phi/kernels/declarations.h using. Thanks!\n"
    check_approval 1 zyfncg zhangbo9674 phlrain
fi

HAS_USED_CCTESTOLD=`git diff -U0 upstream/$BRANCH |grep "cc_test_old" || true`
if [ "${HAS_USED_CCTESTOLD}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by phlrain or risemeup1 or zhangbo9674 for using cc_test_old. Thanks!\n"
    check_approval 1 phlrain risemeup1 zhangbo9674
fi

HAS_USED_CCTEST=`git diff -U0 upstream/$BRANCH |grep -w "cc_test" || true`
if [ "${HAS_USED_CCTEST}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Paddle utest will gradually discard cc_test\n  instead, the paddle_test is recommended,\n if you must use cc_test, you must be approved by risemeup1 or zhangbo9674 for using cc_test. Thanks!\n"
    check_approval 1 risemeup1 zhangbo9674
fi

HAS_CREATE_NEW_PASS=`git diff -U0 upstream/$BRANCH |grep "paddle/pir/include/pass/pass.h" || true`
if [ "${HAS_CREATE_NEW_PASS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="\n If you implement a new Pass, you must be approved by yuanlehome or zyfncg. Thanks!\n"
    check_approval 1 yuanlehome zyfncg
fi

HAS_MODIFIED_API_COMPAT_YAML=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/ops/yaml/op_compat.yaml" || true`
if [ "${HAS_MODIFIED_API_COMPAT_YAML}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by zyfncg or heavyrain-lzy for paddle/phi/ops/yaml/op_compat.yaml changes, which manages the extra params of Op and name mapping between Yaml and OpMaker. In order to ensure compatibility of framework, this file isn't allowed to be modified at will!\n"
    check_approval 1 zyfncg heavyrain-lzy
fi

HAS_MODIFIED_API_FW_BW_YAML=`git diff --name-only upstream/$BRANCH | grep -E "paddle/phi/ops/yaml/ops.yaml|paddle/phi/ops/yaml/backward.yaml" || true`
if [ "${HAS_MODIFIED_API_FW_BW_YAML}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by zyfncg or heavyrain-lzy for paddle/phi/ops/yaml/ops.yaml or paddle/phi/ops/yaml/backward.yaml changes, which manage the generated code for the C++ OP. You can only change them according to the specification at the beginning of this two file.\n Recommend you obtain approval from gongshaotian or Hongqing-work, if only modified the InferSymbolicShapeInterface interfaces in the YAML file.\n"
    check_approval 1 zyfncg heavyrain-lzy gongshaotian Hongqing-work
fi

HAS_MODIFIED_PRIMITIVE_YAML=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/primitive/primitive/primitive.yaml" || true`
if [ "${HAS_MODIFIED_PRIMITIVE_YAML}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must be approved by jeff41404(gaoxiang) or xiaoguoguo626807(wangruting) or cubehan3(hanqiukun) for paddle/fluid/primitive/primitive/primitive.yaml changes.\n"
    check_approval 1 jeff41404 xiaoguoguo626807 cubehan3
fi

HAS_MODIFIED_FRAMEWORK_EXECUTOR=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/framework/new_executor" || true`
if [ "${HAS_MODIFIED_FRAMEWORK_EXECUTOR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (From00, zhangbo9674) approval for file changes in paddle/fluid/framework/new_executor.\n"
    check_approval 1 From00 zhangbo9674
fi

HAS_MODIFIED_FRAMEWORK_EXECUTOR=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/pir/serialize_deserialize" || true`
if [ "${HAS_MODIFIED_FRAMEWORK_EXECUTOR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (xiaoguoguo626807 changeyoung98) approval for file changes in paddle/fluid/pir/serialize_deserialize.\n"
    check_approval 1 xiaoguoguo626807 changeyoung98
fi

HAS_MODIFIED_DRR_INCLUDE_DIR=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/pir/drr/include" || true`
if [ "${HAS_MODIFIED_DRR_INCLUDE_DIR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (yuanlehome, zyfncg) approval for file changes in paddle/fluid/pir/drr/include.\n"
    check_approval 1 yuanlehome zyfncg
fi


HAS_MODIFIED_PIR_INCLUDE_DIR=`git diff --name-only upstream/$BRANCH | grep "paddle/pir/include" || true`
if [ "${HAS_MODIFIED_PIR_INCLUDE_DIR}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (yuanlehome, zhangbo9674) approval for file changes in paddle/pir/include.\n"
    check_approval 1 yuanlehome zhangbo9674
fi

HAS_MODIFIED_API_GENE=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/api/generator" || true`
if [ "${HAS_MODIFIED_API_GENE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zyfncg, YuanRisheng, phlrain, heavyrain-lzy) approval for file changes in paddle/phi/api/generator, which manages the generated code for C++ API in paddle/phi/api/lib/api.cc.\n"
    check_approval 1 zyfncg YuanRisheng phlrain heavyrain-lzy
fi

HAS_MODIFIED_EAGER_GENE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/eager/auto_code_generator" || true`
if [ "${HAS_MODIFIED_EAGER_GENE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (JiabinYang, zyfncg, phlrain, heavyrain-lzy) approval for file changes in paddle/fluid/eager/auto_code_generator, which manages the generated code for dygraph functions in paddle/fluid/eager/api/generated.\n"
    check_approval 1 JiabinYang zyfncg phlrain heavyrain-lzy
fi

HAS_MODIFIED_OPERATOR_GENE=`git diff --name-only upstream/$BRANCH | grep "paddle/fluid/operators/generator" || true`
if [ "${HAS_MODIFIED_OPERATOR_GENE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zyfncg, iclementine, phlrain, heavyrain-lzy) approval for file changes in paddle/fluid/operators/generator, which manages the generated code for OpMaker in paddle/fluid/operators/(generated_op.cc | sparse_generated_op.cc)\n"
    check_approval 1 zyfncg iclementine phlrain heavyrain-lzy
fi

HAS_MODIFIED_SETUP_IN=`git diff --name-only upstream/$BRANCH | grep "python/setup.py.in" || true`
if [ "${HAS_MODIFIED_SETUP_IN}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zyfncg, YuanRisheng, phlrain) approval for file changes in python/setup.py.in, which manages the header files that can be used from outside of framework.\n"
    check_approval 1 zyfncg YuanRisheng phlrain
fi

HAS_MODIFIED_SETUP=`git diff --name-only upstream/$BRANCH | grep "${PADDLE_ROOT}/setup.py" || true`
if [ "${HAS_MODIFIED_SETUP}" != "" ] || ([ "${HAS_MODIFIED_SETUP_IN}" != "" ] && [ "${HAS_MODIFIED_SETUP}" == "" ]); then
    echo_line="You must have one RD (risemeup1, zhangbo9674) approval for file changes in setup.py or setup.py and python/setup.py.in are not changed synchronously.\n"
    check_approval 1 risemeup1 zhangbo9674
fi

HAS_MODIFIED_STATIC_BUILD=`git diff --name-only upstream/$BRANCH | grep "new_executor/interpreter/static_build.cc" || true`
if [ "${HAS_MODIFIED_STATIC_BUILD}" != "" ] && [ "${GIT_PR_ID}" != ""]; then
    echo_line="You must have one RD (From00 or zhiqiu) approval for file changes in new_executor/interpreter/static_build.cc.\n"
    check_approval 1 From00 zhiqiu
fi

HAS_MODIFIED_TARGET_FOR_AUTO_PARALLEL_CI=`git diff --name-only upstream/$BRANCH | grep "tools/auto_parallel/target_path_lists.sh" || true`
if [ "${HAS_MODIFIED_TARGET_FOR_AUTO_PARALLEL_CI}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zhiqiu approval for file changes in tools/auto_parallel/target_path_lists.sh.\n"
    check_approval 1 zhiqiu
fi

HAS_MODIFIED_PY_FLUID=`git diff --name-only upstream/$BRANCH | grep "python/paddle/fluid" || true`
if [ "${HAS_MODIFIED_PY_FLUID}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zoooo0820(Recommend), or jeff41404) approval for file changes in python/paddle/fluid, because fluid API has been removed.\n"
    check_approval 1 zoooo0820 jeff41404
fi

HAS_MODIFIED_PADDLE_DISTRIBUTED=`git diff --name-only upstream/$BRANCH | grep "distributed/collective" || true`
if [ "${HAS_MODIFIED_PADDLE_DISTRIBUTED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (zhiqiu, ForFishes, gongweibao or sneaxiy) approval for file changes in distributed/collective. Thanks!\n"
    check_approval 1 zhiqiu ForFishes gongweibao sneaxiy
fi

ALL_PADDLE_ENFORCE=`git diff -U0 upstream/$BRANCH |grep "^+" |grep -zoE "PADDLE_ENFORCE\(.[^,\);]+.[^;]*\);\s" || true`
if [ "${ALL_PADDLE_ENFORCE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="PADDLE_ENFORCE is not recommended. Please use PADDLE_ENFORCE_EQ/NE/GT/GE/LT/LE or PADDLE_ENFORCE_NOT_NULL or PADDLE_ENFORCE_GPU_SUCCESS instead, see [ https://github.com/PaddlePaddle/Paddle/wiki/PADDLE_ENFORCE-Rewriting-Specification ] for details.\nYou must have one RD (luotao1 (Recommend) or zhangbo9674 or phlrain) approval for the usage (either add or delete) of PADDLE_ENFORCE.\n${ALL_PADDLE_ENFORCE}\n"
    check_approval 1 luotao1 zhangbo9674 phlrain
fi

ALL_ADDED_LINES=$(git diff -U0 upstream/$BRANCH |grep "^+" || true)
ALL_PADDLE_CHECK=$(echo $ALL_ADDED_LINES |grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" || true)
VALID_PADDLE_CHECK=$(echo "$ALL_PADDLE_CHECK" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(([^,;]+,)*[^";]*errors::.[^"]*".[^";]{20,}.[^;]*\);\s' || true)
INVALID_PADDLE_CHECK=$(echo "$ALL_PADDLE_CHECK" |grep -vxF "$VALID_PADDLE_CHECK" || true)
if [ "${INVALID_PADDLE_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The error message you wrote in PADDLE_ENFORCE{_**} or PADDLE_THROW does not meet our error message writing specification. Possible errors include 1. the error message is empty / 2. the error message is too short / 3. the error type is not specified. Please read the specification [ https://github.com/PaddlePaddle/Paddle/wiki/Paddle-Error-Message-Writing-Specification ], then refine the error message. If it is a mismatch, please request zhangbo9674 or phlrain or luotao1 review and approve.\nThe PADDLE_ENFORCE{_**} or PADDLE_THROW entries that do not meet the specification are as follows:\n${INVALID_PADDLE_CHECK}\n"
    check_approval 1 luotao1 zhangbo9674 phlrain
fi

EMPTY_GRAD_OP_REGISTERED=`echo $ALL_ADDED_LINES |grep -zoE "REGISTER_OP_WITHOUT_GRADIENT\([^;.]*\)[;\s]" || echo $ALL_ADDED_LINES |grep -zoE "[[:graph:]]*EmptyGradOpMaker<[[:graph:]]*>" || true`
if [ "${EMPTY_GRAD_OP_REGISTERED}" != "" ] && [ "${GIT_PT_ID}" != "" ]; then
    echo_line="You must have one RD (phlrain, XiaoguangHu01, kolinwei or JiabinYang) approval for the usage of REGISTER_OP_WITHOUT_GRADIENT or EmptyGradOpMaker.\nThe code that do not meet the specification are as follows:\n${EMPTY_GRAD_OP_REGISTERED}\n"
    check_approval 1 phlrain XiaoguangHu01 kolinwei JiabinYang
fi

INVALID_UNITTEST_ASSERT_CHECK=`echo "$ALL_ADDED_LINES" | grep -zoE '\+\s+((assert\s+)|(self\.assert(True|Equal)\())(\s*\+\s*)?(np|numpy)\.(allclose|array_equal)[^+]*' || true`
if [ "${INVALID_UNITTEST_ASSERT_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="It is recommended to use 'np.testing.assert_allclose' and 'np.testing.assert_array_equal' instead of 'self.assertTrue(np.allclose(...))' and 'self.assertTrue(np.array_equal(...))'.\nPlease modify the code below. If anything is unclear, please read the specification [ https://github.com/PaddlePaddle/community/blob/master/rfcs/CodeStyle/20220805_code_style_improvement_for_unittest.md#background ]. If it is a mismatch, please request SigureMo (Recommend) or luotao1 review and approve.\nThe code that do not meet the specification are as follows:\n${INVALID_UNITTEST_ASSERT_CHECK}\n"
    check_approval 1 SigureMo luotao1
fi

TEST_FILE_ADDED_LINES=$(git diff -U0 upstream/$BRANCH -- test |grep "^+")
ENABLE_TO_STATIC_CHECK=`echo "$TEST_FILE_ADDED_LINES" | grep "enable_to_static(" || true`
if [ "${ENABLE_TO_STATIC_CHECK}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (SigureMo, DrRyanHuang, zrr1999 or gouzil) approval for using 'paddle.jit.enable_to_static', we recommend using 'enable_to_static_guard' in the related test files.\n"
    check_approval 1 SigureMo DrRyanHuang zrr1999 gouzil
fi

HAS_MODIFIED_DY2ST_TEST_FILES=$(git diff --name-only --diff-filter=ACMR upstream/$BRANCH | grep "test/dygraph_to_static/test_" || true)
if [ "${HAS_MODIFIED_DY2ST_TEST_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    error_lines=`python ${PADDLE_ROOT}/test/dygraph_to_static/check_approval.py ${HAS_MODIFIED_DY2ST_TEST_FILES}`
    if [ $? -ne 0 ]; then
        echo_line="Your PR does not meet Dy2St unittest dev guide, please check https://github.com/PaddlePaddle/Paddle/issues/61464 for details.\n"
        echo_line=${echo_line}"Errors are as follows:\n"
        echo_line=${echo_line}${error_lines}"\n"
        echo_line=${echo_line}"You can run following command to fix the errors:\n"
        echo_line=${echo_line}"    python test/dygraph_to_static/check_approval.py "$(echo ${HAS_MODIFIED_DY2ST_TEST_FILES} | tr "\n" " ")"\n"
        echo_line=${echo_line}"If you believe this is a false positive, please request one of the RD (SigureMo, DrRyanHuang, zrr1999 or gouzil) approval for the changes.\n"
        check_approval 1 SigureMo DrRyanHuang zrr1999 gouzil
    fi
fi

HAS_MODIFIED_DY2ST_TEST_TENSOR_ATTR_CONSISTENCY=$(git diff --name-only upstream/$BRANCH | grep "test/dygraph_to_static/test_tensor_attr_consistency.py" || true)
if [ "${HAS_MODIFIED_DY2ST_TEST_TENSOR_ATTR_CONSISTENCY}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (SigureMo, DrRyanHuang, zrr1999 or gouzil) approval for file changes in test/dygraph_to_static/test_tensor_attr_consistency.py.\n"
    check_approval 1 SigureMo DrRyanHuang zrr1999 gouzil
fi

HAS_USED_AUTO_PARALLEL_ALIGN_MODE=`git diff -U0 upstream/$BRANCH |grep -o -m 1 "auto_parallel_align_mode" || true`
if [ ${HAS_USED_AUTO_PARALLEL_ALIGN_MODE} ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (sneaxiy, zhiqiu, ForFishes, or From00) approval for the usage of auto-parallel align mode.\n"
    check_approval 1 sneaxiy zhiqiu ForFishes From00
fi

HAS_MODIFIED_PHI_FILES=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/" || true`
PHI_INCLUDE_FLUID_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_FILES}; do
    PHI_DIR_ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep "^+" | grep "#include \"paddle/fluid/" || true`
    if [ "${PHI_DIR_ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        PHI_INCLUDE_FLUID_FILES="${PHI_INCLUDE_FLUID_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${PHI_INCLUDE_FLUID_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (YuanRisheng or zyfncg) approval for the including paddle/fluid header in paddle/phi files(${PHI_INCLUDE_FLUID_FILES}).\n"
    check_approval 1 YuanRisheng zyfncg
fi

HAS_MODIFIED_PHI_HEADER_FILES=`git diff --name-only upstream/$BRANCH | grep "paddle/phi/.*\.h" || true`
PHI_INCLUDE_THIRD_PARTY_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_HEADER_FILES}; do
    PHI_DIR_ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep "^+" | grep -E "#include \"gflags/gflags.h\"|#include \"glog/logging.h\"" || true`
    if [ "${PHI_DIR_ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        PHI_INCLUDE_THIRD_PARTY_FILES="${PHI_INCLUDE_THIRD_PARTY_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${PHI_INCLUDE_THIRD_PARTY_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (jiahy0825, zyfncg, YuanRisheng or heavyrain-lzy) approval for including \"gflags/gflags.h\" or \"glog/logging.h\" headerfile in paddle/phi headerfiles(${PHI_INCLUDE_THIRD_PARTY_FILES}). Recommend including third party headers in phi source files(*.cc) instead of phi headerfiles(*.h). Because if phi headerfiles include third party headers like \"gflags.h\" or \"logging.h\", error might occur when outside developers use phi headerfiles directly.\n"
    check_approval 1 jiahy0825 zyfncg YuanRisheng heavyrain-lzy
fi

HAS_MODIFIED_PADDLE_API_FILES=`git diff --name-only upstream/$BRANCH | grep "paddle/.*\.h" || true`
INCLUDE_PADDLE_API_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_HEADER_FILES}; do
    PADDLE_API_ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep -w "PADDLE_API" || true`
    if [ "${PADDLE_API_ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        INCLUDE_PADDLE_API_FILES="${INCLUDE_PADDLE_API_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${INCLUDE_PADDLE_API_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You must have one RD (jiahy0825, zyfncg) or PM (sunzhongkai588, Ligoml) approval for code changes about PADDLE_API. If you add a new PADDLE_API, please make sure you have written detailed comments about the parameter and usage of this PADDLE_API .\n"
    check_approval 1 jiahy0825 zyfncg sunzhongkai588 Ligoml
fi

HAS_MODIFIED_PHI_OR_FLUID_FILES=`git diff --name-only upstream/$BRANCH | grep -E "paddle/phi|paddle/fluid" || true`
USE_MUTABLE_DATA_FILES=""
for CHANGE_FILE in ${HAS_MODIFIED_PHI_OR_FLUID_FILES}; do
    ADDED_LINES=`git diff -U0 upstream/$BRANCH -- ${PADDLE_ROOT}/${CHANGE_FILE} | grep "^+" | grep -w "mutable_data" || true`
    if [ "${ADDED_LINES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
        USE_MUTABLE_DATA_FILES="${USE_MUTABLE_DATA_FILES} ${CHANGE_FILE}"
    fi
done
if [ "${USE_MUTABLE_DATA_FILES}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="You can not use the DenseTensor::mutable_data() method in files(${USE_MUTABLE_DATA_FILES}). If you want to alloc memory, use phi::DeviceContext::Alloc() or phi::DeviceContext::HostAlloc() instead and if you want to get mutable data, use DenseTensor::data(). If you have any questions, you can have one RD (YuanRisheng, zyfncg or From00) review and approve.\n"
    check_approval 1 YuanRisheng zyfncg From00
fi

ALL_CHANGE_FILES=`git diff --numstat upstream/$BRANCH | awk '{print $3}' | grep ".py"`
ALL_OPTEST_BAN_DYGRAPH_MESSAGE=""
for CHANGE_FILE in ${ALL_CHANGE_FILES}; do
    ALL_OPTEST_BAN_DYGRAPH=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CHANGE_FILE} | grep "+" | grep "check_dygraph=" || true`
    if [ "${ALL_OPTEST_BAN_DYGRAPH}" != "" ]; then
        ALL_OPTEST_BAN_DYGRAPH_MESSAGE="${ALL_OPTEST_BAN_DYGRAPH_MESSAGE} ${CHANGE_FILE} : \n${ALL_OPTEST_BAN_DYGRAPH} \n"
    fi
done
if [ "${ALL_OPTEST_BAN_DYGRAPH_MESSAGE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="Developers are not allowed to set the check_dygraph field directly, which is set to True by default. If you need to change the check_dygraph field, you must have one RD (phlrain (Recommend), QingshuChen (Recommend for kunlun) review and approve. \nThe code that do not meet the specification are as follows:\n${ALL_OPTEST_BAN_DYGRAPH_MESSAGE}\n"
    check_approval 1 phlrain QingshuChen
fi

ALL_CHANGE_YAML_FILES=`git diff --numstat upstream/$BRANCH | awk '{print $3}' | grep ".yaml"`
BAN_COMP_MESSAGE=""
for CHANGE_FILE in ${ALL_CHANGE_YAML_FILES}; do
    ALL_ITEM_BAN_COMP=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CHANGE_FILE} | grep "composite" || true`
    if [ "${ALL_ITEM_BAN_COMP}" != "" ]; then
        BAN_COMP_MESSAGE="${BAN_COMP_MESSAGE} ${CHANGE_FILE} : \n${ALL_ITEM_BAN_COMP} \n"
    fi
done
if [ "${BAN_COMP_MESSAGE}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="If you need to change the key composite, you must have one RD (xiaoguoguo626807(wangruting), cubehan3(hanqiukun)) review and approve. \nThe code that do not meet the specification are as follows:\n${BAN_COMP_MESSAGE}\n"
    check_approval 1 xiaoguoguo626807 cubehan3
fi

NEW_OP_ADDED=`git diff --name-only --diff-filter=A upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${NEW_OP_ADDED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    GET_KERNEL_TYPE_FUNC_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -czoE "GetExpectedKernelType[(][^(){}]+[)][^{]+[{][^}]+[}]" || true`
    INDICATE_VAR_DTYPE_CNT=`git diff -U0 --diff-filter=A upstream/$BRANCH |grep "+" |grep -co "IndicateVarDataType" || true`
    if [ ${GET_KERNEL_TYPE_FUNC_CNT} -gt ${INDICATE_VAR_DTYPE_CNT} ]; then
        echo_line="If you override GetExpectedKernelType method of OperatorWithKernel, please use OperatorWithKernel::IndicateVarDataType() method to get specific input variable's dtype, which checked whether the input variable is initialized (The details in https://github.com/PaddlePaddle/FluidDoc/pull/1527). If you don't use this method to check, you must have one RD (zhangbo9674 or phlrain or luotao1) approval for the usage of other methods.\n"
        check_approval 1 luotao1 zhangbo9674 phlrain
    fi
fi

HAS_OPERATORBASE_FLAG=`git diff -U0 --diff-filter=A upstream/$BRANCH | grep -E "public[[:space:]]+.*OperatorBase" || true`
if [ "${HAS_OPERATORBASE_FLAG}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="In order to support dynamic graph, all ops are not recommended to inherit OperatorBase. Please use OperatorWithKernel instead.\nYou must have one RD (phlrain (Recommend), luotao1, XiaoguangHu01) approval for the inherit of OperatorBase.\nYou inherit the OperatorBase class. The corresponding lines are as follows:\n${HAS_OPERATORBASE_FLAG}"
    check_approval 1 phlrain luotao1 XiaoguangHu01
fi

HAS_INPLACE_TESTS=`git diff -U0 upstream/$BRANCH |grep "+" |grep -E "inplace_atol[[:space:]]*=.*" || true`
if [ "${HAS_INPLACE_TESTS}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    echo_line="The calculation results of setting inplace enabled and disabled must be equal, that is, it's not recommended to set inplace_atol.\n If you do need to use inplace_atol, you must have one RD (XiaoguangHu01, phlrain, luotao1, QingshuChen) approval for the usage of inplace_atol.\nThe corresponding lines are as follows:\n${HAS_INPLACE_TESTS}\n"
    check_approval 1 XiaoguangHu01 phlrain luotao1 QingshuChen
fi

OP_FILE_CHANGED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -oE ".+_op..*" || true`
if [ "${OP_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    ERROR_LINES=""
    for OP_FILE in ${OP_FILE_CHANGED};
    do
        CHECK_OBJECT_FLAGS=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${OP_FILE} |grep "+" |grep -E "ShareDataWith[(]|ShareBufferWith[(]" || true`
        if [ "${CHECK_OBJECT_FLAGS}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${OP_FILE}${CHECK_OBJECT_FLAGS}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="Using ShareDataWith or ShareBufferWith is not recommended. You must have one RD's (zhhsplendid (Recommend), zhiqiu or luotao1) approval to use these methods. For more information, please refer to https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-is-prohibited-in-OP. The error lines are as follows:${ERROR_LINES}"
        check_approval 1 zhhsplendid zhiqiu luotao1
    fi
fi

CMAKE_FILE_CHANGED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -E "\.cmake|CMakeLists\.txt"  || true`
if [ "${CMAKE_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    ERROR_LINES=""
    for CMAKE_FILE in ${CMAKE_FILE_CHANGED};
    do
        CHECK_OBJECT_FLAGS=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CMAKE_FILE} |grep "+" |grep -E "\-Wno\-error" || true`
        if [ "${CHECK_OBJECT_FLAGS}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${CMAKE_FILE}${CHECK_OBJECT_FLAGS}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="Change compilation flag of warnings is not recommended. You must have one RD's (zhiqiu (Recommend), luotao1 or phlrain) approval to use these methods. "
        check_approval 1 zhiqiu luotao1 phlrain
    fi
fi

NEW_OP_TEST_ADDED=`git diff --name-only --diff-filter=AMR upstream/$BRANCH |grep -oE "test_.*.\.py" || true`
if [ "${NEW_OP_TEST_ADDED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    CHECK_OUTPUT=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep "self\.check_output(a*t*o*l*=*[0-9]"|grep "+" || true`
    CHECK_OUTPUT_WITH_PLACE=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep -A2 "self\.check_output_with_place" |grep ", [atol*,0-9]"|grep "+" || true`
    CHECK_GRAD=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep -A5 -E "self\.check_grad|self\.check_grad_with_place"|grep "max_relative_error=" |grep "+" || true`
    CHECK_GRAD_CHECK=`git diff -U5 --diff-filter=AMR upstream/$BRANCH |grep -A2 -E "checker\.double_grad_check"|grep "eps=|atol=|rtol=" |grep "+" || true`
    CHECK_WHOLE=$CHECK_OUTPUT$CHECK_OUTPUT_WITH_PLACE$CHECK_GRAD$CHECK_GRAD_CHECK
    if [ "${CHECK_WHOLE}" != "" ] ; then
        CHECK_OP=${CHECK_WHOLE//+/'\n+'}
        echo_line="Please use the default precision parameters of 'atol, rtol, eps, max_relative_error'. If you don't use the default value, you must have one RD (Xreki (Recommend), QingshuChen(Recommend for kunlun), zhiqiu, luotao1, phlrain or ZzSean) approval for the usage of other values. The detailed information is in the link: https://github.cor/PaddlePaddle/Paddle/wiki/OP-test-accuracy-requirements. The error line is ${CHECK_OP}\n"
        check_approval 1 Xreki QingshuChen zhiqiu luotao1 phlrain
    fi
fi

UNITTEST_FILE_CHANGED=`git diff --name-only --diff-filter=AM upstream/$BRANCH |grep -E "test_.*.\.py" || true`
if [ "${UNITTEST_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    ERROR_LINES=""
    for TEST_FILE in ${UNITTEST_FILE_CHANGED};
    do
        HAS_SKIP_CHECK_GRAD_CI=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${TEST_FILE} |grep "@skip_check_grad_ci" || true`
        if [ "${HAS_SKIP_CHECK_GRAD_CI}" != "" ]; then
            ERROR_LINES="${ERROR_LINES}\n${TEST_FILE}\n${HAS_SKIP_CHECK_GRAD_CI}\n"
        fi
    done
    if [ "${ERROR_LINES}" != "" ]; then
        ERROR_LINES=${ERROR_LINES//+/'\n+\t'}
        echo_line="It is an Op accuracy problem, please take care of it. You must have one RD (zhangting2020 (Recommend), luotao1 or phlrain, QingshuChen) approval for the usage (either add or delete) of @skip_check_grad_ci. For more information, please refer to: https://github.com/PaddlePaddle/Paddle/wiki/Gradient-Check-Is-Required-for-Op-Test. The corresponding lines are as follows:\n${ERROR_LINES}\n"
        check_approval 1 zhangting2020 luotao1 phlrain QingshuChen
    fi
fi

RUNTYPE_FILE_CHANGED=`git diff --name-only --diff-filter=AM upstream/$BRANCH|grep -E "CMakeLists.txt"||true`
if [ "${RUNTYPE_FILE_CHANGED}" != "" ] && [ "${GIT_PR_ID}" != "" ]; then
    for CMAKELISTS_FILE in ${RUNTYPE_FILE_CHANGED};
    do
        RUNTYPE_ADD=`git diff -U0 upstream/$BRANCH ${PADDLE_ROOT}/${CMAKELISTS_FILE} |grep "^+" |grep -E "SERIAL|RUN_TYPE=EXCLUSIVE|RUN_TYPE=DIST|RUN_TYPE=HYBRID|RUN_TYPE=NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY|RUN_TYPE=DIST:NIGHTLY|PROPERTIES[[:space:]]+TIMEOUT" || true`
    if [[ ${RUNTYPE_ADD} != "" ]];then
        RUNTYPE_ADD_LINES="${RUNTYPE_ADD_LINES}\n${CMAKELISTS_FILE}\n${RUNTYPE_ADD}\n"
    fi
    done
    if [[ ${RUNTYPE_ADD_LINES} != "" ]];then
        echo_line="You must have one QA (XieYunshen(Recommend) or chalsliu) approval for setting parameter RUN_TYPE as EXCLUSIVE, DIST, HYBRID, NIGHTLY, EXCLUSIVE:NIGHTLY or DISTNIGHTLY, or setting parameter SERIAL, or setting TIMEOUT properties.\nThe corresponding lines are as follows:\n${RUNTYPE_ADD_LINES}\nFor more information, please refer to:https://github.com/PaddlePaddle/Paddle/wiki/PaddlePaddle-Unit-test-specification"
    check_approval 1 XieYunshen chalsliu
    fi
fi

SKIP_CI=`git log --pretty=oneline|grep $AGILE_REVISION |grep -w "test=document_fix" || true`
if [[ ${SKIP_CI} ]];then
    echo_line="You must have one RD (tianshuo78520a (Recommend), zhiqiu, phlrain ) or PM (Ligoml) approval you add test=document_fix method in commit skips CI"
    check_approval 1 tianshuo78520a zhiqiu phlrain Ligoml
fi

# Get the list of PR authors with unresolved unit test issues
pip install PyGithub
# For getting PR related data
wget https://sys-p0.bj.bcebos.com/blk/block.txt --no-check-certificate --no-proxy
wget https://sys-p0.bj.bcebos.com/bk-ci/bk.txt --no-check-certificate --no-proxy
HASUTFIXED=`python ${PADDLE_ROOT}/tools/check_ut.py | grep "has unit-test to be fixed" || true`
if [ "${HASUTFIXED}" != "" ]; then
    echo_line="${HASUTFIXED} You must have one RD (chalsliu (Recommend) or kolinwei) approval.\n"
    check_approval 1 chalsliu kolinwei
fi

HASUTFIXED=`python ${PADDLE_ROOT}/tools/check_ut.py | grep "has benchmark issue to be fixed" || true`
if [ "${HASUTFIXED}" != "" ]; then
    echo_line="${HASUTFIXED} You must have one RD (hysunflower or xiegegege or Xreki) approval.\n"
    check_approval 1 hysunflower xiegegege Xreki
fi

# NOTE(Avin0323): Files with the name "unity_build_rule.cmake" are rules used
# by Unity Build to combine source files. Changes to these rules may cause
# errors in the compilation. Specific personal are required to approve the
# modification of these files.
UNITYBUILD_RULE_CHANGED=$(git diff --name-only upstream/$BRANCH |
                          grep "unity_build_rule.cmake" || true)
if [ -n "${UNITYBUILD_RULE_CHANGED}" -a -n "${GIT_PR_ID}" ]; then
    echo_line="You must have one RD (Avin0323(Recommend) or zhwesky2010 or
               wanghuancoder or luotao1) approval for modifying
               unity_build_rule.cmake which the rules of Unity Build."
    echo_line=$(echo ${echo_line})
    # Avin0323(23427135) zhwesky2010(52485244)
    # wanghuancoder(26922892) luotao1(6836917)
    check_approval 1 Avin0323 zhwesky2010 wanghuancoder luotao1
fi

if [ -n "${echo_list}" ];then
  echo "****************"
  echo -e "${echo_list[@]}"
  echo "There are ${failed_num} approved errors."
  echo "****************"
fi

if [ -n "${echo_list}" ]; then
  exit 6
fi
