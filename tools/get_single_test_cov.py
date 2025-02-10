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

import json
import os
import re
import subprocess
import sys
import time


def getFNDAFile(rootPath, test):
    # load base fnda
    fnda_base_dict = {}
    find_file_cmd = os.popen(f"find {rootPath} -name {test}.cc")
    if find_file_cmd.read() != "":
        print(f"{test} is a c++ unittest")
        with open(
            f"{rootPath}/build/ut_map/simple_precision_test/base_fnda.json",
            'r',
        ) as load_f:
            fnda_base_dict = json.load(load_f)
    # analyse fnda
    filename = f'{rootPath}/build/ut_map/{test}/coverage.info.tmp'
    fn_filename = f'{rootPath}/build/ut_map/{test}/fnda.tmp'
    os.system(f'touch {fn_filename}')
    try:
        f = open(filename)
        print(f"open {filename} successfully")
    except FileNotFoundError:
        print(f"{filename} is not found.")
        return
    all_data = f.read().split('TN:')
    del all_data[0]
    for gcov_data in all_data:
        message_list = gcov_data.split('\n')
        os.system(f'echo {message_list[1]} >> {fn_filename}')
        if 'FNH:0' not in gcov_data:
            for message in message_list:
                if message.startswith('FNDA:') and (
                    not message.startswith('FNDA:0,')
                ):
                    tmp_data = message.split('FNDA:')[1].split(',')
                    hit = int(tmp_data[0])
                    symbol = tmp_data[1]
                    if symbol in fnda_base_dict:
                        if (hit - fnda_base_dict[symbol]) > 0:
                            fnda_str = (
                                f'FNDA:{hit - fnda_base_dict[symbol]},{symbol}'
                            )
                            os.system(f'echo {fnda_str} >> {fn_filename}')
                    else:
                        os.system(f'echo {message} >> {fn_filename}')
    f.close()


def analysisFNDAFile(rootPath, test):
    related_ut_map_file = f'{rootPath}/build/ut_map/{test}/related_{test}.txt'
    notrelated_ut_map_file = (
        f'{rootPath}/build/ut_map/{test}/notrelated_{test}.txt'
    )
    os.system(f'touch {related_ut_map_file}')
    os.system(f'touch {notrelated_ut_map_file}')

    if os.path.isfile(related_ut_map_file) and os.path.isfile(
        notrelated_ut_map_file
    ):
        print(
            f"make {related_ut_map_file} and {related_ut_map_file} successfully"
        )
    else:
        print(f"make {related_ut_map_file} and {related_ut_map_file} failed")
        return

    fn_filename = f'{rootPath}/build/ut_map/{test}/fnda.tmp'
    try:
        f = open(fn_filename)
        print(f"oepn {fn_filename} successfully")
    except FileNotFoundError:
        print(f"{fn_filename} is not found.")
        return
    data = f.read().split('SF:')
    related_file_list = []
    for message in data:
        message_list = message.split('\n')
        clazz_filename = message_list[0]
        if '/build/' in clazz_filename:
            clazz_filename = clazz_filename.replace('/build', '')
        if '.pb.h' in clazz_filename:
            clazz_filename = clazz_filename.replace('.pb.h', '.proto')
        if '.pb.cc' in clazz_filename:
            clazz_filename = clazz_filename.replace('.pb.cc', '.proto')
        if 'FNDA:' in message:
            OP_REGISTER = True
            for i in range(1, len(message_list) - 1):
                fn = message_list[i]
                matchObj = re.match(
                    r'(.*)Maker(.*)|(.*)Touch(.*)Regist(.*)|(.*)Touch(.*)JitKernel(.*)|(.*)converterC2Ev(.*)',  # typos: disable-line
                    fn,
                    re.IGNORECASE,
                )
                if matchObj is None:
                    OP_REGISTER = False
                    break
            if not OP_REGISTER:
                related_file_list.append(clazz_filename)
                os.system(f'echo {clazz_filename} >> {related_ut_map_file}')
            else:
                os.system(f'echo {clazz_filename} >> {notrelated_ut_map_file}')
        else:
            if clazz_filename != '':
                if (
                    clazz_filename not in related_file_list
                ):  # xx.pb.cc in RELATED xx.pb.h not in RELATED
                    os.system(
                        f'echo {clazz_filename} >> {notrelated_ut_map_file}'
                    )
    f.close()


def getBaseFnda(rootPath, test):
    filename = f'{rootPath}/build/ut_map/{test}/coverage.info.tmp'
    try:
        f = open(filename)
        print(f"oepn {filename} successfully")
    except FileNotFoundError:
        print(f"{filename} is not found.")
    symbol_fnda = {}
    all_data = f.read().split('TN:')
    del all_data[0]
    for gcov_data in all_data:
        message_list = gcov_data.split('\n')
        # only for cc file
        if ".cc" in message_list[1]:
            for message in message_list:
                if message.startswith('FNDA:') and (
                    not message.startswith('FNDA:0,')
                ):
                    tmp_data = message.split('FNDA:')[1].split(',')
                    symbol_fnda[tmp_data[1]] = int(tmp_data[0])
    f.close()

    with open(f"{rootPath}/build/ut_map/{test}/base_fnda.json", "w") as f:
        json.dump(symbol_fnda, f, indent=4)


def getCovinfo(rootPath, test):
    ut_map_path = f'{rootPath}/build/ut_map/{test}'
    print("start get fluid ===>")
    cmd_fluid = f'cd {ut_map_path} && lcov --capture -d paddle/fluid/ -o paddle/fluid/coverage_fluid.info --rc lcov_branch_coverage=0'
    p_fluid = subprocess.Popen(cmd_fluid, shell=True, stdout=subprocess.DEVNULL)

    print("start get phi ===>")
    cmd_phi = f'cd {ut_map_path} && lcov --capture -d paddle/phi -o paddle/phi/coverage_phi.info --rc lcov_branch_coverage=0'
    if os.path.exists(f"{ut_map_path}/paddle/phi"):
        p_phi = subprocess.Popen(cmd_phi, shell=True, stdout=subprocess.DEVNULL)

    print("start get utils ===>")
    cmd_utils = f'cd {ut_map_path} && lcov --capture -d paddle/utils -o paddle/utils/coverage_utils.info --rc lcov_branch_coverage=0'
    if os.path.exists(f"{ut_map_path}/paddle/utils"):
        p_utils = subprocess.Popen(
            cmd_utils, shell=True, stdout=subprocess.DEVNULL
        )
    print("start wait fluid ===>")
    p_fluid.wait()
    print("start wait phi ===>")
    p_phi.wait()
    print("start wait utils ===>")
    p_utils.wait()
    print("end wait...")
    coverage_utils_info_path = f"{ut_map_path}/paddle/utils/coverage_utils.info"
    if (
        os.path.exists(coverage_utils_info_path)
        and os.path.getsize(coverage_utils_info_path) > 4
    ):
        os.system(
            f'cd {ut_map_path} && lcov -a paddle/fluid/coverage_fluid.info -a paddle/phi/coverage_phi.info -a paddle/utils/coverage_utils.info -o coverage.info --rc lcov_branch_coverage=0 > /dev/null 2>&1'
        )
    else:
        os.system(
            f'cd {ut_map_path} && lcov -a paddle/fluid/coverage_fluid.info -a paddle/phi/coverage_phi.info -o coverage.info --rc lcov_branch_coverage=0 > /dev/null 2>&1'
        )
    coverage_info_path = ut_map_path + '/coverage.info'
    file_size = os.path.getsize(coverage_info_path)
    if file_size == 0:
        print(
            f"coverage.info of {ut_map_path} is empty,collect coverage rate failed"
        )
        return
    else:
        print(f"get coverage.info of {ut_map_path} successfully")
    os.system(
        f"cd {ut_map_path} && lcov --extract coverage.info '/paddle/paddle/phi/*' '/paddle/paddle/utils/*' '/paddle/paddle/fluid/*' '/paddle/build/*' -o coverage.info.tmp --rc lcov_branch_coverage=0 > /dev/null 2>&1"
    )
    coverage_info_tmp = ut_map_path + '/coverage.info.tmp'
    coverage_tmp_size = os.path.getsize(coverage_info_tmp)
    if coverage_tmp_size == 0:
        print("coverage.info.tmp is empty,collect coverage rate failed")
        return
    else:
        print("get coverage.info.tmp successfully")

    os.system(f'rm -rf {ut_map_path}/paddle')
    os.system(f'rm -rf {ut_map_path}/coverage.info')
    if test == "simple_precision_test":
        getBaseFnda(rootPath, test)
    else:
        start_getFNDAFile = time.time()
        getFNDAFile(rootPath, test)
        end_getFNDAFile = time.time()
        print("getFNDAFile time:", end_getFNDAFile - start_getFNDAFile)
        start_analysisFNDAFile = time.time()
        analysisFNDAFile(rootPath, test)
        end_analysisFNDAFile = time.time()
        print(
            "analysisFNDAFile time :",
            end_analysisFNDAFile - start_analysisFNDAFile,
        )
    os.system(f'rm -rf {ut_map_path}/coverage.info.tmp')


if __name__ == "__main__":
    rootPath = sys.argv[1]
    case = sys.argv[2]
    start_getCovinfo = time.time()
    getCovinfo(rootPath, case)
    end_getCovinfo = time.time()
    print("getConvinfo time :", end_getCovinfo - start_getCovinfo)
