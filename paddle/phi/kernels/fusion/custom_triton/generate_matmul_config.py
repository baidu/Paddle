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

import re
import os
import shutil
import threading

# AOT生成命令模板
template = '''
python3.8   {compile_file}      \
/nishirong/Paddle/paddle/phi/kernels/fusion/custom_triton/matmul_triton.py   \
-n matmul_kernel   \
-o {matmul_dir}/matmul_fp16     \
--out-name matmul_kernel_fp16     \
-w {num_warps}   -ns {num_stages}  \
-s   "*fp16:16, *fp16:16, *fp16:16, *fp16:16, i32, i32:16, i32:16, i32:16, i32:1, i32:16, i32:1,  i32:16, i32:1, {block_m}, {block_n}, {block_k}, {group_m}, {split_k}, 0"   \
-g   "((M+{block_m}-1)/{block_m}) * ((N+{block_n}-1)/{block_n}), {split_k}, 1" \
'''

# 新建生成kernel的文件夹
matmul_dir = "generated/matmul/"
compile_file="triton/python/triton/tools/compile.py"
link_file="triton/python/triton/tools/link.py"
shutil.rmtree(matmul_dir, ignore_errors=True)
os.makedirs(matmul_dir, exist_ok=True)

os.system("export TRITON_USE_PADDLE=TRUE")
config_num = 0
thread_num = 200
# AOT kernel
codegen_commands = []
for num_stages in [2, 3]:
     for block_m in [32, 128, 128]:
         for block_n in [32, 128, 256]:
            for group_m in [8]:
                for block_k in [64, 128, 256]:
                    num_warps = 4
                    if block_m * block_n >= 128 * 256:
                        num_warps = 8
                    for split_k in [1, 2]:
                        codegen_command = template.format(
                            compile_file=compile_file,
                            matmul_dir=matmul_dir,
                            num_stages=num_stages,
                            block_m=block_m,
                            block_n=block_n,
                            block_k=block_k,
                            group_m=group_m,
                            split_k=split_k,
                            num_warps=num_warps
                        )
                        # print(codegen_command)
                        codegen_commands.append(codegen_command)

def execute_commands(commands, thread_id):
    i = thread_id
    while (i < len(commands)):
        return_value = os.system(commands[i])
        print(return_value)
        i += thread_num

# 创建线程并启动
thread_list = []
for thread_id in range(0, thread_num):
    thread = threading.Thread(target=execute_commands, args=(codegen_commands,thread_id,))
    thread_list.append(thread)
    thread.start()

# 等待所有线程执行完毕
for thread in thread_list:
    thread.join()

# 链接
link_command = "python3.8  {link_file}  {matmul_dir}/*.h -o {matmul_dir}/fp16".format(
    link_file=link_file, matmul_dir=matmul_dir)
re = os.system(link_command)

# 遍历目录中的文件,重命名
for filename in os.listdir(matmul_dir):
    if filename.endswith(".c"):
        old_path = os.path.join(matmul_dir, filename)
        new_path = os.path.join(matmul_dir, filename + "u")
        os.rename(old_path, new_path)

