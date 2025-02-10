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

set -e

TMP_DIR=patchelf_tmp

gcc_version=$(gcc --version |awk 'NR==1{print $3}')
if [ "$gcc_version" == "5.4.0" ];then
    patchelf_version=0.10
else
    patchelf_version=0.15.0
fi

rm -rf "$TMP_DIR"

git clone -b $patchelf_version https://github.com/NixOS/patchelf "$TMP_DIR"

cd "$TMP_DIR"
./bootstrap.sh
./configure
make
make install

cd ..
rm -rf "$TMP_DIR"
