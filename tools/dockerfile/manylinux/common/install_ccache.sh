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

wget -q https://paddle-ci.gz.bcebos.com/ccache-4.8.2.tar.gz
tar xf ccache-4.8.2.tar.gz && mkdir /usr/local/ccache-4.8.2 && cd ccache-4.8.2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ccache-4.8.2 ..
make -j8 && make install
ln -s /usr/local/ccache-4.8.2/bin/ccache /usr/local/bin/ccache
cd ../../ && rm -rf ccache-4.8.2.tar.gz
