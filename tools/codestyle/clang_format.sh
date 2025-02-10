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

set -e

readonly VERSION="13.0.0"

version=$(clang-format -version)

if ! [[ $(python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1$2}') -ge 36 ]]; then
    echo "clang-format installation by pip need python version great equal 3.6,
          please change the default python to higher version."
    exit 1
fi

if ! [[ $version == *"$VERSION"* ]]; then
    # low version of pip may not have the source of clang-format whl
    pip install --upgrade pip
    pip install clang-format==13.0.0
    pip install clang-tidy==15.0.2.1
fi

clang-format $@
