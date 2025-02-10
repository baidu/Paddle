#!/bin/bash

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

function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}

function check_sha256sum {
    local fname=$1
    check_var ${fname}
    local sha256=$2
    check_var ${sha256}

    echo "${sha256}  ${fname}" > ${fname}.sha256
    sha256sum -c ${fname}.sha256
    rm ${fname}.sha256
}

function do_openssl_build {
    ./config -fPIC --prefix=/usr/local/ssl > /dev/null
    make > /dev/null
    make install > /dev/null
    ln -sf /usr/lib64/libcrypto.so.1.1 /usr/local/ssl/lib/libcrypto.so.1.1
}


function build_openssl {
    local openssl_fname=$1
    check_var ${openssl_fname}
    local openssl_sha256=$2
    check_var ${openssl_sha256}
    check_var ${OPENSSL_DOWNLOAD_URL}
    curl -sLO ${OPENSSL_DOWNLOAD_URL}/${openssl_fname}.tar.gz
    check_sha256sum ${openssl_fname}.tar.gz ${openssl_sha256}
    tar -xzf ${openssl_fname}.tar.gz
    (cd ${openssl_fname} && do_openssl_build)
    rm -rf ${openssl_fname} ${openssl_fname}.tar.gz
}

OPENSSL_ROOT=openssl-1.1.1
OPENSSL_HASH=2836875a0f89c03d0fdf483941512613a50cfb421d6fd94b9f41d7279d586a3d
OPENSSL_DOWNLOAD_URL=https://www.openssl.org/source
build_openssl $OPENSSL_ROOT $OPENSSL_HASH
