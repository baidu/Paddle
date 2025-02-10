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

function lex_pyver {
    # Echoes Python version string padded with zeros
    # Thus:
    # 3.2.1 -> 003002001
    # 3     -> 003000000
    echo $1 | awk -F "." '{printf "%03d%03d%03d", $1, $2, $3}'
}


function do_cpython_build {
    local py_ver=$1
    check_var $py_ver
    local ucs_setting=$2
    check_var $ucs_setting
    tar -xzf Python-$py_ver.tgz
    pushd Python-$py_ver
    if [ "$ucs_setting" = "none" ]; then
        unicode_flags=""
        dir_suffix=""
    else
        local unicode_flags="--enable-unicode=$ucs_setting"
        local dir_suffix="-$ucs_setting"
    fi
    local prefix="/opt/_internal/cpython-${py_ver}${dir_suffix}"
    mkdir -p ${prefix}/lib

    #if [ $1 -eq '3.13.0t' ];then
    #    GIL='--disable-gil'
    #fi

    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} CFLAGS="-Wformat" ./configure --prefix=${prefix} --enable-shared $unicode_flags > /dev/null
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} make -j8 > /dev/null
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} make install > /dev/null

    popd
    echo "ZZZ looking for libpython"
    find / -name 'libpython*.so*'
    rm -rf Python-$py_ver
    # Some python's install as bin/python3. Make them available as bin/python.
    if [ -e ${prefix}/bin/python3.8 ]; then
        ln -s python3.8 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.9 ]; then
        ln -s python3.9 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.10 ]; then
        ln -s python3.10 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.11 ]; then
        ln -s python3.11 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.12 ]; then
        ln -s python3.12 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.13 ]; then
        ln -s python3.13 ${prefix}/bin/python
    fi
    if [ -e ${prefix}/bin/python3.13t ]; then
        ln -s python3.13t ${prefix}/bin/python
    fi

    # NOTE Make libpython shared library visible to python calls below
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python -m pip config set global.trusted-host mirrors.aliyun.com
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/python get-pip.py
    LD_LIBRARY_PATH="/usr/local/ssl/lib:${prefix}/lib" ${prefix}/bin/pip install wheel==0.40.0
    cd /
    ls ${MY_DIR}
    abi_version=$(LD_LIBRARY_PATH="${prefix}/lib" ${prefix}/bin/python -V|awk '{print $2}'|awk -F '.' '{print $1$2}')
    local abi_tag=$(echo cp$abi_version-cp$abi_version)
    ln -s ${prefix} /opt/python/${abi_tag}
}


function build_cpython {
    local py_ver=$1
    check_var $py_ver
    check_var $PYTHON_DOWNLOAD_URL
    wget -q $PYTHON_DOWNLOAD_URL/$py_ver/Python-$py_ver.tgz
    do_cpython_build $py_ver none
    rm -f Python-$py_ver.tgz
}


function build_cpythons {
    for py_ver in $@; do
        check_var $GET_PIP_URL
        curl -sLO $GET_PIP_URL
        build_cpython $py_ver
    done
    rm -f get-pip.py
    rm -f ez_setup.py
}

PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python
GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py
CPYTHON_VERSIONS="3.13.0 3.12.0 3.11.0 3.10.0 3.9.0 3.8.0"

mkdir -p /opt/python
build_cpythons $CPYTHON_VERSIONS


mkdir -p /opt/python
build_cpythons $CPYTHON_VERSIONS
#PY38_BIN=/opt/python/cp38-cp38/bin
#PY39_BIN=/opt/python/cp39-cp39/bin
#PY310_BIN=/opt/python/cp310-cp310/bin
#PY311_BIN=/opt/python/cp311-cp311/bin
#PY312_BIN=/opt/python/cp312-cp312/bin
#PY313_BIN=/opt/python/cp313-cp313/bin
#PY313T_BIN=/opt/python/cp313-cp313t/bin
#
#LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname ${PY38_BIN})/lib" $PY38_BIN/pip install certifi
#ln -s $($PY38_BIN/python -c 'import certifi; print(certifi.where())') \
#      /opt/_internal/certs.pem
#
#find /opt/_internal -name '*.a' -print0 | xargs -0 rm -f
#find /opt/_internal -type f -print0 \
#    | xargs -0 -n1 strip --strip-unneeded 2>/dev/null || true
#
#find /opt/_internal \
#     \( -type d -a -name test -o -name tests \) \
#  -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
#  -print0 | xargs -0 rm -f
#
#for PYTHON in /opt/python/*/bin/python; do
#    # Add matching directory of libpython shared library to library lookup path
#    LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname $(dirname ${PYTHON}))/lib"
#
#    if [ "$(dirname $(dirname ${PYTHON}))" != "/opt/python/cp310-cp310" -a "$(dirname $(dirname ${PYTHON}))" != "/opt/python/cp311-cp311" ]; then
#        # Smoke test to make sure that our Pythons work, and do indeed detect as
#        # being manylinux compatible:
#        LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname $(dirname ${PYTHON}))/lib" $PYTHON $MY_DIR/manylinux1-check.py
#        # Make sure that SSL cert checking works
#        LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}:$(dirname $(dirname ${PYTHON}))/lib"
#    fi
#done
