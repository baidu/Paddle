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

import ssl
import sys

import httpx

import paddle
from paddle.base import core
from paddle.device import cuda


def get_disable_ut_by_url(url):
    ssl._create_default_https_context = ssl._create_unverified_context
    f = httpx.get(url, timeout=None, follow_redirects=True)
    data = f.text
    status_code = f.status_code
    if len(data.strip()) == 0 or status_code != 200:
        sys.exit(1)
    else:
        lt = data.strip().split('\n')
        lt = '^' + '$|^'.join(lt) + '$'
        return lt


def download_file():
    """Get disabled unit tests"""
    sysstr = sys.platform
    if sysstr == 'win32':
        url = "https://sys-p0.bj.bcebos.com/prec/{}".format('disable_ut_win')
    else:
        url = "https://sys-p0.bj.bcebos.com/prec/{}".format('disable_ut')

    if paddle.is_compiled_with_rocm():
        if cuda.get_device_name() == 'K100_AI':
            url = "https://sys-p0.bj.bcebos.com/prec/{}".format(
                'disable_ut_rocm_k100'
            )
        else:
            url = "https://sys-p0.bj.bcebos.com/prec/{}".format(
                'disable_ut_rocm'
            )

    disabled_ut_list = get_disable_ut_by_url(url)

    if paddle.is_compiled_with_xpu():
        xpu_version = core.get_xpu_device_version(0)
        if xpu_version != core.XPUVersion.XPU3:
            url = "https://sys-p0.bj.bcebos.com/prec/{}".format(
                'disable_ut_xpu_kl2'
            )
            external_xpu = get_disable_ut_by_url(url)
        else:
            # part 1: "quick" list on bos
            url = "https://sys-p0.bj.bcebos.com/prec/{}".format(
                'quick_disable_ut_xpu_kl3'
            )
            external_xpu = get_disable_ut_by_url(url)

            # part 2: local list
            import os

            paddle_root = os.getenv('PADDLE_ROOT')
            file_path = paddle_root + "/tools/xpu/disable_ut_xpu_kl3.local"
            with open(file_path, 'r') as file:
                data = file.read()
            local_list = data.strip().split('\n')
            local_list = '^' + '$|^'.join(local_list) + '$'
            external_xpu = external_xpu + "|" + local_list
        disabled_ut_list = disabled_ut_list + "|" + external_xpu

    print(disabled_ut_list)
    sys.exit(0)


if __name__ == '__main__':
    try:
        download_file()
    except Exception as e:
        print(e)
        sys.exit(1)
