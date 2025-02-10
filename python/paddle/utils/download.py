#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import hashlib
import os
import os.path as osp
import shutil
import sys
import tarfile
import time
import zipfile
from typing import Literal

import httpx

try:
    from tqdm import tqdm
except:

    class tqdm:
        def __init__(self, total=None):
            self.total = total
            self.n = 0

        def update(self, n):
            self.n += n
            if self.total is None:
                sys.stderr.write(f"\r{self.n:.1f} bytes")
            else:
                sys.stderr.write(f"\r{100 * self.n / float(self.total):.1f}%")
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr.write('\n')


import logging

logger = logging.getLogger(__name__)

__all__ = ['get_weights_path_from_url']

WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/hapi/weights")

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path: str) -> bool:
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def get_weights_path_from_url(url: str, md5sum: str | None = None) -> str:
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.

    Args:
        url (str): download url
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded weights.

    Examples:
        .. code-block:: python

            >>> from paddle.utils.download import get_weights_path_from_url

            >>> resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
            >>> local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)

    """
    path = get_path_from_url(url, WEIGHTS_HOME, md5sum)
    return path


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def _get_unique_endpoints(trainer_endpoints):
    # Sorting is to avoid different environmental variables for each card
    trainer_endpoints.sort()
    ips = set()
    unique_endpoints = set()
    for endpoint in trainer_endpoints:
        ip = endpoint.split(":")[0]
        if ip in ips:
            continue
        ips.add(ip)
        unique_endpoints.add(endpoint)
    logger.info(f"unique_endpoints {unique_endpoints}")
    return unique_endpoints


def get_path_from_url(
    url: str,
    root_dir: str,
    md5sum: str | None = None,
    check_exist: bool = True,
    decompress: bool = True,
    method: Literal['wget', 'get'] = 'get',
) -> str:
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str|None, optional): md5 sum of download package
        decompress (bool, optional): decompress zip or tar file. Default is `True`
        method (str, optional): which download method to use. Support `wget` and `get`. Default is `get`.

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    from paddle.distributed import ParallelEnv

    assert is_url(url), f"downloading from {url} not a url"
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)
    # Mainly used to solve the problem of downloading data from different
    # machines in the case of multiple machines. Different ips will download
    # data, and the same ip will only download data once.
    unique_endpoints = _get_unique_endpoints(ParallelEnv().trainer_endpoints[:])
    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info(f"Found {fullpath}")
    else:
        if ParallelEnv().current_endpoint in unique_endpoints:
            fullpath = _download(url, root_dir, md5sum, method=method)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)

    if ParallelEnv().current_endpoint in unique_endpoints:
        if decompress and (
            tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath)
        ):
            fullpath = _decompress(fullpath)

    return fullpath


def _get_download(url, fullname):
    # using requests.get method
    fname = osp.basename(fullname)
    try:
        with httpx.stream(
            "GET", url, timeout=None, follow_redirects=True
        ) as req:
            if req.status_code != 200:
                raise RuntimeError(
                    f"Downloading from {url} failed with code "
                    f"{req.status_code}!"
                )

            tmp_fullname = fullname + "_tmp"
            total_size = req.headers.get('content-length')
            with open(tmp_fullname, 'wb') as f:
                if total_size:
                    with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                        for chunk in req.iter_bytes(chunk_size=1024):
                            f.write(chunk)
                            pbar.update(1)
                else:
                    for chunk in req.iter_bytes(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            shutil.move(tmp_fullname, fullname)
            return fullname

    except Exception as e:  # requests.exceptions.ConnectionError
        logger.info(f"Downloading {fname} from {url} failed with exception {e}")
        return False


_download_methods = {'get': _get_download}


def _download(url, path, md5sum=None, method='get'):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    md5sum (str): md5 sum of download package
    method (str): which download method to use. Support `wget` and `get`. Default is `get`.

    """
    assert method in _download_methods, f'make sure `{method}` implemented'

    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    logger.info(f"Downloading {fname} from {url}")
    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        logger.info(f"md5check {fullname} and {md5sum}")
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError(
                f"Download from {url} failed. " "Retry limit reached"
            )

        if not _download_methods[method](url, fullname):
            time.sleep(1)
            continue

    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info(f"File {fullname} md5 checking...")
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info(
            f"File {fullname} md5 check failed, {calc_md5sum}(calc) != "
            f"{md5sum}(base)"
        )
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info(f"Decompressing {fname}...")

    # For protecting decompressing interrupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # succeeded, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError(f"Unsupported compress file type {fname}")

    return uncompressed_path


def _uncompress_file_zip(filepath):
    with zipfile.ZipFile(filepath, 'r') as files:
        file_list_tmp = files.namelist()
        file_list = []
        for file in file_list_tmp:
            file_list.append(file.replace("../", ""))

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir)

        elif _is_a_single_dir(file_list):
            # `strip(os.sep)` to remove `os.sep` in the tail of path
            rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                os.sep
            )[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)

            files.extractall(file_dir)
        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)
            files.extractall(os.path.join(file_dir, rootpath))

        return uncompressed_path


def _uncompress_file_tar(filepath, mode="r:*"):
    with tarfile.open(filepath, mode) as files:
        file_list_tmp = files.getnames()
        file_list = []
        for file in file_list_tmp:
            assert (
                file[0] != "/"
            ), f"uncompress file path {file} should not start with /"
            file_list.append(file.replace("../", ""))

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir)
        elif _is_a_single_dir(file_list):
            rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                os.sep
            )[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir)
        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)

            files.extractall(os.path.join(file_dir, rootpath))

        return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < 0:
        return True
    return False


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if '/' in file_path:
            file_path = file_path.replace('/', os.sep)
        elif '\\' in file_path:
            file_path = file_path.replace('\\', os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True
