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

from __future__ import print_function

import six
import numpy as np

import paddle.dataset.common
from paddle.io import Dataset
from .utils import _check_exists_and_download

__all__ = ["UCIHousing"]

URL = 'http://paddlemodels.bj.bcebos.com/uci_housing/housing.data'
MD5 = 'd4accdce7a25600298819f8e28e8d593'
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT'
]


class UCIHousing(Dataset):
    """
    Implement of UCI housing dataset

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): whether auto download cifar dataset if
            :attr:`data_file` unset. Default
            True

    Returns:
        Dataset: instance of UCI housing dataset.

    Examples:
        
        .. code-block:: python

            from paddle.incubate.hapi.datasets import UCIHousing	

            uci_housing = UCIHousing(mode='test')

            for i in range(len(uci_housing)):
                sample = uci_housing[i]
                print(sample)

    """

    def __init__(self, data_file=None, mode='train', download=True):
        assert mode.lower() in ['train', 'test'], \
                "mode should be 'train' or 'test', but got {}".format(mode)
        self.mode = mode.lower()

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file not set and auto download disabled"
            self.data_file = _check_exists_and_download(data_file, URL, MD5,
                                                        'uci_housing', download)

        # read dataset into memory
        self._load_data()

    def _load_data(self, feature_num=14, ratio=0.8):
        data = np.fromfile(self.data_file, sep=' ')
        data = data.reshape(data.shape[0] // feature_num, feature_num)
        maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
            axis=0) / data.shape[0]
        for i in six.moves.range(feature_num - 1):
            data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
        offset = int(data.shape[0] * ratio)
        if self.mode == 'train':
            self.data = data[:offset]
        elif self.mode == 'test':
            self.data = data[offset:]

    def __getitem__(self, idx):
        data = self.data[idx]
        return np.array(data[:-1]), np.array(data[-1:])

    def __len__(self):
        return len(self.data)
