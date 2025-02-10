#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from . import core
from .data_feeder import DataToDenseTensorConverter

__all__ = []


def create_lod_tensor(data, recursive_seq_lens, place):
    """
    Create a DenseTensor from a numpy array, list or existing DenseTensor.

    The implementation is as follows:

    1. Check whether the length-based LoD, i.e., :code:`recursive_seq_lens`
       is valid.

    2. Convert :code:`recursive_seq_lens` to a offset-based LoD.

    3. Based on :code:`place` , copy the :code:`data` from a numpy array, list
       or existing DenseTensor to CPU or GPU device.

    4. Set offset-based LoD to the output DenseTensor.

    Suppose we want to create a DenseTensor to hold data for word sequences,
    where each word is represented by an integer. If we want to create
    a DenseTensor to represent two sentences, one of 2 words, and one of 3 words.

    Then :code:`data` would be a numpy array of integers with shape (5, 1).
    :code:`recursive_seq_lens` would be [[2, 3]], indicating the word number
    in each sentence. This length-based :code:`recursive_seq_lens` [[2, 3]]
    would be converted to offset-based LoD [[0, 2, 5]] inside the function
    call.


    Args:
        data (numpy.ndarray|list|DenseTensor): a numpy array, a list or ad DenseTensor
                holding the data to be copied.
        recursive_seq_lens (list[list[int]]): a list of lists indicating the
                length-based LoD info.
        place (CPUPlace|CUDAPlace): CPU or GPU place indicating where the data
                in the created DenseTensor will be stored.

    Returns:
         A DenseTensor with tensor data and recursive_seq_lens info.

    Examples:

        .. code-block:: python

            >>> import paddle.base as base
            >>> import numpy as np

            >>> t = base.create_lod_tensor(np.ndarray([5, 30]), [[2, 3]], base.CPUPlace())
    """
    if isinstance(data, core.DenseTensor):
        return create_lod_tensor(np.array(data), recursive_seq_lens, place)
    elif isinstance(data, list):
        # dtype and shape are not important here,
        # we only want to reuse code of DataToDenseTensorConverter
        converter = DataToDenseTensorConverter(
            place=place,
            lod_level=len(recursive_seq_lens),
            shape=[],
            dtype=core.VarDesc.VarType.FP32,
        )

        new_recursive_seq_lens = []
        for seq in data:
            new_recursive_seq_lens.append(len(seq))
            converter.feed(seq)

        assert [
            new_recursive_seq_lens
        ] == recursive_seq_lens, "data and recursive_seq_lens do not match"

        arr = np.array(converter.data)

        # FIXME(zjl): the original logic of create_lod_tensor would append
        # 1 to the shape. Maybe it is not a right way? Currently, we only
        # follow the previous logic
        arr = arr.reshape((*arr.shape, 1))
        tensor = core.DenseTensor()
        tensor.set(arr, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        return tensor
    elif isinstance(data, np.ndarray):
        tensor = core.DenseTensor()
        tensor.set(data, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        return tensor
    else:
        raise TypeError(
            "data should be either a DenseTensor, a Numpy array or a list"
        )
