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

import sys
import unittest
import warnings

import numpy as np

import paddle
from paddle.utils import deprecated

LOWEST_WARNING_POSITION = 3
ERROR_WARNING_POSITION = sys.maxsize

# custom paddle version
paddle.version.major = '0'
paddle.version.minor = '0'
paddle.version.patch = '0'
paddle.version.rc = '0'
paddle.__version__ = '0.0.0'
paddle.version.full_version = '0.0.0'
print("current paddle version: ", paddle.__version__)


def get_warning_index(api):
    """
    Given an paddle API, return the index of the Warning information in its doc string if exists;
    If Warning information doesn't exist, return the default ERROR_WARNING_POSITION, sys.maxsize.

    Args:
        API (python object)

    Returns:
        index (int): the index of the Warning information in its doc string if exists.
    """

    doc_list = api.__doc__.splitlines()
    if len(doc_list) < 2:
        return ERROR_WARNING_POSITION
    for idx, (current_line, next_line) in enumerate(
        zip(doc_list[:-1], doc_list[1:])
    ):
        if (
            current_line == "Warning:"
            and next_line.endswith(" instead.")
            and "and will be removed in future versions." in next_line
        ):
            return idx
    return ERROR_WARNING_POSITION


class TestDeprecatedDecorator(unittest.TestCase):
    """
    tests for paddle's deprecated decorator.
    test_new_multiply: test for new api, which should not insert warning information.
    """

    def test_new_multiply(self):
        """
        Test for new multiply api, expected result should be False.
        """

        a = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        b = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        x = paddle.to_tensor(a)
        y = paddle.to_tensor(b)
        res = paddle.multiply(x, y)

        # expected
        expected = LOWEST_WARNING_POSITION

        # captured
        captured = get_warning_index(paddle.multiply)

        # testing
        self.assertLess(expected, captured)

    def test_indent_level(self):
        # test for different indent_level
        dataset = paddle.base.DatasetFactory().create_dataset("InMemoryDataset")
        with warnings.catch_warnings(record=True):
            dataset.set_merge_by_lineid()
            assert (
                '\nSet merge by'
                in paddle.base.InMemoryDataset.set_merge_by_lineid.__doc__
            )

    def test_tensor_gradient(self):
        paddle.__version__ = '2.1.0'

        x = paddle.to_tensor([5.0], stop_gradient=False)
        y = paddle.pow(x, 4.0)
        y.backward()

        with warnings.catch_warnings(record=True) as w:
            grad = x.gradient()
            assert (
                'API "paddle.base.dygraph.tensor_patch_methods.gradient" is '
                'deprecated since 2.1.0'
            ) in str(w[-1].message)

    def test_softmax_with_cross_entropy(self):
        paddle.__version__ = '2.0.0'

        data = np.random.rand(128).astype("float32")
        label = np.random.rand(1).astype("int64")
        data = paddle.to_tensor(data)
        label = paddle.to_tensor(label)
        linear = paddle.nn.Linear(128, 100)
        x = linear(data)

        with warnings.catch_warnings(record=True) as w:
            out = paddle.nn.functional.softmax_with_cross_entropy(
                logits=x, label=label
            )
            assert (
                'API "paddle.nn.functional.loss.softmax_with_cross_entropy" is '
                'deprecated since 2.0.0'
            ) in str(w[-1].message)

    def test_deprecated_error(self):
        paddle.__version__ = '2.1.0'

        @deprecated(since="2.1.0", level=2)
        def deprecated_error_func():
            pass

        self.assertRaises(RuntimeError, deprecated_error_func)


if __name__ == '__main__':
    unittest.main()
