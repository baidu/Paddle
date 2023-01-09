#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np

import paddle
from paddle.static import Program, program_guard


# In static graph mode, inplace strategy will not be used in Inplace APIs.
class TestStaticAutoGeneratedAPI(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.init_data()
        self.set_np_compare_func()

    def init_data(self):
        self.dtype = 'float32'
        self.shape = [10, 20]
        self.np_x = np.random.uniform(-5, 5, self.shape).astype(self.dtype)

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def executed_paddle_api(self, x):
        return x.ceil()

    def executed_numpy_api(self, x):
        return np.ceil(x)

    def test_api(self):
        main_prog = Program()
        with program_guard(main_prog, Program()):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            out = self.executed_paddle_api(x)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        fetch_x, fetch_out = exe.run(
            main_prog, feed={"x": self.np_x}, fetch_list=[x, out]
        )

        np.testing.assert_array_equal(fetch_x, self.np_x)
        self.assertTrue(
            self.np_compare(fetch_out, self.executed_numpy_api(self.np_x))
        )


class TestStaticInplaceAutoGeneratedAPI(TestStaticAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.ceil_()


class TestStaticFloorAPI(TestStaticAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.floor()

    def executed_numpy_api(self, x):
        return np.floor(x)


class TestStaticInplaceFloorAPI(TestStaticFloorAPI):
    def executed_paddle_api(self, x):
        return x.floor_()


class TestStaticExpAPI(TestStaticAutoGeneratedAPI):
    def set_np_compare_func(self):
        self.np_compare = np.allclose

    def executed_paddle_api(self, x):
        return x.exp()

    def executed_numpy_api(self, x):
        return np.exp(x)


class TestStaticInplaceExpAPI(TestStaticExpAPI):
    def executed_paddle_api(self, x):
        return x.exp_()


class TestStaticReciprocalAPI(TestStaticAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.reciprocal()

    def executed_numpy_api(self, x):
        return np.reciprocal(x)


class TestStaticInplaceReciprocalAPI(TestStaticReciprocalAPI):
    def executed_paddle_api(self, x):
        return x.reciprocal_()


class TestStaticRoundAPI(TestStaticAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.round()

    def executed_numpy_api(self, x):
        return np.round(x)


class TestStaticInplaceRoundAPI(TestStaticRoundAPI):
    def executed_paddle_api(self, x):
        return x.round_()


class TestStaticSqrtAPI(TestStaticAutoGeneratedAPI):
    def init_data(self):
        self.dtype = 'float32'
        self.shape = [10, 20]
        self.np_x = np.random.uniform(0, 5, self.shape).astype(self.dtype)

    def set_np_compare_func(self):
        self.np_compare = np.allclose

    def executed_paddle_api(self, x):
        return x.sqrt()

    def executed_numpy_api(self, x):
        return np.sqrt(x)


class TestStaticInplaceSqrtAPI(TestStaticSqrtAPI):
    def executed_paddle_api(self, x):
        return x.sqrt_()


class TestStaticRsqrtAPI(TestStaticSqrtAPI):
    def executed_paddle_api(self, x):
        return x.rsqrt()

    def executed_numpy_api(self, x):
        return 1 / np.sqrt(x)


class TestStaticInplaceRsqrtAPI(TestStaticRsqrtAPI):
    def executed_paddle_api(self, x):
        return x.rsqrt_()


# In dygraph mode, inplace strategy will be used in Inplace APIs.
class TestDygraphAutoGeneratedAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.init_data()
        self.set_np_compare_func()

    def init_data(self):
        self.dtype = 'float32'
        self.shape = [10, 20]
        self.np_x = np.random.uniform(-5, 5, self.shape).astype(self.dtype)

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def executed_paddle_api(self, x):
        return x.ceil()

    def executed_numpy_api(self, x):
        return np.ceil(x)

    def test_api(self):
        x = paddle.to_tensor(self.np_x, dtype=self.dtype)
        out = self.executed_paddle_api(x)

        self.assertTrue(
            self.np_compare(out.numpy(), self.executed_numpy_api(self.np_x))
        )


class TestDygraphInplaceAutoGeneratedAPI(TestDygraphAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.ceil_()


class TestDygraphFloorAPI(TestDygraphAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.floor()

    def executed_numpy_api(self, x):
        return np.floor(x)


class TestDygraphInplaceFloorAPI(TestDygraphFloorAPI):
    def executed_paddle_api(self, x):
        return x.floor_()


class TestDygraphExpAPI(TestDygraphAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.exp()

    def executed_numpy_api(self, x):
        return np.exp(x)

    def set_np_compare_func(self):
        self.np_compare = np.allclose


class TestDygraphInplaceExpAPI(TestDygraphExpAPI):
    def executed_paddle_api(self, x):
        return x.exp_()


class TestDygraphReciprocalAPI(TestDygraphAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.reciprocal()

    def executed_numpy_api(self, x):
        return np.reciprocal(x)


class TestDygraphInplaceReciprocalAPI(TestDygraphReciprocalAPI):
    def executed_paddle_api(self, x):
        return x.reciprocal_()


class TestDygraphRoundAPI(TestDygraphAutoGeneratedAPI):
    def executed_paddle_api(self, x):
        return x.round()

    def executed_numpy_api(self, x):
        return np.round(x)


class TestDygraphInplaceRoundAPI(TestDygraphRoundAPI):
    def executed_paddle_api(self, x):
        return x.round_()


class TestDygraphSqrtAPI(TestDygraphAutoGeneratedAPI):
    def init_data(self):
        self.dtype = 'float32'
        self.shape = [10, 20]
        self.np_x = np.random.uniform(0, 100, self.shape).astype(self.dtype)

    def set_np_compare_func(self):
        self.np_compare = np.allclose

    def executed_paddle_api(self, x):
        return x.sqrt()

    def executed_numpy_api(self, x):
        return np.sqrt(x)


class TestDygraphInplaceSqrtAPI(TestDygraphSqrtAPI):
    def executed_paddle_api(self, x):
        return x.sqrt_()


class TestDygraphRsqrtAPI(TestDygraphSqrtAPI):
    def executed_paddle_api(self, x):
        return x.rsqrt()

    def executed_numpy_api(self, x):
        return 1.0 / np.sqrt(x)


class TestDygraphInplaceRsqrtAPI(TestDygraphRsqrtAPI):
    def executed_paddle_api(self, x):
        return x.rsqrt_()


if __name__ == "__main__":
    unittest.main()
