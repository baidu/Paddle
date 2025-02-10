#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import scipy.linalg

sys.path.append("..")
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


def scipy_lu_solution(lu, pivots, b):
    pivots = pivots - 1
    return scipy.linalg.lu_solve((lu, pivots), b)


# Auxiliary function for broadcasting shapes
def broadcast_shape(matA, matB):
    shapeA = matA.shape
    shapeB = matB.shape
    broad_shape = []
    for idx in range(len(shapeA) - 2):
        if shapeA[idx] == shapeB[idx]:
            broad_shape.append(shapeA[idx])
        elif shapeA[idx] == 1 or shapeB[idx] == 1:
            broad_shape.append(max(shapeA[idx], shapeB[idx]))
        else:
            raise Exception(
                f'shapeA and shapeB cannot be broadcast: got {shapeA} and {shapeB}'
            )
    bsA = broad_shape + list(shapeA[-2:])
    bsB = broad_shape + list(shapeB[-2:])
    return np.broadcast_to(matA, bsA), np.broadcast_to(matB, bsB)


# Batch implementation of LU solve using LU factors
def scipy_lu_solution_batch(lu_batch, pivots_batch, b_batch):
    pivots_batch = pivots_batch - 1
    # Broadcast inputs if necessary
    lu_batch, b_batch = broadcast_shape(lu_batch, b_batch)
    ushape = lu_batch.shape
    bshape = b_batch.shape
    lu_batch = lu_batch.reshape((-1, ushape[-2], ushape[-1]))
    b_batch = b_batch.reshape((-1, bshape[-2], bshape[-1]))
    batch = 1
    for d in ushape[:-2]:
        batch *= d
    solutions = []
    for i in range(batch):
        sol = scipy.linalg.lu_solve((lu_batch[i], pivots_batch[i]), b_batch[i])
        solutions.append(sol)
    return np.array(solutions).reshape(bshape)


class TestLUSolveOp(OpTest):
    """
    Test case 1:
    A is a square matrix, and B is the corresponding right-hand side.
    """

    def config(self):
        self.A_shape = [15, 15]
        self.B_shape = [15, 5]
        self.dtype = np.float64

    def set_output(self):
        # LU and pivots are obtained from matrix A
        lu = self.inputs['lu']
        pivots = self.inputs['pivots']
        self.output = scipy_lu_solution(lu, pivots, self.inputs['b'])

    def setUp(self):
        self.op_type = "lu_solve"
        self.python_api = paddle.linalg.lu_solve
        self.config()
        # Generate a non-singular matrix A: random matrix plus identity matrix
        A = np.random.random(self.A_shape).astype(self.dtype) + np.eye(
            self.A_shape[0], dtype=self.dtype
        )
        # Compute LU factorization of A using scipy.linalg.lu_factor (returns LU and pivots)
        lu, pivots = scipy.linalg.lu_factor(A)
        b = np.random.random(self.B_shape).astype(self.dtype)
        self.inputs = {
            "lu": lu,
            "pivots": pivots,  # pivots is an integer array
            "b": b,
        }
        self.set_output()
        self.outputs = {"Out": self.output}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        # Only check gradients with respect to the right-hand side B,
        # because LU factors and pivots do not participate in backpropagation.
        self.check_grad(["b"], "out", max_relative_error=0.01, check_pir=True)


class TestLUSolveOpBroadcast(TestLUSolveOp):
    """
    Test case 2 (broadcast/batch scenario):
    A is a 3D tensor with a batch dimension; B also has the same batch dimension.
    """

    def config(self):
        self.A_shape = [2, 15, 15]
        self.B_shape = [2, 15, 5]
        self.dtype = np.float64

    def setUp(self):
        self.op_type = "lu_solve"
        self.python_api = paddle.linalg.lu_solve
        self.config()
        batch = self.A_shape[0]
        A_batch = np.random.random(self.A_shape).astype(self.dtype)
        # Add identity matrix to each batch slice to ensure non-singularity
        for i in range(batch):
            A_batch[i] += np.eye(self.A_shape[-1], dtype=self.dtype)
        # Compute LU factorization for each A in the batch
        lu_batch = []
        pivots_batch = []
        for i in range(batch):
            lu_i, pivots_i = scipy.linalg.lu_factor(A_batch[i])
            lu_batch.append(lu_i)
            pivots_batch.append(pivots_i)
        lu_batch = np.array(lu_batch)
        pivots_batch = np.array(pivots_batch)
        b = np.random.random(self.B_shape).astype(self.dtype)
        self.inputs = {
            "lu": lu_batch,
            "pivots": pivots_batch,
            "b": b,
        }
        # Compute reference result: call lu_solve for each batch individually
        outs = []
        for i in range(batch):
            outs.append(
                scipy.linalg.lu_solve((lu_batch[i], pivots_batch[i] - 1), b[i])
            )
        self.output = np.array(outs)
        self.outputs = {"Out": self.output}


class TestLUSolveAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2021)
        self.place = [paddle.CPUPlace()]
        self.dtype = "float64"
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # Define input data
            lu = paddle.static.data(name="lu", shape=[15, 15], dtype=self.dtype)
            pivots = paddle.static.data(
                name="pivots", shape=[15], dtype="int32"
            )
            b = paddle.static.data(name="b", shape=[15, 5], dtype=self.dtype)
            out = paddle.linalg.lu_solve(lu, pivots, b)

            # Generate matrix A and compute its LU factorization
            A = np.random.random([15, 15]).astype(self.dtype) + np.eye(
                15, dtype=self.dtype
            )
            lu, Pivots = scipy.linalg.lu_factor(A)
            b_np = np.random.random([15, 5]).astype(self.dtype)
            ref_out = scipy.linalg.lu_solve((lu, Pivots - 1), b_np)

            exe = base.Executor(place)
            fetches = exe.run(
                feed={"lu": lu, "pivots": Pivots.astype("int32"), "b": b_np},
                fetch_list=[out],
            )
            np.testing.assert_allclose(fetches[0], ref_out, rtol=1e-05)

    def test_static(self):
        for place in self.place:
            self.check_static_result(place=place)

    def test_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            A = np.random.random([15, 15]).astype(self.dtype) + np.eye(
                15, dtype=self.dtype
            )
            lu, Pivots = scipy.linalg.lu_factor(A)
            b_np = np.random.random([15, 5]).astype(self.dtype)
            ref_out = scipy.linalg.lu_solve((lu, Pivots - 1), b_np)
            lu_tensor = paddle.to_tensor(lu)
            pivots_tensor = paddle.to_tensor(Pivots.astype("int32"))
            b_tensor = paddle.to_tensor(b_np)
            out = paddle.linalg.lu_solve(lu_tensor, pivots_tensor, b_tensor)
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-05)
            self.assertEqual(out.numpy().shape, ref_out.shape)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_broadcast(self):
        def run(place):
            paddle.disable_static(place)
            # Construct a broadcast scenario: A with shape [1, 15, 15] and B with shape [2, 15, 5]
            A = np.random.random([1, 15, 15]).astype(self.dtype) + np.eye(
                15, dtype=self.dtype
            )
            # Expand A to simulate broadcasting for 2 batches
            lu_list = []
            pivots_list = []
            for i in range(2):
                LU_i, pivots_i = scipy.linalg.lu_factor(A[0])
                lu_list.append(LU_i)
                pivots_list.append(pivots_i)
            LU_np = np.array(lu_list)
            Pivots_np = np.array(pivots_list)
            b_np = np.random.random([2, 15, 5]).astype(self.dtype)

            ref_out = []
            for i in range(2):
                ref_out.append(
                    scipy.linalg.lu_solve((LU_np[i], Pivots_np[i] - 1), b_np[i])
                )
            ref_out = np.array(ref_out)

            # Create a tensor for A and broadcast it to shape [2, 15, 15]
            lu_tensor = paddle.to_tensor(A)
            lu_tensor = paddle.broadcast_to(lu_tensor, [2, 15, 15])
            pivots_tensor = paddle.to_tensor(Pivots_np)
            b_tensor = paddle.to_tensor(b_np)
            out = paddle.linalg.lu_solve(lu_tensor, pivots_tensor, b_tensor)
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-05)

        for place in self.place:
            run(place)


class TestLUSolveOpError(unittest.TestCase):
    def test_errors_1(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type for lu_solve must be a Variable. Passing a LoDTensor should raise an error.
            x1 = base.create_lod_tensor(np.array([[1]]), [[1]], base.CPUPlace())
            self.assertRaises(TypeError, paddle.linalg.lu_solve, x1, x1, x1)

    def test_errors_2(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # Data type check: lu_solve requires the inputs to be float32 or float64.
            # The pivots tensor must have dtype int32 or int64, not bool or float16.
            lu_int = paddle.static.data(
                name="lu_int", shape=[15, 15], dtype="int32"
            )
            pivots_int = paddle.static.data(
                name="pivots_int", shape=[15], dtype="int32"
            )
            b_int = paddle.static.data(
                name="b_int", shape=[15, 5], dtype="int32"
            )
            self.assertRaises(
                TypeError, paddle.linalg.lu_solve, lu_int, pivots_int, b_int
            )

            lu_bool = paddle.static.data(
                name="lu_bool", shape=[15, 15], dtype="bool"
            )
            pivots_bool = paddle.static.data(
                name="pivots_bool", shape=[15], dtype="int32"
            )
            b_bool = paddle.static.data(
                name="b_bool", shape=[15, 5], dtype="bool"
            )
            self.assertRaises(
                TypeError, paddle.linalg.lu_solve, lu_bool, pivots_bool, b_bool
            )

            lu_f16 = paddle.static.data(
                name="lu_f16", shape=[15, 15], dtype="float16"
            )
            pivots_f16 = paddle.static.data(
                name="pivots_f16", shape=[15], dtype="int32"
            )
            b_f16 = paddle.static.data(
                name="b_f16", shape=[15, 5], dtype="float16"
            )
            self.assertRaises(
                TypeError, paddle.linalg.lu_solve, lu_f16, pivots_f16, b_f16
            )

            # Check if the shapes of the tensors match: LU must be square.
            lu_wrong = paddle.static.data(
                name="lu_wrong", shape=[15, 10], dtype="float64"
            )
            pivots_wrong = paddle.static.data(
                name="pivots_wrong", shape=[15], dtype="int32"
            )
            b_wrong = paddle.static.data(
                name="b_wrong", shape=[15, 5], dtype="float64"
            )
            self.assertRaises(
                ValueError,
                paddle.linalg.lu_solve,
                lu_wrong,
                pivots_wrong,
                b_wrong,
            )

            # Check that the number of rows in B matches the dimensions of LU.
            lu_15 = paddle.static.data(
                name="lu_15", shape=[15, 15], dtype="float64"
            )
            pivots_15 = paddle.static.data(
                name="pivots_15", shape=[15], dtype="int32"
            )
            b_wrong2 = paddle.static.data(
                name="b_wrong2", shape=[10, 5], dtype="float64"
            )
            self.assertRaises(
                ValueError, paddle.linalg.lu_solve, lu_15, pivots_15, b_wrong2
            )

    def test_errors_3(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The shape of the pivots tensor must match the first dimension of LU.
            lu_wrong = paddle.static.data(
                name="lu_wrong", shape=[15, 15], dtype="float64"
            )
            pivots_wrong = paddle.static.data(
                name="pivots_wrong", shape=[5], dtype="int32"
            )
            b_wrong = paddle.static.data(
                name="b_wrong", shape=[15, 5], dtype="float64"
            )
            self.assertRaises(
                ValueError,
                paddle.linalg.lu_solve,
                lu_wrong,
                pivots_wrong,
                b_wrong,
            )


if __name__ == "__main__":
    unittest.main()
