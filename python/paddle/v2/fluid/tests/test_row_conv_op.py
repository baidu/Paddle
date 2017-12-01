import unittest
import numpy as np
from op_test import OpTest


def row_conv_forward(x, lod, wt):
    out = np.zeros_like(x)
    seq_info = lod[0]
    num_sequences = len(seq_info) - 1
    context_length = wt.shape[0]

    for i in range(num_sequences):  # loop over number of sequences
        start = seq_info[i]
        end = seq_info[i + 1]
        curinput = x[start:end, :]
        curoutput = out[start:end, :]

        cur_timesteps = end - start
        for j in range(cur_timesteps):  # loop over different timesteps
            for k in range(context_length):

                if j + k >= cur_timesteps:
                    continue
                curoutput[j, :] += curinput[j + k, :] * wt[k, :]

    return out


class TestRowConvOp1(OpTest):
    def setUp(self):

        self.op_type = "row_conv"
        lod = [[0, 2, 5, 7]]
        T = lod[0][-1]
        D = 16
        context_length = 2

        x = np.random.random((T, D)).astype("float32")
        wt = np.random.random((context_length, D)).astype("float32")
        self.inputs = {'X': (x, lod), 'Filter': wt}

        out = row_conv_forward(x, lod, wt)
        self.outputs = {'Out': (out, lod)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Filter'], 'Out', max_relative_error=0.05)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Filter'], 'Out', max_relative_error=0.05, no_grad_set=set('X'))

    def test_check_grad_ignore_wt(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Filter'))


class TestRowConvOp2(OpTest):
    def setUp(self):

        self.op_type = "row_conv"
        lod = [[0, 20, 50, 100]]
        T = lod[0][-1]
        D = 30
        context_length = 35

        x = np.random.random((T, D)).astype("float32")
        wt = np.random.random((context_length, D)).astype("float32")
        self.inputs = {'X': (x, lod), 'Filter': wt}

        out = row_conv_forward(x, lod, wt)
        self.outputs = {'Out': (out, lod)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Filter'], 'Out', max_relative_error=0.05)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Filter'], 'Out', max_relative_error=0.05, no_grad_set=set('X'))

    def test_check_grad_ignore_wt(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Filter'))


if __name__ == '__main__':
    unittest.main()
