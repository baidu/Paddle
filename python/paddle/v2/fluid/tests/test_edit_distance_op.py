import unittest
import numpy as np
from op_test import OpTest


def Levenshtein(hyp, ref):
    """ Compute the Levenshtein distance between two strings.

    :param hyp: hypothesis string in index
    :type hyp: list
    :param ref: reference string in index
    :type ref: list
    """
    m = len(hyp)
    n = len(ref)
    if m == 0:
        return n
    if n == 0:
        return m

    dist = np.zeros((m + 1, n + 1)).astype("float32")
    for i in range(0, m + 1):
        dist[i][0] = i
    for j in range(0, n + 1):
        dist[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            deletion = dist[i - 1][j] + 1
            insertion = dist[i][j - 1] + 1
            substitution = dist[i - 1][j - 1] + cost
            dist[i][j] = min(deletion, insertion, substitution)
    return dist[m][n]


class TestEditDistanceOp(OpTest):
    def setUp(self):
        self.op_type = "edit_distance"
        normalized = False
        x1 = np.array([[0, 12, 3, 5, 8, 2]]).astype("int32")
        x2 = np.array([[0, 12, 4, 7, 8]]).astype("int32")
        x1 = np.transpose(x1)
        x2 = np.transpose(x2)
        x1_lod = [0, 1, 5]
        x2_lod = [0, 3, 4]

        num_strs = len(x1_lod) - 1
        distance = np.zeros((num_strs, 1)).astype("float32")
        for i in range(0, num_strs):
            distance[i] = Levenshtein(
                hyp=x1[x1_lod[i]:x1_lod[i + 1]],
                ref=x2[x2_lod[i]:x2_lod[i + 1]])
            if normalized is True:
                len_ref = x2_lod[i + 1] - x2_lod[i]
                distance[i] = distance[i] / len_ref
        self.attrs = {'normalized': normalized}
        self.inputs = {'Hyps': (x1, [x1_lod]), 'Refs': (x2, [x2_lod])}
        self.outputs = {'Out': distance}

    def test_check_output(self):
        self.check_output()


class TestEditDistanceOpNormalized(OpTest):
    def setUp(self):
        self.op_type = "edit_distance"
        normalized = True
        x1 = np.array([[0, 10, 3, 6, 5, 8, 2]]).astype("int32")
        x2 = np.array([[0, 10, 4, 6, 7, 8]]).astype("int32")
        x1 = np.transpose(x1)
        x2 = np.transpose(x2)
        x1_lod = [0, 1, 3, 6]
        x2_lod = [0, 2, 3, 5]

        num_strs = len(x1_lod) - 1
        distance = np.zeros((num_strs, 1)).astype("float32")
        for i in range(0, num_strs):
            distance[i] = Levenshtein(
                hyp=x1[x1_lod[i]:x1_lod[i + 1]],
                ref=x2[x2_lod[i]:x2_lod[i + 1]])
            if normalized is True:
                len_ref = x2_lod[i + 1] - x2_lod[i]
                distance[i] = distance[i] / len_ref
        self.attrs = {'normalized': normalized}
        self.inputs = {'Hyps': (x1, [x1_lod]), 'Refs': (x2, [x2_lod])}
        self.outputs = {'Out': distance}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
