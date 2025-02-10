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

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

import paddle.vision.transforms as T
from paddle.dataset.common import _check_exists_and_download
from paddle.vision.datasets import (
    MNIST,
    DatasetFolder,
    FashionMNIST,
    Flowers,
    ImageFolder,
)


class TestFolderDatasets(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.empty_dir = tempfile.mkdtemp()
        for i in range(2):
            sub_dir = os.path.join(self.data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                fake_img = (np.random.random((32, 32, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_dataset(self):
        dataset_folder = DatasetFolder(self.data_dir)

        for _ in dataset_folder:
            pass

        assert len(dataset_folder) == 4
        assert len(dataset_folder.classes) == 2

        dataset_folder = DatasetFolder(self.data_dir)
        for _ in dataset_folder:
            pass

    def test_folder(self):
        loader = ImageFolder(self.data_dir)

        for _ in loader:
            pass

        loader = ImageFolder(self.data_dir)
        for _ in loader:
            pass

        assert len(loader) == 4

    def test_transform(self):
        def fake_transform(img):
            return img

        transform = fake_transform
        dataset_folder = DatasetFolder(self.data_dir, transform=transform)

        for _ in dataset_folder:
            pass

        loader = ImageFolder(self.data_dir, transform=transform)
        for _ in loader:
            pass

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            ImageFolder(self.empty_dir)
        with self.assertRaises(RuntimeError):
            DatasetFolder(self.empty_dir)

        with self.assertRaises(ValueError):
            _check_exists_and_download('temp_paddle', None, None, None, False)


class TestMNISTTest(unittest.TestCase):
    def test_main(self):
        transform = T.Transpose()
        mnist = MNIST(mode='test', transform=transform)
        self.assertTrue(len(mnist) == 10000)

        i = np.random.randint(0, len(mnist) - 1)
        image, label = mnist[i]
        self.assertTrue(image.shape[0] == 1)
        self.assertTrue(image.shape[1] == 28)
        self.assertTrue(image.shape[2] == 28)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 9)


class TestMNISTTrain(unittest.TestCase):
    def test_main(self):
        transform = T.Transpose()
        mnist = MNIST(mode='train', transform=transform)
        self.assertTrue(len(mnist) == 60000)

        i = np.random.randint(0, len(mnist) - 1)
        image, label = mnist[i]
        self.assertTrue(image.shape[0] == 1)
        self.assertTrue(image.shape[1] == 28)
        self.assertTrue(image.shape[2] == 28)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 9)

        # test cv2 backend
        mnist = MNIST(mode='train', transform=transform, backend='cv2')
        self.assertTrue(len(mnist) == 60000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 1)
            self.assertTrue(image.shape[1] == 28)
            self.assertTrue(image.shape[2] == 28)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)
            break

        with self.assertRaises(ValueError):
            mnist = MNIST(mode='train', transform=transform, backend=1)


class TestFASHIONMNISTTest(unittest.TestCase):
    def test_main(self):
        transform = T.Transpose()
        mnist = FashionMNIST(mode='test', transform=transform)
        self.assertTrue(len(mnist) == 10000)

        i = np.random.randint(0, len(mnist) - 1)
        image, label = mnist[i]
        self.assertTrue(image.shape[0] == 1)
        self.assertTrue(image.shape[1] == 28)
        self.assertTrue(image.shape[2] == 28)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 9)


class TestFASHIONMNISTTrain(unittest.TestCase):
    def test_main(self):
        transform = T.Transpose()
        mnist = FashionMNIST(mode='train', transform=transform)
        self.assertTrue(len(mnist) == 60000)

        i = np.random.randint(0, len(mnist) - 1)
        image, label = mnist[i]
        self.assertTrue(image.shape[0] == 1)
        self.assertTrue(image.shape[1] == 28)
        self.assertTrue(image.shape[2] == 28)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 9)

        # test cv2 backend
        mnist = FashionMNIST(mode='train', transform=transform, backend='cv2')
        self.assertTrue(len(mnist) == 60000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 1)
            self.assertTrue(image.shape[1] == 28)
            self.assertTrue(image.shape[2] == 28)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)
            break

        with self.assertRaises(ValueError):
            mnist = FashionMNIST(mode='train', transform=transform, backend=1)

    def test_dataset_value(self):
        fmnist = FashionMNIST(mode='train')
        value = np.mean([np.array(x[0]) for x in fmnist])

        # 72.94035223214286 was getted from competitive products
        np.testing.assert_allclose(value, 72.94035223214286)


class TestFlowersTrain(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='train')
        self.assertTrue(len(flowers) == 6149)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 6149)
        image, label = flowers[idx]
        image = np.array(image)
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersValid(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='valid')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        image = np.array(image)
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersTest(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='test')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        image = np.array(image)
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)

        # test cv2 backend
        flowers = Flowers(mode='test', backend='cv2')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]

        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)

        with self.assertRaises(ValueError):
            flowers = Flowers(mode='test', backend=1)


if __name__ == '__main__':
    unittest.main()
