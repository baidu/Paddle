
import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = '1'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'

import unittest
import numpy as np
import paddle


class CinnMonkeyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        tensor93540 = x1

        # tensor93540: (128, 64, 1)
        tensor93549 = -tensor93540
        assert tuple(tensor93549.shape) == (128, 64, 1)
        
        # tensor93549: (128, 64, 1)
        expanded10156 = paddle.expand(tensor93549, shape=(128, 64, 32))
        assert tuple(expanded10156.shape) == (128, 64, 32)
        
        
        # expanded10156: (128, 64, 32)
        tensor93541 = -expanded10156
        assert tuple(tensor93541.shape) == (128, 64, 32)
        
        tensor93542 = paddle.zeros((128, 64, 32))
        
        
        # tensor93540: (128, 64, 1)
        # tensor93541: (128, 64, 32)
        tensor93543 = tensor93540 + tensor93541
        assert tuple(tensor93543.shape) == (128, 64, 32)
        
        # tensor93543: (128, 64, 32)
        tensor93553 = -tensor93543
        assert tuple(tensor93553.shape) == (128, 64, 32)
        
        # tensor93553: (128, 64, 32)
        reduced15119 = tensor93553.sum(axis=(1,), keepdim=True)
        assert tuple(reduced15119.shape) == (128, 1, 32)
        
        # reduced15119: (128, 1, 32)
        tensor93544 = -reduced15119
        assert tuple(tensor93544.shape) == (128, 1, 32)
        
        # tensor93544: (128, 1, 32)
        expanded10157 = paddle.expand(tensor93544, shape=(128, 64, 32))
        assert tuple(expanded10157.shape) == (128, 64, 32)
        
        # expanded10157: (128, 64, 32)
        # expanded10156: (128, 64, 32)
        tensor93545 = expanded10157 + expanded10156
        assert tuple(tensor93545.shape) == (128, 64, 32)
        
        # tensor93545: (128, 64, 32)
        tensor93556 = -tensor93545
        assert tuple(tensor93556.shape) == (128, 64, 32)
        
        # tensor93556: (128, 64, 32)
        reduced15120 = tensor93556.sum(axis=(0, 1, 2), keepdim=True)
        assert tuple(reduced15120.shape) == (1, 1, 1)
        
        
        # reduced15120: (1, 1, 1)
        tensor93557 = -reduced15120
        assert tuple(tensor93557.shape) == (1, 1, 1)
        
        # tensor93557: (1, 1, 1)
        expanded10158 = paddle.expand(tensor93557, shape=(128, 64, 32))
        assert tuple(expanded10158.shape) == (128, 64, 32)
        
        # expanded10158: (128, 64, 32)
        tensor93546 = -expanded10158
        assert tuple(tensor93546.shape) == (128, 64, 32)
        
        # reduced15120: (1, 1, 1)
        # tensor93546: (128, 64, 32)
        tensor93547 = reduced15120 + tensor93546
        assert tuple(tensor93547.shape) == (128, 64, 32)
        return tensor93547


class TestCinnMonkey(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x1 = paddle.uniform([128, 64, 1], dtype="float32", min=-0.5, max=0.5)
        self.x1.stop_gradient = True

    def apply_to_static(self, net, use_cinn, input_spec=None):
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def train(self, use_cinn):
        net = CinnMonkeyNet()
        net.eval()
        net = self.apply_to_static(net, use_cinn)
        out = net(self.x1)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)
        dy_out = self.train(use_cinn=False)

        # np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-6)
        np.testing.assert_allclose(cinn_out.numpy(), dy_out.numpy(), atol=1e-1)


if __name__ == '__main__':
    unittest.main()
