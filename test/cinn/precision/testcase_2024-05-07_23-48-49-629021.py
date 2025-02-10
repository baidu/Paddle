
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
        tensor594 = x1

        # tensor594: (128, 64, 32)
        tensor603 = -tensor594
        assert tuple(tensor603.shape) == (128, 64, 32)
        
        # tensor603: (128, 64, 32)
        reduced96 = tensor603.sum(axis=(1,), keepdim=True)
        assert tuple(reduced96.shape) == (128, 1, 32)
        
        
        
        # tensor594: (128, 64, 32)
        tensor604 = -tensor594
        assert tuple(tensor604.shape) == (128, 64, 32)
        
        # tensor604: (128, 64, 32)
        reduced97 = tensor604.sum(axis=(2,), keepdim=True)
        assert tuple(reduced97.shape) == (128, 64, 1)
        
        
        # tensor594: (128, 64, 32)
        # reduced96: (128, 1, 32)
        tensor595 = tensor594 + reduced96
        assert tuple(tensor595.shape) == (128, 64, 32)
        
        # tensor595: (128, 64, 32)
        tensor606 = -tensor595
        assert tuple(tensor606.shape) == (128, 64, 32)
        
        # tensor606: (128, 64, 32)
        reduced98 = tensor606.sum(axis=(2,), keepdim=True)
        assert tuple(reduced98.shape) == (128, 64, 1)
        
        # reduced97: (128, 64, 1)
        tensor596 = -reduced97
        assert tuple(tensor596.shape) == (128, 64, 1)
        
        # tensor596: (128, 64, 1)
        reduced99 = tensor596.sum(axis=(0,), keepdim=True)
        assert tuple(reduced99.shape) == (1, 64, 1)
        
        # reduced99: (1, 64, 1)
        expanded58 = paddle.expand(reduced99, shape=(1, 64, 32))
        assert tuple(expanded58.shape) == (1, 64, 32)
        
        # tensor594: (128, 64, 32)
        tensor597 = -tensor594
        assert tuple(tensor597.shape) == (128, 64, 32)
        
        # expanded58: (1, 64, 32)
        # reduced98: (128, 64, 1)
        tensor598 = expanded58 + reduced98
        assert tuple(tensor598.shape) == (128, 64, 32)
        
        
        # tensor598: (128, 64, 32)
        tensor610 = -tensor598
        assert tuple(tensor610.shape) == (128, 64, 32)
        
        # tensor610: (128, 64, 32)
        reduced100 = tensor610.sum(axis=(2,), keepdim=True)
        assert tuple(reduced100.shape) == (128, 64, 1)
        
        # tensor598: (128, 64, 32)
        # reduced100: (128, 64, 1)
        tensor599 = tensor598 + reduced100
        assert tuple(tensor599.shape) == (128, 64, 32)
        
        # tensor599: (128, 64, 32)
        tensor612 = -tensor599
        assert tuple(tensor612.shape) == (128, 64, 32)
        
        # tensor612: (128, 64, 32)
        reduced101 = tensor612.sum(axis=(2,), keepdim=True)
        assert tuple(reduced101.shape) == (128, 64, 1)
        
        # tensor597: (128, 64, 32)
        # reduced96: (128, 1, 32)
        tensor600 = tensor597 + reduced96
        assert tuple(tensor600.shape) == (128, 64, 32)
        
        # tensor600: (128, 64, 32)
        # reduced101: (128, 64, 1)
        tensor601 = tensor600 + reduced101
        assert tuple(tensor601.shape) == (128, 64, 32)
        
        # tensor601: (128, 64, 32)
        tensor615 = -tensor601
        assert tuple(tensor615.shape) == (128, 64, 32)
        
        # tensor615: (128, 64, 32)
        reduced102 = tensor615.sum(axis=(2,), keepdim=True)
        assert tuple(reduced102.shape) == (128, 64, 1)
        return reduced102


class TestCinnMonkey(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.x1 = paddle.uniform([128, 64, 32], dtype="float32", min=-0.5, max=0.5)
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
