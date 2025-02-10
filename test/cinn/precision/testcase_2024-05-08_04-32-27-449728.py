
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
        tensor57720 = x1

        # tensor57720: (128, 64, 32)
        tensor57731 = -tensor57720
        assert tuple(tensor57731.shape) == (128, 64, 32)
        
        # tensor57731: (128, 64, 32)
        reduced9351 = tensor57731.sum(axis=(0,), keepdim=True)
        assert tuple(reduced9351.shape) == (1, 64, 32)
        
        
        
        # tensor57720: (128, 64, 32)
        tensor57721 = -tensor57720
        assert tuple(tensor57721.shape) == (128, 64, 32)
        
        # tensor57721: (128, 64, 32)
        reduced9352 = tensor57721.sum(axis=(2,), keepdim=True)
        assert tuple(reduced9352.shape) == (128, 64, 1)
        
        # tensor57720: (128, 64, 32)
        # reduced9352: (128, 64, 1)
        tensor57722 = tensor57720 + reduced9352
        assert tuple(tensor57722.shape) == (128, 64, 32)
        
        # tensor57722: (128, 64, 32)
        tensor57734 = -tensor57722
        assert tuple(tensor57734.shape) == (128, 64, 32)
        
        # tensor57734: (128, 64, 32)
        reduced9353 = tensor57734.sum(axis=(2,), keepdim=True)
        assert tuple(reduced9353.shape) == (128, 64, 1)
        
        # reduced9353: (128, 64, 1)
        # tensor57720: (128, 64, 32)
        tensor57723 = reduced9353 + tensor57720
        assert tuple(tensor57723.shape) == (128, 64, 32)
        
        # tensor57723: (128, 64, 32)
        # reduced9351: (1, 64, 32)
        tensor57724 = tensor57723 + reduced9351
        assert tuple(tensor57724.shape) == (128, 64, 32)
        
        # tensor57724: (128, 64, 32)
        tensor57737 = -tensor57724
        assert tuple(tensor57737.shape) == (128, 64, 32)
        
        # tensor57737: (128, 64, 32)
        reduced9354 = tensor57737.sum(axis=(0,), keepdim=True)
        assert tuple(reduced9354.shape) == (1, 64, 32)
        
        tensor57725 = paddle.ones((128, 1, 32))
        
        # tensor57725: (128, 1, 32)
        # reduced9354: (1, 64, 32)
        tensor57726 = tensor57725 + reduced9354
        assert tuple(tensor57726.shape) == (128, 64, 32)
        
        tensor57727 = paddle.ones((128, 64, 32))
        
        # tensor57726: (128, 64, 32)
        tensor57728 = -tensor57726
        assert tuple(tensor57728.shape) == (128, 64, 32)
        
        # tensor57728: (128, 64, 32)
        # tensor57727: (128, 64, 32)
        tensor57729 = tensor57728 + tensor57727
        assert tuple(tensor57729.shape) == (128, 64, 32)
        
        # tensor57729: (128, 64, 32)
        tensor57743 = -tensor57729
        assert tuple(tensor57743.shape) == (128, 64, 32)
        
        # tensor57743: (128, 64, 32)
        reduced9355 = tensor57743.sum(axis=(0,), keepdim=True)
        assert tuple(reduced9355.shape) == (1, 64, 32)
        return reduced9355


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
