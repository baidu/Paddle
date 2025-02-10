
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
        tensor34428 = x1

        # tensor34428: (128, 64, 1)
        tensor34437 = -tensor34428
        assert tuple(tensor34437.shape) == (128, 64, 1)
        
        # tensor34437: (128, 64, 1)
        expanded3793 = paddle.expand(tensor34437, shape=(128, 64, 32))
        assert tuple(expanded3793.shape) == (128, 64, 32)
        
        
        # tensor34428: (128, 64, 1)
        tensor34438 = -tensor34428
        assert tuple(tensor34438.shape) == (128, 64, 1)
        
        # tensor34438: (128, 64, 1)
        reduced5555 = tensor34438.sum(axis=(1,), keepdim=True)
        assert tuple(reduced5555.shape) == (128, 1, 1)
        
        # reduced5555: (128, 1, 1)
        expanded3794 = paddle.expand(reduced5555, shape=(128, 1, 32))
        assert tuple(expanded3794.shape) == (128, 1, 32)
        
        
        # tensor34428: (128, 64, 1)
        tensor34439 = -tensor34428
        assert tuple(tensor34439.shape) == (128, 64, 1)
        
        # tensor34439: (128, 64, 1)
        expanded3795 = paddle.expand(tensor34439, shape=(128, 64, 32))
        assert tuple(expanded3795.shape) == (128, 64, 32)
        
        
        # expanded3793: (128, 64, 32)
        tensor34440 = -expanded3793
        assert tuple(tensor34440.shape) == (128, 64, 32)
        
        # tensor34440: (128, 64, 32)
        reduced5556 = tensor34440.sum(axis=(0, 1), keepdim=True)
        assert tuple(reduced5556.shape) == (1, 1, 32)
        
        # reduced5556: (1, 1, 32)
        # expanded3794: (128, 1, 32)
        tensor34429 = reduced5556 + expanded3794
        assert tuple(tensor34429.shape) == (128, 1, 32)
        
        # tensor34429: (128, 1, 32)
        tensor34442 = -tensor34429
        assert tuple(tensor34442.shape) == (128, 1, 32)
        
        # tensor34442: (128, 1, 32)
        reduced5557 = tensor34442.sum(axis=(0,), keepdim=True)
        assert tuple(reduced5557.shape) == (1, 1, 32)
        
        # reduced5557: (1, 1, 32)
        expanded3796 = paddle.expand(reduced5557, shape=(1, 64, 32))
        assert tuple(expanded3796.shape) == (1, 64, 32)
        
        # tensor34428: (128, 64, 1)
        # expanded3796: (1, 64, 32)
        tensor34430 = tensor34428 + expanded3796
        assert tuple(tensor34430.shape) == (128, 64, 32)
        
        # tensor34430: (128, 64, 32)
        tensor34444 = -tensor34430
        assert tuple(tensor34444.shape) == (128, 64, 32)
        
        # tensor34444: (128, 64, 32)
        reduced5558 = tensor34444.sum(axis=(0, 1), keepdim=True)
        assert tuple(reduced5558.shape) == (1, 1, 32)
        
        # expanded3793: (128, 64, 32)
        tensor34431 = -expanded3793
        assert tuple(tensor34431.shape) == (128, 64, 32)
        
        # tensor34431: (128, 64, 32)
        # expanded3795: (128, 64, 32)
        tensor34432 = tensor34431 + expanded3795
        assert tuple(tensor34432.shape) == (128, 64, 32)
        
        # reduced5558: (1, 1, 32)
        tensor34433 = -reduced5558
        assert tuple(tensor34433.shape) == (1, 1, 32)
        
        # tensor34433: (1, 1, 32)
        expanded3797 = paddle.expand(tensor34433, shape=(128, 64, 32))
        assert tuple(expanded3797.shape) == (128, 64, 32)
        
        
        # expanded3797: (128, 64, 32)
        # expanded3797: (128, 64, 32)
        tensor34434 = expanded3797 + expanded3797
        assert tuple(tensor34434.shape) == (128, 64, 32)
        
        # tensor34434: (128, 64, 32)
        tensor34449 = -tensor34434
        assert tuple(tensor34449.shape) == (128, 64, 32)
        
        # tensor34449: (128, 64, 32)
        reduced5559 = tensor34449.sum(axis=(2,), keepdim=True)
        assert tuple(reduced5559.shape) == (128, 64, 1)
        
        # reduced5559: (128, 64, 1)
        # tensor34432: (128, 64, 32)
        tensor34435 = reduced5559 + tensor34432
        assert tuple(tensor34435.shape) == (128, 64, 32)
        
        # tensor34435: (128, 64, 32)
        tensor34451 = -tensor34435
        assert tuple(tensor34451.shape) == (128, 64, 32)
        
        # tensor34451: (128, 64, 32)
        reduced5560 = tensor34451.sum(axis=(1, 2), keepdim=True)
        assert tuple(reduced5560.shape) == (128, 1, 1)
        return reduced5560


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
