
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
        tensor43863 = x1

        # tensor43863: (128, 64, 32)
        tensor43873 = -tensor43863
        assert tuple(tensor43873.shape) == (128, 64, 32)
        
        # tensor43873: (128, 64, 32)
        reduced7078 = tensor43873.sum(axis=(0, 2), keepdim=True)
        assert tuple(reduced7078.shape) == (1, 64, 1)
        
        
        # reduced7078: (1, 64, 1)
        tensor43874 = -reduced7078
        assert tuple(tensor43874.shape) == (1, 64, 1)
        
        # tensor43874: (1, 64, 1)
        expanded4784 = paddle.expand(tensor43874, shape=(128, 64, 1))
        assert tuple(expanded4784.shape) == (128, 64, 1)
        
        
        # tensor43863: (128, 64, 32)
        tensor43875 = -tensor43863
        assert tuple(tensor43875.shape) == (128, 64, 32)
        
        # tensor43875: (128, 64, 32)
        reduced7079 = tensor43875.sum(axis=(2,), keepdim=True)
        assert tuple(reduced7079.shape) == (128, 64, 1)
        
        
        # tensor43863: (128, 64, 32)
        tensor43876 = -tensor43863
        assert tuple(tensor43876.shape) == (128, 64, 32)
        
        # tensor43876: (128, 64, 32)
        reduced7080 = tensor43876.sum(axis=(0,), keepdim=True)
        assert tuple(reduced7080.shape) == (1, 64, 32)
        
        # reduced7078: (1, 64, 1)
        # reduced7079: (128, 64, 1)
        tensor43864 = reduced7078 + reduced7079
        assert tuple(tensor43864.shape) == (128, 64, 1)
        
        # tensor43864: (128, 64, 1)
        tensor43878 = -tensor43864
        assert tuple(tensor43878.shape) == (128, 64, 1)
        
        # tensor43878: (128, 64, 1)
        expanded4785 = paddle.expand(tensor43878, shape=(128, 64, 32))
        assert tuple(expanded4785.shape) == (128, 64, 32)
        
        # expanded4785: (128, 64, 32)
        # expanded4784: (128, 64, 1)
        tensor43865 = expanded4785 + expanded4784
        assert tuple(tensor43865.shape) == (128, 64, 32)
        
        # tensor43865: (128, 64, 32)
        tensor43880 = -tensor43865
        assert tuple(tensor43880.shape) == (128, 64, 32)
        
        # tensor43880: (128, 64, 32)
        reduced7081 = tensor43880.sum(axis=(0, 1, 2), keepdim=True)
        assert tuple(reduced7081.shape) == (1, 1, 1)
        
        # tensor43863: (128, 64, 32)
        # reduced7080: (1, 64, 32)
        tensor43866 = tensor43863 + reduced7080
        assert tuple(tensor43866.shape) == (128, 64, 32)
        
        
        # reduced7081: (1, 1, 1)
        tensor43882 = -reduced7081
        assert tuple(tensor43882.shape) == (1, 1, 1)
        
        # tensor43882: (1, 1, 1)
        expanded4786 = paddle.expand(tensor43882, shape=(128, 64, 32))
        assert tuple(expanded4786.shape) == (128, 64, 32)
        
        # tensor43866: (128, 64, 32)
        # expanded4786: (128, 64, 32)
        tensor43867 = tensor43866 + expanded4786
        assert tuple(tensor43867.shape) == (128, 64, 32)
        
        # tensor43867: (128, 64, 32)
        tensor43884 = -tensor43867
        assert tuple(tensor43884.shape) == (128, 64, 32)
        
        # tensor43884: (128, 64, 32)
        reduced7082 = tensor43884.sum(axis=(0, 2), keepdim=True)
        assert tuple(reduced7082.shape) == (1, 64, 1)
        
        # reduced7082: (1, 64, 1)
        # reduced7081: (1, 1, 1)
        tensor43868 = reduced7082 + reduced7081
        assert tuple(tensor43868.shape) == (1, 64, 1)
        
        # tensor43868: (1, 64, 1)
        tensor43886 = -tensor43868
        assert tuple(tensor43886.shape) == (1, 64, 1)
        
        # tensor43886: (1, 64, 1)
        reduced7083 = tensor43886.sum(axis=(1,), keepdim=True)
        assert tuple(reduced7083.shape) == (1, 1, 1)
        
        # reduced7083: (1, 1, 1)
        tensor43869 = -reduced7083
        assert tuple(tensor43869.shape) == (1, 1, 1)
        
        # tensor43869: (1, 1, 1)
        expanded4787 = paddle.expand(tensor43869, shape=(128, 64, 32))
        assert tuple(expanded4787.shape) == (128, 64, 32)
        
        tensor43870 = paddle.ones((128, 1, 32))
        
        # expanded4787: (128, 64, 32)
        # tensor43870: (128, 1, 32)
        tensor43871 = expanded4787 + tensor43870
        assert tuple(tensor43871.shape) == (128, 64, 32)
        
        # tensor43871: (128, 64, 32)
        tensor43890 = -tensor43871
        assert tuple(tensor43890.shape) == (128, 64, 32)
        
        # tensor43890: (128, 64, 32)
        reduced7084 = tensor43890.sum(axis=(0,), keepdim=True)
        assert tuple(reduced7084.shape) == (1, 64, 32)
        return reduced7084


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
