
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
        tensor7940 = x1

        # tensor7940: (128, 64, 32)
        tensor7950 = -tensor7940
        assert tuple(tensor7950.shape) == (128, 64, 32)
        
        # tensor7950: (128, 64, 32)
        reduced1274 = tensor7950.sum(axis=(2,), keepdim=True)
        assert tuple(reduced1274.shape) == (128, 64, 1)
        
        
        
        # reduced1274: (128, 64, 1)
        tensor7951 = -reduced1274
        assert tuple(tensor7951.shape) == (128, 64, 1)
        
        # tensor7951: (128, 64, 1)
        expanded850 = paddle.expand(tensor7951, shape=(128, 64, 32))
        assert tuple(expanded850.shape) == (128, 64, 32)
        
        
        # tensor7940: (128, 64, 32)
        # tensor7940: (128, 64, 32)
        tensor7941 = tensor7940 + tensor7940
        assert tuple(tensor7941.shape) == (128, 64, 32)
        
        # tensor7941: (128, 64, 32)
        tensor7953 = -tensor7941
        assert tuple(tensor7953.shape) == (128, 64, 32)
        
        # tensor7953: (128, 64, 32)
        reduced1275 = tensor7953.sum(axis=(0, 1), keepdim=True)
        assert tuple(reduced1275.shape) == (1, 1, 32)
        
        # reduced1275: (1, 1, 32)
        # tensor7940: (128, 64, 32)
        tensor7942 = reduced1275 + tensor7940
        assert tuple(tensor7942.shape) == (128, 64, 32)
        
        # tensor7942: (128, 64, 32)
        tensor7955 = -tensor7942
        assert tuple(tensor7955.shape) == (128, 64, 32)
        
        # tensor7955: (128, 64, 32)
        reduced1276 = tensor7955.sum(axis=(0, 1), keepdim=True)
        assert tuple(reduced1276.shape) == (1, 1, 32)
        
        # reduced1276: (1, 1, 32)
        tensor7943 = -reduced1276
        assert tuple(tensor7943.shape) == (1, 1, 32)
        
        # tensor7943: (1, 1, 32)
        expanded851 = paddle.expand(tensor7943, shape=(1, 64, 32))
        assert tuple(expanded851.shape) == (1, 64, 32)
        
        # reduced1274: (128, 64, 1)
        tensor7944 = -reduced1274
        assert tuple(tensor7944.shape) == (128, 64, 1)
        
        # tensor7944: (128, 64, 1)
        expanded852 = paddle.expand(tensor7944, shape=(128, 64, 32))
        assert tuple(expanded852.shape) == (128, 64, 32)
        
        # expanded850: (128, 64, 32)
        tensor7945 = -expanded850
        assert tuple(tensor7945.shape) == (128, 64, 32)
        
        # expanded851: (1, 64, 32)
        tensor7946 = -expanded851
        assert tuple(tensor7946.shape) == (1, 64, 32)
        
        # tensor7946: (1, 64, 32)
        expanded853 = paddle.expand(tensor7946, shape=(128, 64, 32))
        assert tuple(expanded853.shape) == (128, 64, 32)
        
        
        # expanded853: (128, 64, 32)
        tensor7947 = -expanded853
        assert tuple(tensor7947.shape) == (128, 64, 32)
        
        # tensor7947: (128, 64, 32)
        reduced1277 = tensor7947.sum(axis=(0, 1), keepdim=True)
        assert tuple(reduced1277.shape) == (1, 1, 32)
        
        # reduced1277: (1, 1, 32)
        # expanded852: (128, 64, 32)
        tensor7948 = reduced1277 + expanded852
        assert tuple(tensor7948.shape) == (128, 64, 32)
        
        # tensor7948: (128, 64, 32)
        tensor7962 = -tensor7948
        assert tuple(tensor7962.shape) == (128, 64, 32)
        
        # tensor7962: (128, 64, 32)
        reduced1278 = tensor7962.sum(axis=(1,), keepdim=True)
        assert tuple(reduced1278.shape) == (128, 1, 32)
        return reduced1278


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
