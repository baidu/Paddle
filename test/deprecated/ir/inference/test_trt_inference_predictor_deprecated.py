#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import sys
import tempfile
import unittest

import numpy as np
import yaml

import paddle
from paddle import nn

try:
    import paddle.inference as paddle_infer
except Exception as e:
    sys.stderr.write("Cannot import paddle, maybe paddle is not installed.\n")

paddle.set_device('cpu')
paddle.disable_signal_handler()


def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False


def getdtype(dtype="float32"):
    if dtype == "float32" or dtype == "float":
        return np.float32
    if dtype == "float16":
        return np.float16
    if dtype == "float64":
        return np.float64
    if dtype == "int32":
        return np.int32
    if dtype == "int64":
        return np.int64


class BackendPaddle:
    def __init__(self):
        super().__init__()
        self.h2d_time = []
        self.compute_time = []
        self.d2h_time = []

    def version(self):
        return paddle.version.full_version

    def name(self):
        return "paddle"

    def load(self, config_arg, inputs=None, outputs=None):
        self.args = config_arg
        if os.path.exists(self.args.model_dir):
            model_file = os.path.join(
                self.args.model_dir + "/" + self.args.paddle_model_file
            )
            model_params = os.path.join(
                self.args.model_dir + "/" + self.args.paddle_params_file
            )
            config = paddle_infer.Config(model_file, model_params)
        else:
            raise ValueError(
                f"The model dir {self.args.model_dir} does not exists!"
            )

        # enable memory optim
        if not self.args.enable_tune:
            config.enable_memory_optim()

        config.set_cpu_math_library_num_threads(self.args.cpu_threads)
        config.switch_ir_optim(True)
        # debug
        if self.args.enable_debug:
            config.switch_ir_debug()
        precision_mode = paddle_infer.PrecisionType.Float32
        if self.args.precision == 'fp16':
            precision_mode = paddle_infer.PrecisionType.Half
        elif self.args.precision == 'int8':
            precision_mode = paddle_infer.PrecisionType.Int8

        if self.args.enable_mkldnn and not self.args.enable_gpu:
            config.disable_gpu()
            config.enable_mkldnn()
            if self.args.precision == 'int8':
                config.enable_mkldnn_int8(
                    {"conv2d", "depthwise_conv2d", "transpose2", "pool2d"}
                )
        if not self.args.enable_mkldnn and not self.args.enable_gpu:
            config.disable_gpu()
            # config.enable_mkldnn()
        if self.args.enable_profile:
            config.enable_profile()
        shape_range_file = os.path.join(
            self.args.model_dir, self.args.shape_range_file
        )
        if self.args.enable_tune:
            config.collect_shape_range_info(shape_range_file)
        if self.args.enable_gpu:
            config.enable_use_gpu(256, self.args.gpu_id)
            if self.args.enable_trt:
                max_batch_size = self.args.batch_size
                if (
                    self.args.yaml_config["input_shape"]["0"]["shape"][
                        self.args.test_num
                    ][0]
                    != -1
                ):
                    max_batch_size = self.args.yaml_config["input_shape"]["0"][
                        "shape"
                    ][self.args.test_num][0]
                config.enable_tensorrt_engine(
                    workspace_size=1 << 25,
                    precision_mode=precision_mode,
                    max_batch_size=max_batch_size,
                    min_subgraph_size=self.args.subgraph_size,
                    use_static=False,
                    use_calib_mode=(
                        False if self.args.precision == 'int8' else False
                    ),
                )
                if self.args.enable_dynamic_shape:
                    if os.path.exists(shape_range_file):
                        config.enable_tuned_tensorrt_dynamic_shape(
                            shape_range_file, True
                        )
        config.disable_glog_info()
        config.exp_disable_tensorrt_ops(["range"])

        self.predictor = paddle_infer.create_predictor(config)

        input_shape = self.args.yaml_config["input_shape"]
        if len(input_shape) <= 0:
            raise Exception("input shape is empty.")

        if "input_data" in self.args.yaml_config:
            input_file = self.args.yaml_config["input_data"]["data"][
                self.args.test_num
            ]
            self.numpy_input = np.load(input_file, allow_pickle=True)

        return self

    def set_input(self):
        # set input tensor
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            if "input_data" not in self.args.yaml_config:
                if (
                    self.args.yaml_config["input_shape"][str(i)]["shape"][
                        self.args.test_num
                    ][0]
                    == -1
                ):
                    input_shape = [
                        self.args.batch_size
                    ] + self.args.yaml_config["input_shape"][str(i)]["shape"][
                        self.args.test_num
                    ][
                        1:
                    ]
                    dtype = self.args.yaml_config["input_shape"][str(i)][
                        "dtype"
                    ][self.args.test_num]
                else:
                    input_shape = self.args.yaml_config["input_shape"][str(i)][
                        "shape"
                    ][self.args.test_num]
                    dtype = self.args.yaml_config["input_shape"][str(i)][
                        "dtype"
                    ][self.args.test_num]
                if hasattr(self.args, "test_data"):
                    fake_input = self.args.test_data[i].astype(getdtype(dtype))
                else:
                    fake_input = np.ones(input_shape, dtype=getdtype(dtype))
                input_tensor.copy_from_cpu(fake_input)
            else:
                real_input = np.expand_dims(self.numpy_input[i], 0).repeat(
                    self.args.batch_size, axis=0
                )
                input_tensor.copy_from_cpu(real_input)

    def set_output(self):
        results = []
        # get out data from output tensor
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            if self.args.return_result or self.args.save_result:
                results.append(output_data)
        if self.args.return_result or self.args.save_result:
            return results

    def reset(self):
        self.h2d_time.clear()
        self.d2h_time.clear()
        self.compute_time.clear()

    def warmup(self):
        pass

    def predict(self, feed=None):
        self.set_input()
        self.predictor.run()
        output = self.set_output()
        if self.args.return_result or self.args.save_result:
            return output

    def predict_nocopy(self, feed=None):
        self.predictor.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--inter_op_threads', type=int, default=1)
    parser.add_argument(
        '--precision', type=str, choices=["fp32", "fp16", "int8"]
    )
    parser.add_argument(
        '--backend_type',
        type=str,
        choices=["paddle", "onnxruntime", "openvino", "tensorrt"],
        default="paddle",
    )
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--subgraph_size', type=int, default=1)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument(
        '--paddle_model_file', type=str, default="model.pdmodel"
    )
    parser.add_argument(
        '--paddle_params_file', type=str, default="model.pdiparams"
    )
    parser.add_argument('--enable_mkldnn', type=str2bool, default=False)
    parser.add_argument('--enable_gpu', type=str2bool, default=True)
    parser.add_argument('--enable_trt', type=str2bool, default=True)
    parser.add_argument('--enable_dynamic_shape', type=str2bool, default=True)
    parser.add_argument('--enable_tune', type=str2bool, default=False)
    parser.add_argument('--enable_profile', type=str2bool, default=False)
    parser.add_argument('--enable_benchmark', type=str2bool, default=True)
    parser.add_argument('--save_result', type=str2bool, default=False)
    parser.add_argument('--return_result', type=str2bool, default=False)
    parser.add_argument('--enable_debug', type=str2bool, default=False)
    parser.add_argument(
        '--config_file', type=str, required=False, default="config/model.yaml"
    )
    parser.add_argument(
        '--shape_range_file', type=str, default="shape_range.pbtxt"
    )
    args, unknown = parser.parse_known_args()
    return args


def run_infer(model_path):
    conf = parse_args()

    yaml_config = yaml.safe_load(
        '''
    input_shape:
      '0':
        dtype: [float32]
        shape:
        - [-1, 3, 32, 32]
    '''
    )

    conf.yaml_config = yaml_config
    conf.test_num = 0
    conf.model_dir = model_path

    conf.enable_tune = True
    # collect shape use CPU
    conf.enable_gpu = False
    backend = BackendPaddle()
    backend.load(conf)
    backend.predict()

    # collect shape use GPU
    conf.enable_gpu = True
    backend = BackendPaddle()
    backend.load(conf)
    backend.predict()

    # run inference predictor
    conf.enable_tune = False
    backend = BackendPaddle()
    backend.load(conf)
    backend.predict()


class ConvBNLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
    ):
        super().__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False,
        )

        self._batch_norm = paddle.nn.BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class Test(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=3, stride=2, act='relu'
        )
        self.pool2d_max = paddle.nn.MaxPool2D(
            kernel_size=3, stride=1, padding=1
        )
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool2d_avg(x)

        x = paddle.reshape(
            x,
            shape=[
                paddle.to_tensor([-1], dtype=paddle.int64),
                paddle.to_tensor([8], dtype=paddle.int64),
            ],
        )
        return x


class TestInferencePredictor(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, './inference/model')
        self.path = "./inference/model"

    def tearDown(self):
        self.temp_dir.cleanup()

    def SaveInferenceModel(self):
        paddle.disable_static()
        net = Test()
        net.eval()

        net(paddle.rand(shape=[1, 3, 32, 32], dtype='float32'))
        input_spec = [
            paddle.static.InputSpec(
                shape=[-1, 3, 32, 32], dtype=paddle.float32, name='input'
            )
        ]

        static_model = paddle.jit.to_static(
            net, input_spec=input_spec, full_graph=True
        )
        paddle.jit.save(static_model, self.path)

    def testInferencePredictor(self):
        self.SaveInferenceModel()
        run_infer(os.path.dirname(self.path))


if __name__ == '__main__':
    unittest.main()
