# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
import unittest

import numpy as np

import paddle
from paddle.dataset.common import DATA_HOME, download
from paddle.static.quantization import WeightQuantization

paddle.enable_static()


def _load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())


def _set_variable_data(scope, place, var_name, np_value):
    '''
    Set the value of var node by name, if the node exits,
    '''
    assert isinstance(
        np_value, np.ndarray
    ), 'The type of value should be numpy array.'
    var_node = scope.find_var(var_name)
    if var_node is not None:
        tensor = var_node.get_tensor()
        tensor.set(np_value, place)


class TestWeightQuantization(unittest.TestCase):
    def setUp(self):
        self.weight_quantization_dir = 'weight_quantization'
        self.cache_folder = os.path.join(
            DATA_HOME, self.weight_quantization_dir
        )

    def download_model(self, model_name, data_url, data_md5):
        download(data_url, self.weight_quantization_dir, data_md5)
        file_name = data_url.split('/')[-1]
        file_path = os.path.join(self.cache_folder, file_name)
        print(model_name + ' is downloaded at ' + file_path)

        unzipped_path = os.path.join(self.cache_folder, model_name)
        self.cache_unzipping(unzipped_path, file_path)
        print(model_name + ' is unzipped at ' + unzipped_path)
        return unzipped_path

    def cache_unzipping(self, target_folder, zip_path):
        if not os.path.exists(target_folder):
            cmd = (
                f'mkdir {target_folder} && tar xf {zip_path} -C {target_folder}'
            )
            os.system(cmd)

    def quantize_to_int(
        self,
        model_name,
        model_filename,
        params_filename,
        model_data_url,
        model_data_md5,
        weight_bits,
        quantizable_op_type,
        weight_quantize_type,
        generate_test_model,
        threshold_rate,
    ):
        model_dir = self.download_model(
            model_name, model_data_url, model_data_md5
        )
        load_model_dir = os.path.join(model_dir, model_name)

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        save_model_dir = os.path.join(
            os.getcwd(),
            model_name + "_wq_" + str(weight_bits) + "_" + timestamp,
        )

        weight_quant = WeightQuantization(
            model_dir=load_model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
        )
        weight_quant.quantize_weight_to_int(
            save_model_dir=save_model_dir,
            weight_bits=weight_bits,
            quantizable_op_type=quantizable_op_type,
            weight_quantize_type=weight_quantize_type,
            generate_test_model=generate_test_model,
            threshold_rate=threshold_rate,
        )
        print("finish weight quantization for " + model_name + "\n")

        try:
            os.system(f"rm -rf {save_model_dir}")
        except Exception as e:
            print(f"Failed to delete {save_model_dir} due to {e}")

    def convert_to_fp16(
        self,
        model_name,
        model_data_url,
        model_data_md5,
        model_filename,
        params_filename,
    ):
        model_dir = self.download_model(
            model_name, model_data_url, model_data_md5
        )
        load_model_dir = os.path.join(model_dir, model_name)

        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        save_model_dir = os.path.join(
            os.getcwd(), model_name + "_wq_fp16_" + timestamp
        )

        weight_quant = WeightQuantization(
            load_model_dir, model_filename, params_filename
        )

        weight_quant.convert_weight_to_fp16(save_model_dir)

        print(
            "finish converting the data type of weights to fp16 for "
            + model_name
        )
        print("fp16 model saved in " + save_model_dir + "\n")

        input_data = np.ones([1, 3, 224, 224], dtype=np.float32)
        res_fp32 = self.run_models(
            load_model_dir, model_filename, params_filename, input_data, False
        )
        res_fp16 = self.run_models(
            save_model_dir, model_filename, params_filename, input_data, True
        )

        np.testing.assert_allclose(
            res_fp32,
            res_fp16,
            rtol=1e-05,
            atol=1e-08,
            equal_nan=True,
            err_msg='Failed to test the accuracy of the fp32 and fp16 model.',
        )

        try:
            os.system(f"rm -rf {save_model_dir}")
        except Exception as e:
            print(f"Failed to delete {save_model_dir} due to {e}")

    def run_models(
        self,
        model_dir,
        model_filename,
        params_filename,
        input_data,
        is_fp16_model,
    ):
        print(model_dir)

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = paddle.static.load_inference_model(
                model_dir,
                exe,
                model_filename=model_filename,
                params_filename=params_filename,
            )

        if is_fp16_model:
            for var in inference_program.list_vars():
                if (
                    (var.type == paddle.framework.core.VarDesc.VarType.RAW)
                    or (not var.persistable)
                    or (var.name in ['feed', 'fetch'])
                    or (var.dtype != paddle.framework.core.VarDesc.VarType.FP16)
                ):
                    continue
                tensor = _load_variable_data(scope, var.name)
                _set_variable_data(
                    scope, place, var.name, tensor.astype(np.float32)
                )

        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: input_data},
            fetch_list=fetch_targets,
        )
        return np.array(results[0])


class TestWeightQuantizationMobilenetv1(TestWeightQuantization):
    nocomb_model_name = "mobilenetv1_fp32_nocombined"
    nocomb_model_data_url = "https://paddle-inference-dist.cdn.bcebos.com/Paddle-Inference-Demo/mobilenetv1_fp32_nocombined.tar.gz"
    nocomb_model_data_md5 = "c9aae3b04d9d535c84590ae557be0a0b"

    comb_model_name = "mobilenetv1_fp32_combined"
    comb_model_data_url = "https://paddle-inference-dist.cdn.bcebos.com/Paddle-Inference-Demo/mobilenetv1_fp32_combined.tar.gz"
    comb_model_data_md5 = "087c67e2b2b0a8b689fcc570a56c005f"

    def test_weight_quantization_mobilenetv1_8bit_abs_max(self):
        weight_bits = 8
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        weight_quantize_type = "abs_max"
        generate_test_model = True
        threshold_rate = 0.0
        self.quantize_to_int(
            self.comb_model_name,
            '__model__',
            '__params__',
            self.comb_model_data_url,
            self.comb_model_data_md5,
            weight_bits,
            quantizable_op_type,
            weight_quantize_type,
            generate_test_model,
            threshold_rate,
        )

    def test_weight_quantization_mobilenetv1_8bit_channel_wise_abs_max(self):
        weight_bits = 8
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        weight_quantize_type = "channel_wise_abs_max"
        generate_test_model = True
        threshold_rate = 0.0
        self.quantize_to_int(
            self.comb_model_name,
            '__model__',
            '__params__',
            self.comb_model_data_url,
            self.comb_model_data_md5,
            weight_bits,
            quantizable_op_type,
            weight_quantize_type,
            generate_test_model,
            threshold_rate,
        )

    def test_weight_quantization_mobilenetv1_16bit_abs_max(self):
        weight_bits = 16
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        weight_quantize_type = "abs_max"
        generate_test_model = False
        threshold_rate = 0
        self.quantize_to_int(
            self.comb_model_name,
            '__model__',
            '__params__',
            self.comb_model_data_url,
            self.comb_model_data_md5,
            weight_bits,
            quantizable_op_type,
            weight_quantize_type,
            generate_test_model,
            threshold_rate,
        )

    def test_weight_quantization_mobilenetv1_16bit_channel_wise_abs_max(self):
        weight_bits = 16
        quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
        weight_quantize_type = "channel_wise_abs_max"
        generate_test_model = False
        threshold_rate = 1e-9
        self.quantize_to_int(
            self.comb_model_name,
            '__model__',
            '__params__',
            self.comb_model_data_url,
            self.comb_model_data_md5,
            weight_bits,
            quantizable_op_type,
            weight_quantize_type,
            generate_test_model,
            threshold_rate,
        )

    def test_mobilenetv1_fp16_combined(self):
        model_filename = '__model__'
        params_filename = '__params__'
        self.convert_to_fp16(
            self.comb_model_name,
            self.comb_model_data_url,
            self.comb_model_data_md5,
            model_filename,
            params_filename,
        )


if __name__ == '__main__':
    unittest.main()
