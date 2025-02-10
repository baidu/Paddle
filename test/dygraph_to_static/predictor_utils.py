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

import numpy as np

from paddle import base
from paddle.base.core import AnalysisConfig, create_paddle_predictor
from paddle.framework import use_pir_api

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


class PredictorTools:
    '''
    Paddle-Inference predictor
    '''

    def __init__(self, model_path, model_file, params_file, feeds_var):
        '''
        __init__
        '''
        self.model_path = model_path
        self.model_file = model_file
        self.params_file = params_file

        self.feeds_var = feeds_var

    def _load_model_and_set_config(self):
        '''
        load model from file and set analysis config
        '''
        if os.path.exists(os.path.join(self.model_path, self.params_file)):
            config = AnalysisConfig(
                os.path.join(self.model_path, self.model_file),
                os.path.join(self.model_path, self.params_file),
            )
        else:
            config = AnalysisConfig(os.path.join(self.model_path))

        if base.is_compiled_with_cuda():
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        config.switch_specify_input_names(True)
        config.switch_use_feed_fetch_ops(False)
        config.enable_memory_optim()
        # config.disable_glog_info()
        # TODO: set it to True after PaddleInference fix the precision error
        # in CUDA11
        config.switch_ir_optim(False)

        if use_pir_api():
            config.enable_new_ir()
            config.enable_new_executor()
            if os.name == 'nt':
                config.delete_pass("conv2d_bn_fuse_pass")

        return config

    def _get_analysis_outputs(self, config):
        '''
        Return outputs of paddle inference
        Args:
            config (AnalysisConfig): predictor configs
        Returns:
            outs (numpy array): forward network prediction outputs
        '''
        predictor = create_paddle_predictor(config)
        names = predictor.get_input_names()
        for i, name in enumerate(names):
            # assert name in self.feeds_var, '{} not in feeded dict'.format(name)
            tensor = predictor.get_input_tensor(name)
            feed_data = self.feeds_var[i]
            tensor.copy_from_cpu(np.array(feed_data))
            if type(feed_data) == base.DenseTensor:
                tensor.set_lod(feed_data.lod())

        # ensure no diff in multiple repeat times
        repeat_time = 2
        for i in range(repeat_time):
            predictor.zero_copy_run()

        output_names = predictor.get_output_names()
        outs = [
            predictor.get_output_tensor(out_name).copy_to_cpu()
            for out_name in output_names
        ]

        return outs

    def __call__(self):
        '''
        __call__
        '''
        config = self._load_model_and_set_config()
        outputs = self._get_analysis_outputs(config)

        return outputs
