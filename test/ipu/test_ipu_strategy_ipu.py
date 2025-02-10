#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import paddle
import paddle.static

paddle.enable_static()


class TestIpuStrategy(unittest.TestCase):
    def test_set_options(self):
        ipu_strategy = paddle.static.IpuStrategy()
        all_option_names = ipu_strategy._ipu_strategy.get_all_option_names()
        skip_options = []
        skip_options.append(
            'mean_accumulation_and_replication_reduction_strategy'
        )
        skip_options.append('random_seed')

        for option_name in all_option_names:
            if option_name in skip_options:
                continue
            option = ipu_strategy._ipu_strategy.get_option(option_name)
            option_type = option['type']
            option_value = option['value']
            if option_type in ['double']:
                set_value = option_value + 0.5
            elif option_type == 'uint64':
                set_value = option_value + 1
            elif option_type == 'bool':
                set_value = not option_value
            else:
                continue

            try:
                ipu_strategy.set_options({option_name: set_value})
                new_value = ipu_strategy.get_option(option_name)
                assert (
                    new_value == set_value
                ), f"set {option_name} to {set_value} failed"
            except:
                raise Exception(f"set {option_name} to {set_value} failed")

    def test_set_string_options(self):
        ipu_strategy = paddle.static.IpuStrategy()
        options = {
            'cache_path': 'paddle_cache',
            'log_dir': 'paddle_log',
            'partials_type_matmuls': 'float',
        }
        ipu_strategy.set_options(options)
        for k, v in options.items():
            assert v == ipu_strategy.get_option(k), f"set {k} to {v} failed "

    def test_set_other_options(self):
        ipu_strategy = paddle.static.IpuStrategy()
        options = {}
        options['dot_checks'] = ['Fwd0', 'Fwd1', 'Bwd0', 'PreAlias', "Final"]
        options['engine_options'] = {
            'debug.allowOutOfMemory': 'true',
            'autoReport.directory': 'path',
            'autoReport.all': 'true',
        }
        options['random_seed'] = 1234
        for k, v in options.items():
            ipu_strategy.set_options({k: v})
            if isinstance(v, list):
                assert (
                    v.sort() == ipu_strategy.get_option(k).sort()
                ), f"set {k} to {v} failed "
            else:
                assert v == ipu_strategy.get_option(
                    k
                ), f"set {k} to {v} failed "

        # The custom logger need 2 int as inputs
        logger = lambda progress, total: print(
            f"compile progress: {progress}/{total}"
        )
        ipu_strategy.set_options({'compilation_progress_logger': logger})


if __name__ == "__main__":
    unittest.main()
