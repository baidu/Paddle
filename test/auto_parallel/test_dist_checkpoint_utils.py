# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import collective.test_communication_api_base as test_base
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.checkpoint.load_state_dict import get_checkpoint_files
from paddle.distributed.checkpoint.utils import (
    flatten_state_dict,
    unflatten_state_dict,
)

os.environ['FLAGS_enable_pir_api'] = '0'


class TestDistCheckpointUtils(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=120, nnode=1)
        self._default_envs = {}
        self._changeable_envs = {"backend": ["gpu"]}

    def test_flatten_mapping(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            ckpt_path = ckpt_path_tmp.name
            envs["ckpt_path"] = ckpt_path
            self.run_test_case(
                "semi_auto_parallel_checkpoint_flatten_mapping.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()

    def test_dedup_tensor(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path_tmp = tempfile.TemporaryDirectory()
            ckpt_path = ckpt_path_tmp.name
            envs["ckpt_path"] = ckpt_path
            self.run_test_case(
                "semi_auto_parallel_checkpoint_dedup_tensor.py",
                user_defined_envs=envs,
            )
            ckpt_path_tmp.cleanup()

    def test_flatten_state_dict(self):
        state_dict = {
            "model": {
                "a.0": paddle.to_tensor([1, 2]),
                "b": paddle.to_tensor([3, 4]),
            },
            "optimizer": {
                "c": paddle.to_tensor([5, 6]),
                "d.2": paddle.to_tensor([7, 8]),
            },
        }
        expected_flat_state_dict = {
            "model.a.0": paddle.to_tensor([1, 2]),
            "model.b": paddle.to_tensor([3, 4]),
            "optimizer.c": paddle.to_tensor([5, 6]),
            "optimizer.d.2": paddle.to_tensor([7, 8]),
        }
        flat_state_dict, mapping = flatten_state_dict(state_dict)
        self.assertTrue(len(expected_flat_state_dict) == len(flat_state_dict))
        for k, v in flat_state_dict.items():
            self.assertTrue(isinstance(v, paddle.Tensor))
            self.assertTrue(k in expected_flat_state_dict)
            np.testing.assert_equal(
                v.numpy(), expected_flat_state_dict[k].numpy()
            )
        recover_state_dict = unflatten_state_dict(flat_state_dict, mapping)

        def check_state_dict(d1, d2):
            self.assertTrue(len(d1) == len(d2))
            self.assertTrue(type(d1) == type(d2))
            if isinstance(d1, dict):
                for k in d1:
                    self.assertTrue(k in d2)
                    check_state_dict(d1[k], d2[k])
            elif isinstance(d1, paddle.Tensor):
                np.testing.assert_equal(d1.numpy(), d2.numpy())
            else:
                raise ValueError(f"Invalid type of state_dict:{d1} != {d2}")

        check_state_dict(recover_state_dict, state_dict)

    def test_get_rank_to_files(self):
        process_group = None
        use_dist = False
        ckpt_dir_tmp = tempfile.TemporaryDirectory()
        ckpt_dir = ckpt_dir_tmp.name
        state_dict = {
            "w1": paddle.to_tensor([1, 2]),
            "w2": paddle.to_tensor([3, 4]),
        }
        dist.save_state_dict(state_dict, ckpt_dir)

        metadata_files, local_load_files = get_checkpoint_files(ckpt_dir)
        metadata_list = []

        for metadata_file in metadata_files:
            metadata_list.append(
                paddle.load(os.path.join(ckpt_dir, metadata_file))
            )

        new_state_dict = {
            "w1": paddle.to_tensor([1, 2]),
            "w2": paddle.to_tensor([3, 4]),
        }
        (
            rank_to_files,
            missing_keys,
            mw_name_compatibility_mapping,
        ) = dist.checkpoint.load_state_dict.get_rank_to_files(
            metadata_list,
            local_load_files,
            new_state_dict,
            process_group,
            use_dist,
        )
        self.assertTrue(len(rank_to_files) == 1 and 0 in rank_to_files)
        self.assertTrue(rank_to_files[0] == ["0_0.distcp"])
        self.assertTrue(len(missing_keys) == 0)
        self.assertTrue(len(mw_name_compatibility_mapping) == 0)

        new_state_dict = {
            "w1": paddle.to_tensor([1, 2]),
            "w3": paddle.to_tensor([3, 4]),
        }
        (
            rank_to_files,
            missing_keys,
            mw_name_compatibility_mapping,
        ) = dist.checkpoint.load_state_dict.get_rank_to_files(
            metadata_list,
            local_load_files,
            new_state_dict,
            process_group,
            use_dist,
        )
        self.assertTrue(len(rank_to_files) == 1 and 0 in rank_to_files)
        self.assertTrue(rank_to_files[0] == ["0_0.distcp"])
        self.assertTrue(len(missing_keys) == 1)
        self.assertTrue(len(mw_name_compatibility_mapping) == 0)
        self.assertTrue("w3" in missing_keys)

        new_state_dict = {
            "w3": paddle.to_tensor([3, 4]),
            "w4": paddle.to_tensor([5, 6]),
        }
        (
            rank_to_files,
            missing_keys,
            mw_name_compatibility_mapping,
        ) = dist.checkpoint.load_state_dict.get_rank_to_files(
            metadata_list,
            local_load_files,
            new_state_dict,
            process_group,
            use_dist,
        )
        self.assertTrue(len(rank_to_files) == 0)
        self.assertTrue(len(missing_keys) == 2)
        self.assertTrue(len(mw_name_compatibility_mapping) == 0)
        self.assertTrue("w3" in missing_keys)
        self.assertTrue("w4" in missing_keys)

        ckpt_dir_tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
