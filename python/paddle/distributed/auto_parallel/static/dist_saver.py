# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import errno
import logging
import os
import pickle
import re

import numpy as np

import paddle
from paddle.framework import core

from ...utils.log_utils import get_logger
from .process_group import _g_process_group_map
from .utils import get_dist_attr


def check_filename(re_exp, filename):
    if re.search(re_exp, filename):
        return True
    else:
        return False


def _process_path(path):
    filename = os.path.basename(path)
    if filename == "":
        raise ValueError(
            "path should be of 'dirname/filename' format, but received filename is empty string"
        )
    try:
        dirname = os.path.dirname(path)
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return dirname, filename


class DistributedSaver:
    def __init__(self):
        self._logger = get_logger(logging.INFO)

    def save(self, path, serial_program, dist_main_program, dist_context):
        def _save_state(program, path, mode="param"):
            state = {
                k: np.array(v) for k, v in program.state_dict(mode).items()
            }
            with open(path, "wb") as f:
                pickle.dump(state, f)

        dirname, filename = _process_path(path)

        rank_id = paddle.distributed.get_rank()
        # save serial program when rank id is 0
        if rank_id == 0:
            self._save_rank_mapping(dirname)
            serial_model_filename = filename + "_serial.pdmodel"
            serial_model_path = os.path.join(dirname, serial_model_filename)
            with open(serial_model_path, "wb") as f:
                f.write(serial_program.desc.serialize_to_string())

        # save distributed main program
        dist_model_filename = filename + "_dist" + str(rank_id) + ".pdmodel"
        dist_model_path = os.path.join(dirname, dist_model_filename)
        with open(dist_model_path, "wb") as f:
            f.write(dist_main_program.desc.serialize_to_string())

        # save distributed attribute
        dist_attr_filename = filename + "_dist" + str(rank_id) + ".pdattr"
        dist_attr_path = os.path.join(dirname, dist_attr_filename)
        dist_attrs = get_dist_attr(dist_main_program, dist_context)
        with open(dist_attr_path, "wb") as f:
            pickle.dump(dist_attrs, f)

        # save distributed params
        dist_param_filename = filename + "_dist" + str(rank_id) + ".pdparams"
        dist_param_path = os.path.join(dirname, dist_param_filename)
        _save_state(dist_main_program, dist_param_path)

        # save distributed opt states
        dist_opt_filename = filename + "_dist" + str(rank_id) + ".pdopt"
        dist_opt_path = os.path.join(dirname, dist_opt_filename)
        _save_state(dist_main_program, dist_opt_path, "opt")

        # TODO:save cluster.json

    def load(self, path, load_optimizer=True):
        # TODO: if `program` is None, load `path.pdmodel`.
        def _load_file(filename, dirname, suffix="pdparams"):
            file_list = []
            for file in os.listdir(dirname):
                if check_filename(f'{filename}(.*)_dist(.*).{suffix}', file):
                    file_list.append(os.path.join(dirname, file))
            file_list.sort()
            return file_list

        def _load_state(filename, dirname, suffix="pdparams"):
            file_list = _load_file(filename, dirname, suffix)
            state_dict = {}
            for file in file_list:
                with open(file, 'rb') as f:
                    state_dict_info = pickle.load(f, encoding='latin1')
                for name, value in state_dict_info.items():
                    if name in state_dict:
                        state_dict[name].append(np.array(value))
                    else:
                        state_dict[name] = [np.array(value)]
            self._logger.info(f"Load param file: {file_list}")
            return state_dict

        filename = os.path.basename(path)
        if filename == "":
            raise ValueError(
                "path should be of 'dirname/filename' format, but received filename is empty string"
            )
        dirname = os.path.dirname(path)

        # load path.pdparam and path.pdopt
        param_state_dict = _load_state(filename, dirname)
        opt_state_dict = (
            _load_state(filename, dirname, "pdopt") if load_optimizer else {}
        )
        state_dict = dict(param_state_dict, **opt_state_dict)

        # load path.pdattr
        dist_attr_file_list = _load_file(filename, dirname, "pdattr")
        self._logger.info(
            f"Load distributed attribute file: {dist_attr_file_list}"
        )
        dist_attr = {}
        for dist_attr_file in dist_attr_file_list:
            with open(dist_attr_file, 'rb') as f:
                dist_attr_info = pickle.load(f, encoding='latin1')
            for name, attr in dist_attr_info.items():
                if name not in dist_attr:
                    dist_attr[name] = attr

        return state_dict, dist_attr

    def save_inference_model(self, path, feed_vars, fetch_vars, exe, **kwargs):
        dirname, filename = _process_path(path)

        # save distributed inference program
        rank_id = paddle.distributed.get_rank()
        if rank_id == 0:
            self._save_rank_mapping(dirname)
        op_role_key = core.op_proto_and_checker_maker.kOpRoleAttrName()
        op_role_forward = int(core.op_proto_and_checker_maker.OpRole.Forward)

        dist_main_prog = kwargs.get('program', None)
        if not dist_main_prog:
            dist_main_prog = paddle.static.default_main_program()
        global_block = dist_main_prog.global_block()

        ops = global_block.ops
        feed_vars_names = [x.name for x in feed_vars]
        fetch_vars_names = [x.name for x in fetch_vars]

        last_idx = -1
        for idx, op in enumerate(ops):
            if op.attr(op_role_key) != op_role_forward:
                continue
            if op.type == "read" or op.type == "feed":
                feed_vars_names += op.output("Out")
            if op.type == 'p_recv':
                feed_vars_names += op.output("out")
            if op.type == "p_send":
                fetch_vars_names += op.input("x")
            for out_name in op.output_arg_names:
                if out_name in fetch_vars_names:
                    last_idx = max(idx, last_idx)

        used_inputs = []
        used_outputs = []
        for idx, op in enumerate(ops):
            if idx > last_idx:
                break
            used_inputs += op.input_arg_names
            used_outputs += op.output_arg_names

        # delete duplicated elements and keep order
        feed_vars_names = list({}.fromkeys(feed_vars_names).keys())
        used_inputs = list({}.fromkeys(used_inputs).keys())
        fetch_vars_names = list({}.fromkeys(fetch_vars_names).keys())
        used_outputs = list({}.fromkeys(used_outputs).keys())

        dist_feed_vars_names = [
            var_name for var_name in feed_vars_names if var_name in used_inputs
        ]
        dist_fetch_vars_names = [
            var_name
            for var_name in fetch_vars_names
            if var_name in used_outputs
        ]

        dist_feed_vars = list(
            reversed([global_block.vars[name] for name in dist_feed_vars_names])
        )
        dist_fetch_vars = [
            global_block.vars[name] for name in dist_fetch_vars_names
        ]

        dist_filename = filename + "_dist" + str(rank_id)
        dist_path = os.path.join(dirname, dist_filename)
        legacy_format = kwargs.get("legacy_format", False)
        paddle.static.save_inference_model(
            dist_path,
            dist_feed_vars,
            dist_fetch_vars,
            exe,
            program=dist_main_prog,
            legacy_format=legacy_format,
        )

    def _save_rank_mapping(self, dirname):
        path = os.path.join(dirname, 'rank_mapping.csv')
        f = open(path, 'w')
        f.write('[ring_id -> ranks]\n')
        for process_group in _g_process_group_map.values():
            ring_id = process_group._group_id
            ranks = [str(rank) for rank in process_group._ranks]
            id_to_rank = str(ring_id) + "," + ",".join(ranks) + '\n'
            f.write(id_to_rank)
            id_to_rank = ""
        f.write('[rank -> ring_ids]\n')
        rank_to_id_dict = {}
        for process_group in _g_process_group_map.values():
            ring_id = process_group._group_id
            for rank in process_group._ranks:
                if rank in rank_to_id_dict:
                    rank_to_id_dict[rank].append(str(ring_id))
                else:
                    rank_to_id_dict[rank] = [str(ring_id)]
        rank_to_id = ""
        for item, val in rank_to_id_dict.items():
            rank_to_id += str(item) + ","
            rank_to_id += ",".join(val) + "\n"
            f.write(rank_to_id)
            rank_to_id = ""
        f.close()
