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

from . import auto_checkpoint as acp

g_train_epoch_ranges = {}

CONST_GENERATOR_BEGIN = "begin"
CONST_GENERATOR_END = "end"


class TrainEpochRangeWrapper(object):
    def __init__(self, name):
        self._running_status = CONST_GENERATOR_BEGIN
        self._epoch_no = -1

        self._train_epoch_range = acp.TrainEpochRange(
            -1,
            name,
            save_checkpoint_inter=acp._get_checker().save_checkpoint_inter,
            load_last=-2)


logger = acp._get_logger(20)


def _check():
    checker = acp._get_checker()
    if not checker.valid():
        return False

    if acp.g_acp_type == acp.CONST_ACP_TYPE:
        return


def _begin(name):
    if not _check():
        return False

    t = None
    if name not in g_train_epoch_ranges:
        g_train_epoch_ranges[name] = TrainEpochRangeWrapper(name)
    t = g_train_epoch_ranges[name]

    acp.g_train_epoch_range = t

    if t._train_epoch_range.restord_from != acp.CONST_CHECKPOINT:
        return True

    logger.info("begin generator epoch_no:{} checkpoint_epoch_no:{}",
                t._epoch_no, t._train_epoch_range.get())

    if t._epoch_no <= t._train_epoch_range.epoch_no:
        raise fluid.core.EOFException

    return True


def _end(name):
    if not _check():
        return False

    assert name in g_train_epoch_ranges, \
        "internal error: g_train_epoch_ranges must contain the name:{}, now:{}".format(name, g_train_epoch_ranges.keys())

    t = g_train_epoch_ranges[name]
    t._runing_status = CONST_GENERATOR_END

    assert t == acp.g_train_epoch_range, "interal error, current running range must equal"

    t._epoch_no += 1

    acp.g_train_epoch_range.save_checkpoint()
    acp.g_train_epoch_range = None

    logger.info("end generator epoch_no:{} checkpoint_epoch_no:{}", t._epoch_no,
                t._train_epoch_range.get())
    return True
