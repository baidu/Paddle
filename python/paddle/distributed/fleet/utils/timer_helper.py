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
import time

import paddle
from paddle.base import core

_GLOBAL_TIMERS = None


def is_timer_initialized():
    return _GLOBAL_TIMERS is not None


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, f"{name} has been already initialized."


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, f"{name} is not initialized."


def get_timers():
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers()


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already started"
        paddle.device.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timers."""
        assert self.started_, "timer is not started."
        paddle.device.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class _GPUEventTimer:
    """GPUEventTimer."""

    def __init__(self, name):
        self.name = name
        dev_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        self.timer = core.GPUEventTimer(core.CUDAPlace(dev_id))

    def __getattr__(self, name):
        return getattr(self.timer, name)


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name, use_event=True):
        clazz = _GPUEventTimer if use_event else _Timer
        timer = self.timers.get(name)
        if timer is None:
            timer = clazz(name)
            self.timers[name] = timer
        else:
            assert (
                type(timer) == clazz
            ), f"Invalid timer type: {clazz} vs {type(timer)}"
        return timer

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = (
                self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            )
            string += f" | {name}: {elapsed_time:.2f}"
        print(string, flush=True)
