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

import multiprocessing
import os
import signal
import sys
import time
import unittest

from paddle.base import core


def set_child_signal_handler(self, child_pid):
    core._set_process_pids(id(self), (child_pid,))
    current_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)

    signal.signal(signal.SIGCHLD, __handler__)


class DygraphDataLoaderSignalHandler(unittest.TestCase):
    def test_child_process_exit_with_error(self):
        def __test_process__():
            core._set_process_signal_handler()
            sys.exit(1)

        def try_except_exit():
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()

                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn("Fatal", str(ex))
                exception = ex
            return exception

        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break

        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigsegv(self):
        def __test_process__():
            core._set_process_signal_handler()
            os.kill(os.getpid(), signal.SIGSEGV)

        def try_except_exit():
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()

                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn("Segmentation fault", str(ex))
                exception = ex
            return exception

        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break

        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigbus(self):
        def __test_process__():
            core._set_process_signal_handler()
            os.kill(os.getpid(), signal.SIGBUS)

        def try_except_exit():
            exception = None
            try:
                test_process = multiprocessing.Process(target=__test_process__)
                test_process.start()

                set_child_signal_handler(id(self), test_process.pid)
                time.sleep(5)
            except SystemError as ex:
                self.assertIn("Bus error", str(ex))
                exception = ex
            return exception

        try_time = 10
        exception = None
        for i in range(try_time):
            exception = try_except_exit()
            if exception is not None:
                break

        self.assertIsNotNone(exception)

    def test_child_process_killed_by_sigterm(self):
        def __test_process__():
            core._set_process_signal_handler()
            time.sleep(10)

        test_process = multiprocessing.Process(target=__test_process__)
        test_process.daemon = True
        test_process.start()

        set_child_signal_handler(id(self), test_process.pid)
        time.sleep(1)


if __name__ == '__main__':
    unittest.main()
