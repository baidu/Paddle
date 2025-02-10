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
from __future__ import annotations

import traceback


class SotErrorBase(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..opcode_translator.breakpoint import BreakpointManager

        BreakpointManager().on_event(f"{self.__class__.__name__}")

    def print(self):
        lines = traceback.format_tb(self.__traceback__)
        print("".join(lines))


class InnerError(SotErrorBase):
    pass


class HasNoAttributeError(InnerError):
    pass


class FallbackError(SotErrorBase):
    def __init__(self, msg, disable_eval_frame=False):
        super().__init__(msg)
        self.disable_eval_frame = disable_eval_frame


# raise in inline function call strategy.
class BreakGraphError(SotErrorBase):
    pass


def inner_error_default_handler(func, message_fn):
    """Wrap function and an error handling function and throw an InnerError."""

    def impl(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SotErrorBase as e:
            raise e
        except Exception as e:
            message = message_fn(*args, **kwargs)
            origin_exception_message = "\n".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            raise InnerError(
                f"{message}\nOrigin Exception is: \n {origin_exception_message}"
            ) from e

    return impl


class ExportError(SotErrorBase):
    pass


class SotExtraInfo:
    SOT_EXTRA_INFO_ATTR_NAME = "__SOT_EXTRA_INFO__"

    def __init__(self, *, need_breakgraph: bool = False):
        self.need_breakgraph = need_breakgraph

    def set_need_breakgraph(self, need_breakgraph: bool):
        self.need_breakgraph = need_breakgraph

    def attach(self, err: BaseException):
        setattr(err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, self)

    @staticmethod
    def default() -> SotExtraInfo:
        return SotExtraInfo()

    @staticmethod
    def from_exception(err: BaseException) -> SotExtraInfo:
        info = getattr(
            err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, SotExtraInfo.default()
        )
        setattr(err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, info)
        return info
