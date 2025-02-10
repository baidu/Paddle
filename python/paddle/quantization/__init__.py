"""Quantization Module"""

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

from .base_observer import BaseObserver
from .base_quanter import BaseQuanter
from .config import QuantConfig
from .factory import quanter
from .imperative.ptq import (  # noqa: F401
    ImperativePTQ,
)
from .imperative.ptq_config import (  # noqa: F401
    PTQConfig,
    default_ptq_config,
)
from .imperative.ptq_quantizer import (  # noqa: F401
    SUPPORT_ACT_QUANTIZERS,
    SUPPORT_WT_QUANTIZERS,
    AbsmaxQuantizer,
    BaseQuantizer,
    HistQuantizer,
    KLQuantizer,
    PerChannelAbsmaxQuantizer,
)
from .imperative.ptq_registry import (  # noqa: F401
    PTQRegistry,
)
from .imperative.qat import (  # noqa: F401
    ImperativeQuantAware,
)
from .ptq import PTQ
from .qat import QAT
from .fp8ptq import FP8PTQ
__all__ = [
    "QuantConfig",
    "BaseQuanter",
    "BaseObserver",
    "quanter",
    "QAT",
    "PTQ",
    "FP8PTQ"
]
