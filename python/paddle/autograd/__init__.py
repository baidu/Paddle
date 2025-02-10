# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ..base.dygraph.base import (  # noqa: F401
    enable_grad,
    grad,
    is_grad_enabled,
    no_grad_ as no_grad,
    set_grad_enabled,
)
from . import (  # noqa: F401
    backward_mode,
    forward_mode,
    ir_backward,
)
from .autograd import hessian, jacobian
from .backward_mode import backward
from .forward_mode import (
    dual_level,
    enter_dual_level,
    exit_dual_level,
    make_dual,
)
from .py_layer import PyLayer, PyLayerContext
from .saved_tensors_hooks import saved_tensors_hooks

__all__ = [
    'jacobian',
    'hessian',
    'backward',
    'PyLayer',
    'PyLayerContext',
    'saved_tensors_hooks',
    'enter_dual_level',
    'exit_dual_level',
    'make_dual',
    'dual_level',
]
