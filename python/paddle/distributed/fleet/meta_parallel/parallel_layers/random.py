#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from ...layers.mpu.random import RNGStatesTracker  # noqa: F401
from ...layers.mpu.random import get_rng_state_tracker  # noqa: F401
from ...layers.mpu.random import model_parallel_random_seed  # noqa: F401
from ...layers.mpu.random import determinate_seed  # noqa: F401
from ...layers.mpu.random import dropout  # noqa: F401

__all__ = [
    "RNGStatesTracker", "get_rng_state_tracker", "model_parallel_random_seed",
    "determinate_seed", "dropout"
]
