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

from .call_ast_utils import get_static_function, try_ast_func  # noqa: F401
from .envs import (  # noqa: F401
    ENV_CLEAN_CODE,
    ENV_COST_MODEL,
    ENV_MIN_GRAPH_SIZE,
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    ENV_SOT_ENABLE_FASTER_GUARD,
    ENV_SOT_ENABLE_GUARD_TREE,
    ENV_SOT_EXPORT,
    ENV_SOT_FORCE_FALLBACK_SIR_IDS,
    ENV_SOT_LOG_LEVEL,
    ENV_SOT_WITH_CONTROL_FLOW,
    ENV_STRICT_MODE,
    allow_dynamic_shape_guard,
    cost_model_guard,
    export_guard,
    faster_guard_guard,
    guard_tree_guard,
    min_graph_size_guard,
    sot_step_profiler_guard,
    strict_mode_guard,
    with_control_flow_guard,
)
from .exceptions import (  # noqa: F401
    BreakGraphError,
    ExportError,
    FallbackError,
    InnerError,
    inner_error_default_handler,
)
from .info_collector import (  # noqa: F401
    CompileCountInfo,
    InfoCollector,
    NewSymbolHitRateInfo,
    SubGraphRelationInfo,
)
from .magic_methods import magic_method_builtin_dispatch  # noqa: F401
from .paddle_api_config import (  # noqa: F401
    get_tensor_methods,
    is_break_graph_tensor_methods,
    is_directly_run_api,
    is_inplace_api,
    is_not_supported_paddle_layer,
)
from .utils import (  # noqa: F401
    Cache,
    ConstTypes,
    GraphLogger,
    NameGenerator,
    ResumeFnNameFactory,
    Singleton,
    SotUndefinedVar,
    StepInfoManager,
    StepState,
    count_if,
    current_symbol_registry,
    execute_time,
    flatten,
    flatten_extend,
    get_api_fullname,
    get_unbound_method,
    hashable,
    in_paddle_module,
    is_break_graph_api,
    is_builtin_fn,
    is_clean_code,
    is_comprehensive_name,
    is_paddle_api,
    is_strict_mode,
    list_contain_by_id,
    list_find_index_by_id,
    log,
    log_do,
    log_enabled,
    log_format,
    map_if,
    map_if_extend,
    meta_str,
    no_eval_frame,
    printable,
    switch_symbol_registry,
)
