// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Python.h>
// Avoid a problem with copysign defined in pyconfig.h on Windows.
#ifdef copysign
#undef copysign
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>  // NOLINT // for call_once
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/compiled_program.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/dense_tensor_array.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/ir/coalesce_grad_tensor_pass.h"
#include "paddle/fluid/framework/ir/cost_model.h"
#include "paddle/fluid/framework/ir/generate_pass.h"
#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/new_executor/executor_statistics.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/scope_pool.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/phi/core/framework/reader.h"
#include "paddle/phi/core/memory/allocation/allocator_strategy.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/core/memory/allocation/cuda_ipc_allocator.h"
#endif
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/profiler.h"
#include "paddle/fluid/pybind/bind_cost_model.h"
#include "paddle/fluid/pybind/box_helper_py.h"
#include "paddle/fluid/pybind/communication.h"
#include "paddle/fluid/pybind/compatible.h"
#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/pybind/cuda_streams_py.h"
#include "paddle/fluid/pybind/data_set_py.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/fleet_wrapper_py.h"
#include "paddle/fluid/pybind/generator_py.h"
#include "paddle/fluid/pybind/global_value_getter_setter.h"
#include "paddle/fluid/pybind/gloo_context_py.h"
#include "paddle/fluid/pybind/gloo_wrapper_py.h"
#include "paddle/fluid/pybind/graph.h"
#include "paddle/fluid/pybind/heter_wrapper_py.h"
#include "paddle/fluid/pybind/imperative.h"
#include "paddle/fluid/pybind/inference_api.h"
#include "paddle/fluid/pybind/io.h"
#include "paddle/fluid/pybind/metrics_py.h"
#include "paddle/fluid/pybind/ps_gpu_wrapper_py.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/core/memory/allocation/mmap_allocator.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/monitor.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"
#include "paddle/utils/none.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/pybind/nccl_wrapper_py.h"
#endif
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/fluid/pybind/reader_py.h"
#include "paddle/fluid/pybind/tensor_py.h"
#include "paddle/utils/string/to_string.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#endif
#ifndef PADDLE_WITH_HIP
#include "paddle/phi/core/platform/device/gpu/cuda/cuda_profiler.h"
#endif
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#endif

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/phi/core/platform/device/xpu/xpu_op_list.h"
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/capi/capi.h"
#endif

#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"

#ifdef PADDLE_WITH_IPU
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_info.h"
#endif

#ifdef PADDLE_WITH_CRYPTO
#include "paddle/fluid/pybind/crypto.h"
#endif

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/pybind/fleet_py.h"
#endif

#include "paddle/common/flags.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/layout_autotune.h"
#include "paddle/fluid/pybind/compiled_program.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "pybind11/stl.h"

COMMON_DECLARE_bool(use_mkldnn);

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(phi::TensorArray);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchUnmergedList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchList);
PYBIND11_MAKE_OPAQUE(paddle::framework::FetchType);

namespace paddle {
namespace pybind {
using namespace paddle::framework;               // NOLINT
void BindCompiledProgram(pybind11::module &m) {  // NOLINT
  // -- python binds for compiled_program.
  py::class_<CompiledProgram> cp(m, "CompiledProgram");

  py::enum_<paddle::platform::DeviceType>(m, "DeviceType", py::arithmetic())
      .value("CPU", paddle::platform::DeviceType::CPU)
      .value("CUDA", paddle::platform::DeviceType::CUDA)
      .value("XPU", paddle::platform::DeviceType::XPU);

  py::class_<BuildStrategy> build_strategy(cp, "BuildStrategy", R"DOC(
    BuildStrategy allows the user to more preciously control how to
    build the SSA Graph in CompiledProgram by setting the property.

    Returns:
        BuildStrategy: An BuildStrategy object.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> data = static.data(name="x", shape=[None, 1], dtype="float32")
            >>> hidden = static.nn.fc(data, size=10)
            >>> loss = paddle.mean(hidden)
            >>> paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

            >>> build_strategy = static.BuildStrategy()
            >>> build_strategy.enable_inplace = True
            >>> build_strategy.memory_optimize = True
            >>> build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
            >>> program = static.CompiledProgram(static.default_main_program(), build_strategy=build_strategy)
)DOC");

  py::enum_<BuildStrategy::ReduceStrategy>(build_strategy, "ReduceStrategy")
      .value("Reduce", BuildStrategy::ReduceStrategy::kReduce)
      .value("AllReduce", BuildStrategy::ReduceStrategy::kAllReduce)
      .value("_NoReduce", BuildStrategy::ReduceStrategy::kNoReduce);

  build_strategy.def(py::init())
      .def("_clear_finalized", &BuildStrategy::ClearFinalized)
      .def_property(
          "reduce_strategy",
          [](const BuildStrategy &self) { return self.reduce_; },
          [](BuildStrategy &self, BuildStrategy::ReduceStrategy strategy) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.reduce_ = strategy;
          },
          R"DOC((fluid.BuildStrategy.ReduceStrategy, optional): there are two reduce
                strategies in CompiledProgram, AllReduce and Reduce. If you want
                that all the parameters' optimization are done on all devices independently,
                you should choose AllReduce; otherwise, if you choose Reduce, all the parameters'
                optimization will be evenly distributed to different devices, and then
                broadcast the optimized parameter to other devices.
                Default is 'AllReduce'.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
          )DOC")
      .def_property(
          "debug_graphviz_path",
          [](const BuildStrategy &self) { return self.debug_graphviz_path_; },
          [](BuildStrategy &self, const std::string &path) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.debug_graphviz_path_ = path;
          },
          R"DOC((str, optional): debug_graphviz_path indicates the path that
                writing the SSA Graph to file in the form of graphviz.
                It is useful for debugging. Default is empty string, that is, ""

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.debug_graphviz_path = "./graph"
          )DOC")
      .def_property(
          "num_trainers",
          [](const BuildStrategy &self) { return self.num_trainers_; },
          [](BuildStrategy &self, int num_trainers) {
#ifdef WIN32
            PADDLE_THROW(common::errors::Unavailable(
                "Distribution mode is not supported on Windows platform."));
#endif
            self.num_trainers_ = num_trainers;
          })
      .def_property(
          "trainers_endpoints",
          [](const BuildStrategy &self) { return self.trainers_endpoints_; },
          [](BuildStrategy &self,
             const std::vector<std::string> &trainers_endpoints) {
            self.trainers_endpoints_ = trainers_endpoints;
          })
      .def_property(
          "trainer_id",
          [](const BuildStrategy &self) { return self.trainer_id_; },
          [](BuildStrategy &self, int trainer_id) {
            self.trainer_id_ = trainer_id;
          })
      .def_property(
          "nccl_comm_num",
          [](const BuildStrategy &self) { return self.nccl_comm_num_; },
          [](BuildStrategy &self, int nccl_comm_num) {
            self.nccl_comm_num_ = nccl_comm_num;
          })
      .def_property(
          "bkcl_comm_num",
          [](const BuildStrategy &self) { return self.bkcl_comm_num_; },
          [](BuildStrategy &self, int bkcl_comm_num) {
            self.bkcl_comm_num_ = bkcl_comm_num;
          })
      .def_property(
          "use_hierarchical_allreduce",
          [](const BuildStrategy &self) {
            return self.use_hierarchical_allreduce_;
          },
          [](BuildStrategy &self, bool use) {
            self.use_hierarchical_allreduce_ = use;
          })
      .def_property(
          "hierarchical_allreduce_inter_nranks",
          [](const BuildStrategy &self) {
            return self.hierarchical_allreduce_inter_nranks_;
          },
          [](BuildStrategy &self, int nranks) {
            self.hierarchical_allreduce_inter_nranks_ = nranks;
          })
      .def_property(
          "build_cinn_pass",
          [](const BuildStrategy &self) { return self.build_cinn_pass_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, "
                                  "cannot be configured again."));
            self.build_cinn_pass_ = b;
          },
          R"DOC((bool, optional): build_cinn_pass indicates whether
                      to lowering some operators in graph into cinn ops
                      to execute, which will speed up the process of execution.
                      Default False.

                      Examples:
                            .. code-block:: python

                                >>> import paddle
                                >>> import paddle.static as static
                                >>> paddle.enable_static()
                                >>> build_strategy = static.BuildStrategy()
                                >>> build_strategy.build_cinn_pass = True
          )DOC")
      .def_property(
          "fuse_elewise_add_act_ops",
          [](const BuildStrategy &self) {
            return self.fuse_elewise_add_act_ops_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_elewise_add_act_ops_ = b;
          },
          R"DOC((bool, optional): fuse_elewise_add_act_ops indicate whether
                to fuse elementwise_add_op and activation_op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fuse_elewise_add_act_ops = True
          )DOC")
      .def_property(
          "fuse_gemm_epilogue",
          [](const BuildStrategy &self) { return self.fuse_gemm_epilogue_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_gemm_epilogue_ = b;
          },
          R"DOC((bool, optional): fuse_gemm_epilogue indicate whether
                to fuse matmul_op, elemenewist_add_op and activation_op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fuse_gemm_epilogue = True
          )DOC")
      .def_property(
          "fuse_dot_product_attention",
          [](const BuildStrategy &self) {
            return self.fuse_dot_product_attention_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finlaized, cannot be "
                                  "configured again."));
            self.fuse_dot_product_attention_ = b;
          },
          R"DOC((bool, optional): fuse_dot_product_attention indicate whether
                to fuse dot product attention,
                it would make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.fuse_dot_product_attention = True
                     )DOC")
      .def_property(
          "fuse_adamw",
          [](const BuildStrategy &self) { return self.fuse_adamw_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_adamw_ = b;
          },
          R"DOC((bool, optional): fuse_adamw indicate whether
                to fuse all adamw optimizers with multi_tensor_adam,
                it may make the execution faster. Default is False.
                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static
                        >>> paddle.enable_static()
                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fuse_adamw = True
          )DOC")
      .def_property(
          "fused_attention",
          [](const BuildStrategy &self) { return self.fused_attention_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fused_attention_ = b;
          },
          R"DOC((bool, optional): fused_attention indicate whether
                to fuse the whole multi head attention part with one op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fused_attention = True
          )DOC")
      .def_property(
          "fused_feedforward",
          [](const BuildStrategy &self) { return self.fused_feedforward_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fused_feedforward_ = b;
          },
          R"DOC((bool, optional): fused_feedforward indicate whether
                to fuse the whole feed_forward part with one op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fused_feedforward = True
          )DOC")
      .def_property(
          "sequential_run",
          [](const BuildStrategy &self) { return self.sequential_run_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.sequential_run_ = b;
          },
          R"DOC((bool, optional): sequential_run is used to let the `StandaloneExecutor` run ops by the
          order of `ProgramDesc`. Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.sequential_run = True
          )DOC")
      .def_property(
          "fuse_resunit",
          [](const BuildStrategy &self) { return self.fuse_resunit_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_resunit_ = b;
#ifndef PADDLE_WITH_CUDNN_FRONTEND
            if (self.fuse_resunit_) {
              PADDLE_THROW(common::errors::PreconditionNotMet(
                  "Paddle is not built with CUDNN Frontend support."));
            }
#endif
          },
          R"DOC((bool, optional): fuse_resunit Default is False.

                Examples:
                    .. code-block:: python

                        import paddle
                        import paddle.static as static

                        paddle.enable_static()

                        build_strategy = static.BuildStrategy()
                        build_strategy.fuse_resunit = True
                     )DOC")
      .def_property(
          "fuse_bn_act_ops",
          [](const BuildStrategy &self) { return self.fuse_bn_act_ops_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_bn_act_ops_ = b;
          },
          R"DOC((bool, optional): fuse_bn_act_ops indicate whether
                to fuse batch_norm and activation_op,
                it may make the execution faster. Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fuse_bn_act_ops = True
          )DOC")
      .def_property(
          "fuse_bn_add_act_ops",
          [](const BuildStrategy &self) { return self.fuse_bn_add_act_ops_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_bn_add_act_ops_ = b;
          },
          R"DOC((bool, optional): fuse_bn_add_act_ops indicate whether
                to fuse batch_norm, elementwise_add and activation_op,
                it may make the execution faster. Default is True

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fuse_bn_add_act_ops = True
          )DOC")
      .def_property(
          "enable_auto_fusion",
          [](const BuildStrategy &self) { return self.enable_auto_fusion_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.enable_auto_fusion_ = b;
          },
          R"DOC((bool, optional): Whether to enable fusing subgraph to a
                fusion_group. Now we only support fusing subgraph that composed
                of elementwise-like operators, such as elementwise_add/mul
                without broadcast and activations.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.enable_auto_fusion = True
          )DOC")
      .def_property(
          "fuse_relu_depthwise_conv",
          [](const BuildStrategy &self) {
            return self.fuse_relu_depthwise_conv_;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.fuse_relu_depthwise_conv_ = b;
          },
          R"DOC((bool, optional): fuse_relu_depthwise_conv indicate whether
                to fuse relu and depthwise_conv2d,
                it will save GPU memory and may make the execution faster.
                This options is only available in GPU devices.
                Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.fuse_relu_depthwise_conv = True
          )DOC")
      .def_property(
          "fuse_broadcast_ops",
          [](const BuildStrategy &self) {
            return self.fuse_broadcast_ops_ == true ||
                   self.fuse_broadcast_ops_ == paddle::none;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, "
                                  "cannot be configured again."));
            self.fuse_broadcast_ops_ = b;
          },
          R"DOC((bool, optional): fuse_broadcast_op indicates whether
                      to fuse the broadcast ops. Note that, in Reduce mode,
                      fusing broadcast ops may make the program faster. Because
                      fusing broadcast OP equals delaying the execution of all
                      broadcast Ops, in this case, all nccl streams are used only
                      for NCCLReduce operations for a period of time. Default False.

                      Examples:
                            .. code-block:: python

                                >>> import paddle
                                >>> import paddle.static as static
                                >>> paddle.enable_static()

                                >>> build_strategy = static.BuildStrategy()
                                >>> build_strategy.fuse_broadcast_ops = True
          )DOC")
      .def_property(
          "fuse_all_optimizer_ops",
          [](const BuildStrategy &self) {
            return self.fuse_all_optimizer_ops_ == true ||
                   self.fuse_all_optimizer_ops_ == paddle::none;
          },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, "
                                  "cannot be configured again."));
            self.fuse_all_optimizer_ops_ = b;
          })
      .def_property(
          "sync_batch_norm",
          [](const BuildStrategy &self) { return self.sync_batch_norm_; },
          [](BuildStrategy &self, bool b) {
            PADDLE_ENFORCE_NE(self.IsFinalized(),
                              true,
                              common::errors::PreconditionNotMet(
                                  "BuildStrategy has been finalized, cannot be "
                                  "configured again."));
            self.sync_batch_norm_ = b;
          },
          R"DOC((bool, optional): sync_batch_norm indicates whether to use
                synchronous batch normalization which synchronizes the mean
                and variance through multi-devices in training phase.
                Current implementation doesn't support FP16 training and CPU.
                And only synchronous on one machine, not all machines.
                Default is False.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.sync_batch_norm = True
          )DOC")
      .def_property(
          "memory_optimize",
          [](const BuildStrategy &self) -> py::object {
            if (self.memory_optimize_) {  // NOLINT
              return py::cast(self.memory_optimize_.get());
            } else {
              return py::cast(nullptr);
            }
          },
          [](BuildStrategy &self, const py::handle &value) {
            auto *py_obj = value.ptr();
            if (py_obj == nullptr || py_obj == Py_None) {
              self.memory_optimize_ = paddle::none;
            } else if (PyBool_Check(py_obj)) {
              self.memory_optimize_ = (py_obj == Py_True);
            } else {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "BuildStrategy.memory_optimize must be set to None, False "
                  "or True"));
            }
          },
          R"DOC((bool, optional): memory optimize aims to save total memory
                consumption, set to True to enable it.

                Default None. None means framework would choose to use or not use
                this strategy automatically. Currently, None means that it is
                enabled when GC is disabled, and disabled when GC is enabled.
                True means enabling and False means disabling. Default is None.

                Examples:
                    .. code-block:: python

                        >>> import paddle
                        >>> import paddle.static as static

                        >>> paddle.enable_static()

                        >>> build_strategy = static.BuildStrategy()
                        >>> build_strategy.memory_optimize = True

          )DOC")
      .def_property(
          "async_mode",
          [](const BuildStrategy &self) { return self.async_mode_; },
          [](BuildStrategy &self, bool b) { self.async_mode_ = b; })
      .def_property(
          "enable_inplace",
          [](const BuildStrategy &self) { return self.enable_inplace_; },
          [](BuildStrategy &self, bool b) { self.enable_inplace_ = b; })
      .def_property(
          "enable_addto",
          [](const BuildStrategy &self) { return self.enable_addto_; },
          [](BuildStrategy &self, bool b) { self.enable_addto_ = b; })
      .def_property(
          "fuse_all_reduce_ops",
          [](const BuildStrategy &self) {
            return self.fuse_all_reduce_ops_ == true ||
                   self.fuse_all_reduce_ops_ == paddle::none;
          },
          [](BuildStrategy &self, bool b) { self.fuse_all_reduce_ops_ = b; })
      .def_property(
          "enable_backward_optimizer_op_deps",
          [](const BuildStrategy &self) {
            return self.enable_backward_optimizer_op_deps_;
          },
          [](BuildStrategy &self, bool b) {
            self.enable_backward_optimizer_op_deps_ = b;
          })
      .def_property(
          "cache_runtime_context",
          [](const BuildStrategy &self) { return self.cache_runtime_context_; },
          [](BuildStrategy &self, bool b) { self.cache_runtime_context_ = b; })
      .def_property(
          "mkldnn_enabled_op_types",
          [](const BuildStrategy &self) {
            return self.mkldnn_enabled_op_types_;
          },
          [](BuildStrategy &self,
             const std::unordered_set<std::string> &mkldnn_enabled_op_types) {
            self.mkldnn_enabled_op_types_ = mkldnn_enabled_op_types;
          })
      .def_property(
          "allow_cuda_graph_capture",
          [](const BuildStrategy &self) {
            return self.allow_cuda_graph_capture_;
          },
          [](BuildStrategy &self, bool allow_cuda_graph_capture) {
            self.allow_cuda_graph_capture_ = allow_cuda_graph_capture;
          })
      .def("_copy",
           [](const BuildStrategy &self) {
             auto new_bs = self;
             new_bs.ClearFinalized();
             return new_bs;
           })
      .def("__str__",
           [](const BuildStrategy &self) {
             std::stringstream ss;
             ss << self;
             return ss.str();
           })
      .def(
          "_finalize_strategy_and_create_passes",
          [](BuildStrategy &self) -> std::shared_ptr<ir::PassBuilder> {
            return self.CreatePassesFromStrategy(true);
          },
          R"DOC(Allow user to customized passes. Normally model-specific
                optimization passes should be defined in this way. BuildStrategy
                cannot be updated after being finalized.)DOC");

  cp.def(py::init<const std::vector<phi::Place> &,
                  const std::vector<std::string> &,
                  const std::string &,
                  Scope *,
                  std::vector<Scope *> &,
                  const BuildStrategy &,
                  ir::Graph *>())
      // NOTE: even we return a vec<Scope*>* to Python use reference policy.
      // We still cannot get local_scope from this vector, since the element
      // of vec<Scope*> will be freed by Python GC. We can only return Scope*
      // one by one and mark them as reference.
      .def(
          "local_scopes",
          [](CompiledProgram &self) -> std::vector<Scope *> * {
            return &self.GetLocalScopes();
          },
          py::return_value_policy::reference);
  using VarQuantScale =
      std::unordered_map<std::string, std::pair<bool, phi::DenseTensor>>;
  py::class_<ir::Pass, std::shared_ptr<ir::Pass>> pass(m, "Pass");
  pass.def(py::init())
      .def("has", &ir::Pass::Has)
      .def("set_not_owned",
           [](ir::Pass &self, const std::string &attr_name, ProgramDesc &attr) {
             self.SetNotOwned<ProgramDesc>(attr_name, &attr);
           })
      .def(
          "set",
          [](ir::Pass &self, const std::string &name, const std::string &attr) {
            self.Set<std::string>(name, new std::string(attr));
          })
      .def("set",
           [](ir::Pass &self, const std::string &name, bool val) {
             self.Set<bool>(name, new bool(val));
           })
      .def("set",
           [](ir::Pass &self, const std::string &name, int val) {
             self.Set<const int>(name, new int(val));
           })
      .def("set",
           [](ir::Pass &self,
              const std::string &name,
              std::vector<std::string> set) {
             self.Set(name, new std::vector<std::string>(set));
           })
      .def("set",
           [](ir::Pass &self,
              const std::string &name,
              std::unordered_set<std::string> set) {
             self.Set(name, new std::unordered_set<std::string>(set));
           })
      .def("set",
           [](ir::Pass &self,
              const std::string &name,
              std::unordered_set<int> set) {
             self.Set(name, new std::unordered_set<int>(set));
           })
      .def("set",
           [](ir::Pass &self, const std::string &name, VarQuantScale scales) {
             self.Set(name, new VarQuantScale(scales));
           })
      .def("type", &ir::Pass::Type)
      .def("apply", [](ir::Pass &self, std::shared_ptr<ir::Graph> graph) {
        self.Apply(graph.get());
      });

  py::class_<ir::PassBuilder, std::shared_ptr<ir::PassBuilder>> pb(
      m, "PassBuilder");
  pb.def(py::init())
      .def("append_pass",
           [](ir::PassBuilder &self,
              const std::string &pass_type) -> std::shared_ptr<ir::Pass> {
             return self.AppendPass(pass_type);
           })
      .def("all_passes", [](ir::PassBuilder &self) { return self.AllPasses(); })
      .def("insert_pass",
           [](ir::PassBuilder &self, size_t idx, const std::string &pass_type) {
             return self.InsertPass(idx, pass_type);
           })
      .def("remove_pass",
           [](ir::PassBuilder &self, size_t idx) { self.RemovePass(idx); });
}

}  // namespace pybind
}  // namespace paddle
