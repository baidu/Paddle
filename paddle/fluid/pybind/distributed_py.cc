/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fcntl.h>
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/distributed/collective/reducer.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/pybind/distributed_py.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/process_group_utils.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/distributed/types.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/async_load.h"
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#endif

#if defined(PADDLE_WITH_MPI)
#include "paddle/fluid/distributed/collective/process_group_mpi.h"
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/fluid/distributed/collective/process_group_custom.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include "paddle/fluid/distributed/collective/process_group_gloo.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/distributed/collective/process_group_bkcl.h"
#endif

#if defined(PADDLE_WITH_UCC)
#include "paddle/fluid/distributed/collective/process_group_ucc.h"
#endif

#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

namespace paddle::pybind {

using Tensor = paddle::Tensor;

std::shared_ptr<distributed::EagerReducer> CreateEagerReducer(
    py::handle py_tensors,
    const std::vector<std::vector<size_t>> &group_indices,
    const std::vector<bool> &is_sparse_gradient,
    std::shared_ptr<distributed::ProcessGroup> process_group,
    const std::vector<size_t> &group_size_limits,
    bool find_unused_parameters) {
  auto params = CastPyArg2VectorOfTensor(py_tensors.ptr(), 0);
  return std::make_shared<distributed::EagerReducer>(params,
                                                     group_indices,
                                                     is_sparse_gradient,
                                                     process_group,
                                                     group_size_limits,
                                                     find_unused_parameters);
}

#if defined(PADDLE_WITH_GLOO)
using ProcessGroupGloo = paddle::distributed::ProcessGroupGloo;
using GlooStore = paddle::distributed::ProcessGroupGloo::GlooStore;
using GlooOptions = paddle::distributed::ProcessGroupGloo::GlooOptions;
#endif

static UNUSED void *use_ccl_comm_func =
    phi::detail::GetCCLComm(phi::CPUPlace());

void BindDistributed(py::module *m) {
  py::enum_<distributed::ReduceOp>(*m, "ReduceOp")
      .value("SUM", distributed::ReduceOp::SUM)
      .value("AVG", distributed::ReduceOp::AVG)
      .value("MAX", distributed::ReduceOp::MAX)
      .value("MIN", distributed::ReduceOp::MIN)
      .value("PRODUCT", distributed::ReduceOp::PRODUCT);

  py::class_<distributed::AllreduceOptions>(*m, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduce_op", &distributed::AllreduceOptions::reduce_op);

  py::class_<distributed::BroadcastOptions>(*m, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("source_rank", &distributed::BroadcastOptions::source_rank)
      .def_readwrite("source_root",
                     &distributed::BroadcastOptions::source_root);

  py::class_<distributed::BarrierOptions>(*m, "BarrierOptions")
      .def(py::init<>())
      .def_readwrite("device_id", &distributed::BarrierOptions::device_id);

  py::class_<distributed::ReduceOptions>(*m, "ReduceOptions")
      .def(py::init<>())
      .def_readwrite("reduce_op", &distributed::ReduceOptions::reduce_op)
      .def_readwrite("source_root", &distributed::ReduceOptions::root_rank);

  py::class_<distributed::GatherOptions>(*m, "GatherOptions")
      .def(py::init<>())
      .def_readwrite("root_rank", &distributed::GatherOptions::root_rank);

  auto ProcessGroup =
      py::class_<distributed::ProcessGroup,
                 std::shared_ptr<distributed::ProcessGroup>>(*m, "ProcessGroup")
          .def("rank",
               &distributed::ProcessGroup::GetRank,
               py::call_guard<py::gil_scoped_release>())
          .def("size",
               &distributed::ProcessGroup::GetSize,
               py::call_guard<py::gil_scoped_release>())
          .def("name",
               &distributed::ProcessGroup::GetBackendName,
               py::call_guard<py::gil_scoped_release>())
          .def("_start_coalescing",
               &distributed::ProcessGroup::StartCoalescing,
               py::call_guard<py::gil_scoped_release>())
          .def("_end_coalescing",
               &distributed::ProcessGroup::EndCoalescing,
               py::arg("tasks") = std::nullopt,
               py::call_guard<py::gil_scoped_release>())
          .def("eager_connect",
               &distributed::ProcessGroup::EagerConnect,
               py::call_guard<py::gil_scoped_release>())
          .def("eager_connect_ring_exchange",
               &distributed::ProcessGroup::EagerConnectRingExchange,
               py::call_guard<py::gil_scoped_release>())
          .def(
              "all_reduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::AllreduceOptions opts{op};
                return self.AllReduce(out_dense, in_dense, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("op"),
              py::arg("sync_op"))

          .def(
              "broadcast",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::BroadcastOptions opts{src};
                return self.Broadcast(out_dense, in_dense, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("sync_op"))

          .def(
              "send",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto out_dense = *p_dense;
                return self.Send(out_dense, dst, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("sync_op"))

          .def(
              "send_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst_rank,
                 int nranks,
                 int rank_id,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto out_dense = *p_dense;

                int64_t numel = p_dense->numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;
                return self.Send(
                    out_dense, dst_rank, offset, send_numel, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("num"),
              py::arg("id"),
              py::arg("sync_op") = true)

          .def(
              "recv",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *in_dense = p_dense.get();
                return self.Recv(in_dense, src, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("sync_op"))
          .def(
              "recv_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src_rank,
                 int nranks,
                 int rank_id,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();

                int64_t numel = p_dense->numel();
                int64_t recv_numel = numel / nranks;
                int64_t offset = recv_numel * rank_id;

                return self.Recv(
                    out_dense, src_rank, offset, recv_numel, sync_op);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("num"),
              py::arg("id"),
              py::arg("sync_op") = true)

          .def(
              "all_gather",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor_list,
                 py::handle py_in_tensor,
                 bool sync_op) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                Tensor stack_out_tensor = paddle::stack(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                auto task = self.AllGather(out_dense, in_dense, sync_op);
                auto *dev_ctx = self.GetDeviceContext(in_tensor.place());
                SplitTensor(*dev_ctx, *out_dense, &out_tensor_list);
                task->UpdateWaitChain(*dev_ctx);
                return task;
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("sync_op"))

          .def(
              "all_gather_into_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllGather(out_dense, in_dense, sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("sync_op"))

          .def(
              "all_to_all",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor_list,
                 py::handle py_in_tensor_list,
                 bool sync_op) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                py::gil_scoped_release release;

                Tensor stack_out_tensor = paddle::stack(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                Tensor stack_in_tensor = paddle::stack(in_tensor_list, 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_in_tensor.impl());
                auto in_dense = *p_in_tensor;

                // in_tensor_list should not be empty
                int world_size = self.GetSize();
                auto task =
                    self.AllToAll(out_dense,
                                  in_dense,
                                  GetDefaultSplitSizes(*out_dense, world_size),
                                  GetDefaultSplitSizes(in_dense, world_size),
                                  sync_op);
                auto *dev_ctx =
                    self.GetDeviceContext(in_tensor_list.back().place());
                SplitTensor(*dev_ctx, *out_dense, &out_tensor_list);
                task->UpdateWaitChain(*dev_ctx);
                return task;
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("sync_op"))

          .def(
              "all_to_all_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                int world_size = self.GetSize();

                return self.AllToAll(
                    out_dense,
                    in_dense,
                    GetDefaultSplitSizes(*out_dense, world_size),
                    GetDefaultSplitSizes(in_dense, world_size),
                    sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("sync_op"))

          .def(
              "all_to_all_single",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 const std::vector<int64_t> &out_sizes,
                 const std::vector<int64_t> &in_sizes,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllToAll(
                    out_dense, in_dense, out_sizes, in_sizes, sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("out_sizes"),
              py::arg("in_sizes"),
              py::arg("sync_op"))

          .def(
              "reduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::ReduceOptions opts{op, dst};
                return self.Reduce(out_dense, in_dense, opts, sync_op);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("op"),
              py::arg("sync_op"))

          .def(
              "reduce_scatter",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor_list,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto out_dense = p_out_tensor.get();

                Tensor stack_in_tensor = paddle::stack(in_tensor_list, 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(out_dense, in_dense, opts, sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("op"),
              py::arg("sync_op"))

          .def(
              "reduce_scatter_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 distributed::ReduceOp op,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(out_dense, in_dense, opts, sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("op"),
              py::arg("sync_op"))

          .def(
              "scatter",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor_list,
                 int src,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();
                Tensor stack_in_tensor = paddle::stack(in_tensor_list, 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ScatterOptions opts{src};
                return self.Scatter(out_dense, in_dense, opts, sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("src"),
              py::arg("sync_op"))

          .def(
              "scatter_tensor",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 int src,
                 bool sync_op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ScatterOptions opts{src};
                return self.Scatter(out_dense, in_dense, opts, sync_op);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("src"),
              py::arg("sync_op"))
          .def(
              "gather",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_gather_tensor_list,
                 int dst,
                 bool sync_op,
                 bool use_calc_stream) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_gather_tensor_list.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                Tensor stack_out_tensor = paddle::stack(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::GatherOptions gather_opts{dst};
                auto task = self.Gather(
                    out_dense, in_dense, gather_opts, sync_op, use_calc_stream);
                auto *dev_ctx =
                    self.GetDeviceContext(in_tensor.place(), use_calc_stream);
                SplitTensor(*dev_ctx, *out_dense, &out_tensor_list);
                if (!use_calc_stream &&
                    dev_ctx->GetPlace() != phi::CPUPlace()) {
                  // calculate stream will wait comm stream
                  task->UpdateWaitChain(*dev_ctx);
                }
                return task;
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("dst"),
              py::arg("sync_op"),
              py::arg("use_calc_stream") = false)
          .def(
              "barrier",
              [](distributed::ProcessGroup &self, int8_t device_id) {
                distributed::BarrierOptions opts{};
                opts.device_id = device_id;
                return self.Barrier(opts);
              },
              py::arg("device_id") = -1,
              py::call_guard<py::gil_scoped_release>())

          // TODO(liyurui): Interface below will be removed in the future.
          .def(
              "allreduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                distributed::AllreduceOptions opts{};
                opts.reduce_op = op;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                return self.AllReduce(dense.get(), *dense, opts, false);
              },
              py::arg("tensor"),
              py::arg("op") = distributed::ReduceOp::SUM)

          .def(
              "broadcast",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int source_rank) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                distributed::BroadcastOptions opts{};
                opts.source_rank = source_rank;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                return self.Broadcast(dense.get(), *dense, opts, false);
              },
              py::arg("tensor"),
              py::arg("source_rank"))

          .def(
              "send",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                return self.Send(*dense, dst, false);
              },
              py::arg("tensor"),
              py::arg("dst"))

          .def(
              "recv",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                return self.Recv(dense.get(), src, false);
              },
              py::arg("tensor"),
              py::arg("src"))

          .def(
              "all_gather",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                return self.AllGather(out_dense.get(), *in_dense, false);
              },
              py::arg("in"),
              py::arg("out"))

          .def(
              "all_gather_partial",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 int nranks,
                 int rank_id) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                int64_t numel = in_dense.numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;
                return self.AllGather(
                    out_dense, in_dense, offset, send_numel, /*sync_op*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("num"),
              py::arg("id"))

          .def(
              "alltoall",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());

                int world_size = self.GetSize();
                return self.AllToAll(
                    out_dense.get(),
                    *in_dense,
                    GetDefaultSplitSizes(*out_dense, world_size),
                    GetDefaultSplitSizes(*in_dense, world_size),
                    false);
              },
              py::arg("in"),
              py::arg("out"))

          .def(
              "alltoall_single",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 const std::vector<int64_t> in_sizes,
                 const std::vector<int64_t> out_sizes) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllToAll(
                    out_dense, in_dense, out_sizes, in_sizes, /*sync_op*/ true);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("in_sizes"),
              py::arg("out_sizes"))

          .def(
              "reduce",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 int dst,
                 distributed::ReduceOp op) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;
                distributed::ReduceOptions opts{};
                opts.reduce_op = op;
                opts.root_rank = dst;
                auto dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                return self.Reduce(dense.get(), *dense, opts, false);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("op") = distributed::ReduceOp::SUM)

          .def(
              "scatter",
              [](distributed::ProcessGroup &self,
                 py::handle py_in_tensor,
                 py::handle py_out_tensor,
                 int src) {
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                py::gil_scoped_release release;
                distributed::ScatterOptions opts{};
                opts.root_rank = src;
                auto in_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto out_dense = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                return self.Scatter(out_dense.get(), *in_dense, opts, false);
              },
              py::arg("in"),
              py::arg("out"),
              py::arg("src"))

          .def(
              "all_gather_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor_list,
                 py::handle py_in_tensor) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                Tensor stack_out_tensor = paddle::stack(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;
                auto task = self.AllGather(out_dense,
                                           in_dense,
                                           /*sync_op*/ true,
                                           /*use_calc_stream*/ true);
                auto *dev_ctx = self.GetDeviceContext(in_tensor.place(), true);
                SplitTensor(*dev_ctx, *out_dense, &out_tensor_list);
                return task;
              },
              py::arg("out"),
              py::arg("in"))

          .def(
              "all_gather_into_tensor_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllGather(out_dense,
                                      in_dense,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"))

          .def(
              "all_gather_partial_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 int nranks,
                 int rank_id) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                int64_t numel = in_dense.numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;

                return self.AllGather(out_dense,
                                      in_dense,
                                      offset,
                                      send_numel,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("num"),
              py::arg("id"))

          .def(
              "all_reduce_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 distributed::ReduceOp op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto in_dense = *p_dense;
                auto *out_dense = p_dense.get();
                distributed::AllreduceOptions opts{op};
                return self.AllReduce(out_dense,
                                      in_dense,
                                      opts,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("op") = distributed::ReduceOp::SUM)

          .def(
              "all_to_all_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor_list,
                 py::handle py_in_tensor_list) {
                auto out_tensor_list =
                    CastPyArg2VectorOfTensor(py_out_tensor_list.ptr(), 0);
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                py::gil_scoped_release release;

                Tensor stack_out_tensor = paddle::stack(out_tensor_list, 0);
                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                Tensor stack_in_tensor = paddle::stack(in_tensor_list, 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_in_tensor.impl());
                auto in_dense = *p_in_tensor;

                // in_tensor_list should not be empty
                int world_size = self.GetSize();
                auto task =
                    self.AllToAll(out_dense,
                                  in_dense,
                                  GetDefaultSplitSizes(*out_dense, world_size),
                                  GetDefaultSplitSizes(in_dense, world_size),
                                  /*sync_op*/ true,
                                  /*use_calc_stream*/ true);
                auto *dev_ctx = self.GetDeviceContext(
                    in_tensor_list.back().place(), /*use_calc_stream*/ true);
                SplitTensor(*dev_ctx, *out_dense, &out_tensor_list);
                return task;
              },
              py::arg("out"),
              py::arg("in"))

          .def(
              "all_to_all_tensor_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                int world_size = self.GetSize();
                return self.AllToAll(
                    out_dense,
                    in_dense,
                    GetDefaultSplitSizes(*out_dense, world_size),
                    GetDefaultSplitSizes(in_dense, world_size),
                    /*sync_op*/ true,
                    /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"))

          .def(
              "all_to_all_single_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 const std::vector<int64_t> &out_sizes,
                 const std::vector<int64_t> &in_sizes) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                return self.AllToAll(out_dense,
                                     in_dense,
                                     out_sizes,
                                     in_sizes,
                                     /*sync_op*/ true,
                                     /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("out_sizes"),
              py::arg("in_sizes"))

          .def(
              "broadcast_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::BroadcastOptions opts{src};
                return self.Broadcast(out_dense,
                                      in_dense,
                                      opts,
                                      /*sync_op*/ true,
                                      /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("src"))

          .def(
              "reduce_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst,
                 distributed::ReduceOp op) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();
                auto in_dense = *p_dense;
                distributed::ReduceOptions opts{op, dst};
                return self.Reduce(out_dense,
                                   in_dense,
                                   opts,
                                   /*sync_op*/ true,
                                   /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("op"))

          .def(
              "reduce_scatter_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor_list,
                 distributed::ReduceOp op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto out_dense = p_out_tensor.get();

                Tensor stack_in_tensor = paddle::stack(in_tensor_list, 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(out_dense,
                                          in_dense,
                                          opts,
                                          /*sync_op*/ true,
                                          /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("op"))

          .def(
              "reduce_scatter_tensor_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 distributed::ReduceOp op) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ReduceScatterOptions opts{op};
                return self.ReduceScatter(out_dense,
                                          in_dense,
                                          opts,
                                          /*sync_op*/ true,
                                          /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("op"))

          .def(
              "scatter_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor_list,
                 int src) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor_list =
                    CastPyArg2VectorOfTensor(py_in_tensor_list.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                Tensor stack_in_tensor = paddle::stack(in_tensor_list, 0);
                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    stack_in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ScatterOptions opts{src};
                return self.Scatter(out_dense,
                                    in_dense,
                                    opts,
                                    /*sync_op*/ true,
                                    /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("src"))

          .def(
              "scatter_tensor_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_out_tensor,
                 py::handle py_in_tensor,
                 int src) {
                auto out_tensor = CastPyArg2Tensor(py_out_tensor.ptr(), 0);
                auto in_tensor = CastPyArg2Tensor(py_in_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_out_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    out_tensor.impl());
                auto *out_dense = p_out_tensor.get();

                auto p_in_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    in_tensor.impl());
                auto in_dense = *p_in_tensor;

                distributed::ScatterOptions opts{src};
                return self.Scatter(out_dense,
                                    in_dense,
                                    opts,
                                    /*sync_op*/ true,
                                    /*use_calc_stream*/ true);
              },
              py::arg("out"),
              py::arg("in"),
              py::arg("src"))

          .def(
              "send_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto out_dense = *p_dense;
                return self.Send(out_dense,
                                 dst,
                                 /*sync_op*/ true,
                                 /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("dst"))

          .def(
              "send_partial_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int dst_rank,
                 int nranks,
                 int rank_id) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto out_dense = *p_dense;

                int64_t numel = p_dense->numel();
                int64_t send_numel = numel / nranks;
                int64_t offset = send_numel * rank_id;

                return self.Send(out_dense,
                                 dst_rank,
                                 offset,
                                 send_numel,
                                 /*sync_op*/ true,
                                 /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("dst"),
              py::arg("num"),
              py::arg("id"))

          .def(
              "recv_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *in_dense = p_dense.get();
                return self.Recv(in_dense,
                                 src,
                                 /*sync_op*/ true,
                                 /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("src"))

          .def(
              "recv_partial_on_calc_stream",
              [](distributed::ProcessGroup &self,
                 py::handle py_tensor,
                 int src_rank,
                 int nranks,
                 int rank_id) {
                auto tensor = CastPyArg2Tensor(py_tensor.ptr(), 0);
                py::gil_scoped_release release;
                auto p_dense =
                    std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
                auto *out_dense = p_dense.get();

                int64_t numel = p_dense->numel();
                int64_t recv_numel = numel / nranks;
                int64_t offset = recv_numel * rank_id;

                return self.Recv(out_dense,
                                 src_rank,
                                 offset,
                                 recv_numel,
                                 /*sync_op*/ true,
                                 /*use_calc_stream*/ true);
              },
              py::arg("tensor"),
              py::arg("src"),
              py::arg("num"),
              py::arg("id"));

#if defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)
  py::class_<distributed::ProcessGroupNCCL,
             std::shared_ptr<distributed::ProcessGroupNCCL>>(
      *m, "ProcessGroupNCCL", ProcessGroup)
      .def_static("create",
                  distributed::ProcessGroupNCCL::CreateProcessGroupNCCL,
                  py::arg("store"),
                  py::arg("rank"),
                  py::arg("world_size"),
                  py::arg("group_id") = 0,
                  py::arg("timeout") = 30 * 60 * 1000,
                  py::arg("nccl_comm_init_option") = 0,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("group_start", distributed::ProcessGroupNCCL::GroupStart)
      .def_static("group_end", distributed::ProcessGroupNCCL::GroupEnd);

  py::class_<distributed::AsyncLoad::Task,
             std::shared_ptr<distributed::AsyncLoad::Task>>(*m, "AsyncLoadTask")
      .def("is_completed",
           &distributed::AsyncLoad::Task::IsCompleted,
           py::call_guard<py::gil_scoped_release>())
      .def("cuda_wait",
           &distributed::AsyncLoad::Task::CudaSynchronize,
           py::call_guard<py::gil_scoped_release>())
      .def("cpu_wait",
           &distributed::AsyncLoad::Task::CpuSynchronize,
           py::call_guard<py::gil_scoped_release>());

  auto AsyncLoad =
      py::class_<distributed::AsyncLoad>(*m, "AsyncLoad")
          .def(py::init<>())
          .def(
              "offload",
              [](distributed::AsyncLoad &self,
                 py::handle py_dst_tensor,
                 py::handle py_src_tensor) {
                auto dst_tensor = CastPyArg2Tensor(py_dst_tensor.ptr(), 0);
                auto src_tensor = CastPyArg2Tensor(py_src_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_dst_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    dst_tensor.impl());
                auto *dst_dense = p_dst_tensor.get();

                auto p_src_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    src_tensor.impl());
                auto src_dense = *p_src_tensor;

                return self.Offload(dst_dense, src_dense);
              },
              py::arg("dst"),
              py::arg("src"))
          .def(
              "offload_with_offset",
              [](distributed::AsyncLoad &self,
                 py::handle py_dst_tensor,
                 py::handle py_src_tensor,
                 size_t dst_offset,
                 size_t src_offset,
                 size_t offload_size) {
                auto dst_tensor = CastPyArg2Tensor(py_dst_tensor.ptr(), 0);
                auto p_dst_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    dst_tensor.impl());
                auto *dst_dense = p_dst_tensor.get();

                auto src_tensor = CastPyArg2Tensor(py_src_tensor.ptr(), 0);
                auto p_src_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    src_tensor.impl());
                auto src_dense = *p_src_tensor;

                return self.OffloadWithOffset(
                    dst_dense, src_dense, dst_offset, src_offset, offload_size);
              },
              py::arg("dst"),
              py::arg("src"),
              py::arg("dst_offset"),
              py::arg("src_offset"),
              py::arg("offload_size"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reload",
              [](distributed::AsyncLoad &self,
                 py::handle py_dst_tensor,
                 py::handle py_src_tensor) {
                auto dst_tensor = CastPyArg2Tensor(py_dst_tensor.ptr(), 0);
                auto src_tensor = CastPyArg2Tensor(py_src_tensor.ptr(), 0);
                py::gil_scoped_release release;

                auto p_dst_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    dst_tensor.impl());
                auto *dst_dense = p_dst_tensor.get();

                auto p_src_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
                    src_tensor.impl());
                auto src_dense = *p_src_tensor;

                return self.Reload(dst_dense, src_dense);
              },
              py::arg("dst"),
              py::arg("src"));

#endif

#if defined(PADDLE_WITH_MPI)
  py::class_<distributed::ProcessGroupMPI,
             std::shared_ptr<distributed::ProcessGroupMPI>>(
      *m, "ProcessGroupMPI", ProcessGroup)
      .def_static(
          "create",
          [](const std::vector<int> &ranks,
             int gid) -> std::shared_ptr<distributed::ProcessGroupMPI> {
            return paddle::distributed::ProcessGroupMPI::CreateProcessGroupMPI(
                ranks, gid);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_rank",
           &distributed::ProcessGroup::GetRank,
           py::call_guard<py::gil_scoped_release>())
      .def("get_world_size",
           &distributed::ProcessGroup::GetSize,
           py::call_guard<py::gil_scoped_release>());
#endif

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  py::class_<distributed::ProcessGroupCustom,
             std::shared_ptr<distributed::ProcessGroupCustom>>(
      *m, "ProcessGroupCustom", ProcessGroup)
      .def_static("create",
                  distributed::ProcessGroupCustom::CreateProcessGroupCustom,
                  py::arg("store"),
                  py::arg("device_type"),
                  py::arg("rank"),
                  py::arg("world_size"),
                  py::arg("group_id") = 0,
                  py::return_value_policy::reference_internal,
                  py::call_guard<py::gil_scoped_release>())
      .def("get_comm_name",
           &distributed::ProcessGroupCustom::GetCommName,
           py::arg("rank"),
           py::call_guard<py::gil_scoped_release>());

#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  auto processGroupBKCL =
      py::class_<distributed::ProcessGroupBKCL,
                 std::shared_ptr<distributed::ProcessGroupBKCL>>(
          *m, "ProcessGroupBKCL", ProcessGroup)
          .def_static("create",
                      distributed::ProcessGroupBKCL::CreateProcessGroupBKCL,
                      py::arg("store"),
                      py::arg("rank"),
                      py::arg("world_size"),
                      py::arg("group_id") = 0,
                      py::call_guard<py::gil_scoped_release>())
          .def_static("group_start",
                      distributed::ProcessGroupBKCL::GroupStart,
                      py::call_guard<py::gil_scoped_release>())
          .def_static("group_end",
                      distributed::ProcessGroupBKCL::GroupEnd,
                      py::call_guard<py::gil_scoped_release>());
#endif

#if defined(PADDLE_WITH_UCC)
  auto processGroupUCC =
      py::class_<distributed::ProcessGroupUCC,
                 std::shared_ptr<distributed::ProcessGroupUCC>>(
          *m, "ProcessGroupUCC", ProcessGroup)
          .def_static("create",
                      distributed::ProcessGroupUCC::CreateProcessGroupUCC,
                      py::arg("store"),
                      py::arg("rank"),
                      py::arg("world_size"),
                      py::arg("group_id") = 0,
                      py::call_guard<py::gil_scoped_release>());
  //   .def_static("group_start",
  //               distributed::ProcessGroupBKCL::GroupStart,
  //               py::call_guard<py::gil_scoped_release>())
  //   .def_static("group_end",
  //               distributed::ProcessGroupBKCL::GroupEnd,
  //               py::call_guard<py::gil_scoped_release>());
#endif

  py::class_<distributed::ProcessGroup::Task,
             std::shared_ptr<distributed::ProcessGroup::Task>>(*m, "task")
      .def("is_completed",
           &distributed::ProcessGroup::Task::IsCompleted,
           py::call_guard<py::gil_scoped_release>())
      .def("is_sync",
           &distributed::ProcessGroup::Task::IsSync,
           py::call_guard<py::gil_scoped_release>())
      .def("wait",
           &distributed::ProcessGroup::Task::Wait,
           py::arg("timeout") = kWaitTimeout,
           py::call_guard<py::gil_scoped_release>())
      .def("synchronize",
           &distributed::ProcessGroup::Task::Synchronize,
           py::call_guard<py::gil_scoped_release>());

#if defined(PADDLE_WITH_GLOO)
  py::class_<ProcessGroupGloo, std::shared_ptr<ProcessGroupGloo>>(
      *m, "ProcessGroupGloo", ProcessGroup)
      .def_static("create",
                  distributed::ProcessGroupGloo::CreateProcessGroupGloo,
                  py::arg("store"),
                  py::arg("rank"),
                  py::arg("world_size"),
                  py::arg("group_id") = 0,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("create_default_device",
                  &ProcessGroupGloo::createDefaultDevice,
                  py::call_guard<py::gil_scoped_release>());
#endif

  m->def(
      "eager_assign_group_by_size",
      [](py::handle py_tensors,
         std::vector<bool> is_sparse_gradient,
         std::vector<size_t> group_size_limits,
         std::vector<int64_t> tensor_indices) {
        auto tensors = CastPyArg2VectorOfTensor(py_tensors.ptr(), 0);
        py::gil_scoped_release release;
        return distributed::Eager_AssignGroupBySize(
            tensors, is_sparse_gradient, group_size_limits, tensor_indices);
      },
      py::arg("tensors"),
      py::arg("is_sparse_gradient"),
      py::arg("group_size_limits") = std::vector<size_t>{25 * 1024 * 1024},
      py::arg("tensor_indices") = std::vector<int64_t>{});

  py::class_<distributed::EagerReducer,
             std::shared_ptr<distributed::EagerReducer>>(
      *m, "EagerReducer", R"DOC()DOC")
      .def(py::init(&CreateEagerReducer))
      .def(
          "prepare_for_backward",
          [](distributed::EagerReducer &self, py::handle py_tensors) {
            auto params = CastPyArg2VectorOfTensor(py_tensors.ptr(), 0);
            py::gil_scoped_release release;
            self.PrepareForBackward(params);
          },
          py::arg("tensors"));

  py::class_<distributed::ProcessGroupIdMap,
             std::shared_ptr<distributed::ProcessGroupIdMap>>(
      *m, "ProcessGroupIdMap")
      .def_static("destroy",
                  distributed::ProcessGroupIdMap::DestroyProcessGroup,
                  py::call_guard<py::gil_scoped_release>());
}

}  // namespace paddle::pybind
