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

#include "paddle/fluid/operators/custom_device_common_op_registry.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/operators/collective/c_concat_op.h"
#include "paddle/fluid/operators/load_combine_op.h"
#include "paddle/fluid/operators/save_combine_op.h"
#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

#define REGISTER_OP_CUSTOM_DEVICE_KERNEL(op_type, dev_type, ...)             \
  static paddle::framework::OpKernelRegistrar<phi::CustomPlace, __VA_ARGS__> \
      __op_custom_device_kernel_registrar_##op_type##_##__acosf##__(         \
          #op_type,                                                          \
          dev_type,                                                          \
          paddle::framework::OpKernelType::kDefaultCustomizedTypeValue);     \
  __op_custom_device_kernel_registrar_##op_type##_##__acosf##__.Touch();

#define REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL(                             \
    kernel_name, dev_type, layout, kernel_fn)                              \
  static phi::KernelRegistrar                                              \
      __reg_custom_device_phi_kernel_##kernel_name##_##backend##_##layout( \
          phi::RegType::INNER,                                             \
          #kernel_name,                                                    \
          dev_type,                                                        \
          DATA_LAYOUT(layout),                                             \
          ::phi::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,      \
          [](const phi::KernelKey& kernel_key, phi::Kernel* kernel) {},    \
          PHI_KERNEL(kernel_fn),                                           \
          PHI_VARIADIC_KERNEL(kernel_fn))

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CConcatOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();

    PADDLE_ENFORCE_GE(rank,
                      0,
                      common::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      common::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_concat must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      common::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

    auto& dev_ctx = ctx.template device_context<phi::CustomContext>();
    phi::DenseTensor temp_out;
    phi::DDim temp_out_dims = x->dims();
    temp_out_dims[0] *= nranks;
    temp_out.Resize(temp_out_dims);
    dev_ctx.template Alloc<T>(&temp_out);

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*x);
      out_tensor.push_back(temp_out);
      auto task = pg->AllGather(in_tensor, out_tensor);
      task->Wait();
    } else {
      auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
          phi::distributed::CommContextManager::GetInstance().Get(
              std::to_string(rid)));
      PADDLE_ENFORCE_EQ(
          nranks,
          comm->GetSize(),
          common::errors::InvalidArgument(
              "nranks: %s should equal to %s", nranks, comm->GetSize()));

      int64_t send_numel = x->numel();
      const T* send_buff = x->data<T>();
      T* recv_buff = temp_out.data<T>();
      // should ExecutionContext for calc stream.
      auto& stream = *dev_ctx.GetStream();
      phi::DeviceManager::CCLAllGather(
          place.GetDeviceType(),
          reinterpret_cast<void*>(const_cast<T*>(send_buff)),
          recv_buff,
          send_numel,
          x->dtype(),
          comm->GetXcclComm(),
          stream);
    }
    std::vector<phi::DenseTensor> inputs;
    int axis = x->dims().size() - 1;
    auto out_dims = x->dims();
    out_dims[out_dims.size() - 1] *= nranks;
    int rows_per_tensor = x->dims()[0];
    int offset = 0;
    for (int i = 0; i < nranks; i++) {
      phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
      inputs.emplace_back(temp);
      offset += rows_per_tensor;
    }

    out->Resize(out_dims);
    std::vector<paddle::Tensor> inputs_t(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      auto t = std::make_shared<phi::DenseTensor>();
      t->ShareDataWith(inputs[i]);
      inputs_t[i].set_impl(t);
    }
    auto output = paddle::experimental::concat(inputs_t, axis);
    out->ShareDataWith(
        *reinterpret_cast<phi::DenseTensor*>(output.impl().get()));
  }
};

template <typename DeviceContext, typename T>
class CIdentityOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int rid = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        rid,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for c_identity op must be non-negative.", rid));
    ctx.device_context().Alloc<T>(out);

    paddle::framework::TensorCopy(*x, out->place(), out);
  }
};

template <typename DeviceContext, typename T>
class CSplitOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");

    PADDLE_ENFORCE_GE(rank,
                      0,
                      common::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      common::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_split must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      common::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

    auto dims = x->dims();
    auto dims_size = dims.size();

    dims[dims_size - 1] /= nranks;
    out->Resize(dims);
    std::vector<int64_t> split_list(nranks, dims[dims_size - 1]);
    int axis = dims_size - 1;

    auto x_tmp = std::make_shared<phi::DenseTensor>();
    x_tmp->ShareDataWith(*x);
    paddle::Tensor x_tensor(x_tmp);
    auto outputs = paddle::experimental::split(x_tensor, split_list, axis);
    out->ShareDataWith(
        *reinterpret_cast<phi::DenseTensor*>(outputs[rank].impl().get()));
  }
};

template <typename DeviceContext, typename T>
class CEmbeddingOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* ids_t = ctx.Input<phi::DenseTensor>("Ids");
    auto* table_t = ctx.Input<phi::DenseTensor>("W");
    auto* output_t = ctx.Output<phi::DenseTensor>("Out");
    auto out_dims = output_t->dims();
    auto start_index = ctx.Attr<int64_t>("start_index");

    auto K = ids_t->numel();
    auto N = table_t->dims()[0];
    auto D = table_t->dims()[1];
    auto index_type = ids_t->dtype();
    if (index_type == phi::DataType::INT32 ||
        index_type == phi::DataType::INT64) {
      auto x_tmp = std::make_shared<phi::DenseTensor>();
      x_tmp->ShareDataWith(*ids_t).Resize({K});
      auto w_tmp = std::make_shared<phi::DenseTensor>();
      w_tmp->ShareDataWith(*table_t).Resize({N, D});
      paddle::Tensor x_tensor(x_tmp), w_tensor(w_tmp);
      auto start_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index, x_tensor.dtype(), x_tensor.place());
      auto end_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index + N, x_tensor.dtype(), x_tensor.place());
      auto ids_mask_tensor = paddle::experimental::logical_and(
          x_tensor.greater_equal(start_index_tensor),
          x_tensor.less_than(end_index_tensor));
      auto ids_tensor = (x_tensor - start_index_tensor)
                            .multiply(paddle::experimental::cast(
                                ids_mask_tensor, x_tensor.dtype()));
      auto out_tensor =
          paddle::experimental::reshape(
              paddle::experimental::cast(ids_mask_tensor, w_tensor.dtype()),
              {K, 1})
              .multiply(paddle::experimental::reshape(
                  paddle::experimental::embedding(
                      ids_tensor, w_tensor, -1, false),
                  {K, D}));
      output_t
          ->ShareDataWith(
              *reinterpret_cast<phi::DenseTensor*>(out_tensor.impl().get()))
          .Resize(out_dims);
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "CustomDevice c_embedding ids only support int32 or int64."));
    }
  }
};

template <typename DeviceContext, typename T>
class CEmbeddingGradOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto start_index = ctx.Attr<int64_t>("start_index");
    auto ids_t = ctx.Input<phi::DenseTensor>("Ids");
    auto d_output_t =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto table_t = ctx.Input<phi::DenseTensor>("W");
    auto table_grad_t =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("W"));
    table_grad_t->Resize(table_t->dims());
    auto& dev_ctx = ctx.template device_context<phi::CustomContext>();

    auto K = ids_t->numel();
    auto N = table_t->dims()[0];
    auto D = table_t->dims()[1];
    const auto& index_type = ids_t->dtype();
    if (index_type == phi::DataType::INT32 ||
        index_type == phi::DataType::INT64) {
      auto x_tmp = std::make_shared<phi::DenseTensor>();
      x_tmp->ShareDataWith(*ids_t).Resize({K});
      auto w_tmp = std::make_shared<phi::DenseTensor>();
      w_tmp->set_meta(table_t->meta());
      dev_ctx.Alloc(w_tmp.get(), w_tmp->dtype());
      auto out_grad_tmp = std::make_shared<phi::DenseTensor>();
      out_grad_tmp->ShareDataWith(*d_output_t).Resize({K, D});
      paddle::Tensor x_tensor(x_tmp), w_tensor(w_tmp),
          out_grad_tensor(out_grad_tmp);
      auto start_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index, x_tensor.dtype(), x_tensor.place());
      auto end_index_tensor = paddle::experimental::full_like(
          x_tensor, start_index + N, x_tensor.dtype(), x_tensor.place());
      auto ids_mask_tensor = paddle::experimental::logical_and(
          x_tensor.greater_equal(start_index_tensor),
          x_tensor.less_than(end_index_tensor));
      auto real_ids_tensor = (x_tensor - start_index_tensor)
                                 .multiply(paddle::experimental::cast(
                                     ids_mask_tensor, x_tensor.dtype()));
      auto out_grad_tensor_mul_mask =
          paddle::experimental::reshape(out_grad_tensor, {K, D})
              .multiply(paddle::experimental::reshape(
                  paddle::experimental::cast(ids_mask_tensor, table_t->dtype()),
                  {K, 1}));
      paddle::Tensor table_grad_tensor;
      paddle::experimental::embedding_grad(real_ids_tensor,
                                           w_tensor,
                                           out_grad_tensor_mul_mask,
                                           -1,
                                           false,
                                           &table_grad_tensor);
      table_grad_t->ShareDataWith(
          *reinterpret_cast<phi::DenseTensor*>(table_grad_tensor.impl().get()));
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "CustomDevice c_embedding ids only support int32 or int64."));
    }
  }
};

template <typename DeviceContext, typename T>
class CSoftmaxWithCrossEntropyOpCustomDeviceKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      const phi::DenseTensor* logits = ctx.Input<phi::DenseTensor>("Logits");
      const phi::DenseTensor* labels = ctx.Input<phi::DenseTensor>("Label");
      phi::DenseTensor* softmax = ctx.Output<phi::DenseTensor>("Softmax");
      phi::DenseTensor* loss = ctx.Output<phi::DenseTensor>("Loss");
      auto softmax_dims = softmax->dims();
      auto loss_dims = loss->dims();

      const int64_t ignore_index = ctx.Attr<int64_t>("ignore_index");
      const int rid = ctx.Attr<int>("ring_id");
      const int rank = ctx.Attr<int>("rank");

      distributed::ProcessGroup* pg = map->get(rid);
      distributed::AllreduceOptions opts;

      // allocate memory on device.
      const auto& logits_dims = logits->dims();

      const int axis = logits_dims.size() - 1;
      const int N = phi::funcs::SizeToAxis(axis, logits_dims);
      const int D = phi::funcs::SizeFromAxis(axis, logits_dims);

      auto logits_2d = std::make_shared<phi::DenseTensor>();
      auto labels_1d = std::make_shared<phi::DenseTensor>();
      logits_2d->ShareDataWith(*logits).Resize({N, D});
      labels_1d->ShareDataWith(*labels).Resize({N});
      paddle::Tensor logits_2d_tensor(logits_2d), labels_1d_tensor(labels_1d);

      // step 1, obtain logit_max
      auto logits_2d_max_tensor = logits_2d_tensor.max({1}, true);
      std::vector<phi::DenseTensor> in_out;
      in_out.push_back(*reinterpret_cast<phi::DenseTensor*>(
          logits_2d_max_tensor.impl().get()));
      opts.reduce_op = distributed::ReduceOp::MAX;
      pg->AllReduce(in_out, in_out, opts)->Synchronize();

      // step 2, obtain logit - logit_max
      auto logits_2d_sub_max = paddle::experimental::clip(
          logits_2d_tensor - logits_2d_max_tensor, -64., 0.);

      // step 3, obtain predict target
      const int start_index = rank * D;
      auto start_index_tensor =
          paddle::experimental::full_like(labels_1d_tensor,
                                          start_index,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place());
      auto end_index_tensor =
          paddle::experimental::full_like(labels_1d_tensor,
                                          start_index + D,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place());
      auto labels_1d_mask = paddle::experimental::logical_and(
          labels_1d_tensor.greater_equal(start_index_tensor),
          labels_1d_tensor.less_than(end_index_tensor));
      auto real_label_tensor =
          (labels_1d_tensor - start_index_tensor)
              .multiply(paddle::experimental::cast(labels_1d_mask,
                                                   labels_1d_tensor.dtype()));

      auto predicted_logits_tensor =
          logits_2d_sub_max
              .multiply(paddle::experimental::cast(
                  paddle::experimental::one_hot(real_label_tensor, D),
                  logits_2d_sub_max.dtype()))
              .sum({1}, logits_2d_sub_max.dtype(), false)
              .multiply(paddle::experimental::cast(labels_1d_mask,
                                                   logits_2d_sub_max.dtype()));

      in_out.clear();
      in_out.push_back(*reinterpret_cast<phi::DenseTensor*>(
          predicted_logits_tensor.impl().get()));
      opts.reduce_op = distributed::ReduceOp::SUM;
      pg->AllReduce(in_out, in_out, opts)->Synchronize();

      // step 4, obtain exp(logit)
      auto softmax_2d_tensor = logits_2d_sub_max.exp();

      // step 5, obtain sum_exp_logits
      auto sum_exp_logits_tensor =
          softmax_2d_tensor.sum({1}, softmax_2d_tensor.dtype(), false);

      in_out.clear();
      in_out.push_back(*reinterpret_cast<phi::DenseTensor*>(
          sum_exp_logits_tensor.impl().get()));
      opts.reduce_op = distributed::ReduceOp::SUM;
      pg->AllReduce(in_out, in_out, opts)->Synchronize();

      auto softmax_out = softmax_2d_tensor.divide(
          paddle::experimental::reshape(sum_exp_logits_tensor, {N, 1}));
      auto labels_1d_not_equal_ignore = labels_1d_tensor.not_equal(
          paddle::experimental::full_like(labels_1d_tensor,
                                          ignore_index,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place()));
      auto loss_out =
          (sum_exp_logits_tensor.log() - predicted_logits_tensor)
              .multiply(paddle::experimental::cast(
                  labels_1d_not_equal_ignore, sum_exp_logits_tensor.dtype()));
      softmax
          ->ShareDataWith(
              *reinterpret_cast<phi::DenseTensor*>(softmax_out.impl().get()))
          .Resize(softmax_dims);
      loss->ShareDataWith(
              *reinterpret_cast<phi::DenseTensor*>(loss_out.impl().get()))
          .Resize(loss_dims);
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "CustomDevice c_softmax_with_cross_entropy "
          "only support ProcessGroup"));
    }
  }
};

template <typename DeviceContext, typename T>
class CSoftmaxWithCrossEntropyGradCustomDeviceKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const phi::DenseTensor* labels = context.Input<phi::DenseTensor>("Label");
    const phi::DenseTensor* loss_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    const phi::DenseTensor* softmax =
        context.Input<phi::DenseTensor>("Softmax");
    phi::DenseTensor* logit_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Logits"));

    const int64_t ignore_index = context.Attr<int64_t>("ignore_index");
    const int rank = context.Attr<int>("rank");
    if (logit_grad != softmax) {
      framework::TensorCopy(
          *softmax, context.GetPlace(), context.device_context(), logit_grad);
    }
    const auto softmax_dims = softmax->dims();
    const int axis = softmax_dims.size() - 1;
    const int N = phi::funcs::SizeToAxis(axis, softmax_dims);
    const int D = phi::funcs::SizeFromAxis(axis, softmax_dims);
    const auto& label_type = labels->dtype();

    if (label_type == phi::DataType::INT32 ||
        label_type == phi::DataType::INT64) {
      auto logit_grad_t = std::make_shared<phi::DenseTensor>();
      logit_grad_t->ShareDataWith(*logit_grad).Resize({N, D});
      auto loss_grad_t = std::make_shared<phi::DenseTensor>();
      loss_grad_t->ShareDataWith(*loss_grad).Resize({N});
      auto labels_1d = std::make_shared<phi::DenseTensor>();
      labels_1d->ShareDataWith(*labels).Resize({N});
      paddle::Tensor logits_grad_tensor(logit_grad_t),
          loss_grad_tensor(loss_grad_t), labels_1d_tensor(labels_1d);

      auto labels_1d_not_equal_ignore = paddle::experimental::reshape(
          paddle::experimental::not_equal(
              labels_1d_tensor,
              paddle::experimental::full_like(labels_1d_tensor,
                                              ignore_index,
                                              labels_1d_tensor.dtype(),
                                              labels_1d_tensor.place())),
          {N, 1});
      auto start_index_tensor =
          paddle::experimental::full_like(labels_1d_tensor,
                                          rank * D,
                                          labels_1d_tensor.dtype(),
                                          labels_1d_tensor.place());

      auto logits_grad_out_tensor1 = paddle::experimental::subtract(
          paddle::experimental::multiply(
              logits_grad_tensor,
              paddle::experimental::cast(labels_1d_not_equal_ignore,
                                         logits_grad_tensor.dtype())),
          paddle::experimental::cast(
              paddle::experimental::one_hot(
                  paddle::experimental::subtract(labels_1d_tensor,
                                                 start_index_tensor),
                  D),
              logits_grad_tensor.dtype()));

      auto logits_grad_out_tensor2 = paddle::experimental::multiply(
          logits_grad_out_tensor1,
          paddle::experimental::reshape(loss_grad_tensor, {N, 1}));
      logit_grad
          ->ShareDataWith(*reinterpret_cast<phi::DenseTensor*>(
              logits_grad_out_tensor2.impl().get()))
          .Resize(softmax_dims);
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "CustomDevice c_softmax_with_cross_entropy_grad "
          "label_type only support int32/int64"));
    }
  }
};

template <typename DeviceContext, typename T>
class CSyncCalcStreamCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    auto dev_ctx = static_cast<DeviceContext*>(
        phi::DeviceContextPool::Instance().Get(place));
    dev_ctx->GetStream()->Synchronize();
  }
};

template <typename DeviceContext, typename T, phi::ccl::CCLReduceOp red_type>
class CAllReduceOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    if (ctx.HasInput("Cond")) {
      auto cond = ctx.Input<phi::DenseTensor>("Cond");
      auto place = cond->place();
      PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::CPU,
                        true,
                        common::errors::PreconditionNotMet(
                            "The input `cond` tensor should be on cpu place"));
      PADDLE_ENFORCE_EQ(cond->numel(),
                        1,
                        common::errors::PreconditionNotMet(
                            "The input `cond` should be shape [1]"));
      if (!cond->data<bool>()[0]) {
        VLOG(4) << "Skip all reduce Op since cond is 0";
        return;
      }
    }

    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int rid = ctx.Attr<int>("ring_id");

    auto place = ctx.GetPlace();
    auto dtype = in->dtype();
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = ctx.device_context().Alloc<T>(out);

    auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      paddle::distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*in);
      out_tensor.push_back(*out);

      paddle::distributed::AllreduceOptions opts;
      switch (red_type) {
        case phi::ccl::CCLReduceOp::SUM:
          opts.reduce_op = paddle::distributed::ReduceOp::SUM;
          break;

        case phi::ccl::CCLReduceOp::MAX:
          opts.reduce_op = paddle::distributed::ReduceOp::MAX;
          break;

        case phi::ccl::CCLReduceOp::MIN:
          opts.reduce_op = paddle::distributed::ReduceOp::MIN;
          break;

        case phi::ccl::CCLReduceOp::PRODUCT:
          opts.reduce_op = paddle::distributed::ReduceOp::PRODUCT;
          break;

        default:
          PADDLE_THROW(common::errors::InvalidArgument(
              "Invalid reduce type: %d", red_type));
      }

      auto task = pg->AllReduce(in_tensor, out_tensor, opts);
      task->Wait();
      return;
    }

    auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
        phi::distributed::CommContextManager::GetInstance().Get(
            std::to_string(rid)));

    std::shared_ptr<phi::stream::Stream> stream;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::CustomContext*>(dev_ctx)->GetStream();
    } else {
      stream = comm->GetStream();
    }
    phi::DeviceManager::CCLAllReduce(place.GetDeviceType(),
                                     const_cast<void*>(sendbuff),
                                     recvbuff,
                                     numel,
                                     dtype,
                                     red_type,
                                     comm->GetXcclComm(),
                                     *stream);
  }
};

template <typename T>
class CBroadcastOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    const auto& place = ctx.GetPlace();
    ctx.device_context().Alloc<T>(out);
    int root = ctx.Attr<int>("root");
    int rid = ctx.Attr<int>("ring_id");

    auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
        phi::distributed::CommContextManager::GetInstance().Get(
            std::to_string(rid)));

    std::shared_ptr<phi::stream::Stream> stream;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::CustomContext*>(dev_ctx)->GetStream();
    } else {
      stream = comm->GetStream();
    }

    int numel = x->numel();
    auto dtype = x->dtype();
    if (root == comm->GetRank()) {
      phi::DeviceManager::CCLBroadcast(place.GetDeviceType(),
                                       const_cast<void*>(x->data()),
                                       numel,
                                       dtype,
                                       root,
                                       comm->GetXcclComm(),
                                       *stream);
      VLOG(3) << "rank " << comm->GetRank() << " invoke Bcast. sent "
              << x->numel();
      if (out != x) {
        framework::TensorCopy(*static_cast<const phi::DenseTensor*>(x),
                              place,
                              *phi::DeviceContextPool::Instance().Get(place),
                              static_cast<phi::DenseTensor*>(out));
      }
    } else {
      phi::DeviceManager::CCLBroadcast(place.GetDeviceType(),
                                       out->data(),
                                       numel,
                                       dtype,
                                       root,
                                       comm->GetXcclComm(),
                                       *stream);
      VLOG(3) << "rank " << comm->GetRank() << " invoke Bcast. received "
              << common::product(out->dims());
    }
    out->set_lod(x->lod());
  }
};

template <typename T>
class BarrierOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    auto place = ctx.GetPlace();
    int64_t numel = in->numel();
    const void* sendbuff = in->data();
    void* recvbuff = ctx.device_context().Alloc<T>(out);
    int rid = ctx.Attr<int>("ring_id");

    auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
        phi::distributed::CommContextManager::GetInstance().Get(
            std::to_string(rid)));

    std::shared_ptr<phi::stream::Stream> stream;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::CustomContext*>(dev_ctx)->GetStream();
    } else {
      stream = comm->GetStream();
    }
    phi::DeviceManager::CCLAllReduce(place.GetDeviceType(),
                                     const_cast<void*>(sendbuff),
                                     recvbuff,
                                     numel,
                                     in->dtype(),
                                     phi::ccl::CCLReduceOp::SUM,
                                     comm->GetXcclComm(),
                                     *stream);
  }
};

template <typename T>
class NumberCountOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto numbers = context.Input<phi::DenseTensor>("numbers");
    auto upper_range = context.Attr<int>("upper_range");
    auto number_count = context.Output<phi::DenseTensor>("Out");
    const auto& dev_ctx = context.template device_context<phi::CustomContext>();
    number_count->Resize({upper_range});
    dev_ctx.template Alloc<T>(number_count);
    phi::DenseTensor cpu_tensor;
    framework::TensorCopySync(*numbers, phi::CPUPlace(), &cpu_tensor);
    std::vector<T> count(upper_range);
    for (auto i = 0; i < cpu_tensor.numel(); ++i) {
      auto idx = static_cast<int64_t>(cpu_tensor.data<T>()[i]);
      if (idx >= 0 && idx < upper_range) {
        count[idx] += 1;
      }
    }
    framework::TensorFromVector<T>(count, dev_ctx, number_count);
    number_count->Resize({upper_range});
  }
};

template <typename T>
class LimitByCapacityOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto expert_count = context.Input<phi::DenseTensor>("expert_count");
    auto capacity = context.Input<phi::DenseTensor>("capacity");
    auto out = context.Output<phi::DenseTensor>("Out");
    auto n_worker = context.Attr<int>("n_worker");
    auto n_expert = expert_count->numel() / n_worker;

    const auto& dev_ctx = context.template device_context<phi::CustomContext>();

    dev_ctx.template Alloc<T>(out);
    std::vector<T> out_data(out->numel());
    phi::DenseTensor expert_count_cpu, capacity_cpu;
    framework::TensorCopySync(
        *expert_count, phi::CPUPlace(), &expert_count_cpu);
    framework::TensorCopySync(*capacity, phi::CPUPlace(), &capacity_cpu);

    auto* ec_data = expert_count_cpu.data<T>();
    auto* capacity_data = capacity_cpu.data<T>();
    int eid, wid;
    for (int64_t i = 0; i < expert_count->numel(); ++i) {
      wid = i / n_expert;
      eid = i % n_expert;
      auto proposal = ec_data[i];
      auto cap_left = capacity_data[eid];
      capacity_data[eid] -= proposal;
      if (cap_left >= proposal) {
        out_data[wid * n_expert + eid] = proposal;
      } else if (cap_left >= 0) {
        out_data[wid * n_expert + eid] = cap_left;
      } else {
        out_data[wid * n_expert + eid] = 0;
      }
    }

    auto out_dims = out->dims();
    framework::TensorFromVector<T>(out_data, dev_ctx, out);
    out->Resize(out_dims);
  }
};

template <typename T>
class PruneGateByCapacityCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* gate_idx = context.Input<phi::DenseTensor>("GateIdx");
    auto* expert_count = context.Input<phi::DenseTensor>("ExpertCount");
    auto* new_gate_idx = context.Output<phi::DenseTensor>("NewGateIdx");
    const auto& dev_ctx = context.template device_context<phi::CustomContext>();
    dev_ctx.template Alloc<T>(new_gate_idx);

    phi::DenseTensor expert_count_cpu, gate_idx_cpu;
    framework::TensorCopySync(
        *expert_count, phi::CPUPlace(), &expert_count_cpu);
    framework::TensorCopySync(*gate_idx, phi::CPUPlace(), &gate_idx_cpu);
    auto expert_count_data = expert_count_cpu.data<T>();
    auto gate_idx_data = gate_idx_cpu.data<T>();
    std::vector<T> new_gate_idx_data(gate_idx->numel());
    for (auto i = 0; i < gate_idx->numel(); ++i) {
      auto orig_cap = expert_count_data[gate_idx_data[i]]--;
      if (orig_cap <= 0) {
        new_gate_idx_data[i] = -1;
      } else {
        new_gate_idx_data[i] = gate_idx_data[i];
      }
    }
  }
};

template <typename T>
class RandomRoutingOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto topk_idx = context.Input<phi::DenseTensor>("TopK_Idx");
    auto topk_value = context.Input<phi::DenseTensor>("TopK_Value");
    auto prob = context.Input<phi::DenseTensor>("Prob");
    auto out = context.Output<phi::DenseTensor>("Out");

    const auto& dev_ctx = context.template device_context<phi::CustomContext>();
    size_t D = topk_idx->dims()[1];

    phi::DenseTensor topk_value_cpu, prob_cpu;
    framework::TensorCopySync(*topk_value, phi::CPUPlace(), &topk_value_cpu);
    framework::TensorCopySync(*prob, phi::CPUPlace(), &prob_cpu);
    auto* topk_value_data = topk_value_cpu.data<T>();
    auto* prob_data = prob_cpu.data<T>();
    std::vector<int64_t> out_data(topk_idx->numel());

    for (int64_t idx = 0; idx < topk_idx->numel(); ++idx) {
      size_t row = idx / D;
      size_t col = idx % D;
      if (col == 1 &&
          static_cast<T>(2) * topk_value_data[idx] < prob_data[row]) {
        out_data[idx] = static_cast<int64_t>(-1);
      }
    }
    auto out_dims = out->dims();
    framework::TensorFromVector<int64_t>(out_data, dev_ctx, out);
    out->Resize(out_dims);
  }
};

template <typename T>
class AssignPosCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // assign pos decides which tokens should be fetched belong to specially
    // counter orderly.
    auto cum_count = context.Input<phi::DenseTensor>(
        "cum_count");  // (counter number) int32 | int64
    auto numbers = context.Input<phi::DenseTensor>(
        "X");  // (batch_size * seq_len, topk) int32
    auto eff_num_len =
        context.Input<phi::DenseTensor>("eff_num_len");  // (sum(cum_count))
    auto out =
        context.Output<phi::DenseTensor>("Out");  // (cum_count) value ranges
                                                  // from 0 to batch_size *
                                                  // seq_len * topk
    const auto& dev_ctx = context.template device_context<phi::CustomContext>();

    phi::DenseTensor cpu_eff_num_len;
    int64_t cpu_eff_num_len_data = 0;
    if (eff_num_len->place().GetType() == phi::AllocationType::CPU) {
      cpu_eff_num_len_data = eff_num_len->data<T>()[0];
    } else {
      framework::TensorCopySync(
          *eff_num_len, phi::CPUPlace(), &cpu_eff_num_len);
      cpu_eff_num_len_data = cpu_eff_num_len.data<T>()[0];
    }

    out->Resize({cpu_eff_num_len_data});
    dev_ctx.template Alloc<T>(out);

    phi::DenseTensor numbers_cpu, cum_count_cpu;
    framework::TensorCopySync(*numbers, phi::CPUPlace(), &numbers_cpu);
    framework::TensorCopySync(*cum_count, phi::CPUPlace(), &cum_count_cpu);
    auto* numbers_data = numbers_cpu.data<T>();
    auto* cum_count_data = cum_count_cpu.data<T>();

    std::vector<T> out_data(cpu_eff_num_len_data);
    for (int64_t i = 0; i < numbers->numel(); ++i) {
      int number_idx = numbers_data[i];
      if (number_idx > -1) {
        cum_count_data[number_idx] -= 1;
        int p = cum_count_data[number_idx];
        out_data[p] = i;
      }
    }
    framework::TensorFromVector<int64_t>(out_data, dev_ctx, out);
  }
};

template <typename T>
class GlobalScatterOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto local_count = ctx.Input<phi::DenseTensor>("local_count");
    auto global_count = ctx.Input<phi::DenseTensor>("global_count");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    const int rid = ctx.Attr<int>("ring_id");
    const auto& dev_ctx = ctx.template device_context<phi::CustomContext>();
    auto place = ctx.GetPlace();

    PADDLE_ENFORCE_EQ(local_count->dtype(),
                      phi::DataType::INT64,
                      common::errors::InvalidArgument(
                          "Please use int64 type in local_count."));
    PADDLE_ENFORCE_EQ(global_count->dtype(),
                      phi::DataType::INT64,
                      common::errors::InvalidArgument(
                          "Please use int64 type in global_count."));

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    phi::DenseTensor cpu_local_count;
    if (local_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_local_count_data = local_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *local_count, phi::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
    }
    auto global_count_len = 0;
    phi::DenseTensor cpu_global_count;
    if (global_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_global_count_data = global_count->data<int64_t>();
      global_count_len = global_count->numel();
    } else {
      framework::TensorCopySync(
          *global_count, phi::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
      global_count_len = cpu_global_count.numel();
    }

    if (map->has(rid)) {
      distributed::ProcessGroup* pg = map->get(rid);
      auto stream =
          reinterpret_cast<phi::CustomContext*>(pg->GetDeviceContext(place))
              ->GetStream();
      int nranks = pg->GetSize();
      int rank = pg->GetRank();
      auto in_feat = x->dims()[1];
      auto n_expert = local_count->dims()[0] / nranks;
      int64_t fwd_count = 0;

      for (auto i = 0; i < global_count_len; ++i) {
        fwd_count += cpu_global_count_data[i];
      }
      phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
      int64_t* expert_ptr = new int64_t[n_expert * nranks];
      expert_ptr[0] = 0;
      auto tot_experts = n_expert * nranks;
      for (auto i = 1; i < tot_experts; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
      }

      auto recv_ptr = 0;
      out->Resize(out_dims);
      dev_ctx.template Alloc<T>(out);

      for (auto i = 0; i < n_expert; ++i) {
        for (auto j = 0; j < rank; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            pg->Recv(out,
                     j,
                     recv_ptr * in_feat,
                     cpu_global_count_data[idx] * in_feat,
                     /*sync_op*/ true);
            recv_ptr += cpu_global_count_data[idx];
          }
        }
        for (auto j = 0; j < nranks; ++j) {
          if (j != rank) {
            int idx = i + j * n_expert;
            if (cpu_local_count_data[idx]) {
              phi::DenseTensor tmp = *x;
              pg->Send(tmp,
                       j,
                       expert_ptr[idx] * in_feat,
                       cpu_local_count_data[idx] * in_feat,
                       /*sync_op*/ true);
            }
          }
        }
        if (cpu_local_count_data[i + rank * n_expert]) {
          phi::DeviceManager::GetDeviceWithPlace(place)->MemoryCopyD2D(
              reinterpret_cast<void*>(out->data<T>() + recv_ptr * in_feat),
              reinterpret_cast<const void*>(x->data<T>() +
                                            expert_ptr[rank] * in_feat),
              (cpu_local_count_data[rank] * in_feat) * phi::SizeOf(x->dtype()),
              stream.get());
          recv_ptr += cpu_global_count_data[rank];
        }
        for (auto j = rank + 1; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            pg->Recv(out,
                     j,
                     recv_ptr * in_feat,
                     cpu_global_count_data[idx] * in_feat,
                     /*sync_op*/ true);
            recv_ptr += cpu_global_count_data[idx];
          }
        }
      }
    } else {
      auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
          phi::distributed::CommContextManager::GetInstance().Get(
              std::to_string(rid)));

      std::shared_ptr<phi::stream::Stream> stream;
      if (ctx.Attr<bool>("use_calc_stream")) {
        auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
        stream = static_cast<phi::CustomContext*>(dev_ctx)->GetStream();
      } else {
        stream = comm->GetStream();
      }

      int nranks = comm->GetSize();
      int rank = comm->GetRank();
      auto in_feat = x->dims()[1];
      auto n_expert = local_count->dims()[0] / nranks;
      int64_t fwd_count = 0;

      for (auto i = 0; i < global_count_len; ++i) {
        fwd_count += cpu_global_count_data[i];
      }
      phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
      int64_t* expert_ptr = new int64_t[n_expert * nranks];
      expert_ptr[0] = 0;
      auto tot_experts = n_expert * nranks;
      for (auto i = 1; i < tot_experts; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
      }

      auto recv_ptr = 0;
      auto send_buf = x->data<T>();
      out->Resize(out_dims);
      auto recv_buf = dev_ctx.template Alloc<T>(out);

      for (auto i = 0; i < n_expert; ++i) {
        for (auto j = 0; j < rank; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            phi::DeviceManager::CCLRecv(
                place.GetDeviceType(),
                reinterpret_cast<void*>(recv_buf + recv_ptr * in_feat),
                cpu_global_count_data[idx] * in_feat,
                x->dtype(),
                j,
                comm->GetXcclComm(),
                *stream);
            recv_ptr += cpu_global_count_data[idx];
          }
        }
        for (auto j = 0; j < nranks; ++j) {
          if (j != rank) {
            int idx = i + j * n_expert;
            if (cpu_local_count_data[idx]) {
              phi::DeviceManager::CCLSend(
                  place.GetDeviceType(),
                  const_cast<void*>(reinterpret_cast<const void*>(
                      send_buf + expert_ptr[idx] * in_feat)),
                  cpu_local_count_data[idx] * in_feat,
                  x->dtype(),
                  j,
                  comm->GetXcclComm(),
                  *stream);
            }
          }
        }
        if (cpu_local_count_data[i + rank * n_expert]) {
          phi::DeviceManager::GetDeviceWithPlace(place)->MemoryCopyD2D(
              reinterpret_cast<void*>(recv_buf + recv_ptr * in_feat),
              reinterpret_cast<const void*>(send_buf +
                                            expert_ptr[rank] * in_feat),
              (cpu_local_count_data[rank] * in_feat) * phi::SizeOf(x->dtype()),
              stream.get());
          recv_ptr += cpu_global_count_data[rank];
        }
        for (auto j = rank + 1; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            phi::DeviceManager::CCLRecv(
                place.GetDeviceType(),
                reinterpret_cast<void*>(recv_buf + recv_ptr * in_feat),
                cpu_global_count_data[idx] * in_feat,
                x->dtype(),
                j,
                comm->GetXcclComm(),
                *stream);
            recv_ptr += cpu_global_count_data[idx];
          }
        }
      }
    }

    phi::DeviceManager::SynchronizeDevice(ctx.GetPlace());
  }
};

template <typename T>
class GlobalGatherOpCustomDeviceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto local_count = ctx.Input<phi::DenseTensor>("local_count");
    auto global_count = ctx.Input<phi::DenseTensor>("global_count");
    const int rid = ctx.Attr<int>("ring_id");
    const auto& dev_ctx = ctx.template device_context<phi::CustomContext>();
    auto place = ctx.GetPlace();
    auto out = ctx.Output<phi::DenseTensor>("Out");

    PADDLE_ENFORCE_EQ(local_count->dtype(),
                      phi::DataType::INT64,
                      common::errors::InvalidArgument(
                          "Please use int64 type in local_count."));
    PADDLE_ENFORCE_EQ(global_count->dtype(),
                      phi::DataType::INT64,
                      common::errors::InvalidArgument(
                          "Please use int64 type in global_count."));

    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;
    phi::DenseTensor cpu_local_count;
    if (local_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(
          *local_count, phi::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }
    phi::DenseTensor cpu_global_count;
    if (global_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *global_count, phi::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    auto map = distributed::ProcessGroupMapFromGid::getInstance();

    if (map->has(rid)) {
      distributed::ProcessGroup* pg = map->get(rid);
      auto stream =
          reinterpret_cast<phi::CustomContext*>(pg->GetDeviceContext(place))
              ->GetStream();
      int nranks = pg->GetSize();
      int rank = pg->GetRank();
      auto in_feat = x->dims()[1];
      auto n_expert = local_count->dims()[0] / nranks;
      auto fwd_count = 0;
      for (auto i = 0; i < local_count_len; ++i) {
        fwd_count += cpu_local_count_data[i];
      }
      phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
      int64_t* expert_ptr = new int64_t[n_expert * nranks];
      expert_ptr[0] = 0;
      auto tot_experts = n_expert * nranks;
      for (auto i = 1; i < tot_experts; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
      }
      auto send_ptr = 0;
      out->Resize(out_dims);
      dev_ctx.template Alloc<T>(out);

      for (auto i = 0; i < n_expert; ++i) {
        for (auto j = 0; j < rank; ++j) {
          int idx = i + j * n_expert;
          if (cpu_local_count_data[idx]) {
            pg->Recv(out,
                     j,
                     expert_ptr[idx] * in_feat,
                     cpu_local_count_data[idx] * in_feat,
                     /*sync_op*/ true);
          }
        }
        for (auto j = 0; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            if (j != rank) {
              phi::DenseTensor tmp = *x;
              pg->Send(tmp,
                       j,
                       send_ptr * in_feat,
                       cpu_global_count_data[idx] * in_feat,
                       /*sync_op*/ true);
            } else {
              phi::DeviceManager::GetDeviceWithPlace(place)->MemoryCopyD2D(
                  reinterpret_cast<void*>(out->data<T>() +
                                          expert_ptr[idx] * in_feat),
                  reinterpret_cast<const void*>(x->data<T>() +
                                                send_ptr * in_feat),
                  (cpu_global_count_data[idx] * in_feat) *
                      phi::SizeOf(x->dtype()),
                  stream.get());
            }
            send_ptr += cpu_global_count_data[idx];
          }
        }
        for (auto j = rank + 1; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_local_count_data[idx]) {
            pg->Recv(out,
                     j,
                     expert_ptr[idx] * in_feat,
                     cpu_local_count_data[idx] * in_feat,
                     /*sync_op*/ true);
          }
        }
      }
    } else {
      auto comm = reinterpret_cast<phi::distributed::XCCLCommContext*>(
          phi::distributed::CommContextManager::GetInstance().Get(
              std::to_string(rid)));

      std::shared_ptr<phi::stream::Stream> stream;
      if (ctx.Attr<bool>("use_calc_stream")) {
        auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
        stream = static_cast<phi::CustomContext*>(dev_ctx)->GetStream();
      } else {
        stream = comm->GetStream();
      }
      int nranks = comm->GetSize();
      int rank = comm->GetRank();
      auto in_feat = x->dims()[1];
      auto n_expert = local_count->dims()[0] / nranks;

      auto fwd_count = 0;

      for (auto i = 0; i < local_count_len; ++i) {
        fwd_count += cpu_local_count_data[i];
      }
      phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
      int64_t* expert_ptr = new int64_t[n_expert * nranks];
      expert_ptr[0] = 0;
      auto tot_experts = n_expert * nranks;
      for (auto i = 1; i < tot_experts; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
      }
      auto send_ptr = 0;
      auto send_buf = x->data<T>();
      out->Resize(out_dims);
      auto recv_buf = dev_ctx.template Alloc<T>(out);

      for (auto i = 0; i < n_expert; ++i) {
        for (auto j = 0; j < rank + 1; ++j) {
          int idx = i + j * n_expert;
          if (cpu_local_count_data[idx]) {
            phi::DeviceManager::CCLRecv(place.GetDeviceType(),
                                        recv_buf + expert_ptr[idx] * in_feat,
                                        cpu_local_count_data[idx] * in_feat,
                                        x->dtype(),
                                        j,
                                        comm->GetXcclComm(),
                                        *stream);
          }
        }
        for (auto j = 0; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            if (j != rank) {
              phi::DeviceManager::CCLSend(
                  place.GetDeviceType(),
                  const_cast<void*>(reinterpret_cast<const void*>(
                      send_buf + send_ptr * in_feat)),
                  cpu_global_count_data[idx] * in_feat,
                  x->dtype(),
                  j,
                  comm->GetXcclComm(),
                  *stream);
            } else {
              phi::DeviceManager::GetDeviceWithPlace(place)->MemoryCopyD2D(
                  reinterpret_cast<void*>(recv_buf + expert_ptr[idx] * in_feat),
                  reinterpret_cast<const void*>(send_buf + send_ptr * in_feat),
                  (cpu_global_count_data[idx] * in_feat) *
                      phi::SizeOf(x->dtype()),
                  stream.get());
            }
            send_ptr += cpu_global_count_data[idx];
          }
        }
        for (auto j = rank + 1; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_local_count_data[idx]) {
            phi::DeviceManager::CCLRecv(place.GetDeviceType(),
                                        recv_buf + expert_ptr[idx] * in_feat,
                                        cpu_local_count_data[idx] * in_feat,
                                        x->dtype(),
                                        j,
                                        comm->GetXcclComm(),
                                        *stream);
          }
        }
      }
    }

    phi::DeviceManager::SynchronizeDevice(ctx.GetPlace());
  }
};

template <typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out);

void RegisterCustomDeviceCommonKernel(const std::string& dev_type) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE

#endif
}

}  // namespace operators
}  // namespace paddle

#undef REGISTER_OP_CUSTOM_DEVICE_KERNEL
#undef REGISTER_CUSTOM_DEVICE_GENERAL_KERNEL
