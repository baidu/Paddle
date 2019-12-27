// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/gradient_accumulator.h"
#include <algorithm>
#include <memory>
#include <utility>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

template <typename T>
class TensorAddFunctor : public boost::static_visitor<> {
 public:
  TensorAddFunctor(int64_t numel, const T* x, T* y)
      : numel_(numel), x_(x), y_(y) {}

  void operator()(const platform::CPUPlace& place) {
    platform::CPUDeviceContext* ctx = dynamic_cast<platform::CPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CPUDeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }

#ifdef PADDLE_WITH_CUDA
  void operator()(const platform::CUDAPlace& place) {
    platform::CUDADeviceContext* ctx =
        dynamic_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CUDADeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }
#else
  void operator()(const platform::CUDAPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }
#endif

  // there is NO blas in CUDAPinnedPlace
  void operator()(const platform::CUDAPinnedPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }

 private:
  int64_t numel_;
  const T* x_;
  T* y_;
};

void TensorAdd(const framework::Variable& src, framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_tensor = src.Get<framework::LoDTensor>();

  auto numel = src_tensor.numel();

  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(dst_tensor->numel() == numel, true,
                    "dst_numel %d vs. src_numel %d", dst_tensor->numel(),
                    numel);

  auto data_type = src_tensor.type();
  auto place = src_tensor.place();

#define PADDLE_TENSOR_ADD(cpp_type)                                  \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) { \
    TensorAddFunctor<cpp_type> func(                                 \
        numel, src_tensor.data<cpp_type>(),                          \
        dst_tensor->mutable_data<cpp_type>(place));                  \
    boost::apply_visitor(func, place);                               \
    return;                                                          \
  }

  PADDLE_TENSOR_ADD(float);
  PADDLE_TENSOR_ADD(double);

#undef PADDLE_TENSOR_ADD

  PADDLE_THROW("Not supported data type %s for AddTo",
               framework::DataTypeToString(data_type));
}

void SelectedRowsAddToTensor(const framework::Variable& src,
                             framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_selected_rows = src.Get<framework::SelectedRows>();
  auto place = dst_tensor->place();
  auto data_type = src_selected_rows.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

#define PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(dev_ctx_type, cpp_type)           \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {         \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);              \
    paddle::operators::math::SelectedRowsAddToTensor<dev_ctx_type, cpp_type> \
        functor;                                                             \
    functor(*(dynamic_cast<dev_ctx_type*>(dev_ctx)), src_selected_rows,      \
            dst_tensor);                                                     \
    return;                                                                  \
  }

#ifdef PADDLE_WITH_CUDA
  if (paddle::platform::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CUDADeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CUDADeviceContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CPUDeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CPUDeviceContext, double);
#ifdef PADDLE_WITH_CUDA
  }
#endif

#undef PADDLE_SELECTED_ROWS_ADD_TO_TENSOR

  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsAddToTensor",
      framework::DataTypeToString(data_type)));
}

// Note(chenweihang): when two selected rows need to be added,
//   adding one to another is not equal to merging two selected rows
//   to one then add it to a empty selected rows, the after is correct.
//   Parameter has_src2 indicates whether src2 needs to participate in the
//   compute.
std::shared_ptr<VarBase> SelectedRowsMerge(const framework::Variable& src1,
                                           const framework::Variable& src2,
                                           bool has_src2 = true) {
  auto& src_selected_rows1 = src1.Get<framework::SelectedRows>();
  auto place = src_selected_rows1.value().place();
  auto data_type = src_selected_rows1.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  std::vector<const framework::SelectedRows*> src_selected_rows;
  src_selected_rows.emplace_back(&src_selected_rows1);
  if (has_src2) {
    auto& src_selected_rows2 = src2.Get<framework::SelectedRows>();
    src_selected_rows.emplace_back(&src_selected_rows2);
  }
  auto dst_var = std::make_shared<VarBase>(false, "Temp");
  auto* dst_selected_rows =
      dst_var->MutableVar()->GetMutable<framework::SelectedRows>();

#define PADDLE_SELECTED_ROWS_ADD(dev_ctx_type, cpp_type)                  \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {      \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);           \
    paddle::operators::math::scatter::MergeAdd<dev_ctx_type, cpp_type>    \
        merge_add;                                                        \
    merge_add(*(dynamic_cast<dev_ctx_type*>(dev_ctx)), src_selected_rows, \
              dst_selected_rows);                                         \
    return dst_var;                                                       \
  }

#ifdef PADDLE_WITH_CUDA
  if (paddle::platform::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD(platform::CUDADeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD(platform::CUDADeviceContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD(platform::CPUDeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD(platform::CPUDeviceContext, double);
#ifdef PADDLE_WITH_CUDA
  }
#endif

#undef PADDLE_SELECTED_ROWS_ADD

  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsMerge",
      framework::DataTypeToString(data_type)));
}

void VarBaseAdd(std::shared_ptr<VarBase> var, VarBase* var_) {
  auto& src = var->Var();
  auto* dst = var_->MutableVar();
  if (dst->IsType<framework::LoDTensor>()) {
    if (src.IsType<framework::LoDTensor>()) {
      TensorAdd(src, dst);
    } else if (src.IsType<framework::SelectedRows>()) {
      SelectedRowsAddToTensor(src, dst);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unexpected branch, output variable type is %s",
          framework::ToTypeName(dst->Type())));
    }
  } else {
    if (src.IsType<framework::LoDTensor>()) {
      auto* src_mutable = var->MutableVar();
      SelectedRowsAddToTensor(*dst, src_mutable);
      *dst = std::move(*(var->MutableVar()));
      var_->SetType(framework::proto::VarType::LOD_TENSOR);
    } else if (src.IsType<framework::SelectedRows>()) {
      std::shared_ptr<VarBase> temp = SelectedRowsMerge(src, *dst);
      *dst = std::move(*(temp->MutableVar()));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unexpected branch, output variable type is %s",
          framework::ToTypeName(dst->Type())));
    }
  }
}

platform::Place GetPlaceOfVarBase(const std::shared_ptr<VarBase>& var) {
  platform::Place place;
  if (var->Var().IsType<framework::LoDTensor>()) {
    place = var->Var().Get<framework::LoDTensor>().place();
  } else if (var->Var().IsType<framework::SelectedRows>()) {
    place = var->Var().Get<framework::SelectedRows>().place();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "only support LoDTensor and SelectedRows in dygraph"));
  }
  return place;
}

void GradientAccumulate(std::shared_ptr<VarBase> src_varbase,
                        VarBase* dst_varbase) {
  auto* dst_var = dst_varbase->MutableVar();
  auto* src_var = src_varbase->MutableVar();

  if (src_var->IsType<framework::SelectedRows>()) {
    if ((!dst_var->IsInitialized()) ||
        (dst_var->IsInitialized() &&
         dst_var->IsType<framework::SelectedRows>() &&
         (!dst_var->Get<framework::SelectedRows>().value().IsInitialized()))) {
      dst_varbase->SetType(framework::proto::VarType::SELECTED_ROWS);
      std::shared_ptr<VarBase> temp =
          SelectedRowsMerge(*src_var, *dst_var, false);
      *dst_var = std::move(*(temp->MutableVar()));
    } else if (dst_var->IsInitialized() &&
               dst_var->IsType<framework::LoDTensor>() &&
               (!dst_var->Get<framework::LoDTensor>().IsInitialized())) {
      *dst_var = std::move(*(src_varbase->MutableVar()));
    } else {
      if (dst_varbase->Trainable()) {
        VarBaseAdd(src_varbase, dst_varbase);
      } else {
        dst_varbase->SetType(framework::proto::VarType::SELECTED_ROWS);
        std::shared_ptr<VarBase> temp =
            SelectedRowsMerge(*src_var, *dst_var, false);
        *dst_var = std::move(*(temp->MutableVar()));
      }
    }
  } else {
    if ((!dst_var->IsInitialized()) ||
        (dst_var->IsInitialized() && dst_var->IsType<framework::LoDTensor>() &&
         (!dst_var->Get<framework::LoDTensor>().IsInitialized())) ||
        (dst_var->IsInitialized() &&
         dst_var->IsType<framework::SelectedRows>() &&
         (!dst_var->Get<framework::SelectedRows>().value().IsInitialized()))) {
      *dst_var = std::move(*(src_varbase->MutableVar()));
    } else {
      if (dst_varbase->Trainable()) {
        VarBaseAdd(src_varbase, dst_varbase);
      } else {
        *dst_var = std::move(*(src_varbase->MutableVar()));
      }
    }
  }
}

void EagerGradientAccumulator::Add(std::shared_ptr<VarBase> var,
                                   size_t trace_id) {
  platform::Place place = GetPlaceOfVarBase(var);
  if (!var_->OverridedStopGradient()) {
    VLOG(3) << "Sum Gradient for: " << var_->Name();
    if (cur_cnt_ == 0) {
      GradientAccumulate(var, var_);
    } else {
      VarBaseAdd(var, var_);
    }
  } else {
    if (!var_->Var().IsInitialized() ||
        !var_->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var_->Name() << " as zero ";

      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      if (!var_->Var().IsInitialized()) {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        VLOG(6) << "Dims of " << var_->Name() << " is set as: "
                << var->Var().Get<framework::LoDTensor>().dims();
        tensor->Resize(var->Var().Get<framework::LoDTensor>().dims());
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      } else {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
  }
  ++cur_cnt_;
}

void SortedGradientAccumulator::Add(std::shared_ptr<VarBase> var,
                                    size_t trace_id) {
  platform::Place place = GetPlaceOfVarBase(var);
  if (!var_->OverridedStopGradient()) {
    if (ref_cnt_ == 1) {
      GradientAccumulate(var, var_);
    } else {
      if (tmp_grad_vars_.empty()) {
        tmp_grad_vars_.reserve(ref_cnt_);
      }

      tmp_grad_vars_.emplace_back(std::move(var), trace_id);

      if (tmp_grad_vars_.size() != ref_cnt_) {
        return;
      }

      std::sort(tmp_grad_vars_.begin(), tmp_grad_vars_.end(),
                [](const std::pair<std::shared_ptr<VarBase>, size_t>& p1,
                   const std::pair<std::shared_ptr<VarBase>, size_t>& p2) {
                  return p1.second > p2.second;
                });

#ifdef PADDLE_WITH_CUDA
      auto* dst_var = var_->MutableVar();
      if (paddle::platform::is_gpu_place(place)) {
        bool dst_varbase_is_initialized = false;
        if (dst_var->IsInitialized()) {
          if (dst_var->IsType<framework::SelectedRows>()) {
            if (dst_var->Get<framework::SelectedRows>()
                    .value()
                    .IsInitialized()) {
              dst_varbase_is_initialized = true;
            }
          } else {
            if (dst_var->Get<framework::LoDTensor>().IsInitialized()) {
              dst_varbase_is_initialized = true;
            }
          }
        }
        // accumulate selected rows firstly
        for (size_t i = 0; i < tmp_grad_vars_.size(); ++i) {
          if (tmp_grad_vars_[i]
                  .first->Var()
                  .IsType<framework::SelectedRows>()) {
            if (!dst_varbase_is_initialized) {
              dst_varbase_is_initialized = true;
              var_->SetType(framework::proto::VarType::SELECTED_ROWS);
              std::shared_ptr<VarBase> temp = SelectedRowsMerge(
                  tmp_grad_vars_[i].first->Var(), *dst_var, false);
              *dst_var = std::move(*(temp->MutableVar()));
            } else {
              VarBaseAdd(tmp_grad_vars_[i].first, var_);
            }
          }
        }
        // accumulate lod tensor
        for (size_t i = 0; i < tmp_grad_vars_.size(); ++i) {
          if (!dst_varbase_is_initialized) {
            dst_varbase_is_initialized = true;
            *dst_var = std::move(*(tmp_grad_vars_[0].first->MutableVar()));
          }
          if (tmp_grad_vars_[i].first->Var().IsType<framework::LoDTensor>()) {
            VarBaseAdd(tmp_grad_vars_[i].first, var_);
          }
        }
      } else {
#endif
        GradientAccumulate(tmp_grad_vars_[0].first, var_);
        for (size_t i = 1; i < tmp_grad_vars_.size(); ++i) {
          VarBaseAdd(tmp_grad_vars_[i].first, var_);
        }
#ifdef PADDLE_WITH_CUDA
      }
#endif
      tmp_grad_vars_.clear();
    }
  } else {
    if (!var_->Var().IsInitialized() ||
        !var_->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var->Name() << " as zero";
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      if (!var_->Var().IsInitialized()) {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        VLOG(6) << "Dims of " << var_->Name() << " is set as: "
                << var->Var().Get<framework::LoDTensor>().dims();
        tensor->Resize(var->Var().Get<framework::LoDTensor>().dims());
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      } else {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
    // looks like tmp_grad_vars will not have any member but just in case
    tmp_grad_vars_.clear();
  }
}

}  // namespace imperative
}  // namespace paddle
