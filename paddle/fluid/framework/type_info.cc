/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_selected_rows.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_sparse_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/phi/core/raw_tensor.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {

template <typename BaseT, typename DerivedT>
TypeInfoTraits<BaseT, DerivedT>::TypeInfoTraits() {
  static_cast<BaseT*>(static_cast<DerivedT*>(this))->type_info_ = kType;
}

template <typename BaseT, typename DerivedT>
const TypeInfo<BaseT> TypeInfoTraits<BaseT, DerivedT>::kType =
    RegisterStaticType<BaseT>(DerivedT::name());

template <typename BaseT, typename DerivedT>
bool TypeInfoTraits<BaseT, DerivedT>::classof(const BaseT* obj) {
  return obj->type_info() == kType;
}

template class TypeInfoTraits<phi::TensorBase, egr::VariableCompatTensor>;
template class TypeInfoTraits<phi::TensorBase, paddle::prim::DescTensor>;
template class TypeInfoTraits<phi::TensorBase, paddle::primitive::LazyTensor>;
template class TypeInfoTraits<phi::TensorBase,
                              paddle::framework::VariableRefArray>;
template class TypeInfoTraits<phi::TensorBase, paddle::dialect::IrTensor>;
template class TypeInfoTraits<phi::TensorBase, paddle::dialect::IrSelectedRows>;
template class TypeInfoTraits<phi::TensorBase,
                              paddle::dialect::IrSparseCooTensor>;
template class TypeInfoTraits<phi::TensorBase,
                              paddle::dialect::IrSparseCsrTensor>;
}  // namespace phi
