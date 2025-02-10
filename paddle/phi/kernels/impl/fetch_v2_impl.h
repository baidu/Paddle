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

#pragma once

#include "glog/logging.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/framework/feed_fetch_type.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace phi {

template <typename Context>
static void DeepCopy(const Context &dev_ctx,
                     const phi::DenseTensor &src_item,
                     const std::string &fetch_var_name,
                     phi::DenseTensor *dst_item) {
  if (src_item.IsInitialized()) {
#ifdef PADDLE_WITH_DNNL
    // Conversion from MKL-DNN to Paddle
    if (src_item.layout() == phi::DataLayout::ONEDNN) {
      phi::DenseTensor out;
      // Convert to desired Paddle layout, apart from grads of filter
      // as params are not a subject to paddle's data_format
      phi::funcs::TransDataLayoutFromOneDNN(
          src_item.layout(),
          fetch_var_name == "Filter@GRAD"
              ? phi::DataLayout::kNCHW
              : phi::OneDNNContext::tls().get_cur_paddle_data_layout(),
          src_item,
          &out,
          phi::CPUPlace());
      phi::Copy(dev_ctx, out, phi::CPUPlace(), true, dst_item);
    } else {
      phi::Copy(dev_ctx, src_item, phi::CPUPlace(), true, dst_item);
    }
#else
    phi::Copy(dev_ctx, src_item, phi::CPUPlace(), true, dst_item);
#endif
  }
  dst_item->set_lod(src_item.lod());
}

template <typename T, typename Context>
void FetchV2Kernel(const Context &dev_ctx,
                   const DenseTensor &x,
                   int col,
                   bool deepcopy,
                   phi::FetchList *out) {
  PADDLE_ENFORCE_GE(col,
                    0,
                    errors::InvalidArgument(
                        "Expected the column index (the attribute 'col' of "
                        "operator 'Fetch') of current fetching variable to be "
                        "no less than 0. But received column index = %d.",
                        col));
  auto *fetch_list = out;
  if (static_cast<size_t>(col) >= fetch_list->size()) {
    fetch_list->resize(col + 1);
  }

  auto &src_item = x;
  if (!src_item.initialized()) {
    return;
  }
  bool check_place =
      src_item.place().GetType() == phi::AllocationType::CPU ||
      src_item.place().GetType() == phi::AllocationType::GPUPINNED ||
      src_item.place().GetType() == phi::AllocationType::CUSTOM;
  PADDLE_ENFORCE_EQ(check_place,
                    true,
                    errors::InvalidArgument("Tensor's place of input(X) must "
                                            "be CPUPlace or CUDAPinnedPlace."));
  phi::DenseTensor tmp = src_item;
  fetch_list->at(col) = tmp;
  auto *dst_item = &(PADDLE_GET(phi::DenseTensor, fetch_list->at(col)));
  if (deepcopy) {
    DeepCopy(dev_ctx, src_item, "", dst_item);
  } else {
    dst_item->ShareDataWith(src_item);
    dst_item->set_lod(src_item.lod());
  }
}

template <typename T, typename Context>
void FetchV2ArrayKernel(const Context &dev_ctx,
                        const TensorArray &x,
                        int col,
                        bool deepcopy,
                        phi::FetchList *out) {
  PADDLE_ENFORCE_GE(col,
                    0,
                    errors::InvalidArgument(
                        "Expected the column index (the attribute 'col' of "
                        "operator 'Fetch') of current fetching variable to be"
                        " no less than 0. But received column index = % d.",
                        col));
  auto *fetch_list = out;
  if (static_cast<size_t>(col) >= fetch_list->size()) {
    fetch_list->resize(col + 1);
  }
  auto &src_item = x;
  phi::TensorArray tmp(src_item.size());
  fetch_list->at(col) = tmp;
  auto &dst_item = PADDLE_GET(phi::TensorArray, fetch_list->at(col));
  for (size_t i = 0; i < src_item.size(); ++i) {
    PADDLE_ENFORCE_EQ(src_item[i].place().GetType() == phi::AllocationType::CPU,
                      true,
                      errors::InvalidArgument(
                          "Tensor's place of input(X) must be CPUPlace."));
    if (deepcopy) {
      DeepCopy(dev_ctx, src_item[i], "", &dst_item[i]);
    } else {
      dst_item[i].ShareDataWith(src_item[i]);
      dst_item[i].set_lod(src_item[i].lod());
    }
  }
}

}  // namespace phi
