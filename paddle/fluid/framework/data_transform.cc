/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_transform.h"

#include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/phi/api/lib/data_transform.h"

namespace paddle::framework {
class Variable;
}  // namespace paddle::framework

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif

namespace paddle::framework {

static void PassTensorData(phi::DenseTensor *from, phi::DenseTensor *to) {
  to->ShareDataWith(*from);
  *from = phi::DenseTensor();
}

void TransformData(const phi::KernelKey &expected_kernel_type,
                   const phi::KernelKey &kernel_type_for_var,
                   const phi::DenseTensor &input_tensor,
                   phi::DenseTensor *output_tensor,
                   const phi::Place &place) {
  bool transformed = false;
  phi::DenseTensor in;
  in.ShareDataWith(input_tensor);
  phi::DenseTensor out;
  const DataLayout lin = kernel_type_for_var.layout();
  const DataLayout lout = expected_kernel_type.layout();

  if (NeedTransform2Contiguous(in.meta().is_contiguous())) {
    out = paddle::experimental::Trans2Contiguous(in);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do layout transform
  if (NeedTransformLayout(lout, lin)) {
#ifdef PADDLE_WITH_DNNL
    if (lin == DataLayout::ONEDNN || lout == DataLayout::ONEDNN) {
      PADDLE_ENFORCE_EQ(
          !(lin == DataLayout::ONEDNN && lout == DataLayout::ONEDNN),
          true,
          common::errors::PreconditionNotMet(
              "No layout transform needed between two oneDNN OPKernels."));

      if (lin != DataLayout::ONEDNN && lout == DataLayout::ONEDNN) {
        // Case1 - transform from Non-ONEDNN OPKernel to ONEDNN OPKernel
        // Just set layout/format. No real transform occur
        out.ShareDataWith(input_tensor);
        // For NHWC data we need reshape of tensors as MKL-DNN
        // is expecting NHWC dims description order
        if (lin == DataLayout::kNHWC || lin == DataLayout::kNDHWC) {
          phi::funcs::MatchShapeToLayout(&out, lin, lout);
          // We register only NHWC assuming that model is consistent e.g. either
          // NHWC or NCHW
          phi::OneDNNContext::tls().set_cur_paddle_data_layout(lin);
        }

        dnnl::memory::desc out_mem_desc =
            phi::funcs::make_memory_desc(out, lin);
        out.set_mem_desc(out_mem_desc);
      } else {
        // Case2 - transform from ONEDNN OPKernel to Non-ONEDNN OPKernel
        // Do transform via ONEDNN lib
        PADDLE_ENFORCE(lin == DataLayout::ONEDNN && lout != DataLayout::ONEDNN,
                       common::errors::InvalidArgument(
                           "TransDataLayoutFromOneDNN only supports "
                           "transform from ONEDNN to non-ONEDNN"));

        phi::funcs::TransDataLayoutFromOneDNN(
            lin,
            phi::OneDNNContext::tls().get_cur_paddle_data_layout(),
            in,
            &out,
            place);
      }
    } else {
      // Case3 - transform between Non-ONEDNN OPKernels
      TransDataLayout(
          kernel_type_for_var, expected_kernel_type, in, &out, place);
    }
#else
    // Case3 - transform between Non-ONEDNN OPKernels
    TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out, place);
#endif
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do data type transform
  if (NeedTransformDataType(expected_kernel_type, kernel_type_for_var)) {
    TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do device transform
  if (kernel_type_for_var.backend() != phi::Backend::ALL_BACKEND &&
      !phi::is_same_place(in.place(), place)) {
    TransDataDevice(in, place, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  PADDLE_ENFORCE_EQ(
      transformed,
      true,
      common::errors::PreconditionNotMet(
          "No transform is applied for the data needs to be transformed."));
  // get output data
  output_tensor->ShareDataWith(in);
}

void SetTensorToVariable(const Variable &in_var,
                         const phi::DenseTensor &tensor,
                         Variable *out_var) {
  if (in_var.IsType<phi::DenseTensor>()) {
    auto &in_dense_tensor = in_var.Get<phi::DenseTensor>();
    auto *tran_dense_tensor = out_var->GetMutable<phi::DenseTensor>();
    tran_dense_tensor->set_lod(in_dense_tensor.lod());
    tran_dense_tensor->set_layout(in_dense_tensor.layout());
#ifdef PADDLE_WITH_DNNL
    tran_dense_tensor->set_mem_desc(in_dense_tensor.mem_desc());
#endif
    tran_dense_tensor->ShareDataWith(tensor);
  } else if (in_var.IsType<phi::SelectedRows>()) {
    auto &in_selected_rows = in_var.Get<phi::SelectedRows>();
    auto *trans_selected_rows = out_var->GetMutable<phi::SelectedRows>();
    trans_selected_rows->set_height(in_selected_rows.height());
    trans_selected_rows->set_rows(in_selected_rows.rows());
    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "Unsupported variable type, only supports phi::DenseTensor or "
        "SelectedRows, "
        "but the input variable type is %s.",
        ToTypeName(in_var.Type())));
  }
}

phi::GetKernelTypeForVarContext BuildGetKernelTypeForVarContext(
    const phi::KernelKey &kernel_key,
    const AttributeMap &fluid_attrs,
    phi::AttributeMap *phi_attrs,
    bool has_infer_varkernel_fn) {
  // According to "GetKernelTypeForVar" in some ops executed with oneDNN,
  // the only "string" member, such as "data_layout" 、"data_format" of
  // AttributeMap is useful. In the future the other args maybe used. Because
  // the "phi" module should not depend on the "fluid", transform
  // "framework::AttributeMap" to "phi::AttributeMap".
  if (has_infer_varkernel_fn) {
    for (auto &attr : fluid_attrs) {
      switch (attr.second.index()) {
        case 3:  // string type in framework::Attribute
          (*phi_attrs)[attr.first] = PADDLE_GET_CONST(std::string, attr.second);
          break;
        default:
          VLOG(6) << "GetKernelTypeForVarContext currently only use "
                     "std::string. You add other type if need.";
          break;
      }
    }
  }
  return phi::GetKernelTypeForVarContext(&kernel_key, phi_attrs);
}

}  // namespace paddle::framework
