// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/autograd_meta.h"

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "test/cpp/eager/data_structure_tests/grad_node_test.h"

TEST(AutogradMeta, Constructor) {
  paddle::Tensor et1;
  auto auto_grad = std::make_shared<egr::AutogradMeta>();
  et1.set_autograd_meta(auto_grad);
  auto* tmp_auto = static_cast<egr::AutogradMeta*>(et1.get_autograd_meta());
  PADDLE_ENFORCE_EQ(
      tmp_auto->OutRankInfo().first,
      size_t(0),
      common::errors::InvalidArgument(
          "The first element of OutRankInfo should be 0, but received %d.",
          tmp_auto->OutRankInfo().first));
  PADDLE_ENFORCE_EQ(
      tmp_auto->OutRankInfo().second,
      size_t(0),
      common::errors::InvalidArgument(
          "The second element of OutRankInfo should be 0, but received %d.",
          tmp_auto->OutRankInfo().second));
  CHECK(tmp_auto->IsInitialized() == false);
}

TEST(AutogradMeta, MemberFunction) {
  paddle::Tensor et1;
  auto auto_grad = std::make_shared<egr::AutogradMeta>();
  et1.set_autograd_meta(auto_grad);
  auto* tmp_auto = static_cast<egr::AutogradMeta*>(et1.get_autograd_meta());
  VLOG(6) << "Test Grad";
  PADDLE_ENFORCE_EQ(tmp_auto->Grad().defined(),
                    false,
                    common::errors::Fatal("grad should not be defined now"));
  auto* grad_t = tmp_auto->MutableGrad();
  phi::DenseTensorMeta meta =
      phi::DenseTensorMeta(phi::DataType::FLOAT32, common::make_ddim({1, 2}));
  std::shared_ptr<phi::DenseTensor> dt = std::make_shared<phi::DenseTensor>(
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace())
          .get(),
      meta);
  auto* dt_ptr = dt->mutable_data<float>(phi::CPUPlace());
  dt_ptr[0] = 5.0f;
  dt_ptr[1] = 10.0f;
  grad_t->set_impl(dt);
  VLOG(6) << "Test Mutable Grad";
  auto impl_ptr =
      std::dynamic_pointer_cast<phi::DenseTensor>(tmp_auto->Grad().impl());
  PADDLE_ENFORCE_EQ(
      impl_ptr->data<float>()[0],
      5.0f,
      common::errors::InvalidArgument(
          "The first element of grad tensor should be 5.0, but received %f.",
          impl_ptr->data<float>()[0]));
  PADDLE_ENFORCE_EQ(
      impl_ptr->data<float>()[1],
      10.0f,
      common::errors::InvalidArgument(
          "The second element of grad tensor should be 10.0, but received %f.",
          impl_ptr->data<float>()[1]));
  VLOG(6) << "Test IsInitialized";
  PADDLE_ENFORCE_EQ(
      tmp_auto->IsInitialized(),
      false,
      common::errors::Fatal(
          "egr::AutogradMeta variable tmp_auto should not be initialized now"));
  VLOG(6) << "Test GradNodeSetter Getter";
  auto grad_node = std::make_shared<eager_test::GradTestNode>();
  tmp_auto->SetGradNode(grad_node);
  PADDLE_ENFORCE_EQ(
      tmp_auto->IsInitialized(),
      true,
      common::errors::Fatal(
          "egr::AutogradMeta variable tmp_auto should be initialized now"));
  auto tmp_grad_node = tmp_auto->GetMutableGradNode();
  std::dynamic_pointer_cast<eager_test::GradTestNode>(tmp_grad_node)->val_ =
      5.0;
  PADDLE_ENFORCE_EQ(
      dynamic_cast<eager_test::GradTestNode*>(tmp_auto->GradNode())->val_,
      5.0,
      common::errors::InvalidArgument(
          "The value of GradTestNode should be 5.0, but received %f.",
          dynamic_cast<eager_test::GradTestNode*>(tmp_auto->GradNode())->val_));
  VLOG(6) << "Test rank Setter Getter";
  PADDLE_ENFORCE_EQ(
      tmp_auto->OutRankInfo().first,
      size_t(0),
      common::errors::InvalidArgument(
          "The first element of OutRankInfo should be 0, but received %d.",
          tmp_auto->OutRankInfo().first));
  PADDLE_ENFORCE_EQ(
      tmp_auto->OutRankInfo().second,
      size_t(0),
      common::errors::InvalidArgument(
          "The second element of OutRankInfo should be 0, but received %d.",
          tmp_auto->OutRankInfo().second));
  tmp_auto->SetSingleOutRankWithSlot(2, 3);
  PADDLE_ENFORCE_EQ(
      tmp_auto->OutRankInfo().first,
      size_t(2),
      common::errors::InvalidArgument(
          "The first element of OutRankInfo should be 2, but received %d.",
          tmp_auto->OutRankInfo().first));
  PADDLE_ENFORCE_EQ(
      tmp_auto->OutRankInfo().second,
      size_t(3),
      common::errors::InvalidArgument(
          "The second element of OutRankInfo should be 3, but received %d.",
          tmp_auto->OutRankInfo().second));
  VLOG(6) << "Test stop gradient Setter Getter";
  PADDLE_ENFORCE_EQ(
      tmp_auto->NumericStopGradient(),
      -1,
      common::errors::InvalidArgument(
          "The NumericStopGradient value should be -1, but received %d.",
          tmp_auto->NumericStopGradient()));
  tmp_auto->SetStopGradient(true);
  PADDLE_ENFORCE_EQ(
      tmp_auto->StopGradient(),
      true,
      common::errors::Fatal("tmp_auto->StopGradient() should be true now"));
  VLOG(6) << "Test Persistable Setter Getter";
  PADDLE_ENFORCE_EQ(
      tmp_auto->Persistable(),
      false,
      common::errors::Fatal("tmp_auto->Persistable() should be false now"));
  tmp_auto->SetPersistable(true);
  PADDLE_ENFORCE_EQ(
      tmp_auto->Persistable(),
      true,
      common::errors::Fatal(
          "tmp_auto->Persistable() should be true now after SetPersistable()"));
}
