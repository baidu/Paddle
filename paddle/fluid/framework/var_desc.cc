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

#include "paddle/fluid/framework/var_desc.h"

#include "glog/logging.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework {

VarDesc::VarDesc(const VarDesc &other)
    : desc_(other.desc_),
      attrs_(other.attrs_),
      original_id_(other.original_id_) {
  if (other.dist_attr_) {
    dist_attr_ = std::make_unique<TensorDistAttr>(*other.dist_attr_);
  }
  need_updated_ = true;
}

VarDesc::~VarDesc() = default;

VarDesc::VarDesc(const proto::VarDesc &desc) : desc_(desc) {
  // Restore attrs_ for auto parallel
  for (const proto::VarDesc::Attr &attr : desc_.attrs()) {
    const std::string& attr_name = attr.name();
    attrs_[attr_name] = GetAttrValue(attr);
  }
  need_updated_ = true;
}

proto::VarType::Type VarDesc::GetType() const { return desc_.type().type(); }

void VarDesc::SetType(proto::VarType::Type type) {
  desc_.mutable_type()->set_type(type);
  need_updated_ = true;
}

void VarDesc::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
  need_updated_ = true;
}

void VarDesc::SetTensorDescNum(size_t num) {
  switch (desc_.type().type()) {
    case proto::VarType::READER: {
      auto *dense_tensors_ptr =
          desc_.mutable_type()->mutable_reader()->mutable_dense_tensor();
      dense_tensors_ptr->Clear();
      for (size_t i = 0; i < num; ++i) {
        dense_tensors_ptr->Add();
      }
      return;
    } break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Setting 'sub_tensor_number' is not "
                                      "supported by the %s type variable.",
                                      this->Name()));
  }
  need_updated_ = true;
}

size_t VarDesc::GetTensorDescNum() const {
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      return desc_.type().reader().dense_tensor_size();
      break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Getting 'sub_tensor_number' is not "
                                      "supported by the %s type variable.",
                                      this->Name()));
  }
}

void VarDesc::SetShapes(
    const std::vector<std::vector<int64_t>> &multiple_dims) {
  if (multiple_dims.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given shapes(" << multiple_dims.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_dims.size());
  }
  std::vector<proto::VarType::TensorDesc *> tensors = mutable_tensor_descs();
  for (size_t i = 0; i < multiple_dims.size(); ++i) {
    VectorToRepeated(multiple_dims[i], tensors[i]->mutable_dims());
  }
  need_updated_ = true;
}

std::vector<int64_t> VarDesc::GetShape() const {
  return RepeatedToVector(tensor_desc().dims());
}

std::vector<std::vector<int64_t>> VarDesc::GetShapes() const {
  std::vector<proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<std::vector<int64_t>> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(RepeatedToVector(tensor_desc.dims()));
  }
  return res;
}

void VarDesc::SetDataType(proto::VarType::Type data_type) {
  mutable_tensor_desc()->set_data_type(data_type);
  need_updated_ = true;
}

void VarDesc::SetDataTypes(
    const std::vector<proto::VarType::Type> &multiple_data_type) {
  if (multiple_data_type.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given data types("
            << multiple_data_type.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_data_type.size());
  }
  std::vector<proto::VarType::TensorDesc *> tensor_descs =
      mutable_tensor_descs();
  for (size_t i = 0; i < multiple_data_type.size(); ++i) {
    tensor_descs[i]->set_data_type(multiple_data_type[i]);
  }
  need_updated_ = true;
}

proto::VarType::Type VarDesc::GetDataType() const {
  return tensor_desc().data_type();
}

size_t VarDesc::ElementSize() const {
  return framework::SizeOfType(GetDataType());
}

std::vector<proto::VarType::Type> VarDesc::GetDataTypes() const {
  std::vector<proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<proto::VarType::Type> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(tensor_desc.data_type());
  }
  return res;
}

void VarDesc::SetLegacyLoDLevel(int32_t legacy_lod_level) {
  switch (desc_.type().type()) {
    case proto::VarType::DENSE_TENSOR:
      desc_.mutable_type()->mutable_dense_tensor()->set_legacy_lod_level(
          legacy_lod_level);
      break;
    case proto::VarType::DENSE_TENSOR_ARRAY:
      desc_.mutable_type()->mutable_tensor_array()->set_legacy_lod_level(
          legacy_lod_level);
      break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Setting 'legacy_lod_level' is not "
                                      "supported by the %s type variable.",
                                      this->Name()));
  }
  need_updated_ = true;
}

void VarDesc::SetLegacyLoDLevels(
    const std::vector<int32_t> &multiple_legacy_lod_level) {
  if (multiple_legacy_lod_level.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given legacy_lod_levels("
            << multiple_legacy_lod_level.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_legacy_lod_level.size());
  }
  switch (desc_.type().type()) {
    case proto::VarType::READER: {
      size_t i = 0;
      for (auto &dense_tensor :
           *desc_.mutable_type()->mutable_reader()->mutable_dense_tensor()) {
        dense_tensor.set_legacy_lod_level(multiple_legacy_lod_level[i++]);
      }
    } break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Setting 'legacy_lod_levels' is not "
                                      "supported by the %s type variable",
                                      this->Name()));
  }
  need_updated_ = true;
}

int32_t VarDesc::GetLegacyLoDLevel() const {
  switch (desc_.type().type()) {
    case proto::VarType::DENSE_TENSOR:
      return desc_.type().dense_tensor().legacy_lod_level();
    case proto::VarType::DENSE_TENSOR_ARRAY:
      return desc_.type().tensor_array().legacy_lod_level();
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Getting 'legacy_lod_level' is not "
                                      "supported by the %s type variable.",
                                      this->Name()));
  }
}

std::vector<int32_t> VarDesc::GetLegacyLoDLevels() const {
  std::vector<int32_t> res;
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      res.reserve(desc_.type().reader().dense_tensor_size());
      for (auto &dense_tensor : desc_.type().reader().dense_tensor()) {
        res.push_back(dense_tensor.legacy_lod_level());
      }
      return res;
      break;
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Getting 'legacy_lod_levels' is not "
                                      "supported by the %s type variable.",
                                      this->Name()));
  }
}

void VarDesc::SetLoDLevel(int32_t lod_level) { SetLegacyLoDLevel(lod_level); }

void VarDesc::SetLoDLevels(const std::vector<int32_t> &multiple_lod_level) {
  SetLegacyLoDLevels(multiple_lod_level);
}

int32_t VarDesc::GetLoDLevel() const { return GetLegacyLoDLevel(); }

std::vector<int32_t> VarDesc::GetLoDLevels() const {
  return GetLegacyLoDLevels();
}

const proto::VarType::TensorDesc &VarDesc::tensor_desc() const {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(),
      true,
      common::errors::NotFound("The variable's type was not set."));
  PADDLE_ENFORCE_EQ(
      desc_.type().has_type(),
      true,
      common::errors::NotFound("The variable's type was not set."));
  switch (desc_.type().type()) {
    case proto::VarType::SELECTED_ROWS:
      return desc_.type().selected_rows();
    case proto::VarType::DENSE_TENSOR:
      return desc_.type().dense_tensor().tensor();
    case proto::VarType::DENSE_TENSOR_ARRAY:
      return desc_.type().tensor_array().tensor();
    case proto::VarType::STRINGS:
      return desc_.type().strings();
    case proto::VarType::VOCAB:
      return desc_.type().vocab();
    case proto::VarType::SPARSE_COO:
      return desc_.type().sparse_coo();
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Getting 'tensor_desc' is not supported by the %s type variable.",
          this->Name()));
  }
}

std::vector<proto::VarType::TensorDesc> VarDesc::tensor_descs() const {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(),
      true,
      common::errors::NotFound("The variable's type was not be set."));
  std::vector<proto::VarType::TensorDesc> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      for (const auto &dense_tensor : desc_.type().reader().dense_tensor()) {
        res.push_back(dense_tensor.tensor());
      }
      return res;
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Getting 'tensor_descs' is not supported by the %s type variable.",
          this->Name()));
  }
}

proto::VarType::TensorDesc *VarDesc::mutable_tensor_desc() {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(),
      true,
      common::errors::NotFound("The variable's type was not be set."));
  PADDLE_ENFORCE_EQ(
      desc_.type().has_type(),
      true,
      common::errors::NotFound("The variable's type was not be set."));
  switch (desc_.type().type()) {
    case proto::VarType::SELECTED_ROWS:
      return desc_.mutable_type()->mutable_selected_rows();
    case proto::VarType::DENSE_TENSOR:
      return desc_.mutable_type()->mutable_dense_tensor()->mutable_tensor();
    case proto::VarType::DENSE_TENSOR_ARRAY:
      return desc_.mutable_type()->mutable_tensor_array()->mutable_tensor();
    case proto::VarType::STRINGS:
      return desc_.mutable_type()->mutable_strings();
    case proto::VarType::VOCAB:
      return desc_.mutable_type()->mutable_vocab();
    case proto::VarType::SPARSE_COO:
      return desc_.mutable_type()->mutable_sparse_coo();
    default:
      PADDLE_THROW(
          common::errors::Unavailable("Getting 'mutable_tensor_desc' is not "
                                      "supported by the %s type variable.",
                                      this->Name()));
  }
  need_updated_ = true;
}

std::vector<proto::VarType::TensorDesc *> VarDesc::mutable_tensor_descs() {
  PADDLE_ENFORCE_EQ(
      desc_.has_type(),
      true,
      common::errors::NotFound("The variable's type was not be set."));
  PADDLE_ENFORCE_EQ(
      desc_.type().has_type(),
      true,
      common::errors::NotFound("The variable's type was not be set."));
  std::vector<proto::VarType::TensorDesc *> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      for (auto &dense_tensor :
           *desc_.mutable_type()->mutable_reader()->mutable_dense_tensor()) {
        res.push_back(dense_tensor.mutable_tensor());
      }
      return res;
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Getting 'tensor_descs' is not supported by the %s type variable.",
          this->Name()));
  }
  need_updated_ = true;
}

std::vector<std::string> VarDesc::AttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

void VarDesc::RemoveAttr(const std::string &name) { attrs_.erase(name); }

void VarDesc::SetAttr(const std::string &name, const Attribute &v) {
  // NOTICE(sandyhouse): pybind11 will take the empty list in python as
  // the std::vector<int> type in C++; so we have to change the attr's type
  // here if we meet this issue
  proto::AttrType attr_type = static_cast<proto::AttrType>(v.index() - 1);
  if (attr_type == proto::AttrType::INTS &&
      PADDLE_GET_CONST(std::vector<int>, v).empty()) {
    // Find current attr via attr name and set the correct attribute value
    this->attrs_[name] = std::vector<int>();
    return;
  }
  bool valid = attr_type == proto::AttrType::INT ||
               attr_type == proto::AttrType::STRING ||
               attr_type == proto::AttrType::INTS;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    common::errors::InvalidArgument(
                        "The value for attr (%s) must be "
                        "one of int, string, list of int for now.",
                        name));

  this->attrs_[name] = v;
  need_updated_ = true;
}

Attribute VarDesc::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE_NE(
      it,
      attrs_.end(),
      common::errors::NotFound("Attribute %s is not found.", name));
  return it->second;
}

struct SetVarAttrDescVisitor {
  explicit SetVarAttrDescVisitor(proto::VarDesc::Attr *attr) : attr_(attr) {}
  mutable proto::VarDesc::Attr *attr_;

  template <typename T>
  void operator()(T &&v) {
    using U = std::decay_t<decltype(v)>;
    if (std::is_same<U, int>::value || std::is_same<U, std::string>::value ||
        std::is_same<U, std::vector<int>>::value) {
      set_attr_value(v);
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "Unsupported calling method of SetAttrDescVisitor object."));
    }
  }

  // This template is used to pass the compilation
  template <typename U>
  void set_attr_value(U v);

  void set_attr_value(int v) { attr_->set_i(v); }

  void set_attr_value(const std::string &v) { attr_->set_s(v); }

  void set_attr_value(const std::vector<int> &v) {
    VectorToRepeated(v, attr_->mutable_ints());
  }
};

// Only need to flush the attrs for auto parallel for now
void VarDesc::Flush() {
  VLOG(4) << "Flush "
          << " " << Name() << " " << need_updated_;
  if (need_updated_) {
    this->desc_.mutable_attrs()->Clear();
    std::vector<std::pair<std::string, Attribute>> sorted_attrs{attrs_.begin(),
                                                                attrs_.end()};
    std::sort(
        sorted_attrs.begin(),
        sorted_attrs.end(),
        [](std::pair<std::string, Attribute> a,
           std::pair<std::string, Attribute> b) { return a.first < b.first; });
    for (auto &attr : sorted_attrs) {
      auto *attr_desc = desc_.add_attrs();
      attr_desc->set_name(attr.first);
      attr_desc->set_type(
          static_cast<proto::AttrType>(attr.second.index() - 1));
      SetVarAttrDescVisitor visitor(attr_desc);
      paddle::visit(visitor, attr.second);
    }
    need_updated_ = false;
  }
}

TensorDistAttr *VarDesc::MutableDistAttr() {
  // If dist_attr_ is nullptr, construct a new one and return.
  if (dist_attr_) {
    return dist_attr_.get();
  } else {
    auto shape = paddle::distributed::auto_parallel::get_tensor_shape(this);
    dist_attr_ = std::make_unique<TensorDistAttr>(shape);
    return dist_attr_.get();
  }
  need_updated_ = true;
}

void VarDesc::SetDistAttr(const TensorDistAttr &dist_attr) {
  // Make sure this dist attr be created
  MutableDistAttr();
  *dist_attr_ = dist_attr;
  need_updated_ = true;
}

bool operator==(const VarDesc &left, const VarDesc &right) {
  return left.Proto()->SerializeAsString() ==
         right.Proto()->SerializeAsString();
}

}  // namespace paddle::framework
