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

#include "paddle/fluid/framework/op_desc.h"
#include <algorithm>
#include <functional>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include "glog/logging.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace framework {

class OpDesc;
class BlockDesc;
class CompileTimeInferShapeContext : public InferShapeContext {
 public:
  CompileTimeInferShapeContext(const OpDesc &op, const BlockDesc &block);

  bool HasInput(const std::string &name) const override;

  bool HasOutput(const std::string &name) const override;

  bool HasInputs(const std::string &name) const override;

  bool HasOutputs(const std::string &name) const override;

  AttrReader Attrs() const override;

  const std::vector<std::string> &Inputs(
      const std::string &name) const override;

  const std::vector<std::string> &Outputs(
      const std::string &name) const override;

  void ShareDim(const std::string &in, const std::string &out, size_t i = 0,
                size_t j = 0) override {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    const std::string &input_n = Inputs(in)[i];
    const std::string &output_n = Outputs(out)[j];

    PADDLE_ENFORCE(input_n != framework::kEmptyVarName, "The %s[%d] is @EMPTY@",
                   in, i);
    PADDLE_ENFORCE(output_n != framework::kEmptyVarName,
                   "The %s[%d] is @EMPTY@", out, j);

    auto *in_var = block_.FindVarRecursive(input_n);
    auto *out_var = block_.FindVarRecursive(output_n);

    PADDLE_ENFORCE(in_var->GetType() == out_var->GetType(),
                   "The type of %s and %s is not the same.", input_n, output_n);

    SetDim(output_n, GetDim(input_n));
  }

  void ShareLoD(const std::string &in, const std::string &out, size_t i = 0,
                size_t j = 0) const override {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    PADDLE_ENFORCE(Inputs(in)[i] != framework::kEmptyVarName,
                   "The %s[%d] is @EMPTY@", in, i);
    PADDLE_ENFORCE(Outputs(out)[j] != framework::kEmptyVarName,
                   "The %s[%d] is @EMPTY@", out, j);
    auto *in_var = block_.FindVarRecursive(Inputs(in)[i]);
    auto *out_var = block_.FindVarRecursive(Outputs(out)[j]);
    if (in_var->GetType() != proto::VarType::LOD_TENSOR &&
        in_var->GetType() != proto::VarType::LOD_TENSOR_ARRAY) {
      VLOG(3) << "input " << in << " is not LodTensor or LodTensorArray.";
      return;
    }
    out_var->SetLoDLevel(in_var->GetLoDLevel());
  }

  void DecreaseLoDLevel(const std::string &in, const std::string &out,
                        size_t i = 0, size_t j = 0) const override {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    PADDLE_ENFORCE(Inputs(in)[i] != framework::kEmptyVarName,
                   "The %s[%d] is @EMPTY@", in, i);
    PADDLE_ENFORCE(Outputs(out)[j] != framework::kEmptyVarName,
                   "The %s[%d] is @EMPTY@", out, j);
    auto *in_var = block_.FindVarRecursive(Inputs(in)[i]);
    auto *out_var = block_.FindVarRecursive(Outputs(out)[j]);
    PADDLE_ENFORCE(out_var->GetType() == proto::VarType::LOD_TENSOR_ARRAY ||
                       out_var->GetType() == proto::VarType::LOD_TENSOR,
                   "The input %s should be LodTensorArray or LodTensor.",
                   out_var->Name());
    PADDLE_ENFORCE(in_var->GetType() == proto::VarType::LOD_TENSOR,
                   "The input %s should be LodTensor.", in_var->Name());
    if (in_var->GetLoDLevel() > 0) {
      out_var->SetLoDLevel(in_var->GetLoDLevel() - 1);
    }
  }

  std::vector<InferShapeVarPtr> GetInputVarPtrs(
      const std::string &name) override {
    const std::vector<std::string> arg_names = Inputs(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(arg_names.size());
    std::transform(arg_names.begin(), arg_names.end(), std::back_inserter(res),
                   [this](const std::string &name) {
                     return block_.FindVarRecursive(name);
                   });
    return res;
  }

  std::vector<InferShapeVarPtr> GetOutputVarPtrs(
      const std::string &name) override {
    const std::vector<std::string> arg_names = Outputs(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(arg_names.size());
    std::transform(arg_names.begin(), arg_names.end(), std::back_inserter(res),
                   [this](const std::string &name) {
                     return block_.FindVarRecursive(name);
                   });
    return res;
  }

  DDim GetInputDim(const std::string &name) const override {
    const std::vector<std::string> &arg_names = Inputs(name);
    PADDLE_ENFORCE_EQ(arg_names.size(), 1UL,
                      "Input(%s) should hold one element, but now it holds %d",
                      name, arg_names.size());
    return this->GetDim(arg_names[0]);
  }

  std::vector<DDim> GetInputsDim(const std::string &name) const override {
    const std::vector<std::string> &arg_names = Inputs(name);
    return GetDims(arg_names);
  }

  bool IsRuntime() const override;

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string &name) const override {
    return GetVarTypes(Inputs(name));
  }

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string &name) const override {
    return GetVarTypes(Outputs(name));
  }

  void SetOutputDim(const std::string &name, const DDim &dim) override {
    auto &arg_names = Outputs(name);
    PADDLE_ENFORCE_EQ(arg_names.size(), 1UL,
                      "Output(%s) should hold one element, but now it holds %d",
                      name, arg_names.size());
    SetDim(arg_names[0], dim);
  }

  void SetOutputsDim(const std::string &name,
                     const std::vector<DDim> &dims) override {
    auto &names = Outputs(name);
    SetDims(names, dims);
  }

 protected:
  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<std::string> &names) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(names.size());
    std::transform(
        names.begin(), names.end(), retv.begin(),
        std::bind(std::mem_fn(&CompileTimeInferShapeContext::GetVarType), this,
                  std::placeholders::_1));
    return retv;
  }

  proto::VarType::Type GetVarType(const std::string &name) const;

  DDim GetDim(const std::string &name) const {
    auto var = block_.FindVarRecursive(name);
    PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s", name);
    DDim res;
    try {
      auto shape = var->GetShape();
      res = shape.empty() ? make_ddim({0UL}) : make_ddim(shape);
    } catch (...) {
      VLOG(5) << "GetDim of variable " << name << " error";
      std::rethrow_exception(std::current_exception());
    }
    return res;
  }

  std::vector<DDim> GetDims(const std::vector<std::string> &names) const {
    std::vector<DDim> ret;
    ret.reserve(names.size());
    std::transform(
        names.begin(), names.end(), std::back_inserter(ret),
        [this](const std::string &name) { return this->GetDim(name); });
    return ret;
  }

  void SetDim(const std::string &name, const DDim &dim);

  void SetDims(const std::vector<std::string> &names,
               const std::vector<DDim> &dims) {
    size_t length = names.size();
    PADDLE_ENFORCE_EQ(length, dims.size());
    for (size_t i = 0; i < length; ++i) {
      if (names[i] == framework::kEmptyVarName) {
        continue;
      }
      SetDim(names[i], dims[i]);
    }
  }

  std::vector<DDim> GetRepeatedDims(const std::string &name) const override;

  void SetRepeatedDims(const std::string &name,
                       const std::vector<DDim> &dims) override;

  const OpDesc &op_;
  const BlockDesc &block_;
};

OpDesc::OpDesc(const std::string &type, const VariableNameMap &inputs,
               const VariableNameMap &outputs, const AttributeMap &attrs) {
  desc_.set_type(type);
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
  need_update_ = true;
  block_ = nullptr;
}

OpDesc::OpDesc(const OpDesc &other, BlockDesc *block) {
  CopyFrom(other);
  block_ = block;
  need_update_ = true;
}

void OpDesc::CopyFrom(const OpDesc &op_desc) {
  desc_.set_type(op_desc.Type());
  inputs_ = op_desc.inputs_;
  outputs_ = op_desc.outputs_;
  attrs_ = op_desc.attrs_;
  need_update_ = true;
}

OpDesc::OpDesc(const proto::OpDesc &desc, BlockDesc *block)
    : desc_(desc), need_update_(false) {
  // restore inputs_
  int input_size = desc_.inputs_size();
  for (int i = 0; i < input_size; ++i) {
    const proto::OpDesc::Var &var = desc_.inputs(i);
    std::vector<std::string> &args = inputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore outputs_
  int output_size = desc_.outputs_size();
  for (int i = 0; i < output_size; ++i) {
    const proto::OpDesc::Var &var = desc_.outputs(i);
    std::vector<std::string> &args = outputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore attrs_
  for (const proto::OpDesc::Attr &attr : desc_.attrs()) {
    std::string attr_name = attr.name();
    // The sub_block referred to by the BLOCK attr hasn't been added
    // to ProgramDesc class yet, we skip setting BLOCK/BLOCKS attr here.
    if (attr.type() != proto::AttrType::BLOCK &&
        attr.type() != proto::AttrType::BLOCKS) {
      attrs_[attr_name] = GetAttrValue(attr);
    }
  }
  this->block_ = block;
}

proto::OpDesc *OpDesc::Proto() {
  Flush();
  return &desc_;
}

const std::vector<std::string> &OpDesc::Input(const std::string &name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Input %s cannot be found in Op %s", name,
                 Type());
  return it->second;
}

std::vector<std::string> OpDesc::InputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->inputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDesc::SetInput(const std::string &param_name,
                      const std::vector<std::string> &args) {
  need_update_ = true;
  inputs_[param_name] = args;
}

const std::vector<std::string> &OpDesc::Output(const std::string &name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(), "Output %s cannot be found in Op %s",
                 name, Type());
  return it->second;
}

std::vector<std::string> OpDesc::OutputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->outputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDesc::SetOutput(const std::string &param_name,
                       const std::vector<std::string> &args) {
  need_update_ = true;
  this->outputs_[param_name] = args;
}

bool OpDesc::HasProtoAttr(const std::string &name) const {
  auto &op_info = OpInfoMap::Instance();
  if (op_info.Has(desc_.type())) {
    auto op_info_ptr = op_info.Get(desc_.type());
    if (op_info_ptr.HasOpProtoAndChecker()) {
      const proto::OpProto &proto = op_info_ptr.Proto();
      for (int i = 0; i != proto.attrs_size(); ++i) {
        const proto::OpProto::Attr &attr = proto.attrs(i);
        if (attr.name() == name) {
          return true;
        }
      }
    }
  }
  return false;
}

proto::AttrType OpDesc::GetAttrType(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return static_cast<proto::AttrType>(it->second.which() - 1);
}

std::vector<std::string> OpDesc::AttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

void OpDesc::RemoveAttr(const std::string &name) {
  attrs_.erase(name);
  need_update_ = true;
}

void OpDesc::SetAttr(const std::string &name, const Attribute &v) {
  // NOTICE(minqiyang): pybind11 will take the empty list in python as
  // the std::vector<int> type in C++; so we have to change the attr's type
  // here if we meet this issue
  proto::AttrType attr_type = static_cast<proto::AttrType>(v.which() - 1);
  if (attr_type == proto::AttrType::INTS &&
      boost::get<std::vector<int>>(v).size() == 0u) {
    // Find current attr via attr name and set the correct attribute value
    const proto::OpProto::Attr &attr = GetProtoAttr(name);
    switch (attr.type()) {
      case proto::AttrType::BOOLEANS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to BOOLEANS";
        this->attrs_[name] = std::vector<bool>();
        break;
      }
      case proto::AttrType::INTS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to INTS";
        this->attrs_[name] = std::vector<int>();
        break;
      }
      case proto::AttrType::LONGS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from LONGS to LONGS";
        this->attrs_[name] = std::vector<int64_t>();
        break;
      }
      case proto::AttrType::FLOATS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to FLOATS";
        this->attrs_[name] = std::vector<float>();
        break;
      }
      case proto::AttrType::STRINGS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to STRINGS";
        this->attrs_[name] = std::vector<std::string>();
        break;
      }
      case proto::AttrType::BLOCKS: {
        VLOG(11) << "SetAttr: " << Type() << ", " << name
                 << " from INTS to BLOCKS";
        this->SetBlocksAttr(name, std::vector<BlockDesc *>());
        return;
      }
      default:
        PADDLE_THROW("Wrong attr type %d", attr.type());
    }
    need_update_ = true;
    return;
  }

  this->attrs_[name] = v;
  need_update_ = true;
}

void OpDesc::SetBlockAttr(const std::string &name, BlockDesc *block) {
  this->attrs_[name] = block;
  need_update_ = true;
}

void OpDesc::SetBlocksAttr(const std::string &name,
                           std::vector<BlockDesc *> blocks) {
  this->attrs_[name] = blocks;
  need_update_ = true;
}

void OpDesc::SetAttrMap(
    const std::unordered_map<std::string, Attribute> &attr_map) {
  attrs_ = attr_map;
  need_update_ = true;
}

Attribute OpDesc::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return it->second;
}

const proto::OpProto::Attr &OpDesc::GetProtoAttr(
    const std::string &name) const {
  const proto::OpProto &proto = OpInfoMap::Instance().Get(Type()).Proto();
  for (int i = 0; i != proto.attrs_size(); ++i) {
    const proto::OpProto::Attr &attr = proto.attrs(i);
    if (attr.name() == name) {
      return attr;
    }
  }

  PADDLE_THROW("Attribute %s is not found in proto %s", name, proto.type());
}

Attribute OpDesc::GetNullableAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  if (it != attrs_.end()) {
    return it->second;
  } else {
    return Attribute();
  }
}

std::vector<int> OpDesc::GetBlocksAttrIds(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  auto blocks = boost::get<std::vector<BlockDesc *>>(it->second);

  std::vector<int> ids;
  for (auto n : blocks) {
    ids.push_back(n->ID());
  }

  return ids;
}

int OpDesc::GetBlockAttrId(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return boost::get<BlockDesc *>(it->second)->ID();
}

const std::unordered_map<std::string, Attribute> &OpDesc::GetAttrMap() const {
  return attrs_;
}

void OpDesc::Rename(const std::string &old_name, const std::string &new_name) {
  RenameInput(old_name, new_name);
  RenameOutput(old_name, new_name);
  need_update_ = true;
}

void OpDesc::RenameOutput(const std::string &old_name,
                          const std::string &new_name) {
  for (auto &output : outputs_) {
    std::replace(output.second.begin(), output.second.end(), old_name,
                 new_name);
  }

  auto it = attrs_.find(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (it != attrs_.end()) {
    auto &op_vars = boost::get<std::vector<std::string>>(it->second);
    std::replace(op_vars.begin(), op_vars.end(), old_name, new_name);
  }

  need_update_ = true;
}

void OpDesc::RenameInput(const std::string &old_name,
                         const std::string &new_name) {
  for (auto &input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }

  auto it = attrs_.find(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName());
  if (it != attrs_.end()) {
    auto &op_vars = boost::get<std::vector<std::string>>(it->second);
    std::replace(op_vars.begin(), op_vars.end(), old_name, new_name);
  }

  need_update_ = true;
}

struct SetAttrDescVisitor : public boost::static_visitor<void> {
  explicit SetAttrDescVisitor(proto::OpDesc::Attr *attr) : attr_(attr) {}
  mutable proto::OpDesc::Attr *attr_;
  void operator()(int v) const { attr_->set_i(v); }
  void operator()(float v) const { attr_->set_f(v); }
  void operator()(const std::string &v) const { attr_->set_s(v); }

  // Please refer to https://github.com/PaddlePaddle/Paddle/issues/7162
  template <class T,
            class = typename std::enable_if<std::is_same<bool, T>::value>::type>
  void operator()(T b) const {
    attr_->set_b(b);
  }

  void operator()(const std::vector<int> &v) const {
    VectorToRepeated(v, attr_->mutable_ints());
  }
  void operator()(const std::vector<float> &v) const {
    VectorToRepeated(v, attr_->mutable_floats());
  }
  void operator()(const std::vector<std::string> &v) const {
    VectorToRepeated(v, attr_->mutable_strings());
  }
  void operator()(const std::vector<bool> &v) const {
    VectorToRepeated(v, attr_->mutable_bools());
  }
  void operator()(const std::vector<BlockDesc *> &v) const {
    std::vector<int> blocks_idx;
    for (auto blk : v) {
      blocks_idx.push_back(blk->ID());
    }
    VectorToRepeated(blocks_idx, attr_->mutable_blocks_idx());
  }

  void operator()(BlockDesc *desc) const { attr_->set_block_idx(desc->ID()); }

  void operator()(int64_t v) const { attr_->set_l(v); }

  void operator()(const std::vector<int64_t> &v) const {
    VectorToRepeated(v, attr_->mutable_longs());
  }

  void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
};

void OpDesc::Flush() {
  if (need_update_) {
    this->desc_.mutable_inputs()->Clear();
    for (auto &ipt : inputs_) {
      auto *input = desc_.add_inputs();
      input->set_parameter(ipt.first);
      VectorToRepeated(ipt.second, input->mutable_arguments());
    }

    this->desc_.mutable_outputs()->Clear();
    for (auto &opt : outputs_) {
      auto *output = desc_.add_outputs();
      output->set_parameter(opt.first);
      VectorToRepeated(opt.second, output->mutable_arguments());
    }

    this->desc_.mutable_attrs()->Clear();
    for (auto &attr : attrs_) {
      auto *attr_desc = desc_.add_attrs();
      attr_desc->set_name(attr.first);
      attr_desc->set_type(
          static_cast<proto::AttrType>(attr.second.which() - 1));
      SetAttrDescVisitor visitor(attr_desc);
      boost::apply_visitor(visitor, attr.second);
    }

    need_update_ = false;
  }
}

static std::once_flag init_infer_shape_funcs;

/**
 * NOTE(paddle-dev): Very tricky code here. Maybe we should find a
 * better way to register compile-time infershape method gentlely.
 *
 * Normally, we can register a class derived from InferShapeBase, so that
 * we can set the field of `infer_shape_` inside OpInfo when registering op.
 *
 * However, there is another way we can set the field of `infer_shape_` inside
 * OpInfo. Usually, we overload InferShape method of OperatorWithKernel. After
 * running the following method InitInferShapeFuncs, `infer_shape_` would be set
 * to be the InferShape method of OperatorWithKernel. That is to say, we borrow
 * the run-time InferShape method of OperatorWithKernel to be the compile-time
 * InferShape method.
 *
 * However, during compiling time, we may not know inputs, outputs and attrs of
 * run-time OperatorWithKernel. So the following code creates a fake
 * OperatorWithKernel object. That is why the field info_ of OperatorBase
 * would be null.
 */
static void InitInferShapeFuncs() {
  std::call_once(init_infer_shape_funcs, [] {
    auto &map = OpInfoMap::Instance();
    auto &info_map = *map.mutable_map();

    for (auto &kern_pair : OperatorWithKernel::AllOpKernels()) {
      auto op_type = kern_pair.first;
      auto it = info_map.find(op_type);
      PADDLE_ENFORCE(it != info_map.end(), "%s has not been registered",
                     op_type);
      auto &op_info = it->second;
      if (op_info.infer_shape_) {  // infer_shape has been registered.
        continue;
      }

      auto op = dynamic_cast<OperatorWithKernel *>(op_info.Creator()(
          "", VariableNameMap{}, VariableNameMap{}, AttributeMap{}));

      PADDLE_ENFORCE_NOT_NULL(
          op, "InferShapeBase is not registered to Operator %s", op_type);

      op_info.infer_shape_ = [op](InferShapeContext *ctx) {
        op->InferShape(ctx);
      };
    }
  });
}

void OpDesc::CheckAttrs() {
  PADDLE_ENFORCE(!Type().empty(),
                 "CheckAttr() can not be called before type is setted.");
  auto *checker = OpInfoMap::Instance().Get(Type()).Checker();
  if (checker == nullptr) {
    // checker is not configured. That operator could be generated by Paddle,
    // not by users.
    return;
  }
  VLOG(10) << "begin to check attribute of " << Type();
  checker->Check(&attrs_);
}

void OpDesc::InferShape(const BlockDesc &block) const {
  VLOG(3) << "CompileTime infer shape on " << Type();
  InitInferShapeFuncs();
  auto &infer_shape = OpInfoMap::Instance().Get(this->Type()).infer_shape_;
  PADDLE_ENFORCE(static_cast<bool>(infer_shape),
                 "%s's infer_shape has not been registered", this->Type());
  CompileTimeInferShapeContext ctx(*this, block);
  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    auto inames = this->InputArgumentNames();
    sout << " From [";
    std::copy(inames.begin(), inames.end(),
              std::ostream_iterator<std::string>(sout, ", "));
    sout << "] to [";
    auto onames = this->OutputArgumentNames();
    std::copy(onames.begin(), onames.end(),
              std::ostream_iterator<std::string>(sout, ", "));
    sout << "]";
    VLOG(10) << sout.str();
  }
  infer_shape(&ctx);
}

void OpDesc::InferVarType(BlockDesc *block) const {
  // There are a few places that var type can be set.
  // When VarDesc is created, default set to LOD_TENSOR.
  // When output variable is created, default is defaut set to LOD_TENSOR.
  // We limit here to be the only place that operator defines its customized
  // var type inference. Hence, we don't do any "default" setting here.
  auto &info = OpInfoMap::Instance().Get(this->Type());
  if (info.infer_var_type_) {
    InferVarTypeContext context(this, block);
    info.infer_var_type_(&context);
  }
}

CompileTimeInferShapeContext::CompileTimeInferShapeContext(
    const OpDesc &op, const BlockDesc &block)
    : op_(op), block_(block) {}

bool CompileTimeInferShapeContext::HasInput(const std::string &name) const {
  const std::vector<std::string> &input_names = op_.Input(name);
  auto length = input_names.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Input(%s) should have only one value, "
                    "but it have %d now",
                    name, length);
  return block_.HasVarRecursive(input_names[0]);
}

bool CompileTimeInferShapeContext::HasOutput(const std::string &name) const {
  const std::vector<std::string> &output_names = op_.Output(name);
  auto length = output_names.size();
  if (length == 0) {
    return false;
  }
  PADDLE_ENFORCE_EQ(length, 1UL,
                    "Output(%s) should have only one value, "
                    "but it have %d now",
                    name, length);
  return block_.HasVarRecursive(output_names[0]);
}

bool CompileTimeInferShapeContext::HasInputs(const std::string &name) const {
  const std::vector<std::string> &input_names = op_.Input(name);
  if (input_names.empty()) {
    return false;
  }
  for (auto &input : input_names) {
    if (!block_.HasVarRecursive(input)) return false;
  }
  return true;
}

bool CompileTimeInferShapeContext::HasOutputs(const std::string &name) const {
  const std::vector<std::string> &output_names = op_.Output(name);
  if (output_names.empty()) {
    return false;
  }
  for (auto &output : output_names) {
    if (!block_.HasVarRecursive(output)) return false;
  }
  return true;
}

AttrReader CompileTimeInferShapeContext::Attrs() const {
  return AttrReader(op_.GetAttrMap());
}

const std::vector<std::string> &CompileTimeInferShapeContext::Inputs(
    const std::string &name) const {
  return op_.Input(name);
}

const std::vector<std::string> &CompileTimeInferShapeContext::Outputs(
    const std::string &name) const {
  return op_.Output(name);
}

std::vector<DDim> CompileTimeInferShapeContext::GetRepeatedDims(
    const std::string &name) const {
  auto var = block_.FindVarRecursive(name);
  PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s", name);
  std::vector<DDim> res;
  try {
    auto shapes = var->GetShapes();
    for (const auto &s : shapes) {
      res.push_back(s.empty() ? make_ddim({0UL}) : make_ddim(s));
    }
  } catch (...) {
    VLOG(5) << "GetRepeatedDim of variable " << name << " error.";
    std::rethrow_exception(std::current_exception());
  }
  return res;
}

void CompileTimeInferShapeContext::SetDim(const std::string &name,
                                          const DDim &dim) {
  block_.FindVarRecursive(name)->SetShape(vectorize(dim));
}

void CompileTimeInferShapeContext::SetRepeatedDims(
    const std::string &name, const std::vector<DDim> &dims) {
  auto var = block_.FindVarRecursive(name);
  PADDLE_ENFORCE(var != nullptr, "Cannot find variable %s", name);
  std::vector<std::vector<int64_t>> dim_vec(dims.size());
  std::transform(dims.begin(), dims.end(), dim_vec.begin(), vectorize);
  var->SetShapes(dim_vec);
}

bool CompileTimeInferShapeContext::IsRuntime() const { return false; }

proto::VarType::Type CompileTimeInferShapeContext::GetVarType(
    const std::string &name) const {
  return block_.FindVarRecursive(name)->GetType();
}

}  // namespace framework
}  // namespace paddle
