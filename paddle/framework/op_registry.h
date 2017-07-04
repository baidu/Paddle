#pragma once

#include <unordered_map>
#include "paddle/framework/attr_checker.h"
#include "paddle/framework/enforce.h"
#include "paddle/framework/op_base.h"
#include "paddle/framework/op_proto.pb.h"

namespace paddle {
namespace framework {

// this class not only make proto but also init attribute checkers.
class OpProtoAndCheckerMaker {
 public:
  OpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : proto_(proto), op_checker_(op_checker) {}

 protected:
  void AddInput(const std::string& name, const std::string& comment) {
    // add input to proto_
  }
  void AddOutput(const std::string& name, const std::string& comment) {
    // add output to proto_
  }
  template <typename T>
  TypedAttrChecker<T>& AddAttr(const std::string& name,
                               const std::string& comment) {
    // add attribute to proto_
    return op_checker_->AddAttrChecker<T>(name);
  }
  void AddType(const std::string& op_type) {
    // add type to proto_
  }
  void AddComment(const std::string& comment) {
    // add comment to proto_
  }

  OpProto* proto_;
  OpAttrChecker* op_checker_;
};

class OpRegistry {
  typedef std::function<OperatorBase*(OpDesc& op_desc)> OpCreator;

 public:
  template <typename OpType, typename ProtoMakerType>
  static void RegisterOp(const std::string& op_type) {
    creators_[op_type] = [](const OpDesc& op_desc) {
      return new OpType(op_desc);
    };
    OpProto& op_proto = protos_[op_type];
    OpAttrChecker& op_checker = op_checkers_[op_type];
    ProtoMakerType(&op_proto, &op_checker);
  }

  static OpBase* CreateOp(const std::string& op_type, OpDesc op_desc) const {
    const OpAttrChecker& op_checker = op_checkers_.at(op_type);
    PADDLE_ENFORCE(op_checker.PassCheck(op_desc), "op attribute error.");
    return (creators_.at(op_type))(op_desc);
  }

 private:
  static std::unordered_map<std::string, OpCreator> creators_;
  static std::unordered_map<std::string, OpProto> protos_;
  static std::unordered_map<std::string, OpAttrChecker> op_checkers_;
};

template <typename OpType, typename ProtoMakerType>
class OpRegisterHelper {
 public:
  OpRegisterHelper(std::string op_type) {
    OpRegistry::RegisterOp<OpType, ProtoMakerType>(op_type);
  }
};

#define REGISTER_OP(op_class, op_maker_class, op_type)             \
  class op_class##Register {                                       \
   private:                                                        \
    const static OpRegisterHelper<#op_class, #op_maker_class> reg; \
  };                                                               \
  const Register op_class##Register::reg(#op_type);

class CosineOp {
  // ...
};

class CosineOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  CosineOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddType("cos");
    AddComment("This is cos op");
  }
};

REGISTER_OP(CosineOp, CosineOpProtoAndCheckerMaker, "cos");

}  // namespace framework
}  // namespace paddle
