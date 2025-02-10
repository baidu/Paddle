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

#include "paddle/fluid/framework/ir/generate_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/utils/blank.h"

namespace paddle::framework::ir {

class element_visitor {
 public:
  explicit element_visitor(int index) : index_(index) {}

  template <typename T>
  Attribute operator()(const T& attr UNUSED) const {
    PADDLE_THROW(common::errors::Unimplemented("Unimplemented operand."));
  }

  template <typename T>
  Attribute operator()(const std::vector<T>& attr) const {
    using ET = std::conditional_t<std::is_same<T, double>::value, float, T>;
    int index = index_;
    if (index < 0) {
      index += attr.size();
    }
    if (index >= 0 && static_cast<size_t>(index) < attr.size()) {
      return static_cast<ET>(attr[index]);
    }
    return paddle::blank();
  }

 private:
  int index_;
};

template <>
Attribute element_visitor::operator()(
    const std::vector<::pir::Value>& attr UNUSED) const {
  PADDLE_THROW(common::errors::Unimplemented("Unimplemented operand."));
}

class operation_visitor {
 public:
  explicit operation_visitor(const proto::PassDesc::OperationType& type)
      : type_(type) {}

  template <typename T1, typename T2>
  Attribute operator()(const T1& attr UNUSED,
                       const T2& operation UNUSED) const {
    PADDLE_THROW(common::errors::Unimplemented("Unimplemented operand."));
  }

  template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  Attribute operator()(const T& attr, const T& operation) const {
    switch (type_) {
      case proto::PassDesc_OperationType_kSub: {
        return attr - operation;
      }

      case proto::PassDesc_OperationType_kMod: {
        return attr % operation;
      }

      default:
        PADDLE_THROW(
            common::errors::Unimplemented("Unimplemented operation type."));
    }
  }

 private:
  proto::PassDesc::OperationType type_;
};

Attribute GetVarAttrValue(const VarDesc* desc,
                          const proto::PassDesc::Attr& attr) {
  if ("shape" == attr.name()) {
    std::vector<int64_t> shape = desc->GetShape();
    if (attr.has_operation()) {
      if (attr.operation() == proto::PassDesc_OperationType_kSize) {
        return static_cast<int>(shape.size());
      }
    } else if (attr.has_element_index()) {
      int element_index = attr.element_index();
      if (attr.element_index() < 0) {
        element_index += static_cast<int>(shape.size());
      }
      if (element_index >= 0 &&
          static_cast<size_t>(element_index) < shape.size()) {
        return static_cast<int>(shape[element_index]);
      }
    } else {
      return shape;
    }
  }
  return paddle::blank();
}

Attribute GetOpAttrValue(const OpDesc* desc,
                         const proto::PassDesc::Attr& attr) {
  Attribute value = desc->GetAttr(attr.name());
  if (attr.has_element_index()) {
    value = paddle::visit(element_visitor(attr.element_index()), value);
  }
  return value;
}

void InitGeneratePattern(const proto::PassDesc& pass_desc, PDPattern* pattern) {
  // Traverse all operators to create subgraph.
  for (int index = 0; index < pass_desc.pattern_size(); ++index) {
    const proto::OpDesc& op = pass_desc.pattern(index);
    // Create a PDNode for current operator. Use the index as name to avoid
    // multiple operators with same type. Get a PDNode from pattern subgraph
    // through index in rewrite phase.
    PDNode* op_pdnode =
        pattern->NewNode(std::to_string(index))->assert_is_op(op.type());
    // Create PDNodes for inputs of current operator.
    for (const proto::OpDesc::Var& var : op.inputs()) {
      for (int n = 0; n < var.arguments_size(); ++n) {
        const std::string& argument = var.arguments(n);
        // The input may be the output of other operator.
        PDNode* var_pdnode = pattern->RetrieveNode(argument);
        if (nullptr == var_pdnode) {
          var_pdnode = pattern->NewNode(argument)->AsInput();
          var_pdnode->assert_is_var();
        } else if (var_pdnode->IsOutput()) {
          var_pdnode->AsIntermediate();
        }
        var_pdnode->assert_more([&](Node* x) {
          for (auto* out : x->outputs) {
            if (out->IsOp() && out->Op()->Type() == op.type()) {
              const auto& inputs = out->Op()->Inputs();
              const auto& iter = inputs.find(var.parameter());
              if (inputs.end() != iter) {
                if (iter->second.end() != std::find(iter->second.begin(),
                                                    iter->second.end(),
                                                    x->Name())) {
                  return true;
                }
              }
            }
          }
          return false;
        });
        pattern->AddEdge(var_pdnode, op_pdnode);
      }
    }
    // Create PDNodes for outputs of current operator.
    for (const proto::OpDesc::Var& var : op.outputs()) {
      for (const std::string& argument : var.arguments()) {
        // The output may be the input of other operator.
        PDNode* var_pdnode = pattern->RetrieveNode(argument);
        if (nullptr == var_pdnode) {
          var_pdnode = pattern->NewNode(argument)->AsOutput();
          var_pdnode->assert_is_var();
          var_pdnode->assert_more([&](Node* x) {
            for (Node* input : x->inputs) {
              if (input && input->IsOp() && input->Op() &&
                  input->Op()->Type() == op.type()) {
                const auto& outputs = input->Op()->Outputs();
                const auto& iter = outputs.find(var.parameter());
                if (outputs.end() != iter) {
                  if (iter->second.end() != std::find(iter->second.begin(),
                                                      iter->second.end(),
                                                      x->Name())) {
                    return true;
                  }
                }
              }
            }
            return false;
          });
        } else if (var_pdnode->IsInput()) {
          var_pdnode->AsIntermediate();
        }
        var_pdnode->assert_is_op_output(op.type());
        pattern->AddEdge(op_pdnode, var_pdnode);
      }
    }
    // Set attribute condition for current operator.
    for (const proto::OpDesc::Attr& attr : op.attrs()) {
      op_pdnode->assert_more([&](Node* x) {
        if (x && x->IsOp()) {
          OpDesc* op_desc = x->Op();
          if (op_desc->HasAttr(attr.name())) {
            return GetAttrValue(attr) == op_desc->GetAttr(attr.name());
          }
          return false;
        }
        return false;
      });
    }
  }
  for (const auto& condition : pass_desc.var_attr_conditions()) {
    if (condition.has_condition_value()) {
      PDNode* pdnode = pattern->RetrieveNode(condition.attr().var_name());
      pdnode->assert_more([&](Node* x) {
        Attribute attr = GetVarAttrValue(x->Var(), condition.attr());
        if (condition.has_operation()) {
          Attribute operation = GetAttrValue(condition.operation().value());
          attr = paddle::visit(
              operation_visitor(condition.operation().type()), attr, operation);
        }
        switch (condition.type()) {
          case proto::PassDesc_ConditionType_kEQ: {
            return attr == GetAttrValue(condition.condition_value());
          }

          default:
            PADDLE_THROW(
                common::errors::Unimplemented("Unimplemented condition type."));
        }
      });
    }
  }

  // The output of the pattern must be marked as AsOutput.
  for (const proto::PassDesc::VarMap& var_map : pass_desc.var_maps()) {
    PDNode* var_pdnode = pattern->RetrieveNode(var_map.pattern_var());
    PADDLE_ENFORCE_NOT_NULL(
        var_pdnode,
        common::errors::NotFound("Not found the var %s in the pattern.",
                                 var_map.pattern_var()));
    var_pdnode->AsOutput();
  }
}

// There are some duplicate patterns.
bool IsDuplicatePattern(const GraphPatternDetector::subgraph_t& subgraph,
                        Graph* graph) {
  for (auto iter : subgraph) {
    if (nullptr == graph->RetrieveNode(iter.second->id())) {
      VLOG(3) << "Node [" << iter.second->Name()
              << "] of subgraph has been removed. So skip this optimize.";
      return true;
    }
  }
  return false;
}

GraphPatternDetector::handle_t GetGenerateDelete(
    const PDPattern& pattern, const proto::PassDesc& pass_desc) {
  GraphPatternDetector::handle_t handler =
      [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
        if (IsDuplicatePattern(subgraph, graph)) {
          return;
        }
        // `var_node_maps` record the mapping of variable to the pattern
        // subgraph.
        std::map<std::string, Node*> var_node_maps;
        for (const proto::PassDesc::VarMap& var_map : pass_desc.var_maps()) {
          Node* node = subgraph.at(pattern.RetrieveNode(var_map.pattern_var()));
          const auto& iter = var_node_maps.find(var_map.replace_var());
          if (var_node_maps.end() == iter) {
            // first node is input
            var_node_maps.insert({var_map.replace_var(), node});
          } else {
            // output node
            for (Node* s_node : node->outputs) {
              iter->second->outputs.push_back(s_node);
              std::replace(s_node->inputs.begin(),
                           s_node->inputs.end(),
                           node,
                           iter->second);
              s_node->Op()->RenameInput(node->Name(), iter->second->Name());
            }
          }
        }
        // Remove nodes that are intermediate.
        std::unordered_set<const Node*> remove_nodes;
        for (const std::unique_ptr<PDNode>& pdnode : pattern.nodes()) {
          remove_nodes.emplace(subgraph.at(pdnode.get()));
        }
        for (auto const& iter : var_node_maps) {
          remove_nodes.erase(iter.second);
        }
        GraphSafeRemoveNodes(graph, remove_nodes);
      };
  return handler;
}

GraphPatternDetector::handle_t GetGenerateRewrite(
    const PDPattern& pattern, const proto::PassDesc& pass_desc) {
  GraphPatternDetector::handle_t handler =
      [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
        if (IsDuplicatePattern(subgraph, graph)) {
          return;
        }
        for (const auto& condition : pass_desc.var_attr_conditions()) {
          if (condition.has_condition_attr()) {
            Node* node =
                subgraph.at(pattern.RetrieveNode(condition.attr().var_name()));
            Attribute node_attr =
                GetVarAttrValue(node->Var(), condition.attr());
            Attribute condition_attr;
            if (condition.condition_attr().role() ==
                proto::PassDesc_RoleType_kVariable) {
              Node* condition_node = subgraph.at(
                  pattern.RetrieveNode(condition.attr().var_name()));
              condition_attr = GetVarAttrValue(condition_node->Var(),
                                               condition.condition_attr());
            } else {
              PADDLE_THROW(common::errors::Unimplemented(
                  "Unimplemented for operation."));
            }
            bool check_failed = false;
            if (condition.type() == proto::PassDesc_ConditionType_kEQ) {
              check_failed = !(node_attr == condition_attr);
            }
            if (check_failed) {
              VLOG(3) << "Check var [" << node->Name() << "] with attr ["
                      << condition.attr().name()
                      << "] failed, skip this pattern.";
              return;
            }
          }
        }
        // `var_node_maps` record the mapping of variable to the pattern
        // subgraph.
        std::map<std::string, std::vector<Node*>> var_node_maps;
        for (const proto::PassDesc::VarMap& var_map : pass_desc.var_maps()) {
          Node* node = subgraph.at(pattern.RetrieveNode(var_map.pattern_var()));
          var_node_maps[var_map.replace_var()].emplace_back(node);
        }
        // Traverse all operators to create subgraph.
        for (int index = 0; index < pass_desc.replace_size(); ++index) {
          const proto::OpDesc& op = pass_desc.replace(index);
          OpDesc op_desc;
          std::vector<Node*> in_nodes, out_nodes;
          op_desc.SetType(op.type());
          // Create Nodes for inputs of current operator.
          for (const proto::OpDesc::Var& var : op.inputs()) {
            std::vector<std::string> arguments;
            for (const std::string& argument : var.arguments()) {
              // The input may be mapped on the operator of pattern subgraph.
              if (var_node_maps[argument].empty()) {
                VarDesc var_desc(patterns::UniqueKey(argument));
                var_node_maps[argument].emplace_back(
                    graph->CreateVarNode(&var_desc));
              }
              in_nodes.push_back(var_node_maps[argument][0]);
              arguments.push_back(var_node_maps[argument][0]->Name());
            }
            op_desc.SetInput(var.parameter(), arguments);
          }
          // Create Nodes for outputs of current operator.
          for (const proto::OpDesc::Var& var : op.outputs()) {
            std::vector<std::string> arguments;
            for (const std::string& argument : var.arguments()) {
              // The output may be mapped on the operator of pattern subgraph.
              if (var_node_maps[argument].empty()) {
                VarDesc var_desc(patterns::UniqueKey(argument));
                var_node_maps[argument].emplace_back(
                    graph->CreateVarNode(&var_desc));
              }
              if (in_nodes.end() == std::find(in_nodes.begin(),
                                              in_nodes.end(),
                                              var_node_maps[argument][0])) {
                out_nodes.push_back(var_node_maps[argument][0]);
              } else {
                out_nodes.push_back(
                    graph->CreateVarNode(var_node_maps[argument][0]->Var()));
              }
              arguments.push_back(var_node_maps[argument][0]->Name());
            }
            op_desc.SetOutput(var.parameter(), arguments);
          }
          // Set attribute for current operator.
          for (const proto::OpDesc::Attr& attr : op.attrs()) {
            op_desc.SetAttr(attr.name(), GetAttrValue(attr));
          }
          for (const auto& attr_map : pass_desc.op_attr_maps()) {
            if (attr_map.replace_attr().op_index() == index) {
              Attribute attr;
              if (attr_map.pattern_attr().role() ==
                  proto::PassDesc_RoleType_kVariable) {
                Node* condition_node = subgraph.at(
                    pattern.RetrieveNode(attr_map.pattern_attr().var_name()));
                attr = GetVarAttrValue(condition_node->Var(),
                                       attr_map.pattern_attr());
              } else {
                Node* condition_node = subgraph.at(pattern.RetrieveNode(
                    std::to_string(attr_map.pattern_attr().op_index())));
                attr = GetOpAttrValue(condition_node->Op(),
                                      attr_map.pattern_attr());
              }
              if (attr_map.has_operation()) {
                Attribute operation =
                    GetAttrValue(attr_map.operation().value());
                attr = paddle::visit(
                    operation_visitor(attr_map.operation().type()),
                    attr,
                    operation);
              }
              op_desc.SetAttr(attr_map.replace_attr().name(), attr);
            }
          }
          // Create a Node for current operator.
          Node* op_node = graph->CreateOpNode(&op_desc);
          for (Node* node : in_nodes) {
            IR_NODE_LINK_TO(node, op_node);
          }
          for (Node* node : out_nodes) {
            IR_NODE_LINK_TO(op_node, node);
          }
        }
        // Remove nodes that are intermediate.
        std::unordered_set<const Node*> remove_nodes;
        for (const std::unique_ptr<PDNode>& pdnode : pattern.nodes()) {
          remove_nodes.emplace(subgraph.at(pdnode.get()));
        }
        for (auto const& iter : var_node_maps) {
          for (auto& node : iter.second) {
            remove_nodes.erase(node);
          }
        }
        GraphSafeRemoveNodes(graph, remove_nodes);

        // Replace the redundant node by the first node in var_nodes_nmaps.
        remove_nodes.clear();
        for (auto iter : var_node_maps) {
          auto var_node = iter.second[0];
          for (size_t i = 1; i < iter.second.size(); ++i) {
            auto replaced_var_node = iter.second[i];
            for (auto op_node : replaced_var_node->outputs) {
              auto index = std::find(op_node->inputs.begin(),
                                     op_node->inputs.end(),
                                     replaced_var_node) -
                           op_node->inputs.begin();
              op_node->inputs[index] = var_node;

              auto& input_name_maps = *op_node->Op()->MutableInputs();
              for (auto& item : input_name_maps) {
                auto iter = std::find(item.second.begin(),
                                      item.second.end(),
                                      replaced_var_node->Name());
                if (iter != item.second.end()) {
                  item.second[iter - item.second.begin()] = var_node->Name();
                  input_name_maps[item.first] = item.second;
                  break;
                }
              }
              op_node->Op()->Flush();
            }
            remove_nodes.emplace(replaced_var_node);
          }
        }
        GraphSafeRemoveNodes(graph, remove_nodes);
      };
  return handler;
}

GeneratePass::GeneratePass(const std::string& binary_str,
                           const std::string& pass_type) {
  RegisterType(pass_type);
  multi_pass_desc_.ParseFromString(binary_str);
  VerifyDesc();
}

GeneratePass::GeneratePass(const proto::MultiPassDesc& multi_pass_desc,
                           const std::string& pass_type)
    : multi_pass_desc_(multi_pass_desc) {
  RegisterType(pass_type);
  VerifyDesc();
}

void GeneratePass::ApplyImpl(Graph* graph) const {
  for (const proto::PassDesc& pass_desc : multi_pass_desc_.pass_descs()) {
    GraphPatternDetector detector;
    InitGeneratePattern(pass_desc, detector.mutable_pattern());
    if (pass_desc.replace_size() == 0) {
      detector(graph, GetGenerateDelete(detector.pattern(), pass_desc));
    } else {
      detector(graph, GetGenerateRewrite(detector.pattern(), pass_desc));
    }
    // The rewritten graph needs to be verified. Current Pass should be skipped
    // if validation failed. Rewrite based on the original graph cannot
    // implement rollback operation.
    VerifyGraph(*graph);
  }
}

void GeneratePass::VerifyDesc() const {
  PADDLE_ENFORCE_NE(
      multi_pass_desc_.pass_descs_size(),
      0,
      common::errors::InvalidArgument("Size of PassDesc should not be empty."));
}

bool GeneratePass::VerifyGraph(const Graph& graph) {
  // Return true temporarily.
  return true;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::generate_pass {

VarHelper::VarHelper(const char* name) : name_(name), type_(Type::kInput) {}
VarHelper::VarHelper(const std::string& name, Type type)
    : name_(name), type_(type) {}

OpHelper::OpHelper(const char* type, SubgraphHelper* subgraph_helper)
    : type_(type), subgraph_helper_(subgraph_helper) {
  op_desc_ = subgraph_helper_->ProgramDesc()->mutable_blocks(0)->add_ops();
  op_desc_->set_type(type_);
}

OpHelper::Arguments::Arguments(const char* parameter,
                               const VarHelper& var_helper)
    : parameter_(parameter) {
  var_helpers_.push_back(var_helper);
}

OpHelper::Arguments::Arguments(const char* parameter,
                               std::initializer_list<VarHelper> var_helpers)
    : parameter_(parameter), var_helpers_(var_helpers) {}

OpHelper& OpHelper::operator()(const Arguments& input) {
  proto::OpDesc::Var* var = op_desc_->add_inputs();
  var->set_parameter(input.parameter_);
  for (const VarHelper& var_helper : input.var_helpers_) {
    var->add_arguments()->assign(var_helper.name_);
    if (VarHelper::Type::kInput == var_helper.type_) {
      subgraph_helper_->AddInputVar(var_helper.name_);
    }
  }
  return *this;
}

OpHelper& OpHelper::operator()(std::initializer_list<Arguments> inputs) {
  for (const auto& input : inputs) {
    operator()(input);
  }
  return *this;
}

VarHelper OpHelper::Out(const char* name) {
  std::string argument = patterns::UniqueKey(type_);
  proto::OpDesc::Var* var = op_desc_->add_outputs();
  var->set_parameter(name);
  var->add_arguments()->assign(argument);
  return VarHelper(argument, VarHelper::Type::kOutput);
}

proto::ProgramDesc* SubgraphHelper::ProgramDesc() { return &program_desc_; }

const proto::ProgramDesc& SubgraphHelper::ProgramDesc() const {
  return program_desc_;
}

const std::vector<std::string>& SubgraphHelper::InputVars() const {
  return input_vars_;
}

const std::vector<std::string>& SubgraphHelper::OutputVars() const {
  return output_vars_;
}

void SubgraphHelper::AddInputVar(const std::string& name) {
  auto iter = std::find(input_vars_.begin(), input_vars_.end(), name);
  if (input_vars_.end() == iter) {
    input_vars_.push_back(name);
  }
}

void SubgraphHelper::AddOutputVars(const VarHelper& var_helper) {
  output_vars_.push_back(var_helper.name_);
}

}  // namespace paddle::framework::ir::generate_pass
namespace paddle::framework::ir {

PassPairs::PassPairs(const SubgraphType& pattern, const SubgraphType& replace) {
  AddPassDesc(pattern, replace);
}

void PassPairs::AddPassDesc(const SubgraphType& pattern,
                            const SubgraphType& replace) {
  proto::PassDesc* pass_desc = multi_pass_desc_.add_pass_descs();
  pass_desc->mutable_pattern()->CopyFrom(pattern.ProgramDesc().blocks(0).ops());
  pass_desc->mutable_replace()->CopyFrom(replace.ProgramDesc().blocks(0).ops());
  PADDLE_ENFORCE_EQ(pattern.InputVars().size(),
                    replace.InputVars().size(),
                    common::errors::InvalidArgument(
                        "Size of lambda expression arguments is not equal "
                        "between pattern/replace subgraph."));
  for (size_t i = 0; i < pattern.InputVars().size(); i++) {
    proto::PassDesc::VarMap* var_map = pass_desc->add_var_maps();
    var_map->set_pattern_var(pattern.InputVars()[i]);
    var_map->set_replace_var(replace.InputVars()[i]);
  }
  PADDLE_ENFORCE_EQ(pattern.OutputVars().size(),
                    replace.OutputVars().size(),
                    common::errors::InvalidArgument(
                        "Size of lambda expression returns is not equal "
                        "between pattern/replace subgraph."));
  for (size_t i = 0; i < pattern.OutputVars().size(); i++) {
    proto::PassDesc::VarMap* var_map = pass_desc->add_var_maps();
    var_map->set_pattern_var(pattern.OutputVars()[i]);
    var_map->set_replace_var(replace.OutputVars()[i]);
  }
}

const proto::MultiPassDesc& PassPairs::MultiPassDesc() const {
  return multi_pass_desc_;
}

}  // namespace paddle::framework::ir
