/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/onednn/cpu_bfloat16_pass.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

namespace {
class Quanter {
 public:
  void AddQuantOps() {
    if (IsNotPermittedOpType()) return;

    std::unordered_map<std::string, std::string> linked_xputs;

    for (const auto& logical_xput : op_xputs) {
      std::vector<std::string> quant_xput_names;
      quant_xput_names.reserve(xputs_map.size());

      const auto& logical_xput_name = logical_xput.first;
      if (IsNotPermittedName(logical_xput_name)) continue;

      const auto& physical_xputs_names = logical_xput.second;
      for (const auto& physical_xput_name : physical_xputs_names) {
        // In case the input is repetitively used, where the input should be
        // still added.
        if (IsAlreadyLinked(linked_xputs, physical_xput_name)) {
          quant_xput_names.emplace_back(linked_xputs[physical_xput_name]);
          continue;
        }

        VarDesc quant_x_desc(
            patterns::PDNodeName(get_op_type(), get_op_edge()));
        auto quant_x_node = graph->CreateVarNode(&quant_x_desc);
        const auto xput_name = quant_x_node->Name();
        quant_xput_names.emplace_back(xput_name);

        auto quant_op = create_quant_op(physical_xput_name, xput_name);

        auto physical_xput_node = xputs_map[physical_xput_name];
        link_nodes(physical_xput_node, quant_op, quant_x_node);
        counter++;
        linked_xputs[physical_xput_name] = xput_name;
      }

      set_edge(logical_xput_name, quant_xput_names);
    }
  }

  int get_counter() const { return counter; }

  virtual ~Quanter() = default;

 protected:
  Graph* graph;
  ir::Node* const op;

  std::map<std::string, ir::Node*> xputs_map;
  const VariableNameMap& op_xputs;

  int counter = 0;

  Quanter(Graph* const graph,
          ir::Node* const op,
          const VariableNameMap& op_xputs)
      : graph(graph), op(op), op_xputs(op_xputs) {}

  virtual bool IsNotPermittedOpType() const = 0;
  virtual bool IsNotPermittedName(const std::string& input_name) const = 0;
  virtual std::string get_op_type() const = 0;
  virtual std::string get_op_edge() const = 0;
  virtual void link_nodes(ir::Node* const physical_xput_node,
                          ir::Node* const quant_op,
                          ir::Node* const quant_x_node) = 0;
  virtual void set_edge(const std::string& logical_xput_name,
                        const std::vector<std::string>& quant_xput_names) = 0;

  bool IsAlreadyLinked(
      const std::unordered_map<std::string, std::string>& node_names_map,
      const std::string& node_name) const {
    return node_names_map.find(node_name) != node_names_map.end();
  }

  virtual ir::Node* create_quant_op(const std::string& input_name,
                                    const std::string& output_name) const {
    OpDesc op_desc;
    op_desc.SetType(get_op_type());

    op_desc.SetInput("Input", std::vector<std::string>({input_name}));
    op_desc.SetOutput("Output", std::vector<std::string>({output_name}));
    op_desc.SetAttr("Scale", 1.f);
    op_desc.SetAttr("Shift", 0.0f);
    op_desc.SetAttr("bfloat16", true);
    op_desc.SetAttr("output_format",
                    op->Op()->HasAttr("data_layout")
                        ? op->Op()->GetAttr("data_layout")
                        : std::string("NCHW"));
    return graph->CreateOpNode(&op_desc);  // OpDesc will be copied.
  }

  void UnlinkNodes(ir::Node* a, ir::Node* b) const {
    a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                     a->outputs.end());
    b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                    b->inputs.end());
  }
};

class Quantizer final : public Quanter {
 public:
  Quantizer(Graph* const graph, ir::Node* const op)
      : Quanter(graph, op, op->Op()->Inputs()) {
    auto inputs = op->inputs;
    PADDLE_ENFORCE_GE(
        inputs.size(),
        1,
        common::errors::InvalidArgument(
            "OP(%s)'s inputs(%d) must be equal or greater than 1.",
            op->Name(),
            inputs.size()));

    for (auto input : inputs) xputs_map[input->Name()] = input;
  }

 protected:
  bool IsNotPermittedOpType() const override { return false; }

  // Checking whether a reorder from FP32 to BF16
  // should be added before the input to the operator
  bool IsNotPermittedName(const std::string& input_name) const override {
    // Only the inputs listed in \"permitted_names\"
    // requires quanitization before the bfloat16 operator.
    // Other inputs, such as Filter and Bias are reordered in the kernel.
    const std::vector<std::string> permitted_names = {
        "X", "Y", "Input", "ResidualData"};

    return std::none_of(
        permitted_names.begin(),
        permitted_names.end(),
        [&input_name](const std::string& name) { return name == input_name; });
  }

  std::string get_op_type() const override { return "quantize"; };
  std::string get_op_edge() const override { return "out"; };

  void link_nodes(ir::Node* const physical_xput_node,
                  ir::Node* const quant_op,
                  ir::Node* const quant_x_node) override {
    UnlinkNodes(physical_xput_node, op);
    IR_NODE_LINK_TO(physical_xput_node, quant_op);
    IR_NODE_LINK_TO(quant_op, quant_x_node);
    IR_NODE_LINK_TO(quant_x_node, op);
  }

  void set_edge(const std::string& logical_xput_name,
                const std::vector<std::string>& quant_xput_names) override {
    op->Op()->SetInput(logical_xput_name, quant_xput_names);
  }
};

class DeQuantizer final : public Quanter {
 public:
  DeQuantizer(Graph* const graph, ir::Node* const op)
      : Quanter(graph, op, op->Op()->Outputs()) {
    auto outputs = op->outputs;
    PADDLE_ENFORCE_GE(
        outputs.size(),
        1,
        common::errors::InvalidArgument(
            "OP(%s)'s outputs(%d) must be equal or greater than 1.",
            op->Name(),
            outputs.size()));

    for (auto output : outputs) xputs_map[output->Name()] = output;
  }

 protected:
  bool IsNotPermittedOpType() const override {
    // Prior_box operator output is always FP32 so no dequantization is needed.
    return op->Op()->Type() == "prior_box";
  }

  // Checking whether a reorder from BF16 to FP32
  // should be added after the output to the operator
  bool IsNotPermittedName(const std::string& output_name) const override {
    std::unordered_map<std::string, std::vector<std::string>> block_list{
        {"layer_norm",
         {"Mean", "Variance"}}};  // not used in inference in oneDNN

    std::vector<std::string> blocked_outputs{"XShape"};  // blocklist for any op
    auto op_name = op->Name();
    if (block_list.count(op_name)) {
      const auto& op_blocklist = block_list[op_name];
      blocked_outputs.insert(
          blocked_outputs.begin(), op_blocklist.begin(), op_blocklist.end());
    }

    return std::any_of(blocked_outputs.begin(),
                       blocked_outputs.end(),
                       [&output_name](const std::string& name) {
                         return name == output_name;
                       });
  }

  std::string get_op_type() const override { return "dequantize"; };
  std::string get_op_edge() const override { return "in"; };

  void link_nodes(ir::Node* const physical_xput_node,
                  ir::Node* const quant_op,
                  ir::Node* const quant_x_node) override {
    UnlinkNodes(op, physical_xput_node);
    IR_NODE_LINK_TO(quant_op, physical_xput_node);
    IR_NODE_LINK_TO(quant_x_node, quant_op);
    IR_NODE_LINK_TO(op, quant_x_node);
  }

  void set_edge(const std::string& logical_xput_name,
                const std::vector<std::string>& quant_xput_names) override {
    op->Op()->SetOutput(logical_xput_name, quant_xput_names);
  }

  ir::Node* create_quant_op(const std::string& input_name,
                            const std::string& output_name) const override {
    return Quanter::create_quant_op(output_name, input_name);
  }
};
}  // namespace
using string::PrettyLogDetail;

void CPUBFloat16Pass::ApplyImpl(ir::Graph* graph) const {
  int quantize_counter = 0;
  int dequantize_counter = 0;

  GraphPatternDetector gpd;
  patterns::Bloat16Ops Bloat16Ops{gpd.mutable_pattern(), "Bloat16Ops"};
  Bloat16Ops();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, Bloat16Ops);

    Quantizer quantizer(graph, op);
    quantizer.AddQuantOps();
    quantize_counter += quantizer.get_counter();

    DeQuantizer dequantizer(graph, op);
    dequantizer.AddQuantOps();
    dequantize_counter += dequantizer.get_counter();
  };
  gpd(graph, handler);

  PrettyLogDetail("---    added %d quantize ops before bfloat16 op",
                  quantize_counter);
  PrettyLogDetail("---    added %d dequantize ops after bfloat16 op",
                  dequantize_counter);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(cpu_bfloat16_pass, paddle::framework::ir::CPUBFloat16Pass);

REGISTER_PASS_CAPABILITY(cpu_bfloat16_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().GE(
            "quantize", 1));
