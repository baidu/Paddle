#include "paddle/fluid/framework/ir/seq_concat_fc_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"

namespace paddle {
namespace framework {
namespace ir {

struct FuseExpr {};

// sequence expand, concat fuse pattern, return concat's output
PDNode* BuildSeqExpandConcatPattern(PDPattern* pattern) {
  // The following operators will be fused:
  // concat
  // sequence_expand
  // sequence_expand

  // The following variables will be treat as inputs:
  // concat mid input, 0th input for fused op
  // sequence_expand input, 1th input for fused op
  // sequence_expand input, 2th input for fused op

  // The following variables will be treat as outputs:
  // concat output

  // So the following variables will be removed:
  // sequence-expand output
  // sequence-expand output

  // Three operators
  auto* sequence_expand0 = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "sequence_expand";
      },
      "sequence_expand0");

  auto* sequence_expand1 = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "sequence_expand";
      },
      "sequence_expand1");

  auto* concat = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "concat" &&  // basic check
               x->Op()->Input("X").size() == 3;                  // Special case
      },
      "concat");

  auto* sequence_expand0_in = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() && VarLinksToOp(x, "sequence_expand");
      },
      "sequence_expand0_in");
  auto* sequence_expand1_in = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() && VarLinksToOp(x, "sequence_expand");
      },
      "sequence_expand1_in");

  // The variables
  auto* sequence_expand0_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&
               VarLinksFromOp(x, "sequence_expand") &&  // basic check
               VarLinksToOp(x, "concat") &&             // is concat's input
               IsNthInput(x, x->outputs[0], "X", 1);    // X[0]
      },
      "sequence_expand0_out");

  auto* sequence_expand1_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&
               VarLinksFromOp(x, "sequence_expand") &&  // basic check
               VarLinksToOp(x, "concat") &&             // is concat's input
               IsNthInput(x, x->outputs[0], "X", 2);    // x[2]
      },
      "sequence_expand1_out");

  auto* concat_in0 = pattern->NewNode(
      [](Node* x) { return x && x->IsVar() && VarLinksToOp(x, "concat"); },
      "concat_in0");

  auto* concat_out = pattern->NewNode(
      [](Node* x) { return x && x->IsVar() && VarLinksFromOp(x, "concat"); },
      "concat_out");

  // Links
  PDLINK(sequence_expand0_in, sequence_expand0);
  PDLINK(sequence_expand1_in, sequence_expand1);
  PDLINK(sequence_expand0, sequence_expand0_out);
  PDLINK(sequence_expand1, sequence_expand1_out);
  PDLINK(sequence_expand0_out, concat);
  PDLINK(sequence_expand1_out, concat);
  PDLINK(concat_in0, concat);
  PDLINK(concat, concat_out);
  return concat_out;
}

PDNode* BuildFCPattern(PDPattern* pattern, PDNode* fc_x) {
  PDNode* fc_w = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                 // basic
               VarLinksToOp(x, "mul") &&          // link
               x->Var()->Proto()->persistable();  // is a parameter
      },
      "fc_w");

  PDNode* mul_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                     // basic
               VarLinksFromOp(x, "mul") &&            // link
               VarLinksToOp(x, "elementwise_add") &&  //
               !x->Var()->Proto()->persistable();     // is a parameter
      },
      "mul_out");

  PDNode* fc_mul = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "mul";  // basic
      },
      "fc_mul");

  PDNode* fc_bias = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                     // basic
               VarLinksToOp(x, "elementwise_add") &&  // link
               x->Var()->Proto()->persistable();      // is a parameter
      },
      "fc_bias");

  PDNode* elementwise_add = pattern->NewNode(
      [](Node* x) {
        return x && x->IsOp() && x->Op()->Type() == "elementwise_add";
      },
      "elementwise_add");

  PDNode* add_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                       // basic
               VarLinksFromOp(x, "elementwise_add") &&  // link
               !x->Var()->Proto()->persistable();       // is a parameter
      },
      "add_out");

  std::set<std::string> acts({"sigmoid", "tanh", "relu", "identity"});
  PDNode* act = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsOp() && acts.count(x->Op()->Type());

      },
      "act");

  PDNode* fc_out = pattern->NewNode(
      [](Node* x) {
        return x && x->IsVar() &&                  // basic
               !x->Var()->Proto()->persistable();  // is a parameter
      },
      "fc_out");

  PDLINK(fc_w, fc_mul);
  PDLINK(fc_x, fc_mul);
  PDLINK(fc_mul, mul_out);

  PDLINK(mul_out, elementwise_add);
  PDLINK(fc_bias, elementwise_add);
  PDLINK(elementwise_add, add_out);
  PDLINK(add_out, act);
  PDLINK(act, fc_out);

  return fc_out;
}

std::unique_ptr<ir::Graph> SeqConcatFcFusePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  GraphPatternDetector detector;
  auto* pattern = detector.mutable_pattern();
  auto* concat_out = BuildSeqExpandConcatPattern(pattern);
  BuildFCPattern(pattern, concat_out);

  LOG(INFO) << "\n" << pattern->DotString();

  if (!graph->Has(kGraphvizMarkedNodeAttr)) {
    graph->Set(kGraphvizMarkedNodeAttr, new GraphVizPass::marked_nodes_t);
  }
  auto& marked_nodes =
      graph->Get<GraphVizPass::marked_nodes_t>(kGraphvizMarkedNodeAttr);

#define GET_NODE(id, pattern)                              \
  PADDLE_ENFORCE(subgraph.count(pattern.RetriveNode(#id)), \
                 "pattern has no Node called %s", #id);    \
  auto* id = subgraph.at(pattern.RetriveNode(#id));        \
  PADDLE_ENFORCE_NOT_NULL(id, "subgraph has no node %s", #id);

  detector(graph.get(), [&](const GraphPatternDetector::subgraph_t& subgraph,
                            Graph* graph) {
    LOG(INFO) << "get one concat pattern";
    // fc
    GET_NODE(fc_w, detector.pattern());
    GET_NODE(fc_bias, detector.pattern());
    GET_NODE(act, detector.pattern());
    GET_NODE(fc_out, detector.pattern());

    // concat
    GET_NODE(concat_in0, detector.pattern());
    GET_NODE(sequence_expand0_in, detector.pattern());
    GET_NODE(sequence_expand1_in, detector.pattern());

    OpDesc op_desc;
    op_desc.SetType("fusion_seqexpand_concat_fc");
    op_desc.SetInput("X", {concat_in0->Name(), sequence_expand0_in->Name(),
                           sequence_expand1_in->Name()});
    op_desc.SetInput("FCWeight", {fc_w->Name()});
    op_desc.SetInput("FCBias", {fc_bias->Name()});
    op_desc.SetOutput("FCOut", {});
    op_desc.SetOutput("Out", {fc_out->Name()});
    op_desc.SetAttr("fc_activation", act->Op()->Type());

    auto* op_node = graph->CreateOpNode(&op_desc);
// Add links
#define NODE_LINKS(a, b)   \
  a->outputs.push_back(b); \
  b->inputs.push_back(a);
    NODE_LINKS(fc_w, op_node);
    NODE_LINKS(fc_bias, op_node);
    NODE_LINKS(concat_in0, op_node);
    NODE_LINKS(sequence_expand0_in, op_node);
    NODE_LINKS(sequence_expand1_in, op_node);
    NODE_LINKS(op_node, fc_out);

    GraphSafeRemoveNodes(graph, marked_nodes);
  });

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(seq_concat_fc_fuse_pass,
              paddle::framework::ir::SeqConcatFcFusePass);
