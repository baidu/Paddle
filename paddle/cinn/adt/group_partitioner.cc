// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/group_partitioner.h"
#include <algorithm>

namespace cinn::adt::equation {

std::vector<Variable> InitCandidateIndex(const Graph& graph) {
  std::unordered_set<Variable> variables = graph.GetVariables();
  std::vector<Variable> candidate_index;
  for (auto iter = variables.begin(); iter != variables.end(); ++iter) {
    *iter >> match{[&](const Index& index) {
      candidate_index.emplace_back(Variable(index));
    }};
  }
  return candidate_index;
}

Variable PickAnchorTensor(const std::vector<Variable>& candidate_index) {
  // Heuristic optimization will be added later
  // such as choosing the one with the biggest rank number as the anchor tensor
  // first
  return *(candidate_index.begin());
}

FakeOpPlaceHolders GenerateIGroup(const Graph& graph,
                                  const Variable& anchor_tensor) {
  FakeOpPlaceHolders igroup;
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();
  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const FakeOpPlaceHolder& fakeOpPlaceholder) {
          igroup->emplace_back(fakeOpPlaceholder);
        }};
      };
  walker(anchor_tensor, variableVisitor);
  return igroup;
}

bool IsContain(const FakeOpPlaceHolders& pre_igroup,
               const FakeOpPlaceHolders& igroup) {
  for (const auto& pre_op : *pre_igroup) {
    auto iter = std::find(igroup->begin(), igroup->end(), pre_op);
    if (iter == igroup->end()) {
      return false;
    }
  }
  return true;
}

void UpdateIGroupMap(
    const FakeOpPlaceHolders& igroup,
    const Variable& anchor_tensor,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup) {
  for (const auto& [pre_anchor_tensor, pre_igroup] : *index2IGroup) {
    if (pre_igroup->size() >= igroup->size()) {
      continue;
    }
    if (IsContain(pre_igroup, igroup)) {
      index2IGroup->erase(pre_anchor_tensor);
    }
  }
  index2IGroup->emplace(anchor_tensor, igroup);
}

void UpdateCandidateSet(const Graph& graph,
                        const FakeOpPlaceHolders& igroup,
                        const Variable& anchor_tensor,
                        std::vector<Variable>* candidate_index) {
  EquationGraphTopoWalker<const Variable, const Function*> walker =
      graph.GetWalker();

  std::function<void(Variable)> variableVisitor =
      [&](const Variable& variable) {
        variable >> match{[&](const Index& index) {
          auto iter = std::find(candidate_index->begin(),
                                candidate_index->end(),
                                Variable(index));
          if (iter != candidate_index->end()) {
            candidate_index->erase(iter);
          }
        }};
      };
  walker(anchor_tensor, variableVisitor);
}

void TopoSort4IGroup(
    const cinn::hlir::framework::Graph::Group& group,
    std::unordered_map<Variable, FakeOpPlaceHolders>* index2IGroup) {
  std::vector<cinn::hlir::framework::Node*> sorted_ops = group.nodes;
  for (auto& [index, igroup] : *index2IGroup) {
    FakeOpPlaceHolders tmp_igroup;
    for (const auto& sorted_op : sorted_ops) {
      auto iter = std::find(igroup->begin(), igroup->end(), sorted_op);
      if (iter != igroup->end()) {
        tmp_igroup->emplace_back(sorted_op);
      }
    }
    igroup = std::move(tmp_igroup);
  }
}

std::unordered_map<Variable, FakeOpPlaceHolders> PartitionGraph(
    const cinn::hlir::framework::Graph::Group& group, const Graph& graph) {
  std::vector<Variable> candidate_index = InitCandidateIndex(graph);
  std::unordered_map<Variable, FakeOpPlaceHolders> index2IGroup;
  while (!candidate_index.empty()) {
    Variable anchor_tensor = PickAnchorTensor(candidate_index);
    FakeOpPlaceHolders igroup = GenerateIGroup(graph, anchor_tensor);
    UpdateIGroupMap(igroup, anchor_tensor, &index2IGroup);
    UpdateCandidateSet(graph, igroup, anchor_tensor, &candidate_index);
  }

  TopoSort4IGroup(group, &index2IGroup);
  return index2IGroup;
}

}  // namespace cinn::adt::equation
