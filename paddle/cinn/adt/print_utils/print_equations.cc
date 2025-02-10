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

#include "paddle/cinn/adt/print_utils/print_equations.h"

#include <sstream>
#include <string>

#include "paddle/cinn/adt/equation_function.h"
#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/print_utils/print_dim_expr.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::adt {

namespace {

std::string OpImpl(const ::pir::Operation* op) {
  return hlir::framework::pir::CompatibleInfo::OpName(*op);
}

std::string OpImpl(const tReduceInit<const ::pir::Operation*>& op) {
  return OpImpl(op.value()) + "_init";
}

std::string OpImpl(const tReduceAcc<const ::pir::Operation*>& op) {
  return OpImpl(op.value()) + "_acc";
}

}  // namespace

std::string ToTxtString(const Iterator& iterator) {
  std::size_t iterator_unique_id = iterator.value().unique_id();
  return "i_" + std::to_string(iterator_unique_id);
}

std::string ToTxtString(const Index& index) {
  std::size_t index_unique_id = index.value().unique_id();
  return "idx_" + std::to_string(index_unique_id);
}

std::string ToTxtString(const FakeOpPlaceHolder& op) {
  std::size_t op_unique_id = op.value().unique_id();
  return "op_" + std::to_string(op_unique_id);
}

std::string ToTxtString(const List<Index>& indexes) {
  std::string ret;
  ret += "[";

  for (std::size_t idx = 0; idx < indexes->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(indexes.Get(idx));
  }

  ret += "]";
  return ret;
}

std::string ToTxtString(const List<std::optional<Index>>& indexes) {
  std::string ret;
  ret += "[";

  for (std::size_t idx = 0; idx < indexes->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    if (indexes->at(idx).has_value()) {
      ret += ToTxtString(indexes.Get(idx).value());
    }
  }

  ret += "]";
  return ret;
}

std::string ToTxtString(const List<Iterator>& iterators) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < iterators->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(iterators.Get(idx));
  }
  ret += "]";
  return ret;
}

std::string ToTxtString(const tInMsg<List<Index>>& in_msg_indexes) {
  std::string ret;
  const List<Index>& index_list = in_msg_indexes.value();
  ret += ToTxtString(index_list);
  return ret;
}

std::string ToTxtString(const tOutMsg<List<Index>>& out_msg_indexes) {
  std::string ret;
  const List<Index>& index_list = out_msg_indexes.value();
  ret += ToTxtString(index_list);
  return ret;
}

std::string ToTxtString(const std::vector<Index>& indexes) {
  std::string ret;
  ret += "vector(";
  for (std::size_t idx = 0; idx < indexes.size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(indexes.at(idx));
  }

  ret += ")";
  return ret;
}

std::string ToTxtString(const List<OpStmt>& op_stmts,
                        const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
  std::string ret;
  std::size_t count = 0;
  for (const auto& op_stmt : *op_stmts) {
    if (count++ != 0) {
      ret += "\n";
    }
    const auto& [op, _0, _1] = op_stmt.tuple();
    ret += std::visit([&](const auto& op_impl) { return OpImpl(op_impl); },
                      op.variant());
    ret += ": \n";
    const auto& ctx = EquationCtx4OpStmt(op_stmt);
    ret += ToTxtString(ctx->equations(), "\n");
  }

  return ret;
}

namespace {

struct ToTxtStringStruct {
  std::string operator()(
      const Identity<tOut<Iterator>, tIn<Iterator>>& id) const {
    std::string ret;
    const auto& [out_iter, in_iter] = id.tuple();
    ret += ToTxtString(out_iter.value()) + " = " + ToTxtString(in_iter.value());
    return ret;
  }

  std::string operator()(const Identity<tOut<Index>, tIn<Index>>& id) const {
    std::string ret;
    const auto& [out_index, in_index] = id.tuple();
    ret +=
        ToTxtString(out_index.value()) + " = " + ToTxtString(in_index.value());
    return ret;
  }

  std::string operator()(
      const IndexDot<List<DimExpr>, tOut<Index>, tIn<List<Iterator>>>& dot)
      const {
    std::string ret;
    const auto& [dim_list, out_index_tag, in_iterator_list_tag] = dot.tuple();
    const Index& out_index = out_index_tag.value();
    const List<Iterator>& in_iterator_list = in_iterator_list_tag.value();
    ret += ToTxtString(out_index) + " = IndexDot(" +
           ToTxtString(in_iterator_list) + ")";
    return ret;
  }

  std::string operator()(
      const GetBroadcastedIterator<DimExpr, tOut<Iterator>, tIn<Iterator>>&
          broadcast) const {
    std::string ret;
    const auto& [dim, out_iterator, in_iterator] = broadcast.tuple();
    ret += ToTxtString(out_iterator.value()) + " = GetBroadcastedIterator(" +
           ToTxtString(in_iterator.value()) + ", " + ToTxtString(dim) + ")";
    return ret;
  }

  std::string operator()(
      const IndexUnDot<List<DimExpr>, tOut<List<Iterator>>, tIn<Index>>& undot)
      const {
    std::string ret;
    const auto& [dim_list, out_iterator_list_tag, in_index_tag] = undot.tuple();
    const List<Iterator>& out_iterator_list = out_iterator_list_tag.value();
    const Index& in_index = in_index_tag.value();
    ret += ToTxtString(out_iterator_list) + " = IndexUnDot(" +
           ToTxtString(in_index) + ")";
    return ret;
  }

  std::string operator()(
      const InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                         tOut<OpArgIndexes<std::optional<Index>>>,
                         tIn<OpArgIndexes<Index>>>& in_msg2out_msg) const {
    std::string ret;
    const auto& [out_op, out_indices, in_indices] = in_msg2out_msg.tuple();
    const FakeOpPlaceHolder& op = out_op.value();
    const auto& out_index_tuple = out_indices.value();
    const auto& in_index_tuple = in_indices.value();
    const auto& [out_msg_list_in, out_msg_list_out] = out_index_tuple.tuple();
    const auto& [in_msg_list_in, in_msg_list_out] = in_index_tuple.tuple();
    ret += ToTxtString(op) + ", ";
    ret += "(" + ToTxtString(out_msg_list_in.value()) + ", " +
           ToTxtString(out_msg_list_out.value()) + ") = InMsg2OutMsg(";
    ret += ToTxtString(in_msg_list_in.value()) + ", " +
           ToTxtString(in_msg_list_out.value()) + ")";
    return ret;
  }

  std::string operator()(
      const ConstantFunction<tOut<Iterator>, tIn<Index>>& constant) const {
    std::string ret{};
    const auto& [out_iterator, in_index, c] = constant.tuple();
    ret += ToTxtString(out_iterator.value()) + " = ConstantFunction(" +
           ToTxtString(in_index.value()) + ", " + ToTxtString(c) + ")";
    return ret;
  }
};

}  // namespace

std::string ToTxtString(const Equation& equation) {
  return std::visit(ToTxtStringStruct{}, equation.variant());
}

std::string ToTxtString(const Equations& equations,
                        const std::string& separator) {
  std::stringstream ret;
  std::size_t count = 0;

  for (const auto& equation : *equations) {
    if (count++ > 0) {
      ret << separator;
    }
    ret << &equation << ": ";
    ret << ToTxtString(equation);
  }
  return ret.str();
}

std::string ToTxtStringImpl(const Iterator& iterator) {
  return ToTxtString(iterator);
}

std::string ToTxtStringImpl(const Index& index) { return ToTxtString(index); }

std::string ToTxtStringImpl(const FakeOpPlaceHolder& op) {
  return ToTxtString(op);
}

std::string ToTxtString(const Variable& variable) {
  return std::visit([&](const auto& impl) { return ToTxtStringImpl(impl); },
                    variable.variant());
}

std::string ToDotString(
    const Equations& equations,
    const std::optional<Variable>& start,
    const std::unordered_set<Variable>& visited_variables,
    const std::unordered_set<const void*>& visited_functions) {
  std::stringstream ss;

  const auto& GetFunctionUid = [&](const Equation& equation) {
    std::stringstream ss;
    ss << "f" << GetFunctionDataPtr(equation);
    return ss.str();
  };

  ss << "digraph {\n";

  const auto& FillFunctionColor = [&](const Equation& function) -> std::string {
    if (visited_functions.count(GetFunctionDataPtr(function))) {
      return ", style=filled, color=green";
    } else {
      return "";
    }
  };
  std::unordered_set<Variable> variables{};
  for (const auto& equation : *equations) {
    const auto& [in_variables, out_variables] =
        GraphTrait<Variable, Function>::CollectInputAndOutputVariables(
            equation);
    ss << GetFunctionUid(equation) << "["
       << "label=\"" << GetFunctionTypeName(equation) << "<"
       << GetFunctionDataPtr(equation) << ">"
       << "\"" << FillFunctionColor(equation) << "]\n";
    for (const auto& in_variable : in_variables) {
      ss << ToTxtString(in_variable) << " -> " << GetFunctionUid(equation)
         << ";\n";
      variables.insert(in_variable);
    }
    for (const auto& out_variable : out_variables) {
      ss << GetFunctionUid(equation) << " -> " << ToTxtString(out_variable)
         << ";\n";
      variables.insert(out_variable);
    }
  }
  const auto& GetColor = [&](const Variable& variable) {
    if (start.has_value() && start.value() == variable) {
      return "red";
    } else {
      return "green";
    }
  };

  for (const auto& variable : variables) {
    if (visited_variables.count(variable)) {
      ss << ToTxtString(variable)
         << "[style=filled, color=" << GetColor(variable) << "];\n";
    }
  }
  ss << "}\n";
  return ss.str();
}

}  // namespace cinn::adt
