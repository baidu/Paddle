// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <isl/cpp.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/ast_gen_ius/tensor_group.h"
#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/function_base.h"
#include "paddle/cinn/lang/buffer.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {

namespace ast_gen_ius {
class TensorGroup;
}  // namespace ast_gen_ius

namespace ir {

class _Tensor_;

class Tensor : public ir::IrNodeRef {
 public:
  Tensor() = default;
  explicit Tensor(ir::IrNode* n) : IrNodeRef(n) {}
  Tensor(const std::string& name,
         Type dtype,
         const std::vector<Expr>& shape,
         const std::vector<Expr>& domain,
         FunctionRef fn,
         const std::vector<Var>& reduce_axis = {});

  Tensor(const std::string& name,
         Type dtype,
         const std::vector<Dim>& sym_shape,
         const std::vector<Dim>& sym_domain,
         FunctionRef fn,
         const std::vector<Var>& reduce_axis = {});

  //! Get number of dimensions.
  size_t ndims() const;

  /**
   * Take elements from the tensor.
   * This take one or multiple expressions as indices.
   *
   * usage:
   *
   * Tensor A;
   * A(i,j) get the [i][j] element.
   */
  // @{
  Expr operator()(const Expr& a) const {
    return operator()(std::vector<Expr>({a}));
  }
  template <typename... Args>
  inline typename std::enable_if<sizeof...(Args) >= 2, Expr>::type operator()(
      Args&&... args) const {
    return operator()({std::forward<Args>(args)...});
  }
  // @}

  /**
   * Take elements from the tensor.
   * @param indices  The indices.
   * @return The result expression representing a tensor read.
   */
  Expr operator()(const std::vector<Expr>& indices) const;

  friend bool operator<(const Tensor& a, const Tensor& b);

  _Tensor_* self() { return operator->(); }
  const _Tensor_* self() const { return operator->(); }

  inline const _Tensor_* operator->() const { return As<_Tensor_>(); }
  inline _Tensor_* operator->() { return As<_Tensor_>(); }

  //! Cast to an Expr.
  inline operator Expr() const { return Expr(get()); }
};

/**
 * \brief Generate the name of the reduce init tensor of \p tensor.
 * This is used for retrieving the corresponding reduction-init tensor from a
 * stage map by name.
 */
std::string GenReduceInitTensorNameOf(const std::string& tensor_name);

bool IsReduceInitTensorName(const std::string& tensor_name);

bool IsSplitTransformTensorName(const std::string& tensor_name);

std::string GetOriginalReduceTensorName(const std::string& tensor_name);

class ComputeOp;
class PlaceholderOp;
struct ReadCacheRelation;
struct WriteCacheRelation;

/**
 * _Tensor_ holds the content of a Tensor.
 *
 * NOTE(All) Some rules:
 *
 * 1. a _Tensor_ is a node in SSA, so every tensor's name should be unique,
 * 2. never try to change a tensor's name, that will cause chaos.
 */
class _Tensor_ : public ExprNode<_Tensor_> {
 public:
  //! Symbolic Shape of this tensor(buffer).
  std::vector<Dim> sym_shape;
  //! Shape of this tensor(buffer).
  std::vector<Expr> shape;
  //! The symbolic domain of each axis(without reduce_axis)
  std::vector<Dim> sym_domain;
  //! The domain of each axis(without reduce_axis)
  // TODO(Superjomn) support ISL domain.
  std::vector<Expr> domain;

  std::vector<Var> reduce_axis;
  //! The operation that generates Tensor.
  FunctionRef operation;
  //! Name of this tensor.
  std::string name;
  //! The bound buffer, for each tensor if it is not inline.
  Buffer buffer;
  //! Normal axis.
  mutable std::vector<Var> axis_;

  std::vector<Expr> new_indices{};
  std::vector<Expr> domain_with_reduce_axis() const;
  const std::vector<Expr>& domain_without_reduce_axis() const { return domain; }

  //! Generate a tensor from a function.
  static Tensor Make(const std::string& name,
                     Type dtype,
                     const std::vector<Expr>& shape,
                     const std::vector<Expr>& domain,
                     FunctionRef fn,
                     const std::vector<Var>& reduce_axis = {});

  // Manual tensor construction, no FunctionRef information
  static Tensor Make(const std::string& name,
                     Type dtype,
                     const std::vector<Expr>& shape,
                     const std::vector<Expr>& domain,
                     const std::vector<Var>& reduce_axis = {});

  //! (Symbolic Shape) Generate a tensor from a function.
  static Tensor Make(const std::string& name,
                     Type dtype,
                     const std::vector<Dim>& sym_shape,
                     const std::vector<Dim>& sym_domain,
                     FunctionRef fn,
                     const std::vector<Var>& reduce_axis = {});

  // (Symbolic Shape) Manual tensor construction, no FunctionRef information
  static Tensor Make(const std::string& name,
                     Type dtype,
                     const std::vector<Dim>& sym_shape,
                     const std::vector<Dim>& sym_domain,
                     const std::vector<Var>& reduce_axis = {});

  void Verify() const override;

  //! Tell whether this tensor represents a tuple (consists of one or multiple
  //! tensors as output of a extern Call).
  bool is_tuple() const;
  bool is_tuple_get() const;

  Tensor TupleGet(int offset) const;

  /**
   * Get the names of the dependency(read or write) tensors.
   * e.g. A[i] = C[i]*2 + D[i], A's dependency tensors are {C,D}
   */
  std::set<std::string> GetDependTensorNames() const;

  /**
   * \brief Tell whether this tensor's computation relays on a specific
   * statement.
   * @param statement The name of a statement(equivalent to the id of tensor).
   * @return A boolean.
   */
  bool IsDependOnStatement(absl::string_view statement);

  /**
   * Get the names of the tensors those this tensor depends on.
   */
  std::set<std::string> DependingTensorNames();

  /**
   * Get a new tensor with the \p shape, but the underlying buffer shared.
   * NOTE the tensor to Reshape should not be an inlined computation.
   */
  ir::Tensor Reshape(const std::vector<Expr>& shape) const;

  /**
   * Get a new tensor with the \p shape with a newly allocated buffer.
   * NOTE the tensor to Reshape should not be an inlined computation.
   */
  ir::Tensor ReshapeCopied(const std::vector<Expr>& shape) const;

  /**
   * Tell whether this tensor has same shape with \p other.
   */
  bool HasSameShapeWith(const Tensor& other) const;

  //! Operation related.
  // @{
  bool is_compute_node() const;
  bool is_placeholder_node() const;
  bool is_call_node() const;
  bool is_extern_call_node() const;
  bool is_preceding_view_node() const;
  bool is_buffer_shared_node() const;
  const char* operation_type() const;
  ComputeOp* get_compute_op() const;
  PlaceholderOp* get_placeholder_op() const;
  // @}

  //! The expression generate this tensor, will be empty if it is a PlaceHolder.
  Expr body() const;
  Expr* mutable_body();
  //! Get the expression with `store(tensor)` inserted into the body.
  Expr tensor_store_expanded_body();

  Expr inline_expanded(const std::vector<Expr>& indices);

  //! Tell whether contain a reduce axis.
  bool contains_reduce_axis() const { return !reduce_axis.empty(); }
  bool is_reduce_tensor() const { return contains_reduce_axis(); }
  bool is_reduce_sum() const;
  bool is_reduce_mul() const;
  //! Get the initial value of a reduce tensor.
  Expr GetReduceInitVal() const;

  std::vector<Expr*> expr_fields() override;
  std::vector<const Expr*> expr_fields() const override;

  /**
   * The normal axis without reducing ones.
   */
  const std::vector<Var>& axis() const;

  /**
   * The axis with the reduce ones.
   */
  std::vector<Var> axis_with_reduce() const;

  /**
   * Get the tensors those depend on the same buffer belong to this tensor.
   */
  const std::set<std::string>& buffer_depended_tensor_names() const {
    return buffer_depended_tensor_names_;
  }

  static const IrNodeTy _node_type_ = IrNodeTy::_Tensor_;

  _Tensor_() : ExprNode<_Tensor_>(Float(32)) {}

  bool has_expression() const;

  ~_Tensor_();

  /**
   * Tell if this tensor uses other tensors in the body.
   */
  bool Uses(const ir::Tensor& other) const;

  //! Bind to a buffer, will persist data to the buffer in runtime.
  void Bind(lang::Buffer& buffer);  // NOLINT
  void Bind(const Buffer& buffer);
  void UnBind(lang::Buffer& buffer);  // NOLINT

  //! Create a buffer belong to this tensor.
  void WithBuffer(const Type& type = Void());
  void WithBuffer(const std::string& memory_type,
                  const std::string& buffer_name = "",
                  const Type& type = Void());

  const std::optional<std::vector<Expr>>& value() const { return value_; }

  void set_value(const std::vector<Expr>& value) { value_ = value; }

 private:
  //! Initialize the axis field after the shape field is assigned.
  void InitAxis() const;

  isl::set GenerateIslDomain() const;

  //! The names of the tensors depend the same buffer and should schedule before
  //! this.
  std::set<std::string> buffer_depended_tensor_names_;

  // The flatten compute value of tensor, such as Tensor[[1, 2], [3, 4]] ->
  // Tensor[1, 2, 3, 4]
  std::optional<std::vector<Expr>> value_;

  friend Shared<poly::Stage> CreateStage(Tensor tensor);
};

Shared<poly::Stage> CreateStage(Tensor tensor);

class _Operation_;
class Operation : public FunctionRef {
 public:
  Operation() = default;
  explicit Operation(IrNode* n) : FunctionRef(n) {}

  inline const _Operation_* operator->() const {
    return reinterpret_cast<_Operation_*>(get());
  }
  inline _Operation_* operator->() {
    return reinterpret_cast<_Operation_*>(get());
  }

  //! Get the i-th output of the operation.
  // Tensor output(size_t i) const;

  std::string name;
};

class _Operation_ : public ir::FunctionBase {
 public:
  //! Optional name of the operation.
  std::string name;
  //! Optional tag of the operation.
  std::string tag;
  //! Additional attributes of the operation.
  std::map<std::string, IrNodeRef> attrs;

  const std::string& func_name() const final { return name; }

  void Verify() const override {}

  //! The function type.
  virtual const char* func_type() const = 0;
};

}  // namespace ir
}  // namespace cinn

namespace std {

template <>
struct hash<cinn::ir::Tensor> {
  inline size_t operator()(const cinn::ir::Tensor& x) {
    // We treat the tensor's name as the unique identifier.
    return std::hash<std::string>()(x->name);
  }
};

}  // namespace std
