// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/common/object.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace ir {
namespace stmt {

class StmtRef;
class BlockRef;

class Let;
class Store;
class Alloc;
class Free;
class IfThenElse;
class For;
class Evaluate;
class Schedule;

class _Block_;

/*
 * \brief Define statement type
 * \param Type The statement type
 * \param TypeEnumName The statement type enum
 */
#define CINN_DEFINE_STMT_TYPE(TypeEnumName)                     \
  StmtNodeTy stmt_type() const override { return _stmt_type_; } \
  static const StmtNodeTy _stmt_type_ = StmtNodeTy::TypeEnumName;

/*
 * \brief Define statement reference methods
 * \param RefTypeName The type name of statement ref
 * \param NodeTypeName The type name of statement node
 */
#define CINN_DEFINE_STMT_REF_METHODS(RefTypeName, NodeTypeName)      \
  CINN_DEFINE_OBJECT_REF_METHODS(RefTypeName, StmtRef, NodeTypeName) \
  static const StmtNodeTy _stmt_type_ = NodeTypeName::_stmt_type_;

/**
 * The base of all the statement nodes.
 */
class _Stmt_ : public cinn::common::Object {
 public:
  _Stmt_() = default;
  explicit _Stmt_(Type t) : type_(t) {}
  virtual ~_Stmt_() = default;

  virtual StmtNodeTy stmt_type() const { return StmtNodeTy::kUnk; }

  const BlockRef GetParentBlockRef() const;
  const std::vector<BlockRef>& block_fields() const { return blocks_; }
  virtual void set_block_fields(const std::vector<BlockRef>& blocks);
  virtual Type type() const { return type_; }
  void set_type(Type type) { type_ = type; }

  virtual void Verify() const { CINN_NOT_IMPLEMENTED }

 protected:
  _Block_* parent_;
  std::vector<BlockRef> blocks_;

 private:
  friend class _Block_;
  void set_parent(_Block_* parent) { parent_ = parent; }
  // Forbidden cast these methods of Object.
  // TODO(Hongqing-work): restruct the Object class.
  static constexpr char* __type_info__ = "StmtNode";
  const char* type_info() const override { return __type_info__; }
  using Object::as;
  using Object::is_type;
  using Object::safe_as;

  Type type_;
};

class StmtRef : public cinn::common::Shared<Object> {
 public:
  CINN_DEFINE_OBJECT_REF_METHODS(StmtRef, cinn::common::Shared<Object>, _Stmt_)

  template <typename T>
  T as() {
    static_assert(std::is_base_of<StmtRef, T>());
    PADDLE_ENFORCE_NOT_NULL(
        get(),
        ::common::errors::InvalidArgument(
            "StmtRef holds null. "
            "The get() method should return a non-null value."));
    PADDLE_ENFORCE_EQ(
        (*this)->stmt_type(),
        T::_stmt_type_,
        ::common::errors::InvalidArgument("Type mismatch when cast StmtRef, "
                                          "expected type %d, but got type %d.",
                                          (*this)->stmt_type(),
                                          T::_stmt_type_));
    T& res = static_cast<T&>(*this);
    return res;
  }

  template <typename T>
  const T& as() const {
    static_assert(std::is_base_of<StmtRef, T>());
    PADDLE_ENFORCE_NOT_NULL(
        get(),
        ::common::errors::InvalidArgument(
            "StmtRef holds null. "
            "The get() method should return a non-null value."));
    PADDLE_ENFORCE_EQ(
        (*this)->stmt_type(),
        T::_stmt_type_,
        ::common::errors::InvalidArgument("Type mismatch when cast StmtRef, "
                                          "expected type %d, but got type %d.",
                                          (*this)->stmt_type(),
                                          T::_stmt_type_));
    const T& res = static_cast<const T&>(*this);
    return res;
  }

  template <typename T>
  bool isa() const {
    if ((*this)->stmt_type() == T::_stmt_type_) {
      return true;
    }
    return false;
  }
};

class _Block_ : public cinn::common::Object {
 public:
  const StmtRef GetParentStmtRef() const { return StmtRef{parent_}; }
  const std::vector<StmtRef>& stmts() const { return stmts_; }
  void set_stmts(const std::vector<StmtRef>& stmts) {
    stmts_ = stmts;
    for (auto& stmt : stmts_) stmt->set_parent(this);
  }

  static BlockRef Make(const std::vector<StmtRef>& stmts);

 private:
  _Block_() = default;
  std::vector<StmtRef> stmts_;

  _Stmt_* parent_;
  friend class _Stmt_;
  void set_parent(_Stmt_* parent) { parent_ = parent; }

  // Forbidden cast these methods of Object.
  static constexpr char* __type_info__ = "BlockNode";
  const char* type_info() const override { return __type_info__; }
  using Object::as;
  using Object::is_type;
  using Object::safe_as;
};

class BlockRef : public cinn::common::Shared<Object> {
 public:
  BlockRef() = default;
  explicit BlockRef(const std::vector<StmtRef>& stmts)
      : BlockRef(_Block_::Make(stmts)) {}
  explicit BlockRef(_Block_* n) : cinn::common::Shared<Object>(n) {}
  CINN_DEFINE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_REF(BlockRef);
  const _Block_* operator->() const {
    return static_cast<const _Block_*>(this->p_);
  }
  _Block_* operator->() { return static_cast<_Block_*>(this->p_); }
};

class _Let_ : public _Stmt_ {
 public:
  static Let Make(Expr symbol, Expr body);
  const Expr& symbol() const { return symbol_; }
  const Expr& body() const { return body_; }
  void set_symbol(Expr symbol) { symbol_ = symbol; }
  void set_body(Expr body) { body_ = body; }

  Type type() const override;

  void Verify() const override;

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_Let_)
  CINN_DEFINE_STMT_TYPE(Let)

 private:
  _Let_() = default;
  Expr symbol_;
  Expr body_;
};

class Let : public StmtRef {
 public:
  Let(Expr symbol, Expr body) : Let(_Let_::Make(symbol, body)) {}
  CINN_DEFINE_STMT_REF_METHODS(Let, _Let_)
};

/**
 * Store a `value` to the buffer at a given `index`.
 */
class _Store_ : public _Stmt_ {
 public:
  static Store Make(Expr tensor, Expr value, const std::vector<Expr>& indices);
  const Expr& tensor() const { return addr_mnger_.tensor; }
  const Expr& value() const { return value_; }
  const std::vector<Expr>& indices() const { return indices_; }
  void set_tensor(Expr tensor) { addr_mnger_.tensor = tensor; }
  void set_value(Expr value) { value_ = value; }
  void set_indices(const std::vector<Expr>& indices) { indices_ = indices; }

  void Verify() const override;

  const std::string& name() const;

  void replace(Expr old_op, Expr new_op);

  bool is_addr_tensor() const { return addr_mnger_.is_addr_tensor(); }
  bool is_addr_scalar() const { return addr_mnger_.is_addr_scalar(); }

  Type type() const override;

  Expr index() const;

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_Store_)
  CINN_DEFINE_STMT_TYPE(Store)

 private:
  _Store_() = default;
  Expr value_;
  std::vector<Expr> indices_;
  LoadStoreAddrMnger addr_mnger_;
};

class Store : public StmtRef {
 public:
  Store(Expr tensor, Expr value, const std::vector<Expr>& indices)
      : Store(_Store_::Make(tensor, value, indices)) {}
  CINN_DEFINE_STMT_REF_METHODS(Store, _Store_)
};

/**
 * Allocate a buffer with the given type and size. The buffer lives for at most
 * the duration of the body statement, within which it is freed.
 */
class _Alloc_ : public _Stmt_ {
 public:
  static Alloc Make(Expr dest,
                    Type type,
                    const std::vector<Expr>& extents,
                    Expr condition,
                    Expr body);
  const Expr& destination() const { return destination_; }
  const std::vector<Expr>& extents() const { return extents_; }
  const Expr& condition() const { return condition_; }
  const Expr& body() const { return body_; }
  void set_destination(Expr dest) { destination_ = dest; }
  void set_extents(const std::vector<Expr>& extents) { extents_ = extents; }
  void set_condition(Expr condition) { condition_ = condition; }
  void set_body(Expr body) { body_ = body; }

  void Verify() const override;

  int32_t ConstantAllocationSize() const;
  static int32_t ConstantAllocationSize(const std::vector<Expr>& extents);

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_Alloc_)
  CINN_DEFINE_STMT_TYPE(Alloc)

 private:
  _Alloc_() : _Stmt_(Type()) {}
  //! The destination of the allocation, this might be a buffer or a variable.
  Expr destination_;
  //! Dimensions of this buffer (as a multi-dimensional array).
  std::vector<Expr> extents_;
  // NOTE the condition might be undefined, that means always true.
  Expr condition_;
  // NOTE the body might be undefined, that means no specific logic other than
  // default.
  Expr body_;
};

class Alloc : public StmtRef {
 public:
  Alloc(Expr dest,
        Type type,
        const std::vector<Expr>& extents,
        Expr condition,
        Expr body)
      : Alloc(_Alloc_::Make(dest, type, extents, condition, body)) {}
  CINN_DEFINE_STMT_REF_METHODS(Alloc, _Alloc_)
};

/**
 * Free the resources associated with the given buffer.
 */
class _Free_ : public _Stmt_ {
 public:
  static Free Make(Expr dest);
  const Expr& destination() const { return destination_; }
  void set_destination(Expr dest) { destination_ = dest; }

  void Verify() const override;

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_Free_)
  CINN_DEFINE_STMT_TYPE(Free)

 private:
  _Free_() : _Stmt_(Type()) {}
  Expr destination_;
};

class Free : public StmtRef {
 public:
  explicit Free(Expr dest) : Free(_Free_::Make(dest)) {}
  CINN_DEFINE_STMT_REF_METHODS(Free, _Free_)
};

class _IfThenElse_ : public _Stmt_ {
 public:
  static IfThenElse Make(Expr condition,
                         BlockRef true_case,
                         BlockRef false_case);
  const Expr& condition() const { return condition_; }
  const BlockRef& true_case() const { return blocks_[0]; }
  const BlockRef& false_case() const { return blocks_[1]; }
  void set_condition(Expr condition) { condition_ = condition; }
  void set_true_case(BlockRef true_case) {
    set_block_fields({true_case, blocks_[1]});
  }
  void set_false_case(BlockRef false_case) {
    set_block_fields({blocks_[0], false_case});
  }

  void Verify() const override;

  virtual void set_block_fields(const std::vector<BlockRef>& blocks) {
    PADDLE_ENFORCE_EQ(
        blocks.size() == 2U,
        true,
        ::common::errors::InvalidArgument(
            "blocks size for IfThenElse must be 2, but got %d", blocks.size()));
    _Stmt_::set_block_fields(blocks);
  }

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_IfThenElse_)
  CINN_DEFINE_STMT_TYPE(IfThenElse)

 private:
  _IfThenElse_() : _Stmt_(Type()) { blocks_.resize(2); }
  Expr condition_;
};

class IfThenElse : public StmtRef {
 public:
  IfThenElse(Expr condition,
             BlockRef true_case,
             BlockRef false_case = BlockRef(std::vector<StmtRef>()))
      : IfThenElse(_IfThenElse_::Make(condition, true_case, false_case)) {}
  CINN_DEFINE_STMT_REF_METHODS(IfThenElse, _IfThenElse_)
};

class _For_ : public _Stmt_, public ForBase {
 public:
  static For Make(Var loop_var,
                  Expr min,
                  Expr extent,
                  ForType for_type,
                  DeviceAPI device_api,
                  BlockRef body,
                  VectorizeInfo vector_info = VectorizeInfo(),
                  BindInfo bind_info = BindInfo());
  const Var& loop_var() const { return loop_var_; }
  const Expr& min() const { return min_; }
  const Expr& extent() const { return extent_; }
  const BlockRef& body() const { return blocks_[0]; }
  const DeviceAPI& device_api() const { return device_api_; }
  const LLVMForLoopMeta& metadata() const { return metadata_; }
  void set_loop_var(Var loop_var) { loop_var_ = loop_var; }
  void set_min(Expr min) { min_ = min; }
  void set_extent(Expr extent) { extent_ = extent; }
  void set_body(BlockRef body) { set_block_fields({body}); }
  void set_device_api(DeviceAPI device_api) { device_api_ = device_api; }
  void set_metadata(LLVMForLoopMeta metadata) { metadata_ = metadata; }

  void Verify() const override;

  virtual void set_block_fields(const std::vector<BlockRef>& blocks) {
    PADDLE_ENFORCE_EQ(
        blocks.size() == 1U,
        true,
        ::common::errors::InvalidArgument(
            "blocks size for For must be 1, but got %d", blocks.size()));
    _Stmt_::set_block_fields(blocks);
  }

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_For_)
  CINN_DEFINE_STMT_TYPE(For)

 private:
  _For_() { blocks_.resize(1); }
  //! The loop variable.
  Var loop_var_;
  //! The minimum value of the iteration.
  Expr min_;
  //! The extent of the iteration.
  Expr extent_;
  DeviceAPI device_api_;
  LLVMForLoopMeta metadata_;
};

class For : public StmtRef {
 public:
  For(Var loop_var,
      Expr min,
      Expr extent,
      ForType for_type,
      DeviceAPI device_api,
      BlockRef body,
      VectorizeInfo vector_info = VectorizeInfo(),
      BindInfo bind_info = BindInfo())
      : For(_For_::Make(loop_var,
                        min,
                        extent,
                        for_type,
                        device_api,
                        body,
                        vector_info,
                        bind_info)) {}
  CINN_DEFINE_STMT_REF_METHODS(For, _For_)
};

class _Evaluate_ : public _Stmt_ {
 public:
  static Evaluate Make(Expr value);
  const Expr& value() const { return value_; }
  void set_value(Expr value) { value_ = value; }

  void Verify() const override;

  CINN_DEFINE_STMT_TYPE(Evaluate)
  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_Evaluate_)

 private:
  _Evaluate_() = default;
  Expr value_;
};

class Evaluate : public StmtRef {
 public:
  explicit Evaluate(Expr value) : Evaluate(_Evaluate_::Make(value)) {}
  CINN_DEFINE_STMT_REF_METHODS(Evaluate, _Evaluate_)
};

/**
Schedule contains information to schedule a tensor's computation
*/
class _Schedule_ : public _Stmt_ {
 public:
  static Schedule Make(const std::vector<Var>& iter_vars,
                       const std::vector<Expr>& iter_values,
                       const std::vector<Expr>& read_buffers,
                       const std::vector<Expr>& write_buffers,
                       const std::string& name,
                       const BlockRef& body,
                       const std::map<std::string, attr_t>& attrs = {},
                       const ReduceMethod& reduce_method = {
                           NoneReduceMethod()});
  const std::vector<Var>& iter_vars() const { return iter_vars_; }
  const std::vector<Expr>& iter_values() const { return iter_values_; }
  const std::vector<Expr>& read_buffers() const { return read_buffers_; }
  const std::vector<Expr>& write_buffers() const { return write_buffers_; }
  const BlockRef& body() const { return blocks_[0]; }
  const std::map<std::string, attr_t>& attrs() const { return attrs_; }
  const std::string& name() const { return name_; }
  const ReduceMethod& reduce_method() const { return reduce_method_; }
  void set_iter_vars(const std::vector<Var>& iter_vars) {
    iter_vars_ = iter_vars;
  }
  void set_iter_values(const std::vector<Expr>& iter_values) {
    iter_values_ = iter_values;
  }
  void set_read_buffers(const std::vector<Expr>& read_buffers) {
    read_buffers_ = read_buffers;
  }
  void set_write_buffers(const std::vector<Expr>& write_buffers) {
    write_buffers_ = write_buffers;
  }
  void set_body(const BlockRef& body) { set_block_fields({body}); }
  void set_attrs(const std::map<std::string, attr_t>& attrs) { attrs_ = attrs; }
  void set_name(const std::string& name) { name_ = name; }
  void set_reduce_method(const ReduceMethod& reduce_method) {
    reduce_method_ = reduce_method;
  }

  void Verify() const override;

  virtual void set_block_fields(const std::vector<BlockRef>& blocks) {
    PADDLE_ENFORCE_EQ(
        blocks.size() == 1U,
        true,
        ::common::errors::InvalidArgument(
            "blocks size for For must be 1, but got %d", blocks.size()));
    _Stmt_::set_block_fields(blocks);
  }

  CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(_Schedule_)
  CINN_DEFINE_STMT_TYPE(Schedule)

 private:
  _Schedule_() { blocks_.resize(1); }
  std::vector<Var> iter_vars_;
  // values of the iter_vars
  std::vector<Expr> iter_values_;
  // BufferRange(s) which is read in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> read_buffers_;
  // BufferRange(s) which is written in this schedule block, it is used to
  // analyze, not a real computation expression. Must be AST DFS order.
  std::vector<Expr> write_buffers_;
  // Additional attributes about this schedulable block,
  // which take some auxiliary hints for future transformations.
  std::map<std::string, attr_t> attrs_;
  std::string name_;
  ReduceMethod reduce_method_{NoneReduceMethod()};
};

class Schedule : public StmtRef {
 public:
  Schedule(const std::vector<Var>& iter_vars,
           const std::vector<Expr>& iter_values,
           const std::vector<Expr>& read_buffers,
           const std::vector<Expr>& write_buffers,
           const std::string& name,
           const BlockRef& body,
           const std::map<std::string, attr_t>& attrs = {},
           const ReduceMethod& reduce_method = {NoneReduceMethod()})
      : Schedule(_Schedule_::Make(iter_vars,
                                  iter_values,
                                  read_buffers,
                                  write_buffers,
                                  name,
                                  body,
                                  attrs,
                                  reduce_method)) {}
  CINN_DEFINE_STMT_REF_METHODS(Schedule, _Schedule_)
};
}  // namespace stmt
}  // namespace ir
}  // namespace cinn
