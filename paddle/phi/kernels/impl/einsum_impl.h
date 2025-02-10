// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/diagonal_kernel.h"
#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/tile_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/utils/string/string_helper.h"

PD_DECLARE_bool(einsum_opt);

namespace phi {

// check the validation of the Einsum equation.
// 1. the label must between 'a' - 'z'.
// 2. the dim of the same label must be same.
// 3. the broad cast dims in two operands is broadcastable.
// 4. there must exist '->' and the default output is complete in python.
// may be we can skip validation check in C++ and just put it in python.
inline static void ValidationCheck(const std::string& equation) {
  auto n_part = paddle::string::split_string(equation, "->").size();
  PADDLE_ENFORCE_EQ(n_part,
                    2,
                    common::errors::InvalidArgument(
                        "Required at least one `->` in equation of EinsumOp."));
  size_t pos;
  auto trimmed_equ = equation;
  if ((pos = trimmed_equ.find("->", 0)) != std::string::npos) {
    trimmed_equ.replace(pos, 2, "");
  }
  auto is_valid_char = [](char c) {
    if (c >= 'a' && c <= 'z') return true;
    if (c == ',') return true;
    return false;
  };
  for (auto c : trimmed_equ) {
    if (!is_valid_char(c))
      PADDLE_THROW(common::errors::InvalidArgument(
          "Found invalid char in equation. Einsum only accept `a`-`z` and `...`"
          "but get:`%c`",
          c));
  }
}

enum LabelType {
  ALL_TYPE = 0,
  Batch = 1,    // ABO
  AO,           // AO --  free label
  BO,           // BO --  free label
  Contraction,  // AB
  Reduction,    // A, B
};

// map a label('a' - 'z') -> int, O(1) speed.
class LabelMap {
  constexpr static int N =
      26 + 1;  // 'a' - 'z' + '.', '.' is for broadcast dims
  int default_value;
  int map[N];

 public:
  explicit LabelMap(int default_value = 0) {
    this->default_value = default_value;
    for (size_t i = 0; i < N; ++i) map[i] = default_value;
  }
  int& operator[](int label) {
    int i = label - 'a';
    return map[i];
  }
  int operator[](int label) const {
    int i = label - 'a';
    return map[i];
  }
  bool exist(char label) { return !is_default(label); }

 private:
  // non-exist is present by is_default
  bool is_default(char label) {
    return (*this)[static_cast<int>(label)] == default_value;
  }
};

inline std::string label_to_string(const std::vector<char>& all_labels,
                                   const LabelMap& label2type) {
  std::string str;
  for (int a : all_labels) {
    std::stringstream ss;
    ss << label2type[a];
    str += ss.str();
  }
  return str;
}

template <typename CharIterable1, typename CharIterable2>
inline std::vector<char> union_labels(const CharIterable1& a,
                                      const CharIterable2& b) {
  LabelMap counter(0);
  std::vector<char> res;
  auto f = [&](char c) {
    if (counter[static_cast<int>(c)] == 0) {
      res.push_back(c);
    }
    counter[static_cast<int>(c)] += 1;
  };
  std::for_each(a.begin(), a.end(), f);
  std::for_each(b.begin(), b.end(), f);
  return res;
}

template <typename CharIterable>
inline std::vector<char> unique_labels(const CharIterable& a) {
  return union_labels(a, CharIterable());
}

// Apply transforms to all_labels and get another all_labels
inline std::vector<char> TransformLabelsOrder(
    const std::vector<char>& all_labels,
    const LabelMap& type,
    std::vector<LabelType> new_order) {
  std::vector<char> ret;
  for (auto cnt_type : new_order) {
    std::vector<char> tmp;
    for (int c : all_labels) {
      if (type[c] == cnt_type) tmp.push_back(c);
    }
    ret.insert(ret.end(), tmp.begin(), tmp.end());
  }
  return ret;
}

inline static void GlobalInfo(const std::vector<std::string>& op_labels,
                              const std::string& right,
                              LabelMap* label2type,
                              std::vector<char>* sorted_labels) {
  std::vector<char> all;
  LabelMap counter(0);
  for (auto& ch : right) {  // char
    int c = ch;
    (*label2type)[c] = LabelType::BO;
  }

  for (auto& op : op_labels) {
    for (auto& ch : unique_labels(op)) {  // char
      int c = ch;
      if (!counter.exist(c)) {
        all.push_back(ch);
      }
      counter[c] += 1;
      if ((*label2type)[c] != LabelType::BO && counter[c] == 2)
        (*label2type)[c] = LabelType::Contraction;
      else if (counter[c] == 2)
        (*label2type)[c] = LabelType::Batch;
    }
  }

  // BO is represent Free, so we need find the AO.
  for (int c : op_labels[0]) {
    if ((*label2type)[c] == LabelType::BO) (*label2type)[c] = LabelType::AO;
  }

  if (sorted_labels->size()) {
    std::set<char> exist(all.begin(), all.end());
    all.clear();
    std::for_each(
        sorted_labels->begin(), sorted_labels->end(), [&exist, &all](char c) {
          if (exist.count(c)) all.push_back(c);
        });
  }

  *sorted_labels = TransformLabelsOrder(all,
                                        *label2type,
                                        {LabelType::Batch,
                                         LabelType::AO,
                                         LabelType::BO,
                                         LabelType::Contraction,
                                         LabelType::Reduction});

  VLOG(5) << "GlobalInfo: sorted_labels after: "
          << paddle::string::join_strings(*sorted_labels, ",");
}

inline static void InferLabelShape(
    const std::vector<std::string>& op_labels,
    const std::vector<DDim>& inputs,
    LabelMap* labelshape,
    std::vector<std::vector<int>>* broadcast_shapes) {
  VLOG(5) << "Start InferLabelShape";
  for (size_t i = 0; i < op_labels.size(); ++i) {
    auto& op_str = op_labels[i];
    auto& op_dim = inputs[i];
    int dim_ptr = 0;
    for (auto& c : op_str) {
      if (!labelshape->exist(c) || abs((*labelshape)[c]) == 1) {
        (*labelshape)[c] = static_cast<int>(op_dim[dim_ptr]);
      } else if (abs(op_dim[dim_ptr]) != 1) {
        PADDLE_ENFORCE_EQ(
            (*labelshape)[c],
            op_dim[dim_ptr],
            common::errors::InvalidArgument(
                "Same label have different shapes for label: `%c`", c));
      }
      dim_ptr++;
    }
  }
  for (size_t i = 0; i < op_labels.size(); ++i) {
    for (auto& c : op_labels[i]) {
      (*broadcast_shapes)[i].push_back((*labelshape)[c]);
    }
  }
  for (size_t i = 0; i < op_labels.size(); ++i) {
    VLOG(5) << "InferLabelShape: After broadcast shape is:"
            << paddle::string::join_strings((*broadcast_shapes)[i], ",");
  }
}

template <class CharIterable>
inline static void InferLabelPerm(const CharIterable& op,
                                  LabelMap* label2perm) {
  int cur = 0;
  for (int c : op) {
    if (!label2perm->exist(
            c))  // can appear repeatedly. we just record the first position.
      (*label2perm)[c] = cur;
    cur += 1;
  }
}

inline static void InferOutputDims(const std::string& right,
                                   const LabelMap& labelshape,
                                   std::vector<int>* output_dims) {
  for (int c : right) {
    output_dims->push_back(labelshape[c]);
  }
}
//
inline static void ParseEinsumEquation(
    const std::string& equation,
    const std::vector<DDim>& inputs,
    LabelMap* labelshape,
    LabelMap* labeltype,
    std::vector<char>* all_labels,
    std::vector<LabelMap>* label2perms,
    std::vector<std::vector<int>>* broadcast_shapes,
    std::vector<int>* output_dims,
    std::string* right,
    std::vector<std::string>* input_strs) {
  VLOG(5) << "Start ParseEinsumEquation " << equation;
  auto results = paddle::string::split_string(equation, "->");
  auto left = results[0];
  *right = results[1];
  auto op_labels = paddle::string::split_string(left, ",");
  // split_string("i,") -> ["i", ""], we push back a "".
  // split_string("->") -> [], we push back a "".
  if (op_labels.empty()) op_labels.emplace_back("");
  GlobalInfo(op_labels, *right, labeltype, all_labels);
  InferLabelShape(op_labels, inputs, labelshape, broadcast_shapes);
  VLOG(5) << "Einsum Infershape: right:" << *right;
  VLOG(5) << "Einsum Infershape: left :"
          << paddle::string::join_strings(op_labels, '\n');
  InferOutputDims(*right, *labelshape, output_dims);
  for (size_t i = 0; i < inputs.size(); ++i) {
    InferLabelPerm(op_labels[i], &((*label2perms)[i]));
    (*input_strs).push_back(std::move(op_labels[i]));
  }
}

template <typename T>
std::vector<T> GetLabelIndexByType(const std::vector<char>& all_labels,
                                   const LabelMap& type,
                                   const LabelMap& perm,
                                   LabelType filter) {
  std::vector<T> res;
  for (T c : all_labels) {
    if ((filter == LabelType::ALL_TYPE || type[c] == filter) && perm[c] != -1) {
      res.push_back(perm[c]);
    }
  }
  return res;
}

template <typename T>
std::vector<T> GetShapeByType(const std::vector<char>& all_labels,
                              const LabelMap& type,
                              const LabelMap& perm,
                              const LabelMap& label2shape,
                              std::set<LabelType> filter) {
  std::vector<T> res;
  for (T c : all_labels) {
    if ((filter.count(LabelType::ALL_TYPE) ||
         filter.count(LabelType(type[c]))) &&
        perm[c] != -1) {
      res.push_back(label2shape[c]);
    }
  }
  return res;
}

inline static std::vector<int> perm_moveto(int n, int from, int to) {
  // a permutation means moving `from` to `to`.
  /*
  f => t   permutation
  --------------------
           0 1 2 3 4 5
  5 => 2 : 0 2 5 2 3 4
  2 => 5 : 0 1 3 4 5 2
  we can conclude the following rules.
  */
  if (from < 0) from = n + from;
  if (to < 0) to = n + to;
  std::vector<int> res(n);
  for (int i = 0; i < n; ++i) {
    res[i] = i;
  }
  res[to] = from;
  auto offset = from > to ? -1 : 1;
  auto start = from > to ? to + 1 : from;
  auto end = from > to ? from : to - 1;
  for (int i = start; i <= end; ++i) {
    res[i] += offset;
  }
  return res;
}

template <typename T, typename Context>
DenseTensor Undiagonal(const Context& dev_ctx,
                       const DenseTensor& tensor,
                       size_t insert_pos,
                       size_t axis) {
  // tensor with shape (3, 4, 5, 2, 1), insert_pos = 5, axis = 2.
  // output is (3, 4, 5, 2, 1, 5)
  VLOG(5) << "Start undiagonal with args: insert_pos = " << insert_pos
          << ", axis = " << axis;
  std::vector<int> shape(tensor.dims().size() + 1);
  int point = 0;  // point to the tensor.dims()
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == insert_pos)
      shape[i] = tensor.dims()[axis];
    else
      shape[i] = tensor.dims()[point++];
  }
  auto zeros = Full<T, Context>(dev_ctx, shape, 0);
  auto diags = Transpose<T, Context>(
      dev_ctx, tensor, perm_moveto(tensor.dims().size(), axis, -1));
  return FillDiagonalTensor<T, Context>(
      dev_ctx, zeros, diags, 0, insert_pos, axis + (insert_pos <= axis));
}

template <typename T, typename Context>
DenseTensor PerformUndiagonal(const Context& dev_ctx,
                              const DenseTensor& tensor,
                              const std::string& equ) {
  //  if the equ is 'iijjkij', then the tensor must be 'ijk', so we have enough
  //  information to do un-diagonal with equ.
  auto res = tensor;
  LabelMap label2perm(-1);
  InferLabelPerm(equ, &label2perm);
  // Un-Diagonal
  int tot = equ.size();
  int cur = tot - 1;
  for (auto it = equ.rbegin(); it != equ.rend(); ++it) {
    char c = *it;
    if (cur != label2perm[c]) {
      // do diagonal, followed by movedim().
      auto insert_pos = cur - tot + res.dims().size() + 1;
      res = Undiagonal<T, Context>(dev_ctx, res, insert_pos, label2perm[c]);
    }
    --cur;
  }
  return res;
}

template <typename T, typename Context>
DenseTensor PerformDiagonalAndReduction(const Context& dev_ctx,
                                        const DenseTensor& tensor,
                                        const std::string& equ,
                                        const LabelMap& label2perm,
                                        const std::vector<char>& all_labels,
                                        const std::vector<int>& broadcast_shape,
                                        const LabelMap& label2type) {
  auto res = tensor;
  int tot = equ.size();
  // tiling tensor for broadcast
  std::vector<int> repeat_times;
  auto tensor_origin_shape = common::vectorize(tensor.dims());
  for (size_t i = 0; i < tensor_origin_shape.size(); ++i) {
    VLOG(4) << "broadcast shape is " << broadcast_shape[i]
            << ", tensor shape is " << tensor_origin_shape[i];
    repeat_times.push_back(broadcast_shape[i] / tensor_origin_shape[i]);
  }
  DenseTensor after_tile;
  bool is_all_ones = std::all_of(
      repeat_times.begin(), repeat_times.end(), [](int x) { return x == 1; });
  if (!is_all_ones) {
    TileKernel<T, Context>(dev_ctx, res, repeat_times, &after_tile);
    res = after_tile;
  }
  // Diagonal
  int cur = tot - 1;
  for (auto it = equ.rbegin(); it != equ.rend(); ++it) {
    char c = *it;
    if (cur != label2perm[c]) {
      // do diagonal, followed by movedim().
      VLOG(5) << "Do diagonal with shape="
              << paddle::string::join_strings(
                     common::vectorize<int>(res.dims()), ',')
              << ", axis1=" << cur << ", axis2=" << label2perm[c];
      res = Diagonal<T, Context>(dev_ctx, res, 0, cur, label2perm[c]);
      res = Transpose<T, Context>(
          dev_ctx, res, perm_moveto(res.dims().size(), -1, label2perm[c]));
    }
    --cur;
  }
  // reduction
  auto indices = GetLabelIndexByType<int64_t>(
      all_labels, label2type, label2perm, LabelType::Reduction);
  VLOG(5) << "call PerformDiagonalAndReduction: with axis: "
          << paddle::string::join_strings(indices, ",");
  if (indices.empty()) return res;
  return Sum<T, Context>(
      dev_ctx, res, phi::IntArray(indices), res.dtype(), true);
}

inline bool is_no_need_transpose(const std::vector<int>& axis) {
  for (size_t i = 0; i < axis.size(); ++i) {
    if (i != static_cast<size_t>(axis[i])) return false;
  }
  return true;
}

template <typename T, typename Context>
DenseTensor PerformTranspose(const Context& dev_ctx,
                             const DenseTensor& tensor,
                             const LabelMap& label2perm,
                             const std::vector<char>& all_labels,
                             const LabelMap& label2type) {
  auto axis = GetLabelIndexByType<int>(
      all_labels, label2type, label2perm, LabelType::ALL_TYPE);
  VLOG(5) << "PerformTranspose: " << paddle::string::join_strings(axis, ",");
  if (is_no_need_transpose(axis)) {
    return tensor;
  }
  auto ret = Transpose<T, Context>(dev_ctx, tensor, axis);
  VLOG(5) << "PerformTranspose: do_transpose()";
  return ret;
}

template <typename T, typename Context>
DenseTensor PerformContraction(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& operands,
    const std::vector<std::string>& input_strs,
    const std::vector<LabelMap>& label2perm,
    const std::vector<char>& all_labels,
    const LabelMap& label2type,
    const LabelMap& label2shape,
    const std::vector<std::vector<int>>& broadcast_shapes,
    std::vector<DenseTensor*> cache,
    bool use_cache) {
  auto all_valid = LabelMap(1);
  auto recover_dim = GetShapeByType<int>(
      all_labels, label2type, all_valid, label2shape, {LabelType::Batch});
  auto preprocess = [&](const DenseTensor& t,
                        const LabelMap& perm,
                        const std::vector<int>& broadcast,
                        int operand_idx) -> DenseTensor {
    // reshape
    auto frees = GetShapeByType<int>(all_labels,
                                     label2type,
                                     perm,
                                     label2shape,
                                     {LabelType::AO, LabelType::BO});
    auto conts = GetShapeByType<int>(
        all_labels, label2type, perm, label2shape, {LabelType::Contraction});
    std::vector<char> reordered_all_labels = all_labels;
    if (operand_idx == 1) {
      reordered_all_labels = TransformLabelsOrder(all_labels,
                                                  label2type,
                                                  {LabelType::Batch,
                                                   LabelType::Contraction,
                                                   LabelType::AO,
                                                   LabelType::BO,
                                                   LabelType::Reduction});
    }
    // reduction
    DenseTensor trans_t;
    if (use_cache && cache[operand_idx] != nullptr &&
        cache[operand_idx]->IsInitialized()) {
      trans_t.ShareBufferWith(*(cache[operand_idx]));
      VLOG(5) << "Cache Used!";
    } else {
      auto reduct_t =
          PerformDiagonalAndReduction<T, Context>(dev_ctx,
                                                  t,
                                                  input_strs[operand_idx],
                                                  perm,
                                                  all_labels,
                                                  broadcast_shapes[operand_idx],
                                                  label2type);
      trans_t = PerformTranspose<T, Context>(
          dev_ctx, reduct_t, perm, reordered_all_labels, label2type);
      if (cache[operand_idx] != nullptr) {
        cache[operand_idx]->ShareBufferWith(trans_t);
        cache[operand_idx]->Resize(trans_t.dims());
        VLOG(5) << "Set dims of cache[" << operand_idx
                << "]: " << trans_t.dims();
      }
    }
    auto mul_dims = GetShapeByType<int>(
        all_labels, label2type, perm, label2shape, {LabelType::Batch});
    recover_dim.insert(recover_dim.end(), frees.begin(), frees.end());
    if (operand_idx == 0) {
      mul_dims.push_back(std::accumulate(
          frees.begin(), frees.end(), 1, std::multiplies<int>()));
      mul_dims.push_back(std::accumulate(
          conts.begin(), conts.end(), 1, std::multiplies<int>()));
    } else {
      mul_dims.push_back(std::accumulate(
          conts.begin(), conts.end(), 1, std::multiplies<int>()));
      mul_dims.push_back(std::accumulate(
          frees.begin(), frees.end(), 1, std::multiplies<int>()));
    }
    VLOG(5) << "PerformContraction: mul_dims: "
            << paddle::string::join_strings(mul_dims, ",");
    trans_t.Resize(common::make_ddim(mul_dims));
    return trans_t;
  };

  // Reduction, Reshape and Matmul
  DenseTensor after_contraction;
  if (operands.size() == 2) {
    auto trans_a =
        preprocess(*(operands[0]), label2perm[0], broadcast_shapes[0], 0);
    auto trans_b =
        preprocess(*(operands[1]), label2perm[1], broadcast_shapes[1], 1);
    after_contraction =
        Matmul<T, Context>(dev_ctx, trans_a, trans_b, false, false);
  } else if (operands.size() == 1) {
    after_contraction =
        preprocess(*(operands[0]), label2perm[0], broadcast_shapes[0], 0);
  }
  if (recover_dim.empty()) recover_dim.push_back(1);
  VLOG(5) << "PerformContraction: recover_dim: "
          << paddle::string::join_strings(recover_dim, ",");
  after_contraction.Resize(common::make_ddim(recover_dim));
  return after_contraction;
}

template <typename T, typename Context>
DenseTensor TransposeToOutput(const Context& dev_ctx,
                              const DenseTensor& to_trans,
                              const std::vector<char>& right,
                              const std::vector<char>& all_labels) {
  std::vector<int> axis;
  for (char c : right) {
    auto it = std::find(all_labels.begin(), all_labels.end(), c);
    PADDLE_ENFORCE_NE(it,
                      all_labels.end(),
                      common::errors::InvalidArgument("Must in all_labels."));
    axis.push_back(it - all_labels.begin());
  }
  if (is_no_need_transpose(axis)) {
    return to_trans;
  }
  VLOG(5) << "call TransposeToOutput: with axis: "
          << paddle::string::join_strings(axis, ",")
          << "  to trans dims is: " << to_trans.dims();
  auto output = Transpose<T, Context>(dev_ctx, to_trans, axis);
  VLOG(5) << "After Transpose.";
  return output;
}

template <typename T, typename Context>
void EinsumKernelImpl(const Context& dev_ctx,
                      const std::vector<char>& forward_all_labels,
                      const std::vector<const DenseTensor*>& inputs,
                      const std::string& equation,
                      DenseTensor* out,
                      std::vector<DenseTensor*> cache,
                      bool is_forward = true) {
  VLOG(5) << "Start EinsumKernelImpl with inputs(" << inputs.size() << "): ";
  for (auto& i : inputs) {
    VLOG(5) << "      inputs [ " << i << " ].shape=" << i->dims();
  }
  ValidationCheck(equation);
  // collect the following information to prepare einsum.
  LabelMap labelshape(0);
  LabelMap labeltype(LabelType::Reduction);
  std::vector<LabelMap> label2perms(inputs.size(), LabelMap(-1));
  std::vector<char> all_labels;  // order: ABO, AO, BO, AB, Reduce
  std::vector<std::vector<int>> broadcast_shapes(2);
  std::vector<int> output_dims;

  std::vector<DDim> input_dims;
  for (auto& i : inputs) {
    input_dims.push_back(i->dims());
  }
  std::vector<std::string> input_strs;
  std::string right;
  if (!is_forward) {
    all_labels = forward_all_labels;
  }
  ParseEinsumEquation(equation,
                      input_dims,
                      &labelshape,
                      &labeltype,
                      &all_labels,
                      &label2perms,
                      &broadcast_shapes,
                      &output_dims,
                      &right,
                      &input_strs);
  if (inputs.size() > 2) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "EinsumOp kernel only support len(operands) between (0, 2]. Use "
        "opt_einsum first to convert multi-variable to binary-variable."));
  }
  auto after_contraction = PerformContraction<T, Context>(dev_ctx,
                                                          inputs,
                                                          input_strs,
                                                          label2perms,
                                                          all_labels,
                                                          labeltype,
                                                          labelshape,
                                                          broadcast_shapes,
                                                          cache,
                                                          !is_forward);
  *out = TransposeToOutput<T, Context>(
      dev_ctx, after_contraction, unique_labels(right), all_labels);
  *out = PerformUndiagonal<T, Context>(dev_ctx, *out, right);
  out->Resize(common::make_ddim(output_dims));
}

template <typename T, typename Context>
void EinsumKernel(const Context& dev_ctx,
                  const std::vector<const DenseTensor*>& inputs,
                  const std::string& equation,
                  DenseTensor* out,
                  std::vector<DenseTensor*> cache,
                  std::vector<DenseTensor*> xshape UNUSED) {
  std::vector<char> tmp;
  // for the sake of compatibility, we may load and run v2.3 EinsumOp. Output
  // may have nullptr and the cache.size() is not equal to inputs.size(). refer
  // to BuildPhiKernelContext for details.
  int diff = inputs.size() - cache.size();
  for (int i = 0; i < diff; ++i) {
    cache.push_back(nullptr);
  }
  EinsumKernelImpl<T, Context>(
      dev_ctx, tmp, inputs, equation, out, cache, /*forward=*/true);
}

template <typename T, typename Context>
void EinsumInferKernel(const Context& dev_ctx,
                       const std::vector<const DenseTensor*>& inputs,
                       const std::string& equation,
                       DenseTensor* out) {
  std::vector<char> place_holder;
  std::vector<DenseTensor*> cache_tensor(
      inputs.size());  // set empty; TA, TB, TdC
  for (size_t i = 0; i < inputs.size(); ++i) {
    cache_tensor[i] = nullptr;
  }
  EinsumKernelImpl<T, Context>(
      dev_ctx, place_holder, inputs, equation, out, cache_tensor, true);
}

}  // namespace phi
