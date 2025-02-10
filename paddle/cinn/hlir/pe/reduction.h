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
#include <string>
#include <vector>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {
/**
 * @brief sums array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes along which a sum is performed. If axis is empty,
 * the operation will sum over all elements of the input array. If axis is
 * negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param initial Starting value for the sum.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensors.
 */
ir::Tensor ReduceSum(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     const bool keep_dims = false,
                     const std::string& output_name = "T_Reduce_Sum_out");

/**
 * @brief product array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes along which a production is performed. If axis is
 * empty, the operation will product over all elements of the input array. If
 * axis is negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param initial Starting value for the production.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensors.
 */
ir::Tensor ReduceProd(const ir::Tensor& A,
                      const std::vector<int>& axis,
                      const bool keep_dims = false,
                      const std::string& output_name = "T_Reduce_Prod_out");

/**
 * @brief find the maximum of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the maximum over. If axis is empty, the
 * operation will product over all elements of the input array. If axis is
 * negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceMax(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     const bool keep_dims = false,
                     const std::string& output_name = "T_Reduce_Max_out");

/**
 * @brief find the minimum of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the minimum over. If axis is empty, the
 * operation will product over all elements of the input array. If axis is
 * negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceMin(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     const bool keep_dims = false,
                     const std::string& output_name = "T_Reduce_Min_out");

/**
 * @brief find the logic and of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the logic and over. If axis is empty, the
 * operation will product over all elements of the input array. If axis is
 * negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceAll(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     const bool keep_dims = false,
                     const std::string& output_name = "T_Reduce_All_out");

/**
 * @brief find the logic or of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the logic or over. If axis is empty, the
 * operation will product over all elements of the input array. If axis is
 * negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceAny(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     const bool keep_dims = false,
                     const std::string& output_name = "T_Reduce_Any_out");

/**
 * @brief find the logic or of array elements over a given axis
 *
 * @param A The input Tensor
 * @param stages The stage map
 * @param axis Axis or axes to find the logic or over. If axis is empty, the
 * operation will product over all elements of the input array. If axis is
 * negative it counts from the last to the first axis.
 * @param keep_dims If it is set true, the axes which are reduced are left in
 * the result as dimensions with size one. With this option, the result will
 * broadcast correctly against the input array.
 * @param output_name The name of the output Tensor
 *
 * @return The result Tensor.
 */
ir::Tensor ReduceVar(const ir::Tensor& A,
                     const std::vector<int>& axis,
                     const bool keep_dims = false,
                     const std::string& output_name = "T_Reduce_Var_out");

/**
 * @brief find the max of array elements over the last dimension
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> WarpReduceMax(
    const ir::Tensor& A,
    const int last_reduce_dim_num,
    const bool keep_dim = false,
    const std::string& output_name = "T_Warp_Reduce_Max_out");

/**
 * @brief compute the sum of array elements over the last dimension
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> WarpReduceSum(
    const ir::Tensor& A,
    const int last_reduce_dim_num,
    const bool keep_dim = false,
    const std::string& output_name = "T_Warp_Reduce_Sum_out");

/**
 * @brief compute the average of array elements over the last dimension
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> WarpReduceAvg(
    const ir::Tensor& A,
    const int last_reduce_dim_num,
    const bool keep_dim = false,
    const std::string& output_name = "T_Warp_Reduce_Avg_out");

/**
 * @brief compute the sum of array elements over the last dimension with block
 * reduce. 'BlockReduceSumInternal' is used as the internal compute of reduce
 * sum, do not use it directly.
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceSumInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Sum_Internal_out");

/**
 * @brief compute the Product of array elements over the last dimension with
 * block reduce. 'BlockReduceSumInternal' is used as the internal compute of
 * reduce sum, do not use it directly.
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceProdInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Prod_Internal_out");

/**
 * @brief compute the Max of array elements over the last dimension with block
 * reduce. 'BlockReduceSumInternal' is used as the internal compute of reduce
 * sum, do not use it directly.
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceMaxInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Max_Internal_out");

/**
 * @brief compute the Min of array elements over the last dimension with block
 * reduce. 'BlockReduceSumInternal' is used as the internal compute of reduce
 * sum, do not use it directly.
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceMinInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Min_Internal_out");

/**
 * @brief compute the logic and of array elements over the last dimension with
 * block reduce. 'BlockReduceSumInternal' is used as the internal compute of
 * reduce sum, do not use it directly.
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceAllInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_All_Internal_out");

/**
 * @brief compute the logic or of array elements over the last dimension with
 * block reduce. 'BlockReduceSumInternal' is used as the internal compute of
 * reduce sum, do not use it directly.
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceAnyInternal(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Any_Internal_out");

/**
 * @brief compute the Sum of array elements over the last dimension with block
 * reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceSum(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const int block_size,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Sum_out");

/**
 * @brief compute the Product of array elements over the last dimension with
 * block reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceProd(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const int block_size,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Prod_out");

/**
 * @brief compute the Max of array elements over the last dimension with block
 * reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceMax(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const int block_size,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Max_out");

/**
 * @brief compute the Min of array elements over the last dimension with block
 * reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceMin(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const int block_size,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Min_out");

/**
 * @brief compute the logic and of array elements over the last dimension with
 * block reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceAll(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const int block_size,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_All_out");

/**
 * @brief compute the logic or of array elements over the last dimension with
 * block reduce
 *
 * @param A The input Tensor.
 * @param last_reduce_dim_num the number of last reduce dimension.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockReduceAny(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const int block_size,
    const bool keep_dim = false,
    const std::string& output_name = "T_Block_Reduce_Any_out");

/**
 * @brief compute the value of array elements over the last dimension with block
 * reduce
 *
 * @param A The input Tensor.
 * @param axes the reduce axes.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */
std::vector<ir::Tensor> BlockShuffleReduceSum(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Sum_out");

std::vector<ir::Tensor> BlockShuffleReduceProd(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Prod_out");

std::vector<ir::Tensor> BlockShuffleReduceMax(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Max_out");

std::vector<ir::Tensor> BlockShuffleReduceMin(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Min_out");

std::vector<ir::Tensor> BlockShuffleReduceAll(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_All_out");

std::vector<ir::Tensor> BlockShuffleReduceAny(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Any_out");

/**
 * @brief compute the value of array elements over the last dimension with block
 * reduce
 *
 * @param A The input Tensor.
 * @param axes the reduce axes.
 * @param keep_dim keep the output tensor shape size as input.
 * @param output_name The name of the output Tensor.
 */

std::vector<ir::Tensor> TwoStepBlockReduceSum(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Sum_out");

std::vector<ir::Tensor> TwoStepBlockReduceProd(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Prod_out");

std::vector<ir::Tensor> TwoStepBlockReduceMax(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Max_out");

std::vector<ir::Tensor> TwoStepBlockReduceMin(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Min_out");

std::vector<ir::Tensor> TwoStepBlockReduceAll(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_All_out");

std::vector<ir::Tensor> TwoStepBlockReduceAny(
    const ir::Tensor& A,
    const std::vector<int>& axes,
    const bool keep_dim,
    const std::string& output_name = "T_Reduce_Any_out");

std::string CrossThreadReduceExternalFuncName(const ir::Expr& op,
                                              const ir::Expr& tensor);

std::string DiscreteReduceExternalFuncName(const ir::Expr& op,
                                           const ir::Expr& tensor);

std::string GridReduceExternalFuncName(const ir::Expr& op,
                                       const cinn::common::Type type);

std::string Type2StrForReduce(cinn::common::Type type);
}  // namespace pe
}  // namespace hlir
}  // namespace cinn
