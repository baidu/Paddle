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

#pragma once

#include <string>
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"

namespace paddle {
namespace framework {
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace phi {
class DenseTensor;
class Place;
class CPUPlace;
}  // namespace phi

namespace pir {

class Operation;
class Block;
class Value;

using Variable = paddle::framework::Variable;
using Scope = paddle::framework::Scope;

/**
 * @brief Copy a DenseTensor to another.
 * default dst_plaxce is CPU.
 *
 * @param const phi::DenseTensor& src
 * @param phi::DenseTensor* dst
 * @param const phi::Place& dst_place
 *
 * @return
 */
void TensorCopySync(const phi::DenseTensor& src,
                    phi::DenseTensor* dst,
                    const phi::Place& dst_place = phi::CPUPlace());

/**
 * @brief Cast a DenseTensor to fp32.
 * world_size represents the maximum number of devices, defaulting to 1.
 * The result is either stored in 'out' or overwritten in 'in' if 'out' is
 * nullptr.
 *
 * @param phi::DenseTensor* in
 * @param phi::DenseTensor* out
 * @param int world_size
 *
 * @return
 */
void DenseTensorCastToFp32(phi::DenseTensor* in,
                           phi::DenseTensor* out = nullptr,
                           int world_size = 1);

/**
 * @brief Translate a DenseTensor to Ir's Type.
 *
 * @param phi::DenseTensor* tensor
 *
 * @return pir::Type
 */
pir::Type TranslateToIrDataType(phi::DataType dtype);

/**
 * @brief Create an Operation by name.
 * This method is typically used to directly construct operations under the
 * namespaces `pd_op.xxx` and `custom_op.xxx`.
 *
 * @param const std::string& op_name
 * @param const std::vector<pir::Value>& inputs
 * @param const pir::AttributeMap& attrs
 * @param const pir::PatternRewriter& rewriter
 *
 * @return pir::Operation*
 */
pir::Operation* CreateOperationByName(const std::string& op_name,
                                      const std::vector<pir::Value>& inputs,
                                      const pir::AttributeMap& attrs,
                                      const pir::PatternRewriter& rewriter);

/**
 * @brief Create a DataType attribute.
 *
 * @param pir::IrContext * ctx
 * @param phi::DataType dtype
 **/
pir::Attribute CreateDataTypeAttr(pir::IrContext* ctx, phi::DataType dtype);

/**
 * @brief Get the mutable data of a Variable.
 *
 * @param Variable* var
 *
 * @return T*
 */
template <typename T>
T* VarGetMutable(Variable* var);

/**
 * @brief Check if a Variable is of the specified type.
 *
 * @param Variable* var
 *
 * @return bool
 */
template <typename T>
bool VarIsType(Variable* var);

/**
 * @brief Find a Variable in the scope.
 *
 * @param Scope* scope_
 *
 * @return Variable*
 */
Variable* ScopeFindVar(Scope* scope_, const std::string& name);

/**
 * @brief Get a Variable in the scope.
 *
 * @param Scope* scope_
 *
 * @return Variable*
 */
Variable* ScopeGetVar(Scope* scope_, const std::string& name);

/**
 * @brief Get a Variable in the scope.
 *
 * @param Scope* scope_
 * @param const std::string& name
 *
 * @return Variable*
 */
Variable* ScopeVar(Scope* scope_, const std::string& name);

/**
 * @brief Get all the names of Variables in the scope.
 *
 * @param Scope* scope_
 *
 * @return std::vector<std::string>
 */
std::vector<std::string> ScopeGetVarNames(Scope* scope_);

/**
 * @brief Get the scope of a pass.
 *
 * @param pir::Pass* pass
 *
 * @return Scope*
 */
Scope* GetScopeImpl(pir::Pass* pass);

/**
 * @brief Get the name of parameter from a value.
 *
 * @note The value must be a output of a ParameterOp or a ConstantTensorOp.
 *
 * @param const pir::Value&
 *
 * @return std::string
 */
std::string GetParameterNameFromValue(const pir::Value& value);

/**
 * @brief Get tensor's shape from a value.
 *
 * @param const pir::Value&
 *
 * @return std::vector<int64_t>
 */
std::vector<int64_t> GetShapeFromValue(const pir::Value& value);

/**
 * @brief Get tensor's data type from a value.
 *
 * @param const pir::Value&
 *
 * @return pir::Type
 */
pir::Type GetDataTypeFromValue(const pir::Value& value);

/**
 * @brief Get an operation that defines the specific input of the operation.
 *
 * @param const Operation* const pointer to an operation
 * @param uint32_t index of operand of the operation
 *
 * @return Operation*
 */
TEST_API Operation* GetDefiningOpForInput(const Operation* op, uint32_t index);

/**
 * @brief Get operations and the index of designative op operand (op result)
 that use the specific output of the operation.
 *
 * @param const Operation* const pointer to an operation
 * @param uint32_t index of result of the operation

 * @return std::vector<std::pair<Operation*, int32_t>>
 */
std::vector<std::pair<Operation*, int32_t>> GetUseOpsForOutput(
    const Operation* op, uint32_t index);

/**
* @brief Get the value of the input and output of the specified op in the
external block.
*
* @param const Operation& const reference to an operation

* @return std::vector<Value>
*/
std::vector<Value> GetUsedExternalValue(const Operation& op);

/**
 * @brief Get the external value of the input and output of all op which in the
 specified block.
 *
 * @param const Block& const reference to an block

 * @return std::vector<Value>
 */
std::vector<Value> GetUsedExternalValue(const Block& block);

/**
 * @brief Determine whether a value comes from a weight or has no input op. That
 is to say, it is permissible.
 *
 * @param const pir::Value&

 * @return bool
 */
bool ValueIsPersistable(const pir::Value& value);

phi::DataType GetTensorDtype(pir::Type type);
phi::DataType GetValueDtype(const pir::Value& val);

}  // namespace pir
