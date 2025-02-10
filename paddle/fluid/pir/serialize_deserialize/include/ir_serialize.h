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
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"
#include "paddle/pir/include/core/program.h"

namespace pir {
/**
 * ProgramWriter is used to serialize pir program to json object.
 *
 */

class ProgramWriter {
 public:
  explicit ProgramWriter(const uint64_t version) : version_(version) {}
  explicit ProgramWriter(const uint64_t version, const bool trainable)
      : version_(version), trainable_(trainable) {}

  ProgramWriter(ProgramWriter&&) = delete;
  ProgramWriter(const ProgramWriter& ProgramWriter) = delete;
  ProgramWriter& operator=(const ProgramWriter&) = delete;
  ProgramWriter& operator=(ProgramWriter&&);

  /** GetProgramJson is used by writeModulde api*/
  Json GetProgramJson(const pir::Program* program);
  Json GetTypeJson(const pir::Type& type);
  Json GetAttributesMapJson(const AttributeMap& attr_map);

  ~ProgramWriter() = default;

 private:
  /** version_ is the version of paddlepaddle. which is used to
   * Conduct version compatibility judgment and modification.*/
  uint64_t version_;

  /** program_json is the json object of pir program. */
  Json program_json;

  /** value_id_map is used to record the serialize id of pir::Value.
   * which is used to serialize op's operands. */
  std::map<pir::Value, int64_t> value_id_map;

  /** xxx_id_ is used to record current id of IR structure
   * which should be serialized.*/

  int64_t region_id_ = 0;
  int64_t block_id_ = 0;
  int64_t value_id_ = 1;
  int64_t blockarg_id_ = -1;

  bool trainable_ = true;

  Json WriteProgram(const pir::Program* program);
  Json WriteRegion(const pir::Region* region, const std::string& region_name);
  Json WriteBlock(pir::Block* block, const std::string& block_name);
  Json WriteOp(const pir::Operation& op);
  Json WriteBlockArg(const pir::Value& value);
  Json WriteValue(const pir::Value& value);
  Json WriteOpOperand(const pir::OpOperand& op_operand);
  Json WriteAttributesMapOpinfo(pir::Operation* op,
                                const AttributeMap& attr_map);
  Json WriteAttributesMapOther(const AttributeMap& attr_map);
  /** WriteAttribute is used to write attribute of op.
   * which call writeAttr to get Derived Class‘s json object.
   * same as WriteType
   */

  Json WriteAttribute(const std::string& op_attr_name,
                      const pir::Attribute& attr);
  Json WriteType(const pir::Type& type);

  // special op for optimize json file size
  Json WriteParameterOP(const pir::Operation& op);
};

}  // namespace pir
