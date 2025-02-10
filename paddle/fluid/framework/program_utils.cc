/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/program_utils.h"

#include <google/protobuf/text_format.h>
#include "paddle/fluid/framework/block_desc.h"

namespace paddle::framework {

template <typename Container, typename Visitor>
inline void VisitAllElements(Container &&container,
                             Visitor &&visitor,
                             bool reverse) {
  if (reverse) {
    std::for_each(container.rbegin(), container.rend(), visitor);
  } else {
    std::for_each(container.begin(), container.end(), visitor);
  }
}

void MergePrograms(ProgramDesc *dst,
                   const std::vector<ProgramDesc> &srcs,
                   bool append) {
  PADDLE_ENFORCE_NOT_NULL(
      dst, common::errors::InvalidArgument("Dst program must be provided."));
  bool reverse = !append;

  auto create_var_visitor = [dst](const ProgramDesc &src) {
    PADDLE_ENFORCE_EQ(
        src.Size(),
        1,
        common::errors::Unimplemented("MergePrograms can only support to "
                                      "merge program with only one block."));
    const auto &src_block = src.Block(0);
    auto *dst_block = dst->MutableBlock(0);
    for (const auto *src_new_var : src_block.AllVars()) {
      if (dst_block->FindVar(src_new_var->Name())) continue;
      auto *dst_new_var = dst_block->Var(src_new_var->Name());
      *dst_new_var = *src_new_var;
      VLOG(10) << "Create new variable " << dst_new_var->Name()
               << ", persistable:" << dst_new_var->Persistable();
    }
  };

  VisitAllElements(srcs, create_var_visitor, reverse);

  auto create_op_visitor = [dst, reverse](const ProgramDesc &src) {
    auto ops = src.Block(0).AllOps();
    auto copy_op_visitor = [dst, reverse](const OpDesc *src_op) {
      auto *dst_block = dst->MutableBlock(0);
      auto *op = reverse ? dst_block->PrependOp() : dst_block->AppendOp();
      op->CopyFrom(*src_op);
      VLOG(10) << (reverse ? "Prepend" : "Append") << " op " << op->Type();
      // FIXME(zjl): some passes does not add VarDesc to program,
      // we should fix this bug later...
      for (const auto &in_var_name : op->InputArgumentNames()) {
        dst_block->Var(in_var_name);
      }
      for (const auto &out_var_name : op->OutputArgumentNames()) {
        dst_block->Var(out_var_name);
      }
    };
    VisitAllElements(ops, copy_op_visitor, reverse);
  };
  VisitAllElements(srcs, create_op_visitor, reverse);
}

void ProgramProcessor::GetInputsOutputsInBlock(
    const BlockDesc &current_block,
    std::set<std::string> *inner_inputs,
    std::set<std::string> *inner_outputs) {
  /* Find inputs and outputs in current control flow block.
  :param current_block: Current control flow block.
  :param inner_inputs: Input var vector of ops in current block.
  :param inner_outputs: Output var vector of ops in current block. */

  // Step1: update inner_inputs and inner_outputs
  // NOTE: Here assumes that all variables are input or output of Ops,

  for (OpDesc *op : current_block.AllOps()) {
    for (auto const &iname : op->InputNames()) {
      for (auto const &in_var_name : op->Input(iname)) {
        VLOG(3) << "insert inner_inputs_name:" << in_var_name;
        inner_inputs->insert(in_var_name);
      }
    }

    for (auto const &oname : op->OutputNames()) {
      for (auto const &out_var_name : op->Output(oname)) {
        VLOG(3) << "insert out_var_name:" << out_var_name;
        inner_outputs->insert(out_var_name);
      }
    }
  }

  // Step2: Remove variable created in current control flow block.
  BlockDesc *parent_block = current_block.ParentBlock();

  if (parent_block) {
    for (auto iter = inner_inputs->begin(); iter != inner_inputs->end();) {
      const std::string &in_var_name = *iter;
      if (current_block.HasVar(in_var_name)) {
        VLOG(3) << "remove inner input var:" << in_var_name;
        iter = inner_inputs->erase(iter);
      } else {
        ++iter;
      }
    }

    for (auto iter = inner_outputs->begin(); iter != inner_outputs->end();) {
      const std::string &out_var_name = *iter;
      if (current_block.HasVar(out_var_name)) {
        VLOG(3) << "remove inner output  var:" << out_var_name;
        iter = inner_outputs->erase(iter);
      } else {
        ++iter;
      }
    }
  }
}

void ProgramProcessor::AddDepToBlockOp(const BlockDesc &block) {
  VLOG(3) << "Op size:" << block.AllOps().size();
  for (OpDesc *op : block.AllOps()) {
    if (op->HasAttr("sub_block")) {
      auto op_type = op->Type();
      BlockDesc *sub_block =
          PADDLE_GET_MUTABLE(BlockDesc *, op->GetAttr("sub_block"));

      // recursively processing
      AddDepToBlockOp(*sub_block);

      std::set<std::string> sub_inputs;
      std::set<std::string> sub_outputs;
      ProgramProcessor::GetInputsOutputsInBlock(
          *sub_block, &sub_inputs, &sub_outputs);
      VLOG(3) << "sub_inputs.size:" << sub_inputs.size();
      VLOG(3) << "sub_outputs.size:" << sub_outputs.size();

      auto *op_inputs = op->MutableInputs();
      std::vector<std::string> *op_input_var_vec = nullptr;
      VLOG(3) << "op_type:>>>>>>" << op_type;
      if (op_type == "while") {
        op_input_var_vec = &((*op_inputs)["kX"]);
      } else if (op_type == "conditional_block") {
        op_input_var_vec = &((*op_inputs)["kInputs"]);
      } else {
        // Only support while_op and conditional_block_op now
        LOG(WARNING)
            << "Currently, only support while_op and conditional_block_op.\n";
        continue;
      }

      for (auto const &sub_input : sub_inputs) {
        if (std::find(op_input_var_vec->begin(),
                      op_input_var_vec->end(),
                      sub_input) == op_input_var_vec->end())
          op_input_var_vec->push_back(sub_input);
        VLOG(3) << "modified private inputs, inputs.size():"
                << op_input_var_vec->size();
      }

      auto *op_outputs = op->MutableOutputs();
      auto *op_output_var_vec = &((*op_outputs)["kOutputs"]);

      for (auto const &sub_output : sub_outputs) {
        if (std::find(op_output_var_vec->begin(),
                      op_output_var_vec->end(),
                      sub_output) == op_output_var_vec->end())
          op_output_var_vec->push_back(sub_output);
        VLOG(3) << "modified private outputs, outputs.size():"
                << op_output_var_vec->size();
      }
    }
  }
}

ProgramProcessor::ProgramProcessor() = default;

// write to file
void WriteToFile(const std::string &file_path, const std::string &msg) {
  FILE *fp = fopen(file_path.c_str(), "w");
  if (fp == nullptr) {
    LOG(WARNING) << "open write file path=" << file_path << " failed";
    return;
  }
  fwrite(msg.c_str(), 1, msg.length(), fp);
  fclose(fp);
}
void DumpProgramDescFile(const std::string &name, const ProgramDesc &program) {
  ProgramDesc *new_prog = const_cast<ProgramDesc *>(&program);
  std::string print_str;
  google::protobuf::TextFormat::Printer printer;
  printer.SetUseShortRepeatedPrimitives(true);
  printer.SetSingleLineMode(false);
  const ::google::protobuf::Message *message =
      reinterpret_cast<const ::google::protobuf::Message *>(new_prog->Proto());
  printer.PrintToString(*message, &print_str);

  std::string filename = "./" + name + "_" + std::to_string(time(0)) + ".proto";
  WriteToFile(filename.c_str(), print_str);
}

}  // namespace paddle::framework
