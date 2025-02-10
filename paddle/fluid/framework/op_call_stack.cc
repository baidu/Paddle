/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_call_stack.h"
#include <string>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle::framework {

std::string InsertIndentationIntoEachLine(const std::string &str) {
  std::ostringstream sout;
  size_t start_pos = 0;
  size_t end_pos = 0;
  while ((end_pos = str.find_first_of('\n', start_pos)) != std::string::npos) {
    sout << "    " << str.substr(start_pos, end_pos - start_pos + 1);
    start_pos = end_pos + 1;
  }
  sout << "    " << str.substr(start_pos, end_pos - start_pos + 1);
  return sout.str();
}

void InsertCallStackInfo(const std::string &type,
                         const paddle::framework::AttributeMap &attrs,
                         platform::EnforceNotMet *exception) {
  if (attrs.count("sub_block") != 0) {
    return;
  }

  const std::vector<std::string> *callstack = nullptr;
  auto iter = attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
  if (iter != attrs.end()) {
    callstack = &PADDLE_GET_CONST(std::vector<std::string>, iter->second);
    if (callstack->empty()) callstack = nullptr;
  }

  std::ostringstream sout;
  // Step 1. Construct python call stack string
  if (callstack) {
    if (FLAGS_call_stack_level > 1) {
      sout << "\n\n  Compile Traceback (most recent call last):";
    } else {
      sout << "In user code:\n";
    }
    for (auto &line : *callstack) {
      sout << "\n  " << line;
    }
  }
  VLOG(1) << exception->error_str();
  // Step 2. Construct final call stack & append error op name
  if (FLAGS_call_stack_level > 1) {
    sout << exception->what();
  } else {
    // If callstack exists, use err_str_ instead sub_err_str_
    if (callstack) {
      sout << "\n\n";
      sout << InsertIndentationIntoEachLine(exception->error_str());
    } else {
      sout << exception->simple_error_str();
    }
  }
  sout << "  [operator < " << type << " > error]";
  exception->set_error_str(sout.str());
}

void InsertCallStackInfoDygraph(
    const std::string &node_name,
    const std::vector<std::string> &forward_callstack_str,
    platform::EnforceNotMet *exception) {
  const std::vector<std::string> *callstack = &forward_callstack_str;
  std::ostringstream sout;
  // Step 1. Construct python call stack string
  if (callstack) {
    if (FLAGS_call_stack_level > 1) {
      sout << "\n\n  Forward Traceback (most recent call last):";
    } else {
      sout << "In user code:\n";
    }
    for (auto &line : *callstack) {
      sout << "\n  " << line;
    }
  }
  VLOG(1) << exception->error_str();
  // Step 2. Construct final call stack & append error op name
  if (FLAGS_call_stack_level > 1) {
    sout << exception->what();
  } else {
    // If callstack exists, use err_str_ instead sub_err_str_
    if (callstack) {
      sout << "\n\n";
      sout << InsertIndentationIntoEachLine(exception->error_str());
    } else {
      sout << exception->simple_error_str();
    }
  }
  sout << "  [GradNode < " << node_name << " > error]";
  exception->set_error_str(sout.str());
}

void InsertCallStackInfo(const std::string &type,
                         const std::vector<std::string> &callstack_attr_str,
                         platform::EnforceNotMet *exception) {
  const std::vector<std::string> *callstack = &callstack_attr_str;
  std::ostringstream sout;
  // Step 1. Construct python call stack string
  if (callstack) {
    if (FLAGS_call_stack_level > 1) {
      sout << "\n\n  Compile Traceback (most recent call last):";
    } else {
      sout << "In user code:\n";
    }
    for (auto &line : *callstack) {
      sout << "\n  " << line;
    }
  }
  VLOG(1) << exception->error_str();
  // Step 2. Construct final call stack & append error op name
  if (FLAGS_call_stack_level > 1) {
    sout << exception->what();
  } else {
    // If callstack exists, use err_str_ instead sub_err_str_
    if (callstack) {
      sout << "\n\n";
      sout << InsertIndentationIntoEachLine(exception->error_str());
    } else {
      sout << exception->simple_error_str();
    }
  }
  sout << "  [operator < " << type << " > error]";
  exception->set_error_str(sout.str());
}

void AppendErrorOpHint(const std::string &type,
                       platform::EnforceNotMet *exception) {
  std::ostringstream sout;
  sout << exception->what();
  sout << "  [operator < " << type << " > error]";
  exception->set_error_str(sout.str());
}

}  // namespace paddle::framework
