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
#include <vector>
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {

void InsertCallStackInfo(const std::string &type, const AttributeMap &attrs,
                         platform::EnforceNotMet *exception) {
  if (attrs.count("sub_block") != 0) {
    return;
  }

  const std::vector<std::string> *callstack = nullptr;
  auto iter = attrs.find(OpProtoAndCheckerMaker::OpCreationCallstackAttrName());
  if (iter != attrs.end()) {
    callstack = &BOOST_GET_CONST(std::vector<std::string>, iter->second);
    if (callstack->empty()) callstack = nullptr;
  }

  std::ostringstream sout;
  // Step 1. Construct python call stack string
  if (callstack) {
    sout << "\n\n  Compile Traceback (most recent call last):\n";
    for (auto &line : *callstack) {
      sout << "  " << line;
    }
  }
  // Step 2. Construct final call stack & append error op name
  sout << exception->err_str_;
  sout << "  [operator < " << type << " > error]";
  exception->err_str_ = sout.str();
}

void AppendErrorOpHint(const std::string &type,
                       platform::EnforceNotMet *exception) {
  std::ostringstream sout;
  sout << exception->err_str_;
  sout << "  [operator < " << type << " > error]";
  exception->err_str_ = sout.str();
}

}  // namespace framework
}  // namespace paddle
