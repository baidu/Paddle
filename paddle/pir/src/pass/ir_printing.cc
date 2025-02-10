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

#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_instrumentation.h"
#include "paddle/pir/include/pass/pass_manager.h"

namespace pir {

namespace {
void PrintIR(Operation *op, bool print_module, std::ostream &os) {
  if (!print_module) {
    op->Print(os << "\n");
    return;
  }

  auto *program = op->GetParentProgram();
  program->Print(os);
}
}  // namespace

class IRPrinting : public PassInstrumentation {
 public:
  explicit IRPrinting(std::unique_ptr<PassManager::IRPrinterOption> option)
      : option_(std::move(option)) {}

  ~IRPrinting() override = default;

  void RunBeforePass(Pass *pass, Operation *op) override {
    if (option_->print_on_change()) {
      // TODO(liuyuanle): support print on change
    }

    option_->PrintBeforeIfEnabled(pass, op, [&]() {
      std::ostringstream oss;
      std::string header =
          "IRPrinting on " + op->name() + " before " + pass->name() + " pass";
      detail::PrintHeader(header, oss);
      PrintIR(op, option_->print_module(), oss);
      oss << "\n";
      std::cout << oss.str() << std::endl;
    });
  }

  void RunAfterPass(Pass *pass, Operation *op) override {
    if (option_->print_on_change()) {
      // TODO(liuyuanle): support print on change
    }

    option_->PrintAfterIfEnabled(pass, op, [&]() {
      std::ostringstream oss;
      std::string header =
          "IRPrinting on " + op->name() + " after " + pass->name() + " pass";
      detail::PrintHeader(header, oss);
      PrintIR(op, option_->print_module(), oss);
      oss << "\n";
      std::cout << oss.str() << std::endl;

      FILE *fp = fopen(pass->name().c_str(), "w");
      if (fp == nullptr) {
        return;
      }
      fwrite(oss.str().c_str(), 1, oss.str().length(), fp);
      fclose(fp);
    });
  }

 private:
  std::unique_ptr<PassManager::IRPrinterOption> option_;

  // TODO(liuyuanle): Add IRFingerPrint to support print on change.
};

void PassManager::EnableIRPrinting(std::unique_ptr<IRPrinterOption> option) {
  AddInstrumentation(std::make_unique<IRPrinting>(std::move(option)));
}

}  // namespace pir
