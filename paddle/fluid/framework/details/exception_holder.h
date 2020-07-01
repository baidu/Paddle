// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {

class ExceptionHolder {
 public:
  void Catch(std::exception_ptr eptr) {
    try {
      std::rethrow_exception(eptr);
    } catch (memory::allocation::BadAlloc& exp) {
      Catch(exp);
    } catch (platform::EOFException& exp) {
      Catch(exp);
    } catch (platform::EnforceNotMet& exp) {
      Catch(exp);
    } catch (std::exception& ex) {
      Catch(ex);
    } catch (...) {
      LOG(FATAL) << "Unknown exception caught.";
    }
  }

  bool IsCaught() const {
    std::lock_guard<std::mutex> lock(mu_);
    return exception_.get() != nullptr;
  }

  void ReThrow() {
    std::lock_guard<std::mutex> lock(mu_);
    switch (type_) {
      case kNone:
        break;
      case kEnforceNotMet: {
        auto e = *static_cast<platform::EnforceNotMet*>(exception_.get());
        throw e;
      }
      case kEOF: {
        auto e = *static_cast<platform::EOFException*>(exception_.get());
        throw e;
      }
      case kBadAlloc: {
        auto e = *static_cast<paddle::memory::allocation::BadAlloc*>(
            exception_.get());
        throw e;
      }
      case kBaseException: {
        auto e = *static_cast<std::exception*>(exception_.get());
        throw e;
      }
    }
    ClearImpl();
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    ClearImpl();
  }

  std::string Type() {
    std::lock_guard<std::mutex> lock(mu_);
    switch (type_) {
      case kNone:
        return "None";
      case kEnforceNotMet: {
        return "EnforceNotMet";
      }
      case kEOF: {
        return "EOF";
      }
      case kBadAlloc: {
        return "BadAlloc";
      }
      case kBaseException: {
        return "BaseException";
      }
    }
    return "unknown";
  }

 private:
  void ClearImpl() {
    exception_.reset();
    type_ = kNone;
  }

  void Catch(const platform::EnforceNotMet& exp) {
    std::lock_guard<std::mutex> lock(mu_);
    exception_.reset(new platform::EnforceNotMet(exp));
    type_ = kEnforceNotMet;
  }

  void Catch(const memory::allocation::BadAlloc& exp) {
    std::lock_guard<std::mutex> lock(mu_);
    // BadAlloc have the highest priority
    if (exception_.get() != nullptr) {
      VLOG(2) << "exception is reset by BadAlloc, the message is"
              << exception_->what();
    }
    exception_.reset(new paddle::memory::allocation::BadAlloc(exp));
    type_ = kBadAlloc;
  }

  void Catch(const platform::EOFException& exp) {
    std::lock_guard<std::mutex> lock(mu_);
    // EOFException will not cover up existing EnforceNotMet.
    if (exception_.get() == nullptr) {
      exception_.reset(new platform::EOFException(exp));
      type_ = kEOF;
    } else {
      VLOG(2) << "EOFException is skip, the error message of EOFException is "
              << exception_->what();
    }
  }

  void Catch(const std::exception& exp) {
    std::lock_guard<std::mutex> lock(mu_);
    // std::exception will not cover anything
    if (exception_.get() == nullptr) {
      exception_.reset(new std::exception(exp));
      type_ = kBaseException;
    }
  }

  enum ExceptionType { kNone, kEnforceNotMet, kEOF, kBadAlloc, kBaseException };
  ExceptionType type_{kNone};

  std::unique_ptr<std::exception> exception_;
  mutable std::mutex mu_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
