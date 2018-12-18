/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/scope.h"

#include <memory>  // for unique_ptr
#include <queue>
#include <set>
#include <unordered_set>
#include "glog/logging.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/string/printf.h"

DEFINE_bool(benchmark, false,
            "Doing memory benchmark. It will make deleting scope synchronized, "
            "and add some memory usage logs."
            "Default cuda is asynchronous device, set to True will"
            "force op run in synchronous mode.");

DEFINE_bool(
    eager_delete_scope, true,
    "Delete local scope eagerly. It will reduce GPU memory usage but "
    "slow down the destruction of variables.(around 1% performance harm)");

DEFINE_double(
    eager_delete_tensor_gb, -1.0,
    "Memory size threshold (GB) when the garbage collector clear tensors."
    "Disabled when this value is less than 0");

DEFINE_bool(fast_eager_deletion_mode, false,
            "Fast eager deletion mode. If enabled, memory would release "
            "immediately without waiting GPU kernel ends.");

namespace paddle {
namespace framework {

#ifdef PADDLE_ON_INFERENCE
Variable* Scope::FindVarLocally(const std::string& name) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) return it->second.get();
  return nullptr;
}

Variable* Scope::VarInternal(const std::string& name) {
  auto* v = FindVarLocally(name);
  if (v != nullptr) return v;

  v = new Variable();
  vars_[name].reset(v);
  VLOG(3) << "Create variable " << name;
  return v;
}

void Scope::RenameInternal(const std::string& origin_name,
                           const std::string& new_name) const {
  auto origin_it = vars_.find(origin_name);
  PADDLE_ENFORCE(origin_it != vars_.end(),
                 "Cannot find original variable with name %s", origin_name);
  auto new_it = vars_.find(new_name);
  PADDLE_ENFORCE(new_it == vars_.end(),
                 "The variable with name %s is already in the scope", new_name);
  vars_[new_name].reset(origin_it->second.release());
  vars_.erase(origin_it);
}

#else
// concurrent hash map has a different access do find/erase
Variable* Scope::FindVarLocally(const std::string& name) const {
  ScopeHashMap::const_accessor it;
  auto exsits = vars_.find(it, name);
  if (exsits) return it->second.get();
  return nullptr;
}

Variable* Scope::VarInternal(const std::string& name) {
  auto* v = FindVarLocally(name);
  if (v != nullptr) return v;

  ScopeHashMap::accessor it;
  vars_.insert(it,
               std::make_pair(name, std::unique_ptr<Variable>(new Variable())));
  VLOG(3) << "Create variable " << name;
  return it->second.get();
}

void Scope::RenameInternal(const std::string& origin_name,
                           const std::string& new_name) const {
  ScopeHashMap::const_accessor new_it, origin_it;
  auto origin_exsits = vars_.find(origin_it, origin_name);
  PADDLE_ENFORCE(origin_exsits, "Cannot find original variable with name %s",
                 origin_name);
  auto new_exsits = vars_.find(new_it, new_name);
  PADDLE_ENFORCE(new_exsits == false,
                 "The variable with name %s is already in the scope", new_name);
  vars_.insert(
      new_it,
      std::make_pair(new_name, std::unique_ptr<Variable>(new Variable())));
  vars_.erase(origin_it);
}
#endif

int64_t GetEagerDeletionThreshold() {
  return FLAGS_eager_delete_tensor_gb < 0
             ? -1
             : static_cast<int64_t>(FLAGS_eager_delete_tensor_gb *
                                    (static_cast<int64_t>(1) << 30));
}

bool IsFastEagerDeletionModeEnabled() { return FLAGS_fast_eager_deletion_mode; }

Scope::~Scope() { DropKids(); }

Scope& Scope::NewScope() const {
  kids_.push_back(new Scope(this));
  return *kids_.back();
}

Variable* Scope::Var(const std::string& name) { return VarInternal(name); }

Variable* Scope::Var(std::string* name) {
  auto new_name = string::Sprintf("%p.%d", this, vars_.size());
  if (name != nullptr) {
    *name = new_name;
  }
  return VarInternal(new_name);
}

Variable* Scope::FindVar(const std::string& name) const {
  return FindVarInternal(name);
}

void Scope::EraseVars(const std::vector<std::string>& var_names) {
  std::set<std::string> var_set(var_names.begin(), var_names.end());
  for (auto it = var_names.begin(); it != var_names.end(); ++it) {
    vars_.erase(*it);
  }
}

Variable* Scope::FindLocalVar(const std::string& name) const {
  return FindVarLocally(name);
}

const Scope* Scope::FindScope(const Variable* var) const {
  return FindScopeInternal(var);
}

void Scope::DropKids() {
  for (Scope* s : kids_) delete s;
  kids_.clear();
}

bool Scope::HasKid(const Scope* scope) const {
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  return it != this->kids_.end();
}

std::vector<std::string> Scope::LocalVarNames() const {
  std::vector<std::string> known_vars;
  known_vars.reserve(this->vars_.size());
  for (auto& p : vars_) {
    known_vars.emplace_back(p.first);
  }
  return known_vars;
}

void Scope::DeleteScope(Scope* scope) const {
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  PADDLE_ENFORCE(it != this->kids_.end(), "%p Cannot find %p as kid scope",
                 this, scope);
  this->kids_.erase(it);
  // When making memory benchmark on Fluid, we have to delete scope sync.
  if (FLAGS_benchmark || FLAGS_eager_delete_scope) {
    delete scope;
  } else {
    Async([scope] { delete scope; });
  }
}

void Scope::Rename(const std::string& origin_name,
                   const std::string& new_name) const {
  RenameInternal(origin_name, new_name);
}

std::string Scope::Rename(const std::string& origin_name) const {
  auto new_name = string::Sprintf("%p.%d", this, vars_.size());
  RenameInternal(origin_name, new_name);
  return new_name;
}

const Scope* Scope::FindScopeInternal(const Variable* var) const {
  for (auto& kv : vars_) {
    if (kv.second.get() == var) {
      return this;
    }
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindScope(var);
}

Variable* Scope::FindVarInternal(const std::string& name) const {
  auto var = FindVarLocally(name);
  if (var != nullptr) {
    return var;
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
}

std::string GenScopeTreeDebugInfo(Scope* root) {
  std::stringstream os;

  if (!root) return "";

  // level traversal
  std::queue<Scope*> queue;
  queue.push(root);

  std::vector<Scope*> scopes;

  while (!queue.empty()) {
    auto* end = queue.back();
    Scope* q = nullptr;
    while (q != end) {
      q = queue.front();
      queue.pop();
      os << q << " ";
      scopes.push_back(q);

      for (auto* c : q->kids()) {
        queue.push(c);
      }
    }
    // end of a level
    os << "\n------------------------------------------\n";
  }

  os << "\nDetails:\n\n";

  for (Scope* q : scopes) {
    os << "====\n";
    os << q << ":\n";
    for (auto& var : q->LocalVarNames()) {
      os << "  - " << var << "\n";
    }
  }

  return os.str();
}

}  // namespace framework
}  // namespace paddle
