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

#include "paddle/common/flags.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace paddle::flags {

std::stringstream& ErrorStream() {
  static std::stringstream err_ss;
  return err_ss;
}

inline void exit_with_errors() {
  std::cerr << ErrorStream().str();
  exit(-1);
}

#define LOG_FLAG_ERROR(message)                                             \
  ErrorStream() << "paddle flags error: " << message << " (at " << __FILE__ \
                << ":" << __LINE__ << ")" << std::endl

#define LOG_FLAG_FATAL_ERROR(message) \
  LOG_FLAG_ERROR(message);            \
  exit_with_errors()

enum class FlagType : uint8_t {
  BOOL = 0,
  INT32 = 1,
  UINT32 = 2,
  INT64 = 3,
  UINT64 = 4,
  DOUBLE = 5,
  STRING = 6,
  UNDEFINED = 7,
};

class Flag {
 public:
  Flag(std::string name,
       std::string description,
       std::string file,
       FlagType type,
       const void* default_value,
       void* value)
      : name_(name),
        description_(description),
        file_(file),
        type_(type),
        default_value_(default_value),
        value_(value) {}
  ~Flag() = default;

  // Summary: --name_: type_, description_ (default: default_value_)
  std::string Summary() const;

  void SetValueFromString(const std::string& value);

 private:
  friend class FlagRegistry;

  const std::string name_;
  const std::string description_;
  const std::string file_;
  const FlagType type_;
  const void* default_value_;
  void* value_;
};

class FlagRegistry {
 public:
  static FlagRegistry* Instance() {
    static FlagRegistry* global_registry_ = new FlagRegistry();
    return global_registry_;
  }

  void RegisterFlag(Flag* flag);

  bool SetFlagValue(const std::string& name, const std::string& value);

  bool HasFlag(const std::string& name) const;

  void PrintAllFlagHelp(std::ostream& os) const;

 private:
  FlagRegistry() = default;

  std::map<std::string, Flag*> flags_;

  struct FlagCompare {
    bool operator()(const Flag* flag1, const Flag* flag2) const {
      return flag1->name_ < flag2->name_;
    }
  };

  std::map<std::string, std::set<Flag*, FlagCompare>> flags_by_file_;

  std::mutex mutex_;
};

template <typename T>
struct FlagTypeTraits {
  static constexpr FlagType Type = FlagType::UNDEFINED;
};

#define DEFINE_FLAG_TYPE_TRAITS(type, flag_type) \
  template <>                                    \
  struct FlagTypeTraits<type> {                  \
    static constexpr FlagType Type = flag_type;  \
  }

DEFINE_FLAG_TYPE_TRAITS(bool, FlagType::BOOL);
DEFINE_FLAG_TYPE_TRAITS(int32_t, FlagType::INT32);
DEFINE_FLAG_TYPE_TRAITS(uint32_t, FlagType::UINT32);
DEFINE_FLAG_TYPE_TRAITS(int64_t, FlagType::INT64);
DEFINE_FLAG_TYPE_TRAITS(uint64_t, FlagType::UINT64);
DEFINE_FLAG_TYPE_TRAITS(double, FlagType::DOUBLE);
DEFINE_FLAG_TYPE_TRAITS(std::string, FlagType::STRING);

#undef DEFINE_FLAG_TYPE_TRAITS

template <typename T>
FlagRegisterer::FlagRegisterer(std::string name,
                               std::string help,
                               std::string file,
                               const T* default_value,
                               T* value) {
  FlagType type = FlagTypeTraits<T>::Type;
  Flag* flag = new Flag(name, help, file, type, default_value, value);
  FlagRegistry::Instance()->RegisterFlag(flag);
}

// Instantiate FlagRegisterer for supported types.
#define INSTANTIATE_FLAG_REGISTERER(type)                                     \
  template TEST_API FlagRegisterer::FlagRegisterer(std::string name,          \
                                                   std::string help,          \
                                                   std::string file,          \
                                                   const type* default_value, \
                                                   type* value)

INSTANTIATE_FLAG_REGISTERER(bool);
INSTANTIATE_FLAG_REGISTERER(int32_t);
INSTANTIATE_FLAG_REGISTERER(uint32_t);
INSTANTIATE_FLAG_REGISTERER(int64_t);
INSTANTIATE_FLAG_REGISTERER(uint64_t);
INSTANTIATE_FLAG_REGISTERER(double);
INSTANTIATE_FLAG_REGISTERER(std::string);

#undef INSTANTIATE_FLAG_REGISTERER

std::string FlagType2String(FlagType type) {
  switch (type) {
    case FlagType::BOOL:
      return "bool";
    case FlagType::INT32:
      return "int32";
    case FlagType::UINT32:
      return "uint32";
    case FlagType::INT64:
      return "int64";
    case FlagType::UINT64:
      return "uint64";
    case FlagType::DOUBLE:
      return "double";
    case FlagType::STRING:
      return "string";
    default:
      return "undefined";
  }
}

std::string Value2String(const void* value, FlagType type) {
  switch (type) {
    case FlagType::BOOL: {
      const bool* val = static_cast<const bool*>(value);
      return *val ? "true" : "false";
    }
    case FlagType::INT32: {
      const int32_t* val = static_cast<const int32_t*>(value);
      return std::to_string(*val);
    }
    case FlagType::UINT32: {
      const uint32_t* val = static_cast<const uint32_t*>(value);
      return std::to_string(*val);
    }
    case FlagType::INT64: {
      const int64_t* val = static_cast<const int64_t*>(value);
      return std::to_string(*val);
    }
    case FlagType::UINT64: {
      const uint64_t* val = static_cast<const uint64_t*>(value);
      return std::to_string(*val);
    }
    case FlagType::DOUBLE: {
      const double* val = static_cast<const double*>(value);
      return std::to_string(*val);
    }
    case FlagType::STRING: {
      const std::string* val = static_cast<const std::string*>(value);
      return *val;
    }
    default:
      LOG_FLAG_FATAL_ERROR("flag type is undefined.");
      return "";
  }
}

std::string Flag::Summary() const {
  return "--" + name_ + ": " + FlagType2String(type_) + ", " + description_ +
         " (default: " + Value2String(default_value_, type_) + ")";
}

void Flag::SetValueFromString(const std::string& value) {
  try {
    switch (type_) {
      case FlagType::BOOL: {
        bool* val = static_cast<bool*>(value_);
        if (value == "true" || value == "True" || value == "TRUE" ||
            value == "1") {
          *val = true;
        } else if (value == "false" || value == "False" || value == "FALSE" ||
                   value == "0") {
          *val = false;
        } else {
          throw std::invalid_argument(
              ", please use [true, True, TRUE, 1] or [false, False, FALSE, "
              "0].");
        }
        break;
      }
      case FlagType::INT32: {
        int32_t* val = static_cast<int32_t*>(value_);
        *val = std::stoi(value);
        break;
      }
      case FlagType::UINT32: {
        uint32_t* val = static_cast<uint32_t*>(value_);
        *val = std::stoul(value);
        break;
      }
      case FlagType::INT64: {
        int64_t* val = static_cast<int64_t*>(value_);
        *val = std::stoll(value);
        break;
      }
      case FlagType::UINT64: {
        uint64_t* val = static_cast<uint64_t*>(value_);
        *val = std::stoull(value);
        break;
      }
      case FlagType::DOUBLE: {
        double* val = static_cast<double*>(value_);
        *val = std::stod(value);
        break;
      }
      case FlagType::STRING: {
        std::string* val = static_cast<std::string*>(value_);
        *val = value;
        break;
      }
      default: {
        LOG_FLAG_FATAL_ERROR("flag type is undefined.");
      }
    }
  } catch (const std::exception& e) {
    std::string error_msg = "value: \"" + value + "\" is invalid for " +
                            FlagType2String(type_) + " flag \"" + name_ + "\"";
    if (type_ == FlagType::BOOL) {
      error_msg += e.what();
    } else {
      error_msg += ".";
    }
    LOG_FLAG_ERROR(error_msg);
  }
}

void FlagRegistry::RegisterFlag(Flag* flag) {
  auto iter = flags_.find(flag->name_);
  if (iter != flags_.end()) {
    std::string error_msg = "flag multiple definition, flag \"";
    error_msg += flag->name_;
    error_msg += "\" was defined both in ";
    error_msg += iter->second->file_;
    error_msg += " and ";
    error_msg += flag->file_;
    LOG_FLAG_FATAL_ERROR(error_msg);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    flags_[flag->name_] = flag;
    flags_by_file_[flag->file_].insert(flag);
  }
}

bool FlagRegistry::SetFlagValue(const std::string& name,
                                const std::string& value) {
  if (HasFlag(name)) {
    std::lock_guard<std::mutex> lock(mutex_);
    flags_[name]->SetValueFromString(value);
    return true;
  } else {
    LOG_FLAG_ERROR("illegal SetFlagValue, flag \"" + name +
                   "\" is not defined.");
    return false;
  }
}

bool FlagRegistry::HasFlag(const std::string& name) const {
  return flags_.find(name) != flags_.end();
}

void FlagRegistry::PrintAllFlagHelp(std::ostream& os) const {
  for (const auto& iter : flags_by_file_) {
    os << std::endl << "Flags defined in " << iter.first << ":" << std::endl;
    for (const auto& flag : iter.second) {
      os << "  " << flag->Summary() << std::endl;
    }
  }
  os << std::endl;
}

void PrintAllFlagHelp(bool to_file, const std::string& file_path) {
  if (to_file) {
    std::ofstream fout(file_path);
    FlagRegistry::Instance()->PrintAllFlagHelp(fout);
  } else {
    FlagRegistry::Instance()->PrintAllFlagHelp(std::cout);
  }
}

bool SetFlagValue(const std::string& name, const std::string& value) {
  return FlagRegistry::Instance()->SetFlagValue(name, value);
}

bool FindFlag(const std::string& name) {
  return FlagRegistry::Instance()->HasFlag(name);
}

bool GetValueFromEnv(const std::string& name, std::string* value) {
  const char* env_var = std::getenv(name.c_str());
  if (env_var == nullptr) {
    return false;
  }
  *value = std::string(env_var);
  return true;
}

/**
 * @brief Set flags from environment variables.
 *
 * It receives a list of flags name, and will find the corresponding environment
 * variables named "FLAGS_name", if found, it will set the environment variable
 * values to the flags. If error_fatal is true, the program will exit when the
 * environment variable is not set or the flag is not defined, that is the same
 * effect as using commandline argument "--fromenv=var_name1,var_name2,...".
 * Otherwise, the errors above will be ignored, that is the same effect as using
 * commandline argument "--tryfromenv=var_name1,var_name2,...".
 */

void SetFlagsFromEnv(const std::vector<std::string>& flags, bool error_fatal) {
  bool success = true;
  for (const std::string& flag_name : flags) {
    std::string env_var_name = std::string("FLAGS_") + flag_name;
    std::string env_var_value;
    if (GetValueFromEnv(env_var_name, &env_var_value)) {
      success =
          FlagRegistry::Instance()->SetFlagValue(flag_name, env_var_value);
    } else if (error_fatal) {
      LOG_FLAG_ERROR("environment variable \"" + env_var_name +
                     "\" is not set.");
      success = false;
    }
  }
  if (error_fatal && !success) {
    exit_with_errors();
  }
}

static bool allow_undefined_flags = false;

TEST_API void AllowUndefinedFlags() { allow_undefined_flags = true; }

TEST_API void ParseCommandLineFlags(int* pargc, char*** pargv) {
  assert(*pargc > 0);
  size_t argv_num = *pargc - 1;
  std::vector<std::string> argvs(*pargv + 1, *pargv + *pargc);

  std::string arg_format_help =
      "please follow the formats: \"--help(h)\", \"--name=value\""
      " or \"--name value\".";
  for (size_t i = 0; i < argv_num; i++) {
    const std::string& argv = argvs[i];

    if (argv.empty()) {
      continue;
    }

    if (argv.size() < 2 || argv[0] != '-') {
      LOG_FLAG_FATAL_ERROR("invalid commandline argument: \"" + argv + "\", " +
                           arg_format_help);
    }

    // parse arg name and value
    size_t hyphen_num = argv[1] == '-' ? 2 : 1;
    std::string name, value;
    size_t split_pos = argv.find('=');
    if (split_pos == std::string::npos) {
      // the argv format is "--name" or "--name value"
      name = argv.substr(hyphen_num);
      if (name.empty()) {
        std::string error_msg = "invalid commandline argument: \"";
        error_msg += argv;
        error_msg += "\", ";
        error_msg += arg_format_help;
        LOG_FLAG_FATAL_ERROR(error_msg);
      }

      // print help message
      if (name == "help" || name == "h") {
        FlagRegistry::Instance()->PrintAllFlagHelp(std::cout);
        exit(1);
      }

      // get the value from next argv.
      if (++i == argv_num) {
        std::string error_msg = "expected value of flag \"";
        error_msg += name;
        error_msg += "\" but found none.";
        LOG_FLAG_FATAL_ERROR(error_msg);
      } else {
        value = argvs[i];
      }
    } else {
      // the argv format is "--name=value"
      if (split_pos == hyphen_num || split_pos == argv.size() - 1) {
        std::string error_msg = "invalid commandline argument: \"";
        error_msg += argv;
        error_msg += "\", ";
        error_msg += arg_format_help;
        LOG_FLAG_FATAL_ERROR(error_msg);
      }
      name = argv.substr(hyphen_num, split_pos - hyphen_num);
      value = argv.substr(split_pos + 1);
    }

    // special case for flag value enclosed in ""
    if (value[0] == '"') {
      value = value.substr(1);
      if (value.back() == '"') {
        value.pop_back();
      } else {
        while (i < argv_num) {
          value += " ";
          value += argvs[++i];
          if (value.back() == '"') {
            break;
          }
        }
        if (value.back() == '"') {
          value.pop_back();
        } else {
          std::string error_msg = "unexpected end of flag \"";
          error_msg += name;
          error_msg += "\" value while looking for matching `\"'";
          LOG_FLAG_FATAL_ERROR(error_msg);
        }
      }
    }

    if (name == "fromenv" || name == "tryfromenv") {
      // Value of --fromenv or --tryfromenv should be
      // a comma separated list of env var names.
      std::vector<std::string> env_flag_names;
      for (size_t start_pos = 0, end_pos = 0;
           start_pos < value.size() && end_pos != std::string::npos;
           start_pos = end_pos + 1) {
        end_pos = value.find(',', start_pos);
        env_flag_names.push_back(value.substr(start_pos, end_pos - start_pos));
      }
      if (name == "fromenv") {
        SetFlagsFromEnv(env_flag_names, true);
      } else {
        SetFlagsFromEnv(env_flag_names, false);
      }
      continue;
    }

    FlagRegistry::Instance()->SetFlagValue(name, value);
  }
  if (!allow_undefined_flags && !ErrorStream().str().empty()) {
    exit_with_errors();
  }
}

template <typename T>
T GetFromEnv(const std::string& name, const T& default_val) {
  std::string value_str;
  if (GetValueFromEnv(name, &value_str)) {
    T value;
    FlagType type = FlagTypeTraits<T>::Type;
    Flag flag("tmp_" + name, "", "", type, &value, &value);
    flag.SetValueFromString(value_str);
    if (!ErrorStream().str().empty()) {
      ErrorStream().str("");
      std::string error_msg = "value \"";
      error_msg += value_str;
      error_msg += "\" of environment";
      error_msg += "variable \"";
      error_msg += name;
      error_msg += "\" is invalid when ";
      error_msg += "using GetFromEnv with ";
      error_msg += FlagType2String(type);
      error_msg += " type.";
      LOG_FLAG_FATAL_ERROR(error_msg);
    }
    return value;
  } else {
    return default_val;
  }
}

#define INSTANTIATE_GET_FROM_ENV(type)                       \
  template TEST_API type GetFromEnv(const std::string& name, \
                                    const type& default_val)

INSTANTIATE_GET_FROM_ENV(bool);
INSTANTIATE_GET_FROM_ENV(int32_t);
INSTANTIATE_GET_FROM_ENV(uint32_t);
INSTANTIATE_GET_FROM_ENV(int64_t);
INSTANTIATE_GET_FROM_ENV(uint64_t);
INSTANTIATE_GET_FROM_ENV(double);
INSTANTIATE_GET_FROM_ENV(std::string);

#undef INSTANTIATE_GET_FROM_ENV

}  // namespace paddle::flags
