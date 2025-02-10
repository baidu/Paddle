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

#include "paddle/cinn/utils/string.h"

#include <stdarg.h>

#include <cstring>
#include <iomanip>

#include "glog/logging.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace utils {

std::string StringFormat(const std::string &fmt_str, ...) {
  /* Reserve two times as much as the length of the fmt_str */
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(
        new char[n]); /* Wrap the plain char array into the unique_ptr */
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

std::string RemoveSuffix(const std::string &name) {
  std::string res = name;
  while (EndsWith(res, "_outer") || EndsWith(res, "_inner")) {
    res = res.substr(0, res.size() - 6);
  }
  return res;
}

std::string Trim(const std::string &s, const char *empty) {
  if (s.empty()) return s;
  auto start = s.find_first_not_of(empty);
  if (start == std::string::npos) return "";
  auto end = s.find_last_not_of(empty);
  return s.substr(start, end - start + 1);
}

std::string Uppercase(const std::string &x) {
  auto res = x;
  for (auto &c : res) {
    c = toupper(c);
  }
  return res;
}

bool StartsWith(const std::string &x, const std::string &str) {
  return x.find(str) == 0;
}
bool EndsWith(const std::string &x, const std::string &str) {
  if (x.length() >= str.length()) {
    return std::equal(str.rbegin(), str.rend(), x.rbegin());
  }
  return false;
}

std::vector<std::string> Split(const std::string &str,
                               const std::string &splitter) {
  std::vector<std::string> results;
  std::string::size_type pos1, pos2;
  pos2 = str.find(splitter);
  pos1 = 0;
  while (std::string::npos != pos2) {
    results.push_back(str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + splitter.size();
    pos2 = str.find(splitter, pos1);
  }
  if (pos1 != str.length()) {
    results.push_back(str.substr(pos1));
  }
  return results;
}

void Replace(std::string *s, const std::string &from, const std::string &to) {
  size_t pos = 0;
  while ((pos = s->find(from, pos)) != std::string::npos) {
    s->replace(pos, from.size(), to);
    pos += to.length();
  }
}

size_t Count(std::string *s, const std::string &sub) {
  size_t pos = 0;
  size_t times = 0;
  while ((pos = s->find(sub, pos)) != std::string::npos) {
    if ((pos == 0 || !IsPrefix(s->at(pos - 1))) &&
        (pos + sub.length() == s->size() ||
         !IsSuffix(s->at(pos + sub.length())))) {
      pos += sub.length();
      times++;
    } else {
      pos++;
    }
  }
  return times;
}

bool IsPrefix(const char &c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
}

bool IsSuffix(const char &c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_') ||
         (c >= '0' && c <= '9') || (c == '\'');
}

std::string TransValidVarName(std::string name) {
  utils::Replace(&name, ".", "__");
  utils::Replace(&name, "/", "___");
  utils::Replace(&name, "@", "____");
  name.erase(0, name.find_first_not_of("_"));
  return name;
}

std::string Attribute2String(const utils::Attribute &attr) {
  std::stringstream ss;
  if (absl::holds_alternative<bool>(attr)) {
    ss << (absl::get<bool>(attr) ? "True" : "False");
  } else if (absl::holds_alternative<float>(attr)) {
    ss << std::setprecision(std::numeric_limits<float>::max_digits10)
       << std::showpoint << absl::get<float>(attr);
  } else if (absl::holds_alternative<double>(attr)) {
    ss << std::setprecision(std::numeric_limits<double>::max_digits10)
       << std::showpoint << absl::get<double>(attr);
  } else if (absl::holds_alternative<int>(attr)) {
    ss << absl::get<int>(attr);
  } else if (absl::holds_alternative<int64_t>(attr)) {
    ss << absl::get<int64_t>(attr);
  } else if (absl::holds_alternative<std::string>(attr)) {
    ss << "\"" << absl::get<std::string>(attr) << "\"";
  } else if (absl::holds_alternative<std::vector<bool>>(attr)) {
    ss << "[" + cinn::utils::Join(absl::get<std::vector<bool>>(attr), ", ") +
              "]";
  } else if (absl::holds_alternative<std::vector<int>>(attr)) {
    ss << "[" + cinn::utils::Join(absl::get<std::vector<int>>(attr), ", ") +
              "]";
  } else if (absl::holds_alternative<std::vector<int64_t>>(attr)) {
    ss << "[" + cinn::utils::Join(absl::get<std::vector<int64_t>>(attr), ", ") +
              "]";
  } else if (absl::holds_alternative<std::vector<float>>(attr)) {
    ss << "[" + cinn::utils::Join(absl::get<std::vector<float>>(attr), ", ") +
              "]";
  } else if (absl::holds_alternative<std::vector<double>>(attr)) {
    ss << "[" + cinn::utils::Join(absl::get<std::vector<double>>(attr), ", ") +
              "]";
  } else if (absl::holds_alternative<std::vector<std::string>>(attr)) {
    auto attrs = absl::get<std::vector<std::string>>(attr);
    for (auto &str : attrs) {
      str = "\"" + str + "\"";
    }
    ss << "[" + cinn::utils::Join(attrs, ", ") + "]";
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Unknown attribute data type! Please check."));
  }
  return ss.str();
}

}  // namespace utils
}  // namespace cinn
