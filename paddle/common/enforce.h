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

#ifdef __GNUC__
#include <cxxabi.h>  // for __cxa_demangle
#endif               // __GNUC__
#include <exception>
#include <iostream>
#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/utils/test_macros.h"

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

// msvc conflict logging with windows.h
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "paddle/common/type_traits.h"
#include "paddle/utils/string/printf.h"
#include "paddle/utils/string/to_string.h"
#include "paddle/utils/variant.h"

namespace common {
#ifdef __GNUC__
inline std::string demangle(std::string name) {
  int status = -4;  // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
}
#else
inline std::string demangle(std::string name) { return name; }
#endif

namespace enforce {

TEST_API void SkipPaddleFatal(bool skip = true);
TEST_API bool IsPaddleFatalSkip();

namespace details {

class PaddleFatalGuard {
 public:
  PaddleFatalGuard() : skip_paddle_fatal_(IsPaddleFatalSkip()) {
    if (!skip_paddle_fatal_) SkipPaddleFatal(true);
  }
  ~PaddleFatalGuard() {
    if (!skip_paddle_fatal_) SkipPaddleFatal(false);
  }

 private:
  bool skip_paddle_fatal_;
};
template <typename T>
struct CanToString {
 private:
  using YesType = uint8_t;
  using NoType = uint16_t;

  template <typename U>
  static YesType Check(decltype(std::cout << std::declval<U>())) {
    return 0;
  }

  template <typename U>
  static NoType Check(...) {
    return 0;
  }

 public:
  static constexpr bool kValue =
      std::is_same<YesType, decltype(Check<T>(std::cout))>::value;
};
template <bool kCanToString /* = true */>
struct BinaryCompareMessageConverter {
  template <typename T>
  static std::string Convert(const char* expression, const T& value) {
    return expression + std::string(":") + paddle::string::to_string(value);
  }
};

template <>
struct BinaryCompareMessageConverter<false> {
  template <typename T>
  static const char* Convert(const char* expression, const T& value UNUSED) {
    return expression;
  }
};

// Note: This Macro can only be used within enforce.h
#define __THROW_ERROR_INTERNAL__(__ERROR_SUMMARY) \
  do {                                            \
    HANDLE_THE_ERROR                              \
    throw ::common::enforce::EnforceNotMet(       \
        __ERROR_SUMMARY, __FILE__, __LINE__);     \
    END_HANDLE_THE_ERROR                          \
  } while (0)

}  // namespace details

TEST_API int GetCallStackLevel();
TEST_API std::string SimplifyErrorTypeFormat(const std::string& str);
TEST_API std::string GetCurrentTraceBackString(bool for_signal = false);
template <typename StrType>
static std::string GetErrorSummaryString(StrType&& what,
                                         const char* file,
                                         int line) {
  std::ostringstream sout;
  if (GetCallStackLevel() > 1) {
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";
  }
  sout << paddle::string::Sprintf(
              "%s (at %s:%d)", std::forward<StrType>(what), file, line)
       << std::endl;
  return sout.str();
}

template <typename StrType>
static std::string GetTraceBackString(StrType&& what,
                                      const char* file,
                                      int line) {
  if (GetCallStackLevel() > 1) {
    // FLAGS_call_stack_level>1 means showing c++ call stack
    return ::common::enforce::GetCurrentTraceBackString() +
           GetErrorSummaryString(what, file, line);
  } else {
    return GetErrorSummaryString(what, file, line);
  }
}

struct EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(std::exception_ptr e, const char* file, int line) {
    try {
      std::rethrow_exception(e);
    } catch (EnforceNotMet& e) {
      code_ = e.code();
      err_str_ = GetTraceBackString(e.what(), file, line);
      simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    } catch (std::exception& e) {
      err_str_ = GetTraceBackString(e.what(), file, line);
      simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }
  }

  EnforceNotMet(const std::string& str, const char* file, int line)
      : err_str_(GetTraceBackString(str, file, line)) {
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }

  EnforceNotMet(const common::ErrorSummary& error, const char* file, int line)
      : code_(error.code()),
        err_str_(GetTraceBackString(error.to_string(), file, line)) {
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }

  const char* what() const noexcept override {
    if (GetCallStackLevel() > 1) {
      return err_str_.c_str();
    } else {
      return simple_err_str_.c_str();
    }
  }

  common::ErrorCode code() const { return code_; }

  const std::string& error_str() const { return err_str_; }

  const std::string& simple_error_str() const { return simple_err_str_; }

  void set_error_str(std::string str) {
    if (GetCallStackLevel() > 1) {
      err_str_ = str;
    } else {
      simple_err_str_ = str;
    }
  }

  ~EnforceNotMet() override = default;

 private:
  // Used to determine the final type of exception thrown
  common::ErrorCode code_ = common::ErrorCode::LEGACY;
  // Complete error message
  // e.g. InvalidArgumentError: ***
  std::string err_str_;
  // Simple error message used when no C++ stack and python compile stack
  // e.g. (InvalidArgument) ***
  std::string simple_err_str_;

  details::PaddleFatalGuard paddle_fatal_guard_;
};
/** HELPER MACROS AND FUNCTIONS **/
#ifndef PADDLE_MAY_THROW
#define PADDLE_MAY_THROW noexcept(false)
#endif

#if defined _WIN32 && defined PADDLE_ON_INFERENCE && defined PADDLE_NO_PYTHON
#define HANDLE_THE_ERROR try {
#define END_HANDLE_THE_ERROR            \
  }                                     \
  catch (const std::exception& e) {     \
    std::cout << e.what() << std::endl; \
    throw;                              \
  }
#else
#define HANDLE_THE_ERROR
#define END_HANDLE_THE_ERROR
#endif
template <typename T>
inline constexpr bool IsArithmetic() {
  return std::is_arithmetic<T>::value;
}

template <typename T1, typename T2, bool kIsArithmetic /* = true */>
struct TypeConverterImpl {
  using Type1 = typename paddle::common_type<T1, T2>::type;
  using Type2 = Type1;
};

template <typename T1, typename T2>
struct TypeConverterImpl<T1, T2, false> {
  using Type1 = T1;
  using Type2 = T2;
};

template <typename T1, typename T2>
struct TypeConverter {
  static constexpr bool kIsArithmetic =
      IsArithmetic<T1>() && IsArithmetic<T2>();
  using Type1 = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type1;
  using Type2 = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type2;
};

template <typename T1, typename T2>
using CommonType1 = typename std::add_lvalue_reference<
    typename std::add_const<typename TypeConverter<T1, T2>::Type1>::type>::type;

template <typename T1, typename T2>
using CommonType2 = typename std::add_lvalue_reference<
    typename std::add_const<typename TypeConverter<T1, T2>::Type2>::type>::type;

#define PADDLE_THROW(...)                                         \
  do {                                                            \
    HANDLE_THE_ERROR                                              \
    throw ::common::enforce::EnforceNotMet(                       \
        ::common::ErrorSummary(__VA_ARGS__), __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                          \
  } while (0)

#define PADDLE_FATAL(...)                                          \
  if (!::common::enforce::IsPaddleFatalSkip()) {                   \
    auto info = ::common::enforce::EnforceNotMet(                  \
        paddle::string::Sprintf(__VA_ARGS__), __FILE__, __LINE__); \
    std::cerr << info.what() << std::endl;                         \
    std::abort();                                                  \
  }

#define __PADDLE_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)         \
  do {                                                                         \
    auto __val1 = (__VAL1);                                                    \
    auto __val2 = (__VAL2);                                                    \
    using __TYPE1__ = decltype(__val1);                                        \
    using __TYPE2__ = decltype(__val2);                                        \
    using __COMMON_TYPE1__ =                                                   \
        ::common::enforce::CommonType1<__TYPE1__, __TYPE2__>;                  \
    using __COMMON_TYPE2__ =                                                   \
        ::common::enforce::CommonType2<__TYPE1__, __TYPE2__>;                  \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP(        \
        static_cast<__COMMON_TYPE2__>(__val2));                                \
    if (UNLIKELY(!__is_not_error)) {                                           \
      auto __summary__ = ::common::ErrorSummary(__VA_ARGS__);                  \
      constexpr bool __kCanToString__ =                                        \
          ::common::details::CanToString<__TYPE1__>::kValue &&                 \
          ::common::details::CanToString<__TYPE2__>::kValue;                   \
      auto __message__ = ::paddle::string::Sprintf(                            \
          "%s\n  [Hint: Expected %s " #__CMP                                   \
          " %s, but received %s " #__INV_CMP " %s.]",                          \
          __summary__.error_message(),                                         \
          #__VAL1,                                                             \
          #__VAL2,                                                             \
          ::common::details::BinaryCompareMessageConverter<                    \
              __kCanToString__>::Convert(#__VAL1, __val1),                     \
          ::common::details::BinaryCompareMessageConverter<                    \
              __kCanToString__>::Convert(#__VAL2, __val2));                    \
      __THROW_ERROR_INTERNAL__(                                                \
          ::common::ErrorSummary(__summary__.code(), std::move(__message__))); \
    }                                                                          \
  } while (0)

#define PADDLE_ENFORCE_NOT_NULL(__VAL, ...)                                    \
  do {                                                                         \
    if (UNLIKELY(nullptr == (__VAL))) {                                        \
      auto __summary__ = ::common::ErrorSummary(__VA_ARGS__);                  \
      auto __message__ = ::paddle::string::Sprintf(                            \
          "%s\n  [Hint: " #__VAL " should not be null.]",                      \
          __summary__.error_message());                                        \
      __THROW_ERROR_INTERNAL__(                                                \
          ::common::ErrorSummary(__summary__.code(), std::move(__message__))); \
    }                                                                          \
  } while (0)

#define PADDLE_ENFORCE_EQ(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, ==, !=, __VA_ARGS__)
#define PADDLE_ENFORCE_NE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, !=, ==, __VA_ARGS__)
#define PADDLE_ENFORCE_GT(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, >, <=, __VA_ARGS__)
#define PADDLE_ENFORCE_GE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, >=, <, __VA_ARGS__)
#define PADDLE_ENFORCE_LT(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, <, >=, __VA_ARGS__)
#define PADDLE_ENFORCE_LE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, <=, >, __VA_ARGS__)

TEST_API bool RegisterLogSimplyStr(const std::string& type,
                                   const std::string& simply);
template <typename T>
class LogSimplyStrRegistrar {
 public:
  static bool success;
};

#define REGISTER_LOG_SIMPLY_STR(Type)                            \
  template <>                                                    \
  bool ::common::enforce::LogSimplyStrRegistrar<Type>::success = \
      ::common::enforce::RegisterLogSimplyStr(                   \
          ::common::demangle(typeid(Type).name()), #Type);
}  // namespace enforce
using namespace enforce;  // NOLINT
}  // namespace common

// TODO(zhangbopd): This is a copy from pir, and should be removed after merge
// this into common enforce namespace above.
template <typename T>
inline bool is_error(const T& stat) {
  return !stat;
}

namespace pir {
#define IR_THROW(...) PADDLE_THROW(common::errors::Fatal(__VA_ARGS__))
}  // namespace pir
