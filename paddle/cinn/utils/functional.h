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

#pragma once

#include <absl/types/optional.h>

#include <algorithm>
#include <functional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "paddle/common/enforce.h"
namespace cinn {
namespace utils {

template <typename InT, typename OutValT>
std::vector<OutValT> Map(
    const InT &in,
    std::function<OutValT(const typename InT::value_type &)> fn) {
  std::vector<OutValT> res;
  std::transform(in.begin(),
                 in.end(),
                 std::back_inserter(res),
                 [&](const typename InT::value_type &x) { return fn(x); });
  return res;
}

template <typename T>
auto Min(T &&t) {
  return t;
}

template <typename T, typename... Ts>
auto Min(T &&t, Ts &&...ts) {
  return std::min(t, Min(ts...));
}

template <typename T>
auto Max(T &&t) {
  return t;
}

template <typename T, typename... Ts>
auto Max(T &&t, Ts &&...ts) {
  return std::max(t, Max(ts...));
}

template <typename T>
struct IsVector {
  template <typename U>
  static auto infer(U *) -> std::enable_if_t<
      std::is_same<std::vector<typename U::value_type>, U>::value,
      std::true_type>;

  template <typename U>
  static std::false_type infer(...);

  static constexpr bool value =
      decltype(infer<std::decay_t<std::remove_pointer_t<T>>>(nullptr))::value;
};

template <class T>
struct IsString : std::integral_constant<
                      bool,
                      std::is_same<std::string, std::decay_t<T>>::value> {};

template <typename T>
auto Flatten(const absl::optional<std::reference_wrapper<const T>> &c)
    -> std::enable_if_t<std::is_scalar<T>::value || IsString<T>::value,
                        std::vector<T>> {
  return c ? std::vector<T>{c->get()} : std::vector<T>{};
}

template <template <typename...> class C, typename E>
auto Flatten(const absl::optional<std::reference_wrapper<const C<E>>> &c)
    -> std::enable_if_t<std::is_scalar<E>::value &&
                            !IsString<decltype(c->get())>::value,
                        std::vector<E>> {
  return c ? std::vector<E>(c->get().begin(), c->get().end())
           : std::vector<E>{};
}

template <typename T,
          typename E = std::enable_if_t<
              !IsString<T>::value,
              std::decay_t<decltype(*std::declval<const T>().begin())>>>
auto Flatten(const absl::optional<std::reference_wrapper<const T>> &c) {
  absl::optional<std::reference_wrapper<const E>> val;
  if (c && !c->get().empty()) {
    val = *(c->get().begin());
  }

  auto res = Flatten(val);

  if (val) {
    auto it = ++(c->get().begin());
    while (it != c->get().end()) {
      val = *it;
      auto tmp = Flatten(val);
      res.insert(res.end(), tmp.begin(), tmp.end());
      ++it;
    }
  }
  return res;
}

template <typename T>
auto Flatten(const T &v) {
  absl::optional<std::reference_wrapper<const T>> w = v;
  return Flatten(w);
}

/*!
 * \brief hash an object and combines it with previous keys
 * \param seed The previous hash value
 * \param value The object to be hashed and combined into seed
 * \return the combined hash.
 */
template <typename T>
inline uint64_t HashCombine(uint64_t seed, const T &value) {
  return seed ^
         (std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

std::vector<int> GetPositiveAxes(const std::vector<int> &axes, int rank);

int GetPositiveAxes(int axes, int rank);

template <typename T>
std::set<T> VectorToSet(const std::vector<T> &vec) {
  return std::set<T>(vec.begin(), vec.end());
}

}  // namespace utils
}  // namespace cinn
