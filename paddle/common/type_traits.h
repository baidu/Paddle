// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#if __cplusplus < 201103L
#include <bits/c++0x_warning.h>
#else
namespace paddle {

template <typename...>
using __void_t = void;

template <typename _Tp, typename = void>
struct __add_pointer_helper {
  using type = _Tp;
};

template <typename _Tp>
struct __add_pointer_helper<_Tp, __void_t<_Tp*>> {
  using type = _Tp*;
};

/// add_pointer
template <typename _Tp>
struct add_pointer : public __add_pointer_helper<_Tp> {};

template <typename _Tp>
struct add_pointer<_Tp&> {
  using type = _Tp*;
};

template <typename _Tp>
struct add_pointer<_Tp&&> {
  using type = _Tp*;
};

template <typename _Tp>
struct remove_cv {
  using type = _Tp;
};

template <typename _Tp>
struct remove_cv<const _Tp> {
  using type = _Tp;
};

template <typename _Tp>
struct remove_cv<volatile _Tp> {
  using type = _Tp;
};

template <typename _Tp>
struct remove_cv<const volatile _Tp> {
  using type = _Tp;
};

template <bool>
struct __conditional {
  template <typename _Tp, typename>
  using type = _Tp;
};

template <>
struct __conditional<false> {
  template <typename, typename _Up>
  using type = _Up;
};

// More efficient version of std::conditional_t for internal use (and C++11)
template <bool _Cond, typename _If, typename _Else>
using __conditional_t =
    typename __conditional<_Cond>::template type<_If, _Else>;

template <typename _Up>
struct __decay_selector
    : __conditional_t<is_const<const _Up>::value,  // false for functions
                      remove_cv<_Up>,              // N.B. DR 705.
                      add_pointer<_Up>>            // function decays to pointer
{};

template <typename _Up, size_t _Nm>
struct __decay_selector<_Up[_Nm]> {
  using type = _Up*;
};

template <typename _Up>
struct __decay_selector<_Up[]> {
  using type = _Up*;
};

/// decay
template <typename _Tp>
struct decay {
  using type = typename __decay_selector<_Tp>::type;
};

template <typename _Tp>
struct decay<_Tp&> {
  using type = typename __decay_selector<_Tp>::type;
};

template <typename _Tp>
struct decay<_Tp&&> {
  using type = typename __decay_selector<_Tp>::type;
};

template <typename _Tp>
using __decay_t = typename decay<_Tp>::type;

template <typename _Tp>
struct __success_type {
  typedef _Tp type;
};

struct __failure_type {};

struct __do_common_type_impl {
  template <typename _Tp, typename _Up>
  using __cond_t = decltype(true ? std::declval<_Tp>() : std::declval<_Up>());

  // if decay_t<decltype(false ? declval<D1>() : declval<D2>())>
  // denotes a valid type, let C denote that type.
  template <typename _Tp, typename _Up>
  static __success_type<__decay_t<__cond_t<_Tp, _Up>>> _S_test(int);

  template <typename, typename>
  static __failure_type _S_test_2(...);

  template <typename _Tp, typename _Up>
  static decltype(_S_test_2<_Tp, _Up>(0)) _S_test(...);
};

// If sizeof...(T) is two, ...
template <typename _Tp1,
          typename _Tp2,
          typename _Dp1 = __decay_t<_Tp1>,
          typename _Dp2 = __decay_t<_Tp2>>
struct __common_type_impl {
  // If is_same_v<T1, D1> is false or is_same_v<T2, D2> is false,
  // let C denote the same type, if any, as common_type_t<D1, D2>.
  using type = common_type<_Dp1, _Dp2>;
};

template <typename _Tp1, typename _Tp2>
struct __common_type_impl<_Tp1, _Tp2, _Tp1, _Tp2>
    : private __do_common_type_impl {
  // Otherwise, if decay_t<decltype(false ? declval<D1>() : declval<D2>())>
  // denotes a valid type, let C denote that type.
  using type = decltype(_S_test<_Tp1, _Tp2>(0));
};

/// common_type
template <typename... _Tp>
struct common_type;

// If sizeof...(T) is two, ...
template <typename _Tp1, typename _Tp2>
struct common_type<_Tp1, _Tp2> : public __common_type_impl<_Tp1, _Tp2>::type {};

}  // namespace paddle
#endif
