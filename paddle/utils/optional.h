// This file copy from boost/optional/optional.hpp and boost version: 1.41.0
// Modified the following points:
// 1. modify namespace from boost::optional to paddle::optional
// 2. remove the depending boost header files
// 3. remove/modify some macro
// 4. copy some necessary data structures which are the depended by optional
// 5. replace type_with_alignment with std::aligned_storage

// Copyright (C) 2003, Fernando Luis Cacciola Carballal.
//
// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/lib/optional for documentation.
//
// You are welcome to contact the author at:
//  fernando_cacciola@hotmail.com
//
#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <new>
#include <type_traits>

#include "paddle/utils/none.h"

namespace paddle {

// Daniel Wallin discovered that bind/apply.hpp badly interacts with the apply<>
// member template of a factory as used in the optional<> implementation.
// He proposed this simple fix which is to move the call to apply<> outside
// namespace boost.
namespace paddle_optional_detail {
template <class T, class Factory>
void construct(Factory const& factory, void* address) {
  factory.template apply<T>(address);
}
}  // namespace paddle_optional_detail

template <typename T>
class optional;

class in_place_factory_base {};
class typed_in_place_factory_base {};

// template<class OP> bool equal_pointees(OP const& x, OP const& y);
// template<class OP> struct equal_pointees_t;
//
// Being OP a model of OptionalPointee (either a pointer or an optional):
//
// If both x and y have valid pointees, returns the result of (*x == *y)
// If only one has a valid pointee, returns false.
// If none have valid pointees, returns true.
// No-throw
template <class OptionalPointee>
inline bool equal_pointees(OptionalPointee const& x, OptionalPointee const& y) {
  return (!x) != (!y) ? false : (!x ? true : (*x) == (*y));
}

template <class OptionalPointee>
struct equal_pointees_t {
  using first_argument_type = OptionalPointee;
  using second_argument_type = OptionalPointee;
  using result_type = bool;
  bool operator()(OptionalPointee const& x, OptionalPointee const& y) const {
    return equal_pointees(x, y);
  }
};

// template<class OP> bool less_pointees(OP const& x, OP const& y);
// template<class OP> struct less_pointees_t;
//
// Being OP a model of OptionalPointee (either a pointer or an optional):
//
// If y has not a valid pointee, returns false.
// ElseIf x has not a valid pointee, returns true.
// ElseIf both x and y have valid pointees, returns the result of (*x < *y)
// No-throw
template <class OptionalPointee>
inline bool less_pointees(OptionalPointee const& x, OptionalPointee const& y) {
  return !y ? false : (!x ? true : (*x) < (*y));
}

template <class OptionalPointee>
struct less_pointees_t {
  using first_argument_type = OptionalPointee;
  using second_argument_type = OptionalPointee;
  using result_type = bool;
  bool operator()(OptionalPointee const& x, OptionalPointee const& y) const {
    return less_pointees(x, y);
  }
};

namespace detail {

template <typename RefT>
class reference_content {
 private:  // representation
  RefT content_;

 public:  // structors
  ~reference_content() {}

  reference_content(RefT r) : content_(r) {}  // NOLINT

  reference_content(const reference_content& operand)
      : content_(operand.content_) {}

 private:  // non-Assignable
  reference_content& operator=(const reference_content&);

 public:  // queries
  RefT get() const { return content_; }
};

template <typename T>
struct make_reference_content {
  typedef T type;
};

template <typename T>
struct make_reference_content<T&> {
  typedef reference_content<T&> type;
};

}  // namespace detail

namespace optional_detail {

// This local class is used instead of that in "aligned_storage.hpp"
// because I've found the 'official' class to ICE BCB5.5
// when some types are used with optional<>
// (due to sizeof() passed down as a non-type template parameter)
template <class T>
class aligned_storage {
  // Borland ICEs if unnamed unions are used for this!
  union dummy_u {
    char data[sizeof(T)];
    typename std::aligned_storage<::std::alignment_of<T>::value>::type aligner_;
  } dummy_;

 public:
  void const* address() const { return &dummy_.data[0]; }
  void* address() { return &dummy_.data[0]; }
};

template <class T>
struct types_when_isnt_ref {
  typedef T const& reference_const_type;
  typedef T& reference_type;
  typedef T const* pointer_const_type;
  typedef T* pointer_type;
  typedef T const& argument_type;
};
template <class T>
struct types_when_is_ref {
  typedef typename std::remove_reference<T>::type raw_type;

  typedef raw_type& reference_const_type;
  typedef raw_type& reference_type;
  typedef raw_type* pointer_const_type;
  typedef raw_type* pointer_type;
  typedef raw_type& argument_type;
};

struct optional_tag {};

template <class T>
class optional_base : public optional_tag {
 private:
  typedef
      typename ::paddle::detail::make_reference_content<T>::type internal_type;

  typedef aligned_storage<internal_type> storage_type;

  typedef types_when_isnt_ref<T> types_when_not_ref;
  typedef types_when_is_ref<T> types_when_ref;

  typedef optional_base<T> this_type;

 protected:
  typedef T value_type;

  typedef std::true_type is_reference_tag;
  typedef std::false_type is_not_reference_tag;

  typedef typename std::is_reference<T>::type is_reference_predicate;

  typedef typename std::conditional<is_reference_predicate::value,
                                    types_when_ref,
                                    types_when_not_ref>::type types;

  typedef bool (this_type::*unspecified_bool_type)() const;

  typedef typename types::reference_type reference_type;
  typedef typename types::reference_const_type reference_const_type;
  typedef typename types::pointer_type pointer_type;
  typedef typename types::pointer_const_type pointer_const_type;
  typedef typename types::argument_type argument_type;

  // Creates an optional<T> uninitialized.
  // No-throw
  optional_base() : m_initialized(false) {}

  // Creates an optional<T> uninitialized.
  // No-throw
  optional_base(none_t) : m_initialized(false) {}  // NOLINT

  // Creates an optional<T> initialized with 'val'.
  // Can throw if T::T(T const&) does
  optional_base(argument_type val) : m_initialized(false) {  // NOLINT
    construct(val);
  }

  // Creates an optional<T> initialized with 'val' IFF cond is true, otherwise
  // creates an uninitialized optional<T>.
  // Can throw if T::T(T const&) does
  optional_base(bool cond, argument_type val) : m_initialized(false) {
    if (cond) construct(val);
  }

  // Creates a deep copy of another optional<T>
  // Can throw if T::T(T const&) does
  optional_base(optional_base const& rhs) : m_initialized(false) {
    if (rhs.is_initialized()) construct(rhs.get_impl());
  }

  // This is used for both converting and in-place constructions.
  // Derived classes use the 'tag' to select the appropriate
  // implementation (the correct 'construct()' overload)
  template <class Expr>
  explicit optional_base(Expr const& expr, Expr const* tag)
      : m_initialized(false) {
    construct(expr, tag);
  }

  // No-throw (assuming T::~T() doesn't)
  ~optional_base() { destroy(); }

  // Assigns from another optional<T> (deep-copies the rhs value)
  void assign(optional_base const& rhs) {
    if (is_initialized()) {
      if (rhs.is_initialized())
        assign_value(rhs.get_impl(), is_reference_predicate());
      else
        destroy();
    } else {
      if (rhs.is_initialized()) construct(rhs.get_impl());
    }
  }

  // Assigns from another _convertible_ optional<U> (deep-copies the rhs value)
  template <class U>
  void assign(optional<U> const& rhs) {
    if (is_initialized()) {
      if (rhs.is_initialized())
        assign_value(static_cast<value_type>(rhs.get()),
                     is_reference_predicate());
      else
        destroy();
    } else {
      if (rhs.is_initialized()) construct(static_cast<value_type>(rhs.get()));
    }
  }

  // Assigns from a T (deep-copies the rhs value)
  void assign(argument_type val) {
    if (is_initialized())
      assign_value(val, is_reference_predicate());
    else
      construct(val);
  }

  // Assigns from "none", destroying the current value, if any, leaving this
  // UNINITIALIZED
  // No-throw (assuming T::~T() doesn't)
  void assign(none_t) { destroy(); }

  template <class Expr>
  void assign_expr(Expr const& expr, Expr const* tag) {
    if (is_initialized())
      assign_expr_to_initialized(expr, tag);
    else
      construct(expr, tag);
  }

 public:
  // Destroys the current value, if any, leaving this UNINITIALIZED
  // No-throw (assuming T::~T() doesn't)
  void reset() { destroy(); }

  // Replaces the current value -if any- with 'val'
  void reset(argument_type val) { assign(val); }

  // Returns a pointer to the value if this is initialized, otherwise,
  // returns NULL.
  // No-throw
  pointer_const_type get_ptr() const {
    return m_initialized ? get_ptr_impl() : 0;
  }
  pointer_type get_ptr() { return m_initialized ? get_ptr_impl() : 0; }

  bool is_initialized() const { return m_initialized; }

 protected:
  void construct(argument_type val) {
    new (m_storage.address()) internal_type(val);
    m_initialized = true;
  }

  // Constructs in-place using the given factory
  template <class Expr>
  void construct(Expr const& factory, in_place_factory_base const*) {
    static_assert(!is_reference_predicate::value,
                  "!is_reference_predicate::value");
    paddle_optional_detail::construct<value_type>(factory, m_storage.address());
    m_initialized = true;
  }

  // Constructs in-place using the given typed factory
  template <class Expr>
  void construct(Expr const& factory, typed_in_place_factory_base const*) {
    static_assert(!is_reference_predicate::value,
                  "!is_reference_predicate::value");
    factory.apply(m_storage.address());
    m_initialized = true;
  }

  template <class Expr>
  void assign_expr_to_initialized(Expr const& factory,
                                  in_place_factory_base const* tag) {
    destroy();
    construct(factory, tag);
  }

  // Constructs in-place using the given typed factory
  template <class Expr>
  void assign_expr_to_initialized(Expr const& factory,
                                  typed_in_place_factory_base const* tag) {
    destroy();
    construct(factory, tag);
  }

  // Constructs using any expression implicitly convertible to the single
  // argument
  // of a one-argument T constructor.
  // Converting constructions of optional<T> from optional<U> uses this function
  // with
  // 'Expr' being of type 'U' and relying on a converting constructor of T from
  // U.
  template <class Expr>
  void construct(Expr const& expr, void const*) {
    new (m_storage.address()) internal_type(expr);
    m_initialized = true;
  }

  // Assigns using a form any expression implicitly convertible to the single
  // argument
  // of a T's assignment operator.
  // Converting assignments of optional<T> from optional<U> uses this function
  // with
  // 'Expr' being of type 'U' and relying on a converting assignment of T from
  // U.
  template <class Expr>
  void assign_expr_to_initialized(Expr const& expr, void const*) {
    assign_value(expr, is_reference_predicate());
  }

  void assign_value(argument_type val, is_not_reference_tag) {
    get_impl() = val;
  }
  void assign_value(argument_type val, is_reference_tag) { construct(val); }

  void destroy() {
    if (m_initialized) destroy_impl(is_reference_predicate());
  }

  unspecified_bool_type safe_bool() const {
    return m_initialized ? &this_type::is_initialized : 0;
  }

  reference_const_type get_impl() const {
    return dereference(get_object(), is_reference_predicate());
  }
  reference_type get_impl() {
    return dereference(get_object(), is_reference_predicate());
  }

  pointer_const_type get_ptr_impl() const {
    return cast_ptr(get_object(), is_reference_predicate());
  }
  pointer_type get_ptr_impl() {
    return cast_ptr(get_object(), is_reference_predicate());
  }

 private:
  // internal_type can be either T or reference_content<T>
  internal_type const* get_object() const {
    return static_cast<internal_type const*>(m_storage.address());
  }
  internal_type* get_object() {
    return static_cast<internal_type*>(m_storage.address());
  }

  // reference_content<T> lacks an implicit conversion to T&, so the following
  // is needed to obtain a proper reference.
  reference_const_type dereference(internal_type const* p,
                                   is_not_reference_tag) const {
    return *p;
  }
  reference_type dereference(internal_type* p, is_not_reference_tag) {
    return *p;
  }
  reference_const_type dereference(internal_type const* p,
                                   is_reference_tag) const {
    return p->get();
  }
  reference_type dereference(internal_type* p, is_reference_tag) {
    return p->get();
  }

  void destroy_impl(is_not_reference_tag) {
    get_ptr_impl()->T::~T();
    m_initialized = false;
  }

  void destroy_impl(is_reference_tag) { m_initialized = false; }

  // If T is of reference type, trying to get a pointer to the held value must
  // result in a compile-time error.
  // Decent compilers should disallow conversions from reference_content<T>* to
  // T*, but just in case,
  // the following olverloads are used to filter out the case and guarantee an
  // error in case of T being a reference.
  pointer_const_type cast_ptr(internal_type const* p,
                              is_not_reference_tag) const {
    return p;
  }
  pointer_type cast_ptr(internal_type* p, is_not_reference_tag) { return p; }
  pointer_const_type cast_ptr(internal_type const* p, is_reference_tag) const {
    return &p->get();
  }
  pointer_type cast_ptr(internal_type* p, is_reference_tag) {
    return &p->get();
  }

  bool m_initialized;
  storage_type m_storage;
};

}  // namespace optional_detail

template <class T>
class optional : public optional_detail::optional_base<T> {
  typedef optional_detail::optional_base<T> base;

  typedef typename base::unspecified_bool_type unspecified_bool_type;

 public:
  typedef optional<T> this_type;

  typedef typename base::value_type value_type;
  typedef typename base::reference_type reference_type;
  typedef typename base::reference_const_type reference_const_type;
  typedef typename base::pointer_type pointer_type;
  typedef typename base::pointer_const_type pointer_const_type;
  typedef typename base::argument_type argument_type;

  // Creates an optional<T> uninitialized.
  // No-throw
  optional() : base() {}

  // Creates an optional<T> uninitialized.
  // No-throw
  optional(none_t none_) : base(none_) {}  // NOLINT

  // Creates an optional<T> initialized with 'val'.
  // Can throw if T::T(T const&) does
  optional(argument_type val) : base(val) {}  // NOLINT

  // Creates an optional<T> initialized with 'val' IFF cond is true, otherwise
  // creates an uninitialized optional.
  // Can throw if T::T(T const&) does
  optional(bool cond, argument_type val) : base(cond, val) {}

  // Creates a deep copy of another convertible optional<U>
  // Requires a valid conversion from U to T.
  // Can throw if T::T(U const&) does
  template <class U>
  explicit optional(optional<U> const& rhs) : base() {
    if (rhs.is_initialized()) this->construct(rhs.get());
  }

  // Creates an optional<T> with an expression which can be either
  //  (a) An instance of InPlaceFactory (i.e. in_place(a,b,...,n);
  //  (b) An instance of TypedInPlaceFactory ( i.e. in_place<T>(a,b,...,n);
  //  (c) Any expression implicitly convertible to the single type
  //      of a one-argument T's constructor.
  //  (d*) Weak compilers (BCB) might also resolved Expr as optional<T> and
  //  optional<U>
  //       even though explicit overloads are present for these.
  // Depending on the above some T ctor is called.
  // Can throw is the resolved T ctor throws.
  template <class Expr>
  explicit optional(Expr const& expr) : base(expr, &expr) {}

  // Creates a deep copy of another optional<T>
  // Can throw if T::T(T const&) does
  optional(optional const& rhs) : base(rhs) {}

  // No-throw (assuming T::~T() doesn't)
  ~optional() {}

  // Assigns from an expression. See corresponding constructor.
  // Basic Guarantee: If the resolved T ctor throws, this is left UNINITIALIZED
  template <class Expr>
  optional& operator=(Expr expr) {
    this->assign_expr(expr, &expr);
    return *this;
  }

  // Assigns from another convertible optional<U> (converts && deep-copies the
  // rhs value)
  // Requires a valid conversion from U to T.
  // Basic Guarantee: If T::T( U const& ) throws, this is left UNINITIALIZED
  template <class U>
  optional& operator=(optional<U> const& rhs) {
    this->assign(rhs);
    return *this;
  }

  // Assigns from another optional<T> (deep-copies the rhs value)
  // Basic Guarantee: If T::T( T const& ) throws, this is left UNINITIALIZED
  //  (NOTE: On BCB, this operator is not actually called and left is left
  //  UNMODIFIED in case of a throw)
  optional& operator=(optional const& rhs) {
    this->assign(rhs);
    return *this;
  }

  // Assigns from a T (deep-copies the rhs value)
  // Basic Guarantee: If T::( T const& ) throws, this is left UNINITIALIZED
  optional& operator=(argument_type val) {
    this->assign(val);
    return *this;
  }

  // Assigns from a "none"
  // Which destroys the current value, if any, leaving this UNINITIALIZED
  // No-throw (assuming T::~T() doesn't)
  optional& operator=(none_t none_) {
    this->assign(none_);
    return *this;
  }

  // Returns a reference to the value if this is initialized, otherwise,
  // the behaviour is UNDEFINED
  // No-throw
  reference_const_type get() const {
    assert(this->is_initialized());
    return this->get_impl();
  }
  reference_type get() {
    assert(this->is_initialized());
    return this->get_impl();
  }

  // Returns a copy of the value if this is initialized, 'v' otherwise
  reference_const_type get_value_or(reference_const_type v) const {
    return this->is_initialized() ? get() : v;
  }
  reference_type get_value_or(reference_type v) {
    return this->is_initialized() ? get() : v;
  }

  // Returns a pointer to the value if this is initialized, otherwise,
  // the behaviour is UNDEFINED
  // No-throw
  pointer_const_type operator->() const {
    assert(this->is_initialized());
    return this->get_ptr_impl();
  }
  pointer_type operator->() {
    assert(this->is_initialized());
    return this->get_ptr_impl();
  }

  // Returns a reference to the value if this is initialized, otherwise,
  // the behaviour is UNDEFINED
  // No-throw
  reference_const_type operator*() const { return this->get(); }
  reference_type operator*() { return this->get(); }

  // implicit conversion to "bool"
  // No-throw
  operator unspecified_bool_type() const { return this->safe_bool(); }

  // This is provided for those compilers which don't like the conversion to
  // bool
  // on some contexts.
  bool operator!() const { return !this->is_initialized(); }
};

// Returns optional<T>(v)
template <class T>
inline optional<T> make_optional(T const& v) {
  return optional<T>(v);
}

// Returns optional<T>(cond,v)
template <class T>
inline optional<T> make_optional(bool cond, T const& v) {
  return optional<T>(cond, v);
}

// Returns a reference to the value if this is initialized, otherwise, the
// behaviour is UNDEFINED.
// No-throw
template <class T>
inline typename optional<T>::reference_const_type get(optional<T> const& opt) {
  return opt.get();
}

template <class T>
inline typename optional<T>::reference_type get(optional<T>& opt) {  // NOLINT
  return opt.get();
}

// Returns a pointer to the value if this is initialized, otherwise, returns
// NULL.
// No-throw
template <class T>
inline typename optional<T>::pointer_const_type get(optional<T> const* opt) {
  return opt->get_ptr();
}

template <class T>
inline typename optional<T>::pointer_type get(optional<T>* opt) {
  return opt->get_ptr();
}

// Returns a reference to the value if this is initialized, otherwise, the
// behaviour is UNDEFINED.
// No-throw
template <class T>
inline typename optional<T>::reference_const_type get_optional_value_or(
    optional<T> const& opt, typename optional<T>::reference_const_type v) {
  return opt.get_value_or(v);
}

template <class T>
inline typename optional<T>::reference_type get_optional_value_or(
    optional<T>& opt, typename optional<T>::reference_type v) {  // NOLINT
  return opt.get_value_or(v);
}

// Returns a pointer to the value if this is initialized, otherwise, returns
// NULL.
// No-throw
template <class T>
inline typename optional<T>::pointer_const_type get_pointer(
    optional<T> const& opt) {
  return opt.get_ptr();
}

template <class T>
inline typename optional<T>::pointer_type get_pointer(
    optional<T>& opt) {  // NOLINT
  return opt.get_ptr();
}

// optional's relational operators ( ==, !=, <, >, <=, >= ) have deep-semantics
// (compare values).
// WARNING: This is UNLIKE pointers. Use equal_pointees()/less_pointess() in
// generic code instead.

//
// optional<T> vs optional<T> cases
//

template <class T>
inline bool operator==(optional<T> const& x, optional<T> const& y) {
  return equal_pointees(x, y);
}

template <class T>
inline bool operator<(optional<T> const& x, optional<T> const& y) {
  return less_pointees(x, y);
}

template <class T>
inline bool operator!=(optional<T> const& x, optional<T> const& y) {
  return !(x == y);
}

template <class T>
inline bool operator>(optional<T> const& x, optional<T> const& y) {
  return y < x;
}

template <class T>
inline bool operator<=(optional<T> const& x, optional<T> const& y) {
  return !(y < x);
}

template <class T>
inline bool operator>=(optional<T> const& x, optional<T> const& y) {
  return !(x < y);
}

//
// optional<T> vs T cases
//
template <class T>
inline bool operator==(optional<T> const& x, T const& y) {
  return equal_pointees(x, optional<T>(y));
}

template <class T>
inline bool operator<(optional<T> const& x, T const& y) {
  return less_pointees(x, optional<T>(y));
}

template <class T>
inline bool operator!=(optional<T> const& x, T const& y) {
  return !(x == y);
}

template <class T>
inline bool operator>(optional<T> const& x, T const& y) {
  return y < x;
}

template <class T>
inline bool operator<=(optional<T> const& x, T const& y) {
  return !(y < x);
}

template <class T>
inline bool operator>=(optional<T> const& x, T const& y) {
  return !(x < y);
}

//
// T vs optional<T> cases
//

template <class T>
inline bool operator==(T const& x, optional<T> const& y) {
  return equal_pointees(optional<T>(x), y);
}

template <class T>
inline bool operator<(T const& x, optional<T> const& y) {
  return less_pointees(optional<T>(x), y);
}

template <class T>
inline bool operator!=(T const& x, optional<T> const& y) {
  return !(x == y);
}

template <class T>
inline bool operator>(T const& x, optional<T> const& y) {
  return y < x;
}

template <class T>
inline bool operator<=(T const& x, optional<T> const& y) {
  return !(y < x);
}

template <class T>
inline bool operator>=(T const& x, optional<T> const& y) {
  return !(x < y);
}

//
// optional<T> vs none cases
//

template <class T>
inline bool operator==(optional<T> const& x, none_t) {
  return equal_pointees(x, optional<T>());
}

template <class T>
inline bool operator<(optional<T> const& x, none_t) {
  return less_pointees(x, optional<T>());
}

template <class T>
inline bool operator!=(optional<T> const& x, none_t y) {
  return !(x == y);
}

template <class T>
inline bool operator>(optional<T> const& x, none_t y) {
  return y < x;
}

template <class T>
inline bool operator<=(optional<T> const& x, none_t y) {
  return !(y < x);
}

template <class T>
inline bool operator>=(optional<T> const& x, none_t y) {
  return !(x < y);
}

//
// none vs optional<T> cases
//

template <class T>
inline bool operator==(none_t x, optional<T> const& y) {
  return equal_pointees(optional<T>(), y);
}

template <class T>
inline bool operator<(none_t x, optional<T> const& y) {
  return less_pointees(optional<T>(), y);
}

template <class T>
inline bool operator!=(none_t x, optional<T> const& y) {
  return !(x == y);
}

template <class T>
inline bool operator>(none_t x, optional<T> const& y) {
  return y < x;
}

template <class T>
inline bool operator<=(none_t x, optional<T> const& y) {
  return !(y < x);
}

template <class T>
inline bool operator>=(none_t x, optional<T> const& y) {
  return !(x < y);
}

namespace optional_detail {

// optional's swap:
// If both are initialized, calls swap(T&, T&). If this swap throws, both will
// remain initialized but their values are now unspecified.
// If only one is initialized, calls U.reset(*I), THEN I.reset().
// If U.reset(*I) throws, both are left UNCHANGED (U is kept uinitialized and I
// is never reset)
// If both are uninitialized, do nothing (no-throw)
template <class T>
inline void optional_swap(optional<T>& x, optional<T>& y) {
  if (!x && !!y) {
    x.reset(*y);
    y.reset();
  } else if (!!x && !y) {
    y.reset(*x);
    x.reset();
  } else if (!!x && !!y) {
    // allow for Koenig lookup
    using std::swap;
    swap(*x, *y);
  }
}

}  // namespace optional_detail

}  // namespace paddle
