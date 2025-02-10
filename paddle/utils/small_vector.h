// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// This file copy from llvm/ADT/SmallVector.h, version: 12.0.0
// Modified the following points
// 1. remove  macro
// 2. remove LLVM_LIKELY and LLVM_UNLIKELY
// 3. add at(index) method for small vector
// 4. wrap the call to max and min with parenthesis to prevent the macro
// expansion to fix the build error on windows platform
// 5. change SmallVector to small_vector to unify naming style of utils

//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallVector class.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace paddle {

/// A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

 public:
  // TODO(others): Add SFINAE to test that the Container's iterators match the
  // range's
  //      iterators.
  template <typename Container>
  explicit iterator_range(Container &&c)
      // TODO(others): Consider ADL/non-member begin/end calls.
      : begin_iterator(c.begin()), end_iterator(c.end()) {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
  bool empty() const { return begin_iterator == end_iterator; }
};

/// Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template <class T>
iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T>
iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

/// This is all the stuff common to all SmallVectors.
///
/// The template parameter specifies the type which should be used to hold the
/// Size and Capacity of the small_vector, so it can be adjusted.
/// Using 32 bit size is desirable to shrink the size of the small_vector.
/// Using 64 bit size is desirable for cases like small_vector<char>, where a
/// 32 bit size would limit the vector to ~4GB. SmallVectors are used for
/// buffering bitcode output - which can exceed 4GB.
template <class Size_T>
class small_vector_base {
 protected:
  void *BeginX;
  Size_T Size = 0, Capacity;

  /// The maximum value of the Size_T used.
  static constexpr size_t SizeTypeMax() {
    return (std::numeric_limits<Size_T>::max)();
  }

  small_vector_base() = delete;
  small_vector_base(void *FirstEl, size_t TotalCapacity)
      : BeginX(FirstEl), Capacity(TotalCapacity) {}

  /// This is a helper for \a grow() that's out of line to reduce code
  /// duplication.  This function will report a fatal error if it can't grow at
  /// least to \p MinSize.
  void *mallocForGrow(size_t MinSize, size_t TSize, size_t *NewCapacity);

  /// This is an implementation of the grow() method which only works
  /// on POD-like data types and is out of line to reduce code duplication.
  /// This function will report a fatal error if it cannot increase capacity.
  void grow_pod(void *FirstEl, size_t MinSize, size_t TSize);

 public:
  size_t size() const { return Size; }
  size_t capacity() const { return Capacity; }

  bool empty() const { return !Size; }

  /// Set the array size to \p N, which the current array must have enough
  /// capacity for.
  ///
  /// This does not construct or destroy any elements in the vector.
  ///
  /// Clients can use this in conjunction with capacity() to write past the end
  /// of the buffer when they know that more elements are available, and only
  /// update the size later. This avoids the cost of value initializing elements
  /// which will only be overwritten.
  void set_size(size_t N) {
    assert(N <= capacity());
    Size = N;
  }
};

template <class T>
using SmallVectorSizeType = typename std::
    conditional<sizeof(T) < 4 && sizeof(void *) >= 8, uint64_t, uint32_t>::type;

/// Figure out the offset of the first element.
template <class T, typename = void>
struct SmallVectorAlignmentAndSize {
  alignas(small_vector_base<SmallVectorSizeType<T>>) char Base[sizeof(
      small_vector_base<SmallVectorSizeType<T>>)];
  alignas(T) char FirstEl[sizeof(T)];
};

/// This is the part of small_vector_template_base which does not depend on
/// whether
/// the type T is a POD. The extra dummy template argument is used by array_ref
/// to avoid unnecessarily requiring T to be complete.
template <typename T, typename = void>
class small_vector_template_common
    : public small_vector_base<SmallVectorSizeType<T>> {
  using Base = small_vector_base<SmallVectorSizeType<T>>;

  /// Find the address of the first element.  For this pointer math to be valid
  /// with small-size of 0 for T with lots of alignment, it's important that
  /// small_vector_storage is properly-aligned even for small-size of 0.
  void *getFirstEl() const {
    return const_cast<void *>(reinterpret_cast<const void *>(
        reinterpret_cast<const char *>(this) +
        offsetof(SmallVectorAlignmentAndSize<T>, FirstEl)));
  }
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.

 protected:
  explicit small_vector_template_common(size_t Size)
      : Base(getFirstEl(), Size) {}

  void grow_pod(size_t MinSize, size_t TSize) {
    Base::grow_pod(getFirstEl(), MinSize, TSize);
  }

  /// Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const { return this->BeginX == getFirstEl(); }

  /// Put this vector in a state of being small.
  void resetToSmall() {
    this->BeginX = getFirstEl();
    this->Size = this->Capacity =
        0;  // FIXME: Setting Capacity to 0 is suspect.
  }

  /// Return true if V is an internal reference to the given range.
  bool isReferenceToRange(const void *V,
                          const void *First,
                          const void *Last) const {
    // Use std::less to avoid UB.
    std::less<> LessThan;
    return !LessThan(V, First) && LessThan(V, Last);
  }

  /// Return true if V is an internal reference to this vector.
  bool isReferenceToStorage(const void *V) const {
    return isReferenceToRange(V, this->begin(), this->end());
  }

  /// Return true if First and Last form a valid (possibly empty) range in this
  /// vector's storage.
  bool isRangeInStorage(const void *First, const void *Last) const {
    // Use std::less to avoid UB.
    std::less<> LessThan;
    return !LessThan(First, this->begin()) && !LessThan(Last, First) &&
           !LessThan(this->end(), Last);
  }

  /// Return true unless Elt will be invalidated by resizing the vector to
  /// NewSize.
  bool isSafeToReferenceAfterResize(const void *Elt, size_t NewSize) {
    // Past the end.
    if (!isReferenceToStorage(Elt)) return true;

    // Return false if Elt will be destroyed by shrinking.
    if (NewSize <= this->size()) return Elt < this->begin() + NewSize;

    // Return false if we need to grow.
    return NewSize <= this->capacity();
  }

  /// Check whether Elt will be invalidated by resizing the vector to NewSize.
  void assertSafeToReferenceAfterResize(const void *Elt, size_t NewSize) {
    (void)Elt;
    (void)NewSize;  // just remove [-Wunused-parameter]
    assert(isSafeToReferenceAfterResize(Elt, NewSize) &&
           "Attempting to reference an element of the vector in an operation "
           "that invalidates it");
  }

  /// Check whether Elt will be invalidated by increasing the size of the
  /// vector by N.
  void assertSafeToAdd(const void *Elt, size_t N = 1) {
    this->assertSafeToReferenceAfterResize(Elt, this->size() + N);
  }

  /// Check whether any part of the range will be invalidated by clearing.
  void assertSafeToReferenceAfterClear(const T *From, const T *To) {
    if (From == To) return;
    this->assertSafeToReferenceAfterResize(From, 0);
    this->assertSafeToReferenceAfterResize(To - 1, 0);
  }
  template <
      class ItTy,
      std::enable_if_t<!std::is_same<std::remove_const_t<ItTy>, T *>::value,
                       bool> = false>
  void assertSafeToReferenceAfterClear(ItTy, ItTy) {}

  /// Check whether any part of the range will be invalidated by growing.
  void assertSafeToAddRange(const T *From, const T *To) {
    if (From == To) return;
    this->assertSafeToAdd(From, To - From);
    this->assertSafeToAdd(To - 1, To - From);
  }
  template <
      class ItTy,
      std::enable_if_t<!std::is_same<std::remove_const_t<ItTy>, T *>::value,
                       bool> = false>
  void assertSafeToAddRange(ItTy, ItTy) {}

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  template <class U>
  static const T *reserveForParamAndGetAddressImpl(U *This,
                                                   const T &Elt,
                                                   size_t N) {
    size_t NewSize = This->size() + N;
    if (NewSize <= This->capacity()) return &Elt;

    bool ReferencesStorage = false;
    int64_t Index = -1;
    if (!U::TakesParamByValue) {
      if (This->isReferenceToStorage(&Elt)) {
        ReferencesStorage = true;
        Index = &Elt - This->begin();
      }
    }
    This->grow(NewSize);
    return ReferencesStorage ? This->begin() + Index : &Elt;
  }

 public:
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = std::reverse_iterator<iterator>;

  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;

  using Base::capacity;
  using Base::empty;
  using Base::size;

  // forward iterator creation methods.
  iterator begin() { return (iterator)this->BeginX; }
  const_iterator begin() const { return (const_iterator)this->BeginX; }
  iterator end() { return begin() + size(); }
  const_iterator end() const { return begin() + size(); }

  // reverse iterator creation methods.
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  size_type size_in_bytes() const { return size() * sizeof(T); }
  size_type max_size() const {
    return (std::min)(this->SizeTypeMax(), size_type(-1) / sizeof(T));
  }

  size_t capacity_in_bytes() const { return capacity() * sizeof(T); }

  /// Return a pointer to the vector's buffer, even if empty().
  pointer data() { return pointer(begin()); }
  /// Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const { return const_pointer(begin()); }

  reference operator[](size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const_reference operator[](size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference at(size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const_reference at(size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference front() {
    assert(!empty());
    return begin()[0];
  }
  const_reference front() const {
    assert(!empty());
    return begin()[0];
  }

  reference back() {
    assert(!empty());
    return end()[-1];
  }
  const_reference back() const {
    assert(!empty());
    return end()[-1];
  }
};

/// small_vector_template_base<TriviallyCopyable = false> - This is where we put
/// method implementations that are designed to work with non-trivial T's.
///
/// We approximate is_trivially_copyable with trivial move/copy construction and
/// trivial destruction. While the standard doesn't specify that you're allowed
/// copy these types with memcpy, there is no way for the type to observe this.
/// This catches the important case of std::pair<POD, POD>, which is not
/// trivially assignable.
template <typename T,
          bool = (std::is_trivially_copy_constructible<T>::value) &&
                 (std::is_trivially_move_constructible<T>::value) &&
                 std::is_trivially_destructible<T>::value>
class small_vector_template_base : public small_vector_template_common<T> {
  friend class small_vector_template_common<T>;

 protected:
  static constexpr bool TakesParamByValue = false;
  using ValueParamT = const T &;

  explicit small_vector_template_base(size_t Size)
      : small_vector_template_common<T>(Size) {}

  static void destroy_range(T *S, T *E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// Move the range [I, E) into the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(
        std::make_move_iterator(I), std::make_move_iterator(E), Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  /// Grow the allocated memory (without initializing new elements), doubling
  /// the size of the allocated memory. Guarantees space for at least one more
  /// element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

  /// Create a new allocation big enough for \p MinSize and pass back its size
  /// in \p NewCapacity. This is the first section of \a grow().
  T *mallocForGrow(size_t MinSize, size_t *NewCapacity) {
    return static_cast<T *>(
        small_vector_base<SmallVectorSizeType<T>>::mallocForGrow(
            MinSize, sizeof(T), NewCapacity));
  }

  /// Move existing elements over to the new allocation \p NewElts, the middle
  /// section of \a grow().
  void moveElementsForGrow(T *NewElts);

  /// Transfer ownership of the allocation, finishing up \a grow().
  void takeAllocationForGrow(T *NewElts, size_t NewCapacity);

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  T *reserveForParamAndGetAddress(const T &Elt, size_t N = 1) {
    return const_cast<T *>(
        this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  static T &&forward_value_param(T &&V) { return std::move(V); }
  static const T &forward_value_param(const T &V) { return V; }

  void growAndAssign(size_t NumElts, const T &Elt) {
    // Grow manually in case Elt is an internal reference.
    size_t NewCapacity = 0;
    T *NewElts = mallocForGrow(NumElts, &NewCapacity);
    std::uninitialized_fill_n(NewElts, NumElts, Elt);
    this->destroy_range(this->begin(), this->end());
    takeAllocationForGrow(NewElts, NewCapacity);
    this->set_size(NumElts);
  }

  template <typename... ArgTypes>
  T &growAndEmplaceBack(ArgTypes &&...Args) {
    // Grow manually in case one of Args is an internal reference.
    size_t NewCapacity = 0;
    T *NewElts = mallocForGrow(0, &NewCapacity);
    ::new (reinterpret_cast<void *>(NewElts + this->size()))
        T(std::forward<ArgTypes>(Args)...);
    moveElementsForGrow(NewElts);
    takeAllocationForGrow(NewElts, NewCapacity);
    this->set_size(this->size() + 1);
    return this->back();
  }

 public:
  void push_back(const T &Elt) {
    const T *EltPtr = reserveForParamAndGetAddress(Elt);
    ::new (reinterpret_cast<void *>(this->end())) T(*EltPtr);
    this->set_size(this->size() + 1);
  }

  void push_back(T &&Elt) {
    T *EltPtr = reserveForParamAndGetAddress(Elt);
    ::new (reinterpret_cast<void *>(this->end())) T(::std::move(*EltPtr));
    this->set_size(this->size() + 1);
  }

  void pop_back() {
    this->set_size(this->size() - 1);
    this->end()->~T();
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool TriviallyCopyable>
void small_vector_template_base<T, TriviallyCopyable>::grow(size_t MinSize) {
  size_t NewCapacity = 0;
  T *NewElts = mallocForGrow(MinSize, &NewCapacity);
  moveElementsForGrow(NewElts);
  takeAllocationForGrow(NewElts, NewCapacity);
}

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool TriviallyCopyable>
void small_vector_template_base<T, TriviallyCopyable>::moveElementsForGrow(
    T *NewElts) {
  // Move the elements over.
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // Destroy the original elements.
  destroy_range(this->begin(), this->end());
}

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool TriviallyCopyable>
void small_vector_template_base<T, TriviallyCopyable>::takeAllocationForGrow(
    T *NewElts, size_t NewCapacity) {
  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall()) free(this->begin());

  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

/// small_vector_template_base<TriviallyCopyable = true> - This is where we put
/// method implementations that are designed to work with trivially copyable
/// T's. This allows using memcpy in place of copy/move construction and
/// skipping destruction.
template <typename T>
class small_vector_template_base<T, true>
    : public small_vector_template_common<T> {
  friend class small_vector_template_common<T>;

 protected:
  /// True if it's cheap enough to take parameters by value. Doing so avoids
  /// overhead related to mitigations for reference invalidation.
  static constexpr bool TakesParamByValue = sizeof(T) <= 2 * sizeof(void *);

  /// Either const T& or T, depending on whether it's cheap enough to take
  /// parameters by value.
  using ValueParamT =
      typename std::conditional<TakesParamByValue, T, const T &>::type;

  explicit small_vector_template_base(size_t Size)
      : small_vector_template_common<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T *, T *) {}

  /// Move the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    // Just do a copy.
    uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    // Arbitrary iterator types; just use the basic implementation.
    std::uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1 *I,
      T1 *E,
      T2 *Dest,
      std::enable_if_t<std::is_same<typename std::remove_const<T1>::type,
                                    T2>::value> * = nullptr) {
    // Use memcpy for PODs iterated by pointers (which includes small_vector
    // iterators): std::uninitialized_copy optimizes to memmove, but we can
    // use memcpy here. Note that I and E are iterators and thus might be
    // invalid for memcpy if they are equal.
    if (I != E) memcpy(reinterpret_cast<void *>(Dest), I, (E - I) * sizeof(T));
  }

  /// Double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) { this->grow_pod(MinSize, sizeof(T)); }

  /// Reserve enough space to add one element, and return the updated element
  /// pointer in case it was a reference to the storage.
  T *reserveForParamAndGetAddress(const T &Elt, size_t N = 1) {
    return const_cast<T *>(
        this->reserveForParamAndGetAddressImpl(this, Elt, N));
  }

  /// Copy \p V or return a reference, depending on \a ValueParamT.
  static ValueParamT forward_value_param(ValueParamT V) { return V; }

  void growAndAssign(size_t NumElts, T Elt) {
    // Elt has been copied in case it's an internal reference, side-stepping
    // reference invalidation problems without losing the realloc optimization.
    this->set_size(0);
    this->grow(NumElts);
    std::uninitialized_fill_n(this->begin(), NumElts, Elt);
    this->set_size(NumElts);
  }

  template <typename... ArgTypes>
  T &growAndEmplaceBack(ArgTypes &&...Args) {
    // Use push_back with a copy in case Args has an internal reference,
    // side-stepping reference invalidation problems without losing the realloc
    // optimization.
    push_back(T(std::forward<ArgTypes>(Args)...));
    return this->back();
  }

 public:
  void push_back(ValueParamT Elt) {
    const T *EltPtr = reserveForParamAndGetAddress(Elt);
    memcpy(reinterpret_cast<void *>(this->end()), EltPtr, sizeof(T));
    this->set_size(this->size() + 1);
  }

  void pop_back() { this->set_size(this->size() - 1); }
};

/// This class consists of common code factored out of the small_vector class to
/// reduce code duplication based on the small_vector 'N' template parameter.
template <typename T>
class small_vector_impl : public small_vector_template_base<T> {
  using SuperClass = small_vector_template_base<T>;

 public:
  using iterator = typename SuperClass::iterator;
  using const_iterator = typename SuperClass::const_iterator;
  using reference = typename SuperClass::reference;
  using size_type = typename SuperClass::size_type;

 protected:
  using small_vector_template_base<T>::TakesParamByValue;
  using ValueParamT = typename SuperClass::ValueParamT;

  // Default ctor - Initialize to empty.
  explicit small_vector_impl(unsigned N) : small_vector_template_base<T>(N) {}

 public:
  small_vector_impl(const small_vector_impl &) = delete;

  ~small_vector_impl() {
    // Subclass has already destructed this vector's elements.
    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall()) free(this->begin());
  }

  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->Size = 0;
  }

 private:
  template <bool ForOverwrite>
  void resizeImpl(size_type N) {
    if (N < this->size()) {
      this->pop_back_n(this->size() - N);
    } else if (N > this->size()) {
      this->reserve(N);
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I)
        if (ForOverwrite)
          new (&*I) T;
        else
          new (&*I) T();
      this->set_size(N);
    }
  }

 public:
  void resize(size_type N) { resizeImpl<false>(N); }

  /// Like resize, but \ref T is POD, the new values won't be initialized.
  void resize_for_overwrite(size_type N) { resizeImpl<true>(N); }

  void resize(size_type N, ValueParamT NV) {
    if (N == this->size()) return;

    if (N < this->size()) {
      this->pop_back_n(this->size() - N);
      return;
    }

    // N > this->size(). Defer to append.
    this->append(N - this->size(), NV);
  }

  void reserve(size_type N) {
    if (this->capacity() < N) this->grow(N);
  }

  void pop_back_n(size_type NumItems) {
    assert(this->size() >= NumItems);
    this->destroy_range(this->end() - NumItems, this->end());
    this->set_size(this->size() - NumItems);
  }

  T pop_back_val() {
    T Result = ::std::move(this->back());
    this->pop_back();
    return Result;
  }

  void swap(small_vector_impl &RHS);

  /// Add the specified range to the end of the small_vector.
  template <typename in_iter,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>::value>>
  void append(in_iter in_start, in_iter in_end) {
    this->assertSafeToAddRange(in_start, in_end);
    size_type NumInputs = std::distance(in_start, in_end);
    this->reserve(this->size() + NumInputs);
    this->uninitialized_copy(in_start, in_end, this->end());
    this->set_size(this->size() + NumInputs);
  }

  /// Append \p NumInputs copies of \p Elt to the end.
  void append(size_type NumInputs, ValueParamT Elt) {
    const T *EltPtr = this->reserveForParamAndGetAddress(Elt, NumInputs);
    std::uninitialized_fill_n(this->end(), NumInputs, *EltPtr);
    this->set_size(this->size() + NumInputs);
  }

  void append(std::initializer_list<T> IL) { append(IL.begin(), IL.end()); }

  void append(const small_vector_impl &RHS) { append(RHS.begin(), RHS.end()); }

  void assign(size_type NumElts, ValueParamT Elt) {
    // Note that Elt could be an internal reference.
    if (NumElts > this->capacity()) {
      this->growAndAssign(NumElts, Elt);
      return;
    }

    // Assign over existing elements.
    std::fill_n(this->begin(), (std::min)(NumElts, this->size()), Elt);
    if (NumElts > this->size())
      std::uninitialized_fill_n(this->end(), NumElts - this->size(), Elt);
    else if (NumElts < this->size())
      this->destroy_range(this->begin() + NumElts, this->end());
    this->set_size(NumElts);
  }

  // FIXME: Consider assigning over existing elements, rather than clearing &
  // re-initializing them - for all assign(...) variants.

  template <typename in_iter,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<in_iter>::iterator_category,
                std::input_iterator_tag>::value>>
  void assign(in_iter in_start, in_iter in_end) {
    this->assertSafeToReferenceAfterClear(in_start, in_end);
    clear();
    append(in_start, in_end);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  void assign(const small_vector_impl &RHS) { assign(RHS.begin(), RHS.end()); }

  iterator erase(const_iterator CI) {
    // Just cast away constness because this is a non-const member function.
    iterator I = const_cast<iterator>(CI);

    assert(this->isReferenceToStorage(CI) &&
           "Iterator to erase is out of bounds.");

    iterator N = I;
    // Shift all elts down one.
    std::move(I + 1, this->end(), I);
    // Drop the last elt.
    this->pop_back();
    return (N);
  }

  iterator erase(const_iterator CS, const_iterator CE) {
    // Just cast away constness because this is a non-const member function.
    iterator S = const_cast<iterator>(CS);
    iterator E = const_cast<iterator>(CE);

    assert(this->isRangeInStorage(S, E) && "Range to erase is out of bounds.");

    iterator N = S;
    // Shift all elts down.
    iterator I = std::move(E, this->end(), S);
    // Drop the last elts.
    this->destroy_range(I, this->end());
    this->set_size(I - this->begin());
    return (N);
  }

 private:
  template <class ArgType>
  iterator insert_one_impl(iterator I, ArgType &&Elt) {
    // Callers ensure that ArgType is derived from T.
    static_assert(
        std::is_same<std::remove_const_t<std::remove_reference_t<ArgType>>,
                     T>::value,
        "ArgType must be derived from T!");

    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(::std::forward<ArgType>(Elt));
      return this->end() - 1;
    }

    assert(this->isReferenceToStorage(I) &&
           "Insertion iterator is out of bounds.");

    // Grow if necessary.
    size_t Index = I - this->begin();
    std::remove_reference_t<ArgType> *EltPtr =
        this->reserveForParamAndGetAddress(Elt);
    I = this->begin() + Index;

    ::new (reinterpret_cast<void *>(this->end())) T(::std::move(this->back()));
    // Push everything else over.
    std::move_backward(I, this->end() - 1, this->end());
    this->set_size(this->size() + 1);

    // If we just moved the element we're inserting, be sure to update
    // the reference (never happens if TakesParamByValue).
    static_assert(!TakesParamByValue || std::is_same<ArgType, T>::value,
                  "ArgType must be 'T' when taking by value!");
    if (!TakesParamByValue && this->isReferenceToRange(EltPtr, I, this->end()))
      ++EltPtr;

    *I = ::std::forward<ArgType>(*EltPtr);
    return I;
  }

 public:
  iterator insert(iterator I, T &&Elt) {
    return insert_one_impl(I, this->forward_value_param(std::move(Elt)));
  }

  iterator insert(iterator I, const T &Elt) {
    return insert_one_impl(I, this->forward_value_param(Elt));
  }

  iterator insert(iterator I, size_type NumToInsert, ValueParamT Elt) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(NumToInsert, Elt);
      return this->begin() + InsertElt;
    }

    assert(this->isReferenceToStorage(I) &&
           "Insertion iterator is out of bounds.");

    // Ensure there is enough space, and get the (maybe updated) address of
    // Elt.
    const T *EltPtr = this->reserveForParamAndGetAddress(Elt, NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin() + InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end() - I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(I, OldEnd - NumToInsert, OldEnd);

      // If we just moved the element we're inserting, be sure to update
      // the reference (never happens if TakesParamByValue).
      if (!TakesParamByValue && I <= EltPtr && EltPtr < this->end())
        EltPtr += NumToInsert;

      std::fill_n(I, NumToInsert, *EltPtr);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->set_size(this->size() + NumToInsert);
    size_t NumOverwritten = OldEnd - I;
    this->uninitialized_move(I, OldEnd, this->end() - NumOverwritten);

    // If we just moved the element we're inserting, be sure to update
    // the reference (never happens if TakesParamByValue).
    if (!TakesParamByValue && I <= EltPtr && EltPtr < this->end())
      EltPtr += NumToInsert;

    // Replace the overwritten part.
    std::fill_n(I, NumOverwritten, *EltPtr);

    // Insert the non-overwritten middle part.
    std::uninitialized_fill_n(OldEnd, NumToInsert - NumOverwritten, *EltPtr);
    return I;
  }

  template <typename ItTy,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<ItTy>::iterator_category,
                std::input_iterator_tag>::value>>
  iterator insert(iterator I, ItTy From, ItTy To) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(From, To);
      return this->begin() + InsertElt;
    }

    assert(this->isReferenceToStorage(I) &&
           "Insertion iterator is out of bounds.");

    // Check that the reserve that follows doesn't invalidate the iterators.
    this->assertSafeToAddRange(From, To);

    size_t NumToInsert = std::distance(From, To);

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin() + InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end() - I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      std::move_backward(I, OldEnd - NumToInsert, OldEnd);

      std::copy(From, To, I);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->set_size(this->size() + NumToInsert);
    size_t NumOverwritten = OldEnd - I;
    this->uninitialized_move(I, OldEnd, this->end() - NumOverwritten);

    // Replace the overwritten part.
    for (T *J = I; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J;
      ++From;
    }

    // Insert the non-overwritten middle part.
    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  void insert(iterator I, std::initializer_list<T> IL) {
    insert(I, IL.begin(), IL.end());
  }

  template <typename... ArgTypes>
  reference emplace_back(ArgTypes &&...Args) {
    if (this->size() >= this->capacity())
      return this->growAndEmplaceBack(std::forward<ArgTypes>(Args)...);

    ::new (reinterpret_cast<void *>(this->end()))
        T(std::forward<ArgTypes>(Args)...);
    this->set_size(this->size() + 1);
    return this->back();
  }

  small_vector_impl &operator=(const small_vector_impl &RHS);

  small_vector_impl &operator=(small_vector_impl &&RHS);

  bool operator==(const small_vector_impl &RHS) const {
    if (this->size() != RHS.size()) return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
  }
  bool operator!=(const small_vector_impl &RHS) const {
    return !(*this == RHS);
  }

  bool operator<(const small_vector_impl &RHS) const {
    return std::lexicographical_compare(
        this->begin(), this->end(), RHS.begin(), RHS.end());
  }
};

template <typename T>
void small_vector_impl<T>::swap(small_vector_impl<T> &RHS) {
  if (this == &RHS) return;

  // We can only avoid copying elements if neither vector is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->Size, RHS.Size);
    std::swap(this->Capacity, RHS.Capacity);
    return;
  }
  this->reserve(RHS.size());
  RHS.reserve(this->size());

  // Swap the shared elements.
  size_t NumShared = this->size();
  if (NumShared > RHS.size()) NumShared = RHS.size();
  for (size_type i = 0; i != NumShared; ++i) std::swap((*this)[i], RHS[i]);

  // Copy over the extra elts.
  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin() + NumShared, this->end(), RHS.end());
    RHS.set_size(RHS.size() + EltDiff);
    this->destroy_range(this->begin() + NumShared, this->end());
    this->set_size(NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    this->uninitialized_copy(RHS.begin() + NumShared, RHS.end(), this->end());
    this->set_size(this->size() + EltDiff);
    this->destroy_range(RHS.begin() + NumShared, RHS.end());
    RHS.set_size(NumShared);
  }
}

template <typename T>
small_vector_impl<T> &small_vector_impl<T>::operator=(
    const small_vector_impl<T> &RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin() + RHSSize, this->begin());
    else
      NewEnd = this->begin();

    // Destroy excess elements.
    this->destroy_range(NewEnd, this->end());

    // Trim.
    this->set_size(RHSSize);
    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: don't do this if they're efficiently moveable.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->clear();
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::copy(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  // Copy construct the new elements in place.
  this->uninitialized_copy(
      RHS.begin() + CurSize, RHS.end(), this->begin() + CurSize);

  // Set end.
  this->set_size(RHSSize);
  return *this;
}

template <typename T>
small_vector_impl<T> &small_vector_impl<T>::operator=(
    small_vector_impl<T> &&RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If the RHS isn't small, clear this vector and then steal its buffer.
  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall()) free(this->begin());
    this->BeginX = RHS.BeginX;
    this->Size = RHS.Size;
    this->Capacity = RHS.Capacity;
    RHS.resetToSmall();
    return *this;
  }

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd = this->begin();
    if (RHSSize) NewEnd = std::move(RHS.begin(), RHS.end(), NewEnd);

    // Destroy excess elements and trim the bounds.
    this->destroy_range(NewEnd, this->end());
    this->set_size(RHSSize);

    // Clear the RHS.
    RHS.clear();

    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: this may not actually make any sense if we can efficiently move
  // elements.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->clear();
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::move(RHS.begin(), RHS.begin() + CurSize, this->begin());
  }

  // Move-construct the new elements in place.
  this->uninitialized_move(
      RHS.begin() + CurSize, RHS.end(), this->begin() + CurSize);

  // Set end.
  this->set_size(RHSSize);

  RHS.clear();
  return *this;
}

/// Storage for the small_vector elements.  This is specialized for the N=0 case
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct small_vector_storage {
  alignas(T) char InlineElts[N * sizeof(T)];
};

/// We need the storage to be properly aligned even for small-size of 0 so that
/// the pointer math in \a small_vector_template_common::getFirstEl() is
/// well-defined.
template <typename T>
struct alignas(T) small_vector_storage<T, 0> {};

/// Forward declaration of small_vector so that
/// calculateSmallVectorDefaultInlinedElements can reference
/// `sizeof(small_vector<T, 0>)`.
template <typename T, unsigned N>
class small_vector;

/// Helper class for calculating the default number of inline elements for
/// `small_vector<T>`.
///
/// This should be migrated to a constexpr function when our minimum
/// compiler support is enough for multi-statement constexpr functions.
template <typename T>
struct CalculateSmallVectorDefaultInlinedElements {
  // Parameter controlling the default number of inlined elements
  // for `small_vector<T>`.
  //
  // The default number of inlined elements ensures that
  // 1. There is at least one inlined element.
  // 2. `sizeof(small_vector<T>) <= kPreferredSmallVectorSizeof` unless
  // it contradicts 1.
  static constexpr size_t kPreferredSmallVectorSizeof = 64;

  // static_assert that sizeof(T) is not "too big".
  //
  // Because our policy guarantees at least one inlined element, it is possible
  // for an arbitrarily large inlined element to allocate an arbitrarily large
  // amount of inline storage. We generally consider it an antipattern for a
  // small_vector to allocate an excessive amount of inline storage, so we want
  // to call attention to these cases and make sure that users are making an
  // intentional decision if they request a lot of inline storage.
  //
  // We want this assertion to trigger in pathological cases, but otherwise
  // not be too easy to hit. To accomplish that, the cutoff is actually somewhat
  // larger than kPreferredSmallVectorSizeof (otherwise,
  // `small_vector<small_vector<T>>` would be one easy way to trip it, and that
  // pattern seems useful in practice).
  //
  // One wrinkle is that this assertion is in theory non-portable, since
  // sizeof(T) is in general platform-dependent. However, we don't expect this
  // to be much of an issue, because most LLVM development happens on 64-bit
  // hosts, and therefore sizeof(T) is expected to *decrease* when compiled for
  // 32-bit hosts, dodging the issue. The reverse situation, where development
  // happens on a 32-bit host and then fails due to sizeof(T) *increasing* on a
  // 64-bit host, is expected to be very rare.
  static_assert(
      sizeof(T) <= 256,
      "You are trying to use a default number of inlined elements for "
      "`small_vector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `small_vector<T, N>` to make "
      "sure you really want that much inline storage.");

  // Discount the size of the header itself when calculating the maximum inline
  // bytes.
  static constexpr size_t PreferredInlineBytes =
      kPreferredSmallVectorSizeof - sizeof(small_vector<T, 0>);
  static constexpr size_t NumElementsThatFit = PreferredInlineBytes / sizeof(T);
  static constexpr size_t value =
      NumElementsThatFit == 0 ? 1 : NumElementsThatFit;
};

/// This is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  It contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number of
/// elements is below that threshold.  This allows normal "small" cases to be
/// fast without losing generality for large inputs.
///
/// \note
/// In the absence of a well-motivated choice for the number of inlined
/// elements \p N, it is recommended to use \c small_vector<T> (that is,
/// omitting the \p N). This will choose a default number of inlined elements
/// reasonable for allocation on the stack (for example, trying to keep \c
/// sizeof(small_vector<T>) around 64 bytes).
///
/// \warning This does not attempt to be exception safe.
///
/// \see https://llvm.org/docs/ProgrammersManual.html#llvm-adt-smallvector-h
template <typename T,
          unsigned N = CalculateSmallVectorDefaultInlinedElements<T>::value>
class small_vector : public small_vector_impl<T>, small_vector_storage<T, N> {
 public:
  small_vector() : small_vector_impl<T>(N) {}

  ~small_vector() {
    // Destroy the constructed elements in the vector.
    this->destroy_range(this->begin(), this->end());
  }

  explicit small_vector(size_t Size, const T &Value = T())
      : small_vector_impl<T>(N) {
    this->assign(Size, Value);
  }

  template <typename ItTy,
            typename = std::enable_if_t<std::is_convertible<
                typename std::iterator_traits<ItTy>::iterator_category,
                std::input_iterator_tag>::value>>
  small_vector(ItTy S, ItTy E) : small_vector_impl<T>(N) {
    this->append(S, E);
  }

  template <typename RangeTy>
  explicit small_vector(const iterator_range<RangeTy> &R)
      : small_vector_impl<T>(N) {
    this->append(R.begin(), R.end());
  }

  small_vector(std::initializer_list<T> IL) : small_vector_impl<T>(N) {
    this->assign(IL);
  }

  small_vector(const small_vector &RHS) : small_vector_impl<T>(N) {
    if (!RHS.empty()) small_vector_impl<T>::operator=(RHS);
  }

  small_vector &operator=(const small_vector &RHS) {
    small_vector_impl<T>::operator=(RHS);
    return *this;
  }

  small_vector(small_vector &&RHS) : small_vector_impl<T>(N) {
    if (!RHS.empty()) small_vector_impl<T>::operator=(::std::move(RHS));
  }

  explicit small_vector(small_vector_impl<T> &&RHS) : small_vector_impl<T>(N) {
    if (!RHS.empty()) small_vector_impl<T>::operator=(::std::move(RHS));
  }

  small_vector &operator=(small_vector &&RHS) {
    small_vector_impl<T>::operator=(::std::move(RHS));
    return *this;
  }

  small_vector &operator=(small_vector_impl<T> &&RHS) {
    small_vector_impl<T>::operator=(::std::move(RHS));
    return *this;
  }

  small_vector &operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

template <typename T, unsigned N>
inline size_t capacity_in_bytes(const small_vector<T, N> &X) {
  return X.capacity_in_bytes();
}

/// Given a range of type R, iterate the entire range and return a
/// small_vector with elements of the vector.  This is useful, for example,
/// when you want to iterate a range and then sort the results.
template <unsigned Size, typename R>
small_vector<typename std::remove_const<typename std::remove_reference<
                 decltype(*std::begin(std::declval<R &>()))>::type>::type,
             Size>
to_vector(R &&Range) {
  return {std::begin(Range), std::end(Range)};
}

inline void *safe_malloc(size_t Sz) {
  void *Result = std::malloc(Sz);
  if (Result == nullptr) {
    // It is implementation-defined whether allocation occurs if the space
    // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
    // non-zero, if the space requested was zero.
    if (Sz == 0) return safe_malloc(1);
    throw std::bad_alloc();
  }
  return Result;
}

inline void *safe_calloc(size_t Count, size_t Sz) {
  void *Result = std::calloc(Count, Sz);
  if (Result == nullptr) {
    // It is implementation-defined whether allocation occurs if the space
    // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
    // non-zero, if the space requested was zero.
    if (Count == 0 || Sz == 0) return safe_malloc(1);
    throw std::bad_alloc();
  }
  return Result;
}

inline void *safe_realloc(void *Ptr, size_t Sz) {
  void *Result = std::realloc(Ptr, Sz);
  if (Result == nullptr) {
    // It is implementation-defined whether allocation occurs if the space
    // requested is zero (ISO/IEC 9899:2018 7.22.3). Retry, requesting
    // non-zero, if the space requested was zero.
    if (Sz == 0) return safe_malloc(1);
    throw std::bad_alloc();
  }
  return Result;
}

// Check that no bytes are wasted and everything is well-aligned.

struct Struct16B {
  alignas(16) void *X;
};
struct Struct32B {
  alignas(32) void *X;
};

static_assert(sizeof(small_vector<void *, 0>) ==
                  sizeof(unsigned) * 2 + sizeof(void *),
              "wasted space in small_vector size 0");
static_assert(alignof(small_vector<Struct16B, 0>) >= alignof(Struct16B),
              "wrong alignment for 16-byte aligned T");
static_assert(alignof(small_vector<Struct32B, 0>) >= alignof(Struct32B),
              "wrong alignment for 32-byte aligned T");
static_assert(sizeof(small_vector<Struct16B, 0>) >= alignof(Struct16B),
              "missing padding for 16-byte aligned T");
static_assert(sizeof(small_vector<Struct32B, 0>) >= alignof(Struct32B),
              "missing padding for 32-byte aligned T");
static_assert(sizeof(small_vector<void *, 1>) ==
                  sizeof(unsigned) * 2 + sizeof(void *) * 2,
              "wasted space in small_vector size 1");

static_assert(sizeof(small_vector<char, 0>) ==
                  sizeof(void *) * 2 + sizeof(void *),
              "1 byte elements have word-sized type for size and capacity");

/// Report that MinSize doesn't fit into this vector's size type. Throws
/// std::length_error or calls report_fatal_error.
static void report_size_overflow(size_t MinSize, size_t MaxSize);
static void report_size_overflow(size_t MinSize, size_t MaxSize) {
  std::string Reason = "small_vector unable to grow. Requested capacity (" +
                       std::to_string(MinSize) +
                       ") is larger than maximum value for size type (" +
                       std::to_string(MaxSize) + ")";
  throw std::length_error(Reason);
}

/// Report that this vector is already at maximum capacity. Throws
/// std::length_error or calls report_fatal_error.
static void report_at_maximum_capacity(size_t MaxSize);
static void report_at_maximum_capacity(size_t MaxSize) {
  std::string Reason =
      "small_vector capacity unable to grow. Already at maximum size " +
      std::to_string(MaxSize);
  throw std::length_error(Reason);
}

// Note: Moving this function into the header may cause performance regression.
template <class Size_T>
static size_t getNewCapacity(size_t MinSize, size_t OldCapacity) {
  constexpr size_t MaxSize = (std::numeric_limits<Size_T>::max)();

  // Ensure we can fit the new capacity.
  // This is only going to be applicable when the capacity is 32 bit.
  if (MinSize > MaxSize) report_size_overflow(MinSize, MaxSize);

  // Ensure we can meet the guarantee of space for at least one more element.
  // The above check alone will not catch the case where grow is called with a
  // default MinSize of 0, but the current capacity cannot be increased.
  // This is only going to be applicable when the capacity is 32 bit.
  if (OldCapacity == MaxSize) report_at_maximum_capacity(MaxSize);

  // In theory 2*capacity can overflow if the capacity is 64 bit, but the
  // original capacity would never be large enough for this to be a problem.
  size_t NewCapacity = 2 * OldCapacity + 1;  // Always grow.
  return (std::min)((std::max)(NewCapacity, MinSize), MaxSize);
}

// Note: Moving this function into the header may cause performance regression.
template <class Size_T>
void *small_vector_base<Size_T>::mallocForGrow(size_t MinSize,
                                               size_t TSize,
                                               size_t *NewCapacity) {
  *NewCapacity = getNewCapacity<Size_T>(MinSize, this->capacity());
  return safe_malloc((*NewCapacity) * TSize);
}

// Note: Moving this function into the header may cause performance regression.
template <class Size_T>
void small_vector_base<Size_T>::grow_pod(void *FirstEl,
                                         size_t MinSize,
                                         size_t TSize) {
  size_t NewCapacity = getNewCapacity<Size_T>(MinSize, this->capacity());
  void *NewElts;
  if (BeginX == FirstEl) {
    NewElts = safe_malloc(NewCapacity * TSize);

    // Copy the elements over.  No need to run dtors on PODs.
    memcpy(NewElts, this->BeginX, size() * TSize);
  } else {
    // If this wasn't grown from the inline copy, grow the allocated space.
    NewElts = safe_realloc(this->BeginX, NewCapacity * TSize);
  }

  this->BeginX = NewElts;
  this->Capacity = NewCapacity;
}

template class paddle::small_vector_base<uint32_t>;

// Disable the uint64_t instantiation for 32-bit builds.
// Both uint32_t and uint64_t instantiations are needed for 64-bit builds.
// This instantiation will never be used in 32-bit builds, and will cause
// warnings when sizeof(Size_T) > sizeof(size_t).
#if SIZE_MAX > UINT32_MAX
template class paddle::small_vector_base<uint64_t>;

// Assertions to ensure this #if stays in sync with SmallVectorSizeType.
static_assert(sizeof(SmallVectorSizeType<char>) == sizeof(uint64_t),
              "Expected small_vector_base<uint64_t> variant to be in use.");
#else
static_assert(sizeof(SmallVectorSizeType<char>) == sizeof(uint32_t),
              "Expected small_vector_base<uint32_t> variant to be in use.");
#endif

}  // namespace paddle

namespace std {

/// Implement std::swap in terms of small_vector swap.
template <typename T>
inline void swap(paddle::small_vector_impl<T> &LHS,
                 paddle::small_vector_impl<T> &RHS) {
  LHS.swap(RHS);
}

/// Implement std::swap in terms of small_vector swap.
template <typename T, unsigned N>
inline void swap(paddle::small_vector<T, N> &LHS,
                 paddle::small_vector<T, N> &RHS) {
  LHS.swap(RHS);
}

}  // namespace std
