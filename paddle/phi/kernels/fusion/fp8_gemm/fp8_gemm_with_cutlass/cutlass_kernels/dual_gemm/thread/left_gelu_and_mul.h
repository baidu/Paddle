/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
    typename ElementOutput_,  ///< Data type used to load and store tensors
    int Count,                ///< Number of elements computed per operation.
                ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                ///< but we use 64 or 32 sometimes when there are not enough
                ///< data to store
    typename ElementAccumulator_ = ElementOutput_,  ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_,  ///< Data type used to compute linear combination
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LeftGELUAndMul {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  static FloatRoundStyle const kRound = Round;

  struct Params {
    ElementCompute alpha;

    CUTLASS_HOST_DEVICE
    Params() : alpha(ElementCompute(1)) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha) : alpha(alpha) {}  // NOLINT
  };

 private:
  //
  // Data members
  //

  ElementCompute alpha_;
  ElementCompute beta_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  LeftGELUAndMul(Params const &params) { alpha_ = params.alpha; }  // NOLINT

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    assert(false);
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &lhs,
                            FragmentAccumulator const &rhs) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_to_compute;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        compute_to_output;

    ComputeFragment converted_lhs = accumulator_to_compute(lhs);
    ComputeFragment converted_rhs = accumulator_to_compute(rhs);

    cutlass::epilogue::thread::GELU_taylor<ComputeFragment> gelu;
    cutlass::multiplies<ComputeFragment> mul;
    auto gelu_lhs = gelu(converted_lhs);
    // return compute_to_output(mul(gelu_lhs, converted_rhs));
    auto tmp = mul(gelu_lhs, converted_rhs);
    return compute_to_output(mul(alpha_, tmp));
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(ElementAccumulator const &lhs,
                           ElementAccumulator const &rhs) const {
    ElementCompute convert_lhs(lhs);
    ElementCompute convert_rhs(rhs);
    cutlass::epilogue::thread::GELU_taylor<ElementCompute> gelu;
    cutlass::multiplies<ElementCompute> mul;
    auto gelu_lhs = gelu(convert_lhs);
    // return ElementOutput(mul(gelu_lhs, convert_rhs));
    auto tmp = mul(gelu_lhs, convert_rhs);
    return compute_to_output(mul(alpha_, tmp));
  }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass
