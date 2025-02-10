/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"

// Note(chenweihang): In order to be compatible with the original custom
// operator Tensor interface, only available to external users, the file
// cannot be included in paddle

namespace paddle {
// using several Tensor initialize functions in paddle namespace
using experimental::abs;
using experimental::acos;
using experimental::acosh;
using experimental::add;
using experimental::addmm;
using experimental::allclose;
using experimental::argsort;
using experimental::asin;
using experimental::asinh;
using experimental::atan;
using experimental::atan2;
using experimental::atanh;
// using experimental::baddbmm;
using experimental::bernoulli;
using experimental::ceil;
using experimental::cholesky;
using experimental::cholesky_solve;
using experimental::clip;
using experimental::concat;
using experimental::conj;
using experimental::cos;
using experimental::cosh;
using experimental::cross;
using experimental::det;
using experimental::diag;
using experimental::diagonal;
using experimental::digamma;
using experimental::dist;
using experimental::divide;
using experimental::dot;
using experimental::elu;
using experimental::empty;
using experimental::empty_like;
using experimental::equal;
using experimental::equal_all;
using experimental::erf;
using experimental::erfinv;
using experimental::exp;
using experimental::expand;
using experimental::expm1;
using experimental::flatten;
using experimental::flip;
using experimental::floor;
using experimental::floor_divide;
using experimental::fmax;
using experimental::fmin;
using experimental::frame;
using experimental::full;
using experimental::gather;
using experimental::gather_nd;
using experimental::gelu;
using experimental::greater_equal;
using experimental::greater_than;
using experimental::gumbel_softmax;
using experimental::hardswish;
using experimental::hardtanh;
using experimental::imag;
using experimental::increment;
using experimental::index_sample;
using experimental::is_empty;
using experimental::isclose;
using experimental::isfinite;
using experimental::isinf;
using experimental::isnan;
using experimental::kron;
using experimental::kthvalue;
using experimental::label_smooth;
using experimental::lerp;
using experimental::less_equal;
using experimental::less_than;
using experimental::lgamma;
using experimental::log;
using experimental::log10;
using experimental::log1p;
using experimental::log2;
using experimental::logit;
using experimental::masked_select;
using experimental::matmul;
using experimental::matrix_power;
using experimental::maximum;
using experimental::maxout;
using experimental::meshgrid;
using experimental::minimum;
using experimental::mode;
using experimental::multi_dot;
using experimental::multinomial;
using experimental::multiply;
using experimental::mv;
using experimental::nll_loss;
using experimental::not_equal;
using experimental::npu_identity;
using experimental::one_hot;
using experimental::ones;
using experimental::overlap_add;
using experimental::pixel_shuffle;
using experimental::poisson;
using experimental::put_along_axis;
using experimental::qr;
using experimental::real;
using experimental::reciprocal;
using experimental::relu;
using experimental::relu6;
using experimental::remainder;
using experimental::reshape;
using experimental::roll;
using experimental::round;
using experimental::rsqrt;
using experimental::scatter;
using experimental::scatter_nd_add;
using experimental::selu;
using experimental::send_u_recv;
using experimental::send_ue_recv;
using experimental::send_uv;
using experimental::sigmoid;
using experimental::sign;
using experimental::silu;
using experimental::sin;
using experimental::sinh;
using experimental::split;
using experimental::sqrt;
using experimental::square;
using experimental::stack;
using experimental::standard_gamma;
using experimental::strided_slice;
using experimental::subtract;
using experimental::swish;
using experimental::tanh;
using experimental::thresholded_relu;
using experimental::tile;
using experimental::trace;
using experimental::triangular_solve;
using experimental::tril;
using experimental::unbind;
using experimental::unique;
using experimental::unsqueeze;
using experimental::where;
using experimental::zeros;

}  // namespace paddle
