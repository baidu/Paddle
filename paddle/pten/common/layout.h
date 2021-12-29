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

#include "paddle/pten/api/ext/exception.h"
namespace paddle {
namespace experimental {

enum class DataLayout {
  UNDEFINED = 0,
  // TODO(chenweihang): keep ANY for compatibility, remove it later
  ANY = UNDEFINED,
  NHWC,
  NCHW,
  MKLDNN,
  NUM_DATA_LAYOUTS,
  // See Note [ Why we need ALL in baisc kernel key member? ]
  ALL_LAYOUT = UNDEFINED,
  // Note: Unify pten DataLayout and fluid::framework::DataLayout,
  // for compatible with fluid DataLayout, here need prefix `k`
  kNHWC = NHWC,
  kNCHW = NCHW,
  kAnyLayout = ANY,
  kMKLDNN = MKLDNN,  // all layouts supported by MKLDNN internally
};

}  // namespace experimental

// In order to be compatible with the fluid implementation
namespace framework {

using DataLayout = paddle::experimental::DataLayout;

inline DataLayout StringToDataLayout(const std::string& str) {
  std::string s(str);
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = toupper(s[i]);
  }

  if (s == "NHWC") {
    return DataLayout::kNHWC;
  } else if (s == "NCHW") {
    return DataLayout::kNCHW;
  } else if (s == "ANYLAYOUT") {
    return DataLayout::kAnyLayout;
  } else if (s == "MKLDNNLAYOUT") {
    return DataLayout::kMKLDNN;
  } else {
    PD_THROW("Unknown data layout type string: ", s, ".");
  }
}

inline std::string DataLayoutToString(const DataLayout& layout) {
  switch (layout) {
    case DataLayout::kNHWC:
      return "NHWC";
    case DataLayout::kNCHW:
      return "NCHW";
    case DataLayout::kAnyLayout:
      return "Undefined(AnyLayout)";
    case DataLayout::kMKLDNN:
      return "MKLDNN";
    default:
      PD_THROW("Unknown Data Layout type ", static_cast<int>(layout), ".");
  }
}
}  // namespace framework

namespace experimental {

inline std::ostream& operator<<(std::ostream& os, DataLayout layout) {
  os << framework::DataLayoutToString(layout);
  return os;
}

}  // namespace experimental
}  // namespace paddle

namespace pten {
using DataLayout = paddle::experimental::DataLayout;
}
