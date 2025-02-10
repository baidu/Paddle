// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/nvrtc/header_generator.h"

#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/enforce.h"

PD_DECLARE_string(cinn_nvcc_cmd_path);
PD_DECLARE_string(nvidia_package_dir);
PD_DECLARE_bool(nvrtc_compile_to_cubin);
PD_DECLARE_bool(cinn_nvrtc_cubin_with_fmad);

namespace cinn {
namespace backends {
namespace nvrtc {

HeaderGeneratorBase& JitSafeHeaderGenerator::GetInstance() {
  static JitSafeHeaderGenerator instance;
  return instance;
}

const size_t JitSafeHeaderGenerator::size() const {
  PADDLE_ENFORCE_EQ(include_names_.size(),
                    headers_.size(),
                    ::common::errors::InvalidArgument(
                        "Internal error in size of header files."));
  return include_names_.size();
}

JitSafeHeaderGenerator::JitSafeHeaderGenerator() {
  const auto& headers_map = ::jitify::detail::get_jitsafe_headers_map();

  std::ostringstream oss;

  for (const auto& pair : headers_map) {
    oss << pair.second.data() << "\n";
  }

  static const char* header_preinclude = oss.str().data();

  std::ofstream file("fitify_output.h");
  file << header_preinclude;
  file.close();

  include_names_.emplace_back("fitify_output.h");
  headers_.emplace_back(header_preinclude);

  const char* dummy_code = "";
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  std::string cc = "30";
  int major, minor;

  cudaError_t e1 =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaError_t e2 =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  if (e1 == cudaSuccess && e2 == cudaSuccess) {
    cc = std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
  }
  if (1) {
    compile_options.push_back("-arch=sm_" + cc);
    std::string enable_fmad =
        FLAGS_cinn_nvrtc_cubin_with_fmad ? "true" : "false";
    compile_options.push_back("--fmad=" + enable_fmad);
  } else {
    compile_options.push_back("-arch=compute_" + cc);
  }

  compile_options.push_back("--generate-precompiled-header fitify_output.h");
  compile_options.push_back("--pch-name=jitify_precompiled_header.pch");

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }

  nvrtcProgram prog;
  NVRTC_CALL(nvrtcCreateProgram(&prog,
                                dummy_code,
                                nullptr,
                                headers_.size(),
                                headers_.data(),
                                include_names_.data()));
  nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
