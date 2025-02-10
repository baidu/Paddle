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

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <ctime>

#include <fstream>
#include <iostream>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/nvrtc/header_generator.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
PD_DECLARE_string(cinn_nvcc_cmd_path);
PD_DECLARE_string(nvidia_package_dir);
PD_DECLARE_bool(nvrtc_compile_to_cubin);
PD_DECLARE_bool(cinn_nvrtc_cubin_with_fmad);

namespace cinn {
namespace backends {
namespace nvrtc {

static bool TryLocatePath(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::vector<std::string> GetNvidiaAllIncludePath(
    const std::string& nvidia_package_dir) {
  std::vector<std::string> include_paths;
  const std::string delimiter = "/";
  // Expand this list if necessary.
  const std::vector<std::string> sub_modules = {"cublas",
                                                "cudnn",
                                                "cufft",
                                                "cusparse",
                                                "cusolver",
                                                "cuda_nvrtc",
                                                "curand",
                                                "cuda_runtime"};
  for (auto& sub_module : sub_modules) {
    std::string path =
        nvidia_package_dir + delimiter + sub_module + delimiter + "include";
    include_paths.push_back(path);
  }
  return include_paths;
}

std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  if (runtime::CanUseNvccCompiler()) {
    return CompileWithNvcc(code);
  }
  return CompileCudaSource(code, include_headers);
}

Compiler::Compiler() {
  if (FLAGS_nvrtc_compile_to_cubin) {
#if CUDA_VERSION >= 11010
    compile_to_cubin_ = true;
#endif
  }
  VLOG(4) << "FLAGS_nvrtc_compile_to_cubin: " << FLAGS_nvrtc_compile_to_cubin
          << ", compile_to_cubin_: " << compile_to_cubin_;
}

bool Compiler::compile_to_cubin() { return compile_to_cubin_; }

std::vector<std::string> Compiler::FindCUDAIncludePaths() {
  const std::string delimiter = "/";
  std::string cuda_include_path;
  const char* cuda_path_env = std::getenv("CUDA_PATH");
  if (cuda_path_env != nullptr) {
    cuda_include_path += cuda_path_env;
    cuda_include_path += delimiter + "include";
    VLOG(4) << "FindCUDAIncludePaths from CUDA_PATH: " << cuda_include_path;
    return {cuda_include_path};
  }

#if defined(__linux__)
  if (!FLAGS_nvidia_package_dir.empty() &&
      TryLocatePath(FLAGS_nvidia_package_dir)) {
    VLOG(4) << "FindCUDAIncludePaths from nvidia_package_dir: "
            << FLAGS_nvidia_package_dir;
    return GetNvidiaAllIncludePath(FLAGS_nvidia_package_dir);
  }

  cuda_include_path = "/usr/local/cuda/include";
  if (TryLocatePath(cuda_include_path)) {
    VLOG(4) << "FindCUDAIncludePaths from " << cuda_include_path;
    return {cuda_include_path};
  }
#endif
  std::stringstream ss;
  ss << "Cannot find cuda include path."
     << "CUDA_PATH is not set or CUDA is not installed in the default "
        "installation path."
     << "In other than linux, it is necessary to set CUDA_PATH.";
  PADDLE_THROW(::common::errors::Fatal(ss.str()));
  return {cuda_include_path};
}

std::vector<std::string> Compiler::FindCINNRuntimeIncludePaths() {
  return {Context::Global().runtime_include_dir()};
}

std::string Compiler::CompileCudaSource(const std::string& code,
                                        bool include_headers) {
  const auto& header_gen = JitSafeHeaderGenerator::GetInstance();
  std::vector<std::string> compile_options;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;
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
  if (compile_to_cubin_) {
    compile_options.push_back("-arch=sm_" + cc);
    std::string enable_fmad =
        FLAGS_cinn_nvrtc_cubin_with_fmad ? "true" : "false";
    compile_options.push_back("--fmad=" + enable_fmad);
  } else {
    compile_options.push_back("、" + cc);
  }
  compile_options.push_back("-std=c++14");
  compile_options.push_back("-default-device");

  if (include_headers) {  // prepare include headers
    auto cuda_headers = FindCUDAIncludePaths();
    auto cinn_headers = FindCINNRuntimeIncludePaths();
    std::vector<std::string> include_paths;
    for (auto& header : cuda_headers) {
      VLOG(5) << "add include-path: " << header;
      include_paths.push_back("--include-path=" + header);
    }
    for (auto& header : cinn_headers) {
      include_paths.push_back("--include-path=" + header);
    }
    compile_options.insert(
        std::end(compile_options), include_paths.begin(), include_paths.end());
    compile_options.push_back(
        "--use-precompiled-header=jitify_precompiled_header.pch");
  }

  for (const auto& option : compile_options) {
    param_cstrings.push_back(option.c_str());
  }
  VLOG(3) << "compile options: " << utils::Join(compile_options, " ");

  // std::cout << "================= header time 1================" <<
  // std::endl; std::time_t t = std::time(0);   // get time now std::tm* now =
  // std::localtime(&t); std::cout << now->tm_hour << ':'
  //           << now->tm_min << ':'
  //           << now->tm_sec << "\n";
  // NVRTC_CALL(nvrtcCreateProgram(&prog,
  //                               code.c_str(),
  //                               nullptr,
  //                               header_gen.size(),
  //                               header_gen.headers().data(),
  //                               header_gen.include_names().data()));
  // std::cout << "================= header time 2================" <<
  // std::endl; t = std::time(0);   // get time now now = std::localtime(&t);
  // std::cout << now->tm_hour << ':'
  //           << now->tm_min << ':'
  //           << now->tm_sec << "\n";
  std::cout << "nvrtcCreateProgram done" << std::endl;

  nvrtcResult compile_res =
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  // std::cout << "================= header time 3================" <<
  // std::endl; t = std::time(0);   // get time now now = std::localtime(&t);
  // std::cout << now->tm_hour << ':'
  //           << now->tm_min << ':'
  //           << now->tm_sec << "\n";

  {  // get log
    size_t log_size;
    NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));

    PADDLE_ENFORCE_EQ(
        compile_res,
        NVRTC_SUCCESS,
        ::common::errors::Fatal("NVRTC compilation failed: %s", log));
  }

  size_t size;
  std::string data;
  if (compile_to_cubin_) {
    NVRTC_CALL(nvrtcGetCUBINSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetCUBIN(prog, &data[0]));
  } else {
    NVRTC_CALL(nvrtcGetPTXSize(prog, &size));
    data.resize(size);
    NVRTC_CALL(nvrtcGetPTX(prog, &data[0]));
  }

  NVRTC_CALL(nvrtcDestroyProgram(&prog));
  return data;
}

std::string Compiler::CompileWithNvcc(const std::string& cuda_c) {
  // read dir source
  std::string dir = "./source";
  if (access(dir.c_str(), 0) == -1) {
    PADDLE_ENFORCE_NE(
        mkdir(dir.c_str(), 7),
        -1,
        ::common::errors::PermissionDenied(
            "Failed to create directory %s. Please check the permissions.",
            dir.c_str()));
  }

  // get unique prefix name
  prefix_name_ = dir + "/" + cinn::common::UniqName("rtc_tmp");

  auto cuda_c_file = prefix_name_ + ".cu";
  std::ofstream ofs(cuda_c_file, std::ios::out);
  PADDLE_ENFORCE_EQ(ofs.is_open(),
                    true,
                    ::common::errors::Unavailable(
                        "Failed to open file %s. Please check if the file path "
                        "is correct and the file is accessible.",
                        cuda_c_file.c_str()));
  ofs << cuda_c;
  ofs.close();

  CompileToPtx();
  CompileToCubin();

  return prefix_name_ + ".cubin";
}

// std::string Compiler::GetPtx() { return ReadFile(prefix_name_ + ".ptx",
// std::ios::in); }

void Compiler::CompileToPtx() {
  auto include_dir = cinn::common::Context::Global().runtime_include_dir();
  std::string include_dir_str = "";
  for (auto dir : include_dir) {
    if (include_dir_str.empty()) {
      include_dir_str = dir;
    } else {
      include_dir_str += ":" + dir;
    }
  }

  std::string options = std::string("export PATH=") + FLAGS_cinn_nvcc_cmd_path +
                        std::string(":$PATH && nvcc -std=c++14 --ptx -O3 -I ") +
                        include_dir_str;
  options += " -arch=" + GetDeviceArch();
  options += " -o " + prefix_name_ + ".ptx";
  options += " " + prefix_name_ + ".cu";

  VLOG(2) << "Nvcc Compile Options : " << options;
  PADDLE_ENFORCE_EQ(
      system(options.c_str()),
      0,
      ::common::errors::InvalidArgument("Failed to execute command: %s. Please "
                                        "check the command and try again.",
                                        options.c_str()));
}

void Compiler::CompileToCubin() {
  std::string options = std::string("export PATH=") + FLAGS_cinn_nvcc_cmd_path +
                        std::string(":$PATH && nvcc --cubin -O3");
  options += " -arch=" + GetDeviceArch();
  options += " -o " + prefix_name_ + ".cubin";
  options += " " + prefix_name_ + ".ptx";

  VLOG(2) << "Nvcc Compile Options : " << options;
  PADDLE_ENFORCE_EQ(
      system(options.c_str()),
      0,
      ::common::errors::InvalidArgument("Failed to execute command: %s. Please "
                                        "check the command and try again.",
                                        options.c_str()));
}

std::string Compiler::GetDeviceArch() {
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
          cudaSuccess &&
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) ==
          cudaSuccess) {
    return "sm_" + std::to_string(major) + std::to_string(minor);
  } else {
    LOG(WARNING) << "cannot detect compute capability from your device, "
                 << "fall back to compute_30.";
    return "sm_30";
  }
}

std::string Compiler::ReadFile(const std::string& file_name,
                               std::ios_base::openmode mode) {
  // open cubin file
  std::ifstream ifs(file_name, mode);
  PADDLE_ENFORCE_EQ(ifs.is_open(),
                    true,
                    ::common::errors::Unavailable(
                        "Failed to open file %s. Please check if the file path "
                        "is correct and the file is accessible.",
                        file_name.c_str()));
  ifs.seekg(std::ios::end);
  auto len = ifs.tellg();
  ifs.seekg(0);

  // read cubin file
  std::string file_data(len, ' ');
  ifs.read(&file_data[0], len);
  ifs.close();
  return std::move(file_data);
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
