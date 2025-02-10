/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include <dirent.h>

#include <codecvt>
#include <cstdlib>
#include <string>
#include <vector>
#include "paddle/phi/backends/dynload/cupti_lib_path.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/enforce.h"

#if defined(_WIN32)
#include <windows.h>
#endif

// TODO(wilber): The phi computing library requires a component to manage flags
// (maybe not use gflags).
#include "glog/logging.h"

#include "paddle/common/flags.h"

COMMON_DECLARE_string(cudnn_dir);
COMMON_DECLARE_string(cuda_dir);
COMMON_DECLARE_string(cublas_dir);
COMMON_DECLARE_string(nccl_dir);
COMMON_DECLARE_string(cupti_dir);
COMMON_DECLARE_string(tensorrt_dir);
COMMON_DECLARE_string(mklml_dir);
COMMON_DECLARE_string(lapack_dir);
COMMON_DECLARE_string(mkl_dir);
COMMON_DECLARE_string(op_dir);
COMMON_DECLARE_string(cusparselt_dir);
COMMON_DECLARE_string(curand_dir);
COMMON_DECLARE_string(cusolver_dir);
COMMON_DECLARE_string(cusparse_dir);
COMMON_DECLARE_string(win_cuda_bin_dir);
#ifdef PADDLE_WITH_HIP

PHI_DEFINE_string(miopen_dir,
                  "",
                  "Specify path for loading libMIOpen.so. For instance, "
                  "/opt/rocm/miopen/lib. If empty [default], dlopen "
                  "will search miopen from LD_LIBRARY_PATH");

PHI_DEFINE_string(rocm_dir,
                  "",
                  "Specify path for loading rocm library, such as librocblas, "
                  "libmiopen, libhipsparse. For instance, /opt/rocm/lib. "
                  "If default, dlopen will search rocm from LD_LIBRARY_PATH");

PHI_DEFINE_string(rccl_dir,
                  "",
                  "Specify path for loading rccl library, such as librccl.so. "
                  "For instance, /opt/rocm/rccl/lib. If default, "
                  "dlopen will search rccl from LD_LIBRARY_PATH");
#endif

#ifdef PADDLE_WITH_XPU
PD_DEFINE_string(xpti_dir, "", "Specify path for loading libxpti.so.");
#endif

namespace phi::dynload {

struct PathNode {
  PathNode() = default;
  std::string path = "";
};

static constexpr char cupti_lib_path[] = CUPTI_LIB_PATH;  // NOLINT

// NOTE: In order to adapt to the default installation path of cuda
#if defined(_WIN32) && defined(PADDLE_WITH_CUDA)
static constexpr char cuda_lib_path[] = CUDA_TOOLKIT_ROOT_DIR "/bin";
#else
static constexpr char cuda_lib_path[] = "/usr/local/cuda/lib64";  // NOLINT
#endif

static PathNode s_py_site_pkg_path;

#if defined(_WIN32) && defined(PADDLE_WITH_CUDA)
static constexpr char* win_cudnn_lib = "cudnn64_" CUDNN_MAJOR_VERSION ".dll";
static constexpr char* win_cublas_lib =
    "cublas64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cublas64_" CUDA_VERSION_MAJOR ".dll";
#if CUDA_VERSION >= 11000
static constexpr char* win_curand_lib =
    "curand64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;curand64_" CUDA_VERSION_MAJOR ".dll;curand64_10.dll";
static constexpr char* win_nvjpeg_lib =
    "nvjpeg64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;nvjpeg64_" CUDA_VERSION_MAJOR ".dll;nvjpeg64_10.dll";
static constexpr char* win_cusolver_lib =
    "cusolver64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cusolver64_" CUDA_VERSION_MAJOR
    ".dll;cusolver64_11.dll;cusolver64_10.dll";
static constexpr char* win_cusparse_lib =
    "cusparse64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cusparse64_" CUDA_VERSION_MAJOR ".dll;cusparse64_10.dll";
static constexpr char* win_cufft_lib =
    "cufft64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cufft64_" CUDA_VERSION_MAJOR ".dll;cufft64_11.dll;cufft64_10.dll";
#else
static constexpr char* win_curand_lib =
    "curand64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;curand64_" CUDA_VERSION_MAJOR ".dll";
static constexpr char* win_nvjpeg_lib =
    "nvjpeg64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;nvjpeg64_" CUDA_VERSION_MAJOR ".dll";
static constexpr char* win_cusolver_lib =
    "cusolver64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cusolver64_" CUDA_VERSION_MAJOR ".dll";
static constexpr char* win_cusparse_lib =
    "cusparse64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cusparse64_" CUDA_VERSION_MAJOR ".dll";
static constexpr char* win_cufft_lib =
    "cufft64_" CUDA_VERSION_MAJOR CUDA_VERSION_MINOR
    ".dll;cufft64_" CUDA_VERSION_MAJOR ".dll";
#endif  // CUDA_VERSION
#endif

static inline std::string join(const std::string& part1,
                               const std::string& part2) {
// directory separator
#if defined(_WIN32)
  const char sep = '\\';
#else
  const char sep = '/';
#endif
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

static inline std::vector<std::string> split(
    const std::string& str, const std::string separator = " ") {
  std::vector<std::string> str_list;
  std::string::size_type firstPos = 0;
  firstPos = str.find_first_not_of(separator, 0);
  std::string::size_type lastPos = 0;
  lastPos = str.find_first_of(separator, firstPos);
  while (std::string::npos != firstPos && std::string::npos != lastPos) {
    str_list.push_back(str.substr(firstPos, lastPos - firstPos));
    firstPos = str.find_first_not_of(separator, lastPos);
    lastPos = str.find_first_of(separator, firstPos);
  }
  if (std::string::npos == lastPos) {
    str_list.push_back(str.substr(firstPos, lastPos - firstPos));
  }
  return str_list;
}

void SetPaddleLibPath(const std::string& py_site_pkg_path) {
  s_py_site_pkg_path.path = py_site_pkg_path;
  VLOG(3) << "Set paddle lib path : " << py_site_pkg_path;
}

static inline void* GetDsoHandleFromSpecificPath(const std::string& spec_path,
                                                 const std::string& dso_name,
                                                 int dynload_flags) {
  void* dso_handle = nullptr;
  if (!spec_path.empty()) {
    // search xxx.so from custom path
    VLOG(3) << "Try to find library: " << dso_name
            << " from specific path: " << spec_path;
    std::string dso_path = join(spec_path, dso_name);
    dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  }
  return dso_handle;
}

static inline std::string FindLibAbsolutePath(const std::string& directory,
                                              const std::string& filename) {
  DIR* dir = opendir(directory.c_str());
  struct dirent* ent;

  if (dir != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      if (ent->d_type == DT_REG || ent->d_type == DT_LNK) {
        if (filename == std::string(ent->d_name)) {
          closedir(dir);
          return join(directory, ent->d_name);
        }
      } else if (ent->d_type == DT_DIR) {
        if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0) {
          std::string res =
              FindLibAbsolutePath(join(directory, ent->d_name) + "/", filename);
          if (!res.empty()) {
            closedir(dir);
            return res;
          }
        }
      }
    }
    closedir(dir);
  }
  return "";
}

static inline void* GetDsoHandleFromDefaultPath(const std::string& dso_path,
                                                int dynload_flags) {
  // default search from LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
  // and /usr/local/lib path
  void* dso_handle = dlopen(dso_path.c_str(), dynload_flags);
  VLOG(3) << "Try to find library: " << dso_path
          << " from default system path.";

// TODO(chenweihang): This path is used to search which libs?
// DYLD_LIBRARY_PATH is disabled after Mac OS 10.11 to
// bring System Integrity Projection (SIP), if dso_handle
// is null, search from default package path in Mac OS.
#if defined(__APPLE__) || defined(__OSX__)
#if defined(__arm__) || defined(__aarch64__)
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(FindLibAbsolutePath("/opt/homebrew/Cellar/", dso_path).c_str(),
               dynload_flags);
  }
#else
  if (nullptr == dso_handle) {
    dso_handle =
        dlopen(FindLibAbsolutePath("/usr/local/cuda/lib/", dso_path).c_str(),
               dynload_flags);
  }
#endif
#endif

  return dso_handle;
}

/*
 * We define three priorities for dynamic library search:
 *
 * First: Search for  path specified by the user
 * Second: Search the stheystem default path
 * Third: Search for a special path corresponding to
 *        a specific library to adapt to changes and easy to expand.
 */

static inline void* GetDsoHandleFromSearchPath(
    const std::string& config_path,
    const std::string& dso_name,
    bool throw_on_error = true,
    const std::vector<std::string>& extra_paths = std::vector<std::string>(),
    const std::string& warning_msg = std::string()) {
#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32
#if defined(_WIN32)
  std::vector<std::wstring> cuda_bin_search_path = {
      L"cublas",
      L"cuda_nvrtc",
      L"cuda_runtime",
      L"cudnn",
      L"cufft",
      L"curand",
      L"cusolver",
      L"cusparse",
      L"nvjitlink",
  };
  for (auto search_path : cuda_bin_search_path) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring win_path_wstring =
        converter.from_bytes(FLAGS_win_cuda_bin_dir);
    search_path = win_path_wstring + L"\\" + search_path + L"\\bin";
    AddDllDirectory(search_path.c_str());
  }
#endif
  std::vector<std::string> dso_names = split(dso_name, ";");
  void* dso_handle = nullptr;
  for (auto const& dso : dso_names) {
    // 1. search in user config path by FLAGS
    dso_handle = GetDsoHandleFromSpecificPath(config_path, dso, dynload_flags);
    // 2. search in system default path
    if (nullptr == dso_handle) {
      dso_handle = GetDsoHandleFromDefaultPath(dso, dynload_flags);
    }
    // 3. search in extra paths
    if (nullptr == dso_handle) {
      for (auto const& path : extra_paths) {
        VLOG(3) << "extra_paths: " << path;
        dso_handle = GetDsoHandleFromSpecificPath(path, dso, dynload_flags);
      }
    }
    if (nullptr != dso_handle) break;
  }

  // 4. [If Failed for All dso_names] logging warning if exists
  if (nullptr == dso_handle && !warning_msg.empty()) {
    LOG(WARNING) << warning_msg;
  }

  // 5. [If Failed for All dso_names] logging or throw error info
  if (nullptr == dso_handle) {
    auto error_msg =
        "The third-party dynamic library (%s) that Paddle depends on is not "
        "configured correctly. (error code is %s)\n"
        "  Suggestions:\n"
        "  1. Check if the third-party dynamic library (e.g. CUDA, CUDNN) "
        "is installed correctly and its version is matched with paddlepaddle "
        "you installed.\n"
        "  2. Configure third-party dynamic library environment variables as "
        "follows:\n"
        "  - Linux: set LD_LIBRARY_PATH by `export LD_LIBRARY_PATH=...`\n"
        "  - Windows: set PATH by `set PATH=XXX;%%PATH%%`\n"
        "  - Mac: set  DYLD_LIBRARY_PATH by `export DYLD_LIBRARY_PATH=...` "
        "[Note: After Mac OS 10.11, using the DYLD_LIBRARY_PATH is "
        "impossible unless System Integrity Protection (SIP) is disabled.]";
#if !defined(_WIN32)
    auto errorno = dlerror();
#else
    auto errorno = GetLastError();
#endif  // !_WIN32
    if (throw_on_error) {
      // NOTE: Special error report case, no need to change its format
      PADDLE_THROW(
          common::errors::PreconditionNotMet(error_msg, dso_name, errorno));
    } else {
      LOG(WARNING) << paddle::string::Sprintf(error_msg, dso_name, errorno);
    }
  }

  return dso_handle;
}

void* GetCublasDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cublas64_11.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cublas_lib, true, {cuda_lib_path});
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cublas64_12.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cublas_lib, true, {cuda_lib_path});
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#elif defined(__linux__) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublas.so.11");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublas.so");
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublas.so.12");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublas.so");
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "librocblas.so");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublas.so");
#endif
}

void* GetCublasLtDsoHandle() {
// APIs available after CUDA 10.1
#if defined(__linux__) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublasLt.so.11");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublasLt.so");
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublasLt.so.12");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cublas_dir, "libcublasLt.so");
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cublasLt64_11.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cublas_lib, true, {cuda_lib_path});
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cublasLt64_12.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cublas_lib, true, {cuda_lib_path});
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#elif !defined(__linux__) && defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10010
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcublasLt.so");
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libhipblaslt.so");
#else
  std::string warning_msg(
      "Your CUDA_VERSION less 10.1, not support CublasLt. "
      "If you want to use CublasLt, please upgrade CUDA and rebuild "
      "PaddlePaddle.");
  return nullptr;
#endif
}

void* GetCUDNNDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  std::string mac_warn_meg(
      "Note: [Recommend] copy cudnn into /usr/local/cuda/ \n "
      "For instance, sudo tar -xzf "
      "cudnn-7.5-osx-x64-v5.0-ga.tgz -C /usr/local \n sudo "
      "chmod a+r /usr/local/cuda/include/cudnn.h "
      "/usr/local/cuda/lib/libcudnn*");
  return GetDsoHandleFromSearchPath(
      FLAGS_cudnn_dir, "libcudnn.dylib", false, {}, mac_warn_meg);
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  std::string win_warn_meg(
      "Note: [Recommend] copy cudnn into CUDA installation directory. \n "
      "For instance, download cudnn-10.0-windows10-x64-v7.6.5.32.zip from "
      "NVIDIA's official website, \n"
      "then, unzip it and copy it into C:\\Program Files\\NVIDIA GPU Computing "
      "Toolkit\\CUDA\\v10.0\n"
      "You should do this according to your CUDA installation directory and "
      "CUDNN version.");
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12030) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, "cudnn64_8.dll", true, {cuda_lib_path}, win_warn_meg);
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cudnn_lib, true, {cuda_lib_path}, win_warn_meg);
#endif
  } else if (CUDA_VERSION >= 12030) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, "cudnn64_9.dll", true, {cuda_lib_path}, win_warn_meg);
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cudnn_lib, true, {cuda_lib_path}, win_warn_meg);
#endif
  }
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_miopen_dir, "libMIOpen.so", false);
#else
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
  if (CUDA_VERSION >= 12030) {
    return GetDsoHandleFromSearchPath(
        FLAGS_cudnn_dir, "libcudnn.so.9", false, {cuda_lib_path});
  } else {
    return GetDsoHandleFromSearchPath(
        FLAGS_cudnn_dir, "libcudnn.so.8", false, {cuda_lib_path});
  }
#else
  return GetDsoHandleFromSearchPath(
      FLAGS_cudnn_dir, "libcudnn.so", false, {cuda_lib_path});
#endif
#endif
}

void* GetCUPTIDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(
      FLAGS_cupti_dir, "libcupti.dylib", false, {cupti_lib_path});
#elif defined(__linux__) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(
        FLAGS_cupti_dir, "libcupti.so.11.8", false, {cupti_lib_path});
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cupti_dir, "libcupti.so", false, {cupti_lib_path});
#endif

  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(
        FLAGS_cupti_dir, "libcupti.so.12", false, {cupti_lib_path});
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cupti_dir, "libcupti.so", false, {cupti_lib_path});
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#else
  return GetDsoHandleFromSearchPath(
      FLAGS_cupti_dir, "libcupti.so", false, {cupti_lib_path});
#endif
}

void* GetCurandDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcurand.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
  return GetDsoHandleFromSearchPath(
      FLAGS_cuda_dir, "curand64_10.dll", true, {cuda_lib_path});
#else
  return GetDsoHandleFromSearchPath(
      FLAGS_cuda_dir, win_curand_lib, true, {cuda_lib_path});
#endif
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libhiprand.so");
#else
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
  return GetDsoHandleFromSearchPath(FLAGS_curand_dir, "libcurand.so.10");
#else
  return GetDsoHandleFromSearchPath(FLAGS_curand_dir, "libcurand.so");
#endif

#endif
}

#ifdef PADDLE_WITH_HIP
void* GetROCFFTDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "librocfft.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libhipfft.so");
#endif
}
#endif

void* GetNvjpegDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvjpeg.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  return GetDsoHandleFromSearchPath(
      FLAGS_cuda_dir, win_nvjpeg_lib, true, {cuda_lib_path});
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvjpeg.so");
#endif
}

void* GetCusolverDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
  return GetDsoHandleFromSearchPath(
      FLAGS_cuda_dir, "cusolver64_11.dll", true, {cuda_lib_path});
#else
  return GetDsoHandleFromSearchPath(
      FLAGS_cuda_dir, win_cusolver_lib, true, {cuda_lib_path});
#endif
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "librocsolver.so");
#else
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.so.11");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusolver.so");
#endif
#endif
}

void* GetCusparseDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusparse.dylib");
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cusparse64_11.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cusparse_lib, true, {cuda_lib_path});
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cusparse64_12.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cusparse_lib, true, {cuda_lib_path});
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#elif defined(__linux__) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cusparse_dir, "libcusparse.so.11");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cusparse_dir, "libcusparse.so");
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cusparse_dir, "libcusparse.so.12");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cusparse_dir, "libcusparse.so");
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer.");
    return nullptr;
  }
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "librocsparse.so");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcusparse.so");
#endif
}

void* GetNVRTCDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvrtc.dylib", false);
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libamdhip64.so", false);
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvrtc.so", false);
#endif
}

void* GetCUDADsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcuda.dylib", false);
#elif defined(PADDLE_WITH_HIP)
  return GetDsoHandleFromSearchPath(FLAGS_rocm_dir, "libamdhip64.so", false);
#elif defined(_WIN32)
  char system32_dir[MAX_PATH];
  GetSystemDirectory(system32_dir, MAX_PATH);
  return GetDsoHandleFromSearchPath(system32_dir, "nvcuda.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcuda.so", false);
#endif
}

void* GetWarpCTCDsoHandle() {
  std::string warpctc_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    warpctc_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(warpctc_dir, "warpctc.dll");
#else
  return GetDsoHandleFromSearchPath(warpctc_dir, "libwarpctc.so");
#endif
}

void* GetWarpRNNTDsoHandle() {
  std::string warprnnt_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    warprnnt_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(warprnnt_dir, "libwarprnnt.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(warprnnt_dir, "warprnnt.dll");
#else
  return GetDsoHandleFromSearchPath(warprnnt_dir, "libwarprnnt.so");
#endif
}

void* GetFlashAttnDsoHandle() {
  std::string flashattn_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    flashattn_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(flashattn_dir, "libflashattn.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(flashattn_dir, "flashattn.dll");
#else
  return GetDsoHandleFromSearchPath(flashattn_dir, "libflashattn.so");
#endif
}

void* GetFlashAttnV3DsoHandle() {
  std::string flashattn_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    flashattn_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(flashattn_dir, "libflashattnv3.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(flashattn_dir, "flashattnv3.dll");
#else
  return GetDsoHandleFromSearchPath(flashattn_dir, "libflashattnv3.so");
#endif
}

void* GetAfsApiDsoHandle() {
  std::string afsapi_dir = "";
  if (!s_py_site_pkg_path.path.empty()) {
    afsapi_dir = s_py_site_pkg_path.path;
  }
#if defined(__APPLE__) || defined(__OSX__) || defined(_WIN32)
  return NULL;
#else
  return GetDsoHandleFromSearchPath(afsapi_dir, "libafs-api-so.so");
#endif
}

void* GetNCCLDsoHandle() {
#ifdef PADDLE_WITH_HIP
  std::string warning_msg(
      "You may need to install 'rccl' from ROCM official website: "
      "https://rocmdocs.amd.com/en/latest/Installation_Guide/"
      "Installation-Guide.html before install PaddlePaddle.");
#else
  std::string warning_msg(
      "You may need to install 'nccl2' from NVIDIA official website: "
      "https://developer.nvidia.com/nccl/nccl-download "
      "before install PaddlePaddle.");
#endif

#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(
      FLAGS_nccl_dir, "libnccl.dylib", true, {}, warning_msg);
#elif defined(PADDLE_WITH_HIP) && defined(PADDLE_WITH_RCCL)
  return GetDsoHandleFromSearchPath(
      FLAGS_rccl_dir, "librccl.so", true, {}, warning_msg);
#else
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
  return GetDsoHandleFromSearchPath(
      FLAGS_nccl_dir, "libnccl.so;libnccl.so.2", true, {}, warning_msg);
#else
  return GetDsoHandleFromSearchPath(
      FLAGS_nccl_dir, "libnccl.so", true, {}, warning_msg);
#endif

#endif
}

void* GetTensorRtDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "nvinfer.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_tensorrt_dir, "libnvinfer.so");
#endif
}

void* GetMKLMLDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "libmklml_intel.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "mklml.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_mklml_dir, "libmklml_intel.so");
#endif
}

void* GetLAPACKDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
#if defined(__arm__) || defined(__aarch64__)
  return GetDsoHandleFromSearchPath(FLAGS_lapack_dir, "liblapack.dylib");
#else
  return GetDsoHandleFromSearchPath(FLAGS_lapack_dir, "liblapack.3.dylib");
#endif
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_lapack_dir, "liblapack.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_lapack_dir, "liblapack.so.3");
#endif
}

void* GetOpDsoHandle(const std::string& dso_name) {
  return GetDsoHandleFromSearchPath(FLAGS_op_dir, dso_name);
}

void* GetNvtxDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  PADDLE_THROW(common::errors::Unimplemented("Nvtx do not support Apple."));
#elif defined(_WIN32)
  PADDLE_THROW(common::errors::Unimplemented("Nvtx do not support Windows."));
#elif !defined(PADDLE_WITH_CUDA)
  PADDLE_THROW(
      common::errors::Unimplemented("Nvtx do not support without CUDA."));
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libnvToolsExt.so");
#endif
}

void* GetCUFFTDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcufft.dylib");
#elif defined(__linux__) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcufft.so.10");
#else
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcufft.so");
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcufft.so.11");
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer.");
    return nullptr;
  }
#elif defined(_WIN32) && defined(PADDLE_WITH_CUDA)
  if (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cufft64_10.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cufft_lib, true, {cuda_lib_path});
#endif
  } else if (CUDA_VERSION >= 12000 && CUDA_VERSION < 13000) {
#ifdef PADDLE_WITH_PIP_CUDA_LIBRARIES
    return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "cufft64_11.dll");
#else
    return GetDsoHandleFromSearchPath(
        FLAGS_cuda_dir, win_cufft_lib, true, {cuda_lib_path});
#endif
  } else {
    std::string warning_msg(
        "Your CUDA_VERSION is less than 11 or greater than 12, paddle "
        "temporarily no longer supports");
    return nullptr;
  }
#else
  return GetDsoHandleFromSearchPath(FLAGS_cuda_dir, "libcufft.so");
#endif
}

void* GetMKLRTDsoHandle() {
#if defined(__APPLE__) || defined(__OSX__)
  return GetDsoHandleFromSearchPath(FLAGS_mkl_dir, "libmkl_rt.dylib");
#elif defined(_WIN32)
  return GetDsoHandleFromSearchPath(FLAGS_mkl_dir, "mkl_rt.dll");
#else
  return GetDsoHandleFromSearchPath(FLAGS_mkl_dir, "libmkl_rt.so");
#endif
}

void* GetCusparseLtDsoHandle() {
// APIs available after CUDA 11.2
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
  return GetDsoHandleFromSearchPath(FLAGS_cusparselt_dir, "libcusparseLt.so");
#else
  std::string warning_msg(
      "Your CUDA_VERSION less 11.2, not support cusparseLt. "
      "If you want to use cusparseLt, please upgrade CUDA and rebuild "
      "PaddlePaddle.");
  return nullptr;
#endif
}

void* GetXPTIDsoHandle() {
#ifdef PADDLE_WITH_XPTI
  return GetDsoHandleFromSearchPath(FLAGS_xpti_dir, "libxpti.so");
#else
  return nullptr;
#endif
}
}  // namespace phi::dynload
