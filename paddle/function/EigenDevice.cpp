/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/function/EigenDevice.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __OSX__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// #include <android/log.h>

namespace paddle {

int GetAndroidCpuCount() {
#ifdef __ANDROID__
  FILE* fp = fopen("/sys/devices/system/cpu/possible", "r");
  if (!fp) {
    return 1;
  }
  int rank0, rank1;
  fscanf(fp, "%d-%d", &rank0, &rank1);
  //  __android_log_print(ANDROID_LOG_DEBUG, "Paddle",
  //          "rank0: %d, rank1: %d", rank0, rank1);
  fclose(fp);
  return rank1 + 1;
#else
  return 1;
#endif
}

int GetOSXCpuCount() {
// TODO(hjchen) test on osx device
#ifdef __OSX__
  int count = 0;
  size_t len = sizeof(int);
  sysctlbyname("hw.ncpu", &count, &len, NULL, 0);
  return count > 0 ? count : 1;
#else
  return 1;
#endif
}

int GetCpuCount() {
#ifdef __ANDROID__
  return GetAndroidCpuCount();
#endif
#ifdef __OSX__
  return GetOSXCpuCount();
#endif
  return 1;
}

const Eigen::ThreadPoolDevice& GetThreadPoolDevice() {
  int num_threads = ThreadsNumManager::Get();
  //  __android_log_print(ANDROID_LOG_DEBUG, "Paddle",
  //          "num_threads: %d", num_threads);
  static Eigen::ThreadPool tp(num_threads);
  static Eigen::ThreadPoolDevice device(&tp, num_threads);
  return device;
}

}  // namespace paddle
