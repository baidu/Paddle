// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/profiler/cpu_overview.h"
#include "glog/logging.h"

namespace paddle {
namespace platform {

#ifdef _MSC_VER
static uint64_t FileTimeToUint64(FILETIME time) {
  uint64_t low_part = time.dwLowDateTime;
  uint64_t high_part = time.dwHighDateTime;
  uint64_t result = (high_part << 32) | low_part;
  return result;
}
#endif

void CPUOverview::RecordBeginTimeInfo() {
#ifdef _MSC_VER
  HANDLE process_handle = GetCurrentProcess();
  GetSystemTimeAsFileTime(&start_);
  GetSystemTimes(&system_idle_time_start_, &system_kernel_time_start_,
                 &system_user_time_start_);
  GetProcessTimes(handle, &process_creation_time_, &process_exit_time_,
                  &process_kernel_time_start_, &process_user_time_start_);

#else
  start_ = times(&process_tms_start_);
#define proc_path_size 1024
  static char proc_stat_path[proc_path_size] = "/proc/stat";
  FILE *stat_file = fopen(proc_stat_path, "r");
  char temp_str[200];
  uint64_t temp_lu;
  uint64_t tms_utime;
  uint64_t tms_stime;
  uint64_t idle_start;
  system_tms_start_.tms_utime = 0;
  system_tms_start_.tms_stime = 0;
  idle_start_ = 0;
  while (true) {
    int retval = fscanf(
        stat_file, "%s %" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64
                   "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64,
        temp_str, &tms_utime, &temp_lu, &tms_stime, &idle_start, &temp_lu,
        &temp_lu, &temp_lu, &temp_lu, &temp_lu, &temp_lu);
    if (std::string(temp_str).find("cpu") != 0) {
      break;
    }
    if (retval != 11) {
      return;
    }
    system_tms_start_.tms_utime += tms_utime;
    system_tms_start_.tms_stime += tms_stime;
    idle_start_ += idle_start;
  }
#endif
}

void CPUOverview::RecordEndTimeInfo() {
#ifdef _MSC_VER
  HANDLE process_handle = GetCurrentProcess();
  GetSystemTimeAsFileTime(&end_);
  GetSystemTimes(&system_idle_time_end_, &system_kernel_time_end_,
                 &system_user_time_end_);
  GetProcessTimes(handle, &process_creation_time_, &process_exit_time_,
                  &process_kernel_time_end_, &process_user_time_end_);
#else
  end_ = times(&process_tms_end_);
#define proc_path_size 1024
  static char proc_stat_path[proc_path_size] = "/proc/stat";
  FILE *stat_file = fopen(proc_stat_path, "r");
  char temp_str[200];
  uint64_t temp_lu;
  uint64_t tms_utime;
  uint64_t tms_stime;
  uint64_t idle_end;
  system_tms_end_.tms_utime = 0;
  system_tms_end_.tms_stime = 0;
  idle_end_ = 0;
  while (true) {
    int retval = fscanf(
        stat_file, "%s %" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64
                   "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64 "%" PRIu64,
        temp_str, &tms_utime, &temp_lu, &tms_stime, &idle_end, &temp_lu,
        &temp_lu, &temp_lu, &temp_lu, &temp_lu, &temp_lu);
    if (std::string(temp_str).find("cpu") != 0) {
      break;
    }
    if (retval != 11) {
      return;
    }
    system_tms_end_.tms_utime += tms_utime;
    system_tms_end_.tms_stime += tms_stime;
    idle_end_ += idle_end;
  }

#endif
}

float CPUOverview::GetCpuUtilization() {
  float cpu_utilization = 0.0;
#ifdef _MSC_VER
  uint64_t system_user_time_start = FileTimeToUint64(system_user_time_start_);
  uint64_t system_user_time_end = FileTimeToUint64(system_user_time_end_);
  uint64_t system_kernel_time_start =
      FileTimeToUint64(system_kernel_time_start_);
  uint64_t system_kernel_time_end = FileTimeToUint64(system_kernel_time_end_);
  uint64_t system_idle_time_start = FileTimeToUint64(system_idle_time_start_);
  uint64_t system_idle_time_end = FileTimeToUint64(system_idle_time_end_);
  float busy_time = (system_kernel_time_end - system_kernel_time_start) +
                    (system_user_time_end - system_user_time_start);
  float idle_time = system_idle_time_end - system_idle_time_start;
  cpu_utilization = busy_time / (busy_time + idle_time);
  LOG(INFO) << "CPU Utilization = " << cpu_utilization << std::endl;
#else
  // LOG(INFO) << "start time" << std::endl;
  // LOG(INFO) << "system user_time = " << system_tms_start_.tms_utime <<
  // "system kernel_time = " << system_tms_start_.tms_stime << std::endl;
  // LOG(INFO) << "process user_time = " <<  process_tms_start_.tms_utime <<
  // "process kernel_time = " << process_tms_start_.tms_stime << std::endl;
  // LOG(INFO) << "Idle time" << idle_start_ <<std::endl;
  // LOG(INFO) << "end time" << std::endl;
  // LOG(INFO) << "system user_time = " << system_tms_end_.tms_utime << "system
  // kernel_time = " << system_tms_end_.tms_stime << std::endl;
  // LOG(INFO) << "process user_time = " << process_tms_end_.tms_utime <<
  // "process kernel_time = " << process_tms_end_.tms_stime << std::endl;
  // LOG(INFO) << "Idle time" << idle_end_ <<std::endl;
  // LOG(INFO) << "Duration time" << std::endl;
  // LOG(INFO) << "system user_time = " << system_tms_end_.tms_utime -
  // system_tms_start_.tms_utime <<  "system kernel_time = " <<
  // system_tms_end_.tms_stime - system_tms_start_.tms_stime <<std::endl;
  // LOG(INFO) << "Process user_time = " << process_tms_end_.tms_utime -
  // process_tms_start_.tms_utime <<  "Process kernel_time = " <<
  // process_tms_end_.tms_stime - process_tms_start_.tms_stime <<std::endl;
  // LOG(INFO) << "Idle time = " << idle_end_ - idle_start_<< std::endl;
  // LOG(INFO) << "CLOCK TIME = " << end_ - start_ << std::endl;
  float busy_time = (system_tms_end_.tms_utime - system_tms_start_.tms_utime) +
                    (system_tms_end_.tms_stime - system_tms_start_.tms_stime);
  float idle_time = (idle_end_ - idle_start_);
  cpu_utilization = busy_time / (busy_time + idle_time);
  LOG(INFO) << "CPU Utilization = " << cpu_utilization << std::endl;
#endif
  return cpu_utilization;
}

float CPUOverview::GetCpuCurProcessUtilization() {
  float cpu_process_utilization = 0.0;
#ifdef _MSC_VER
  uint64_t process_user_time_start = FileTimeToUint64(process_user_time_start_);
  uint64_t process_user_time_end = FileTimeToUint64(process_user_time_end_);
  uint64_t process_kernel_time_start =
      FileTimeToUint64(process_kernel_time_start_);
  uint64_t process_kernel_time_end = FileTimeToUint64(process_kernel_time_end_);
  uint64_t start = FileTimeToUint64(start_);
  uint64_t end = FileTimeToUint64(end_);
  float busy_time = (process_kernel_time_end - process_kernel_time_start) +
                    (process_user_time_end - process_user_time_start);
  cpu_process_utilization = busy_time / (end - start);
  LOG(INFO) << "Process Utilization = " << cpu_process_utilization << std::endl;
#else
  float busy_time =
      (process_tms_end_.tms_utime - process_tms_start_.tms_utime) +
      (process_tms_end_.tms_stime - process_tms_start_.tms_stime);
  cpu_process_utilization = busy_time / (end_ - start_);
  LOG(INFO) << "Process Utilization = " << cpu_process_utilization << std::endl;
#endif
  return cpu_process_utilization;
}

}  // namespace platform
}  // namespace paddle
