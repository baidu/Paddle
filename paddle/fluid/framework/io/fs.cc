/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/io/fs.h"

#include <sys/stat.h>

#include <memory>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework {

static void fs_add_read_converter_internal(std::string& path,  // NOLINT
                                           bool& is_pipe,      // NOLINT
                                           const std::string& converter) {
  if (converter.empty()) {
    return;
  }

  if (!is_pipe) {
    path = string::format_string(
        "( %s ) < \"%s\"", converter.c_str(), path.c_str());
    is_pipe = true;
  } else {
    path = string::format_string("%s | %s", path.c_str(), converter.c_str());
  }
}

static void fs_add_write_converter_internal(std::string& path,  // NOLINT
                                            bool& is_pipe,      // NOLINT
                                            const std::string& converter) {
  if (converter.empty()) {
    return;
  }

  if (!is_pipe) {
    path = string::format_string(
        "( %s ) > \"%s\"", converter.c_str(), path.c_str());
    is_pipe = true;
  } else {
    path = string::format_string("%s | %s", converter.c_str(), path.c_str());
  }
}

static std::shared_ptr<FILE> fs_open_internal(const std::string& path,
                                              bool is_pipe,
                                              const std::string& mode,
                                              size_t buffer_size,
                                              int* err_no = nullptr) {
  std::shared_ptr<FILE> fp = nullptr;

  if (!is_pipe) {
    fp = shell_fopen(path, mode);
  } else {
    fp = shell_popen(path, mode, err_no);
  }

  if (buffer_size > 0) {
    char* buffer = new char[buffer_size];
    PADDLE_ENFORCE_EQ(
        0,
        setvbuf(&*fp, buffer, _IOFBF, buffer_size),
        common::errors::InvalidArgument("Set Buffer Failed, Please Check!"));
    fp = {&*fp, [fp, buffer](FILE*) mutable {  // NOLINT
            PADDLE_ENFORCE_EQ(fp.unique(),
                              true,
                              common::errors::PermissionDenied(
                                  "File Pointer is not unique!!!"));  // NOLINT
            fp = nullptr;
            delete[] buffer;
          }};
  }

  return fp;
}

static bool fs_begin_with_internal(const std::string& path,
                                   const std::string& str) {
  return strncmp(path.c_str(), str.c_str(), str.length()) == 0;
}

static bool fs_end_with_internal(const std::string& path,
                                 const std::string& str) {
  return path.length() >= str.length() &&
         strncmp(&path[path.length() - str.length()],
                 str.c_str(),
                 str.length()) == 0;
}

static size_t& localfs_buffer_size_internal() {
  static size_t x = 0;
  return x;
}

size_t localfs_buffer_size() { return localfs_buffer_size_internal(); }

void localfs_set_buffer_size(size_t x) { localfs_buffer_size_internal() = x; }

std::shared_ptr<FILE> localfs_open_read(std::string path,
                                        const std::string& converter) {
  bool is_pipe = false;

  if (fs_end_with_internal(path, ".gz")) {
    fs_add_read_converter_internal(path, is_pipe, "zcat");
  }

  fs_add_read_converter_internal(path, is_pipe, converter);
  return fs_open_internal(path, is_pipe, "r", localfs_buffer_size());
}

std::shared_ptr<FILE> localfs_open_write(std::string path,
                                         const std::string& converter) {
  shell_execute(
      string::format_string("mkdir -p $(dirname \"%s\")", path.c_str()));

  bool is_pipe = false;

  if (fs_end_with_internal(path, ".gz")) {
    fs_add_write_converter_internal(path, is_pipe, "gzip");
  }

  fs_add_write_converter_internal(path, is_pipe, converter);
  return fs_open_internal(path, is_pipe, "w", localfs_buffer_size());
}

std::shared_ptr<FILE> localfs_open_append_write(std::string path,
                                                const std::string& converter) {
  shell_execute(
      string::format_string("mkdir -p $(dirname \"%s\")", path.c_str()));

  bool is_pipe = false;

  if (fs_end_with_internal(path, ".gz")) {
    fs_add_write_converter_internal(path, is_pipe, "gzip");
  }

  fs_add_write_converter_internal(path, is_pipe, converter);
  return fs_open_internal(path, is_pipe, "a", localfs_buffer_size());
}

int64_t localfs_file_size(const std::string& path) {
  struct stat buf = {};
  if (0 != stat(path.c_str(), &buf)) {
    PADDLE_THROW(common::errors::External(
        "Failed to get file status via stat function."));
    return -1;
  }
  return (int64_t)buf.st_size;
}

void localfs_remove(const std::string& path) {
  if (path.empty()) {
    return;
  }

  shell_execute(string::format_string("rm -rf %s", path.c_str()));
}

std::vector<std::string> localfs_list(const std::string& path) {
  if (path.empty()) {
    return {};
  }

  std::shared_ptr<FILE> pipe;
  int err_no = 0;
  pipe = shell_popen(
      string::format_string("find %s -type f -maxdepth 1 | sort", path.c_str()),
      "r",
      &err_no);
  string::LineFileReader reader;
  std::vector<std::string> list;

  while (reader.getline(&*pipe)) {
    list.emplace_back(reader.get());
  }

  return list;
}

std::string localfs_tail(const std::string& path) {
  if (path.empty()) {
    return "";
  }

  return shell_get_command_output(
      string::format_string("tail -1 %s ", path.c_str()));
}

bool localfs_exists(const std::string& path) {
  std::string test_f = shell_get_command_output(
      string::format_string("[ -f %s ] ; echo $?", path.c_str()));

  if (string::trim_spaces(test_f) == "0") {
    return true;
  }

  std::string test_d = shell_get_command_output(
      string::format_string("[ -d %s ] ; echo $?", path.c_str()));

  if (string::trim_spaces(test_d) == "0") {
    return true;
  }

  return false;
}

void localfs_mkdir(const std::string& path) {
  if (path.empty()) {
    return;
  }

  shell_execute(string::format_string("mkdir -p %s", path.c_str()));
}

void localfs_mv(const std::string& src, const std::string& dest) {
  if (src.empty() || dest.empty()) {
    return;
  }
  shell_execute(string::format_string("mv %s %s", src.c_str(), dest.c_str()));
}

static size_t& hdfs_buffer_size_internal() {
  static size_t x = 0;
  return x;
}

size_t hdfs_buffer_size() { return hdfs_buffer_size_internal(); }

void hdfs_set_buffer_size(size_t x) { hdfs_buffer_size_internal() = x; }

static std::string& hdfs_command_internal() {
  static std::string x = "hadoop fs";
  return x;
}

const std::string& hdfs_command() { return hdfs_command_internal(); }

void hdfs_set_command(const std::string& x) { hdfs_command_internal() = x; }

// dataset and model may be on different afs cluster
static std::string& dataset_hdfs_command_internal() {
  static std::string x = "hadoop fs";
  return x;
}

const std::string& dataset_hdfs_command() {
  return dataset_hdfs_command_internal();
}

void dataset_hdfs_set_command(const std::string& x) {
  dataset_hdfs_command_internal() = x;
}

static std::string& customized_download_cmd_internal() {
  static std::string x = "";
  return x;
}

const std::string& download_cmd() { return customized_download_cmd_internal(); }

void set_download_command(const std::string& x) {
  customized_download_cmd_internal() = x;
}

std::shared_ptr<FILE> hdfs_open_read(std::string path,
                                     int* err_no,
                                     const std::string& converter,
                                     bool read_data) {
  if (!download_cmd().empty()) {  // use customized download command
    path = string::format_string(
        "%s \"%s\"", download_cmd().c_str(), path.c_str());
  } else {
    if (fs_end_with_internal(path, ".gz")) {
      if (read_data) {
        path = string::format_string(
            "%s -text \"%s\"", dataset_hdfs_command().c_str(), path.c_str());
      } else {
        path = string::format_string(
            "%s -text \"%s\"", hdfs_command().c_str(), path.c_str());
      }
    } else {
      if (read_data) {
        path = string::format_string(
            "%s -cat \"%s\"", dataset_hdfs_command().c_str(), path.c_str());
      } else {
        path = string::format_string(
            "%s -cat \"%s\"", hdfs_command().c_str(), path.c_str());
      }
    }
  }

  bool is_pipe = true;
  fs_add_read_converter_internal(path, is_pipe, converter);
  return fs_open_internal(path, is_pipe, "r", hdfs_buffer_size(), err_no);
}

std::shared_ptr<FILE> hdfs_open_write(std::string path,
                                      int* err_no,
                                      const std::string& converter) {
  path = string::format_string(
      "%s -put - \"%s\"", hdfs_command().c_str(), path.c_str());
  bool is_pipe = true;

  if (fs_end_with_internal(path, ".gz\"")) {
    fs_add_write_converter_internal(path, is_pipe, "gzip");
  }

  fs_add_write_converter_internal(path, is_pipe, converter);
  return fs_open_internal(path, is_pipe, "w", hdfs_buffer_size(), err_no);
}

void hdfs_remove(const std::string& path) {
  if (path.empty()) {
    return;
  }

  shell_execute(string::format_string(
      "%s -rmr %s &>/dev/null; true", hdfs_command().c_str(), path.c_str()));
}

std::vector<std::string> hdfs_list(const std::string& path) {
  if (path.empty()) {
    return {};
  }

  std::string prefix = "hdfs:";

  if (fs_begin_with_internal(path, "afs:")) {
    prefix = "afs:";
  }
  int err_no = 0;
  std::vector<std::string> list;
  do {
    err_no = 0;
    std::shared_ptr<FILE> pipe;
    pipe = shell_popen(
        string::format_string("%s -ls %s | ( grep ^- ; [ $? != 2 ] )",
                              hdfs_command().c_str(),
                              path.c_str()),
        "r",
        &err_no);
    string::LineFileReader reader;
    list.clear();

    while (reader.getline(&*pipe)) {
      std::vector<std::string> line = string::split_string(reader.get());
      if (line.size() != 8) {
        continue;
      }
      list.push_back(prefix + line[7]);
    }
  } while (err_no == -1);
  return list;
}

std::string hdfs_tail(const std::string& path) {
  if (path.empty()) {
    return "";
  }

  return shell_get_command_output(string::format_string(
      "%s -text %s | tail -1 ", hdfs_command().c_str(), path.c_str()));
}

bool hdfs_exists(const std::string& path) {
  std::string test = shell_get_command_output(string::format_string(
      "%s -test -e %s ; echo $?", hdfs_command().c_str(), path.c_str()));

  if (string::trim_spaces(test) == "0") {
    return true;
  }

  return false;
}

void hdfs_mkdir(const std::string& path) {
  if (path.empty()) {
    return;
  }

  shell_execute(string::format_string(
      "%s -mkdir %s; true", hdfs_command().c_str(), path.c_str()));
}

void hdfs_mv(const std::string& src, const std::string& dest) {
  if (src.empty() || dest.empty()) {
    return;
  }
  shell_execute(string::format_string(
      "%s -mv %s %s; true", hdfs_command().c_str(), src.c_str(), dest.c_str()));
}

int fs_select_internal(const std::string& path) {
  if (fs_begin_with_internal(path, "hdfs:") ||
      fs_begin_with_internal(path, "afs:")) {
    return 1;
  } else {
    return 0;
  }
}

std::shared_ptr<FILE> fs_open_read(const std::string& path,
                                   int* err_no,
                                   const std::string& converter,
                                   bool read_data) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_open_read(path, converter);

    case 1:
      return hdfs_open_read(path, err_no, converter, read_data);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }

  return {};
}

std::shared_ptr<FILE> fs_open_write(const std::string& path,
                                    int* err_no,
                                    const std::string& converter) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_open_write(path, converter);

    case 1:
      return hdfs_open_write(path, err_no, converter);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }

  return {};
}

std::shared_ptr<FILE> fs_open_append_write(const std::string& path,
                                           int* err_no,
                                           const std::string& converter) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_open_append_write(path, converter);

    case 1:
      return hdfs_open_write(path, err_no, converter);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }

  return {};
}

std::shared_ptr<FILE> fs_open(const std::string& path,
                              const std::string& mode,
                              int* err_no,
                              const std::string& converter) {
  if (mode == "r" || mode == "rb") {
    return fs_open_read(path, err_no, converter);
  }

  if (mode == "w" || mode == "wb") {
    return fs_open_write(path, err_no, converter);
  }

  PADDLE_THROW(common::errors::Unavailable(
      "Unsupported file open mode: %s. Only supports 'r', 'rb', 'w' or 'wb'.",
      mode));
  return {};
}

int64_t fs_file_size(const std::string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_file_size(path);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system."));
  }

  return 0;
}

void fs_remove(const std::string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_remove(path);

    case 1:
      return hdfs_remove(path);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }
}

std::vector<std::string> fs_list(const std::string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_list(path);

    case 1:
      return hdfs_list(path);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }

  return {};
}

std::string fs_tail(const std::string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_tail(path);

    case 1:
      return hdfs_tail(path);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }

  return "";
}

bool fs_exists(const std::string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_exists(path);

    case 1:
      return hdfs_exists(path);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }

  return false;
}

void fs_mkdir(const std::string& path) {
  switch (fs_select_internal(path)) {
    case 0:
      return localfs_mkdir(path);

    case 1:
      return hdfs_mkdir(path);

    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported file system. Now only supports local file system and "
          "HDFS."));
  }
}

void fs_mv(const std::string& src, const std::string& dest) {
  int s = fs_select_internal(src);
  int d = fs_select_internal(dest);
  PADDLE_ENFORCE_EQ(
      s,
      d,
      common::errors::InvalidArgument(
          "The source is not equal to destination, Please Check!"));
  switch (s) {
    case 0:
      return localfs_mv(src, dest);

    case 1:
      return hdfs_mv(src, dest);
  }
}

}  // namespace paddle::framework
