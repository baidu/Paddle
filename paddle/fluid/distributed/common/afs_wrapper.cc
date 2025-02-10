// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/common/afs_wrapper.h"

#include "paddle/fluid/framework/io/fs.h"

namespace paddle::distributed {
// AfsClient impl
int AfsClient::initialize(const FsClientParameter& fs_client_param) {
  // temporarily implemented with hdfs-client
  return initialize(fs_client_param.hadoop_bin(),
                    fs_client_param.uri(),
                    fs_client_param.user(),
                    fs_client_param.passwd(),
                    fs_client_param.buffer_size());
}
int AfsClient::initialize(const std::string& hadoop_bin,
                          const std::string& uri,
                          const std::string& user,
                          const std::string& passwd,
                          int buffer_size_param) {
  return initialize(
      hadoop_bin,
      uri,
      paddle::string::format_string("%s,%s", user.c_str(), passwd.c_str()),
      buffer_size_param);
}
int AfsClient::initialize(const std::string& hadoop_bin,
                          const std::string& uri,
                          const std::string& ugi,
                          int buffer_size_param) {
  // temporarily implemented with hdfs-client
  size_t buffer_size = 1L << 25;  // 32MB
  if (buffer_size_param > static_cast<int>(buffer_size)) {
    buffer_size = buffer_size_param;
  }
  paddle::framework::hdfs_set_buffer_size(buffer_size);
  paddle::framework::hdfs_set_command(paddle::string::format_string(
      "2>>./hdfs_err.log %s fs -Dfs.default.name=%s -Dhadoop.job.ugi=%s "
      "-Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000",
      hadoop_bin.c_str(),
      uri.c_str(),
      ugi.c_str()));
  return 0;
}

// open file in 'w' or 'r'
std::shared_ptr<FsReadChannel> AfsClient::open_r(const FsChannelConfig& config,
                                                 uint32_t buffer_size,
                                                 int* err_no) {
  std::shared_ptr<FsReadChannel> channel =
      std::make_shared<FsReadChannel>(buffer_size);
  std::shared_ptr<FILE> fp =
      paddle::framework::fs_open_read(config.path, err_no, config.deconverter);
  channel->open(fp, config);
  return channel;
}
std::shared_ptr<FsWriteChannel> AfsClient::open_w(const FsChannelConfig& config,
                                                  uint32_t buffer_size,
                                                  int* err_no) {
  std::shared_ptr<FsWriteChannel> channel =
      std::make_shared<FsWriteChannel>(buffer_size);
  std::shared_ptr<FILE> fp =
      paddle::framework::fs_open_write(config.path, err_no, config.converter);
  channel->open(fp, config);
  return channel;
}

// remove file in path, path maybe a reg, such as 'part-000-*'
void AfsClient::remove(const std::string& path) {
  return paddle::framework::fs_remove(path);
}
void AfsClient::remove_dir(const std::string& dir) {
  return paddle::framework::fs_remove(dir);
}

// list files in path, path maybe a dir with reg
std::vector<std::string> AfsClient::list(const std::string& path) {
  return paddle::framework::fs_list(path);
}

// exist or not
bool AfsClient::exist(const std::string& dir) {
  return paddle::framework::fs_exists(dir);
}
}  // namespace paddle::distributed
