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

#pragma once
#include <unistd.h>

#include <condition_variable>  // NOLINT
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/distributed/ps/service/env.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_client.h"
#include "paddle/fluid/distributed/ps/service/graph_brpc_server.h"
#include "paddle/fluid/distributed/ps/service/ps_service/service.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/string/printf.h"

namespace paddle {
namespace distributed {
class GraphPyService {
 protected:
  std::vector<std::string> server_list, port_list, host_sign_list;
  int server_size, shard_num;
  int num_node_types;
  std::unordered_map<std::string, int> edge_to_id, feature_to_id;
  std::vector<std::string> id_to_feature, id_to_edge;
  std::vector<std::unordered_map<std::string, int>> table_feat_mapping;
  std::vector<std::vector<std::string>> table_feat_conf_feat_name;
  std::vector<std::vector<std::string>> table_feat_conf_feat_dtype;
  std::vector<std::vector<int>> table_feat_conf_feat_shape;

 public:
  int get_shard_num() { return shard_num; }
  void set_shard_num(int shard_num) { this->shard_num = shard_num; }
  void GetDownpourSparseTableProto(
      ::paddle::distributed::TableParameter* sparse_table_proto) {
    sparse_table_proto->set_table_id(0);
    sparse_table_proto->set_table_class("GraphTable");
    sparse_table_proto->set_shard_num(shard_num);
    sparse_table_proto->set_type(::paddle::distributed::PS_SPARSE_TABLE);
    ::paddle::distributed::TableAccessorParameter* accessor_proto =
        sparse_table_proto->mutable_accessor();

    // ::paddle::distributed::CommonAccessorParameter* common_proto =
    //     sparse_table_proto->mutable_common();

    ::paddle::distributed::GraphParameter* graph_proto =
        sparse_table_proto->mutable_graph_parameter();

    // ::paddle::distributed::GraphFeature* graph_feature =
    //     graph_proto->mutable_graph_feature();

    graph_proto->set_task_pool_size(24);

    graph_proto->set_table_name("cpu_graph_table");
    graph_proto->set_use_cache(false);
    for (size_t i = 0; i < id_to_edge.size(); i++)
      graph_proto->add_edge_types(id_to_edge[i]);
    for (size_t i = 0; i < id_to_feature.size(); i++) {
      graph_proto->add_node_types(id_to_feature[i]);
      auto feat_node = id_to_feature[i];
      ::paddle::distributed::GraphFeature* g_f =
          graph_proto->add_graph_feature();
      for (size_t x = 0; x < table_feat_conf_feat_name[i].size(); x++) {
        g_f->add_name(table_feat_conf_feat_name[i][x]);
        g_f->add_dtype(table_feat_conf_feat_dtype[i][x]);
        g_f->add_shape(table_feat_conf_feat_shape[i][x]);
      }
    }
    // Set GraphTable Parameter
    // common_proto->set_table_name(table_name);
    // common_proto->set_name(table_type);
    // for (size_t i = 0; i < feat_name.size(); i++) {
    //   common_proto->add_params(feat_dtype[i]);
    //   common_proto->add_dims(feat_shape[i]);
    //   common_proto->add_attributes(feat_name[i]);
    // }

    // for (size_t i = 0; i < feat_name.size(); i++) {
    //   graph_feature->add_dtype(feat_dtype[i]);
    //   graph_feature->add_shape(feat_shape[i]);
    //   graph_feature->add_name(feat_name[i]);
    // }
    accessor_proto->set_accessor_class("CommMergeAccessor");
  }

  void set_server_size(int server_size) { this->server_size = server_size; }
  void set_num_node_types(int num_node_types) {
    this->num_node_types = num_node_types;
  }
  int get_server_size(int server_size) { return server_size; }
  std::vector<std::string> split(const std::string& str, const char pattern);
  void set_up(std::string ips_str,
              int shard_num,
              std::vector<std::string> node_types,
              std::vector<std::string> edge_types);

  void add_table_feat_conf(std::string node_type,
                           std::string feat_name,
                           std::string feat_dtype,
                           int32_t feat_shape);
};
class GraphPyServer : public GraphPyService {
 public:
  GraphPyServer() {}
  void set_up(std::string ips_str,
              int shard_num,
              std::vector<std::string> node_types,
              std::vector<std::string> edge_types,
              int rank) {
    set_rank(rank);
    GraphPyService::set_up(ips_str, shard_num, node_types, edge_types);
  }
  int GetRank() { return rank; }
  void set_rank(int rank) { this->rank = rank; }

  void start_server(bool block = true);
  ::paddle::distributed::PSParameter GetServerProto();
  std::shared_ptr<::paddle::distributed::GraphBrpcServer> get_ps_server() {
    return pserver_ptr;
  }

 protected:
  int rank;
  std::shared_ptr<::paddle::distributed::GraphBrpcServer> pserver_ptr;
  std::thread* server_thread;
};
class GraphPyClient : public GraphPyService {
 public:
  void set_up(std::string ips_str,
              int shard_num,
              std::vector<std::string> node_types,
              std::vector<std::string> edge_types,
              int client_id) {
    set_client_id(client_id);
    GraphPyService::set_up(ips_str, shard_num, node_types, edge_types);
  }
  std::shared_ptr<::paddle::distributed::GraphBrpcClient> get_ps_client() {
    return worker_ptr;
  }
  void bind_local_server(int local_channel_index,
                         GraphPyServer& server) {  // NOLINT
    worker_ptr->set_local_channel(local_channel_index);
    worker_ptr->set_local_graph_service(
        (::paddle::distributed::GraphBrpcService*)server.get_ps_server()
            ->get_service());
  }
  void StopServer();
  void FinalizeWorker();
  void load_edge_file(std::string name, std::string filepath, bool reverse);
  void load_node_file(std::string name, std::string filepath);
  void clear_nodes(std::string name);
  void add_graph_node(std::string name,
                      std::vector<int64_t>& node_ids,   // NOLINT
                      std::vector<bool>& weight_list);  // NOLINT
  void remove_graph_node(std::string name,
                         std::vector<int64_t>& node_ids);  // NOLINT
  int get_client_id() { return client_id; }
  void set_client_id(int client_id) { this->client_id = client_id; }
  void start_client();
  std::pair<std::vector<std::vector<int64_t>>, std::vector<float>>
  batch_sample_neighbors(std::string name,
                         std::vector<int64_t> node_ids,
                         int sample_size,
                         bool return_weight,
                         bool return_edges);
  std::vector<int64_t> random_sample_nodes(std::string name,
                                           int server_index,
                                           int sample_size);
  std::vector<std::vector<std::string>> get_node_feat(
      std::string name,
      std::vector<int64_t> node_ids,
      std::vector<std::string> feature_names);
  void set_node_feat(std::string node_type,
                     std::vector<int64_t> node_ids,
                     std::vector<std::string> feature_names,
                     const std::vector<std::vector<std::string>> features);
  std::vector<FeatureNode> pull_graph_list(
      std::string name, int server_index, int start, int size, int step = 1);
  ::paddle::distributed::PSParameter GetWorkerProto();

 protected:
  mutable std::mutex mutex_;
  int client_id;
  std::shared_ptr<::paddle::distributed::GraphBrpcClient> worker_ptr;
  std::thread* client_thread;
  bool stopped_ = false;
};
}  // namespace distributed
}  // namespace paddle
