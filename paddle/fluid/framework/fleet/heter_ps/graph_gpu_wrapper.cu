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

#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include <sstream>
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
COMMON_DECLARE_int32(gpugraph_storage_mode);
COMMON_DECLARE_bool(graph_metapath_split_opt);
COMMON_DECLARE_string(graph_edges_split_mode);
COMMON_DECLARE_bool(multi_node_sample_use_gpu_table);
namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_HETERPS
std::shared_ptr<GraphGpuWrapper> GraphGpuWrapper::s_instance_(nullptr);
void GraphGpuWrapper::set_device(std::vector<int> ids) {
  for (auto device_id : ids) {
    device_id_mapping.push_back(device_id);
  }
}

void GraphGpuWrapper::init_conf(const std::string &first_node_type_str,
                                const std::string &meta_path_str,
                                const std::string &excluded_train_pair,
                                const std::string &pair_label) {
  static std::mutex mutex;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (conf_initialized_) {
      return;
    }
    VLOG(2) << "init path config";
    conf_initialized_ = true;

    std::vector<std::string> first_node_type_vec;
    if (first_node_type_str[0] == '[') {
      assert(first_node_type_str[first_node_type_str.size() - 1] == ']');
      std::string tmp_first_node_type_str(
          first_node_type_str, 1, first_node_type_str.size() - 2);
      auto tmp_first_node_types = paddle::string::split_string<std::string>(
          tmp_first_node_type_str, ",");
      first_node_type_vec.assign(tmp_first_node_types.begin(),
                                 tmp_first_node_types.end());
    } else {
      first_node_type_vec.push_back(first_node_type_str);
    }
    tensor_pair_num_ = first_node_type_vec.size();
    first_node_type_.resize(tensor_pair_num_);
    for (int tensor_pair_idx = 0; tensor_pair_idx < tensor_pair_num_;
         ++tensor_pair_idx) {
      auto &first_node_type = first_node_type_vec[tensor_pair_idx];
      auto node_types =
          paddle::string::split_string<std::string>(first_node_type, ";");
      VLOG(2) << "node_types: " << first_node_type;
      for (auto &type : node_types) {
        auto iter = node_to_id.find(type);
        PADDLE_ENFORCE_NE(
            iter,
            node_to_id.end(),
            common::errors::NotFound("(%s) is not found in node_to_id.", type));
        VLOG(2) << "node_to_id[" << type << "] = " << iter->second;
        first_node_type_[tensor_pair_idx].push_back(iter->second);
        all_node_type_.push_back(iter->second);
      }  // end for (auto &type : node_types)
    }  // end for (int tensor_pair_idx = 0; tensor_pair_idx < tensor_pair_num_;

    std::vector<std::string> meta_path_vec;
    if (meta_path_str[0] == '[') {
      assert(meta_path_str[meta_path_str.size() - 1] == ']');
      std::string tmp_meta_path(meta_path_str, 1, meta_path_str.size() - 2);
      auto tmp_meta_paths =
          paddle::string::split_string<std::string>(tmp_meta_path, ",");
      meta_path_vec.assign(tmp_meta_paths.begin(), tmp_meta_paths.end());
    } else {
      meta_path_vec.push_back(meta_path_str);
    }
    assert(tensor_pair_num_ == meta_path_vec.size());
    meta_path_.resize(tensor_pair_num_);
    for (int tensor_pair_idx = 0; tensor_pair_idx < tensor_pair_num_;
         ++tensor_pair_idx) {
      auto &meta_path = meta_path_vec[tensor_pair_idx];
      meta_path_[tensor_pair_idx].resize(
          first_node_type_[tensor_pair_idx].size());
      auto meta_paths =
          paddle::string::split_string<std::string>(meta_path, ";");

      for (size_t i = 0; i < meta_paths.size(); i++) {
        auto path = meta_paths[i];
        auto edges = paddle::string::split_string<std::string>(path, "-");
        for (auto &edge : edges) {
          auto iter = edge_to_id.find(edge);
          PADDLE_ENFORCE_NE(iter,
                            edge_to_id.end(),
                            common::errors::NotFound(
                                "(%s) is not found in edge_to_id.", edge));
          VLOG(2) << "edge_to_id[" << edge << "] = " << iter->second;
          meta_path_[tensor_pair_idx][i].push_back(iter->second);
          if (edge_to_node_map_.find(iter->second) == edge_to_node_map_.end()) {
            auto nodes = get_ntype_from_etype(edge);
            uint64_t src_node_id = node_to_id.find(nodes[0])->second;
            uint64_t dst_node_id = node_to_id.find(nodes[1])->second;
            edge_to_node_map_[iter->second] = src_node_id << 32 | dst_node_id;
            all_node_type_.push_back(src_node_id);
            all_node_type_.push_back(dst_node_id);
            VLOG(1) << "add all_node_type[" << tensor_pair_idx << "] "
                    << node_types_idx_to_node_type_str(src_node_id) << ", "
                    << node_types_idx_to_node_type_str(dst_node_id);
          }
        }  // end for (auto &edge : edges) {
      }    // end for (size_t i = 0; i < meta_paths.size(); i++) {
    }  // end for (int tensor_pair_idx = 0; tensor_pair_idx < tensor_pair_num_;

    auto paths =
        paddle::string::split_string<std::string>(excluded_train_pair, ";");
    VLOG(2) << "excluded_train_pair[" << excluded_train_pair << "]";
    for (auto &path : paths) {
      auto nodes = get_ntype_from_etype(path);
      for (auto &node : nodes) {
        auto iter = node_to_id.find(node);
        PADDLE_ENFORCE_NE(
            iter,
            edge_to_id.end(),
            common::errors::NotFound("(%s) is not found in edge_to_id.", node));
        VLOG(2) << "edge_to_id[" << node << "] = " << iter->second;
        excluded_train_pair_.push_back(iter->second);
      }
    }

    if (pair_label != "") {
      pair_label_conf_.assign(id_to_feature.size() * id_to_feature.size(), 0);
      auto items = paddle::string::split_string<std::string>(pair_label, ",");
      VLOG(2) << "pair_label[" << pair_label
              << "] id_to_feature.size() = " << id_to_feature.size();
      for (auto &item : items) {
        auto sub_items = paddle::string::split_string<std::string>(item, ":");
        int label = std::stoi(sub_items[1]);
        auto nodes = get_ntype_from_etype(sub_items[0]);
        auto &edge_src = nodes[0];
        auto src_iter = node_to_id.find(edge_src);
        PADDLE_ENFORCE_NE(src_iter,
                          edge_to_id.end(),
                          common::errors::NotFound(
                              "(%s) is not found in edge_to_id.", edge_src));
        auto &edge_dst = nodes[1];
        auto dst_iter = node_to_id.find(edge_dst);
        PADDLE_ENFORCE_NE(dst_iter,
                          edge_to_id.end(),
                          common::errors::NotFound(
                              "(%s) is not found in edge_to_id.", edge_dst));
        VLOG(2) << "pair_label_conf[" << src_iter->second << "]["
                << dst_iter->second << "] = " << label;
        pair_label_conf_[src_iter->second * id_to_feature.size() +
                         dst_iter->second] = label;
        if (pair_label_conf_[dst_iter->second * id_to_feature.size() +
                             src_iter->second] == 0) {
          pair_label_conf_[dst_iter->second * id_to_feature.size() +
                           src_iter->second] = label;
        }
      }
      for (int i = 0; i < id_to_feature.size(); ++i) {
        std::stringstream buf;
        for (int j = 0; j < id_to_feature.size(); ++j) {
          buf << static_cast<int>(
                     pair_label_conf_[i * id_to_feature.size() + j])
              << " ";
        }
        VLOG(2) << "pair_label_conf[" << i << "]: " << buf.str();
      }
    }

    int max_dev_id = 0;
    for (size_t i = 0; i < device_id_mapping.size(); i++) {
      if (device_id_mapping[i] > max_dev_id) {
        max_dev_id = device_id_mapping[i];
      }
    }

    finish_node_type_.resize(tensor_pair_num_);
    node_type_start_.resize(tensor_pair_num_);
    global_infer_node_type_start_.resize(tensor_pair_num_);
    infer_cursor_.resize(tensor_pair_num_);
    for (int tensor_pair_idx = 0; tensor_pair_idx < tensor_pair_num_;
         ++tensor_pair_idx) {
      finish_node_type_[tensor_pair_idx].resize(max_dev_id + 1);
      node_type_start_[tensor_pair_idx].resize(max_dev_id + 1);
      global_infer_node_type_start_[tensor_pair_idx].resize(max_dev_id + 1);
      auto &first_node_type = first_node_type_vec[tensor_pair_idx];
      auto node_types =
          paddle::string::split_string<std::string>(first_node_type, ";");
      for (size_t i = 0; i < device_id_mapping.size(); i++) {
        int dev_id = device_id_mapping[i];
        auto &node_type_start = node_type_start_[tensor_pair_idx][i];
        auto &infer_node_type_start =
            global_infer_node_type_start_[tensor_pair_idx][i];
        auto &finish_node_type = finish_node_type_[tensor_pair_idx][i];
        finish_node_type.clear();

        for (size_t idx = 0; idx < node_to_id.size(); idx++) {
          infer_node_type_start[idx] = 0;
        }
        for (auto &type : node_types) {
          auto iter = node_to_id.find(type);
          node_type_start[iter->second] = 0;
          infer_node_type_start[iter->second] = 0;
        }
        infer_cursor_[tensor_pair_idx].push_back(0);
        cursor_.push_back(0);
      }
    }  // end for (int tensor_pair_idx = 0; tensor_pair_idx < tensor_pair_num_;
  }    // end static std::mutex mutex;
}

void GraphGpuWrapper::init_type_keys(
    std::vector<std::vector<std::shared_ptr<phi::Allocation>>> &keys,
    std::vector<std::vector<uint64_t>> &lens) {
  size_t thread_num = device_id_mapping.size();

  auto &graph_all_type_total_keys = get_graph_type_keys();
  auto &type_to_index = get_graph_type_to_index();
  std::vector<std::vector<uint64_t>> tmp_keys;
  tmp_keys.resize(thread_num);
  int first_node_idx;

  if (keys.size() > 0) {  // not empty
    for (size_t f_idx = 0; f_idx < keys.size(); f_idx++) {
      for (size_t j = 0; j < keys[f_idx].size(); j++) {
        keys[f_idx][j].reset();
      }
    }
  }
  keys.clear();
  lens.clear();
  keys.resize(graph_all_type_total_keys.size());
  lens.resize(graph_all_type_total_keys.size());
  for (size_t f_idx = 0; f_idx < graph_all_type_total_keys.size(); f_idx++) {
    for (size_t j = 0; j < tmp_keys.size(); j++) {
      tmp_keys[j].clear();
    }
    keys[f_idx].resize(thread_num);
    auto &type_total_key = graph_all_type_total_keys[f_idx];
    VLOG(1) << "node_type[" << index_to_node_type_str(f_idx) << "] index["
            << type_to_index[f_idx] << "] graph_all_type_total_keys[" << f_idx
            << "]=" << graph_all_type_total_keys[f_idx].size();
    for (size_t j = 0; j < type_total_key.size(); j++) {
      uint64_t shard = type_total_key[j] % thread_num;
      tmp_keys[shard].push_back(type_total_key[j]);
    }
    for (size_t j = 0; j < thread_num; j++) {
      lens[f_idx].push_back(tmp_keys[j].size());
      VLOG(1) << "node_type[" << index_to_node_type_str(f_idx) << "] index["
              << type_to_index[f_idx] << "] gpu_graph_device_keys[" << f_idx
              << "]=" << tmp_keys[j].size();
    }
    for (size_t j = 0; j < thread_num; j++) {
      auto stream = get_local_stream(j);
      int gpuid = device_id_mapping[j];
      auto place = phi::GPUPlace(gpuid);
      platform::CUDADeviceGuard guard(gpuid);
      keys[f_idx][j] = memory::AllocShared(
          place,
          tmp_keys[j].size() * sizeof(uint64_t),
          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
      cudaMemcpyAsync(keys[f_idx][j]->ptr(),
                      tmp_keys[j].data(),
                      sizeof(uint64_t) * tmp_keys[j].size(),
                      cudaMemcpyHostToDevice,
                      stream);
    }
  }
  for (int i = 0; i < thread_num; i++) {
    auto stream = get_local_stream(i);
    cudaStreamSynchronize(stream);
  }
}

void GraphGpuWrapper::init_metapath(std::string cur_metapath,
                                    int cur_metapath_index,
                                    int cur_metapath_len) {
  cur_metapath_ = cur_metapath;
  cur_metapath_index_ = cur_metapath_index;
  cur_metapath_len_ = cur_metapath_len;

  auto nodes = paddle::string::split_string<std::string>(cur_metapath_, "-");
  cur_parse_metapath_.clear();
  cur_parse_reverse_metapath_.clear();
  for (auto &node : nodes) {
    VLOG(2) << "node: " << node << " , in metapath: " << cur_metapath_;
    auto iter = edge_to_id.find(node);
    PADDLE_ENFORCE_NE(
        iter,
        edge_to_id.end(),
        common::errors::NotFound("(%s) is not found in edge_to_id.", node));
    cur_parse_metapath_.push_back(iter->second);
    std::string reverse_type = get_reverse_etype(node);
    iter = edge_to_id.find(reverse_type);
    PADDLE_ENFORCE_NE(iter,
                      edge_to_id.end(),
                      common::errors::NotFound(
                          "(%s) is not found in edge_to_id.", reverse_type));
    cur_parse_reverse_metapath_.push_back(iter->second);
  }

  size_t thread_num = device_id_mapping.size();
  cur_metapath_start_.resize(thread_num);
  for (size_t i = 0; i < thread_num; i++) {
    cur_metapath_start_[i] = 0;
  }
}

void GraphGpuWrapper::init_metapath_total_keys() {
  auto &graph_all_type_total_keys = get_graph_type_keys();
  auto &type_to_index = get_graph_type_to_index();
  std::vector<std::vector<uint64_t>> tmp_keys;
  size_t thread_num = device_id_mapping.size();
  tmp_keys.resize(thread_num);
  int first_node_idx;
  auto nodes = paddle::string::split_string<std::string>(cur_metapath_, "-");
  std::string first_node = get_ntype_from_etype(nodes[0])[0];
  auto it = node_to_id.find(first_node);
  first_node_idx = it->second;
  d_node_iter_graph_metapath_keys_.resize(thread_num);
  h_node_iter_graph_metapath_keys_len_.resize(thread_num);

  for (size_t j = 0; j < tmp_keys.size(); j++) {
    tmp_keys[j].clear();
  }
  size_t f_idx = type_to_index[first_node_idx];
  auto &type_total_key = graph_all_type_total_keys[f_idx];
  VLOG(2) << "first node type:" << first_node_idx
          << ", node start size:" << type_total_key.size();
  for (size_t j = 0; j < type_total_key.size(); j++) {
    uint64_t shard = type_total_key[j] % thread_num;
    tmp_keys[shard].push_back(type_total_key[j]);
  }
  auto fleet_ptr = framework::FleetWrapper::GetInstance();
  std::shuffle(
      tmp_keys.begin(), tmp_keys.end(), fleet_ptr->LocalRandomEngine());

  for (size_t j = 0; j < thread_num; j++) {
    h_node_iter_graph_metapath_keys_len_[j] = tmp_keys[j].size();
    VLOG(2) << j << " th card, graph train keys len: " << tmp_keys[j].size();
  }

  for (size_t j = 0; j < thread_num; j++) {
    auto stream = get_local_stream(j);
    int gpuid = device_id_mapping[j];
    auto place = phi::GPUPlace(gpuid);
    platform::CUDADeviceGuard guard(gpuid);
    d_node_iter_graph_metapath_keys_[j] = memory::AllocShared(
        place,
        tmp_keys[j].size() * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    cudaMemcpyAsync(d_node_iter_graph_metapath_keys_[j]->ptr(),
                    tmp_keys[j].data(),
                    sizeof(uint64_t) * tmp_keys[j].size(),
                    cudaMemcpyHostToDevice,
                    stream);
  }
  for (size_t j = 0; j < thread_num; j++) {
    auto stream = get_local_stream(j);
    cudaStreamSynchronize(stream);
  }
}

void GraphGpuWrapper::clear_metapath_state() {
  size_t thread_num = device_id_mapping.size();
  for (size_t j = 0; j < thread_num; j++) {
    cur_metapath_start_[j] = 0;
    h_node_iter_graph_metapath_keys_len_[j] = 0;
    d_node_iter_graph_metapath_keys_[j].reset();
    for (size_t k = 0; k < cur_parse_metapath_.size(); k++) {
      reinterpret_cast<GpuPsGraphTable *>(graph_table)
          ->clear_graph_info(j, cur_parse_metapath_[k]);
    }
  }
  std::vector<int> clear_etype;
  for (size_t j = 0; j < cur_parse_metapath_.size(); j++) {
    if (find(clear_etype.begin(), clear_etype.end(), cur_parse_metapath_[j]) ==
        clear_etype.end()) {
      clear_etype.push_back(cur_parse_metapath_[j]);
    }
  }
  for (size_t j = 0; j < cur_parse_reverse_metapath_.size(); j++) {
    if (find(clear_etype.begin(),
             clear_etype.end(),
             cur_parse_reverse_metapath_[j]) == clear_etype.end()) {
      clear_etype.push_back(cur_parse_reverse_metapath_[j]);
    }
  }
  for (size_t j = 0; j < clear_etype.size(); j++) {
    reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->cpu_graph_table_->clear_graph(clear_etype[j]);
  }
}

int GraphGpuWrapper::get_all_id(int table_type,
                                int slice_num,
                                std::vector<std::vector<uint64_t>> *output) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_id(
          (GraphTableType)table_type, slice_num, output);
}

int GraphGpuWrapper::get_all_neighbor_id(
    GraphTableType table_type,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_neighbor_id(table_type, slice_num, output);
}

std::string GraphGpuWrapper::node_types_idx_to_node_type_str(
    int node_types_idx) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->node_types_idx_to_node_type_str(node_types_idx);
}

std::string GraphGpuWrapper::index_to_node_type_str(int index) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->index_to_node_type_str(index);
}

int GraphGpuWrapper::get_all_id(int table_type,
                                int idx,
                                int slice_num,
                                std::vector<std::vector<uint64_t>> *output) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_id(
          (GraphTableType)table_type, idx, slice_num, output);
}

int GraphGpuWrapper::get_all_neighbor_id(
    GraphTableType table_type,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_neighbor_id(
          table_type, idx, slice_num, output);
}

int GraphGpuWrapper::get_all_feature_ids(
    GraphTableType table_type,
    int idx,
    int slice_num,
    std::vector<std::vector<uint64_t>> *output) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_feature_ids(
          table_type, idx, slice_num, output);
}

int GraphGpuWrapper::get_node_embedding_ids(
    int slice_num, std::vector<std::vector<uint64_t>> *output) {
  return (reinterpret_cast<GpuPsGraphTable *>(graph_table))
      ->cpu_graph_table_->get_node_embedding_ids(slice_num, output);
}

std::string GraphGpuWrapper::get_reverse_etype(std::string etype) {
  auto etype_split = paddle::string::split_string<std::string>(etype, "2");
  if (etype_split.size() == 2) {
    std::string reverse_type = etype_split[1] + "2" + etype_split[0];
    return reverse_type;
  } else if (etype_split.size() == 3) {
    std::string reverse_type =
        etype_split[2] + "2" + etype_split[1] + "2" + etype_split[0];
    return reverse_type;
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "The format of edge type should be [src2dst] or [src2etype2dst], "
        "but got [%s].",
        etype));
  }
}

std::vector<std::string> GraphGpuWrapper::get_ntype_from_etype(
    std::string etype) {
  std::vector<std::string> etype_split =
      paddle::string::split_string<std::string>(etype, "2");

  if (etype_split.size() == 2) {
    return etype_split;
  } else if (etype_split.size() == 3) {
    auto iter = etype_split.erase(etype_split.begin() + 1);
    return etype_split;
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "The format of edge type should be [src2dst] or [src2etype2dst], "
        "but got [%s].",
        etype));
  }
}

void GraphGpuWrapper::set_up_types(const std::vector<std::string> &edge_types,
                                   const std::vector<std::string> &node_types) {
  id_to_edge = edge_types;
  edge_to_id.clear();
  for (size_t table_id = 0; table_id < edge_types.size(); table_id++) {
    int res = edge_to_id.size();
    edge_to_id[edge_types[table_id]] = res;
  }
  id_to_feature = node_types;
  node_to_id.clear();
  for (size_t table_id = 0; table_id < node_types.size(); table_id++) {
    int res = node_to_id.size();
    node_to_id[node_types[table_id]] = res;
  }
  table_feat_mapping.resize(node_types.size());
  this->table_feat_conf_feat_name.resize(node_types.size());
  this->table_feat_conf_feat_dtype.resize(node_types.size());
  this->table_feat_conf_feat_shape.resize(node_types.size());
}

void GraphGpuWrapper::set_feature_separator(std::string ch) {
  feature_separator_ = ch;
  if (graph_table != nullptr) {
    reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->cpu_graph_table_->set_feature_separator(feature_separator_);
  }
}

void GraphGpuWrapper::set_feature_info(int slot_num_for_pull_feature,
                                       int float_slot_num) {
  this->slot_num_for_pull_feature_ = slot_num_for_pull_feature;
  this->float_slot_num_ = float_slot_num;
}

void GraphGpuWrapper::set_slot_feature_separator(std::string ch) {
  slot_feature_separator_ = ch;
  if (graph_table != nullptr) {
    reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->cpu_graph_table_->set_slot_feature_separator(slot_feature_separator_);
  }
}

void GraphGpuWrapper::make_partitions(int idx,
                                      int64_t byte_size,
                                      int device_len) {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->make_partitions(idx, byte_size, device_len);
}
int32_t GraphGpuWrapper::load_next_partition(int idx) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->load_next_partition(idx);
}

void GraphGpuWrapper::set_search_level(int level) {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->set_search_level(level);
}

std::vector<uint64_t> GraphGpuWrapper::get_partition(int idx, int num) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_partition(idx, num);
}
int32_t GraphGpuWrapper::get_partition_num(int idx) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_partition_num(idx);
}
void GraphGpuWrapper::make_complementary_graph(int idx, int64_t byte_size) {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->make_complementary_graph(idx, byte_size);
}
void GraphGpuWrapper::load_edge_file(std::string name,
                                     std::string filepath,
                                     bool reverse) {
  // 'e' means load edge
  std::string params = "e";
  if (reverse) {
    // 'e<' means load edges from $2 to $1
    params += "<" + name;
  } else {
    // 'e>' means load edges from $1 to $2
    params += ">" + name;
  }
  if (edge_to_id.find(name) != edge_to_id.end()) {
    reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->cpu_graph_table_->Load(std::string(filepath), params);
  }
}

void GraphGpuWrapper::load_edge_file(
    std::string etype2files,
    std::string graph_data_local_path,
    int part_num,
    bool reverse,
    const std::vector<bool> &is_reverse_edge_map,
    bool use_weight) {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->parse_edge_and_load(etype2files,
                                              graph_data_local_path,
                                              part_num,
                                              reverse,
                                              is_reverse_edge_map,
                                              use_weight);
}

int GraphGpuWrapper::load_node_file(std::string name, std::string filepath) {
  // 'n' means load nodes and 'node_type' follows

  std::string params = "n" + name;

  if (node_to_id.find(name) != node_to_id.end()) {
    return reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->cpu_graph_table_->Load(std::string(filepath), params);
  }
  return 0;
}

int GraphGpuWrapper::load_node_file(std::string ntype2files,
                                    std::string graph_data_local_path,
                                    int part_num) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->parse_node_and_load(
          ntype2files, graph_data_local_path, part_num);
}

void GraphGpuWrapper::shuffle_start_nodes_for_training() {
  size_t thread_num = device_id_mapping.size();

  int shuffle_seed = 0;
  std::random_device rd;
  std::mt19937 rng{rd()};
  std::uniform_int_distribution<int> dice_distribution(
      0, std::numeric_limits<int>::max());

  for (size_t i = 0; i < d_node_iter_graph_all_type_keys_.size(); i++) {
    for (size_t j = 0; j < d_node_iter_graph_all_type_keys_[i].size(); j++) {
      auto stream = get_local_stream(j);
      int gpuid = device_id_mapping[j];
      auto place = phi::GPUPlace(gpuid);
      platform::CUDADeviceGuard guard(gpuid);
      paddle::memory::ThrustAllocator<cudaStream_t> allocator(place, stream);
      const auto &exec_policy = thrust::cuda::par(allocator).on(stream);
      shuffle_seed = dice_distribution(rng);
      thrust::random::default_random_engine engine(shuffle_seed);
      uint64_t *cur_node_iter_ptr = reinterpret_cast<uint64_t *>(
          d_node_iter_graph_all_type_keys_[i][j]->ptr());
      VLOG(2) << "node type: " << i << ", card num: " << j
              << ", len: " << h_node_iter_graph_all_type_keys_len_[i][j];

      thrust::shuffle(exec_policy,
                      thrust::device_pointer_cast(cur_node_iter_ptr),
                      thrust::device_pointer_cast(cur_node_iter_ptr) +
                          h_node_iter_graph_all_type_keys_len_[i][j],
                      engine);
    }
  }
  for (size_t i = 0; i < thread_num; i++) {
    auto stream = get_local_stream(i);
    cudaStreamSynchronize(stream);
  }
}

int GraphGpuWrapper::set_node_iter_from_file(std::string ntype2files,
                                             std::string node_types_file_path,
                                             int part_num,
                                             bool training,
                                             bool shuffle) {
  // 1. load cpu node
  (reinterpret_cast<GpuPsGraphTable *>(graph_table))
      ->cpu_graph_table_->parse_node_and_load(
          ntype2files, node_types_file_path, part_num, false);

  // 2. init node iter keys on cpu and release cpu node shards.
  if (type_keys_initialized_) {
    // release d_graph_all_type_total_keys_ and h_graph_all_type_keys_len_
    for (size_t f_idx = 0; f_idx < d_graph_all_type_total_keys_.size();
         f_idx++) {
      for (size_t j = 0; j < d_graph_all_type_total_keys_[f_idx].size(); j++) {
        d_graph_all_type_total_keys_[f_idx][j].reset();
      }
    }
    d_graph_all_type_total_keys_.clear();
    h_graph_all_type_keys_len_.clear();
    type_keys_initialized_ = false;
  }

  (reinterpret_cast<GpuPsGraphTable *>(graph_table))
      ->cpu_graph_table_->build_node_iter_type_keys();
  (reinterpret_cast<GpuPsGraphTable *>(graph_table))
      ->cpu_graph_table_->clear_node_shard();

  // 3. init train or infer type keys.
  if (!training) {
    init_type_keys(d_node_iter_graph_all_type_keys_,
                   h_node_iter_graph_all_type_keys_len_);
  } else {
    if (FLAGS_graph_metapath_split_opt) {
      init_metapath_total_keys();
    } else {
      init_type_keys(d_node_iter_graph_all_type_keys_,
                     h_node_iter_graph_all_type_keys_len_);

      if (shuffle) {
        shuffle_start_nodes_for_training();
      }
    }
  }
  return 0;
}

int GraphGpuWrapper::set_node_iter_from_graph(bool training, bool shuffle) {
  // 1. init type keys
  if (!type_keys_initialized_) {
    init_type_keys(d_graph_all_type_total_keys_, h_graph_all_type_keys_len_);
    type_keys_initialized_ = true;
  }

  // 2. init train or infer type keys.
  if (!training) {
    d_node_iter_graph_all_type_keys_ = d_graph_all_type_total_keys_;
    h_node_iter_graph_all_type_keys_len_ = h_graph_all_type_keys_len_;
  } else {
    if (FLAGS_graph_metapath_split_opt) {
      init_metapath_total_keys();
    } else {
      d_node_iter_graph_all_type_keys_ = d_graph_all_type_total_keys_;
      h_node_iter_graph_all_type_keys_len_ = h_graph_all_type_keys_len_;

      if (shuffle) {
        shuffle_start_nodes_for_training();
      }
    }
  }
  return 0;
}

void GraphGpuWrapper::load_node_and_edge(
    std::string etype2files,
    std::string ntype2files,
    std::string graph_data_local_path,
    int part_num,
    bool reverse,
    const std::vector<bool> &is_reverse_edge_map) {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->load_node_and_edge_file(etype2files,
                                                  ntype2files,
                                                  graph_data_local_path,
                                                  part_num,
                                                  reverse,
                                                  is_reverse_edge_map,
                                                  false);
}

void GraphGpuWrapper::calc_edge_type_limit() {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->calc_edge_type_limit();
}

void GraphGpuWrapper::add_table_feat_conf(std::string table_name,
                                          std::string feat_name,
                                          std::string feat_dtype,
                                          int feat_shape) {
  if (node_to_id.find(table_name) != node_to_id.end()) {
    int idx = node_to_id[table_name];
    if (table_feat_mapping[idx].find(feat_name) ==
        table_feat_mapping[idx].end()) {
      int res = table_feat_mapping[idx].size();
      table_feat_mapping[idx][feat_name] = res;
    }
    int feat_idx = table_feat_mapping[idx][feat_name];
    VLOG(0) << "table_name " << table_name << " mapping id " << idx;
    VLOG(0) << " feat name " << feat_name << " feat id" << feat_idx;
    if (feat_idx < table_feat_conf_feat_name[idx].size()) {
      // override
      table_feat_conf_feat_name[idx][feat_idx] = feat_name;
      table_feat_conf_feat_dtype[idx][feat_idx] = feat_dtype;
      table_feat_conf_feat_shape[idx][feat_idx] = feat_shape;
    } else {
      // new
      table_feat_conf_feat_name[idx].push_back(feat_name);
      table_feat_conf_feat_dtype[idx].push_back(feat_dtype);
      table_feat_conf_feat_shape[idx].push_back(feat_shape);
    }
  }
  VLOG(0) << "add conf over";
}
void GraphGpuWrapper::init_search_level(int level) { search_level = level; }

gpuStream_t GraphGpuWrapper::get_local_stream(int gpuid) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->get_local_stream(gpuid);
}

void GraphGpuWrapper::init_service() {
  table_proto.set_task_pool_size(64);
  table_proto.set_shard_num(1000);
  table_proto.set_build_sampler_on_cpu(false);
  table_proto.set_search_level(search_level);
  table_proto.set_table_name("cpu_graph_table_");
  table_proto.set_use_cache(false);
  for (int i = 0; i < id_to_edge.size(); i++)
    table_proto.add_edge_types(id_to_edge[i]);
  for (int i = 0; i < id_to_feature.size(); i++) {
    table_proto.add_node_types(id_to_feature[i]);
    auto feat_node = id_to_feature[i];
    ::paddle::distributed::GraphFeature *g_f = table_proto.add_graph_feature();
    for (int x = 0; x < table_feat_conf_feat_name[i].size(); x++) {
      g_f->add_name(table_feat_conf_feat_name[i][x]);
      g_f->add_dtype(table_feat_conf_feat_dtype[i][x]);
      g_f->add_shape(table_feat_conf_feat_shape[i][x]);
    }
  }

  std::shared_ptr<HeterPsResource> resource =
      std::make_shared<HeterPsResource>(device_id_mapping);
  resource->enable_p2p();

#ifdef PADDLE_WITH_GLOO
  auto gloo = paddle::framework::GlooWrapper::GetInstance();
  if (gloo->Size() > 1) {
    multi_node_ = 1;
    resource->set_multi_node(multi_node_);
    VLOG(0) << "init multi node graph gpu server";
  }
#else
  PADDLE_THROW(common::errors::Unavailable("heter ps need compile with GLOO"));
#endif

#ifdef PADDLE_WITH_CUDA
  if (multi_node_) {
    int dev_size = device_id_mapping.size();
    // init inner comm
    inner_comms_.resize(dev_size);
    inter_ncclids_.resize(dev_size);
    phi::dynload::ncclCommInitAll(
        &(inner_comms_[0]), dev_size, &device_id_mapping[0]);
// init inter comm
#ifdef PADDLE_WITH_GLOO
    inter_comms_.resize(dev_size);
    if (gloo->Rank() == 0) {
      for (int i = 0; i < dev_size; ++i) {
        phi::dynload::ncclGetUniqueId(&inter_ncclids_[i]);
      }
    }

    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(),
        true,
        common::errors::PreconditionNotMet(
            "You must initialize the gloo environment first to use it."));
    gloo::BroadcastOptions opts(gloo->GetContext());
    opts.setOutput(&inter_ncclids_[0], dev_size);
    opts.setRoot(0);
    gloo::broadcast(opts);

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGroupStart());
    for (int i = 0; i < dev_size; ++i) {
      platform::CUDADeviceGuard guard(device_id_mapping[i]);
      phi::dynload::ncclCommInitRank(
          &inter_comms_[i], gloo->Size(), inter_ncclids_[i], gloo->Rank());
    }
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGroupEnd());

    rank_id_ = gloo->Rank();
    node_size_ = gloo->Size();
#else
    PADDLE_THROW(
        common::errors::Unavailable("heter ps need compile with GLOO"));
#endif
  }
#endif

  size_t gpu_num = device_id_mapping.size();
  GpuPsGraphTable *g = new GpuPsGraphTable(
      resource, id_to_edge.size(), slot_num_for_pull_feature_, float_slot_num_);
  g->init_cpu_table(table_proto, gpu_num);
  g->set_nccl_comm_and_size(inner_comms_, inter_comms_, node_size_, rank_id_);
  g->cpu_graph_table_->set_feature_separator(feature_separator_);
  g->cpu_graph_table_->set_slot_feature_separator(slot_feature_separator_);
  graph_table = reinterpret_cast<char *>(g);
  upload_num = gpu_num;
  upload_task_pool.reset(new ::ThreadPool(upload_num));
}

void GraphGpuWrapper::finalize() {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)->show_table_collisions();
}

void GraphGpuWrapper::show_mem(const char *msg) {
  show_cpu_mem(msg);
  show_gpu_mem(msg);
}

// edge table
void GraphGpuWrapper::upload_batch(int table_type,
                                   int slice_num,
                                   const std::string &edge_type) {
  VLOG(0) << "begin upload edge, etype[" << edge_type << "]";
  auto iter = edge_to_id.find(edge_type);
  int edge_idx = iter->second;
  VLOG(2) << "cur edge: " << edge_type << ", edge_idx: " << edge_idx;
  std::vector<std::vector<uint64_t>> ids;
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_id(
          (GraphTableType)table_type, edge_idx, slice_num, &ids);
  debug_gpu_memory_info("upload_batch node start");
  GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
  std::vector<std::future<int>> tasks;

  for (int i = 0; i < slice_num; i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, edge_idx, this]() -> int {
      VLOG(0) << "begin make_gpu_ps_graph, node_id[" << i << "]_size["
              << ids[i].size() << "]";
      GpuPsCommGraph sub_graph =
          g->cpu_graph_table_->make_gpu_ps_graph(edge_idx, ids[i]);
      g->build_graph_on_single_gpu(sub_graph, i, edge_idx);
      sub_graph.release_on_cpu();
      VLOG(1) << "sub graph on gpu " << i << " is built";
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  debug_gpu_memory_info("upload_batch node end");
}

// feature table
void GraphGpuWrapper::upload_batch(int table_type,
                                   int slice_num,
                                   int slot_num,
                                   int float_slot_num) {
  if (table_type == GraphTableType::FEATURE_TABLE &&
      (FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                          MEM_EMB_FEATURE_AND_GPU_GRAPH ||
       FLAGS_gpugraph_storage_mode == paddle::framework::GpuGraphStorageMode::
                                          SSD_EMB_AND_MEM_FEATURE_GPU_GRAPH)) {
    return;
  }
  std::vector<std::vector<uint64_t>> node_ids;
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->get_all_id(
          (GraphTableType)table_type, slice_num, &node_ids);
  debug_gpu_memory_info("upload_batch feature start");
  GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
  std::vector<std::future<int>> tasks;
  for (int i = 0; i < slice_num; i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, this]() -> int {
      // build slot feature
      VLOG(0) << "begin make_gpu_ps_graph_fea, node_ids[" << i << "]_size["
              << node_ids[i].size() << "]";
      GpuPsCommGraphFea sub_graph =
          g->cpu_graph_table_->make_gpu_ps_graph_fea(i, node_ids[i], slot_num);
      // sub_graph.display_on_cpu();
      VLOG(0) << "begin build_graph_fea_on_single_gpu, node_ids[" << i
              << "]_size[" << node_ids[i].size() << "]";
      g->build_graph_fea_on_single_gpu(sub_graph, i);
      sub_graph.release_on_cpu();
      VLOG(0) << "sub graph fea on gpu " << i << " is built";
      if (float_slot_num > 0) {
        // build float feature
        VLOG(0) << "begin make_gpu_ps_graph_float_fea, node_ids[" << i
                << "]_size[" << node_ids[i].size() << "]";
        GpuPsCommGraphFloatFea float_sub_graph =
            g->cpu_graph_table_->make_gpu_ps_graph_float_fea(
                i, node_ids[i], float_slot_num);
        // sub_graph.display_on_cpu();
        VLOG(0) << "begin build_graph_float_fea_on_single_gpu, node_ids[" << i
                << "]_size[" << node_ids[i].size() << "]";
        g->build_graph_float_fea_on_single_gpu(float_sub_graph, i);
        float_sub_graph.release_on_cpu();
        VLOG(0) << "float sub graph fea on gpu " << i << " is built";
      }
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  // g->build_graph_from_cpu(vec);
  debug_gpu_memory_info("upload_batch feature end");
}

// get sub_graph_fea
std::vector<GpuPsCommGraphFea> GraphGpuWrapper::get_sub_graph_fea(
    std::vector<std::vector<uint64_t>> &node_ids, int slot_num) {
  GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
  std::vector<std::future<int>> tasks;
  std::vector<GpuPsCommGraphFea> sub_graph_feas(node_ids.size());
  for (int i = 0; i < node_ids.size(); i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, this]() -> int {
      GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
      sub_graph_feas[i] =
          g->cpu_graph_table_->make_gpu_ps_graph_fea(i, node_ids[i], slot_num);
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return sub_graph_feas;
}

// get sub_graph_float_fea
std::vector<GpuPsCommGraphFloatFea> GraphGpuWrapper::get_sub_graph_float_fea(
    std::vector<std::vector<uint64_t>> &node_ids, int float_slot_num) {
  if (float_slot_num == 0) return {};
  GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
  std::vector<std::future<int>> tasks;
  std::vector<GpuPsCommGraphFloatFea> sub_graph_float_feas(node_ids.size());
  for (int i = 0; i < node_ids.size(); i++) {
    tasks.push_back(upload_task_pool->enqueue([&, i, this]() -> int {
      GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
      sub_graph_float_feas[i] =
          g->cpu_graph_table_->make_gpu_ps_graph_float_fea(
              i, node_ids[i], float_slot_num);
      return 0;
    }));
  }
  for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
  return sub_graph_float_feas;
}

// build_gpu_graph_fea
void GraphGpuWrapper::build_gpu_graph_fea(GpuPsCommGraphFea &sub_graph_fea,
                                          int i) {
  GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
  g->build_graph_fea_on_single_gpu(sub_graph_fea, i);
  sub_graph_fea.release_on_cpu();
  VLOG(1) << "sub graph fea on gpu " << i << " is built";
  return;
}

// build_gpu_graph_float_fea
void GraphGpuWrapper::build_gpu_graph_float_fea(
    GpuPsCommGraphFloatFea &sub_graph_float_fea, int i) {
  GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
  g->build_graph_float_fea_on_single_gpu(sub_graph_float_fea, i);
  sub_graph_float_fea.release_on_cpu();
  VLOG(1) << "sub graph float fea on gpu " << i << " is built";
  return;
}

void GraphGpuWrapper::seek_keys_rank(int gpu_id,
                                     const uint64_t *d_in_keys,
                                     int len,
                                     uint32_t *d_out_ranks) {
  platform::CUDADeviceGuard guard(gpu_id);
  phi::GPUPlace place = phi::GPUPlace(gpu_id);
  auto stream = get_local_stream(gpu_id);

  if (FLAGS_graph_edges_split_mode == "fennel") {
    if (FLAGS_multi_node_sample_use_gpu_table) {
      // fennel下，FLAGS_multi_node_sample_use_gpu_table为True
      GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
      g->get_rank_info_of_nodes(gpu_id, d_in_keys, d_out_ranks, len);
      return;
    }
  }

  std::vector<uint64_t> h_keys(len);
  std::vector<uint32_t> h_ranks(len);
  CUDA_CHECK(cudaMemcpyAsync(h_keys.data(),
                             d_in_keys,
                             sizeof(uint64_t) * len,
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (FLAGS_graph_edges_split_mode == "fennel") {
    // fennel下，但 FLAGS_multi_node_sample_use_gpu_table为False
    reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->cpu_graph_table_->query_all_ids_rank(
            len, h_keys.data(), h_ranks.data());
  } else {
    // 硬拆下，cpu上进行取余得到
    auto gpu_num =
        reinterpret_cast<GpuPsGraphTable *>(graph_table)->get_device_num();
    for (size_t i = 0; i < len; ++i) {
      bool hit = false;
      auto &k = h_keys[i];
      h_ranks[i] = (k / gpu_num) % node_size_;
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(d_out_ranks,
                             h_ranks.data(),
                             sizeof(uint32_t) * len,
                             cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

NeighborSampleResult GraphGpuWrapper::graph_neighbor_sample_v3(
    NeighborSampleQuery q, bool cpu_switch, bool compress, bool weighted) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->graph_neighbor_sample_v3(q, cpu_switch, compress, weighted);
}

NeighborSampleResultV2 GraphGpuWrapper::graph_neighbor_sample_sage(
    int gpu_id,
    int edge_type_len,
    const uint64_t *d_keys,
    int sample_size,
    int len,
    std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs,
    bool weighted,
    bool return_weight) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->graph_neighbor_sample_sage(gpu_id,
                                   edge_type_len,
                                   d_keys,
                                   sample_size,
                                   len,
                                   edge_type_graphs,
                                   weighted,
                                   return_weight);
}

std::vector<std::shared_ptr<phi::Allocation>>
GraphGpuWrapper::get_edge_type_graph(int gpu_id, int edge_type_len) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->get_edge_type_graph(gpu_id, edge_type_len);
}

std::shared_ptr<phi::Allocation> GraphGpuWrapper::get_node_degree(int gpu_id,
                                                                  int edge_idx,
                                                                  uint64_t *key,
                                                                  int len) {
  return (reinterpret_cast<GpuPsGraphTable *>(graph_table))
      ->get_node_degree(gpu_id, edge_idx, key, len);
}
void GraphGpuWrapper::set_infer_mode(bool infer_mode) {
  if (graph_table != nullptr) {
    reinterpret_cast<GpuPsGraphTable *>(graph_table)
        ->set_infer_mode(infer_mode);
  }
}
int GraphGpuWrapper::get_feature_info_of_nodes(
    int gpu_id,
    uint64_t *d_nodes,
    int node_num,
    std::shared_ptr<phi::Allocation> &size_list,
    std::shared_ptr<phi::Allocation> &size_list_prefix_sum,
    std::shared_ptr<phi::Allocation> &feature_list,
    std::shared_ptr<phi::Allocation> &slot_list,
    bool sage_mode) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_NOT_NULL(
      graph_table,
      common::errors::InvalidArgument("graph_table should not be null"));
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->get_feature_info_of_nodes(gpu_id,
                                  d_nodes,
                                  node_num,
                                  size_list,
                                  size_list_prefix_sum,
                                  feature_list,
                                  slot_list,
                                  sage_mode);
}

int GraphGpuWrapper::get_float_feature_info_of_nodes(
    int gpu_id,
    uint64_t *d_nodes,
    int node_num,
    std::shared_ptr<phi::Allocation> &size_list,
    std::shared_ptr<phi::Allocation> &size_list_prefix_sum,
    std::shared_ptr<phi::Allocation> &feature_list,
    std::shared_ptr<phi::Allocation> &slot_list,
    bool sage_mode) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_NOT_NULL(
      graph_table,
      common::errors::InvalidArgument("graph_table should not be null"));
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->get_float_feature_info_of_nodes(gpu_id,
                                        d_nodes,
                                        node_num,
                                        size_list,
                                        size_list_prefix_sum,
                                        feature_list,
                                        slot_list,
                                        sage_mode);
}

int GraphGpuWrapper::get_feature_of_nodes(int gpu_id,
                                          uint64_t *d_walk,
                                          uint64_t *d_offset,
                                          uint32_t size,
                                          int slot_num,
                                          int *d_slot_feature_num_map,
                                          int fea_num_per_node) {
  platform::CUDADeviceGuard guard(gpu_id);
  PADDLE_ENFORCE_NOT_NULL(
      graph_table,
      common::errors::InvalidArgument("graph_table should not be null"));
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->get_feature_of_nodes(gpu_id,
                             d_walk,
                             d_offset,
                             size,
                             slot_num,
                             d_slot_feature_num_map,
                             fea_num_per_node);
}

NeighborSampleResult GraphGpuWrapper::graph_neighbor_sample(
    int gpu_id, uint64_t *device_keys, int walk_degree, int len) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto &edge_neighbor_size_limit = get_type_to_neighbor_limit();
  auto neighbor_size_limit = edge_neighbor_size_limit[0];
  VLOG(0) << "use edge type 0 set neighbor size limit";
  auto neighbor_sample_res =
      reinterpret_cast<GpuPsGraphTable *>(graph_table)
          ->graph_neighbor_sample(
              gpu_id, device_keys, walk_degree, len, neighbor_size_limit);

  return neighbor_sample_res;
}

// this function is contributed by Liwb5
std::vector<uint64_t> GraphGpuWrapper::graph_neighbor_sample(
    int gpu_id, int idx, std::vector<uint64_t> &key, int sample_size) {
  std::vector<uint64_t> res;
  if (key.size() == 0) {
    return res;
  }
  uint64_t *cuda_key;
  platform::CUDADeviceGuard guard(gpu_id);

  cudaMalloc(&cuda_key, key.size() * sizeof(uint64_t));
  cudaMemcpy(cuda_key,
             key.data(),
             key.size() * sizeof(uint64_t),
             cudaMemcpyHostToDevice);
  VLOG(0) << "key_size: " << key.size();
  auto &edge_neighbor_size_limit = get_type_to_neighbor_limit();
  auto neighbor_size_limit = edge_neighbor_size_limit[idx];
  auto neighbor_sample_res = reinterpret_cast<GpuPsGraphTable *>(graph_table)
                                 ->graph_neighbor_sample_v2(gpu_id,
                                                            idx,
                                                            cuda_key,
                                                            sample_size,
                                                            key.size(),
                                                            neighbor_size_limit,
                                                            false,
                                                            true,
                                                            false);
  int *actual_sample_size = new int[key.size()];
  cudaMemcpy(actual_sample_size,
             neighbor_sample_res.actual_sample_size,
             key.size() * sizeof(int),
             cudaMemcpyDeviceToHost);  // 3, 1, 3
  int cumsum = 0;
  for (int i = 0; i < key.size(); i++) {
    cumsum += actual_sample_size[i];
  }

  std::vector<uint64_t> cpu_key;
  cpu_key.resize(key.size() * sample_size);

  cudaMemcpy(cpu_key.data(),
             neighbor_sample_res.val,
             key.size() * sample_size * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < key.size(); i++) {
    for (int j = 0; j < actual_sample_size[i]; j++) {
      res.push_back(key[i]);
      res.push_back(cpu_key[i * sample_size + j]);
    }
  }
  delete[] actual_sample_size;
  cudaFree(cuda_key);
  return res;
}

NodeQueryResult GraphGpuWrapper::query_node_list(int gpu_id,
                                                 int idx,
                                                 int start,
                                                 int query_size) {
  PADDLE_ENFORCE_EQ(FLAGS_gpugraph_load_node_list_into_hbm,
                    true,
                    common::errors::PreconditionNotMet(
                        "when use query_node_list should set "
                        "gpugraph_load_node_list_into_hbm true"));
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->query_node_list(gpu_id, idx, start, query_size);
}
void GraphGpuWrapper::load_node_weight(int type_id, int idx, std::string path) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->load_node_weight(type_id, idx, path);
}

std::vector<int> GraphGpuWrapper::slot_feature_num_map() const {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->slot_feature_num_map();
}

void GraphGpuWrapper::export_partition_files(int idx, std::string file_path) {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->export_partition_files(idx, file_path);
}

void GraphGpuWrapper::release_graph() {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->release_graph();
}

void GraphGpuWrapper::release_graph_edge() {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->release_graph_edge();
}

void GraphGpuWrapper::release_graph_node() {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->release_graph_node();
  // fennel 下 构造gpu表
  if (strncasecmp(FLAGS_graph_edges_split_mode.c_str(), "fennel", 6) == 0 &&
      FLAGS_multi_node_sample_use_gpu_table) {
    debug_gpu_memory_info("release_graph_node fennel gputable start");
    VLOG(0) << "begin build rank gpu table";
    GpuPsGraphTable *g = reinterpret_cast<GpuPsGraphTable *>(graph_table);
    std::vector<std::future<int>> tasks;
    for (int i = 0; i < 8; i++) {
      tasks.push_back(upload_task_pool->enqueue([&, i, this]() -> int {
        // build rank feature
        GpuPsCommRankFea sub_graph =
            g->cpu_graph_table_->make_gpu_ps_rank_fea(i);
        g->build_rank_fea_on_single_gpu(sub_graph, i);
        sub_graph.release_on_cpu();
        return 0;
      }));
    }
    for (size_t i = 0; i < tasks.size(); i++) tasks[i].get();
    debug_gpu_memory_info("upload_batch feature end");
    VLOG(0) << "end build rank gpu table";
  }
}

std::vector<uint64_t> &GraphGpuWrapper::get_graph_total_keys() {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->graph_total_keys_;
}

std::vector<std::vector<uint64_t>> &GraphGpuWrapper::get_graph_type_keys() {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->graph_type_keys_;
}

std::unordered_map<int, int> &GraphGpuWrapper::get_type_to_neighbor_limit() {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->type_to_neighbor_limit_;
}

std::unordered_map<int, int> &GraphGpuWrapper::get_graph_type_to_index() {
  return reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->cpu_graph_table_->type_to_index_;
}

void GraphGpuWrapper::set_keys2rank(
    int gpu_id, std::shared_ptr<HashTable<uint64_t, uint32_t>> keys2rank) {
  reinterpret_cast<GpuPsGraphTable *>(graph_table)
      ->set_keys2rank(gpu_id, keys2rank);
}

std::string &GraphGpuWrapper::get_node_type_size(std::string first_node_type) {
  auto node_types =
      paddle::string::split_string<std::string>(first_node_type, ";");
  for (auto &type : node_types) {
    uniq_first_node_.insert(type);
  }

  auto &graph_all_type_total_keys = get_graph_type_keys();
  auto &type_to_index = get_graph_type_to_index();
  std::vector<std::string> node_type_size;
  for (auto node : uniq_first_node_) {
    auto it = node_to_id.find(node);
    auto first_node_idx = it->second;
    size_t f_idx = type_to_index[first_node_idx];
    int type_total_key_size = graph_all_type_total_keys[f_idx].size();
    std::string node_type_str =
        node + ":" + std::to_string(type_total_key_size);
    node_type_size.push_back(node_type_str);
  }
  std::string delim = ";";
  node_type_size_str_ = paddle::string::join_strings(node_type_size, delim);

  return node_type_size_str_;
}

std::string &GraphGpuWrapper::get_edge_type_size() {
  auto edge_type_size = reinterpret_cast<GpuPsGraphTable *>(graph_table)
                            ->cpu_graph_table_->edge_type_size;
  std::string delim = ";";
  edge_type_size_str_ = paddle::string::join_strings(edge_type_size, delim);
  return edge_type_size_str_;
}
#endif

};  // namespace framework
};  // namespace paddle
