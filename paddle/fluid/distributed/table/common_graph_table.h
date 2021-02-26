#pragma once

#include <ThreadPool.h>
#include <assert.h>
#include <pthread.h>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "Eigen/Dense"
#include<list>
#include "paddle/fluid/distributed/table/graph_node.h"
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/common_table.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/string/string_helper.h"
namespace paddle {
namespace distributed {
struct pair_hash {
    inline size_t operator()(const pair<uint64_t,GraphNodeType> & p) const {
        return p.first * 10007 + int(p.second);
    }
};
class GraphShard {
  public:
  static int bucket_low_bound;
  static int gcd(int s,int t){
    if(s % t == 0)
      return t;
    return gcd(t, s % t);
  }
  size_t get_size();
  GraphShard(){

  }
  GraphShard(int shard_num){
    this->shard_num = shard_num;
    bucket_size = init_bucket_size(shard_num);
    bucket.resize(bucket_size);
  }
  vector<GraphNode *> get_batch(int start,int total_size){
      if(start < 0)
        start = 0;
      int size = 0, cur_size;
      vector<GraphNode *> res;
      if(total_size <= 0)
         return res;
      for(int i = 0;i < bucket_size;i++){
          cur_size = bucket[i].size();
          if(size + cur_size <= start){
              size += cur_size;
              continue;
          }
          int read = 0;
          list<GraphNode *>::iterator iter = bucket[i].begin();
          while(size + read < start){
            iter++;
            read++;
          }
          read = 0;
          while(iter != bucket[i].end() && read < total_size){
            res.push_back(*iter);
            iter++;
            read++;
          }
          if(read == total_size)
             break;
          size += cur_size;
          start = size;
          total_size -= read;
      }
      return res;
  }
  int init_bucket_size(int shard_num){
     for(int i = bucket_low_bound;;i++){
       if(gcd(i,shard_num) == 1)
         return i;
     }
     return -1;
  }
  list<GraphNode*>::iterator add_node(GraphNode *node);
  GraphNode * find_node(uint64_t id, GraphNodeType type);
  void  add_neighboor(uint64_t id, GraphNodeType type, GraphEdge *edge);
  private:
    unordered_map<pair<uint64_t,GraphNodeType>,list<GraphNode *>::iterator , pair_hash> node_location;
    int bucket_size, shard_num;
    vector<list<GraphNode *>> bucket; 

};
class GraphTable : public SparseTable {
 public:
  GraphTable() {}
  virtual ~GraphTable() {}  
  virtual int32_t pull_graph_list(int start, int size, char* &buffer,int &actual_size);
  virtual int32_t random_sample(uint64_t node_id, GraphNodeType type, int sampe_size, char* &buffer, int &actual_size);
  virtual int32_t initialize();
  int32_t load(const std::string& path, const std::string& param);
  GraphNode *find_node(uint64_t id, GraphNodeType type);
protected:
  vector<GraphShard> shards;
  unordered_set<uint64_t> id_set;
  size_t shard_start,shard_end, server_num,shard_num_per_table;
  std::unique_ptr<framework::RWLock> rwlock_{nullptr};
  const int task_pool_size_ = 7;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;  
};
}
};