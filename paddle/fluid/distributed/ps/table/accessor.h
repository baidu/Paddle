// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdint.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/common/afs_wrapper.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"

namespace paddle {
namespace distributed {

struct Region {
  Region() : data(NULL), size(0) {}
  Region(char* data, size_t data_num) : data(data), size(data_num) {}
  Region(float* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 2) {}
  Region(int16_t* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 1) {}
  Region(int32_t* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 2) {}
  Region(int64_t* data, size_t data_num)
      : data(reinterpret_cast<char*>(data)), size(data_num << 3) {}
  char* data;
  size_t size;
};

struct DataConverter {
  int param;
  std::string converter;
  std::string deconverter;
};

struct AccessorInfo {
  // value维度
  size_t dim;
  // value各个维度的size
  size_t size;
  // pull value维度
  size_t select_dim;
  // pull value各维度相加总size
  size_t select_size;
  // push value维度
  size_t update_dim;
  // push value各个维度的size
  size_t update_size;
  // value中mf动态长度部分总size大小, sparse下生效
  size_t mf_size;
  // value总维度，dense下生效
  size_t fea_dim;
};

class ValueAccessor {
 public:
  ValueAccessor() {}
  virtual ~ValueAccessor() {}

  virtual int Configure(const TableAccessorParameter& parameter) {
    _config = parameter;
    // data_convert结构体初始化
    if (_config.table_accessor_save_param_size() != 0) {
      for (int i = 0; i < _config.table_accessor_save_param_size(); ++i) {
        int param = _config.table_accessor_save_param(i).param();
        std::string converter =
            _config.table_accessor_save_param(i).converter();
        std::string deconverter =
            _config.table_accessor_save_param(i).deconverter();
        _data_converter_map[param] = std::make_shared<DataConverter>();
        *(_data_converter_map[param]) = {param, converter, deconverter};
      }
    }
    return 0;
  }
  virtual int Initialize() = 0;

  virtual AccessorInfo GetAccessorInfo() { return _accessor_info; }

  virtual bool NeedExtendMF(float* value UNUSED) { return false; }
  virtual bool HasMF(size_t size UNUSED) { return false; }
  // converter for save
  virtual std::string GetConverter(int param) {
    auto itr = _data_converter_map.find(param);
    if (itr == _data_converter_map.end()) {
      return "";
    } else {
      return (*itr).second->converter;
    }
  }
  // deconverter for load
  virtual std::string GetDeconverter(int param) {
    auto itr = _data_converter_map.find(param);
    if (itr == _data_converter_map.end()) {
      return "";
    } else {
      return (*itr).second->deconverter;
    }
  }
  // 判断该value是否进行shrink
  virtual bool Shrink(float* value) = 0;

  // 判断该value是否在save阶段dump,
  // param作为参数用于标识save阶段，如downpour的xbox与batch_model
  virtual bool Save(float* value, int param) = 0;
  // update delta_score and unseen_days after save
  virtual void UpdateStatAfterSave(float* value UNUSED, int param UNUSED) {}
  // 判断该value是否保存到ssd
  virtual bool SaveSSD(float* value) = 0;
  // 判断热启时是否过滤slot对应的feasign
  virtual bool FilterSlot(float* value UNUSED) { return false; }
  virtual bool SaveFilterSlot(float* value UNUSED) { return false; }

  //
  virtual bool SaveCache(float* value,
                         int param,
                         double global_cache_threshold) = 0;

  // keys不存在时，为values生成随机值
  virtual int32_t Create(float** value, size_t num) = 0;
  virtual bool CreateValue(int type UNUSED, const float* value UNUSED) {
    return true;
  }
  // 从values中选取到select_values中
  virtual int32_t Select(float** select_values,
                         const float** values,
                         size_t num) = 0;
  // 将update_values聚合到一起
  virtual int32_t Merge(float** update_values,
                        const float** other_update_values,
                        size_t num) = 0;
  // 将update_values聚合到一起，通过it.next判定是否进入下一个key
  // virtual int32_t Merge(float** update_values, iterator it);
  // 将update_values更新应用到values中
  virtual int32_t Update(float** values,
                         const float** update_values,
                         size_t num) = 0;

  // used to save model, will filter feature
  virtual std::string ParseToString(const float* value, int param) = 0;
  //  parse value from string, used to load model
  virtual int32_t ParseFromString(const std::string& data, float* value) = 0;

  virtual FsDataConverter Converter(int param) {
    FsDataConverter data_convert;
    data_convert.converter = this->GetConverter(param);
    data_convert.deconverter = this->GetDeconverter(param);
    return data_convert;
  }

  virtual int SetWeight(float** values UNUSED,
                        const float** update_values UNUSED,
                        size_t num UNUSED) {
    return 0;
  }

  virtual bool SaveMemCache(float* value UNUSED,
                            int param UNUSED,
                            double global_cache_threshold UNUSED,
                            uint16_t pass_id UNUSED) {
    return true;
  }

  virtual void UpdatePassId(float* value UNUSED, uint16_t pass_id UNUSED) {}

  virtual float GetField(float* value UNUSED, const std::string& name UNUSED) {
    return 0.0;
  }
  virtual robin_hood::unordered_set<float>* GetFilteredSlots() {
    return nullptr;
  }
  virtual robin_hood::unordered_set<float>* GetSaveFilteredSlots() {
    return nullptr;
  }

  virtual void SetDayId(int day_id) {}
  virtual void UpdateTimeDecay(float* value, bool is_update_seen_day) {}

#define DEFINE_GET_INDEX(class, field) \
  virtual int get_##field##_index() { return class ::field##_index(); }

 protected:
  size_t _value_size;
  size_t _select_value_size;
  size_t _update_value_size;
  TableAccessorParameter _config;
  std::unordered_map<int, std::shared_ptr<struct DataConverter>>
      _data_converter_map;
  AccessorInfo _accessor_info;
};
REGISTER_PSCORE_REGISTERER(ValueAccessor);
}  // namespace distributed
}  // namespace paddle
