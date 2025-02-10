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

#include "paddle/fluid/distributed/ps/table/table.h"

#include "glog/logging.h"
#include "paddle/fluid/distributed/common/registerer.h"
#include "paddle/fluid/distributed/ps/table/common_graph_table.h"
#include "paddle/fluid/distributed/ps/table/ctr_accessor.h"
#include "paddle/fluid/distributed/ps/table/ctr_double_accessor.h"
#include "paddle/fluid/distributed/ps/table/ctr_dymf_accessor.h"
#include "paddle/fluid/distributed/ps/table/memory_dense_table.h"
#include "paddle/fluid/distributed/ps/table/memory_sparse_geo_table.h"
#include "paddle/fluid/distributed/ps/table/memory_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/sparse_accessor.h"
#include "paddle/fluid/distributed/ps/table/ssd_sparse_table.h"
#include "paddle/fluid/distributed/ps/table/tensor_accessor.h"

namespace paddle::distributed {
REGISTER_PSCORE_CLASS(Table, GraphTable);
REGISTER_PSCORE_CLASS(Table, MemoryDenseTable);
REGISTER_PSCORE_CLASS(Table, BarrierTable);
// REGISTER_PSCORE_CLASS(Table, TensorTable);
// REGISTER_PSCORE_CLASS(Table, DenseTensorTable);
// REGISTER_PSCORE_CLASS(Table, GlobalStepTable);
REGISTER_PSCORE_CLASS(Table, MemorySparseTable);
REGISTER_PSCORE_CLASS(Table, SSDSparseTable);
REGISTER_PSCORE_CLASS(Table, MemorySparseGeoTable);

REGISTER_PSCORE_CLASS(ValueAccessor, CommMergeAccessor);
REGISTER_PSCORE_CLASS(ValueAccessor, CtrCommonAccessor);
REGISTER_PSCORE_CLASS(ValueAccessor, CtrDoubleAccessor);
REGISTER_PSCORE_CLASS(ValueAccessor, CtrDymfAccessor);
REGISTER_PSCORE_CLASS(ValueAccessor, SparseAccessor);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, StdAdaGradSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdamSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseNaiveSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdaGradSGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseAdaGradV2SGDRule);
REGISTER_PSCORE_CLASS(SparseValueSGDRule, SparseSharedAdamSGDRule);

int32_t TableManager::Initialize() {
  static bool initialized = false;
  if (initialized) {
    return 0;
  }
  initialized = true;
  return 0;
}

int32_t Table::Initialize(const TableParameter &config,
                          const FsClientParameter &fs_config) {
  _config = config;
  if (InitializeAccessor() != 0) {
    LOG(WARNING) << "Table accessor initialize failed";
    return -1;
  }

  if (_afs_client.initialize(fs_config) != 0) {
    LOG(WARNING) << "Table fs_client initialize failed";
    // return -1;
  }
  _fs_name = fs_config.uri();
  _fs_user = fs_config.user();
  _pass_wd = fs_config.passwd();
  return Initialize();
}

int32_t Table::InitializeAccessor() {
  if (!_config.has_accessor() || !_config.accessor().has_accessor_class()) {
    LOG(ERROR) << "missing accessor config in table, table_id:"
               << _config.table_id();
    return -1;
  }

  LOG(INFO) << "accessor initializing: table_id: " << _config.table_id()
            << ", accessor_name: " << _config.accessor().accessor_class();
  auto *accessor =
      CREATE_PSCORE_CLASS(ValueAccessor, _config.accessor().accessor_class());

  if (accessor == nullptr) {
    LOG(ERROR) << "accessor is unregistered, table_id:" << _config.table_id()
               << ", accessor_name:" << _config.accessor().accessor_class();
    return -1;
  }
  if (accessor->Configure(_config.accessor()) || accessor->Initialize() != 0) {
    LOG(ERROR) << " accessor initialize failed, table_id:" << _config.table_id()
               << ", accessor_name:" << _config.accessor().accessor_class();
    return -1;
  }
  _value_accessor.reset(accessor);
  return 0;
}

}  // namespace paddle::distributed
