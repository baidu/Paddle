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

#include <unordered_map>

#include "glog/logging.h"

#include "paddle/common/errors.h"
#include "paddle/phi/core/distributed/ucc_tools.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

ucc_reduction_op_t ToUCCRedType(ReduceOp reduction) {
  static const std::unordered_map<ReduceOp, ucc_reduction_op_t> red_type = {
      {ReduceOp::MIN, UCC_OP_MIN},
      {ReduceOp::MAX, UCC_OP_MAX},
      {ReduceOp::SUM, UCC_OP_SUM},
      {ReduceOp::PRODUCT, UCC_OP_PROD},
      {ReduceOp::AVG, UCC_OP_AVG},
  };
  auto it = red_type.find(reduction);
  PADDLE_ENFORCE_EQ(
      it != red_type.end(),
      true,
      common::errors::InvalidArgument(
          "Invalid ucc reduction. Must be UCC_OP_MIN | UCC_OP_MAX | "
          "UCC_OP_PROD | UCC_OP_SUM | UCC_OP_AVG."));
  return it->second;
}

ucc_memory_type_t ToUCCMemType(phi::AllocationType type) {
  static const std::unordered_map<phi::AllocationType, ucc_memory_type_t>
      mem_type = {
          {phi::AllocationType::CPU, UCC_MEMORY_TYPE_HOST},
          {phi::AllocationType::GPU, UCC_MEMORY_TYPE_CUDA},
      };
  auto it = mem_type.find(type);
  if (it == mem_type.end()) {
    return UCC_MEMORY_TYPE_UNKNOWN;
  } else {
    return it->second;
  }
}

std::string UCCDTypeToString(ucc_datatype_t dtype) {
#define PD_UCC_DTYPE_TO_STR(__ucc_dtype, __str_dtype) \
  if (dtype == __ucc_dtype) return __str_dtype;
  PD_UCC_DTYPE_TO_STR(UCC_DT_FLOAT16, "float16");
  PD_UCC_DTYPE_TO_STR(UCC_DT_FLOAT32, "float32");
  PD_UCC_DTYPE_TO_STR(UCC_DT_FLOAT64, "float64");
  PD_UCC_DTYPE_TO_STR(UCC_DT_BFLOAT16, "bfloat16");
  PD_UCC_DTYPE_TO_STR(UCC_DT_INT8, "int8");
  PD_UCC_DTYPE_TO_STR(UCC_DT_UINT8, "uint8");
  PD_UCC_DTYPE_TO_STR(UCC_DT_INT32, "int32");
  PD_UCC_DTYPE_TO_STR(UCC_DT_INT64, "int64");

#undef PD_UCC_DTYPE_TO_STR
  PADDLE_THROW(common::errors::InvalidArgument(
      "This datatype %d in ucc is not supported.", static_cast<int>(dtype)));
}

std::string UCCRedTypeToString(ucc_reduction_op_t op) {
  if (op == UCC_OP_SUM) return "SUM";
  if (op == UCC_OP_PROD) return "PROD";
  if (op == UCC_OP_MIN) return "MIN";
  if (op == UCC_OP_MAX) return "MAX";
  if (op == UCC_OP_AVG) return "AVG";
  return "UDF_" + std::to_string(op);
}

namespace {
// Constants for store keys.
constexpr char kTeamRank[] = "teamr";
constexpr char kAllGatherDone[] = "ag_done";
constexpr char kAllGatherFree[] = "ag_free";
}  // namespace

ucc_status_t OobAllgather(
    void* sbuf, void* rbuf, size_t msglen, void* coll_info, void** req) {
  auto* info = reinterpret_cast<PaddleUCCOobCollInfo*>(coll_info);
  PADDLE_CHECK(info != nullptr);
  std::vector<uint8_t> val =
      std::vector<uint8_t>(reinterpret_cast<uint8_t*>(sbuf),
                           reinterpret_cast<uint8_t*>(sbuf) + msglen);
  try {
    info->store->set(info->GetKey(kTeamRank + std::to_string(info->rank)), val);
    info->rbuf = rbuf;
    info->msglen = msglen;
    *req = coll_info;
  } catch (std::exception& ex) {
    LOG(ERROR) << "(OobAllgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    return UCC_ERR_NO_MESSAGE;
  }
  return UCC_OK;
}

ucc_status_t OobAllgatherTest(void* req) {
  auto* info = reinterpret_cast<PaddleUCCOobCollInfo*>(req);
  PADDLE_CHECK(info != nullptr);

  try {
    for (int r = 0; r < info->size; r++) {
      if (!info->store->check({info->GetKey(kTeamRank + std::to_string(r))})) {
        return UCC_INPROGRESS;
      }
    }
    for (int r = 0; r < info->size; r++) {
      std::vector<uint8_t> data =
          info->store->get(info->GetKey(kTeamRank + std::to_string(r)));
      memcpy(reinterpret_cast<void*>(reinterpret_cast<ptrdiff_t>(info->rbuf) +
                                     info->msglen * r),
             data.data(),
             info->msglen);
    }
  } catch (std::exception& ex) {
    LOG(ERROR) << "(OobAllgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    return UCC_ERR_NO_MESSAGE;
  }
  return UCC_OK;
}

ucc_status_t OobAllgatherFree(void* req) {
  auto* info = reinterpret_cast<PaddleUCCOobCollInfo*>(req);
  PADDLE_CHECK(info != nullptr);
  try {
    int num_done = info->store->add({info->GetKey(kAllGatherDone)}, 1);
    if (num_done == info->size) {
      info->store->deleteKey(info->GetKey(kAllGatherDone));
      // Note: to avoid race condition, it's important to remove all keys in
      // OobAllgatherFree first and only after that signal completion to
      // other ranks
      for (int i = 0; i < info->size; ++i) {
        info->store->deleteKey(info->GetKey(kTeamRank + std::to_string(i)));
      }
      for (int i = 0; i < info->size; ++i) {
        info->store->add({info->GetKey(kAllGatherFree + std::to_string(i))}, 1);
      }
    } else {
      info->store->wait(
          {info->GetKey(kAllGatherFree + std::to_string(info->rank))});
    }
    info->store->deleteKey(
        info->GetKey(kAllGatherFree + std::to_string(info->rank)));
  } catch (std::exception& ex) {
    LOG(ERROR) << "(OobAllgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    return UCC_ERR_NO_MESSAGE;
  }
  return UCC_OK;
}

UCCCommBase::UCCCommBase(std::shared_ptr<PaddleUCCOobCollInfo> oob) {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  UCC_CHECK(ucc_lib_config_read("PADDLE", nullptr, &lib_config),
            "failed to read UCC lib config");
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  UCC_CHECK(ucc_init(&lib_params, lib_config, &lib), "failed to init UCC lib");
  ucc_lib_config_release(lib_config);
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  UCC_CHECK(ucc_lib_get_attr(lib, &lib_attr), "failed to query for lib attr");
  // ucc library wasn't initialized with multithreading support,
  // please check ucc build options
  PADDLE_CHECK(lib_attr.thread_mode == UCC_THREAD_MULTIPLE);
  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    UCC_CHECK(ucc_finalize(lib),
              "failed to finalize UCC library when failing to read UCC context "
              "config");
    LOG(ERROR) << "failed to read UCC context config: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  st = ucc_context_config_modify(context_config,
                                 NULL,
                                 "ESTIMATED_NUM_EPS",
                                 std::to_string(oob->size).c_str());
  if (st != UCC_OK) {
    ucc_context_config_release(context_config);
    ucc_finalize(lib);
    LOG(ERROR) << "UCC failed to modify UCC context config: "
               << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.n_oob_eps = oob->size;
  context_params.oob.oob_ep = oob->rank;
  context_params.oob.allgather = OobAllgather;
  context_params.oob.req_test = OobAllgatherTest;
  context_params.oob.req_free = OobAllgatherFree;
  context_params.oob.coll_info = oob.get();
  st = ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  if (st != UCC_OK) {
    UCC_CHECK(
        ucc_finalize(lib),
        "failed to finalize UCC library when failing to creat UCC context");
    LOG(ERROR) << "UCC failed to create UCC context: " << ucc_status_string(st);
    throw std::runtime_error(ucc_status_string(st));
  }
}

void UCCCommBase::Progress() {
  UCC_CHECK(ucc_context_progress(context), "failed to progress UCC collective");
}

void UCCCommBase::FreeRequest(ucc_coll_req_h request) {
  UCC_CHECK(ucc_collective_finalize(request), "failed to release UCC request");
}

UCCCommBase::~UCCCommBase() {
  try {
    if (context != nullptr) {
      UCC_CHECK(ucc_context_destroy(context), "failed to destroy UCC context");
    }
    if (lib != nullptr) {
      UCC_CHECK(ucc_finalize(lib), "failed to finalize UCC library");
    }
    context = nullptr;
    lib = nullptr;
  } catch (std::exception& ex) {
    context = nullptr;
    lib = nullptr;
    LOG(ERROR) << "(~UCCCommBase) Caught exception "
               << "[" << ex.what() << "]";
  }
}

}  //  namespace distributed
}  // namespace phi
