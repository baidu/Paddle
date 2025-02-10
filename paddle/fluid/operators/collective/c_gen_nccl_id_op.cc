/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <string>

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/gen_comm_id_helper.h"

COMMON_DECLARE_bool(dynamic_static_unified_comm);
namespace paddle::operators {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
static void GenNCCLID(std::vector<ncclUniqueId>* nccl_ids) {
  for (auto& nccl_id : *nccl_ids) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclGetUniqueId(&nccl_id));
  }
}

static void CopyNCCLIDToVar(const std::vector<ncclUniqueId>& nccl_ids,
                            std::function<std::string(size_t)> func,
                            const framework::Scope& scope) {
  for (size_t i = 0; i < nccl_ids.size(); ++i) {
    std::string var_name = func(i);
    auto var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound("Variable with name %s is not found",
                                 var_name.c_str()));
    auto nccl_id = var->GetMutable<ncclUniqueId>();
    memcpy(nccl_id, &nccl_ids[i], sizeof(ncclUniqueId));
  }
}

class CGenNCCLIdOp : public framework::OperatorBase {
 public:
  CGenNCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {
    int rank = Attr<int>("rank");
    int ring_id = Attr<int>("ring_id");

    std::function<std::string(size_t)> func = [&](size_t i) -> std::string {
      return Output("Out");
    };

    std::string endpoint = Attr<std::string>("endpoint");

    std::vector<ncclUniqueId> nccl_ids;
    nccl_ids.resize(1);

    if (!FLAGS_dynamic_static_unified_comm) {
      int server_fd = platform::SocketServer::GetInstance(endpoint).socket();
      if (rank == 0) {
        GenNCCLID(&nccl_ids);
        std::vector<std::string> endpoint_list =
            Attr<std::vector<std::string>>("other_endpoints");
        platform::SendBroadCastCommID(endpoint_list, &nccl_ids, ring_id);
      } else {
        platform::RecvBroadCastCommID(server_fd, endpoint, &nccl_ids, ring_id);
      }
    }

    CopyNCCLIDToVar(nccl_ids, func, scope);
  }
};

#else
class CGenNCCLIdOp : public framework::OperatorBase {
 public:
  CGenNCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {}
};

#endif

class CGenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Raw variable contains a NCCL UniqueId instances.");
    AddComment(R"DOC(
CGenNCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string), e.g. 127.0.0.1:6175 "
                         "current listen endpoint");
    AddAttr<std::vector<std::string>>(
        "other_endpoints",
        "['trainer1_ip:port', 'trainer2_ip:port', ...] "
        "list of other trainer endpoints")
        .SetDefault({});
    AddAttr<int>("rank",
                 "(int default 0) "
                 "The rank of the trainer in distributed training.")
        .SetDefault(0);
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_gen_nccl_id, ops::CGenNCCLIdOp, ops::CGenNCCLIdOpMaker);
