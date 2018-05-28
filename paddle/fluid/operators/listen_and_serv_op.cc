/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>  // for removing the port file
#include <fstream>
#include <ostream>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/operators/listen_and_serv_op.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

void RunServer(std::shared_ptr<detail::AsyncGRPCServer> service) {
  service->RunSyncUpdate();
  VLOG(4) << "RunServer thread end";
}

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

static void ParallelExecuteBlocks(
    const std::vector<size_t> &parallel_blkids, framework::Executor *executor,
    const std::vector<std::shared_ptr<framework::ExecutorPrepareContext>>
        &prepared,
    framework::ProgramDesc *program, framework::Scope *scope) {
  std::vector<std::future<void>> fs;
  for (size_t idx : parallel_blkids) {
    fs.push_back(
        framework::Async([&executor, &prepared, &program, &scope, idx]() {
          int run_block = idx;  // thread local
          try {
            executor->RunPreparedContext(prepared[run_block].get(), scope);
          } catch (std::exception &e) {
            LOG(ERROR) << "run sub program error " << e.what();
          }
        }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

std::atomic_int ListenAndServOp::selected_port_{0};

ListenAndServOp::ListenAndServOp(const std::string &type,
                                 const framework::VariableNameMap &inputs,
                                 const framework::VariableNameMap &outputs,
                                 const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

void ListenAndServOp::Stop() {
  request_handler_->SetExit();
  rpc_service_->ShutDown();

  server_thread_->join();
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  remove(file_path.c_str());
}

void ListenAndServOp::SavePort() const {
  // NOTE: default write file to /tmp/paddle.selected_port
  selected_port_ = rpc_service_->GetSelectedPort();
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  std::ofstream port_file;
  port_file.open(file_path);
  port_file << selected_port_.load();
  port_file.close();
  VLOG(4) << "selected port written to " << file_path;
}

void ListenAndServOp::WaitServerReady() {
  while (selected_port_.load() == 0) {
  }
}

void ListenAndServOp::RunSyncLoop(framework::Executor *executor,
                                  framework::ProgramDesc *program,
                                  framework::Scope *recv_scope,
                                  framework::BlockDesc *prefetch_block) const {
  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    "server program should have at least 2 blocks");

  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    block_list.push_back(blkid);
  }
  auto optimize_prepared = executor->Prepare(*program, block_list);
  // Insert placeholder for block0 which holds current op itself.
  optimize_prepared.insert(
      optimize_prepared.begin(),
      std::shared_ptr<framework::ExecutorPrepareContext>(nullptr));

  request_handler_->clear_to_init();
  while (true) {
    // Get from multiple trainers, we don't care about the order in which
    // the gradients arrives, just add suffix 0~n and merge the gradient.
    rpc_service_->SetCond(static_cast<int>(detail::GrpcMethod::kSendVariable));
    request_handler_->WaitBarrier();
    request_handler_->clear_to_init();

    if (request_handler_->IsExit()) {
      LOG(WARNING) << "get exit!rpc_processor break!";
      rpc_service_->SetCond(static_cast<int>(detail::GrpcMethod::kGetVariable));
      break;
    }

    // NOTE: if is_gpu_place, CUDA kernels are launched by multiple threads
    // and this will still work.
    // The optimize blocks which have the same parent ID would run parallel
    // TODO(Yancey1989): need to use ParallelExecutor for future
    int32_t last_parent_blkid = program->Block(1).Parent();
    std::vector<size_t> parallel_blkids;
    parallel_blkids.push_back(1);
    double ts = detail::GetTimestamp();
    for (size_t blkid = 2; blkid < num_blocks; ++blkid) {
      if (blkid != static_cast<size_t>(prefetch_block->ID())) {
        if (program->Block(blkid).Parent() != last_parent_blkid) {
          ParallelExecuteBlocks(parallel_blkids, executor, optimize_prepared,
                                program, recv_scope);
          parallel_blkids.clear();
          last_parent_blkid = program->Block(blkid).Parent();
        }
        parallel_blkids.push_back(blkid);
      }
    }
    ParallelExecuteBlocks(parallel_blkids, executor, optimize_prepared, program,
                          recv_scope);
    VLOG(2) << "run all blocks spent " << detail::GetTimestamp() - ts << "(ms)";

    // Reset the received sparse variables, the sum operator would not
    // sum the input sparse variables which rows is empty at the next
    // mini-batch.
    // TODO(Yancey1989): move the reset action into an operator, we couldn't
    // have any hide logic in the operator.
    for (auto &var : request_handler_->sparse_vars()) {
      var->GetMutable<framework::SelectedRows>()->mutable_rows()->clear();
    }

    rpc_service_->SetCond(static_cast<int>(detail::GrpcMethod::kGetVariable));
    request_handler_->WaitBarrier();
    request_handler_->clear_to_init();
  }  // while(true)
}

void ListenAndServOp::RunAsyncLoop(framework::Executor *executor,
                                   framework::ProgramDesc *program) const {
  VLOG(3) << "RunAsyncLoop in";
  // grad name to block id
  std::unordered_map<std::string, int32_t> grad_to_block_id;
  std::unordered_map<int32_t, std::string> id_to_grad;

  auto grad_to_block_id_str =
      Attr<std::vector<std::string>>("grad_to_block_id");
  for (auto &grad_and_id : grad_to_block_id_str) {
    std::vector<std::string> pieces;
    split(grad_and_id, ':', &pieces);
    VLOG(3) << "after split, grad = " << pieces[0] << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2);
    PADDLE_ENFORCE_EQ(grad_to_block_id.count(pieces[0]), 0);

    int block_id = std::stoi(pieces[1]);
    grad_to_block_id[pieces[0]] = block_id;
    id_to_grad[block_id] = pieces[0];
  }
  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    "server program should have at least 2 blocks");

  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    block_list.push_back(blkid);
  }
  auto optimize_prepared = executor->Prepare(*program, block_list);
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      grad_to_prepared_ctx;
  for (size_t i = 0; i < block_list.size(); ++i) {
    grad_to_prepared_ctx[id_to_grad[block_list[i]]] = optimize_prepared[i];
  }

  request_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);

  VLOG(3) << "RunAsyncLoop into while";
  while (true) {
    if (request_handler_->IsExit()) {
      LOG(WARNING) << "get exit!rpc_processor break!";
      break;
    }

    sleep(1);
  }  // while(true)
}

void ListenAndServOp::RunImpl(const framework::Scope &scope,
                              const platform::Place &dev_place) const {
  // Mark this as PS that it should decide profiling by listening from trainer.
  platform::SetProfileListener();
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  framework::Scope &recv_scope = scope.NewScope();

  bool sync_mode = Attr<bool>("sync_mode");
  auto fan_in = Attr<int>("Fanin");

  PADDLE_ENFORCE(!rpc_service_);
  std::string endpoint = Attr<std::string>("endpoint");

  request_handler_.reset(new detail::GRPCRequestHandler(sync_mode, fan_in));
  rpc_service_.reset(
      new detail::AsyncGRPCServer(endpoint, request_handler_.get()));

  rpc_service_->RegisterCond(
      static_cast<int>(detail::GrpcMethod::kSendVariable));
  rpc_service_->RegisterCond(
      static_cast<int>(detail::GrpcMethod::kGetVariable));

  auto *optimize_block = Attr<framework::BlockDesc *>(kOptimizeBlock);
  auto *prefetch_block = Attr<framework::BlockDesc *>(kPrefetchBlock);
  auto *program = optimize_block->Program();
  framework::Executor executor(dev_place);

  // prepare rpc processor
  request_handler_->SetScope(&recv_scope);
  request_handler_->SetDevCtx(&dev_ctx);
  request_handler_->SetProgram(program);
  request_handler_->SetExecutor(&executor);

  // prepare for prefetch
  VLOG(3) << "prefetch block id is " << prefetch_block->ID();
  auto prefetch_prepared = executor.Prepare(*program, prefetch_block->ID());
  request_handler_->SetPrefetchPreparedCtx(std::move(prefetch_prepared));

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, rpc_service_));
  VLOG(3) << "wait server thread to become ready...";
  rpc_service_->WaitServerReady();

  // Write to a file of server selected port for python use.
  std::string file_path = string::Sprintf("/tmp/paddle.%d.selected_port",
                                          static_cast<int>(::getpid()));
  SavePort();
  if (sync_mode) {
    RunSyncLoop(&executor, program, &recv_scope, prefetch_block);
  } else {
    RunAsyncLoop(&executor, program);
  }
}

class ListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(R"DOC(
ListenAndServ operator

This operator will start a RPC server which can receive variables
from send_op and send back variables to recv_op.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<std::vector<std::string>>(
        "grad_to_block_id",
        "['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'] "
        "a map from grad name to it's optimize block id")
        .SetDefault({});
    AddAttr<bool>("sync_mode", "if works at sync_mode or not").SetDefault(true);
    AddAttr<framework::BlockDesc *>(kOptimizeBlock,
                                    "BlockID to run on server side.");
    AddAttr<framework::BlockDesc *>(kPrefetchBlock,
                                    "prefetch block to run on server side.");
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(listen_and_serv, ops::ListenAndServOp,
                  ops::ListenAndServOpMaker);
