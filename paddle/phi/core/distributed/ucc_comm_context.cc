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

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/check/static_check.h"
#include "paddle/phi/core/distributed/ucc_comm_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi::distributed {

namespace {
struct paddle_ucc_config_t {
  std::array<bool, 32> blocking_wait;
  // Sharing UCC communicator among multiple PGs to save resource.
  bool shared_comm;
  // Using allgatherv to achieve allgather, without flattening the list of
  // (potentially non-contiguous) tensors.
  bool use_allgatherv;
} paddle_ucc_config;

std::unordered_map<std::string, std::string> paddle_ucc_envs_map = {
    // PADDLE_UCC_BLOCKING_WAIT allowed syntax:
    // - PADDLE_UCC_BLOCKING_WAIT=none --> blocking wait completely disabled
    // - PADDLE_UCC_BLOCKING_WAIT=all --> blocking wait completely enabled
    // - PADDLE_UCC_BLOCKING_WAIT=allreduce,send,recv --> blocking wait enabled
    //                                                   on selected operations
    // Supported operations:
    // [allgather,allgather_base,allreduce,alltoall,broadcast,
    //  gather,reduce,reduce_scatter,scatter,send,recv]
    {"PADDLE_UCC_BLOCKING_WAIT", "none"},
    {"PADDLE_UCC_SHARED_COMM", "1"},
    {"PADDLE_UCC_USE_ALLGATHERV", "0"},
};

std::vector<CommType> parse_blocking_wait(std::string op_list_string) {
  static const std::unordered_map<std::string, CommType> str2op = {
      {"allgather", CommType::ALLGATHER},
      {"allreduce", CommType::ALLREDUCE},
      {"broadcast", CommType::BROADCAST},
      {"gather", CommType::GATHER},
      {"reduce", CommType::REDUCE},
      {"reduce_scatter", CommType::REDUCE_SCATTER},
      {"scatter", CommType::SCATTER},
      {"send", CommType::SEND},
      {"recv", CommType::RECV},
  };
  auto op_list = ParseList(op_list_string);
  if (op_list == std::vector<std::string>{"none"}) {
    return {};
  }
  std::vector<CommType> result;
  if (op_list == std::vector<std::string>{"all"}) {
    for (auto entry : str2op) {
      result.push_back(entry.second);
    }
  } else {
    for (auto op_string : op_list) {
      result.push_back(str2op.at(op_string));
    }
  }
  return result;
}

void read_config() {
  // default configuration
  paddle_ucc_config.blocking_wait.fill(false);
  paddle_ucc_config.shared_comm = false;
  paddle_ucc_config.use_allgatherv = false;

  // read all paddle_ucc env. variables and update the map
  char* env;
  for (auto& paddle_ucc_env : paddle_ucc_envs_map) {
    env = std::getenv(paddle_ucc_env.first.c_str());
    if (env) {
      paddle_ucc_envs_map[paddle_ucc_env.first] = std::string(env);
    }
  }

  auto blocking_wait_str = paddle_ucc_envs_map.at("PADDLE_UCC_BLOCKING_WAIT");
  for (auto op : parse_blocking_wait(blocking_wait_str)) {
    paddle_ucc_config.blocking_wait[(std::uint8_t)op] = true;
  }
  // barrier is always blocking
  paddle_ucc_config.blocking_wait[(std::uint8_t)CommType::BARRIER] = true;

  paddle_ucc_config.shared_comm =
      std::stoi(paddle_ucc_envs_map.at("PADDLE_UCC_SHARED_COMM"));
  paddle_ucc_config.use_allgatherv =
      std::stoi(paddle_ucc_envs_map.at("PADDLE_UCC_USE_ALLGATHERV"));
}

void check_device(Place place1, Place place2) {
  if (is_gpu_place(place1) && is_gpu_place(place1) && place1 != place1) {
    throw std::invalid_argument("UCCCommContext multidevice is not supported");
  }
}

void check_tensor(const phi::DenseTensor& tensor) {
  if (!tensor.meta().is_contiguous()) {
    throw std::invalid_argument(
        "ProcessGroupUCC input tensor has to be contiguous");
  }
}

}  // namespace

// UCCCommTask
UCCCommContext::UCCCommTask::~UCCCommTask() {}

bool UCCCommContext::UCCCommTask::IsCompleted() { return !entry_; }

bool UCCCommContext::UCCCommTask::Wait(std::chrono::milliseconds timeout) {
  while (!IsCompleted()) {
  }
  return true;
}

// UCCCommContext
UCCCommContext::UCCCommContext(const std::shared_ptr<Store>& store,
                               int rank,
                               int size,
                               std::chrono::duration<float> timeout)
    : CommContext(rank, size), timeout_(timeout) {
  read_config();
  oob = std::make_shared<PaddleUCCOobCollInfo>();
  oob->rank = rank;
  oob->size = size;
  oob->store = store;
  ucc_comm = nullptr;
  cuda_ee = nullptr;

  std::string envs = "";
  for (auto& paddle_ucc_env : paddle_ucc_envs_map) {
    envs += ("\n\t" + paddle_ucc_env.first + "=" + paddle_ucc_env.second);
  }
  VLOG(3) << "UCCCommContext: "
          << "\n\tPaddle UCC Comm Context created with the following env. "
             "variables:"
          << envs;
}

UCCCommContext::~UCCCommContext() {
  if (ucc_comm) {
    ucc_comm->UCCDestroyTeam(&team);
    VLOG(3) << "Successfully destroyed UCC library";
    try {
      if (cuda_ee) {
        ucc_ee_destroy(cuda_ee);
        ucc_ee_destroy(cuda_ee_p2p[0]);
        ucc_ee_destroy(cuda_ee_p2p[1]);
      }
    } catch (std::exception& ex) {
      LOG(ERROR) << "Caught error in Store Operation .. [" << ex.what() << "]";
    }
    ucc_comm = nullptr;
  }
}

void UCCCommContext::InitComm(Place place) {
  if (!ucc_comm) {
    ucc_comm = UCCComm::GetComm(&comm_id, place, oob);
    VLOG(3) << "Successfully initialized UCX library";
    ucc_comm->UCCCreateTeam(&team, oob);
    VLOG(3) << "Successfully initialized UCC library";
  } else {
    if (is_gpu_place(place)) {
      if ((ucc_comm->cuda_device_index != PADDLE_UCC_DEVICE_NOT_SET) &&
          (ucc_comm->cuda_device_index != place.GetDeviceId())) {
        LOG(ERROR)
            << "ucc communicator was initialized with different cuda device,"
            << "multi device is not supported";
        throw std::invalid_argument(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
      }
      ucc_comm->cuda_device_index = place.GetDeviceId();
    }
  }
}

template <typename PreProcess, typename PostProcess>
std::shared_ptr<UCCCommContext::UCCCommTask> UCCCommContext::CollectivePost(
    CommType commType,
    PreProcess preproc,
    PostProcess postproc,
    ucc_coll_args_t* coll,
    Place place) {
  seq_++;
  SetTimeout(coll);
  auto task = std::make_shared<UCCCommContext::UCCCommTask>(seq_);

  if (is_cpu_place(place)) {
    preproc();
    ucc_comm->EnqueueCollective(task, coll, team);
    postproc();
    return task;
  }

  LOG(ERROR) << "unsupported device type: " << place.GetDeviceType();
  throw std::invalid_argument(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
}

std::shared_ptr<UCCCommContext::UCCCommTask> UCCCommContext::Reduce(
    phi::DenseTensor* out_tensor,
    const phi::DenseTensor& in_tensor,
    ucc_reduction_op_t reduce_type,
    int root,
    gpuStream_t stream) {
  check_tensor(in_tensor);
  InitComm(in_tensor.place());

  ucc_coll_args_t coll;
  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
  coll.coll_type = UCC_COLL_TYPE_REDUCE;
  coll.op = reduce_type;
  coll.root = root;

  coll.src.info.buffer = const_cast<void*>(in_tensor.data());
  coll.src.info.count = in_tensor.numel();
  coll.src.info.datatype = phi::ToUCCDataType(in_tensor.dtype());
  coll.src.info.mem_type = ToUCCMemType(in_tensor.place().GetType());
  coll.dst.info.buffer = const_cast<void*>(out_tensor->data());
  coll.dst.info.count = out_tensor->numel();
  coll.dst.info.datatype = phi::ToUCCDataType(out_tensor->dtype());
  coll.dst.info.mem_type = ToUCCMemType(out_tensor->place().GetType());

  return CollectivePost(
      CommType::REDUCE, []() {}, []() {}, &coll, in_tensor.place());
}

void UCCCommContext::SetTimeout(ucc_coll_args_t* args) {
  args->mask |= UCC_COLL_ARGS_FIELD_FLAGS;
  args->flags |= UCC_COLL_ARGS_FLAG_TIMEOUT;
  args->timeout = timeout_.count();
}

UCCComm::UCCComm(std::shared_ptr<PaddleUCCOobCollInfo> oob_, Place place)
    : UCCCommBase(oob_),
      oob(oob_),
      cuda_device_index(PADDLE_UCC_DEVICE_NOT_SET) {
  if (is_gpu_place(place)) {
    cuda_device_index = place.GetDeviceId();
  }
  stop_progress_loop = false;
  collective_inprogress = false;
  progress_thread = std::thread(&UCCComm::ProgressLoop, this);
#ifdef _GNU_SOURCE
  pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
#endif
}

UCCComm::~UCCComm() {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
}

void UCCComm::UCCCreateTeam(ucc_team_h* team,
                            std::shared_ptr<PaddleUCCOobCollInfo> oob) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
                     UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = OobAllgather;
  team_params.oob.req_test = OobAllgatherTest;
  team_params.oob.req_free = OobAllgatherFree;
  team_params.oob.coll_info = oob.get();
  team_params.oob.n_oob_eps = oob->size;
  team_params.oob.oob_ep = oob->rank;
  team_params.ep = oob->rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  UCC_CHECK(ucc_team_create_post(&context, 1, &team_params, team),
            "failed to post team create");
  do {
    st = ucc_team_create_test(*team);
    ucc_context_progress(context);
  } while (st == UCC_INPROGRESS);
  UCC_CHECK(st, "failed to create UCC team");
}

void UCCComm::UCCDestroyTeam(ucc_team_h* team) {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(
      lock, [&] { return progress_queue.empty() && !collective_inprogress; });

  ucc_status_t status;
  while (UCC_INPROGRESS == (status = ucc_team_destroy(*team))) {
    if (UCC_OK != status) {
      LOG(ERROR) << "Failed to destroy UCC team";
      break;
    }
  }

  lock.unlock();
}

std::shared_ptr<UCCComm> UCCComm::GetComm(
    uint32_t* id, Place place, std::shared_ptr<PaddleUCCOobCollInfo> oob) {
  static std::mutex m;
  static std::weak_ptr<UCCComm> comm;
  static uint32_t comm_id;

  std::lock_guard<std::mutex> lock(m);
  *id = comm_id;

  std::string group_id = "group_id";

  std::vector<uint8_t> remote_comm_id;
  oob->store->deleteKey(group_id + std::to_string(0));
  if (oob->rank != 0) {
    std::vector<uint8_t> val =
        std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&id),
                             reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set(group_id + std::to_string(oob->rank), val);
  } else {
    for (int i = 1; i < oob->size; i++) {
      remote_comm_id = oob->store->get(group_id + std::to_string(i));
      oob->store->deleteKey(group_id + std::to_string(i));
      // Find the highest id.
      *id =
          std::max(*id, *(reinterpret_cast<uint32_t*>(remote_comm_id.data())));
    }
    std::vector<uint8_t> val =
        std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&id),
                             reinterpret_cast<uint8_t*>(&id) + sizeof(id));
    oob->store->set(group_id + std::to_string(oob->rank), val);
  }
  remote_comm_id = oob->store->get(group_id + std::to_string(0));
  oob->comm_id = *(reinterpret_cast<uint32_t*>(remote_comm_id.data()));
  // Prepare comm_id (static variable) to the next id.
  comm_id = oob->comm_id + 1;

  if (paddle_ucc_config.shared_comm) {
    std::shared_ptr<UCCComm> shared_comm = comm.lock();
    if (!shared_comm) {
      shared_comm = std::make_shared<UCCComm>(oob, place);
      comm = shared_comm;
    } else {
      if (is_gpu_place(place)) {
        if ((shared_comm->cuda_device_index != PADDLE_UCC_DEVICE_NOT_SET) &&
            (shared_comm->cuda_device_index != place.GetDeviceId())) {
          LOG(ERROR)
              << "ucc communicator was initialized with different cuda device,"
              << "multi device is not supported";
          throw std::invalid_argument(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
        }
        shared_comm->cuda_device_index = place.GetDeviceId();
      }
    }
    return shared_comm;
  } else {
    return std::make_shared<UCCComm>(oob, place);
  }
}

void UCCComm::EnqueueCollective(
    std::shared_ptr<UCCCommContext::UCCCommTask> task,
    ucc_coll_args_t* coll,
    ucc_team_h team) {
  VLOG(3) << "EnqueueCollective : "
          << "coll.src.info.buffer = " << coll->src.info.buffer
          << ", coll.dst.info.buffer = " << coll->dst.info.buffer
          << ", coll.src.info.count = " << coll->src.info.count
          << ", coll.dst.info.count = " << coll->dst.info.count;

  ucc_coll_req_h request;
  UCC_CHECK(ucc_collective_init(coll, &request, team),
            "failed to init collective");
  UCC_CHECK_REQUEST(
      request, ucc_collective_post(request), "failed to post collective");

  auto entry = std::make_shared<UCCCommContext::ProgressEntry>(request);
  task->entry_ = entry;
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(task);
  lock.unlock();
  queue_produce_cv.notify_one();
}

void UCCComm::ProgressLoop() {
  std::unique_lock<std::mutex> lock(mutex);
  while (!stop_progress_loop) {
    if (progress_queue.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    collective_inprogress = true;
    auto task = progress_queue.front();
    auto entry = task->entry_;
    progress_queue.pop_front();
    lock.unlock();

    while (entry->request_->status > 0) {
      this->Progress();
    }
    VLOG(3) << "collective progress finished!";

    if (entry->request_->status < 0) {
      LOG(ERROR) << "Failed to progress communication "
                 << ucc_status_string(entry->request_->status);
      throw std::runtime_error(ucc_status_string(entry->request_->status));
    }
    if (entry->request_ != nullptr) {
      this->FreeRequest(entry->request_);
    }
    entry = nullptr;
    task->entry_ = nullptr;
    collective_inprogress = false;
    queue_consume_cv.notify_one();
    lock.lock();
  }
}

}  // namespace phi::distributed
