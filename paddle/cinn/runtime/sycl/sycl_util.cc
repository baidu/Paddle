// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <dlfcn.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
#include "paddle/cinn/runtime/sycl/sycl_util.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/common/enforce.h"
#ifdef CINN_WITH_CNNL
#include <cn_api.h>
#include <cnnl.h>
#include <CL/sycl/backend/cnrt.hpp>
#endif

namespace cinn {
namespace runtime {
namespace sycl {

void cinn_call_sycl_kernel(void *kernel_fn,
                           void *v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z) {
  VLOG(3) << "cinn_call_sycl_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args;

  std::vector<void *> kernel_args;
  {
    cinn::utils::RecordEvent record_run("prepare_args",
                                        cinn::utils::EventType::kInstruction);
    kernel_args.reserve(num_args);
    cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t *>()) {
        void *addr = static_cast<cinn_buffer_t *>(args[idx])->memory;
        std::stringstream ss;
        ss << std::hex << addr;
        VLOG(4) << "sycl kernel arg[" << idx
                << "] is a buffer, addr=" << ss.str();
        kernel_args.emplace_back(&addr);
      }
      // else {
      //   kernel_args.emplace_back((args[idx].data_addr()));
      // }
    }
  }

  {
    cinn::utils::RecordEvent record_run("syclLaunchKernel",
                                        cinn::utils::EventType::kInstruction);
    void (*kernel_func)(::sycl::queue & Q,
                        ::sycl::range<3> k0_dimGrid,
                        ::sycl::range<3> k0_dimBlock,
                        void **void_args) =
        (void (*)(::sycl::queue & Q,
                  ::sycl::range<3> k0_dimGrid,
                  ::sycl::range<3> k0_dimBlock,
                  void **void_args))(kernel_fn);
    ::sycl::queue *Queue = SYCLBackendAPI::Global()->get_now_queue();
    ::sycl::range<3> Grid(grid_z, grid_y, grid_x);
    ::sycl::range<3> Block(block_z, block_y, block_x);
    SYCL_CALL(kernel_func(*Queue, Grid, Block, kernel_args.data()));
  }
}

void cinn_call_sycl_memset(void *v_args,
                           int num_args,
                           int value,
                           size_t count) {
  PADDLE_ENFORCE_EQ(num_args,
                    1,
                    ::common::errors::PreconditionNotMet(
                        "The cinn_call_sycl_memset only accept a output."));
  VLOG(4) << "call cinn_call_sycl_memset with value=" << value
          << ", count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *output = args[0].operator cinn_buffer_t *()->memory;

  std::stringstream ss;
  ss << std::hex << output;
  VLOG(4) << "cinn_call_sycl_memset: " << ss.str();

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();

  SYCL_CALL(Queue->memset(output, value, count));
}

void cinn_call_sycl_memcpy(void *v_args, int num_args, size_t count) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      ::common::errors::PreconditionNotMet(
          "The cinn_call_sycl_memcpy only accept a input and a output."));
  VLOG(4) << "call cinn_call_sycl_memcpy with count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *input = args[0].operator cinn_buffer_t *()->memory;
  void *output = args[1].operator cinn_buffer_t *()->memory;

  if (input == output) {
    VLOG(3) << "cinn_call_sycl_memcpy: skip copy as input addr is the same as "
               "output.";
    return;
  }

  std::stringstream ss;
  ss << std::hex << input << " -> " << output;
  VLOG(4) << "cinn_call_sycl_memcpy: " << ss.str();

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();

  SYCL_CALL(Queue->memcpy(output, input, count));
}

#ifdef CINN_WITH_CNNL

class CnnlHandle {
 public:
  CnnlHandle(const CnnlHandle &) = delete;
  CnnlHandle &operator=(const CnnlHandle &) = delete;
  ~CnnlHandle() { CNNL_CALL(cnnlDestroy(handle)); }
  static CnnlHandle &GetInstance() {
    static CnnlHandle instance;
    return instance;
  }
  cnnlHandle_t &GetCnnlHandle() { return handle; }

 private:
  CnnlHandle() { CNNL_CALL(cnnlCreate(&handle)); }
  cnnlHandle_t handle;
};

class CnnlRandGenerator {
 public:
  CnnlRandGenerator() {
    CNNL_CALL(cnnlRandCreateGenerator(&generator_, CNNL_RAND_RNG_FAST));
  }

  explicit CnnlRandGenerator(cnnlRandRngType_t rng_type) {
    CNNL_CALL(cnnlRandCreateGenerator(&generator_, rng_type));
  }

  ~CnnlRandGenerator() { CNNL_CALL(cnnlRandDestroyGenerator(generator_)); }

  cnnlRandGenerator_t &GetGenerator() { return generator_; }

  CnnlRandGenerator &SetSeed(uint64_t seed = 0ULL) {
    // set global seed if seed is zero
    auto rand_seed = (seed == 0ULL) ? RandomSeed::GetOrSet() : seed;
    if (rand_seed != 0ULL && rand_seed != seed_) {
      CNNL_CALL(cnnlRandSetPhiloxSeed(generator_, rand_seed));
      VLOG(4) << "Change curand random seed from: " << seed_
              << " to: " << rand_seed;
      seed_ = rand_seed;
    }
    return *this;
  }

 private:
  cnnlRandGenerator_t generator_;
  uint64_t seed_ = 0ULL;
};

cnnlDataType_t convert_to_cnnl_dtype(void *v_args, int num_args) {
  PADDLE_ENFORCE_GT(num_args,
                    0,
                    ::common::errors::PreconditionNotMet(
                        "the number of arguments must larger than zero"));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  int bits = args[0].operator cinn_buffer_t *()->type.bits;
  for (int i = 1; i < num_args; ++i) {
    auto t = args[i].operator cinn_buffer_t *()->type.code;
    int b = args[0].operator cinn_buffer_t *()->type.bits;
    if (t != type_code || bits != b) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "The types of all arguments need to be consistent."));
    }
  }
  cnnlDataType_t data_type;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  if (is_float && bits == 16) {
    data_type = CNNL_DTYPE_HALF;
  } else if (is_float && bits == 32) {
    data_type = CNNL_DTYPE_FLOAT;
  } else if (is_bfloat16) {
    data_type = CNNL_DTYPE_BFLOAT16;
  } else {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("unsupported cudnn data type: ",
                                          static_cast<int>(type_code),
                                          ", bits = ",
                                          bits));
  }
  return data_type;
}

std::string debug_cnnl_tensor_format(cnnlTensorLayout_t tensor_format) {
  switch (tensor_format) {
    case CNNL_LAYOUT_NCHW:
      return "NCHW";
    case CNNL_LAYOUT_NHWC:
      return "NHWC";
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support NCHW and NHWC data layout\n"));
  }
  return "";
}

std::string debug_cnnl_tensor_dtype(cnnlDataType_t tensor_dtype) {
  switch (tensor_dtype) {
    case CNNL_DTYPE_FLOAT:
      return "float32";
    case CNNL_DTYPE_HALF:
      return "float16";
    case CNNL_DTYPE_BFLOAT16:
      return "bfloat16";
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Only support float16/bfloat16/float32 now!"));
  }
  return "";
}

std::string debug_cnnl_pool_mode(cnnlPoolingMode_t pool_mode) {
  switch (pool_mode) {
    case CNNL_POOLING_MAX:
      return "max";
    case CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
      return "avg_include_padding";
    case CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
      return "avg_exclude_padding";
    case CNNL_POOLING_FIXED:
      return "fixed";
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Pool only support max and avg now!"));
  }
  return "";
}

class CnnlRandGeneratorFactory {
 public:
  enum class CnnlRandGeneratorType {
    GENERATOR_DEFAULT,
    GENERATOR_GAUSSIAN,
    GENERATOR_UNIFORM,
    GENERATOR_RANDINT,
  };

  static CnnlRandGenerator &Get(CnnlRandGeneratorType type) {
    switch (type) {
      case CnnlRandGeneratorType::GENERATOR_GAUSSIAN:
        static CnnlRandGenerator gaussian_generator(CNNL_RAND_RNG_PHILOX);
        return gaussian_generator;
      case CnnlRandGeneratorType::GENERATOR_UNIFORM:
        static CnnlRandGenerator uniform_generator(CNNL_RAND_RNG_PHILOX);
        return uniform_generator;
      case CnnlRandGeneratorType::GENERATOR_RANDINT:
        static CnnlRandGenerator randint_generator(CNNL_RAND_RNG_PHILOX);
        return randint_generator;
      default:
        static CnnlRandGenerator default_generator;
        return default_generator;
    }
  }
};

void cinn_call_cnnl_gaussian_random(
    void *v_args, int num_args, float mean, float std, int seed) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();

  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  cnnlRandGenerator_t generator =
      CnnlRandGeneratorFactory::Get(
          CnnlRandGeneratorFactory::CnnlRandGeneratorType::GENERATOR_GAUSSIAN)
          .SetSeed(seed)
          .GetGenerator();

  VLOG(4) << "cinn_call_cnnl_gaussian_random: output_size=" << numel
          << ", mean=" << mean << ", std=" << std << ", seed=" << seed;

  if (dtype == cinn_float32_t()) {
    float *ptr = reinterpret_cast<float *>(output->memory);
    CNNL_CALL(cnnlRandGenerateNormal(
        handle, generator, CNNL_DTYPE_FLOAT, NULL, numel, mean, std, ptr));
    CNRT_CALL(cnrtQueueSync(queue));
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "gaussian_random_sycl only support float32! Please check."));
  }
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_uniform_random(
    void *v_args, int num_args, float min, float max, int seed) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();

  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  cnnlRandGenerator_t generator =
      CnnlRandGeneratorFactory::Get(
          CnnlRandGeneratorFactory::CnnlRandGeneratorType::GENERATOR_UNIFORM)
          .SetSeed(seed)
          .GetGenerator();

  VLOG(4) << "cinn_call_cnnl_uniform_random: output_size=" << numel
          << ", min=" << min << ", max=" << max << ", seed=" << seed;

  if (dtype == cinn_float32_t()) {
    float *ptr = reinterpret_cast<float *>(output->memory);
    CNNL_CALL(cnnlRandGenerateUniform(
        handle, generator, CNNL_DTYPE_FLOAT, NULL, numel, 0.0f, 1.0f, ptr));
    CNRT_CALL(cnrtQueueSync(queue));
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "uniform_random_sycl only support float32! Please check."));
  }
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_randint(void *v_args, int num_args, int seed) {
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();

  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  VLOG(4) << "cinn_call_cnnl_randint: output_size=" << numel
          << ", seed=" << seed;

  cnnlRandGenerator_t generator =
      CnnlRandGeneratorFactory::Get(
          CnnlRandGeneratorFactory::CnnlRandGeneratorType::GENERATOR_RANDINT)
          .SetSeed(seed)
          .GetGenerator();

  if (dtype == cinn_int32_t()) {
    unsigned int *ptr = reinterpret_cast<unsigned int *>(output->memory);
    CNNL_CALL(cnnlRandGenerateDiscreteUniform(
        handle, generator, CNNL_DTYPE_INT32, NULL, numel, 0, 1 << 23, ptr));
    CNRT_CALL(cnrtQueueSync(queue));
  } else {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "randint only support int32! Please check."));
  }
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_matmul(void *v_args,
                           int num_args,
                           bool trans_a,
                           bool trans_b,
                           bool trans_o,
                           float alpha,
                           float beta,
                           int a1,
                           int a2,
                           int a3,
                           int a4,
                           int b1,
                           int b2,
                           int b3,
                           int b4) {
  cinn::utils::RecordEvent record_run("cinn_call_cnnl_matmul",
                                      cinn::utils::EventType::kInstruction);
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of arguments is 3, but received %d.", num_args));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));
  VLOG(3) << "a1 ~ a4: " << a1 << " " << a2 << " " << a3 << " " << a4;
  VLOG(3) << "b1 ~ b4: " << b1 << " " << b2 << " " << b3 << " " << b4;
  VLOG(3) << "trans_a: " << trans_a << ", trans_b: " << trans_b
          << ", trans_o: " << trans_o;

  void *A = args[0].operator cinn_buffer_t *()->memory;
  void *B = args[1].operator cinn_buffer_t *()->memory;
  void *C = args[2].operator cinn_buffer_t *()->memory;

  int m = trans_o ? (trans_b ? b3 : b4) : (trans_a ? a4 : a3);
  int n = trans_o ? (trans_a ? a4 : a3) : (trans_b ? b3 : b4);
  int k = trans_a ? a3 : a4;

  VLOG(3) << "m: " << m << ", n: " << n << ", k: " << k;

  int trans_op_l = trans_o ? !trans_b : trans_a;
  int trans_op_r = trans_o ? !trans_a : trans_b;

  void *lhs = trans_o ? B : A;
  void *rhs = trans_o ? A : B;

  cnnlDataType_t cnnl_dtype;
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  int bytes = args[0].operator cinn_buffer_t *()->type.bits / CHAR_BIT;
  if (is_float && bytes == sizeof(cinn::common::float16)) {
    cnnl_dtype = CNNL_DTYPE_HALF;
  } else if (is_float && bytes == sizeof(float)) {
    cnnl_dtype = CNNL_DTYPE_FLOAT;
  } else if (is_bfloat16) {
    cnnl_dtype = CNNL_DTYPE_BFLOAT16;
  } else {
    std::stringstream ss;
    ss << "unsupported cublas data type: " << static_cast<int>(type_code)
       << ", bytes = " << bytes;
    PADDLE_THROW(
        ::common::errors::InvalidArgument("unsupported cublas data type: ",
                                          static_cast<int>(type_code),
                                          ", bytes = ",
                                          bytes));
  }

  if (a1 * a2 * b1 * b2 == 1) {
    VLOG(3) << "call cnnlMatmul for a1 * a2 * b1 * b2 == 1";
    cinn::utils::RecordEvent record_run("Call cnnlMatmul",
                                        cinn::utils::EventType::kInstruction);
    cnnlTensorDescriptor_t desc_A, desc_B, desc_C;
    int dim_A[2] = {a3, a4}, dim_B[2] = {b3, b4}, dim_C[2] = {m, n};
    CNNL_CALL(cnnlCreateTensorDescriptor(&desc_A));
    CNNL_CALL(cnnlCreateTensorDescriptor(&desc_B));
    CNNL_CALL(cnnlCreateTensorDescriptor(&desc_C));
    CNNL_CALL(cnnlSetTensorDescriptor(
        desc_A, CNNL_LAYOUT_NCHW, cnnl_dtype, 2, dim_A));
    CNNL_CALL(cnnlSetTensorDescriptor(
        desc_B, CNNL_LAYOUT_NCHW, cnnl_dtype, 2, dim_B));
    CNNL_CALL(cnnlSetTensorDescriptor(
        desc_C, CNNL_LAYOUT_NCHW, cnnl_dtype, 2, dim_C));

    cnnlTensorDescriptor_t desc_lhs = trans_o ? desc_B : desc_A;
    cnnlTensorDescriptor_t desc_rhs = trans_o ? desc_A : desc_B;

    cnnlMatMulDescriptor_t matmul_desc;
    CNNL_CALL(cnnlMatMulDescCreate(&matmul_desc));
    CNNL_CALL(cnnlSetMatMulDescAttr(
        matmul_desc, CNNL_MATMUL_DESC_TRANSA, &trans_op_l, sizeof(trans_op_l)));
    CNNL_CALL(cnnlSetMatMulDescAttr(
        matmul_desc, CNNL_MATMUL_DESC_TRANSB, &trans_op_r, sizeof(trans_op_r)));

    size_t workspace_size = 0;
    void *workspace = nullptr;
    cnnlMatMulAlgo_t algo;
    CNNL_CALL(cnnlMatMulAlgoCreate(&algo));
    cnnlMatMulHeuristicResult_t heuristic_result;
    CNNL_CALL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
    int requested_algo_count = 1, return_algo_count = 0;
    CNNL_CALL(cnnlGetMatMulAlgoHeuristic(handle,
                                         matmul_desc,
                                         desc_lhs,
                                         desc_rhs,
                                         desc_C,
                                         desc_C,
                                         nullptr,
                                         requested_algo_count,
                                         &heuristic_result,
                                         &return_algo_count));
    CNNL_CALL(
        cnnlGetMatMulHeuristicResult(heuristic_result, algo, &workspace_size));
    if (workspace_size > 0) {
      CNRT_CALL(
          cnrtMalloc(reinterpret_cast<void **>(&workspace), workspace_size));
    }

    CNNL_CALL(cnnlMatMul_v2(handle,
                            matmul_desc,
                            algo,
                            &alpha,
                            desc_lhs,
                            lhs,
                            desc_rhs,
                            rhs,
                            &beta,
                            desc_C,
                            C,
                            workspace,
                            workspace_size,
                            desc_C,
                            C));
    CNRT_CALL(cnrtQueueSync(queue));

    if (workspace != nullptr) {
      CNRT_CALL(cnrtFree(workspace));
    }
    CNNL_CALL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
    CNNL_CALL(cnnlMatMulAlgoDestroy(algo));
    CNNL_CALL(cnnlMatMulDescDestroy(matmul_desc));
    CNNL_CALL(cnnlDestroyTensorDescriptor(desc_A));
    CNNL_CALL(cnnlDestroyTensorDescriptor(desc_B));
    CNNL_CALL(cnnlDestroyTensorDescriptor(desc_C));
  } else {
    cinn::utils::RecordEvent record_run("Call cnnlBatchMatMulBCast",
                                        cinn::utils::EventType::kInstruction);
    cnnlTensorDescriptor_t desc_A, desc_B, desc_C;
    int dim_A[4] = {a1, a2, a3, a4}, dim_B[4] = {b1, b2, b3, b4},
        dim_C[4] = {std::max(a1, b1), std::max(a2, b2), m, n};
    CNNL_CALL(cnnlCreateTensorDescriptor(&desc_A));
    CNNL_CALL(cnnlCreateTensorDescriptor(&desc_B));
    CNNL_CALL(cnnlCreateTensorDescriptor(&desc_C));
    CNNL_CALL(cnnlSetTensorDescriptor(
        desc_A, CNNL_LAYOUT_NCHW, cnnl_dtype, 4, dim_A));
    CNNL_CALL(cnnlSetTensorDescriptor(
        desc_B, CNNL_LAYOUT_NCHW, cnnl_dtype, 4, dim_B));
    CNNL_CALL(cnnlSetTensorDescriptor(
        desc_C, CNNL_LAYOUT_NCHW, cnnl_dtype, 4, dim_C));

    cnnlTensorDescriptor_t desc_lhs = trans_o ? desc_B : desc_A;
    cnnlTensorDescriptor_t desc_rhs = trans_o ? desc_A : desc_B;

    cnnlMatMulDescriptor_t bmm_bcast_desc;
    CNNL_CALL(cnnlMatMulDescCreate(&bmm_bcast_desc));
    CNNL_CALL(cnnlSetMatMulDescAttr(bmm_bcast_desc,
                                    CNNL_MATMUL_DESC_TRANSA,
                                    &trans_op_l,
                                    sizeof(trans_op_l)));
    CNNL_CALL(cnnlSetMatMulDescAttr(bmm_bcast_desc,
                                    CNNL_MATMUL_DESC_TRANSB,
                                    &trans_op_r,
                                    sizeof(trans_op_r)));

    size_t workspace_size = 0;
    void *workspace = nullptr;
    cnnlMatMulAlgo_t algo;
    CNNL_CALL(cnnlMatMulAlgoCreate(&algo));
    cnnlMatMulHeuristicResult_t heuristic_result;
    CNNL_CALL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
    int requested_algo_count = 1, return_algo_count = 0;
    CNNL_CALL(cnnlGetBatchMatMulAlgoHeuristic(handle,
                                              bmm_bcast_desc,
                                              desc_lhs,
                                              desc_rhs,
                                              desc_C,
                                              nullptr,
                                              requested_algo_count,
                                              &heuristic_result,
                                              &return_algo_count));
    CNNL_CALL(cnnlGetBatchMatMulHeuristicResult(
        heuristic_result, algo, &workspace_size));
    if (workspace_size > 0) {
      CNRT_CALL(
          cnrtMalloc(reinterpret_cast<void **>(&workspace), workspace_size));
    }

    CNNL_CALL(cnnlBatchMatMulBCast_v2(handle,
                                      bmm_bcast_desc,
                                      algo,
                                      &alpha,
                                      desc_lhs,
                                      lhs,
                                      desc_rhs,
                                      rhs,
                                      &beta,
                                      desc_C,
                                      C,
                                      workspace,
                                      workspace_size));
    CNRT_CALL(cnrtQueueSync(queue));

    if (workspace != nullptr) {
      CNRT_CALL(cnrtFree(workspace));
    }
    CNNL_CALL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
    CNNL_CALL(cnnlMatMulAlgoDestroy(algo));
    CNNL_CALL(cnnlMatMulDescDestroy(bmm_bcast_desc));
    CNNL_CALL(cnnlDestroyTensorDescriptor(desc_A));
    CNNL_CALL(cnnlDestroyTensorDescriptor(desc_B));
    CNNL_CALL(cnnlDestroyTensorDescriptor(desc_C));
  }
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_conv2d_forward(void *v_args,
                                   int num_args,
                                   int format,
                                   float alpha,
                                   float beta,
                                   int input_n,
                                   int input_c,
                                   int input_h,
                                   int input_w,
                                   int filter_n,
                                   int filter_c,
                                   int filter_h,
                                   int filter_w,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w,
                                   int dilation_h,
                                   int dilation_w,
                                   int groups,
                                   int output_n,
                                   int output_c,
                                   int output_h,
                                   int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();
  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_w = args[1].operator cinn_buffer_t *()->memory;
  void *_y = args[2].operator cinn_buffer_t *()->memory;
  int pad[4] = {pad_h, pad_h, pad_w, pad_w};
  int stride[4] = {stride_h, stride_h, stride_w, stride_w};
  int dilation[4] = {dilation_h, dilation_h, dilation_w, dilation_w};

  cnnlTensorLayout_t tensor_format = static_cast<cnnlTensorLayout_t>(format);
  cnnlDataType_t data_type = convert_to_cnnl_dtype(v_args, num_args);

  std::string hash_key =
      "conv2d forward, layout=" + debug_cnnl_tensor_format(tensor_format) +
      ", dtype=" + debug_cnnl_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";
  VLOG(4) << hash_key;

  cnnlTensorDescriptor_t x_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&x_desc));
  int dim_x[4] = {input_n, input_c, input_h, input_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(x_desc, tensor_format, data_type, 4, dim_x));

  cnnlTensorDescriptor_t w_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&w_desc));
  int dim_w[4] = {filter_n, filter_c, filter_h, filter_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(w_desc, tensor_format, data_type, 4, dim_w));

  cnnlTensorDescriptor_t y_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&y_desc));
  int dim_y[4] = {output_n, output_c, output_h, output_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(y_desc, tensor_format, data_type, 4, dim_y));

  cnnlConvolutionDescriptor_t conv_desc;
  CNNL_CALL(cnnlCreateConvolutionDescriptor(&conv_desc));
  CNNL_CALL(cnnlSetConvolutionDescriptor(
      conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT));

  cnnlConvolutionForwardAlgo_t algo;
  CNNL_CALL(cnnlGetConvolutionForwardAlgorithm(handle,
                                               conv_desc,
                                               x_desc,
                                               w_desc,
                                               y_desc,
                                               CNNL_CONVOLUTION_FWD_FASTEST,
                                               &algo));
  void *workspace = nullptr;
  size_t workspace_size = 0;
  CNNL_CALL(cnnlGetConvolutionForwardWorkspaceSize(handle,
                                                   x_desc,
                                                   w_desc,
                                                   y_desc,
                                                   nullptr,
                                                   conv_desc,
                                                   algo,
                                                   &workspace_size));
  if (workspace_size > 0) {
    CNRT_CALL(
        cnrtMalloc(reinterpret_cast<void **>(&workspace), workspace_size));
  }

  CNNL_CALL(cnnlConvolutionForward(handle,
                                   conv_desc,
                                   algo,
                                   &alpha,
                                   x_desc,
                                   _x,
                                   w_desc,
                                   _w,
                                   nullptr,
                                   nullptr,
                                   workspace,
                                   workspace_size,
                                   &beta,
                                   y_desc,
                                   _y));
  CNRT_CALL(cnrtQueueSync(queue));

  if (workspace != nullptr) {
    CNRT_CALL(cnrtFree(workspace));
  }
  CNNL_CALL(cnnlDestroyConvolutionDescriptor(conv_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(x_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(w_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(y_desc));
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_conv2d_backward_data(void *v_args,
                                         int num_args,
                                         int format,
                                         float alpha,
                                         float beta,
                                         int input_n,
                                         int input_c,
                                         int input_h,
                                         int input_w,
                                         int filter_n,
                                         int filter_c,
                                         int filter_h,
                                         int filter_w,
                                         int pad_h,
                                         int pad_w,
                                         int stride_h,
                                         int stride_w,
                                         int dilation_h,
                                         int dilation_w,
                                         int groups,
                                         int output_n,
                                         int output_c,
                                         int output_h,
                                         int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();
  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_w = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dx = args[2].operator cinn_buffer_t *()->memory;
  int pad[4] = {pad_h, pad_h, pad_w, pad_w};
  int stride[4] = {stride_h, stride_h, stride_w, stride_w};
  int dilation[4] = {dilation_h, dilation_h, dilation_w, dilation_w};

  cnnlTensorLayout_t tensor_format = static_cast<cnnlTensorLayout_t>(format);
  cnnlDataType_t data_type = convert_to_cnnl_dtype(v_args, num_args);

  std::string hash_key =
      "conv2d backward data, layout=" +
      debug_cnnl_tensor_format(tensor_format) +
      ", dtype=" + debug_cnnl_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  cnnlTensorDescriptor_t x_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&x_desc));
  int dim_x[4] = {input_n, input_c, input_h, input_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(x_desc, tensor_format, data_type, 4, dim_x));

  cnnlTensorDescriptor_t w_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&w_desc));
  int dim_w[4] = {filter_n, filter_c, filter_h, filter_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(w_desc, tensor_format, data_type, 4, dim_w));

  cnnlTensorDescriptor_t y_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&y_desc));
  int dim_y[4] = {output_n, output_c, output_h, output_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(y_desc, tensor_format, data_type, 4, dim_y));

  cnnlConvolutionDescriptor_t conv_desc;
  CNNL_CALL(cnnlCreateConvolutionDescriptor(&conv_desc));
  CNNL_CALL(cnnlSetConvolutionDescriptor(
      conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT));

  cnnlConvolutionBwdDataAlgo_t algo;
  CNNL_CALL(
      cnnlGetConvolutionBackwardDataAlgorithm(handle,
                                              w_desc,
                                              y_desc,
                                              conv_desc,
                                              x_desc,
                                              CNNL_CONVOLUTION_BWD_DATA_FASTEST,
                                              &algo));
  void *workspace = nullptr;
  size_t workspace_size = 0;
  CNNL_CALL(cnnlGetConvolutionBackwardDataWorkspaceSize(
      handle, w_desc, y_desc, conv_desc, x_desc, algo, &workspace_size));
  if (workspace_size > 0) {
    CNRT_CALL(
        cnrtMalloc(reinterpret_cast<void **>(&workspace), workspace_size));
  }

  CNNL_CALL(cnnlConvolutionBackwardData(handle,
                                        &alpha,
                                        w_desc,
                                        _w,
                                        y_desc,
                                        _dy,
                                        conv_desc,
                                        algo,
                                        workspace,
                                        workspace_size,
                                        &beta,
                                        x_desc,
                                        _dx));
  CNRT_CALL(cnrtQueueSync(queue));

  if (workspace != nullptr) {
    CNRT_CALL(cnrtFree(workspace));
  }
  CNNL_CALL(cnnlDestroyConvolutionDescriptor(conv_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(x_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(w_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(y_desc));
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_conv2d_backward_filter(void *v_args,
                                           int num_args,
                                           int format,
                                           float alpha,
                                           float beta,
                                           int input_n,
                                           int input_c,
                                           int input_h,
                                           int input_w,
                                           int filter_n,
                                           int filter_c,
                                           int filter_h,
                                           int filter_w,
                                           int pad_h,
                                           int pad_w,
                                           int stride_h,
                                           int stride_w,
                                           int dilation_h,
                                           int dilation_w,
                                           int groups,
                                           int output_n,
                                           int output_c,
                                           int output_h,
                                           int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 3, but received %d.", num_args));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();
  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dw = args[2].operator cinn_buffer_t *()->memory;
  int pad[4] = {pad_h, pad_h, pad_w, pad_w};
  int stride[4] = {stride_h, stride_h, stride_w, stride_w};
  int dilation[4] = {dilation_h, dilation_h, dilation_w, dilation_w};

  cnnlTensorLayout_t tensor_format = static_cast<cnnlTensorLayout_t>(format);
  cnnlDataType_t data_type = convert_to_cnnl_dtype(v_args, num_args);

  std::string hash_key =
      "conv2d backward filter, layout=" +
      debug_cnnl_tensor_format(tensor_format) +
      ", dtype=" + debug_cnnl_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  cnnlTensorDescriptor_t x_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&x_desc));
  int dim_x[4] = {input_n, input_c, input_h, input_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(x_desc, tensor_format, data_type, 4, dim_x));

  cnnlTensorDescriptor_t w_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&w_desc));
  int dim_w[4] = {filter_n, filter_c, filter_h, filter_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(w_desc, tensor_format, data_type, 4, dim_w));

  cnnlTensorDescriptor_t y_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&y_desc));
  int dim_y[4] = {output_n, output_c, output_h, output_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(y_desc, tensor_format, data_type, 4, dim_y));

  cnnlConvolutionDescriptor_t conv_desc;
  CNNL_CALL(cnnlCreateConvolutionDescriptor(&conv_desc));
  CNNL_CALL(cnnlSetConvolutionDescriptor(
      conv_desc, 4, pad, stride, dilation, groups, CNNL_DTYPE_FLOAT));

  cnnlConvolutionBwdFilterAlgo_t algo;
  CNNL_CALL(cnnlGetConvolutionBackwardFilterAlgorithm(
      handle,
      conv_desc,
      x_desc,
      y_desc,
      w_desc,
      CNNL_CONVOLUTION_BWD_FILTER_FASTEST,
      &algo));
  void *workspace = nullptr;
  size_t workspace_size = 0;
  CNNL_CALL(cnnlGetConvolutionBackwardFilterWorkspaceSize(
      handle, x_desc, y_desc, w_desc, conv_desc, algo, &workspace_size));
  if (workspace_size > 0) {
    CNRT_CALL(
        cnrtMalloc(reinterpret_cast<void **>(&workspace), workspace_size));
  }

  CNNL_CALL(cnnlConvolutionBackwardFilter(handle,
                                          nullptr,
                                          x_desc,
                                          _x,
                                          y_desc,
                                          _dy,
                                          conv_desc,
                                          algo,
                                          workspace,
                                          workspace_size,
                                          nullptr,
                                          w_desc,
                                          _dw));
  CNRT_CALL(cnrtQueueSync(queue));

  if (workspace != nullptr) {
    CNRT_CALL(cnrtFree(workspace));
  }
  CNNL_CALL(cnnlDestroyConvolutionDescriptor(conv_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(x_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(w_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(y_desc));
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_pool2d_forward(void *v_args,
                                   int num_args,
                                   int mode,
                                   int format,
                                   float alpha,
                                   float beta,
                                   int input_n,
                                   int input_c,
                                   int input_h,
                                   int input_w,
                                   int kernel_h,
                                   int kernel_w,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w,
                                   int output_n,
                                   int output_c,
                                   int output_h,
                                   int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 2, but received %d.", num_args));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();
  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;

  cnnlPoolingMode_t pool_mode = static_cast<cnnlPoolingMode_t>(mode);
  cnnlTensorLayout_t tensor_format = static_cast<cnnlTensorLayout_t>(format);
  cnnlDataType_t data_type = convert_to_cnnl_dtype(v_args, num_args);

  std::string hash_key =
      "pool2d forward, layout=" + debug_cnnl_tensor_format(tensor_format) +
      ", pool_type=" + debug_cnnl_pool_mode(pool_mode) +
      ", dtype=" + debug_cnnl_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, kernel_hw={" + std::to_string(kernel_h) + "," +
      std::to_string(kernel_w) + "}, pad_hw={" + std::to_string(pad_h) + "," +
      std::to_string(pad_w) + "}, stride_hw={" + std::to_string(stride_h) +
      "," + std::to_string(stride_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  cnnlPoolingDescriptor_t pool_desc;
  CNNL_CALL(cnnlCreatePoolingDescriptor(&pool_desc));
  CNNL_CALL(cnnlSetPooling2dDescriptor_v2(pool_desc,
                                          pool_mode,
                                          CNNL_NOT_PROPAGATE_NAN,
                                          kernel_h,
                                          kernel_w,
                                          pad_h,
                                          pad_h,
                                          pad_w,
                                          pad_w,
                                          stride_h,
                                          stride_w,
                                          1,
                                          1,
                                          true));

  cnnlTensorDescriptor_t x_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&x_desc));
  int dim_x[4] = {input_n, input_c, input_h, input_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(x_desc, tensor_format, data_type, 4, dim_x));
  cnnlTensorDescriptor_t y_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&y_desc));
  int dim_y[4] = {output_n, output_c, output_h, output_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(y_desc, tensor_format, data_type, 4, dim_y));

  size_t workspace_size = 0;
  void *workspace = nullptr;
  CNNL_CALL(cnnlGetPoolingWorkspaceSize(
      handle, pool_mode, output_w, output_h, &workspace_size));
  if (workspace_size > 0) {
    CNRT_CALL(
        cnrtMalloc((reinterpret_cast<void **>(&workspace), workspace_size)));
  }
  size_t extra_input_size = 0;
  void *extra_host_input = nullptr, *extra_device_input = nullptr;
  CNNL_CALL(cnnlGetPoolingExtraInputSize(
      handle, pool_mode, output_w, output_h, &extra_input_size));
  if (extra_input_size > 0) {
    extra_host_input = std::malloc(extra_input_size);
    CNRT_CALL(cnrtMalloc(reinterpret_cast<void **>(&extra_device_input),
                         extra_input_size));
    CNNL_CALL(cnnlInitPoolingExtraInput(
        handle, pool_desc, x_desc, y_desc, extra_host_input));
    CNRT_CALL(cnrtMemcpy(extra_device_input,
                         extra_host_input,
                         extra_input_size,
                         cnrtMemcpyHostToDev));
  }

  CNNL_CALL(cnnlPoolingForward_v2(handle,
                                  pool_desc,
                                  &alpha,
                                  x_desc,
                                  _x,
                                  &beta,
                                  extra_device_input,
                                  y_desc,
                                  _y,
                                  workspace,
                                  workspace_size));
  CNRT_CALL(cnrtQueueSync(queue));

  if (extra_host_input != nullptr) {
    std::free(extra_host_input);
  }
  if (extra_device_input != nullptr) {
    CNRT_CALL(cnrtFree(extra_device_input));
  }
  if (workspace != nullptr) {
    CNRT_CALL(cnrtFree(workspace));
  }
  CNNL_CALL(cnnlDestroyTensorDescriptor(x_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(y_desc));
  CNNL_CALL(cnnlDestroyPoolingDescriptor(pool_desc));
  CNRT_CALL(cnrtQueueDestroy(queue));
}

void cinn_call_cnnl_pool2d_backward(void *v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      4,
      ::common::errors::InvalidArgument(
          "Expected number of argruments is 4, but received %d.", num_args));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();
  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;
  void *_dy = args[2].operator cinn_buffer_t *()->memory;
  void *_dx = args[3].operator cinn_buffer_t *()->memory;

  cnnlPoolingMode_t pool_mode = static_cast<cnnlPoolingMode_t>(mode);
  cnnlTensorLayout_t tensor_format = static_cast<cnnlTensorLayout_t>(format);
  cnnlDataType_t data_type = convert_to_cnnl_dtype(v_args, num_args);

  std::string hash_key =
      "pool2d backward, layout=" + debug_cnnl_tensor_format(tensor_format) +
      ", pool_type=" + debug_cnnl_pool_mode(pool_mode) +
      ", dtype=" + debug_cnnl_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, kernel_hw={" + std::to_string(kernel_h) + "," +
      std::to_string(kernel_w) + "}, pad_hw={" + std::to_string(pad_h) + "," +
      std::to_string(pad_w) + "}, stride_hw={" + std::to_string(stride_h) +
      "," + std::to_string(stride_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  cnnlPoolingDescriptor_t pool_desc;
  CNNL_CALL(cnnlCreatePoolingDescriptor(&pool_desc));
  CNNL_CALL(cnnlSetPooling2dDescriptor_v2(pool_desc,
                                          pool_mode,
                                          CNNL_NOT_PROPAGATE_NAN,
                                          kernel_h,
                                          kernel_w,
                                          pad_h,
                                          pad_h,
                                          pad_w,
                                          pad_w,
                                          stride_h,
                                          stride_w,
                                          1,
                                          1,
                                          true));

  cnnlTensorDescriptor_t x_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&x_desc));
  int dim_x[4] = {input_n, input_c, input_h, input_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(x_desc, tensor_format, data_type, 4, dim_x));
  cnnlTensorDescriptor_t y_desc;
  CNNL_CALL(cnnlCreateTensorDescriptor(&y_desc));
  int dim_y[4] = {output_n, output_c, output_h, output_w};
  CNNL_CALL(
      cnnlSetTensorDescriptor(y_desc, tensor_format, data_type, 4, dim_y));

  CNNL_CALL(cnnlPoolingBackward(handle,
                                pool_desc,
                                &alpha,
                                nullptr,
                                nullptr,
                                y_desc,
                                _dy,
                                x_desc,
                                _x,
                                &beta,
                                x_desc,
                                _dx));
  CNRT_CALL(cnrtQueueSync(queue));

  CNNL_CALL(cnnlDestroyTensorDescriptor(x_desc));
  CNNL_CALL(cnnlDestroyTensorDescriptor(y_desc));
  CNNL_CALL(cnnlDestroyPoolingDescriptor(pool_desc));
  CNRT_CALL(cnrtQueueDestroy(queue));
}

#endif  // CINN_WITH_CNNL

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
