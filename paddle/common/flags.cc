// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/common/flags.h"

namespace phi {

const ExportedFlagInfoMap &GetExportedFlagInfoMap() {
  return *GetMutableExportedFlagInfoMap();
}

ExportedFlagInfoMap *GetMutableExportedFlagInfoMap() {
  static ExportedFlagInfoMap g_exported_flag_info_map;
  return &g_exported_flag_info_map;
}

}  // namespace phi

PHI_DEFINE_EXPORTED_int32(inner_op_parallelism,
                          0,
                          "number of threads for inner op");

/**
 * NOTE(paddle-dev): This file is designed to define all public FLAGS.
 */

/**
 * Paddle initialization related FLAG
 * Name: FLAGS_paddle_num_threads
 * Since Version: 0.15.0
 * Value Range: int32, default=1
 * Example: FLAGS_paddle_num_threads=2, set the maximum thread number per
 * instance to 2
 * Note:
 */

PHI_DEFINE_EXPORTED_int32(paddle_num_threads,
                          1,
                          "Number of threads for each paddle instance.");

/**
 * Low Precision Op related FLAG
 * Name: FLAGS_low_precision_op_list
 * Since Version: 2.5.0
 * Value Range: int32, default=0
 * Example:
 * Note: Used to debug. Get the low precision op list of current module.
 * FLAGS_check_nan_inf is set.
 * - 1, return the low precision op list of current module.
 * - 2, return the op list of current module.
 */
PHI_DEFINE_EXPORTED_int32(low_precision_op_list,
                          0,
                          "Setting the level of low precision op"
                          "list printing. It will be return the "
                          "low precision op list of current module.");

/**
 * Operator related FLAG
 * Name: FLAGS_check_nan_inf
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: Used to debug. Checking whether operator produce NAN/INF or not.
 */
PHI_DEFINE_EXPORTED_bool(
    check_nan_inf,
    false,
    "Checking whether operator produce NAN/INF or not. It will be "
    "extremely slow so please use this flag wisely.");

/**
 * Operator related FLAG
 * Name: FLAGS_check_nan_inf_level
 * Since Version: 2.5.0
 * Value Range: int32, default=0
 * Example:
 * Note: Used to debug. Setting the check and print level when
 * FLAGS_check_nan_inf is set.
 * - 0, abort the process when any operator produce NAN/INF and only print the
 * information of tensor which holds NAN/INF.
 * - 1, continue the training or inference process and print the information of
 * all tensors which holds NAN/INF.
 * - 2, print the information of float tensors when the max or min value
 * overflowing float16's limit.
 * - 3, print the information of all tensors.
 */
PHI_DEFINE_EXPORTED_int32(
    check_nan_inf_level,
    0,
    "Setting the check and print level when FLAGS_check_nan_inf is set.");

/**
 * Operator related FLAG
 * Name: FLAGS_check_nan_inf
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: Used to debug. Checking whether operator produce NAN/INF or not.
 */
PHI_DEFINE_EXPORTED_bool(
    enable_opt_get_features,
    false,
    "Checking whether operator produce NAN/INF or not. It will be "
    "extremely slow so please use this flag wisely.");

// NOTE(zhiqiu): better to share the flags, otherwise we will have too many
// flags.
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

/**
 * CUDA related related FLAG
 * Name: FLAGS_enable_cublas_tensor_op_math
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use Tensor Core, faster but it may loss precision.
 */
PHI_DEFINE_EXPORTED_bool(
    enable_cublas_tensor_op_math,
    false,
    "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
    "but it may loss precision. Currently, There are two CUDA libraries that"
    " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
    " GEMM computations(the matrices must be either half precision or single "
    "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
    "input and output must be half precision) and recurrent neural networks "
    "(RNNs).");

/**
 * CUDA related related FLAG
 * Name: FLAGS_gemm_use_half_precision_compute_type
 * Since Version: 2.4
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use fp16 compute type when the input and output is fp16,
 * faster but it may loss precision.
 */
PHI_DEFINE_EXPORTED_bool(
    gemm_use_half_precision_compute_type,
    false,
    "Whether to use fp16 compute type when the input and output is fp16, "
    "faster but it may loss precision in most case. If true, the compute "
    "type will be set to fp16. Default is false.");

/**
 * CUDA related FLAG
 * Name: FLAGS_selected_gpus
 * Since Version: 1.3.0
 * Value Range: integer list separated by comma, default empty list
 * Example: FLAGS_selected_gpus=0,1,2,3,4,5,6,7 to train or predict with 0~7 gpu
 * cards
 * Note: A list of device ids separated by comma, like: 0,1,2,3
 */
PHI_DEFINE_EXPORTED_string(
    selected_gpus,
    "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (GPU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between GPU devices, use CUDA_VISIBLE_DEVICES can only use"
    "share-memory only.");
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/**
 * CUDA related FLAG
 * Name: FLAGS_cublaslt_exhaustive_search_times
 * Since Version: 2.3.0
 * Value Range: int64_t, default=0
 * Example:
 * Note: Represents times of exhaustive search to evaluate performance of
 *       cuBlasLt matmul algorithm (with/without epilogue). Set this flag
 *       with value > 0 to enable exhaustive search. Default is 0, means
 *       getting algorithms via heuristic search. There are two search methods
 *       in cuBlasLt, heuristic search and exhaustive search. Exhaustive search
 *       attempts all cuBlasLt algorithms to select the fastest, which is very
 *       time-consuming, and the selected algorithm will be cached for a given
 *       layer specification Once you change the layer specifications
 *       (such as M, N and K), it will re-search again.
 */
PHI_DEFINE_EXPORTED_int64(
    cublaslt_exhaustive_search_times,
    0,
    "The times of exhaustive search for cuBlasLt matmul with/without "
    " epilogue algorithms, default is 0, means disabling exhaustive search.");
#endif

/*
 * Kernel related FLAG
 * Name: FLAGS_enable_api_kernel_fallback
 * Since Version: 2.4
 * Value Range: bool, default=true
 * Example: FLAGS_enable_api_kernel_fallback=true would allow kernel of current
 * backend fallback to CPU one when not found
 */
PHI_DEFINE_EXPORTED_bool(
    enable_api_kernel_fallback,
    true,
    "Whether enable api kernel fallback to CPU one when not found");

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_deterministic
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use deterministic algorithm in cudnn.
 *       If true, it will slow down some operators such as conv and pooling.
 */
PHI_DEFINE_EXPORTED_bool(
    cudnn_deterministic,
    false,
    "Whether allow using an autotuning algorithm for convolution "
    "operator. The autotuning algorithm may be non-deterministic. If "
    "true, the algorithm is deterministic.");

/**
 * CUDA related FLAG
 * Name: FLAGS_embedding_deterministic
 * Since Version: 2.5
 * Value Range: int64, default=0
 * Example:
 * Note: whether to use deterministic algorithm in embedding op.
 *       If it is 1, it will use the optimized deterministic CUDA kernel in
 *       embedding op. If it is 2, it will use the legacy deterministic
 *       CUDA kernel in embedding op.
 */
PHI_DEFINE_EXPORTED_int64(
    embedding_deterministic,
    0,
    "Whether allow using an deterministic algorithm for embedding "
    "operator. The deterministic algorithm may be slower. If "
    "it is larger than 0, the algorithm is deterministic.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_exhaustive_search
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Represents whether an exhaustive search method is used to
 *       select a convolution algorithm. There are two search methods in cuDNN,
 *       heuristic search and exhaustive search. Exhaustive search attempts
 *       all cuDNN algorithms to select the fastest. This method is very
 *       time-consuming, and the selected algorithm will be cached for a given
 *       layer specification. Once you change the layer specifications
 *       (such as batch size, feature map size), it will search again.
 */
PHI_DEFINE_EXPORTED_bool(
    cudnn_exhaustive_search,
    false,
    "Whether enable exhaustive search for cuDNN convolution or "
    "not, default is False.");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_exhaustive_search_times
 * Since Version:
 * Value Range:
 * Example:
 * Note: only used to predict for advanced developer
 */
PHI_DEFINE_EXPORTED_int64(cudnn_exhaustive_search_times,
                          -1,
                          "Exhaustive search times for cuDNN convolution, "
                          "default is -1, not exhaustive search");

#ifdef PADDLE_WITH_HIP
/**
 * MIOPEN related FLAG
 * Name: FLAGS_batch_norm_use_miopen
 * Since Version:
 * Value Range:
 * Example:
 * Note: Use MIOpen batch norm instead of native
 */
PHI_DEFINE_EXPORTED_bool(batch_norm_use_miopen,
                         false,
                         "Whether use MIOpen batch norm or not, "
                         "default is false, not use miopen bn");
#endif

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_batchnorm_spatial_persistent
 * Since Version: 1.4.0
 * Value Range: bool, default=false
 * Example:
 * Note: CUDNN_BATCHNORM_SPATIAL_PERSISTENT in batchnorm. This mode can be
 * faster in
 *       some tasks because an optimized path may be selected for
 * CUDNN_DATA_FLOAT
 *       and CUDNN_DATA_HALF data types, compute capability 6.0 or higher. The
 *       reason we set it to false by default is that this mode may use scaled
 *       atomic integer reduction that may cause a numerical overflow for
 * certain
 *       input data range.
 */
PHI_DEFINE_EXPORTED_bool(
    cudnn_batchnorm_spatial_persistent,
    false,
    "Whether enable CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode for cudnn "
    "batch_norm, default is False.");
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

/**
 * NCCL related FLAG
 * Name: FLAGS_sync_nccl_allreduce
 * Since Version: 1.3
 * Value Range: bool, default=true
 * Example:
 * Note: asynchronous nccl allreduce or synchronous issue:
 *       https://github.com/PaddlePaddle/Paddle/issues/15049
 *       If you want to change this default value, why?(gongwb)
 */
PHI_DEFINE_EXPORTED_bool(
    sync_nccl_allreduce,
    true,
    "If set true, will call `cudaStreamSynchronize(nccl_stream)`"
    "after allreduce, this mode can get better performance in some scenarios.");
#endif

#ifdef PADDLE_WITH_DISTRIBUTE
/**
 * Distributed related FLAG
 * Name: FLAGS_communicator_max_merge_var_num
 * Since Version: 1.5.0
 * Value Range: int32, default=20
 * Example:
 * Note: The maximum number of gradients to be merged into a gradient and
 *       sent through the communicator. The trainer puts all the gradients
 *       into the queue, and then the communicator takes the gradients out
 *       of the queue and sends them after merging.
 */
PHI_DEFINE_EXPORTED_int32(communicator_max_merge_var_num,
                          20,
                          "max var num to merge and send");
PHI_DEFINE_EXPORTED_bool(
    communicator_is_sgd_optimizer,
    true,
    "gradient sent to the server is the sum of the gradients "
    "calculated by each thread if optimizer is sgd");
/**
 * Distributed related FLAG
 * Name: FLAGS_communicator_send_queue_size
 * Since Version: 1.5.0
 * Value Range: int32, default=20
 * Example:
 * Note: Size for each gradient queue. The trainer puts the gradient into
 *       the queue, and then the communicator takes it out of the queue and
 *       sends it out. When the communicator is slow, the queue may be full,
 *       and the trainer will be continuously blocked before the queue has
 *       space. It is used to avoid training much faster than communication,
 *       so that too many gradients are not sent out in time.
 */
PHI_DEFINE_EXPORTED_int32(communicator_send_queue_size,
                          20,
                          "queue size to recv gradient before send");
#endif

/**
 * Distributed related FLAG
 * Name: FLAGS_dist_threadpool_size
 * Since Version: 1.0.0
 * Value Range: int32, default=0
 * Example:
 * Note: Control the number of threads used for distributed modules.
 *       If it is not set, it is set to a hard thread.
 */
PHI_DEFINE_EXPORTED_int32(dist_threadpool_size,
                          0,
                          "number of threads used for distributed executed.");

/**
 * Garbage collector related FLAG
 * Name: FLAGS_eager_delete_tensor_gb
 * Since Version: 1.0.0
 * Value Range: double, default=kDefaultEagerDeleteTensorGB
 * Example: FLAGS_eager_delete_tensor_gb=0.0, Release memory garbage once it is
 * no longer used.
 *          FLAGS_eager_delete_tensor_gb=1.0, Release memory garbage when
 * garbage occupies 1.0GB of memory.
 *          FLAGS_eager_delete_tensor_gb=-1.0, Disable garbage collection
 * policy.
 * Note: Represents whether a garbage collection strategy is used to optimize
 * network memory usage.
 *       It is recommended that users set FLAGS_eager_delete_tensor_gb=0.0 to
 *       enable garbage collection strategy when training large networks.
 */
// Disable gc by default when inference library is built
static const double kDefaultEagerDeleteTensorGB = 0;

PHI_DEFINE_EXPORTED_double(
    eager_delete_tensor_gb,
    kDefaultEagerDeleteTensorGB,
    "Memory size threshold (GB) when the garbage collector clear tensors."
    "Disabled when this value is less than 0");

/**
 * Memory related FLAG
 * Name: FLAGS_fast_eager_deletion_mode
 * Since Version: 1.3.0
 * Value Range: bool, default=true
 * Example:
 * Note: Whether to use fast garbage collection strategy.
 *       If not set, the GPU memory is released at the end of the CUDA kernel.
 *       Otherwise, the GPU memory will be released before the CUDA kernel
 *       has finished, which will make the garbage collection strategy faster.
 *       Only works when garbage collection strategy is enabled.
 */
PHI_DEFINE_EXPORTED_bool(
    fast_eager_deletion_mode,
    true,
    "Fast eager deletion mode. If enabled, memory would release "
    "immediately without waiting GPU kernel ends.");

/**
 * Memory related FLAG
 * Name: FLAGS_memory_fraction_of_eager_deletion
 * Since Version: 1.4
 * Value Range: double [0.0, 1.0], default=1.0
 * Example:
 * Note: The percentage of memory size of garbage collection policy
 *       to release variables.
 *       If FLAGS_memory_fraction_of_eager_deletion = 1.0,
 *       all temporary variables in the network will be released.
 *       If FLAGS_memory_fraction_of_eager_deletion = 0.0,
 *       no temporary variables in the network are released.
 *       If 0.0 < FLAGS_memory_fraction_of_eager_deletion < 1.0,
 *       all temporary variables will be sorted in descending order
 *       according to their memory size, and only variables with the
 *       largest FLAGS_memory_fraction_of_eager_deletion ratio will be released.
 *       The flag is only valid when running parallel data compilers.
 */
PHI_DEFINE_EXPORTED_double(
    memory_fraction_of_eager_deletion,
    1.0,
    "Fraction of eager deletion. If less than 1.0, all variables in "
    "the program would be sorted according to its memory size, and "
    "only the FLAGS_memory_fraction_of_eager_deletion of the largest "
    "variables would be deleted.");

/**
 * Allocator related FLAG
 * Name: FLAGS_allocator_strategy
 * Since Version: 1.2
 * Value Range: string, {naive_best_fit, auto_growth, thread_local},
 * default=auto_growth
 * Example:
 * Note: For selecting allocator policy of PaddlePaddle.
 */
static constexpr char kDefaultAllocatorStrategy[] = "auto_growth";  // NOLINT
PHI_DEFINE_EXPORTED_string(
    allocator_strategy,
    kDefaultAllocatorStrategy,
    "The allocation strategy, enum in [naive_best_fit, auto_growth]. "
    "naive_best_fit means the original pre-allocated allocator of Paddle. "
    "auto_growth means the auto-growth allocator. "
    "These two strategies differ in GPU memory allocation. "
    "naive_best_fit strategy would occupy almost all GPU memory by default, "
    "which prevents users from starting several Paddle jobs on the same GPU "
    "card but leads to less memory fragmentation (i.e., maximum batch "
    "size of models may be larger). auto_growth strategy would allocate "
    "GPU memory on demand, which allows users to start several Paddle jobs "
    "on the same GPU card but may lead to more memory fragmentation "
    "(i.e., maximum batch size of models may be smaller).");

/**
 * Memory related FLAG
 * Name: FLAGS_fraction_of_cpu_memory_to_use
 * Since Version: 0.12.0
 * Value Range: double, [0.0, 1.0], default=1
 * Example:
 * Note: Represents the proportion of allocated CPU memory blocks
 *       to the total memory size of the CPU. Future CPU memory usage
 *       will be allocated from this memory block. If the memory block does
 *       not have enough CUDA pinned memory, new memory blocks of the same
 *       size as the memory block will be allocated from the CUDA pinned
 *       request util the CPU does not have enough memory.
 */
PHI_DEFINE_EXPORTED_double(fraction_of_cpu_memory_to_use,
                           1,
                           "Default use 100% of CPU memory for PaddlePaddle,"
                           "reserve the rest for page tables, etc");

/**
 * Memory related FLAG
 * Name: FLAGS_initial_cpu_memory_in_mb
 * Since Version: 0.14.0
 * Value Range: uint64, default=500 (MB)
 * Example:
 * Note: The CPU memory block size of the initial allocator in MB.
 *       The allocator takes the minimum values of
 *       FLAGS_initial_cpu_memory_in_mb and
 *       FLAGS_fraction_of_cpu_memory_to_use*(total physical memory)
 *       as memory block sizes.
 */
PHI_DEFINE_EXPORTED_uint64(initial_cpu_memory_in_mb,
                           500ul,
                           "Initial CPU memory for PaddlePaddle, in MD unit.");

/**
 * Memory related FLAG
 * Name: FLAGS_fraction_of_cuda_pinned_memory_to_use
 * Since Version: 0.12.0
 * Value Range: double, [0.0, 1.0], default=0.5
 * Example:
 * Note: Represents the proportion of allocated CUDA pinned memory blocks
 *       to the total memory size of the CPU. Future CUDA pinned memory usage
 *       will be allocated from this memory block. If the memory block does
 *       not have enough CPU memory, new memory blocks of the same
 *       size as the memory block will be allocated from the CPU
 *       request util the CPU does not have enough memory.
 */
PHI_DEFINE_EXPORTED_double(
    fraction_of_cuda_pinned_memory_to_use,
    0.5,
    "Default use 50% of CPU memory as the pinned_memory for PaddlePaddle,"
    "reserve the rest for page tables, etc");

// NOTE(zhiqiu): better to share the flags, otherwise we will have too many
// flags.
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE) || defined(PADDLE_WITH_XPU)

/**
 * Memory related FLAG
 * Name: FLAGS_fraction_of_gpu_memory_to_use
 * Since Version: 1.2.0
 * Value Range: double, default=0.5 if win32, 0.92 else
 * Example:
 * Note: Represents the proportion of allocated memory blocks to the total
 * memory size
 *       of the GPU. Future memory usage will be allocated from this memory
 * block.
 *       If the memory block does not have enough GPU memory, new memory blocks
 * of
 *       the same size as the memory block will be allocated from the GPU
 * request
 *       until the GPU does not have enough memory.
 */

#ifndef _WIN32
constexpr static float fraction_of_gpu_memory_to_use = 0.92f;
#else
// fraction_of_gpu_memory_to_use cannot be too high on windows,
// since the win32 graphic sub-system can occupy some GPU memory
// which may lead to insufficient memory left for paddle
constexpr static float fraction_of_gpu_memory_to_use = 0.5f;
#endif
PHI_DEFINE_EXPORTED_double(
    fraction_of_gpu_memory_to_use,
    fraction_of_gpu_memory_to_use,
    "Allocate a trunk of gpu memory that is this fraction of the "
    "total gpu memory size. Future memory usage will be allocated "
    "from the trunk. If the trunk doesn't have enough gpu memory, "
    "additional trunks of the same size will be requested from gpu "
    "until the gpu has no memory left for another trunk.");

/**
 * Memory related FLAG
 * Name: FLAGS_initial_gpu_memory_in_mb
 * Since Version: 1.4.0
 * Value Range: uint64, default=0 (MB)
 * Example:
 * Note: Allocate a specified size of GPU memory block. Later memory usage
 *       will be allocated from that memory block. If the memory block does not
 *       have enough GPU memory, the memory block with the size
 *       FLAGS_reallocate_gpu_memory_in_mb will be requested from the GPU until
 *       the GPU has no remaining memory.
 */
PHI_DEFINE_EXPORTED_uint64(
    initial_gpu_memory_in_mb,
    0ul,
    "Allocate a trunk of gpu memory whose byte size is specified by "
    "the flag. Future memory usage will be allocated from the "
    "trunk. If the trunk doesn't have enough gpu memory, additional "
    "trunks of the gpu memory will be requested from gpu with size "
    "specified by FLAGS_reallocate_gpu_memory_in_mb until the gpu has "
    "no memory left for the additional trunk. Note: if you set this "
    "flag, the memory size set by "
    "FLAGS_fraction_of_gpu_memory_to_use will be overridden by this "
    "flag. If you don't set this flag, PaddlePaddle will use "
    "FLAGS_fraction_of_gpu_memory_to_use to allocate gpu memory");

/**
 * Memory related FLAG
 * Name: FLAGS_reallocate_gpu_memory_in_mb
 * Since Version: 1.4.0
 * Value Range: uint64, default=0 (MB)
 * Example:
 * Note: If the allocated GPU memory blocks are exhausted,
 *       additional GPU memory blocks are reallocated
 */
PHI_DEFINE_EXPORTED_uint64(
    reallocate_gpu_memory_in_mb,
    0ul,
    "If this flag is set, Paddle will reallocate the gpu memory with "
    "size specified by this flag. Else Paddle will reallocate by "
    "FLAGS_fraction_of_gpu_memory_to_use");

PHI_DEFINE_EXPORTED_uint64(
    gpu_memory_limit_mb,
    0UL,
    "The maximum gpu memory limit that the process can allocate. "
    "If it is equal to 0, there would be no limit and all gpu memory "
    "would be available to the process. If it is larger than 0, "
    "the process would raise out of memory error if the allocated "
    "memory exceeds the limit even though there is available "
    "memory on the gpu card. The unit is MB and default value is 0.");

/**
 * Memory related FLAG
 * Name: FLAGS_auto_growth_chunk_size_in_mb
 * Since Version: 2.5.0
 * Value Range: uint64, default=0 (MB)
 * Example:
 * Note: The minimal chunk size of GPU memory block in auto_growth allocator.
 *       The real chunk size is max(request_size,
 *       FLAGS_auto_growth_chunk_size_in_mb).
 */
PHI_DEFINE_EXPORTED_uint64(
    auto_growth_chunk_size_in_mb,
    0ul,
    "The minimal chunk size of GPU memory block in auto_growth allocator.  "
    "The real chunk size is max(request_size, "
    "FLAGS_auto_growth_chunk_size_in_mb).");

#endif

/**
 * Scope related FLAG
 * Name: local_exe_sub_scope_limit
 * Since Version: 1.6.0
 * Value Range: double, default=256 (MB)
 * Example:
 * Note:
 */
PHI_DEFINE_EXPORTED_double(
    local_exe_sub_scope_limit,
    256.0,  // MBytes
    "The memory up limit of sub-scopes of local execution scope for "
    "each CUDAPlace. If you don't need to limit the memory, "
    "you should set FLAGS_local_exe_sub_scope_limit=-1. "
    "The default value is 256 MBytes.");

PHI_DEFINE_EXPORTED_bool(
    reader_queue_speed_test_mode,
    false,
    "If set true, the queue.pop will only get data from queue but not "
    "remove the data from queue for speed testing");

/**
 * MKLDNN related FLAG
 * Name: use_mkldnn
 * Since Version:
 * Value Range: bool, default=false
 * Example:
 * Note:
 */
PHI_DEFINE_EXPORTED_bool(use_mkldnn, false, "Use MKLDNN to run");

/**
 * Debug related FLAG
 * Name: FLAGS_call_stack_level
 * Since Version: 2.0.0
 * Value Range: int, default=2
 * Example:
 * Note: Used to debug. Determine the call stack to print when error or
 * exception happens.
 * If FLAGS_call_stack_level == 0, only the error message summary will be shown.
 * If FLAGS_call_stack_level == 1, the python stack and  error message summary
 * will be shown.
 * If FLAGS_call_stack_level == 2, the python stack, c++ stack, and error
 * message summary will be shown.
 */
#ifdef PADDLE_NO_PYTHON
static const int32_t kDefaultCallStackLevel = 2;
#else
static const int32_t kDefaultCallStackLevel = 1;
#endif

PHI_DEFINE_EXPORTED_int32(
    call_stack_level,
    kDefaultCallStackLevel,
    "Determine the call stack to print when error or exception happens."
    // TODO(zhiqiu): implement logic of FLAGS_call_stack_level==0
    // "If FLAGS_call_stack_level == 0, only the error message summary will be "
    // "shown. "
    "If FLAGS_call_stack_level == 1, the python stack and error message "
    "summary will be shown."
    "If FLAGS_call_stack_level == 2, the python stack, c++ stack, and "
    "error message summary will be shown.");

/**
 * Debug related FLAG
 * Name: sort_sum_gradient
 * Since Version: 2.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, gradients are summed by the reverse order of
 * the forward execution sequence.
 */
PHI_DEFINE_EXPORTED_bool(sort_sum_gradient,
                         false,
                         "Sum gradients by the reverse order of "
                         "the forward execution sequence.");

/**
 * Performance related FLAG
 * Name: max_inplace_grad_add
 * Since Version: 2.0.0
 * Value Range: int32, default=0
 * Example:
 * Note: The maximum number of inplace grad_add.
 */
PHI_DEFINE_EXPORTED_int32(
    max_inplace_grad_add,
    0,
    "The maximum number of inplace grad_add. When doing "
    "gradient accumulation, if the number of gradients need to that "
    "less FLAGS_max_inplace_grad_add, than it will be use several grad_add"
    "instead of sum. Default is 0.");

/**
 * Tensor.numpy() has a hack, and this flag can close this hack
 * [true]: set 0D Tensor to 1D Numpy
 * [false]: not set 0D Tensor to 1D Numpy, close the hack
 *
 * Now, just set true by default in 2.5 transition time
 * which will be removed in future (2.6) .
 */
PHI_DEFINE_EXPORTED_bool(set_to_1d, false, "set 0D Tensor to 1D numpy");

/**
 * Debug related FLAG
 * Name: tracer_onednn_ops_on
 * Since Version: 2.0.0
 * Value Range: string, default=empty
 * Example:
 * Note: Holds list of operation types with OneDNN kernels to be enabled.
 */
PHI_DEFINE_EXPORTED_string(tracer_onednn_ops_on,
                           "",
                           "List of OneDNN operation types to be turned on");

/**
 * Debug related FLAG
 * Name: static_runtime_data_save_path
 * Since Version: 2.6.0
 * Value Range: string, default=./
 * Example:
 * Note: set the static runtime tensor save path.
 */
PHI_DEFINE_EXPORTED_string(static_runtime_data_save_path,
                           "./",
                           "set the static runtime tensor save path");

/**
 * Debug related FLAG
 * Name: tracer_onednn_ops_off
 * Since Version: 2.0.0
 * Value Range: string, default=empty
 * Example:
 * Note: Holds list of operation types with OneDNN kernels to be disabled.
 */
PHI_DEFINE_EXPORTED_string(tracer_onednn_ops_off,
                           "",
                           "List of OneDNN operation types to be turned off");

/**
 * Debug related FLAG
 * Name: check_kernel_launch
 * Since Version: 2.1.0
 * Value Range: bool, default=false
 * Example:
 * Note: Check kernel launch status after every kernel compute.
 */
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PHI_DEFINE_EXPORTED_bool(
    check_kernel_launch,
    false,
    "Check kernel launch status after every kernel compute");
#endif

/**
 * CUDNN related FLAG
 * Name: conv2d_disable_cudnn
 * Since Version:
 * Value Range: bool, default=false
 * Example:
 * Note: Disable cudnn in conv2d.
 */
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PHI_DEFINE_EXPORTED_bool(conv2d_disable_cudnn,
                         false,
                         "Disable cudnn in conv2d");

PHI_DEFINE_EXPORTED_bool(use_fast_math,
                         false,
                         "Whether to use fast math GPU functions.");
#endif

/**
 * Distributed related FLAG
 * Name: FLAGS_get_host_by_name_time
 * Since Version: 2.2.0
 * Value Range: int32, default=120
 * Example:
 * Note: Get host by name time.
 */
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_XPU) || \
    defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_CUSTOM_DEVICE)
PHI_DEFINE_EXPORTED_int32(get_host_by_name_time,
                          120,
                          "The maximum time for get host by name time");
#endif

/**
 * Distributed related FLAG
 * Name: FLAGS_apply_pass_to_program
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example: FLAGS_apply_pass_to_program=true would apply IR Pass to
 *          program when using Fleet APIs.
 * Note: Apply IR pass to program. Be only useful when using Fleet APIs.
 */
PHI_DEFINE_EXPORTED_bool(
    apply_pass_to_program,
    false,
    "It controls whether to apply IR pass to program when using Fleet APIs");

/**
 * Debug related FLAG
 * Name: FLAGS_save_static_runtime_data
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: It controls whether to save runtime tensor in static mode.
 */
PHI_DEFINE_EXPORTED_bool(
    save_static_runtime_data,
    false,
    "It controls whether to save runtime tensor in static mode");

/**
 * Distributed related FLAG
 * Name: FLAGS_graph_load_in_parallel
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Control whether load graph node and edge with multi threads parallelly
 *       If it is not set, load graph data with one thread
 */
PHI_DEFINE_EXPORTED_bool(graph_load_in_parallel,
                         false,
                         "It controls whether load graph node and edge with "
                         "multi threads parallelly.");

/**
 * Distributed related FLAG
 * Name: FLAGS_enable_neighbor_list_use_uva
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: Control whether store neighbor_list with UVA in gpu graph mode
 */
PHI_DEFINE_EXPORTED_bool(enable_neighbor_list_use_uva,
                         false,
                         "It controls whether store neighbor_list with UVA");

/**
 * Distributed related FLAG
 * Name: FLAGS_graph_neighbor_size_percent
 * Since Version: 2.5.0
 * Value Range: double, default=1.0
 * Example:
 * Note: Control whether load graph node and edge with multi threads parallelly
 *       If it is not set, load graph data with one thread
 */
PHI_DEFINE_EXPORTED_double(graph_neighbor_size_percent,
                           1.0,
                           "It controls whether precent of neighbor_size.");

/**
 * Distributed related FLAG
 * Name: FLAGS_graph_metapath_split_opt
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Control whether load graph node and edge with multi threads parallelly
 *       If it is not set, load graph data with one thread
 */
PHI_DEFINE_EXPORTED_bool(graph_metapath_split_opt,
                         false,
                         "It controls whether load graph node and edge with "
                         "multi threads parallelly.");

/**
 * Distributed related FLAG
 * Name: FLAGS_graph_get_neighbor_id
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Control get all neighbor id when running sub part graph
 *       If it is not set, do not need get neighbor id when run all part graph
 */
PHI_DEFINE_EXPORTED_bool(
    graph_get_neighbor_id,
    false,
    "It controls get all neighbor id when running sub part graph.");

/**
 * Distributed related FLAG
 * Name: enable_exit_when_partial_worker
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: Control  whether exit trainer when an worker has no ins.
 *       If it is not set, trainer will exit until all worker finish train.
 */
PHI_DEFINE_EXPORTED_bool(
    enable_exit_when_partial_worker,
    false,
    "It controls whether exit trainer when an worker has no ins.");

/**
 * Distributed related FLAG
 * Name: enable_adjust_op_order
 * Since Version: 2.5.0
 * Value Range: int32, default=0
 * Example:
 * Note: Control  whether adjust op order in worker to reduce hbm cost in gpu
 * graph mode.
 */
PHI_DEFINE_EXPORTED_int32(
    enable_adjust_op_order,
    0,
    "It controls whether adjust op order in worker to reduce hbm cost");

/**
 * Distributed related FLAG
 * Name: enable_exit_when_partial_worker
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: represent gpugraph storage mode, 1 for full hbm, 2 for hbm + mem + ssd.
 */
PHI_DEFINE_EXPORTED_int32(gpugraph_storage_mode,
                          1,
                          "gpugraph storage mode, default 1");

/**
 * KP kernel related FLAG
 * Name: FLAGS_run_kp_kernel
 * Since Version: 2.3.0
 * Value Range: bool, default=false
 * Example: FLAGS_run_kp_kernel=true would use the kp kernel to compute in the
 * Op.
 * Note:
 */
PHI_DEFINE_EXPORTED_bool(run_kp_kernel,
                         false,
                         "It controls whether to run PaddlePaddle using KP");

/**
 * Distributed related FLAG
 * Name: FLAGS_allreduce_record_one_event
 * Since Version: 2.2.0
 * Value Range: bool, default=false
 * Example: FLAGS_allreduce_record_one_event=true makes the allreduce
 *          operations would only wait one event instead of multiple events.
 * Note: Make the allreduce operations would only wait one event instead of
 *       multiple events. Currently, only fuse allreduce supports this.
 *       Otherwise, the precision may be wrong.
 */
PHI_DEFINE_EXPORTED_bool(allreduce_record_one_event,
                         false,
                         "It controls whether the allreduce operations "
                         "would only wait one event instead of multiple "
                         "events. Currently, only fuse allreduce supports "
                         "this. Otherwise, the precision may be wrong.");

#ifdef PADDLE_WITH_CINN
/*
 * CINN related FLAG
 * Name: FLAGS_use_cinn
 * Since Version: 2.3
 * Value Range: bool, default=false
 * Example: FLAGS_use_cinn=true would run PaddlePaddle using CINN
 */
PHI_DEFINE_EXPORTED_bool(use_cinn,
                         false,
                         "It controls whether to run PaddlePaddle using CINN");

/*
 * CINN related FLAG
 * Name: FLAGS_allow_cinn_ops
 * Since Version: 2.3
 * Value Range: string, default=""
 * Example: FLAGS_allow_cinn_ops="mul;relu" would only cover `mul` and `relu`
 * when using CINN
 */
PHI_DEFINE_EXPORTED_string(allow_cinn_ops,
                           "",
                           "It controls the cinn op subset to be used, "
                           "which has the highest priority.");

/*
 * CINN related FLAG
 * Name: FLAGS_deny_cinn_ops
 * Since Version: 2.3
 * Value Range: string, default=""
 * Example: FLAGS_deny_cinn_ops="mul;relu" would block `mul` and `relu` two ops
 * when using CINN
 */
PHI_DEFINE_EXPORTED_string(deny_cinn_ops,
                           "",
                           "It controls the cinn op subset to be not used.");

/*
 * CINN related FLAG
 * Name: FLAGS_deny_cinn_ops
 * Since Version: 3.0 Beta
 * Value Range: bool, default=true
 * Example: FLAGS_enable_cinn_compile_cache=true would reuse cached Kernel
 * function
 */
PHI_DEFINE_EXPORTED_bool(
    enable_cinn_compile_cache,
    true,
    "It controls whether to enable cinn compilation cache.");
/*
 * CINN related FLAG
 * Name: FLAGS_deny_cinn_ops
 * Since Version: 3.0 Beta
 * Value Range: bool, default=-1
 * Example: FLAGS_cinn_compile_thread_nume=8
 */
PHI_DEFINE_EXPORTED_int64(
    cinn_compile_thread_num,
    -1,
    "It controls how many thread numbers applying compilation cache.");
/*
 * CINN related FLAG
 * Name: FLAGS_enable_interpretercore_launch_cinn
 * Since Version: 2.4
 * Value Range: bool, default=true
 * Example: FLAGS_enable_interpretercore_launch_cinn=true would execute the CINN
 * compiled instructions of a paddle graph with InterpreterCore, otherwise with
 * the CINN compiled runtime program in sequential order.
 */
PHI_DEFINE_EXPORTED_bool(enable_interpretercore_launch_cinn,
                         true,
                         "It controls whether to execute cinn compiled "
                         "program with InterpreterCore");

/*
 * CINN related FLAG
 * Name: FLAGS_enable_cinn_auto_tune
 * Since Version: 2.3
 * Value Range: bool, default=false
 * Example: FLAGS_enable_cinn_auto_tune=true would use CINN with its
 * auto-tune feature enabled
 */
PHI_DEFINE_EXPORTED_bool(enable_cinn_auto_tune,
                         false,
                         "It controls whether to use cinn with "
                         "its auto-tune feature enabled");

/*
 * CINN related FLAG
 * Name: FLAGS_cinn_subgraph_graphviz_dir
 * Since Version: 2.3
 * Value Range: string, default=""
 * Example: FLAGS_cinn_subgraph_graphviz_dir="./cinn_graph/" will save the
 * CINN sub-graph into "./cinn_graph/", and each sub-graph will save into
 * "fusion_groups_*"" directory
 */
PHI_DEFINE_EXPORTED_string(cinn_subgraph_graphviz_dir,
                           "",
                           "Specify the directory path of dot file of "
                           "graph, which is used for debug.");

/*
 * CINN related FLAG
 * Name: FLAGS_cinn_specify_input_dynamic_dim
 * Since Version: 3.0 Beta
 * Value Range: bool, default=false
 * Example: FLAGS_cinn_specify_input_dynamic_dim=true will use file set by
 * FLAGS_cinn_input_dynamic_dim_spec_file to specify input dynamic dimention.
 */
PHI_DEFINE_EXPORTED_bool(cinn_specify_input_dynamic_dim,
                         false,
                         "Whether to specify input dynamic dimention.");

/*
 * CINN related FLAG
 * Name: FLAGS_cinn_input_dynamic_dim_spec_file
 * Since Version: 3.0 Beta
 * Value Range: string, default=""
 * Example: FLAGS_cinn_input_dynamic_dim_spec_file="./config.json",
 * FLAGS_cinn_specify_input_dynamic_dim=true would use input dynamic dimention
 * predefined in ./config.json to specify input dynamic dimention.
 */
PHI_DEFINE_EXPORTED_string(
    cinn_input_dynamic_dim_spec_file,
    "",
    "File path of predefined input dynamic dimention specification.");

#endif

/*
 * CUDA Graph related FLAG
 * Name: FLAGS_new_executor_use_cuda_graph
 * Since Version: 2.4
 * Value Range: bool, default=false
 * Example: FLAGS_new_executor_use_cuda_graph=true would allow
 * new executor to use CUDA Graph.
 */
PHI_DEFINE_EXPORTED_bool(new_executor_use_cuda_graph,
                         false,
                         "Use CUDA Graph in new executor");

/*
 * CUDA Graph / Allocator related FLAG
 * Name: FLAGS_use_cuda_malloc_async_allocator
 * Since Version: 2.7
 * Value Range: bool, default=false
 * Example: FLAGS_use_cuda_malloc_async_allocator=true would allow
 * CUDAMallocAsyncAllocator replace StreamSafeCUDAAllocator.
 */
PHI_DEFINE_EXPORTED_bool(use_cuda_malloc_async_allocator,
                         false,
                         "Enable CUDAMallocAsyncAllocator");

/*
 * CUDAMallocAsyncAllocator related FLAG
 * Name: FLAGS_cuda_malloc_async_pool_memory_throttle_ratio
 * Since Version: 3.0
 * Value Range:  double, [0.0, 1.0], default=0.8
 * Note:memory_throttle_ratio provides a threshold that determines when to
 * initiate synchronization operations to deallocate memory. This mechanism
 * helps in ensuring that the system does not exceed its memory capacity while
 * also attempting to minimize performance degradation caused by frequent memory
 * synchronization.
 *
 * Please see Note [cuda_malloc_async_pool_memory_throttle_ratio]
 */
PHI_DEFINE_EXPORTED_double(
    cuda_malloc_async_pool_memory_throttle_ratio,
    0.8,
    "memory_throttle_ratio provides a threshold that determines when to "
    "initiate synchronization operations to deallocate memory. "
    "This mechanism helps in ensuring that the system does not exceed its "
    "memory capacity while also attempting to minimize performance degradation "
    "caused by frequent memory synchronization.");

/*
 * CUDA Graph / Allocator related FLAG
 * Name: FLAGS_auto_free_cudagraph_allocations_on_launch
 * Since Version: 2.7
 * Value Range: bool, default=true
 * Example: When enabling CUDA Graph with CUDAMallocAsyncAllocator, we add
 * cudaGraphInstantiateFlagAutoFreeOnLaunch so it would automatically
 * release graph-owned blocks that have not freed before relaunching.
 */
PHI_DEFINE_EXPORTED_bool(
    auto_free_cudagraph_allocations_on_launch,
    true,
    "When enabling CUDA Graph with CUDAMallocAsyncAllocator, we add "
    "cudaGraphInstantiateFlagAutoFreeOnLaunch so it would automatically "
    "release graph-owned blocks that have not freed before relaunching.");

/*
 * Executor related FLAG
 * Name: FLAGS_executor_log_deps_every_microseconds
 * Since Version: 2.5
 * Value Range: uint64, default=0
 * Example: FLAGS_executor_log_deps_every_microseconds=n (n>0) would
 * allow new executor log deps every n microseconds.
 */
PHI_DEFINE_EXPORTED_uint64(executor_log_deps_every_microseconds,
                           0,
                           "Enable new executor log deps every n microseconds");

PD_DEFINE_int32(record_pool_max_size,
                2000000,
                "SlotRecordDataset slot record pool max size");
PD_DEFINE_int32(slotpool_thread_num,
                1,
                "SlotRecordDataset slot pool thread num");
PD_DEFINE_bool(enable_slotpool_wait_release,  // NOLINT
               false,
               "enable slotrecord object wait release, default false");
PD_DEFINE_bool(enable_slotrecord_reset_shrink,  // NOLINT
               false,
               "enable slotrecord object reset shrink memory, default false");
PD_DEFINE_bool(enable_ins_parser_file,  // NOLINT
               false,
               "enable parser ins file, default false");
PHI_DEFINE_EXPORTED_bool(
    gpugraph_enable_hbm_table_collision_stat,
    false,
    "enable hash collisions stat for hbm table, default false");
PHI_DEFINE_EXPORTED_bool(
    cache_inference_while_scope,
    false,
    "Cache the scope of the while op to avoid repeated creation of the scope "
    "for each iteration and improve inference performance.");
PHI_DEFINE_EXPORTED_double(gpugraph_hbm_table_load_factor,
                           0.75,
                           "the load factor of hbm table, default 0.75");
PHI_DEFINE_EXPORTED_bool(
    gpugraph_enable_gpu_direct_access,
    false,
    "enable direct access between multi gpu cards, default false");
PHI_DEFINE_EXPORTED_bool(
    gpugraph_enable_segment_merge_grads,
    false,
    "enable segment merge gradients while push sparse, default false");
PHI_DEFINE_EXPORTED_uint64(
    gpugraph_merge_grads_segment_size,
    128,
    "segment size with segment gradient merge, default 128");
PHI_DEFINE_EXPORTED_uint64(gpugraph_slot_feasign_max_num,
                           5,
                           "max feasign number in one slot, default 5");
PHI_DEFINE_EXPORTED_int32(
    gpugraph_dedup_pull_push_mode,
    0,
    "enable dedup keys while pull push sparse, default 0");
PHI_DEFINE_EXPORTED_bool(gpugraph_load_node_list_into_hbm,
                         true,
                         "enable load_node_list_into_hbm, default true");
PHI_DEFINE_EXPORTED_int32(gpugraph_sparse_table_storage_mode,
                          0,
                          "parse_table_storage_mode, default 0");
PHI_DEFINE_EXPORTED_bool(enable_auto_detect_gpu_topo,
                         true,
                         "enable auto detect gpu topo, default true");
PHI_DEFINE_EXPORTED_bool(enable_auto_rdma_trans,
                         true,
                         "enable auto gpu rdma trans, default true");
PHI_DEFINE_EXPORTED_bool(enable_tracker_all2all,
                         false,
                         "enable tracker all2all log, default false");
PHI_DEFINE_EXPORTED_bool(enable_all2all_use_fp16,
                         false,
                         "enable all2all use fp16, default false");
PHI_DEFINE_EXPORTED_bool(enable_sparse_inner_gather,
                         false,
                         "enable sparse inner gather, default false");
PHI_DEFINE_EXPORTED_bool(gpugraph_debug_gpu_memory,
                         false,
                         "enable debug gpu memory, default false");
PHI_DEFINE_EXPORTED_bool(
    graph_embedding_split_infer_mode,
    true,
    "graph embedding split infer mode not need nccl barrier in gpu graph mode");
PHI_DEFINE_EXPORTED_bool(enable_graph_multi_node_sampling,
                         false,
                         "control multi-node sample in gpu graph mode");
PHI_DEFINE_EXPORTED_bool(
    query_dest_rank_by_multi_node,
    false,
    "Control whether to query dest rank by multi machine in gpu graph mode");
PHI_DEFINE_EXPORTED_bool(multi_node_sample_use_gpu_table,
                         true,
                         "Control whether to use gpu table in sample multi "
                         "machine in gpu graph mode");

/**
 * ProcessGroupNCCL related FLAG
 * Name: nccl_blocking_wait
 * Since Version:
 * Value Range: bool, default=false
 * Example:
 * Note: nccl blocking wait.
 */

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PHI_DEFINE_EXPORTED_bool(nccl_blocking_wait, false, "nccl blocking wait");
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PHI_DEFINE_EXPORTED_bool(benchmark_nccl,
                         false,
                         "enable nccl debug mode to synchronize nccl comm");
#endif

PHI_DEFINE_EXPORTED_bool(
    benchmark,
    false,
    "Doing memory benchmark. It will make deleting scope synchronized, "
    "and add some memory usage logs."
    "Default cuda is asynchronous device, set to True will"
    "force op run in synchronous mode.");

PHI_DEFINE_EXPORTED_bool(eager_communication_connection,
                         false,
                         "enable eager to create nccl comm");

/**
 * Autotune related FLAG
 * Name: FLAGS_use_autotune
 * Since Version: 2.3.0
 * Value Range: bool, default=false
 * Example:
 */
PHI_DEFINE_EXPORTED_bool(use_autotune, false, "Whether enable autotune.");

/**
 * CINN training related FLAG
 * Name: FLAGS_disable_dyshape_in_train
 * Since Version: 2.7.0
 * Value Range: bool, default=false
 * Example:
 */
PHI_DEFINE_EXPORTED_bool(disable_dyshape_in_train,
                         false,
                         "Whether disable dyshape in training.");

/**
 * CINN accuracy check related FLAG
 * Name: FLAGS_enable_cinn_accuracy_check
 * Since Version: 3.0 beta
 * Value Range: bool, default=false
 */
PHI_DEFINE_EXPORTED_bool(enable_cinn_accuracy_check,
                         false,
                         "Whether enable accuracy check in cinn.");

/**
 * CINN fuse parallel matmul pass related FLAG
 * Name: FLAGS_enable_fuse_parallel_matmul_pass
 * Since Version: 3.0 beta
 * Value Range: bool, default=true
 */
PHI_DEFINE_EXPORTED_bool(enable_fuse_parallel_matmul_pass,
                         true,
                         "Whether enable fuse_parallel_matmul_pass in cinn.");

/**
 * CINN fallback fusion ops FLAG
 * Name: FLAGS_enable_fusion_fallback
 * Since Version: 3.0 beta
 * Value Range: bool, default=false
 */
PHI_DEFINE_EXPORTED_bool(enable_fusion_fallback,
                         false,
                         "Whether enable fallback fusion ops in cinn.");

/**
 * CINN fusion result check FLAG
 * Name: FLAGS_enable_fusion_result_check
 * Since Version: 3.0 beta
 * Value Range: bool, default=false
 */
PHI_DEFINE_EXPORTED_bool(enable_fusion_result_check,
                         false,
                         "Whether enable fusion result check in cinn.");

/**
 * CINN TransposeItesr transform fusion FLAG
 * Name: FLAGS_enable_transpose_iters_in_fusion
 * Since Version: 3.0 beta
 * Value Range: bool, default=true
 */
PHI_DEFINE_EXPORTED_bool(
    enable_transpose_iters_in_fusion,
    true,
    "Whether enable use transpose iters transform in cinn fusion.");

/**
 * CINN ReuseIters transform fusion FLAG
 * Name: FLAGS_enable_reuse_iters_in_fusion
 * Since Version: 3.0 beta
 * Value Range: bool, default=true
 */
PHI_DEFINE_EXPORTED_bool(
    enable_reuse_iters_in_fusion,
    true,
    "Whether enable use reuse iters transform in cinn fusion.");

/**
 * CINN AppendIters transform fusion FLAG
 * Name: FLAGS_enable_append_iters_in_fusion
 * Since Version: 3.0 beta
 * Value Range: bool, default=true
 */
PHI_DEFINE_EXPORTED_bool(
    enable_append_iters_in_fusion,
    true,
    "Whether enable use append iters transform in cinn fusion.");

/**
 * Conv Search cache max number related FLAG
 * Name: FLAGS_search_cache_max_number
 * Since Version: 2.3.0
 * Value Range: int32, default=1000000
 * Example:
 */
PHI_DEFINE_EXPORTED_int32(search_cache_max_number,
                          1000000,
                          "search_cache_max_number.");

/**
 * Performance related FLAG
 * Name: einsum_opt
 * Since Version: 2.3.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, EinsumOp will be optimized by innercache reuse, which
 * uses more gpu memory.
 */
PHI_DEFINE_EXPORTED_bool(
    einsum_opt,
    false,
    "EinsumOp backward will be speedup at the expense of more gpu memory.");

/**
 * Performance related FLAG
 * Name: enable_auto_layout_pass
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, using AutoLayoutInsertPass and AutuLayoutSimplifyPass by
 * default
 */
PHI_DEFINE_EXPORTED_bool(enable_auto_layout_pass,
                         false,
                         "Whether enable auto_layout_pass.");

/**
 * JitLayer related FLAG
 * Name: FLAGS_jit_engine_type
 * Since Version: 2.3.0
 * Value Range: string, {Executor, PE},
 * default=Predictor
 * Example:
 * Note:
 * FLAGS_jit_engine_type == New, using InterpreterEngine by default
 * FLAGS_jit_engine_type == Predictor, using inference Predictor by default
 */
PHI_DEFINE_EXPORTED_string(jit_engine_type,
                           "Predictor",
                           "Choose default function type in JitLayer.");

/**
 * Custom Device NPU related FLAG
 * Name: FLAGS_npu_storage_format
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: Enable NPU Storage Format for Ascend910 performance improvement.
 */
PHI_DEFINE_EXPORTED_bool(npu_storage_format, false, "");

#ifdef PADDLE_WITH_CUDNN_FRONTEND
/**
 * CUDNNv8 related FLAG
 * Name: enable_cudnn_frontend
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: Enable CUDNNv8 Frontend API for CUDNN kernels.
 */
PHI_DEFINE_EXPORTED_bool(enable_cudnn_frontend, false, "");

/**
 * CUDNNv8 related FLAG
 * Name: cudnn_cache_saturation_count
 * Since Version: 2.5.0
 * Value Range: int64_t, default=1
 * Example:
 * Note: Set saturation count for CUDNNv8 cache. A candidate execution
 * plan need to be considered as the fastest plan by exhaustive search
 * N times before it is actually added in the cache. It is useful when
 * the result of exhaustive search is unstable.
 */
PHI_DEFINE_EXPORTED_int32(cudnn_cache_saturation_count, 1, "");
#endif  // PADDLE_WITH_CUDNN_FRONTEND

/**
 * CI related FLAG
 * Name: trt_ibuilder_cache
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: This FLAG is only enabled when CI is running. If True, a persistent
 * IBuilder is added to avoid TensorRT unload/reload kernels.
 */
PHI_DEFINE_EXPORTED_bool(trt_ibuilder_cache,
                         false,
                         "Add a persistent ibuilder.");

/**
 * mmap_allocator related FLAG
 * Name: use_shm_cache
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: . If True, mmap_allocator will cache shm file to decrease munmap
 * operation.
 */
PHI_DEFINE_EXPORTED_bool(use_shm_cache,
                         false,
                         "Use shm cache in mmap_allocator.");

/**
 * mmap_allocator related FLAG
 * Name: dataloader_use_file_descriptor
 * Since Version: 2.6.2
 * Value Range: bool, default=false
 * Example:
 * Note: . If True, mmap_allocator will use file descriptor to open shared
 * memory operation.
 */
PHI_DEFINE_EXPORTED_bool(dataloader_use_file_descriptor,
                         false,
                         "Use file descriptor in mmap_allocator.");

/**
 * Tensor operants related FLAG
 * Name: tensor_operants_mode
 * Since Version: 2.5.0
 * Value Range: string, {eager, phi, static}
 * default=eager
 * Example:
 * Note: For switching tensor operants mode of PaddlePaddle.
 *       - eager mode: tensor operants with dygraph autograd;
 *       - phi mode: tensor operants with only phi forward API;
 *       - static mode: tensor operants within static graph.
 */
PHI_DEFINE_EXPORTED_string(tensor_operants_mode,
                           "eager",
                           "Tensor operants mode");

/**
 * Using PIR in executor  FLAG
 * Name: enable_pir_in_executor
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, executor will use PIR
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_in_executor,
                         false,
                         "Enable PIR in executor");

/**
 * Using PIR by translating legacy program to pir program
 * for dy2st mode  FLAG
 * Name: enable_pir_in_executor
 * Since Version: 2.6.0
 * Value Range: bool, default=true
 * Example:
 * Note: If True, program will be translated to pir program
 * and then run in executor for dy2st mode.
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_with_pt_in_dy2st,
                         true,
                         "Enable PIR in executor");

PHI_DEFINE_EXPORTED_string(logging_pir_py_code_dir,
                           "",
                           "the logging directory to save pir py code");

PHI_DEFINE_EXPORTED_int64(
    logging_pir_py_code_int_tensor_element_limit,
    2048,
    "dump int tensor data if its element count less than this limit.");

PHI_DEFINE_EXPORTED_bool(logging_trunc_pir_py_code,
                         true,
                         "whether truncate the logging files under directory "
                         "FLAGS_logging_pir_py_code_dir");

PHI_DEFINE_EXPORTED_bool(logging_pir_py_code_dump_symbolic_dims,
                         false,
                         "whether dump symbolic dims into pir py code.");

PHI_DEFINE_EXPORTED_bool(
    pir_interpreter_record_stream_for_gc_cache,
    false,
    "whether PirInterpreter::RecordStreamForGC use cache strategy.");

/**
 * Using PIR API in Python
 * Name: enable_pir_api
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, PIR API will be used in Python
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_api, true, "Enable PIR API in Python");

/**
 * Using PIR in executor FLAG
 * Name: enable_pir_in_executor_trace_run
 * Since Version: 2.6.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, executor will use PIR and run in beta version by for trace
 * version.
 */
PHI_DEFINE_EXPORTED_bool(enable_pir_in_executor_trace_run,
                         false,
                         "Enable PIR in executor");

/**
 * Apply inplace pass to PIR FLAG
 * Name: pir_apply_inplace_pass
 * Since Version: 2.6.0
 * Value Range: bool, default=true
 * Example:
 * Note: If True, will apply inplace pass to PIR.
 */
PHI_DEFINE_EXPORTED_bool(pir_apply_inplace_pass,
                         true,
                         "Whether to apply inplace pass on lowering "
                         "::pir::Program to Kernel Dialect");

PHI_DEFINE_EXPORTED_string(
    ir_inplace_kernel_blacklist,
    "",
    "It controls the ir inplace kernel subset do not use.");
/**
 * Specify the directory of saving PIR subgraph from @to_static
 * Name: pir_subgraph_saving_dir
 * Since Version: 2.6.0
 * Value Range: str, default=""
 * Example:
 * Note: "/workspace/my_path", it will save into my_path dir;
 */
PHI_DEFINE_EXPORTED_string(
    pir_subgraph_saving_dir,
    "",
    "Specify the directory of saving PIR subgraph from @to_static.");

PHI_DEFINE_EXPORTED_bool(enable_record_memory, false, "Enable memory recorder");

PHI_DEFINE_EXPORTED_bool(
    eager_delete_scope,
    true,
    "Delete local scope eagerly. It will reduce GPU memory usage but "
    "slow down the destruction of variables.(around 1% performance harm)");

// Used to filter events, works like glog VLOG(level).
// RecordEvent will works if host_trace_level >= level.
PHI_DEFINE_EXPORTED_int64(host_trace_level,
                          1,
                          "RecordEvent will works "
                          "if host_trace_level >= level.");

PHI_DEFINE_EXPORTED_int32(
    multiple_of_cupti_buffer_size,
    1,
    "Multiple of the CUPTI device buffer size. If the timestamps have "
    "been dropped when you are profiling, try increasing this value.");

PHI_DEFINE_EXPORTED_bool(print_ir, false, "Whether print ir debug str.");

PHI_DEFINE_EXPORTED_bool(
    prim_skip_dynamic,
    true,
    "Whether to skip decomposing vjp op with dynamic shape.");
PHI_DEFINE_EXPORTED_bool(
    prim_enable_dynamic,
    false,
    "Whether to enable decomposing composite op with dynamic shape.");
PHI_DEFINE_EXPORTED_bool(prim_check_ops,
                         false,
                         "Whether to check the decomposed program, to ensure "
                         "that only the primitive operator is present.");

// PIR and prim related FLAG
// Example: FLAGS_prim_forward_blacklist="pd_op.relu;pd_op.mean" would block
// `relu` and `mean` two ops in decompsition.
PHI_DEFINE_EXPORTED_string(
    prim_forward_blacklist,
    "",
    "It controls the forward blacklist ops not to be decomposed.");

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_CUSTOM_DEVICE)
/**
 * Communication library related FLAG
 * Name: FLAGS_dynamic_static_unified_comm
 * Since Version: 2.5
 * Value Range: bool, default=true
 * Example:
 * Note: Whether to use new communication library in auto parallel and static
 * mode. If true, it will use unified CommContextManager for communication.
 */
PHI_DEFINE_EXPORTED_bool(dynamic_static_unified_comm,
                         true,
                         "Whether to use new communication library in auto "
                         "parallel and static mode.");
#endif  // FLAGS_dynamic_static_unified_comm

/**
 * ProcessGroupNCCL related FLAG
 * Name: enable_async_trace
 * Since Version:
 * Value Range: bool, default=false
 * Example:
 * Note: enable nccl async trace.
 */

PHI_DEFINE_EXPORTED_bool(enable_async_trace,
                         false,
                         "enable collective async trace");

PHI_DEFINE_EXPORTED_int32(async_trace_count, 5, "collective async trace count");

PHI_DEFINE_EXPORTED_bool(
    use_auto_growth_pinned_allocator,
    false,
    "Whether to use the auto_growth CUDA pinned allocator.");

PHI_DEFINE_EXPORTED_bool(
    sync_after_alloc,
    false,
    "Whether to perform device synchronization after allocation.");
PHI_DEFINE_EXPORTED_int64(alloc_fill_value,
                          -1,
                          "Whether to fill fixed value after allocation. "
                          "This is useful for debugging.");

/**
 * Apply shape optimization pass to PIR FLAG
 * Name: pir_apply_shape_optimization_pass
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will apply shape_optimization pass to PIR.
 */
PHI_DEFINE_EXPORTED_bool(pir_apply_shape_optimization_pass,
                         false,
                         "Whether to apply shape_optimization pass "
                         "to infer symbolic shape");

PHI_DEFINE_EXPORTED_int64(
    pir_broadcast_tree_limit,
    32,
    "Maximum number of broadcast nodes allowed in a tree");

PHI_DEFINE_EXPORTED_string(
    nvidia_package_dir,  // NOLINT
    "",
    "Specify root dir path for nvidia site-package, such as "
    "python3.9/site-packages/nvidia");

PHI_DEFINE_EXPORTED_string(
    cudnn_dir,  // NOLINT
    "",
    "Specify path for loading libcudnn.so. For instance, "
    "/usr/local/cudnn/lib. If empty [default], dlopen "
    "will search cudnn from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(  // NOLINT
    cuda_dir,
    "",
    "Specify path for loading cuda library, such as libcublas, libcublasLt "
    "libcurand, libcusolver. For instance, /usr/local/cuda/lib64. "
    "If default, dlopen will search cuda from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(cublas_dir,  // NOLINT
                           "",
                           "Specify path for loading libcublas.so.");
PHI_DEFINE_EXPORTED_string(
    nccl_dir,  // NOLINT
    "",
    "Specify path for loading nccl library, such as libnccl.so. "
    "For instance, /usr/local/cuda/lib64. If default, "
    "dlopen will search cuda from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(cupti_dir,
                           "",
                           "Specify path for loading cupti.so.");  // NOLINT

PHI_DEFINE_EXPORTED_string(  // NOLINT
    tensorrt_dir,
    "",
    "Specify path for loading tensorrt library, such as libnvinfer.so.");

PHI_DEFINE_EXPORTED_string(
    mklml_dir,
    "",
    "Specify path for loading libmklml_intel.so.");  // NOLINT

PHI_DEFINE_EXPORTED_string(lapack_dir,
                           "",
                           "Specify path for loading liblapack.so.");  // NOLINT

/**
 * Apply check infer symbolic pass FLAG
 * Name: check_infer_symbolic_pass
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will apply check_infer_symbolic pass.
 */
PHI_DEFINE_EXPORTED_bool(
    check_infer_symbolic,
    false,
    "Whether to use check_infer_symbolic_pass. This pass can check "
    "the symbolic inference accuracy by comparing the the value "
    "shape between dynamic shape and static shape.");

/**
 * Name: manually_trans_conv_filter
 * Since Version: 3.0.0 Beta
 * Value Range: bool, default=false
 */
PHI_DEFINE_EXPORTED_bool(
    manually_trans_conv_filter,
    false,
    "Whether to manually transpose the filter of conv2d. This pass can "
    "accelerate the performance of conv2d since it transpose filter ahead");

/**
 * Apply CSE optimize pass in Dy2St
 * Name: enable_cse_in_dy2st
 * Since Version: 3.0.0
 * Value Range: bool, default=true
 * Example:
 * Note: If True, will apply CSE optimize pass in Dy2St.
 */
PHI_DEFINE_EXPORTED_bool(enable_cse_in_dy2st,
                         true,
                         "Apply CSE optimize pass in Dy2St");

/**
 * Max count of eliminate redundant computation in CSE, for debug usage
 * Name: cse_max_count
 * Since Version: 3.0.0
 * Value Range: int32, default=-1
 * Example:
 * Note: If -1, will not limit the max count of eliminate redundant computation.
 */
PHI_DEFINE_EXPORTED_int32(
    cse_max_count,
    -1,
    "Max count of eliminate redundant computation in CSE, for debug usage");

/**
 * Apply global search in cublaslt gemm
 * Name: enable_blaslt_global_search
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will apply global search in blaslt.
 */
PHI_DEFINE_EXPORTED_bool(enable_blaslt_global_search,
                         false,
                         "Whether to use global search in cublaslt gemm.");

/**
 * Apply load search configs file generated by offline in cublaslt gemm
 * Name: cublaslt_device_best_config
 * Since Version: 3.0.0
 * Value Range: string, default="", a absolute file path
 * Example:
 * Note: If set this flag, will load search configs file generated by offline.
 */
PHI_DEFINE_EXPORTED_string(cublaslt_device_best_config,
                           "",
                           "Whether to load search configs file generated by "
                           "offline in cublaslt gemm.");

/**
 * Wether to use xqa optim in block_multihead_attention kernel (GQA)
 * Name: use_xqa_optim
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will use xqa optim in block_multihead_attention kernel (GQA).
 */
PHI_DEFINE_EXPORTED_bool(
    use_xqa_optim,
    false,
    "Enable xqa optim in block_multihead_attention kernel (GQA).");

PHI_DEFINE_EXPORTED_bool(cuda_core_int8_gemm,
                         false,
                         "Enable speed up int8 gemm calculations when m<=4");

PHI_DEFINE_EXPORTED_string(
    mkl_dir,  // NOLINT
    "",
    "Specify path for loading libmkl_rt.so. "
    "For insrance, /opt/intel/oneapi/mkl/latest/lib/intel64/."
    "If default, "
    "dlopen will search mkl from LD_LIBRARY_PATH");

PHI_DEFINE_EXPORTED_string(op_dir,  // NOLINT
                           "",
                           "Specify path for loading user-defined op library.");

PHI_DEFINE_EXPORTED_string(cusparselt_dir,  // NOLINT
                           "",
                           "Specify path for loading libcusparseLt.so.");
PHI_DEFINE_EXPORTED_string(curand_dir,  // NOLINT
                           "",
                           "Specify path for loading libcurand.so.10.");
PHI_DEFINE_EXPORTED_string(cusolver_dir,  // NOLINT
                           "",
                           "Specify path for loading libcusolver.so.*.");
PHI_DEFINE_EXPORTED_string(cusparse_dir,  // NOLINT
                           "",
                           "Specify path for loading libcusparse.so.*.");
PHI_DEFINE_EXPORTED_string(
    win_cuda_bin_dir,  // NOLINT
    "",
    "Specify path for loading *.dll about cuda on windows");

/**
 * Collect shapes of value for TensorRTEngine
 * Name: enable_collect_shape
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Example:
 * Note: If True, will collect shapes of value when run executor.
 */
PHI_DEFINE_EXPORTED_bool(enable_collect_shape,
                         false,
                         "Collect shapes of value for TensorRTEngine");
// Example: FLAGS_accuracy_check_atol=1e-3 would set the atol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_atol_fp32,
                           1e-6,
                           "It controls the atol of accuracy_check op");

// Example: FLAGS_accuracy_check_rtol=1e-3 would set the rtol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_rtol_fp32,
                           1e-6,
                           "It controls the rtol of accuracy_check op");

// Example: FLAGS_accuracy_check_atol=1e-3 would set the atol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_atol_fp16,
                           1e-3,
                           "It controls the atol of accuracy_check op");

// Example: FLAGS_accuracy_check_rtol=1e-3 would set the rtol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_rtol_fp16,
                           1e-3,
                           "It controls the rtol of accuracy_check op");

// Example: FLAGS_accuracy_check_atol=1e-3 would set the atol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_atol_bf16,
                           1e-3,
                           "It controls the atol of accuracy_check op");

// Example: FLAGS_accuracy_check_rtol=1e-3 would set the rtol to 1e-3.
PHI_DEFINE_EXPORTED_double(accuracy_check_rtol_bf16,
                           1e-3,
                           "It controls the rtol of accuracy_check op");

PHI_DEFINE_EXPORTED_bool(
    pinned_memory_as_cpu_backend,
    false,
    "Whether use CPU backend, when tensor is pinned_memory.");

PHI_DEFINE_EXPORTED_int32(
    trt_min_group_size,
    3,
    "when the trt subgraph size is not larger than `trt_min_group_size`, the "
    "group will fallback to original graph.");

/**
 * Enable align mode for auto parallel. If True, the loss results will aligned
 * with dynamic manual-parallel.
 * Name: enable_auto_parallel_align_mode
 * Since Version: 3.0.0
 * Value Range: bool, default=false
 * Note: Just used for testing. Do not use in model trainning.
 */
PHI_DEFINE_EXPORTED_bool(enable_auto_parallel_align_mode,
                         false,
                         "Enable align mode for auto parallel");

/**
 * fused_multi_transformer_op related FLAG
 * Name: fused_multi_transformer_op_use_mbfmha
 * Since Version: 2.5.0
 * Value Range: bool, default=false
 * Example:
 * Note: Enable flash decoding for mmha kernels in fused_multi_transformer_op.
 */
PHI_DEFINE_EXPORTED_bool(fused_multi_transformer_op_use_mbfmha,
                         false,
                         "Enable flash decoding for mmha kernels in "
                         "fused_multi_transformer_op.");

PHI_DEFINE_EXPORTED_int64(multi_block_attention_min_partition_size,
                          1024,
                          "The minimum partition size for flash decoding");

PHI_DEFINE_EXPORTED_bool(save_cf_stack_op,
                         false,
                         "Save cf stack op for higher-order derivatives.");

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/**
 * FlashAttention related FLAG
 * Name: FLAGS_flash_attn_version
 * Value Range: int32, default=2
 * Example:
 * Note: Specify the version of FlashAttention to use, options are 2 or 3.
 *        Version 2 requires Ampere architecture or higher,
 *        while version 3 requires Hopper architecture.
 */
PHI_DEFINE_EXPORTED_int32(
    flash_attn_version,
    2,
    "Specify the version of FlashAttention to use, options are 2 or 3. "
    "Version 2 requires Ampere architecture or higher, "
    "while version 3 requires Hopper architecture.");
#endif

PHI_DEFINE_EXPORTED_bool(
    new_executor_serial_run,
    false,
    "Enable serial execution for standalone executor, used for debug.");
PHI_DEFINE_EXPORTED_bool(
    new_executor_static_build,
    false,
    "Build the interpreterCore statically without running kernels.");
PHI_DEFINE_EXPORTED_bool(new_executor_use_inplace,
                         false,
                         "Use inplace in new executor");
PHI_DEFINE_EXPORTED_bool(new_executor_use_local_scope,
                         true,
                         "Use local_scope in new executor(especially used "
                         "in UT), can turn off for better performance");

PHI_DEFINE_EXPORTED_bool(gpugraph_force_device_batch_num_equal,
                         false,
                         "enable force_device_batch_num_equal, default false");
PHI_DEFINE_EXPORTED_int32(gpugraph_offload_param_stat,
                          0,
                          "enable offload param stat, default 0");
PHI_DEFINE_EXPORTED_string(gpugraph_offload_param_extends,
                           ".w_0_moment,.b_0_moment",
                           "offload param extends list");
PHI_DEFINE_EXPORTED_int32(gpugraph_offload_gather_copy_maxsize,
                          16,
                          "offload gather copy max size , default 16M");
PHI_DEFINE_EXPORTED_int32(gpugraph_parallel_copyer_split_maxsize,
                          64,
                          "offload gather copy max size , default 64M");
PHI_DEFINE_EXPORTED_int32(gpugraph_parallel_stream_num,
                          8,
                          "offload parallel copy stream num");
PHI_DEFINE_EXPORTED_bool(gpugraph_enable_print_op_debug,
                         false,
                         "enable print op debug ,default false");

PHI_DEFINE_EXPORTED_bool(enable_dump_main_program,
                         false,
                         "enable dump main program, default false");
PHI_DEFINE_EXPORTED_bool(
    enable_unused_var_check,
    false,
    "Checking whether operator contains unused inputs, "
    "especially for grad operator. It should be in unittest.");

PHI_DEFINE_EXPORTED_double(
    fuse_parameter_memory_size,
    -1.0,  // MBytes
    "fuse_parameter_memory_size is up limited memory size(MB)"
    "of one group parameters' gradient which is the input "
    "of communication calling(e.g NCCLAllReduce). "
    "The default value is 0, it means that "
    "not set group according to memory_size.");
PHI_DEFINE_EXPORTED_int32(
    fuse_parameter_groups_size,
    1,
    "fuse_parameter_groups_size is the up limited size of one group "
    "parameters' gradient. "
    "The default value is a experimental result. If the "
    "fuse_parameter_groups_size is 1, it means that the groups size is "
    "the number of parameters' gradient. If the fuse_parameter_groups_size is "
    "-1, it means that there are only one group. The default value is 3, it is "
    "an experimental value.");

PHI_DEFINE_EXPORTED_string(print_sub_graph_dir,
                           "",
                           "FLAGS_print_sub_graph_dir is used "
                           "to print the nodes of sub_graphs.");

PHI_DEFINE_EXPORTED_bool(convert_all_blocks,
                         true,
                         "Convert all blocks in program into SSAgraphs");

PHI_DEFINE_EXPORTED_bool(all_blocks_convert_trt,
                         false,
                         "Convert all blocks'Ops into TensorRT Ops");

PHI_DEFINE_EXPORTED_string(static_executor_perfstat_filepath,
                           "",
                           "FLAGS_static_executor_perfstat_filepath "
                           "enables performance statistics for the static "
                           "graph executor.");

PHI_DEFINE_EXPORTED_bool(print_kernel_run_info,
                         false,
                         "Whether print kernel run info.");

PHI_DEFINE_EXPORTED_bool(
    add_dependency_for_communication_op,
    true,
    "Whether to add dependency for communication Ops. It is just a temporary "
    "FLAGS especially for auto parallel to avoid the concurrency damage by the "
    "communication dependency added in standalone executor.");

// The difference between "sequential_run" and "serial_run":
// "sequential_run" dispatches OPs one by one according to the sequence in the
// Program, while "serial_run" ensures that all Ops are scheduled in a signal
// thread. In standalone executor, "sequential_run" is also "serial_run", while
// "serial_run" is not necessarily "sequential_run".
PHI_DEFINE_EXPORTED_bool(new_executor_sequential_run,
                         false,
                         "Enable sequential execution for standalone "
                         "executor, only applied to GPU OPs.");

// add debug info
PHI_DEFINE_EXPORTED_bool(enable_dependency_builder_debug_info,
                         false,
                         "Enable dependency builder debug info");

// FLAGS_force_sync_ops is used to finer control the op-sync in executor.
// The format is: "micro_batch_id, job_name, op_id, op_name | micro_batch_id,
// job_name, op_id, op_name | ...". Keep spaces to syncs all name/id. Example:
// 1. sync the recv_v2 op in the second backward-job of 1F1B scheduling:
// FLAGS_force_sync_ops="1, backward, , recv_v2"
// 2. sync the full op with op_id=5: FLAGS_force_sync_ops=" , , 5, full"
// 3. sync all ops in the first default-job: FLAGS_force_sync_ops="0,default,,
// 4. sync all ops in the forward-job and backward-job: FLAGS_force_sync_ops=" ,
// forward, , | , backward, , , "
PHI_DEFINE_EXPORTED_string(force_sync_ops,
                           "",
                           "Pattern to force sync ops in executor.");

PHI_DEFINE_EXPORTED_bool(
    fleet_executor_with_standalone,
    false,
    "Use standalone executor to run ops. Temporary FLAGS, will be removed "
    "after all fleet executor cases are modified to run ops with standalone "
    "executor.");

PHI_DEFINE_EXPORTED_bool(graph_edges_split_only_by_src_id,
                         false,
                         "multi-node split edges only by src id");
PHI_DEFINE_EXPORTED_string(
    graph_edges_split_mode,
    "hard",
    "graph split split, optional: [dbh,hard,fennel,none], default:hard");
PHI_DEFINE_EXPORTED_bool(graph_edges_split_debug,
                         false,
                         "graph split by debug");
PHI_DEFINE_EXPORTED_int32(graph_edges_debug_node_id, 0, "graph debug node id");
PHI_DEFINE_EXPORTED_int32(graph_edges_debug_node_num,
                          2,
                          "graph debug node num");

PHI_DEFINE_EXPORTED_string(rocksdb_path,
                           "database",
                           "path of sparse table rocksdb file");

PHI_DEFINE_EXPORTED_uint64(dygraph_debug,
                           0,
                           "Debug level of dygraph. This flag is not "
                           "open to users");

PHI_DEFINE_EXPORTED_string(
    tracer_profile_fname,
    "xxgperf",
    "Profiler filename for imperative tracer, which generated by gperftools."
    "Only valid when compiled `WITH_PROFILER=ON`. Empty if disable.");

PHI_DEFINE_EXPORTED_int32(rpc_send_thread_num,
                          12,
                          "number of threads for rpc send");

PHI_DEFINE_EXPORTED_bool(prim_enabled, false, "enable_prim or not");
PHI_DEFINE_EXPORTED_bool(prim_all, false, "enable prim_all or not");
PHI_DEFINE_EXPORTED_bool(prim_forward, false, "enable prim_forward or not");
PHI_DEFINE_EXPORTED_bool(prim_backward, false, "enable prim_backward not");

PD_DEFINE_int32(pserver_push_dense_merge_limit,
                12,
                "limit max push_dense local merge requests");

PD_DEFINE_int32(pserver_push_sparse_merge_limit,
                12,
                "limit max push_sparse local merge requests");

PD_DEFINE_int32(pserver_pull_dense_limit,
                12,
                "limit max push_sparse local merge requests");

PD_DEFINE_int32(pserver_async_push_dense_interval_ms,
                10,
                "async push_dense to server interval");

PD_DEFINE_int32(pserver_async_push_sparse_interval_ms,
                10,
                "async push_sparse to server interval");

PD_DEFINE_bool(pserver_scale_gradient_by_merge,
               false,
               "scale dense gradient when merged");

PD_DEFINE_int32(pserver_communicate_compress_type,
                0,
                "none:0 snappy:1 gzip:2 zlib:3 lz4:4");

PD_DEFINE_int32(pserver_max_async_call_num,
                13,
                "max task num in async_call_server");

PD_DEFINE_int32(pserver_timeout_ms,
                500000,
                "pserver request server timeout_ms");

PD_DEFINE_int32(pserver_connect_timeout_ms,
                10000,
                "pserver connect server timeout_ms");

PD_DEFINE_int32(pserver_sparse_merge_thread,
                1,
                "pserver sparse merge thread num");

PD_DEFINE_int32(pserver_sparse_table_shard_num,
                1000,
                "sparse table shard for save & load");

PD_DEFINE_int32(pserver_timeout_ms_s2s,
                10000,
                "pserver request server timeout_ms");
PD_DEFINE_int32(pserver_connect_timeout_ms_s2s,
                10000,
                "pserver connect server timeout_ms");
PD_DEFINE_string(pserver_connection_type_s2s,
                 "pooled",
                 "pserver connection_type[pooled:single]");

PD_DEFINE_uint64(total_fl_client_size, 100, "supported total fl client size");
PD_DEFINE_uint32(coordinator_wait_all_clients_max_time, 60, "uint32: s");

PD_DEFINE_int32(heter_world_size, 100, "group size");  // group max size
PD_DEFINE_int32(switch_send_recv_timeout_s, 600, "switch_send_recv_timeout_s");

PD_DEFINE_bool(pserver_print_missed_key_num_every_push,
               false,
               "pserver_print_missed_key_num_every_push");
PD_DEFINE_bool(pserver_create_value_when_push,
               true,
               "pserver create value when push");
PD_DEFINE_bool(pserver_enable_create_feasign_randomly,
               false,
               "pserver_enable_create_feasign_randomly");
PD_DEFINE_int32(pserver_table_save_max_retry,
                3,
                "pserver_table_save_max_retry");

PD_DEFINE_bool(enable_show_scale_gradient, true, "enable show scale gradient");

PD_DEFINE_bool(pserver_open_strict_check, false, "pserver_open_strict_check");
PD_DEFINE_int32(pserver_load_batch_size, 5000, "load batch size for ssd");

PD_DEFINE_string(devices,  // NOLINT
                 "",
                 "The devices to be used which is joined by comma.");
PD_DEFINE_int32(math_num_threads,
                1,
                "Number of threads used to run math functions.");

PD_DEFINE_string(inference_model_dir, "", "inference test model dir");

PD_DEFINE_bool(  // NOLINT
    custom_model_save_cpu,
    false,
    "Keep old mode for developers, the model is saved on cpu not device.");

PD_DEFINE_bool(profile, false, "Turn on profiler for fluid");  // NOLINT

PD_DEFINE_string(dirname, "", "dirname to tests.");

PHI_DEFINE_EXPORTED_bool(enable_gpu_memory_usage_log,
                         false,
                         "Whether to print the message of gpu memory usage "
                         "at exit, mainly used for UT and CI.");

PHI_DEFINE_EXPORTED_bool(enable_gpu_memory_usage_log_mb,
                         true,
                         "Whether to print the message of gpu memory usage "
                         "MB as a unit of measurement.");
PHI_DEFINE_EXPORTED_uint64(cuda_memory_async_pool_realease_threshold,
                           ULLONG_MAX,
                           "Amount of reserved memory in bytes to hold onto "
                           "before trying to release memory back to the OS");

// If use_pinned_memory is true, CPUAllocator calls mlock, which
// returns pinned and locked memory as staging areas for data exchange
// between host and device.  Allocates too much would reduce the amount
// of memory available to the system for paging.  So, by default, we
// should set false to use_pinned_memory.
PHI_DEFINE_EXPORTED_bool(use_pinned_memory,  // NOLINT
                         true,
                         "If set, allocate cpu pinned memory.");

PHI_DEFINE_EXPORTED_string(
    selected_xpus,
    "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (XPU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between XPU devices, use XPU_VISIBLE_DEVICES can only use"
    "share-memory only.");

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/cuda/cudnn_workspace_helper.h"
/**
 * CUDNN related FLAG
 * Name: FLAGS_conv_workspace_size_limit
 * Since Version: 0.13.0
 * Value Range: uint64, default=512 (MB)
 * Example:
 * Note: The internal function of cuDNN obtains the fastest matching algorithm
 *       within this memory limit. Usually, faster algorithms can be chosen in
 *       larger workspaces, but memory space can also be significantly
 * increased.
 *       Users need to balance memory and speed.
 */
PHI_DEFINE_EXPORTED_int64(conv_workspace_size_limit,
                          phi::backends::gpu::kDefaultConvWorkspaceSizeLimitMB,
                          "cuDNN convolution workspace limit in MB unit.");
#endif

PHI_DEFINE_EXPORTED_bool(use_stride_kernel,
                         true,
                         "Whether to use stride kernel if op support stride.");

PHI_DEFINE_EXPORTED_bool(
    log_memory_stats,
    false,
    "Log memory stats after each op runs, just used for debug.");

PHI_DEFINE_EXPORTED_int64(
    gpu_allocator_retry_time,
    10000,
    "The retry time (milliseconds) when allocator fails "
    "to allocate memory. No retry if this value is not greater than 0");

PHI_DEFINE_EXPORTED_bool(
    use_system_allocator,
    false,
    "Whether to use system allocator to allocate CPU and GPU memory. "
    "Only used for unittests.");

PHI_DEFINE_EXPORTED_bool(use_virtual_memory_auto_growth,
                         false,
                         "Use VirtualMemoryAutoGrowthBestFitAllocator.");

// NOTE(Ruibiao): This FLAGS is just to be compatible with
// the old single-stream CUDA allocator. It will be removed
// after StreamSafeCudaAllocator has been fully tested.
PHI_DEFINE_EXPORTED_bool(use_stream_safe_cuda_allocator,
                         true,
                         "Enable StreamSafeCUDAAllocator");

PHI_DEFINE_EXPORTED_bool(use_cuda_managed_memory,
                         false,
                         "Whether to use CUDAManagedAllocator to allocate "
                         "managed memory, only available for auto_growth "
                         "strategy");

PHI_DEFINE_EXPORTED_bool(
    use_auto_growth_v2,
    false,
    "Whether to use AutoGrowthBestFitAllocatorV2 for auto_growth "
    "strategy");

PHI_DEFINE_EXPORTED_READONLY_bool(
    free_idle_chunk,
    false,
    "Whether to free idle chunk when each allocation is freed. "
    "If false, all freed allocation would be cached to speed up next "
    "allocation request. If true, no allocation would be cached. This "
    "flag only works when FLAGS_allocator_strategy=auto_growth.");

PHI_DEFINE_EXPORTED_READONLY_bool(
    free_when_no_cache_hit,
    false,
    "Whether to free idle chunks when no cache hit. If true, idle "
    "chunk would be freed when no cache hit; if false, idle "
    "chunk would be freed when out of memory occurs. This flag "
    "only works when FLAGS_allocator_strategy=auto_growth.");

PHI_DEFINE_EXPORTED_READONLY_bool(print_allocator_trace_info,
                                  false,
                                  "print trace memory info");

PHI_DEFINE_EXPORTED_READONLY_bool(dump_chunk_info, false, "dump chunk info");

PHI_DEFINE_EXPORTED_bool(
    init_allocated_mem,
    false,
    "It is a mistake that the values of the memory allocated by "
    "BuddyAllocator are always zeroed in some op's implementation. "
    "To find this error in time, we use init_allocated_mem to indicate "
    "that initializing the allocated memory with a small value "
    "during unit testing.");

PHI_DEFINE_EXPORTED_bool(xpu_top_p_sampling_use_fp16,
                         false,
                         "use fp16 to improve the inference performance of "
                         "top_p_sampling xpu kernel");
PHI_DEFINE_EXPORTED_bool(
    xpu_top_p_sampling_heuristic_threshold,
    20,
    "threshold of heuristic method used for xpu_top_p_sampling, default 20; if "
    "heuristic_threshold = -1, xpu_top_p_sampling don't use heuristic method, "
    "and will fallback to normal top_p_sampling; if heuristic_threshold > 0, "
    "xpu_top_p_sampling will enable heuristic method");

#ifdef PADDLE_WITH_HIP

PHI_DEFINE_string(miopen_dir,
                  "",
                  "Specify path for loading libMIOpen.so. For instance, "
                  "/opt/rocm/miopen/lib. If empty [default], dlopen "
                  "will search miopen from LD_LIBRARY_PATH");

PHI_DEFINE_string(rocm_dir,
                  "",
                  "Specify path for loading rocm library, such as librocblas, "
                  "libmiopen, libhipsparse. For instance, /opt/rocm/lib. "
                  "If default, dlopen will search rocm from LD_LIBRARY_PATH");

PHI_DEFINE_string(rccl_dir,
                  "",
                  "Specify path for loading rccl library, such as librccl.so. "
                  "For instance, /opt/rocm/rccl/lib. If default, "
                  "dlopen will search rccl from LD_LIBRARY_PATH");
#endif

#ifdef PADDLE_WITH_XPU
PD_DEFINE_string(xpti_dir, "", "Specify path for loading libxpti.so.");
#endif

PD_DEFINE_int32(io_threadpool_size,
                100,
                "number of threads used for doing IO, default 100");

PD_DEFINE_int32(burning, 10, "Burning times.");
PD_DEFINE_int32(repeat, 3000, "Repeat times.");
PD_DEFINE_int32(max_size, 1000, "The Max size would be tested.");
PD_DEFINE_string(filter, "", "The Benchmark name would be run.");  // NOLINT

PD_DEFINE_double(acc, 1e-5, "Test accuracy threshold.");

PHI_DEFINE_bool(enable_host_event_recorder_hook,
                false,
                "enable HostEventRecorder, hook Profiler");

PHI_DEFINE_bool(enable_record_op_info,
                false,
                "enable operator supplement info recorder");

PHI_DEFINE_bool(dump_jitcode, false, "Whether to dump the jitcode to file");
