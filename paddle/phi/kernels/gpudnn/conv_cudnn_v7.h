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

#pragma once

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/cuda/cuda_graph_with_memory_pool.h"
#include "paddle/phi/kernels/autotune/switch_autotune.h"
#include "paddle/phi/kernels/gpudnn/conv_gpudnn_base.h"

namespace phi {

using ConvArgs = ConvArgsBase<cudnnHandle_t, cudnnDataType_t>;

template <typename PerfT>
std::string GetPerfResultString(std::string prefix,
                                const std::vector<PerfT>& perf_results,
                                int actual_algo_count,
                                size_t workspace_limit) {
  std::ostringstream out;
  out << prefix << " (workspace limit=" << ToMegaBytes(workspace_limit)
      << " MB):\n";
  for (int i = 0; i < actual_algo_count; ++i) {
    const auto& result = perf_results[i];
    auto math_type_str = (result.mathType == CUDNN_TENSOR_OP_MATH) ? "T" : "F";
    out << "  algo=" << result.algo << ": tensor_core=" << math_type_str
        << ", time=" << result.time
        << " ms, memory=" << ToMegaBytes(result.memory)
        << " MB, status=" << result.status << "\n";
  }
  return out.str();
}

// Choose an algorithm which has the minimize time cost and less memory.
// NOTE: perf_results is ordered by time.
template <typename PerfT, typename AlgoT>
void ChooseAlgoByWorkspace(const std::vector<PerfT>& perf_results,
                           size_t workspace_limit,
                           SearchResult<AlgoT>* search_result) {
  int best_algo_idx = -1;
  for (size_t i = 0; i < perf_results.size(); ++i) {
    const auto& result = perf_results[i];
    if (result.status == CUDNN_STATUS_SUCCESS &&
        result.memory <= workspace_limit) {
      if (best_algo_idx == -1) {
        // The algorithm which has minimize time cost and need a workspace_size
        // fitting the workspace_limit constraint.
        best_algo_idx = i;
        // Each perf_results[i].time is set to be -1 in heuristic search.
        if (perf_results[best_algo_idx].time < 0) {
          break;
        }
      } else {
        // Compared to the next suboptimal algorithm, if the best one only has
        // 1% performance difference, we'd like to pick the one which need less
        // memory.
        if (result.time < 1.01 * perf_results[best_algo_idx].time) {
          best_algo_idx = (result.memory < perf_results[best_algo_idx].memory)
                              ? i
                              : best_algo_idx;
          break;
        }
      }
    }
  }
  if (best_algo_idx != -1) {
    const auto& result = perf_results[best_algo_idx];
    search_result->algo = result.algo;
    search_result->time = result.time;
    search_result->workspace_size = result.memory;
    auto math_type_str = (result.mathType == CUDNN_TENSOR_OP_MATH) ? "T" : "F";
    VLOG(3) << "Choose algo=" << result.algo
            << ", tensor_core=" << math_type_str << ", time=" << result.time
            << " ms, memory=" << ToMegaBytes(result.memory)
            << " MB, status=" << result.status;
  } else {
    VLOG(3) << "Can not find an algorithm that requires memory < "
            << ToMegaBytes(workspace_limit) << " MB";
  }
}

template <ConvKind CK>
struct SearchAlgorithmBase {};

// cuDNN convolution forward algorithm searcher, consisted of three searching
// modes, namely: deterministic, heuristic and exhaustive_search mode.
// As well as one workspace size acquirsition function with respect to
// the chosen algorithm.
template <>
struct SearchAlgorithmBase<ConvKind::kForward> {
  using PerfT = cudnnConvolutionFwdAlgoPerf_t;
  using AlgoT = cudnnConvolutionFwdAlgo_t;

  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvForward;

  static const std::string GetPerfName() { return "ConvForward"; }

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionFwdAlgo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionForwardWorkspaceSize(args.handle,
                                                              args.idesc.desc(),
                                                              args.wdesc.desc(),
                                                              args.cdesc.desc(),
                                                              args.odesc.desc(),
                                                              algo,
                                                              &workspace_size));
    return workspace_size;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const ConvArgs& args) {
    auto workspace_size = GetWorkspaceSize(args, static_cast<AlgoT>(1));
    return SearchResult<AlgoT>(static_cast<AlgoT>(1), -1.0, workspace_size);
  }

  // Heuristic search mode, calling the cudnnGetXxxAlgorithm.
  static SearchResult<AlgoT> FindAlgoHeuristic(const ConvArgs& args,
                                               const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int actual_perf_count;
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_FWD_ALGS);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionForwardAlgorithm_v7(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            kNUM_CUDNN_FWD_ALGS,
            &actual_perf_count,
            perf_results.data()));
    result.algo = perf_results[best_algo_idx].algo;
    result.workspace_size = perf_results[best_algo_idx].memory;

    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      VLOG(4) << GetPerfResultString<PerfT>("[Heuristic] FwdAlgo Perf result",
                                            perf_results,
                                            actual_perf_count,
                                            workspace_size_limit);
      // cudnnGetConvolutionForwardAlgorithm is removed in CUDNN-8
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(3) << "Fallback to non-v7 method to find conv algorithm "
                 "because the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnGetConvolutionForwardAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.wdesc.desc(),
              args.cdesc.desc(),
              args.odesc.desc(),
              CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit,
              &(result.algo)));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionForwardAlgorithm(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &(result.algo)));
#endif
    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const ConvArgs& args, const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    size_t max_workspace_size = GetMaxWorkspaceSize(args, workspace_size_limit);
    VLOG(4) << "max_workspace_size=" << ToMegaBytes(max_workspace_size)
            << " MB";

    int returned_algo_count;
    std::vector<PerfT> perf_results(kNUM_CUDNN_FWD_ALGS);
    auto cudnn_find_func = [&](void* workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnFindConvolutionForwardAlgorithmEx(
              args.handle,
              args.idesc.desc(),
              args.x->data<T>(),
              args.wdesc.desc(),
              args.w->data<T>(),
              args.cdesc.desc(),
              args.odesc.desc(),
              const_cast<T*>(args.o->data<T>()),
              kNUM_CUDNN_FWD_ALGS,
              &returned_algo_count,
              perf_results.data(),
              workspace_ptr,
              max_workspace_size));
    };

    auto workspace_handle = ctx.cudnn_workspace_handle();
    workspace_handle.RunFuncSync(
        cudnn_find_func, max_workspace_size, UseFixedWorkspace());

    VLOG(4) << GetPerfResultString<PerfT>(
        "[Exhaustive Search] FwdAlgo Perf result",
        perf_results,
        returned_algo_count,
        workspace_size_limit);
    ChooseAlgoByWorkspace<PerfT, AlgoT>(
        perf_results, workspace_size_limit, &result);

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_FWD_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status = phi::dynload::cudnnGetConvolutionForwardWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.wdesc.desc(),
            args.cdesc.desc(),
            args.odesc.desc(),
            static_cast<cudnnConvolutionFwdAlgo_t>(algo),
            &workspace_size);
        if (status == CUDNN_STATUS_SUCCESS &&
            workspace_size <= workspace_size_limit) {
          max_workspace_size = std::max(workspace_size, max_workspace_size);
        }
      }
      return max_workspace_size;
    } else {
      return workspace_size_limit;
    }
  }
};

// cuDNN convolution backward data-algorithm searcher, consisting of three
// searching modes, namely: deterministic, heuristic, and exhaustive_search
// mode. Specially, there are 2 patterns of exhaustive search mode, one for
// HALF precision only, one for the rest.
// As well as one workspace size acquisition function with
// respect to the chosen algorithm.
template <>
struct SearchAlgorithmBase<ConvKind::kBackwardData> {
  using PerfT = cudnnConvolutionBwdDataAlgoPerf_t;
  using AlgoT = cudnnConvolutionBwdDataAlgo_t;

  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvBackwardData;

  static const std::string GetPerfName() { return "ConvBackwardData"; }

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionBwdDataAlgo_t algo) {
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const ConvArgs& args) {
    auto workspace_size =
        GetWorkspaceSize(args, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
    return SearchResult<AlgoT>(
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, -1.0, workspace_size);
  }

  static SearchResult<AlgoT> FindAlgoHeuristic(const ConvArgs& args,
                                               const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int actual_perf_count;
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_DATA_ALGS);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataAlgorithm_v7(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            kNUM_CUDNN_BWD_DATA_ALGS,
            &actual_perf_count,
            perf_results.data()));
    result.algo = perf_results[best_algo_idx].algo;

#if CUDNN_VERSION < 7500
    int stride_dim = args.x->dims().size() - 2;
    bool blacklist = std::any_of(args.s.begin(),
                                 args.s.begin() + stride_dim,
                                 [=](int n) { return n != 1; });
    if (blacklist && (perf_results[best_algo_idx].algo ==
                          CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                      perf_results[best_algo_idx].algo ==
                          CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)) {
      result.algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    }
#endif
    result.workspace_size = GetWorkspaceSize(args, result.algo);
    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionBackwardDataAlgorithm is removed in CUDNN-8
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(1) << "Fallback to non-v7 method to find conv algorithm because "
                 "the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
              args.handle,
              args.wdesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.idesc.desc(),
              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit,
              &(result.algo)));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
            args.handle,
            args.wdesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.idesc.desc(),
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &(result.algo)));
#endif
    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const ConvArgs& args, const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    size_t max_workspace_size = GetMaxWorkspaceSize(args, workspace_size_limit);
    VLOG(3) << "max_workspace_size=" << ToMegaBytes(max_workspace_size)
            << " MB";

    int returned_algo_count;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_DATA_ALGS);
    auto cudnn_find_func = [&](void* workspace_ptr) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnFindConvolutionBackwardDataAlgorithmEx(
              args.handle,
              args.wdesc.desc(),
              args.w->data<T>(),
              args.odesc.desc(),
              args.o->data<T>(),
              args.cdesc.desc(),
              args.idesc.desc(),
              const_cast<T*>(args.x->data<T>()),
              kNUM_CUDNN_BWD_DATA_ALGS,
              &returned_algo_count,
              perf_results.data(),
              workspace_ptr,
              max_workspace_size));
    };

    auto workspace_handle = ctx.cudnn_workspace_handle();
    workspace_handle.RunFuncSync(
        cudnn_find_func, max_workspace_size, UseFixedWorkspace());

    VLOG(4) << GetPerfResultString<PerfT>(
        "[Exhaustive Search] BwdDataAlgo Perf result",
        perf_results,
        returned_algo_count,
        workspace_size_limit);
    ChooseAlgoByWorkspace<PerfT, AlgoT>(
        perf_results, workspace_size_limit, &result);

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_BWD_DATA_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status =
            phi::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
                args.handle,
                args.wdesc.desc(),
                args.odesc.desc(),
                args.cdesc.desc(),
                args.idesc.desc(),
                static_cast<cudnnConvolutionBwdDataAlgo_t>(algo),
                &workspace_size);
        if (status == CUDNN_STATUS_SUCCESS &&
            workspace_size <= workspace_size_limit) {
          max_workspace_size = std::max(workspace_size, max_workspace_size);
        }
      }
      return max_workspace_size;
    } else {
      return workspace_size_limit;
    }
  }
};

// cuDNN convution backward filter-algorithm searcher, consisted of three
// algorithm searching modes, namely: deterministic, heuristic, and
// exhaustive_search mode. As well as one workspace size acquirsition function
// with respect to the chosen algorithm.
template <>
struct SearchAlgorithmBase<ConvKind::kBackwardFilter> {
  using PerfT = cudnnConvolutionBwdFilterAlgoPerf_t;
  using AlgoT = cudnnConvolutionBwdFilterAlgo_t;

  constexpr static phi::autotune::AlgorithmType kAlgoType =
      phi::autotune::AlgorithmType::kConvBackwardFilter;

  static const std::string GetPerfName() { return "ConvBackwardFilter"; }

  static size_t GetWorkspaceSize(const ConvArgs& args,
                                 cudnnConvolutionBwdFilterAlgo_t algo) {
    phi::backends::gpu::CUDAGraphCaptureModeGuard guard;
    size_t workspace_size = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            algo,
            &workspace_size));
    return workspace_size;
  }

 protected:
  static SearchResult<AlgoT> FindAlgoDeterministic(const ConvArgs& args) {
    auto workspace_size =
        GetWorkspaceSize(args, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
    return SearchResult<AlgoT>(
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, -1.0, workspace_size);
  }

  static SearchResult<AlgoT> FindAlgoHeuristic(const ConvArgs& args,
                                               const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());

#if CUDNN_VERSION >= 7001
    int actual_perf_count;
    int best_algo_idx = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_FILTER_ALGS);
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            kNUM_CUDNN_BWD_FILTER_ALGS,
            &actual_perf_count,
            perf_results.data()));
    result.algo = perf_results[best_algo_idx].algo;
    result.workspace_size = perf_results[best_algo_idx].memory;

    if (result.workspace_size > workspace_size_limit) {
#if CUDNN_VERSION >= 8000
      // cudnnGetConvolutionBackwardFilterAlgorithm is removed in CUDNN-8
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
#else
      VLOG(1) << "Fallback to non-v7 method to find conv algorithm because "
                 "the workspace size request("
              << result.workspace_size << ") exceeds the limit("
              << workspace_size_limit << ")";
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
              workspace_size_limit,
              &(result.algo)));
#endif
    }
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithm(
            args.handle,
            args.idesc.desc(),
            args.odesc.desc(),
            args.cdesc.desc(),
            args.wdesc.desc(),
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            workspace_size_limit,
            &(result.algo)));
#endif

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  template <typename T>
  static SearchResult<AlgoT> FindAlgoExhaustiveSearch(
      const ConvArgs& args, const phi::GPUContext& ctx) {
    SearchResult<AlgoT> result;
    int returned_algo_count = 0;
    std::vector<PerfT> perf_results(kNUM_CUDNN_BWD_FILTER_ALGS);
    size_t workspace_size_limit =
        CalcWorkspaceLimitInBytes(UseFixedWorkspace());
    auto workspace_handle = ctx.cudnn_workspace_handle();
    if (phi::backends::gpu::CudnnDataType<T>::type != CUDNN_DATA_HALF) {
      size_t max_workspace_size =
          GetMaxWorkspaceSize(args, workspace_size_limit);
      VLOG(3) << "max_workspace_size=" << ToMegaBytes(max_workspace_size)
              << " MB";

      auto cudnn_find_func = [&](void* workspace_ptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnFindConvolutionBackwardFilterAlgorithmEx(
                args.handle,
                args.idesc.desc(),
                args.x->data<T>(),
                args.odesc.desc(),
                args.o->data<T>(),
                args.cdesc.desc(),
                args.wdesc.desc(),
                const_cast<T*>(args.w->data<T>()),
                kNUM_CUDNN_BWD_FILTER_ALGS,
                &returned_algo_count,
                perf_results.data(),
                workspace_ptr,
                max_workspace_size));
      };
      workspace_handle.RunFuncSync(
          cudnn_find_func, max_workspace_size, UseFixedWorkspace());

      VLOG(4) << GetPerfResultString<PerfT>(
          "[Exhaustive Search] BwdFilterAlgo Perf result",
          perf_results,
          returned_algo_count,
          workspace_size_limit);
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
    } else {
      int max_algos = GetAlgorithmMaxCount(args.handle);
      std::vector<PerfT> perf_results(max_algos);
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnFindConvolutionBackwardFilterAlgorithm(
              args.handle,
              args.idesc.desc(),
              args.odesc.desc(),
              args.cdesc.desc(),
              args.wdesc.desc(),
              perf_results.size(),
              &returned_algo_count,
              perf_results.data()));
      perf_results.resize(returned_algo_count);

      VLOG(4) << GetPerfResultString<PerfT>(
          "[Exhaustive Search] BwdFilterAlgo Perf result",
          perf_results,
          perf_results.size(),
          workspace_size_limit);
      ChooseAlgoByWorkspace<PerfT, AlgoT>(
          perf_results, workspace_size_limit, &result);
    }

    result.workspace_size = GetWorkspaceSize(args, result.algo);
    return result;
  }

  static int GetAlgorithmMaxCount(cudnnHandle_t handle) {
#if CUDNN_VERSION_MIN(7, 0, 1)
    int max_algos = 0;
    auto status =
        phi::dynload::cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
            handle, &max_algos);
    if (status == gpuSuccess) {
      VLOG(5) << "[BackwardFilter] max_algos: predefined="
              << kNUM_CUDNN_BWD_FILTER_ALGS << ", actual=" << max_algos;
      return max_algos;
    }
#endif
    return kNUM_CUDNN_BWD_FILTER_ALGS;
  }

  static size_t GetMaxWorkspaceSize(const ConvArgs& args,
                                    size_t workspace_size_limit) {
    if (!UseFixedWorkspace()) {
      size_t max_workspace_size = 0;
      for (size_t algo = 0; algo < kNUM_CUDNN_BWD_FILTER_ALGS; ++algo) {
        size_t workspace_size = 0;
        auto status =
            phi::dynload::cudnnGetConvolutionBackwardFilterWorkspaceSize(
                args.handle,
                args.idesc.desc(),
                args.odesc.desc(),
                args.cdesc.desc(),
                args.wdesc.desc(),
                static_cast<cudnnConvolutionBwdFilterAlgo_t>(algo),
                &workspace_size);
        if (status == CUDNN_STATUS_SUCCESS &&
            workspace_size <= workspace_size_limit) {
          max_workspace_size = std::max(workspace_size, max_workspace_size);
        }
      }
      return max_workspace_size;
    } else {
      return workspace_size_limit;
    }
  }
};

template <ConvKind CK>
struct SearchAlgorithm : public SearchAlgorithmBase<CK> {
  using AlgoT = typename SearchAlgorithmBase<CK>::AlgoT;

  template <typename T>
  static SearchResult<AlgoT> Find(const phi::GPUContext& ctx,
                                  const ConvArgs& args,
                                  bool exhaustive_search,
                                  bool deterministic,
                                  bool enable_autotune = true) {
    SearchResult<AlgoT> result;
    bool use_autotune = false;
    auto dtype = phi::backends::gpu::CudnnDataType<T>::type;
    SetConvMathType(ctx, dtype, args.cdesc);

    if (deterministic) {
      result = SearchAlgorithmBase<CK>::FindAlgoDeterministic(args);
    } else {
      // 1. Once turning on exhaustive FLAGS, always get exhaustive_search.
      // 2. Once turning on auto-tune, run heuristic (default) before
      //    auto-tune process, run exhaustive_search during mentioned process.
      //    Auto tune is only enabled between specified range.
      // 3. After auto-tune process, run cached algorithm if cached, run
      //    default mode for the rest.
      auto key = args.ConvertToConvCacheKey<T>();
      auto& cache = phi::autotune::AutoTuneCache::Instance().GetConv(
          SearchAlgorithmBase<CK>::kAlgoType);
      bool find_in_cache = cache.Find(key);
      if (find_in_cache) {
        auto t = cache.Get(key);
        result.algo = static_cast<AlgoT>(t.algo);
        result.workspace_size = t.workspace_size;
        result.exhaustive_search = t.exhaustive_search;
      }
      if (!result.exhaustive_search) {
        // In conv2d_transpose, enable_autotune is set to false because some
        // algorithm picked by exhaustive search method produce wrong result.
        use_autotune = enable_autotune &&
                       phi::autotune::AutoTuneStatus::Instance().UseAutoTune();
        if (exhaustive_search || use_autotune) {
          // Once autotune is enabled, the autotuned result can rewrite the
          // previous result in cache found by heuristic method.
          result =
              SearchAlgorithmBase<CK>::template FindAlgoExhaustiveSearch<T>(
                  args, ctx);
          cache.Set(key,
                    phi::autotune::ConvAutoTuneResult(
                        static_cast<int64_t>(result.algo),
                        result.workspace_size,
                        true));
        } else if (!find_in_cache) {
          result = SearchAlgorithmBase<CK>::FindAlgoHeuristic(args, ctx);
          cache.Set(key,
                    phi::autotune::ConvAutoTuneResult(
                        static_cast<int64_t>(result.algo),
                        result.workspace_size,
                        false));
        }
      }
    }
    VLOG(3) << "[cuDNN " << SearchAlgorithmBase<CK>::GetPerfName()
            << "] exhaustive_search=" << exhaustive_search
            << ", use_autotune=" << use_autotune
            << ", deterministic=" << deterministic
            << ", choose algo=" << result.algo
            << ", workspace=" << ToMegaBytes(result.workspace_size) << " MB";
    return result;
  }

  static void SetConvMathType(
      const phi::GPUContext& ctx,
      cudnnDataType_t dtype,
      const phi::backends::gpu::ConvolutionDescriptor& cdesc) {
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    if (ctx.GetComputeCapability() >= 70 && dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_TENSOR_OP_MATH));
      VLOG(5) << "Enable Tensor Core for FLOAT16";
#if CUDA_VERSION >= 11000
#if CUDNN_VERSION_MIN(8, 1, 0)
    } else if (ctx.GetComputeCapability() >= 80 &&
               dtype == CUDNN_DATA_BFLOAT16) {
      VLOG(5) << "Enable Tensor Core for BFLOAT16";
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_TENSOR_OP_MATH));
#endif  // CUDNN_VERSION_MIN(8, 1, 0)
    } else if (dtype == CUDNN_DATA_FLOAT && !cdesc.allow_tf32_) {
      VLOG(5) << "Disable TensorFloat (Tensor Core) for FLOAT";
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_FMA_MATH));
#endif  // CUDA_VERSION >= 11000
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetConvolutionMathType(
          cdesc.desc(), CUDNN_DEFAULT_MATH));
    }
#endif
  }
};

template <typename T, ConvKind CK>
struct ConvRunner {};

template <typename T>
struct ConvRunner<T, ConvKind::kForward> {
  static void Apply(
      const phi::GPUContext& ctx,
      const ConvArgs& args,
      const SearchResult<cudnnConvolutionFwdAlgo_t>& search_result,
      const T* input_ptr,
      const T* filter_ptr,
      T* output_ptr,
      int groups,
      int group_offset_in,
      int group_offset_filter,
      int group_offset_out,
      size_t workspace_size,
      phi::DnnWorkspaceHandle* workspace_handle,
      bool use_addto = false) {
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

    auto cudnn_handle = ctx.cudnn_handle();
    for (int i = 0; i < groups; i++) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnConvolutionForward(
                cudnn_handle,
                &alpha,
                args.idesc.desc(),
                input_ptr + i * group_offset_in,
                args.wdesc.desc(),
                filter_ptr + i * group_offset_filter,
                args.cdesc.desc(),
                search_result.algo,
                workspace_ptr,
                workspace_size,
                &beta,
                args.odesc.desc(),
                output_ptr + i * group_offset_out));
          },
          workspace_size);
    }
  }
};

template <typename T>
struct ConvRunner<T, ConvKind::kBackwardData> {
  static void Apply(
      const phi::GPUContext& ctx,
      const ConvArgs& args,
      const SearchResult<cudnnConvolutionBwdDataAlgo_t>& search_result,
      const T* output_grad_ptr,
      const T* filter_ptr,
      T* input_grad_ptr,
      int groups,
      int group_offset_in,
      int group_offset_filter,
      int group_offset_out,
      size_t workspace_size,
      phi::DnnWorkspaceHandle* workspace_handle,
      bool use_addto = false) {
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

    auto cudnn_handle = ctx.cudnn_handle();
    for (int i = 0; i < groups; i++) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::cudnnConvolutionBackwardData(
                    cudnn_handle,
                    &alpha,
                    args.wdesc.desc(),
                    filter_ptr + i * group_offset_filter,
                    args.odesc.desc(),
                    output_grad_ptr + i * group_offset_out,
                    args.cdesc.desc(),
                    search_result.algo,
                    workspace_ptr,
                    workspace_size,
                    &beta,
                    args.idesc.desc(),
                    input_grad_ptr + i * group_offset_in));
          },
          workspace_size);
    }
  }
};

template <typename T>
struct ConvRunner<T, ConvKind::kBackwardFilter> {
  static void Apply(
      const phi::GPUContext& ctx,
      const ConvArgs& args,
      const SearchResult<cudnnConvolutionBwdFilterAlgo_t>& search_result,
      const T* output_grad_ptr,
      const T* input_ptr,
      T* filter_grad_ptr,
      int groups,
      int group_offset_in,
      int group_offset_filter,
      int group_offset_out,
      size_t workspace_size,
      phi::DnnWorkspaceHandle* workspace_handle,
      bool use_addto = false) {
    ScalingParamType<T> alpha = 1.0f;
    ScalingParamType<T> beta = use_addto ? 1.0f : 0.0f;

    auto cudnn_handle = ctx.cudnn_handle();
    for (int i = 0; i < groups; i++) {
      workspace_handle->RunFunc(
          [&](void* workspace_ptr) {
            PADDLE_ENFORCE_GPU_SUCCESS(
                phi::dynload::cudnnConvolutionBackwardFilter(
                    cudnn_handle,
                    &alpha,
                    args.idesc.desc(),
                    input_ptr + i * group_offset_in,
                    args.odesc.desc(),
                    output_grad_ptr + i * group_offset_out,
                    args.cdesc.desc(),
                    search_result.algo,
                    workspace_ptr,
                    workspace_size,
                    &beta,
                    args.wdesc.desc(),
                    filter_grad_ptr + i * group_offset_filter));
          },
          workspace_size);
    }
  }
};

}  // namespace phi
