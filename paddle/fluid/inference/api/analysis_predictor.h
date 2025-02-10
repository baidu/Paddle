// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_compatible_info.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/resource_manager.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/platform/device/gpu/gpu_types.h"
#include "paddle/utils/string/printf.h"

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/program.h"

namespace paddle_infer {
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
namespace experimental {
class InternalUtils;
};
}  // namespace paddle_infer
///
/// \file analysis_predictor.h
///
/// \brief Compared to NativePredictor, AnalysisPredictor is a high-performance
/// predictor that includes many optimizations
///
/// \author paddle-infer@baidu.com
/// \date 2020-01-01
/// \since 1.7.0
///

namespace paddle {

using framework::NaiveExecutor;
using framework::proto::ProgramDesc;
using inference::analysis::Analyzer;
using inference::analysis::Argument;

///
/// \class AnalysisPredictor
///
/// \brief The analysis predictor is based on the original native predictor with
/// IR and Analysis support. It will optimize IR and Parameters in the runtime.
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePaddlePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output tensors
/// \code{cpp}
///   auto input_t = predictor->GetInputTensor(input_names[0]);
///   auto output_t = predictor->GetOutputTensor(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->ZeroCopyRun();
/// \endcode
///
class AnalysisPredictor : public PaddlePredictor {
 public:
  ///
  /// \brief Construct a new Analysis Predictor object
  ///
  /// \param[in] AnalysisConfig config
  ///
  explicit AnalysisPredictor(const AnalysisConfig &config);
  ///
  /// \brief Destroy the Analysis Predictor object
  ///
  ~AnalysisPredictor();

  ///
  /// \brief Initialize predictor
  ///
  /// Initializing predictor mainly includes the following tasks:
  /// preparing scope, creating executor, preparing program, initializing the
  /// variables required by the executor, getting the feed_target_names and
  /// fetch_target_names, etc.
  ///
  /// \param[in] parent_scope parent scope
  /// \param[in] program program
  /// \return Whether the init function executed successfully
  ///
  bool Init(const std::shared_ptr<framework::Scope> &parent_scope,
            const std::shared_ptr<framework::ProgramDesc> &program = nullptr);

  ///
  /// \brief Run the prediction engine. Deprecated. Please refer to ZeroCopyRun
  ///
  /// \param[in] inputs input tensors
  /// \param[out] output_data output tensors
  /// \param[in] batch_size data's batch size
  /// \return Whether the function executed successfully
  ///
  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  ///
  /// \brief Run the prediction engine (Recommended).
  ///
  /// \param[in] inputs input tensors
  /// \param[out] outputs output tensors
  /// \return Whether the function executed successfully
  ///
  bool Run(const std::vector<paddle::Tensor> &inputs,
           std::vector<paddle::Tensor> *outputs) override;

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames() override;
  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  std::vector<std::string> GetOutputNames() override;

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;
  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name output name
  /// \return output tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;
  ///
  /// \brief Get all input names and their corresponding shapes
  ///
  /// \return the map of input names and shapes
  ///
  std::map<std::string, std::vector<int64_t>> GetInputTensorShape() override;
  ///
  /// \brief Get all input names and their corresponding type
  ///
  /// \return the map of input names and type
  ///
  std::map<std::string, paddle_infer::DataType> GetInputTypes() override;
  ///
  /// \brief Get all output names and their corresponding shapes
  ///
  /// \return the map of output names and shapes
  ///
  std::map<std::string, std::vector<int64_t>> GetOutputTensorShape() override;
  ///
  /// \brief Get all output names and their corresponding type
  ///
  /// \return the map of output names and type
  ///
  std::map<std::string, paddle_infer::DataType> GetOutputTypes() override;

  ///
  /// \brief Run the prediction engine
  ///
  /// \param switch_stream Whether the stream is switched
  /// \return Whether the function executed successfully
  ///
  bool ZeroCopyRun(bool switch_stream = false) override;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // Note: Can only be used under thread_local semantics.
  bool ExpRunWithExternalStream(const gpuStream_t stream);
#endif

  // Note: Can only be used under thread_local semantics.
  bool ExpRunWithExternalStream(void *stream);

  // Note: Can only be used under thread_local semantics.
  bool ExpRunWithRuntimeConfig(void *config);

  ///
  /// \brief Get the execution stream on devices with a concept of stream,
  /// otherwise returns nullptr.
  ///
  /// \return The execution stream or nullptr (CPU).
  ///
  void *GetExecStream() const override;

  ///
  /// \brief Create feed fetch variables
  ///
  /// \param[in] scope Scope needed to create variables
  ///
  void CreateFeedFetchVar(framework::Scope *scope);
  ///
  /// \brief Determine the model's inputs and outputs based on the program's
  /// feed fetch op
  ///
  void PrepareFeedFetch();

  ///
  /// \brief Set predictor's argument according to config, which mainly includes
  /// execution information and graph optimization related pass information
  ///
  void PrepareArgument();
  ///
  /// \brief According to argument information, execute the relevant pass
  /// to get the optimized model program
  ///
  void OptimizeInferenceProgram();
  ///
  /// \brief According to argument information, execute the relevant pass
  /// to get the optimized model program
  ///
  void OptimizeInferencePirProgram();
  ///
  /// \brief Clear the intermediate tensors of the predictor
  ///
  ///
  void ClearIntermediateTensor() override;

  ///
  /// \brief Release all tmp tensor to compress the size of the memory pool.
  /// The memory pool is considered to be composed of a list of chunks, if
  /// the chunk is not occupied, it can be released.
  ///
  /// \return Number of bytes released. It may be smaller than the actual
  /// released memory, because part of the memory is not managed by the
  /// MemoryPool.
  ///
  uint64_t TryShrinkMemory() override;

  ///
  /// \brief Get the argument used by predictor
  ///
  /// \return the argument obtained by config
  ///
  Argument &analysis_argument() { return *argument_; }
  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  std::unique_ptr<PaddlePredictor> Clone(void *stream = nullptr) override;
  ///
  /// \brief Get the scope used by predictor
  ///
  /// \return scope
  ///
  framework::Scope *scope() { return scope_.get(); }
  ///
  /// \brief Get the inference program
  ///
  /// \return the inference program
  ///
  framework::ProgramDesc &program() { return *inference_program_; }

  ///
  /// \brief Get the serialized program
  ///
  /// \return the serialized program
  ///
  std::string GetSerializedProgram() const override;

  ///
  /// \brief Get the fusion_statis_t
  ///
  /// \return the fusion_statis_t
  ///
  Argument::fusion_statis_t fusion_statis() { return fusion_statis_; }

  ///
  /// \brief Register a output hook function to operate the intermediate tensor
  /// of op output. when using this function, memory reuse should be turned off.
  /// The hook function signature is void(const std::string&, const
  /// std::string&, const paddle::Tensor&>). Here, the first parameter is op's
  /// type, the second param is output var name of the op, and the third
  /// parameter is output tensor with the var name.
  ///
  void RegisterOutputHook(const OutputTensorHookFunc &hookfunc) override;

  /// \brief Same as RegisterOutputHook
  void RegisterInputHook(const InputTensorHookFunc &hookfunc) override;

  ///
  /// \brief Initialize onednn quantizer and execute onednn quantization pass
  ///
  /// \return Whether the function executed successfully
  ///
  bool MkldnnQuantize();

 protected:
  ///
  /// \brief Prepare predictor's required programs, including loading model
  /// information, graph optimization, and executor creation variables, etc.
  ///
  /// \param[in] program paddle program
  /// \return Whether the function executed successfully
  ///
  bool PrepareProgram(const std::shared_ptr<framework::ProgramDesc> &program);
  ///
  /// \brief Prepare predictor's required programs, including loading model
  /// information, graph optimization, and executor creation variables, etc.
  ///
  /// \return Whether the function executed successfully
  ///
  bool PreparePirProgram();
  ///
  /// \brief Prepare scope environment, each predictor has its own scope
  ///
  /// \param[in] parent_scope The scope of the predictor to be cloned, or null
  /// \return Whether the function executed successfully
  ///
  bool PrepareScope(const std::shared_ptr<framework::Scope> &parent_scope);
  ///
  /// \brief Create an Executor object
  ///
  /// \return Whether the function executed successfully
  ///
  bool CreateExecutor();
  ///
  /// \brief According to the model's program, the executor creates ops
  ///
  /// \return Whether the function executed successfully
  ///
  bool PrepareExecutor();

  ///
  /// \brief Load model program.
  ///
  /// \return Whether the function executed successfully
  ///
  bool LoadProgramDesc();
  ///
  /// \brief Load model parameters.
  ///
  /// \return Whether the function executed successfully
  ///
  bool LoadParameters();

  ///
  /// \brief Save or Load pir model parameters.
  ///
  /// \return Whether the function executed successfully
  ///
  bool SaveOrLoadPirParameters(bool for_save);

  ///
  /// \brief Prepare input data, only used in Run()
  ///
  /// \param[in] input_datas input tensors
  /// \param[in] scope the scope used by predictor
  /// \return Whether the function executed successfully
  ///
  bool SetFeed(const std::vector<PaddleTensor> &input_datas,
               framework::Scope *scope);

  ///
  /// \brief Prepare input data, only used in Run()
  ///
  /// \param[in] inputs input tensors
  /// \param[in] scope the scope used by predictor
  /// \return Whether the function executed successfully
  ///
  bool SetFeed(const std::vector<paddle::Tensor> &inputs,
               framework::Scope *scope);

  ///
  /// \brief Get the output data, only used in Run()
  ///
  /// \param[out] output_data output tensors
  /// \param[in] scope the scope used by predictor
  /// \return Whether the function executed successfully
  ///
  bool GetFetch(std::vector<PaddleTensor> *output_data,
                framework::Scope *scope);

  ///
  /// \brief Get the output data, only used in Run()
  ///
  /// \param[out] outputs output tensors
  /// \param[in] scope the scope used by predictor
  /// \return Whether the function executed successfully
  ///
  bool GetFetch(std::vector<paddle::Tensor> *outputs, framework::Scope *scope);

  ///
  /// \brief Get the output data, only used in GetFetch()
  ///
  /// \param[in] tensor for fetch op
  /// \param[out] output_data output tensor
  ///
  template <typename T>
  void GetFetchOne(const phi::DenseTensor &fetches, PaddleTensor *output_data);
  ///
  /// \brief PreSet for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  /// \param[in] inputs tensors
  ///
  void MkldnnPreSet(const std::vector<PaddleTensor> &inputs);
  ///
  /// \brief PreSet for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run().
  ///
  /// \param[in] inputs tensors
  ///
  void MkldnnPreSet(const std::vector<paddle::Tensor> &inputs);

  ///
  /// \brief PreSet for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  /// \param[in] inputs tensor shape
  ///
  void MkldnnPreSet(const std::vector<std::vector<int>> &inputs_shape);

  ///
  /// \brief PostReset for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  void MkldnnPostReset();

#ifdef PADDLE_WITH_TENSORRT
  ///
  /// \brief save calibration table
  ///
  /// When we use Paddle-TRT INT8 engine, we need to generate calibration table
  /// data first,
  /// the calibration table contains the range for each op's input and output,
  /// this whole process can be divided into several steps:
  /// 1. Builds a 32-bit engine, runs it on the calibration set, and records a
  ///  histogram for each tensor of the distribution of activation values.
  /// 2. Builds a calibration table from the histograms.
  /// After step 2, we need to store the calibration table on disk.
  ///
  /// \return Whether the function executed successfully
  ///
  bool SaveTrtCalibToDisk();
#endif

// Some more detailed tests, they are made the friends of the predictor, so that
// the all the details can be tested.
#if PADDLE_WITH_TESTING
  FRIEND_TEST(AnalysisPredictor, analysis_off);
  FRIEND_TEST(AnalysisPredictor, analysis_on);
  FRIEND_TEST(AnalysisPredictor, with_gpu);
#endif

 protected:
  const void *GetDeviceContexts() const override;

 private:
  void StatisticShapeRangeInfo();
  void HookCollectShapeRangeInfo();
  void InitPlace();
  void InitDeviceContexts();
  void InitResourceManager(void *stream);
  std::string GetOptimizedModelPath();
  void ClearExtraParams();

 private:
  AnalysisConfig config_;
  std::unique_ptr<Argument> argument_ = nullptr;
  Argument::fusion_statis_t fusion_statis_;
  std::unique_ptr<NaiveExecutor> executor_;
  phi::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_{nullptr};
  std::shared_ptr<framework::ProgramDesc> inference_program_;
  std::shared_ptr<pir::Program> pir_program_;
  bool load_pir_model_{false};
  std::vector<framework::OpDesc *> feeds_;
  std::vector<pir::Operation *> pir_feeds_;
  std::map<std::string, size_t> feed_names_;
  // Sorted according to the idx.
  std::map<size_t, std::string> idx2feeds_;
  std::map<std::string, std::vector<int64_t>> feed_name2shapes_;
  std::vector<framework::OpDesc *> fetches_;
  std::vector<pir::Operation *> pir_fetches_;
  std::map<size_t, std::string> idx2fetches_;
  std::map<std::string, std::vector<int64_t>> fetch_name2shapes_;

  phi::DataType model_precision_{phi::DataType::FLOAT32};

  // Memory buffer for feed inputs. The temporary DenseTensor will cause serious
  // concurrency problems, wrong results and memory leak, so cache them.
  std::vector<phi::DenseTensor> feed_tensors_;
  details::TensorArrayBatchCleaner tensor_array_batch_cleaner_;
  // A mutex help to make Clone thread safe.
  std::mutex clone_mutex_;
  static int clone_num_;

  int predictor_id_;
  int root_predictor_id_{-1};

 private:
  std::once_flag register_input_hook_flag_;
  std::once_flag register_output_hook_flag_;
  std::vector<OutputTensorHookFunc> output_hookfuncs_;
  std::vector<InputTensorHookFunc> input_hookfuncs_;
  // Some status here that help to determine the status inside the predictor.
  bool status_is_cloned_{false};

  std::map<std::string, std::vector<std::vector<int32_t>>> shape_info_;
  std::map<std::string, std::vector<std::vector<int32_t>>> shape_tensor_value_;

  bool private_context_{false};
  void *predictor_stream_{nullptr};
  std::map<phi::Place, std::shared_future<std::unique_ptr<phi::DeviceContext>>>
      device_contexts_;

  friend class paddle_infer::experimental::InternalUtils;
};

}  // namespace paddle
