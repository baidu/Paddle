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

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle_infer_declare.h"  // NOLINT

///
/// \file paddle_pass_builder.h
///
/// \brief Class Paddle Pass Builder and its subclasses(pass strategies).
/// \section sec_intro Introduction
/// This class aims to build passes for paddle and define passes' strategies.
///
/// \author paddle-infer@baidu.com
/// \date 2020-3-23
/// \since 1.7

/// \namespace paddle
namespace paddle {

/// \class PaddlePassBuilder
/// \brief This class build passes based on vector<string> input. It is part of
/// inference API. Users can build passes, insert new passes, delete passes
/// using this class and its functions.
///
/// Example Usage:
///     Build a new pass.
/// \code{cpp}
/// const vector<string> passes(1, "conv_relu_onednn_fuse_pass");
/// PaddlePassBuilder builder(passes);
/// \endcode
class PD_INFER_DECL PaddlePassBuilder {
 public:
  /// \brief Constructor of the class. It stores the input passes.
  /// \param[in] passes passes' types.
  explicit PaddlePassBuilder(const std::vector<std::string> &passes)
      : passes_(passes) {}

  /// \brief Stores the input passes.
  /// \param[in] passes passes' types.
  void SetPasses(std::initializer_list<std::string> passes) {
    passes_ = passes;
  }

  /// \brief Append a pass to the end of the passes.
  /// \param[in] pass_type the type of the new pass.
  void AppendPass(const std::string &pass_type);

  /// \brief Insert a pass to a specific position.
  /// \param[in] idx the position to insert.
  /// \param[in] pass_type the type of insert pass.
  void InsertPass(size_t idx, const std::string &pass_type);

  /// \brief Delete the pass at certain position 'idx'.
  /// \param[in] idx the position to delete.
  void DeletePass(size_t idx);

  /// \brief Get the certain position of a pass.
  /// \param[in] pass_type the type of insert pass.
  size_t GetPassIndex(const std::string &pass_type);

  /// \brief Delete all passes that has a certain type 'pass_type'.
  /// \param[in] pass_type the certain pass type to be deleted.
  void DeletePass(const std::string &pass_type);

  /// \brief Delete all the passes.
  void ClearPasses();

  /// \brief Append an analysis pass.
  /// \param[in] pass the type of the new analysis pass.
  void AppendAnalysisPass(const std::string &pass);

  /// \brief Visualize the computation graph after each pass by generating a DOT
  /// language file, one can draw them with the Graphviz toolkit.
  void TurnOnDebug();
  /// \brief Human-readable information of the passes.
  std::string DebugString();

  /// \brief Get information of passes.
  /// \return Return list of the passes.
  const std::vector<std::string> &AllPasses() const { return passes_; }

  /// \brief Get information of analysis passes.
  /// \return Return list of analysis passes.
  std::vector<std::string> AnalysisPasses() const {
    auto passes = analysis_passes_;
    // To make sure the ir_graph_to_program should be the last pass so any
    // modification of IR will persist to the program.
    passes.push_back("ir_graph_to_program_pass");
    return passes;
  }

  const std::unordered_set<std::string> &GetAllDeletedPasses() const {
    return deleted_passes_;
  }

 protected:
  /// \cond Protected
  std::vector<std::string> analysis_passes_{{
      "ir_graph_build_pass",
      "ir_analysis_pass",
      "ir_params_sync_among_devices_pass",
      "adjust_cudnn_workspace_size_pass",
      "inference_op_replace_pass",
      "save_optimized_model_pass",
  }};
  std::vector<std::string> passes_;
  std::unordered_set<std::string> deleted_passes_;
  /// \endcond
};

/// \class PassStrategy
/// \brief This class defines the pass strategies like whether to use gpu/cuDNN
/// kernel/MKLDNN.
class PD_INFER_DECL PassStrategy : public PaddlePassBuilder {
 public:
  /// \brief Constructor of PassStrategy class. It works the same as
  /// PaddlePassBuilder class. \param[in] passes passes' types.
  explicit PassStrategy(const std::vector<std::string> &passes)
      : PaddlePassBuilder(passes) {}

  /// \brief Enable the use of cuDNN kernel.
  virtual void EnableCUDNN() {}

  /// \brief Enable the use of OneDNN.
  /// The OneDNN control exists in both CPU and GPU mode, because there can
  /// still be some CPU kernels running in GPU mode.
  virtual void EnableMKLDNN() {}

  /// \brief Disable the use of OneDNN.
  virtual void DisableMKLDNN() {}

  /// \brief Enable OneDNN bfloat16.
  virtual void EnableMkldnnBfloat16() {}

  /// \brief Enable OneDNN int8.
  virtual void EnableMkldnnInt8() {}

  /// \brief Disable OneDNN fc passes.
  virtual void DisableMkldnnFcPasses() {}

  /// \brief Check if we are using gpu.
  /// \return A bool variable implying whether we are in gpu mode.
  bool use_gpu() const { return use_gpu_; }

  /// \brief Check if we are using xpu.
  /// \return A bool variable implying whether we are in xpu mode.
  bool use_xpu() const { return use_xpu_; }

  /// \brief Check if we are using ipu.
  /// \return A bool variable implying whether we are in ipu mode.
  bool use_ipu() const { return use_ipu_; }

  /// \brief Check if we are using CustomDevice.
  /// \return A bool variable implying whether we are in CustomDevice mode.
  bool use_custom_device() const { return use_custom_device_; }

  /// \brief Default destructor.
  virtual ~PassStrategy() = default;

 protected:
  /// \cond Protected
  bool use_xpu_{false};
  bool use_gpu_{false};
  bool use_ipu_{false};
  bool use_mkldnn_{false};
  bool use_custom_device_{false};
  /// \endcond
};

/// \class CpuPassStrategy
/// \brief The CPU passes controller, it is used in AnalysisPredictor with CPU
/// mode.
class PD_INFER_DECL CpuPassStrategy : public PassStrategy {
 public:
  /// \brief Default constructor of CpuPassStrategy.
  CpuPassStrategy();

  /// \brief Construct by copying another CpuPassStrategy object.
  /// \param[in] other The CpuPassStrategy object we want to copy.
  explicit CpuPassStrategy(const CpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_gpu_ = other.use_gpu_;
    use_mkldnn_ = other.use_mkldnn_;
    use_mkldnn_bfloat16_ = other.use_mkldnn_bfloat16_;
    use_mkldnn_int8_ = other.use_mkldnn_int8_;
    disable_mkldnn_fc_passes_ = other.disable_mkldnn_fc_passes_;
    deleted_passes_ = other.deleted_passes_;
  }
  /// \brief Default destructor.
  virtual ~CpuPassStrategy() = default;

  /// \brief Enable the use of cuDNN kernel.
  void EnableCUDNN() override;

  /// \brief Enable the use of OneDNN.
  void EnableMKLDNN() override;

  /// \brief Disable the use of OneDNN.
  void DisableMKLDNN() override;

  /// \brief Enable OneDNN bfloat16.
  void EnableMkldnnBfloat16() override;

  /// \brief Enable OneDNN int8.
  void EnableMkldnnInt8() override;

  /// \brief Disable OneDNN fc passes.
  void DisableMkldnnFcPasses() override;

 protected:
  /// \brief Erase OneDNN fc passes.
  void EraseFcMkldnnPasses();

  /// \cond Protected
  bool use_mkldnn_bfloat16_{false};
  bool use_mkldnn_int8_{false};
  bool disable_mkldnn_fc_passes_{false};
  /// \endcond
};

/// \class GpuPassStrategy
/// \brief The GPU passes controller, it is used in AnalysisPredictor with GPU
/// mode.
class PD_INFER_DECL GpuPassStrategy : public PassStrategy {
 public:
  /// \brief Default constructor of GpuPassStrategy.
  GpuPassStrategy();

  /// \brief Construct by copying another GpuPassStrategy object.
  /// \param[in] other The GpuPassStrategy object we want to copy.
  explicit GpuPassStrategy(const GpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_gpu_ = true;
    use_cudnn_ = other.use_cudnn_;
    deleted_passes_ = other.deleted_passes_;
  }

  /// \brief Enable the use of cuDNN kernel.
  void EnableCUDNN() override;

  /// \brief Not supported in GPU mode yet.
  void EnableMKLDNN() override;

  /// \brief Not supported in GPU mode yet.
  void EnableMkldnnBfloat16() override;

  /// \brief Not supported in GPU mode yet.
  void EnableMkldnnInt8() override;

  /// \brief Disable OneDNN fc passes.
  void DisableMkldnnFcPasses() override;

  /// \brief Default destructor.
  virtual ~GpuPassStrategy() = default;

 protected:
  /// \cond Protected
  bool use_cudnn_{false};
  /// \endcond
};

/// \class XpuPassStrategy
/// \brief The XPU passes controller, it is used in AnalysisPredictor with XPU
/// mode.
class PD_INFER_DECL XpuPassStrategy final : public PassStrategy {
 public:
  XpuPassStrategy();
  explicit XpuPassStrategy(const XpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_xpu_ = true;
    deleted_passes_ = other.deleted_passes_;
  }
};

/// \class CustomDevicePassStrategy
/// \brief The CustomDevice passes controller, it is used in AnalysisPredictor
/// with CustomDevice
/// mode.
class PD_INFER_DECL CustomDevicePassStrategy final : public PassStrategy {
 public:
  CustomDevicePassStrategy() : PassStrategy({}) { use_custom_device_ = true; }

  /// \brief Construct by copying another CustomDevicePassStrategy object.
  /// \param[in] other The CustomDevicePassStrategy object we want to copy.
  explicit CustomDevicePassStrategy(const CustomDevicePassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_custom_device_ = true;
    deleted_passes_ = other.deleted_passes_;
  }
};

/// \class IpuPassStrategy
/// \brief The IPU passes controller, it is used in AnalysisPredictor with IPU
/// mode.
class PD_INFER_DECL IpuPassStrategy final : public PassStrategy {
 public:
  /// \brief Default constructor of IpuPassStrategy.
  IpuPassStrategy();

  /// \brief Construct by copying another IpuPassStrategy object.
  /// \param[in] other The IpuPassStrategy object we want to copy.
  explicit IpuPassStrategy(const IpuPassStrategy &other)
      : PassStrategy(other.AllPasses()) {
    use_ipu_ = true;
    deleted_passes_ = other.deleted_passes_;
  }
};
#ifdef PADDLE_WITH_OPENVINO
/// \brief List of OpenVINO subgraph passes.
PD_INFER_DECL extern const std::vector<std::string> kOVSubgraphPasses;
#endif

/// \brief List of tensorRT subgraph passes.
PD_INFER_DECL extern const std::vector<std::string> kTRTSubgraphPasses;

/// \brief List of cinn compiler passes.
PD_INFER_DECL extern const std::vector<std::string> kCINNCompilerPasses;

/// \brief TODO(inference): Most of the existing pass fusion operators do not
/// support fp16/bf16 precision, temporarily use low precision pass to prevent
/// running errors. After fusion operator supports low precision, delete this.
PD_INFER_DECL extern const std::vector<std::string> kGpuLowerPrecisionPasses;
PD_INFER_DECL extern const std::vector<std::string> kTrtLowerPrecisionPasses;

PD_INFER_DECL extern const std::vector<std::string> kPirGpuPasses;
PD_INFER_DECL extern const std::vector<std::string> kPirCpuPasses;
PD_INFER_DECL extern const std::vector<std::string> kPirXpuPasses;
PD_INFER_DECL extern const std::vector<std::string> kPirMkldnnPasses;
PD_INFER_DECL extern const std::vector<std::string> kPirMkldnnBf16Passes;

}  // namespace paddle
