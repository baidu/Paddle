// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h"

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/errors.h"
#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/cinn/runtime/cinn_runtime.h"
#endif
PD_DECLARE_bool(cinn_measure_kernel_time);
PD_DECLARE_string(tile_config_policy);
PD_DECLARE_string(cinn_kernel_execution_label);
PD_DECLARE_bool(cinn_check_jit_instruction_shape);

namespace paddle {
namespace framework {

typedef void (*lower_func_ptr_g)(void*, int32_t, void*);
typedef void (*infer_shape_func_ptr_g)(void*, int32_t, int64_t**);

class CinnJitInstruction::FnPtrImpl {
  using CINNKernelInfo = cinn::hlir::framework::pir::CINNKernelInfo;

 public:
  explicit FnPtrImpl(const CINNKernelInfo& cinn_kernel_info)
      : cinn_kernel_info_(cinn_kernel_info) {}

  void InitFuncArgs(const std::vector<phi::DenseTensor*>& kernel_tensor_args) {
    // 1. Create placeholders for tensor args
    for (size_t i = 0; i < kernel_tensor_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      func_args_.emplace_back(buffer);
    }

    // 2. Convert symbol args about dynamic shape to cinn_pod_value_t
    const auto& GetSymbolArg = common::Overloaded{
        [&](const CINNKernelInfo::ArgDimIdx& binding_info) -> int64_t {
          return static_cast<int64_t>(
              kernel_tensor_args[binding_info.arg_idx]->dims().at(
                  binding_info.dim_idx));
        },
        [&](const CINNKernelInfo::ArgValueIdx& binding_info) -> int64_t {
          const auto& tensor = [&]() -> phi::DenseTensor {
            phi::DenseTensor new_tensor =
                *(kernel_tensor_args[binding_info.arg_idx]);
            if (new_tensor.place() == phi::CPUPlace()) {
              return new_tensor;
            }
            framework::TensorCopySync(
                *(kernel_tensor_args[binding_info.arg_idx]),
                phi::CPUPlace(),
                &new_tensor);
            return new_tensor;
          }();
          if (tensor.dtype() == phi::DataType::INT32) {
            std::vector<int> tensor_data;
            framework::TensorToVector(tensor, &tensor_data);
            return tensor_data[binding_info.value_idx];
          } else if (tensor.dtype() == phi::DataType::INT64) {
            std::vector<int64_t> tensor_data;
            framework::TensorToVector(tensor, &tensor_data);
            return tensor_data[binding_info.value_idx];
          }
          PADDLE_THROW(
              ::common::errors::Fatal("Dead code, only support int32 and int64 "
                                      "for dynamic shape arg now"));
        }};

    for (const auto& [_, binding_info] : cinn_kernel_info_.symbol_args_map) {
      func_args_.emplace_back(std::visit(GetSymbolArg, binding_info));
    }

    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Run func_args_ size: " << func_args_.size();
      for (const auto& args : func_args_) {
        VLOG(4) << " args type_code: " << args.type_code();
      }
    }
  }

  void Run(const std::vector<phi::DenseTensor*>& kernel_tensor_args,
           void* stream,
           bool is_gpu) {
    VLOG(6) << "Start Run: " << cinn_kernel_info_.fn_name;

    // Pass real tensor data to cinn_buffer_t func args placeholder
    for (size_t i = 0; i < kernel_tensor_args.size(); ++i) {
      cinn_pod_value_to_buffer_p(&(func_args_[i]))->memory =
          reinterpret_cast<uint8_t*>(kernel_tensor_args[i]->data());
    }

    // Launch host kernel
    if (FLAGS_cinn_measure_kernel_time ||
        FLAGS_tile_config_policy == "search") {
      VLOG(3) << "enter searching config branch";
      ::common::PerformanceStatistician& ps =
          ::common::PerformanceStatistician::Instance();
      auto data_p = static_cast<void*>(func_args_.data());
      phi::gpuStream_t stream;
      phi::InitStream(&stream);
      phi::backends::gpu::GpuDeviceSync();
      if (is_gpu) {
        ps.SetGraphNodesNum(25);
        int graph_nodes_num = ps.GetGraphNodesNum();
        phi::gpuGraph_t graph;
        phi::gpuGraphExec_t instance;
        phi::gpuStreamBeginCapture(
            stream, gpuStreamCaptureMode(0));  // StreamCaptureModeGlobal
        for (int ikrnl = 0; ikrnl < graph_nodes_num; ikrnl++) {
          ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
              static_cast<void*>(func_args_.data()), func_args_.size(), stream);
        }
        phi::gpuStreamEndCapture(stream, &graph);
#ifdef PADDLE_WITH_CUDA
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
#elif defined(PADDLE_WITH_HIP)
        hipGraphInstantiate(&instance, graph, NULL, NULL, 0);
#else
        CINN_NOT_IMPLEMENTED
#endif
        ps.CudaStart(FLAGS_cinn_kernel_execution_label);
        phi::gpuGraphLaunch(instance, stream);
        ps.CudaEnd(FLAGS_cinn_kernel_execution_label);
        phi::gpuGraphDestroy(graph);
        phi::gpuGraphExecDestroy(instance);
        phi::DestroyStream(stream);
      } else {
        ((lower_func_ptr_g)cinn_kernel_info_.CX86_fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      }
      phi::backends::gpu::GpuDeviceSync();
    } else {
      if (is_gpu) {
        ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      } else {
        ((lower_func_ptr_g)cinn_kernel_info_.CX86_fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      }
    }
    VLOG(6) << "End Run: " << cinn_kernel_info_.fn_name;
  }

  void InferShape(const std::vector<phi::DenseTensor*>& kernel_tensor_args,
                  const std::vector<phi::DDim>& ir_dim,
                  int32_t input_tensor_size,
                  int32_t output_tensor_size) {
    VLOG(6) << "Start InferShape: " << cinn_kernel_info_.fn_name;
    // Define an array of Pointers to hold the output tensor shape
    std::vector<int64_t*> output_tensor_shapes(output_tensor_size);
    for (int i = 0; i < output_tensor_size; ++i) {
      output_tensor_shapes[i] = reinterpret_cast<int64_t*>(
          malloc(kernel_tensor_args[input_tensor_size + i]->dims().size() *
                 sizeof(int64_t*)));
    }

    // Launch infer_shape_fn_ptr to infer shape of output tensor
    ((infer_shape_func_ptr_g)cinn_kernel_info_.infer_shape_fn_ptr)(
        static_cast<void*>(func_args_.data()),
        func_args_.size(),
        output_tensor_shapes.data());

    // Resize shape of output tensor
    for (int i = 0; i < output_tensor_size; ++i) {
      DDim dim(output_tensor_shapes[i],
               kernel_tensor_args[input_tensor_size + i]->dims().size());
      if (static_cast<size_t>(i) < ir_dim.size() &&
          FLAGS_cinn_check_jit_instruction_shape) {
        CheckDims(ir_dim[i], dim);
      }
      kernel_tensor_args[input_tensor_size + i]->Resize(dim);
      free(output_tensor_shapes[i]);
    }
    VLOG(6) << "End InferShape: " << cinn_kernel_info_.fn_name;
  }

  void FreeFuncArgs() {
    for (auto& arg : func_args_) {
      if (arg.type_code() == ::cinn_type_code<cinn_buffer_t*>()) {
        delete cinn_pod_value_to_buffer_p(&arg);
      }
    }
    func_args_.clear();
  }

  void CheckDims(const DDim& first, const DDim& second) const {
    VLOG(3) << "Start Check Dims in jit instruction.";
    PADDLE_ENFORCE_EQ(
        first.size(),
        second.size(),
        phi::errors::PreconditionNotMet("The rank of dim MUST be same. "
                                        "But get [%d] and [%d]",
                                        first.size(),
                                        second.size()));
    for (size_t i = 0; i < first.size(); ++i) {
      if (first[i] > 0) {
        PADDLE_ENFORCE_EQ(first[i],
                          second[i],
                          phi::errors::PreconditionNotMet(
                              "Dim MUST be equal"
                              ", but Get first[%d] is [%d], second[%d] is[%d]",
                              i,
                              first[i],
                              i,
                              second[i]));
      }
    }
  }

 private:
  CINNKernelInfo cinn_kernel_info_;

  std::vector<cinn_pod_value_t> func_args_;
};

CinnJitInstruction::CinnJitInstruction(
    size_t id,
    const phi::Place& place,
    ::pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place) {
  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  fn_ptr_impl_ = std::make_shared<FnPtrImpl>(jit_kernel_op.cinn_kernel_info());
  op_ = op;
  input_tensor_size = op->num_operands();
  output_tensor_size = op->num_results();

  place_ = place;

  InitInputsOutputsIds(op, *value_exec_info);

  // prepare input tensors
  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);

    auto var_name = value_exec_info->GetVarName(in);
    auto tensor = value_exec_info->GetScope()
                      ->FindVar(var_name)
                      ->GetMutable<phi::DenseTensor>();
    tensor_args_.push_back(tensor);
  }

  if (op->HasAttribute("exec_backend")) {
    place_ = op->attribute("exec_backend")
                 .dyn_cast<paddle::dialect::PlaceAttribute>()
                 .data();
  }
  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);

  // prepare output tensors
  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value result = op->result(i);
    bool check = result && result.type() &&
                 result.type().isa<paddle::dialect::DenseTensorType>();
    PADDLE_ENFORCE_EQ(check,
                      true,
                      phi::errors::PreconditionNotMet(
                          "cinn jit instruction only support DenseTensorType"));
    auto var_name = value_exec_info->GetVarName(result);

    auto tensor = value_exec_info->GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    ir_dims_.push_back(
        result.type().dyn_cast<paddle::dialect::DenseTensorType>().dims());
    tensor_args_.push_back(tensor);
    auto alloc_tensor_type =
        result.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    tensor->set_type(
        paddle::dialect::TransToPhiDataType(alloc_tensor_type.dtype()));
    for (size_t j = 0; j < alloc_tensor_type.dims().size(); ++j) {
      if (alloc_tensor_type.dims()[j] < 0) {
        need_update_shape = true;
        continue;
      }
    }
    tensor->Resize(alloc_tensor_type.dims());
  }

  // prepare temp_space tensors
  for (int64_t size : jit_kernel_op.cinn_kernel_info().temp_space_sizes) {
    auto& tensor = temp_space_tensors_.emplace_back();
    tensor.set_type(phi::DataType::UINT8);
    tensor.Resize({size});
    if (size < 0) {
      need_update_shape = true;
    }
  }
  for (auto& tensor : temp_space_tensors_) {
    tensor_args_.push_back(&tensor);
  }
  output_tensor_size += temp_space_tensors_.size();
}

void CinnJitInstruction::Run() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void* running_stream = nullptr;
  bool is_gpu = false;

  if (place_.GetType() == phi::AllocationType::GPU) {
    is_gpu = true;
    running_stream =
        static_cast<void*>(static_cast<phi::GPUContext*>(dev_ctx_)->stream());
  }

  // 1. prepare kernel arguments
  fn_ptr_impl_->InitFuncArgs(tensor_args_);

  if (need_update_shape) {
    fn_ptr_impl_->InferShape(
        tensor_args_, ir_dims_, input_tensor_size, output_tensor_size);
  }
  for (size_t i = 0; i < tensor_args_.size(); ++i) {
    dev_ctx_->Alloc(tensor_args_[i], tensor_args_[i]->dtype());
  }

  // 2. exexute kernel
  fn_ptr_impl_->Run(tensor_args_, running_stream, is_gpu);

  // 3. release resource
  fn_ptr_impl_->FreeFuncArgs();
  for (auto& tensor : temp_space_tensors_) {
    tensor.clear();
  }
#else
  VLOG(0) << "Not Supported: cinn jit instruction currently does not "
             "support CUDA/HIP kernel";
#endif
}

const std::string& CinnJitInstruction::Name() const {
  static const std::string name = "cinn_jit";
  return name;
}

}  // namespace framework
}  // namespace paddle
