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

#include "test/cpp/fluid/benchmark/op_tester.h"

#include <fstream>

#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/timer.h"

// phi
#include "paddle/phi/kernels/declarations.h"

namespace paddle {
namespace operators {
namespace benchmark {

PD_DEFINE_string(op_config_list, "", "Path of op config file.");  // NOLINT
PD_DEFINE_int32(specified_config_id, -1, "Test the specified op config.");

void OpTester::Init(const std::string &filename) {
  Init(OpTesterConfig(filename));
}

void OpTester::Init(const OpTesterConfig &config) {
  config_ = config;

  auto &op_desc_info = framework::OpInfoMap::Instance();
  // Initialize the OpDesc
  if (op_desc_info.Has(config_.op_type)) {
    type_ = config_.op_type;

    CreateOpDesc();
    CreateInputVarDesc();
    CreateOutputVarDesc();
  } else {
    PADDLE_THROW(common::errors::NotFound(
        "Operator '%s' is not registered in OpTester.", config_.op_type));
  }

  if (config_.device_id >= 0) {
    place_ = ::phi::GPUPlace(config_.device_id);
  } else {
    place_ = ::phi::CPUPlace();
  }

  framework::InitDevices();
  scope_ = std::make_unique<::paddle::framework::Scope>();

  op_ = framework::OpRegistry::CreateOp(op_desc_);
  CreateVariables(scope_.get());
}

void OpTester::Run() {
  if (config_.print_debug_string) {
    LOG(INFO) << DebugString();
  }

  // Warm up
  RunImpl();

  platform::Timer timer;
  if (config_.profile) {
    if (phi::is_cpu_place(place_)) {
      platform::EnableProfiler(platform::ProfilerState::kCPU);
    } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::EnableProfiler(platform::ProfilerState::kAll);
      platform::SetDeviceId(config_.device_id);
#else
      PADDLE_THROW(common::errors::PermissionDenied(
          "'GPUPlace' is not supported in CPU only device."));
#endif
    }

    timer.Start();
    for (int i = config_.repeat; i > 0; --i) {
      RunImpl();
    }
    timer.Pause();
    platform::DisableProfiler(platform::EventSortingKey::kDefault,
                              "op_tester_profiler");
  } else {
    timer.Start();
    for (int i = config_.repeat; i > 0; --i) {
      RunImpl();
    }
    timer.Pause();
  }
  config_.runtime = timer.ElapsedMS() / config_.repeat;
  LOG(INFO) << "=== Run " << config_.repeat
            << " times, latency: " << config_.runtime << " ms ===";
}

void OpTester::RunImpl() {
  op_->Run(*scope_, place_);
  phi::DeviceContextPool::Instance().Get(place_)->Wait();
  scope_->DropKids();
}

std::vector<std::string> OpTester::GetOpProtoInputNames() {
  std::vector<std::string> input_names;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  for (int i = 0; i != proto.inputs_size(); ++i) {
    const auto &input = proto.inputs(i);
    input_names.push_back(input.name());
  }
  return input_names;
}

std::vector<std::string> OpTester::GetOpProtoOutputNames() {
  std::vector<std::string> output_names;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  for (int i = 0; i != proto.outputs_size(); ++i) {
    const auto &output = proto.outputs(i);
    output_names.push_back(output.name());
  }
  return output_names;
}

std::unordered_map<std::string, framework::proto::AttrType>
OpTester::GetOpProtoAttrNames() {
  std::unordered_map<std::string, framework::proto::AttrType> attr_types;
  const framework::proto::OpProto &proto =
      framework::OpInfoMap::Instance().Get(type_).Proto();
  const std::vector<std::string> skipped_attrs = {
      framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
      framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(),
      framework::OpProtoAndCheckerMaker::OpNamescopeAttrName(),
      framework::OpProtoAndCheckerMaker::OpCreationCallstackAttrName(),
      framework::OpProtoAndCheckerMaker::OpWithQuantAttrName()};
  for (int i = 0; i != proto.attrs_size(); ++i) {
    const auto &attr = proto.attrs(i);
    if (!Has(skipped_attrs, attr.name())) {
      VLOG(4) << "attr: " << attr.name() << ", type: " << attr.type();
      attr_types[attr.name()] = attr.type();
    }
  }
  return attr_types;
}

framework::proto::VarType::Type OpTester::TransToVarType(std::string str) {
  if (str == "int32") {
    return framework::proto::VarType::INT32;
  } else if (str == "int64") {
    return framework::proto::VarType::INT64;
  } else if (str == "fp32") {
    return framework::proto::VarType::FP32;
  } else if (str == "fp64") {
    return framework::proto::VarType::FP64;
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported dtype %s in OpTester.", str.c_str()));
  }
}

void OpTester::CreateInputVarDesc() {
  std::vector<std::string> input_names = GetOpProtoInputNames();
  for (auto &name : input_names) {
    const OpInputConfig *input = config_.GetInput(name);
    PADDLE_ENFORCE_NOT_NULL(
        input,
        common::errors::NotFound(
            "The input %s of operator %s is not correctly provided.",
            name,
            config_.op_type));

    std::string var_name = config_.op_type + "." + name;
    framework::VarDesc *var = Var(var_name);
    // Need to support more type
    var->SetType(framework::proto::VarType::DENSE_TENSOR);
    var->SetPersistable(false);
    var->SetDataType(TransToVarType(input->dtype));
    var->SetShape(input->dims);

    op_desc_.SetInput(name, {var_name});
    inputs_[var_name] = *input;
  }
}

void OpTester::CreateOutputVarDesc() {
  std::vector<std::string> output_names = GetOpProtoOutputNames();
  for (auto &name : output_names) {
    std::string var_name = config_.op_type + "." + name;
    framework::VarDesc *var = Var(var_name);
    // Need to support more type
    var->SetType(framework::proto::VarType::DENSE_TENSOR);
    var->SetPersistable(false);
    var->SetDataType(framework::proto::VarType::FP32);

    op_desc_.SetOutput(name, {var_name});
  }
}

void OpTester::CreateOpDesc() {
  op_desc_.SetType(config_.op_type);
  std::unordered_map<std::string, framework::proto::AttrType> attr_types =
      GetOpProtoAttrNames();
  for (auto item : config_.attrs) {
    const std::string &name = item.first;
    PADDLE_ENFORCE_NE(
        attr_types.find(name),
        attr_types.end(),
        common::errors::NotFound(
            "Operator %s does not have attribute %d.", type_, name));

    const std::string &value_str = item.second;
    const framework::proto::AttrType &type = attr_types[name];
    switch (type) {
      case framework::proto::AttrType::BOOLEAN:
        break;
      case framework::proto::AttrType::INT: {
        int value = StringTo<int>(value_str);
        op_desc_.SetAttr(name, {value});
      } break;
      case framework::proto::AttrType::FLOAT: {
        float value = StringTo<float>(value_str);
        op_desc_.SetAttr(name, {value});
      } break;
      case framework::proto::AttrType::STRING: {
        op_desc_.SetAttr(name, {value_str});
      } break;
      case framework::proto::AttrType::BOOLEANS:
      case framework::proto::AttrType::INTS:
      case framework::proto::AttrType::FLOATS:
      case framework::proto::AttrType::STRINGS:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported STRINGS type in OpTester yet."));
        break;
      case framework::proto::AttrType::LONG: {
        int64_t value = StringTo<int64_t>(value_str);
        op_desc_.SetAttr(name, value);
      } break;
      case framework::proto::AttrType::LONGS:
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported attr type %d in OpTester.", type));
    }
  }
}

framework::VarDesc *OpTester::Var(const std::string &name) {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  auto *var = new framework::VarDesc(name);
  vars_[name].reset(var);
  return var;
}

template <typename T>
void OpTester::SetupTensor(phi::DenseTensor *tensor,
                           const std::vector<int64_t> &shape,
                           T lower,
                           T upper,
                           const std::string &initializer,
                           const std::string &filename) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T *ptr = tensor->mutable_data<T>(common::make_ddim(shape), place_);

  phi::DenseTensor cpu_tensor;
  T *cpu_ptr = nullptr;

  if (!phi::is_cpu_place(place_)) {
    cpu_ptr =
        cpu_tensor.mutable_data<T>(common::make_ddim(shape), phi::CPUPlace());
  } else {
    cpu_ptr = ptr;
  }

  if (initializer == "random") {
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      cpu_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
    }
  } else if (initializer == "natural") {
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      cpu_ptr[i] = static_cast<T>(lower + i);
    }
  } else if (initializer == "zeros") {
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      cpu_ptr[i] = static_cast<T>(0);
    }
  } else if (initializer == "file") {
    std::ifstream is(filename);
    for (int i = 0; i < cpu_tensor.numel(); ++i) {
      T value;
      is >> value;
      cpu_ptr[i] = static_cast<T>(value);
    }
    is.close();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported initializer %s in OpTester.", initializer.c_str()));
  }

  if (!phi::is_cpu_place(place_)) {
    ::paddle::framework::TensorCopySync(cpu_tensor, place_, tensor);
  }
}

void OpTester::CreateVariables(framework::Scope *scope) {
  for (auto &item : vars_) {
    auto &var = item.second;
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    auto *ptr = scope->Var(var->Name());
    framework::InitializeVariable(ptr, var->GetType());
    if (var->Persistable()) {
      VLOG(3) << "Create Variable " << var->Name()
              << " global, which pointer is " << ptr;
    } else {
      VLOG(3) << "Create Variable " << var->Name()
              << " locally, which pointer is " << ptr;
    }
  }

  for (auto &item : inputs_) {
    // Allocate memory for input tensor
    auto &var_name = item.first;
    VLOG(3) << "Allocate memory for tensor " << var_name;

    auto &var_desc = vars_[var_name];
    std::vector<int64_t> shape = var_desc->GetShape();

    auto *var = scope->Var(var_name);
    auto *tensor = var->GetMutable<phi::DenseTensor>();
    const auto &data_type = var_desc->GetDataType();
    if (data_type == framework::proto::VarType::INT32) {
      SetupTensor<int>(
          tensor, shape, 0, 1, item.second.initializer, item.second.filename);
    } else if (data_type == framework::proto::VarType::INT64) {
      SetupTensor<int64_t>(
          tensor, shape, 0, 1, item.second.initializer, item.second.filename);
    } else if (data_type == framework::proto::VarType::FP32) {
      SetupTensor<float>(tensor,
                         shape,
                         static_cast<float>(0.0),
                         static_cast<float>(1.0),
                         item.second.initializer,
                         item.second.filename);
    } else if (data_type == framework::proto::VarType::FP64) {
      SetupTensor<double>(tensor,
                          shape,
                          static_cast<double>(0.0),
                          static_cast<double>(1.0),
                          item.second.initializer,
                          item.second.filename);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported dtype %d in OpTester.", data_type));
    }

    VLOG(3) << "Set lod for tensor " << var_name;
    std::vector<std::vector<size_t>> &lod_vec = item.second.lod;
    phi::LegacyLoD lod;
    for (auto &item : lod_vec) {
      lod.push_back(item);
    }
    tensor->set_lod(lod);
  }
}

static std::string GenSpaces(int count) {
  std::stringstream ss;
  for (int i = 0; i < count; ++i) {
    ss << "  ";
  }
  return ss.str();
}

std::string OpTester::DebugString() {
  std::stringstream ss;
  int count = 0;
  for (auto &item : vars_) {
    auto &var = item.second;
    ss << GenSpaces(count++) << "vars {\n";
    ss << GenSpaces(count) << "name: \"" << var->Name() << "\"\n";
    ss << GenSpaces(count++) << "type: {\n";
    ss << GenSpaces(count) << "type: DENSE_TENSOR\n";
    ss << GenSpaces(count++) << "lod_tensor {\n";
    ss << GenSpaces(count++) << "tensor {\n";
    const auto &data_type = var->GetDataType();
    if (data_type == framework::proto::VarType::INT32) {
      ss << GenSpaces(count) << "data_type: INT32\n";
    } else if (data_type == framework::proto::VarType::INT64) {
      ss << GenSpaces(count) << "data_type: INT64\n";
    } else if (data_type == framework::proto::VarType::FP32) {
      ss << GenSpaces(count) << "data_type: FP32\n";
    } else if (data_type == framework::proto::VarType::FP64) {
      ss << GenSpaces(count) << "data_type: FP64\n";
    }
    std::vector<int64_t> shape = var->GetShape();
    for (auto d : shape) {
      ss << GenSpaces(count) << "dims: " << d << "\n";
    }
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(--count) << "}\n";
    ss << GenSpaces(count) << "persistable: " << var->Persistable() << "\n";
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(count++) << "ops {\n";
  for (auto &name : op_desc_.InputNames()) {
    ss << GenSpaces(count++) << "inputs {\n";
    ss << GenSpaces(count) << "parameters: \"" << name << "\"\n";
    ss << GenSpaces(count) << "arguments: \"" << op_desc_.Input(name)[0]
       << "\"\n";
    ss << GenSpaces(--count) << "}\n";
  }
  for (auto &name : op_desc_.OutputNames()) {
    ss << GenSpaces(count++) << "outputs {\n";
    ss << GenSpaces(count) << "parameters: \"" << name << "\"\n";
    ss << GenSpaces(count) << "arguments: \"" << op_desc_.Output(name)[0]
       << "\"\n";
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(count) << "type: " << op_desc_.Type() << "\n";
  for (auto &name : op_desc_.AttrNames()) {
    ss << GenSpaces(count++) << "attrs {\n";
    const auto &attr_type = op_desc_.GetAttrType(name);
    const auto &attr = op_desc_.GetAttr(name);
    ss << GenSpaces(count) << "name: \"" << name << "\"\n";
    switch (attr_type) {
      case framework::proto::AttrType::BOOLEAN: {
        ss << GenSpaces(count) << "type: BOOLEAN\n";
        ss << GenSpaces(count) << "b: " << PADDLE_GET_CONST(bool, attr) << "\n";
      } break;
      case framework::proto::AttrType::INT: {
        ss << GenSpaces(count) << "type: INT\n";
        ss << GenSpaces(count) << "i: " << PADDLE_GET_CONST(int, attr) << "\n";
      } break;
      case framework::proto::AttrType::FLOAT: {
        ss << GenSpaces(count) << "type: FLOAT\n";
        ss << GenSpaces(count) << "f: " << PADDLE_GET_CONST(float, attr)
           << "\n";
      } break;
      case framework::proto::AttrType::STRING: {
        ss << GenSpaces(count) << "type: STRING\n";
        ss << GenSpaces(count) << "s: \"" << PADDLE_GET_CONST(std::string, attr)
           << "\"\n";
      } break;
      case framework::proto::AttrType::BOOLEANS: {
        ss << GenSpaces(count) << "type: BOOLEANS\n";
        ss << GenSpaces(count) << "bools: "
           << "\n";
      } break;
      case framework::proto::AttrType::INTS: {
        ss << GenSpaces(count) << "type: INTS\n";
        ss << GenSpaces(count) << "ints: "
           << "\n";
      } break;
      case framework::proto::AttrType::FLOATS: {
        ss << GenSpaces(count) << "type: FLOATS\n";
        ss << GenSpaces(count) << "floats: "
           << "\n";
      } break;
      case framework::proto::AttrType::STRINGS: {
        ss << GenSpaces(count) << "type: STRINGS\n";
        ss << GenSpaces(count) << "strings: "
           << "\n";
      } break;
      case framework::proto::AttrType::LONG: {
        ss << GenSpaces(count) << "type: LONG\n";
        ss << GenSpaces(count) << "l: " << PADDLE_GET_CONST(int64_t, attr)
           << "\n";
      } break;
      case framework::proto::AttrType::LONGS: {
        ss << GenSpaces(count) << "type: LONGS\n";
        ss << GenSpaces(count) << "longs: "
           << "\n";
      } break;
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported attr type %d in OpTester.", attr_type));
    }
    ss << GenSpaces(--count) << "}\n";
  }
  ss << GenSpaces(--count) << "}\n";
  return ss.str();
}

TEST(op_tester, base) {
  if (!FLAGS_op_config_list.empty()) {
    std::ifstream fin(FLAGS_op_config_list, std::ios::in | std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin),
        true,
        common::errors::InvalidArgument("OpTester cannot open file %s",
                                        FLAGS_op_config_list.c_str()));
    std::vector<OpTesterConfig> op_configs;
    while (!fin.eof()) {
      VLOG(4) << "Reading config " << op_configs.size() << "...";
      OpTesterConfig config;
      bool result = config.Init(fin);
      if (result) {
        op_configs.push_back(config);
      }
    }
    if (FLAGS_specified_config_id >= 0 &&
        FLAGS_specified_config_id < static_cast<int>(op_configs.size())) {
      OpTester tester;
      tester.Init(op_configs[FLAGS_specified_config_id]);
      tester.Run();
    } else {
      for (auto &op_config : op_configs) {
        OpTester tester;
        tester.Init(op_config);
        tester.Run();
      }
    }
  } else {
    OpTester tester;
    OpTesterConfig config;
    config.op_type = "elementwise_add";
    config.inputs.resize(2);
    config.inputs[0].name = "X";
    config.inputs[0].dims = {64, 64};
    config.inputs[1].name = "Y";
    config.inputs[1].dims = {64, 1};
    tester.Init(config);
    tester.Run();
  }
}

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
