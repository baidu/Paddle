/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"

#include <chrono>
#include "paddle/fluid/framework/convert_utils.h"
namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {

class Scope;

void DeviceWorker::SetRootScope(Scope* root_scope) { root_scope_ = root_scope; }

void DeviceWorker::SetDataFeed(DataFeed* data_feed) {
  device_reader_ = data_feed;
}

template <typename T>
std::string PrintLodTensorType(Tensor* tensor, int64_t start, int64_t end,
                               char separator = ':',
                               bool need_leading_separator = true) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  if (start >= end) return "";
  std::ostringstream os;
  if (!need_leading_separator) {
    os << tensor->data<T>()[start];
    start++;
  }
  for (int64_t i = start; i < end; i++) {
    // os << ":" << tensor->data<T>()[i];
    os << separator << tensor->data<T>()[i];
  }
  return os.str();
}
template <typename T>
void PrintLodTensorType(Tensor* tensor, int64_t start, int64_t end,
                        std::string& out_val, char separator = ':',
                        bool need_leading_separator = true) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    out_val += "access violation";
    return;
  }
  if (start >= end) return;
  if (!need_leading_separator) {
    out_val += std::to_string(tensor->data<T>()[start]);
    // os << tensor->data<T>()[start];
    start++;
  }
  for (int64_t i = start; i < end; i++) {
    // os << ":" << tensor->data<T>()[i];
    // os << separator << tensor->data<T>()[i];
    out_val += separator;
    out_val += std::to_string(tensor->data<T>()[i]);
  }
}

std::string PrintLodTensorIntType(Tensor* tensor, int64_t start, int64_t end,
                                  char separator = ':',
                                  bool need_leading_separator = true) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    return "access violation";
  }
  if (start >= end) return "";
  std::ostringstream os;
  if (!need_leading_separator) {
    os << static_cast<uint64_t>(tensor->data<int64_t>()[start]);
    start++;
  }
  for (int64_t i = start; i < end; i++) {
    // os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
    os << separator << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
  }
  return os.str();
}

void PrintLodTensorIntType(Tensor* tensor, int64_t start, int64_t end,
                           std::string& out_val, char separator = ':',
                           bool need_leading_separator = true) {
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    out_val += "access violation";
    return;
  }
  if (start >= end) return;
  if (!need_leading_separator) {
    out_val +=
        std::to_string(static_cast<uint64_t>(tensor->data<int64_t>()[start]));
    start++;
  }
  for (int64_t i = start; i < end; i++) {
    // os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
    // os << separator << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
    out_val += separator;
    out_val +=
        std::to_string(static_cast<uint64_t>(tensor->data<int64_t>()[i]));
  }
  // return os.str();
}

std::string PrintLodTensor(Tensor* tensor, int64_t start, int64_t end,
                           char separator, bool need_leading_separator) {
  std::string out_val;
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    out_val = PrintLodTensorType<float>(tensor, start, end, separator,
                                        need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::INT64) {
    out_val = PrintLodTensorIntType(tensor, start, end, separator,
                                    need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP64) {
    out_val = PrintLodTensorType<double>(tensor, start, end, separator,
                                         need_leading_separator);
  } else {
    out_val = "unsupported type";
  }
  return out_val;
}

void PrintLodTensor(Tensor* tensor, int64_t start, int64_t end,
                    std::string& out_val, char separator,
                    bool need_leading_separator) {
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    PrintLodTensorType<float>(tensor, start, end, out_val, separator,
                              need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::INT64) {
    PrintLodTensorIntType(tensor, start, end, out_val, separator,
                          need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP64) {
    PrintLodTensorType<double>(tensor, start, end, out_val, separator,
                               need_leading_separator);
  } else {
    out_val += "unsupported type";
  }
}

std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
  auto& dims = tensor->dims();
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}

bool CheckValidOutput(LoDTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != static_cast<int>(batch_size)) {
      return false;
    }
  }
  return true;
}

void DeviceWorker::DumpParam(const Scope& scope, const int batch_id) {
  std::ostringstream os;
  for (auto& param : *dump_param_) {
    os.str("");
    Variable* var = scope.FindVar(param);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    framework::LoDTensor cpu_tensor;
    if (platform::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      tensor = &cpu_tensor;
    }
    int64_t len = tensor->numel();
    os << "(" << batch_id << "," << param << ")"
       << PrintLodTensor(tensor, 0, len);
    writer_ << os.str();
  }
}

void DeviceWorker::InitRandomDumpConfig(const TrainerDesc& desc) {
  bool is_dump_in_simple_mode = desc.is_dump_in_simple_mode();
  if (is_dump_in_simple_mode) {
    dump_mode_ = 3;
    return;
  }
  bool enable_random_dump = desc.enable_random_dump();
  if (!enable_random_dump) {
    dump_mode_ = 0;
  } else {
    if (desc.random_with_lineid()) {
      dump_mode_ = 1;
    } else {
      dump_mode_ = 2;
    }
  }
  dump_interval_ = desc.dump_interval();
}

void DeviceWorker::DumpField(const Scope& scope, int dump_mode,
                             int dump_interval) {  // dump_mode: 0: no random,
                                                   // 1: random with insid hash,
                                                   // 2: random with random
  // 3: simple mode using multi-threads, for gpugraphps-mode
  auto start1 = std::chrono::steady_clock::now();

  size_t batch_size = device_reader_->GetCurBatchSize();
  auto& ins_id_vec = device_reader_->GetInsIdVec();
  auto& ins_content_vec = device_reader_->GetInsContentVec();
  if (ins_id_vec.size() > 0) {
    batch_size = ins_id_vec.size();
  }
  std::vector<std::string> ars(batch_size);
  std::vector<bool> hit(batch_size, false);
  if (dump_mode_ == 3) {
    if (dump_fields_ == NULL || (*dump_fields_).size() == 0) {
      return;
    }
    auto set_output_str = [&, this](size_t begin, size_t end,
                                    LoDTensor* tensor) {
      for (size_t i = begin; i < end; ++i) {
        auto bound = GetTensorBound(tensor, i);
        if (ars[i].size() > 0) ars[i] += "\t";
        PrintLodTensor(tensor, bound.first, bound.second, ars[i], ' ', false);
      }
    };
    std::thread threads[tensor_iterator_thread_num];
    for (auto& field : *dump_fields_) {
      Variable* var = scope.FindVar(field);
      if (var == nullptr) {
        VLOG(0) << "Note: field[" << field
                << "] cannot be find in scope, so it was skipped.";
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (!tensor->IsInitialized()) {
        VLOG(0) << "Note: field[" << field
                << "] is not initialized, so it was skipped.";
        continue;
      }
      framework::LoDTensor cpu_tensor;
      if (platform::is_gpu_place(tensor->place())) {
        TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
        cpu_tensor.set_lod(tensor->lod());
        tensor = &cpu_tensor;
      }
      if (!CheckValidOutput(tensor, batch_size)) {
        VLOG(0) << "Note: field[" << field << "] cannot pass check, so it was "
                                              "skipped. Maybe the dimension is "
                                              "wrong ";
        continue;
      }
      size_t acutal_thread_num =
          std::min((size_t)batch_size, tensor_iterator_thread_num);
      for (size_t i = 0; i < acutal_thread_num; i++) {
        size_t average_size = batch_size / acutal_thread_num;
        size_t begin =
            average_size * i + std::min(batch_size % acutal_thread_num, i);
        size_t end =
            begin + average_size + (i < batch_size % acutal_thread_num ? 1 : 0);
        threads[i] = std::thread(set_output_str, begin, end, tensor);
      }
      for (size_t i = 0; i < acutal_thread_num; i++) threads[i].join();
    }
    auto end1 = std::chrono::steady_clock::now();
    auto tt =
        std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    VLOG(1) << "writing a batch takes " << tt.count() << " us";

    for (size_t i = 0; i < ars.size(); i++) {
      if (ars[i].length() == 0) {
        continue;
      }
      writer_ << ars[i];
    }
    return;
  }
  std::default_random_engine engine(0);
  std::uniform_int_distribution<size_t> dist(0U, INT_MAX);
  for (size_t i = 0; i < batch_size; i++) {
    size_t r = 0;
    if (dump_mode == 1) {
      r = XXH64(ins_id_vec[i].data(), ins_id_vec[i].length(), 0);
    } else if (dump_mode == 2) {
      r = dist(engine);
    }
    if (r % dump_interval != 0) {
      continue;
    }
    hit[i] = true;
  }

  for (size_t i = 0; i < ins_id_vec.size(); i++) {
    if (!hit[i]) {
      continue;
    }
    ars[i] += ins_id_vec[i];
    ars[i] = ars[i] + "\t" + ins_content_vec[i];
  }
  for (auto& field : *dump_fields_) {
    Variable* var = scope.FindVar(field);
    if (var == nullptr) {
      VLOG(0) << "Note: field[" << field
              << "] cannot be find in scope, so it was skipped.";
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (!tensor->IsInitialized()) {
      VLOG(0) << "Note: field[" << field
              << "] is not initialized, so it was skipped.";
      continue;
    }
    framework::LoDTensor cpu_tensor;
    if (platform::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
      cpu_tensor.set_lod(tensor->lod());
      tensor = &cpu_tensor;
    }
    if (!CheckValidOutput(tensor, batch_size)) {
      VLOG(0) << "Note: field[" << field << "] cannot pass check, so it was "
                                            "skipped. Maybe the dimension is "
                                            "wrong ";
      continue;
    }
    for (size_t i = 0; i < batch_size; ++i) {
      if (!hit[i]) {
        continue;
      }
      auto bound = GetTensorBound(tensor, i);
      ars[i] = ars[i] + "\t" + field + ":" +
               std::to_string(bound.second - bound.first);
      ars[i] += PrintLodTensor(tensor, bound.first, bound.second);
    }
  }

  // #pragma omp parallel for
  for (size_t i = 0; i < ars.size(); i++) {
    if (ars[i].length() == 0) {
      continue;
    }
    writer_ << ars[i];
  }
}

}  // namespace framework
}  // namespace paddle
