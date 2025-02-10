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

#include <array>
#include <chrono>
#include "paddle/fluid/framework/convert_utils.h"
namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {

class Scope;

void DeviceWorker::SetRootScope(Scope* root_scope) { root_scope_ = root_scope; }

void DeviceWorker::SetDataFeed(DataFeed* data_feed) {
  device_reader_ = data_feed;
}

template <typename T>
std::string PrintDenseTensorType(phi::DenseTensor* tensor,
                                 int64_t start,
                                 int64_t end,
                                 char separator = ',',
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
void PrintDenseTensorType(phi::DenseTensor* tensor,
                          int64_t start,
                          int64_t end,
                          std::string& out_val,  // NOLINT
                          char separator = ',',
                          bool need_leading_separator = true,
                          int num_decimals = 9) {
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

#define FLOAT_EPS 1e-8
#define MAX_FLOAT_BUFF_SIZE 40
template <>
void PrintDenseTensorType<float>(phi::DenseTensor* tensor,
                                 int64_t start,
                                 int64_t end,
                                 std::string& out_val,  // NOLINT
                                 char separator,
                                 bool need_leading_separator,
                                 int num_decimals) {
  char buf[MAX_FLOAT_BUFF_SIZE];  // NOLINT
  auto count = tensor->numel();
  if (start < 0 || end > count) {
    VLOG(3) << "access violation";
    out_val += "access violation";
    return;
  }
  if (start >= end) return;
  for (int64_t i = start; i < end; i++) {
    if (i != start || need_leading_separator) out_val += separator;
    if (tensor->data<float>()[i] > -FLOAT_EPS &&
        tensor->data<float>()[i] < FLOAT_EPS) {
      out_val += "0";
    } else {
      std::string format = "%." + std::to_string(num_decimals) + "f";
      sprintf(buf, &format[0], tensor->data<float>()[i]);  // NOLINT
      out_val += buf;
    }
  }
}
std::string PrintDenseTensorIntType(phi::DenseTensor* tensor,
                                    int64_t start,
                                    int64_t end,
                                    char separator = ',',
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

void PrintDenseTensorIntType(phi::DenseTensor* tensor,
                             int64_t start,
                             int64_t end,
                             std::string& out_val,  // NOLINT
                             char separator = ',',
                             bool need_leading_separator = true,
                             int num_decimals = 9) {
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

std::string PrintDenseTensor(phi::DenseTensor* tensor,
                             int64_t start,
                             int64_t end,
                             char separator,
                             bool need_leading_separator) {
  std::string out_val;
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    out_val = PrintDenseTensorType<float>(
        tensor, start, end, separator, need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::INT64) {
    out_val = PrintDenseTensorIntType(
        tensor, start, end, separator, need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP64) {
    out_val = PrintDenseTensorType<double>(
        tensor, start, end, separator, need_leading_separator);
  } else {
    out_val = "unsupported type";
  }
  return out_val;
}

void PrintDenseTensor(phi::DenseTensor* tensor,
                      int64_t start,
                      int64_t end,
                      std::string& out_val,  // NOLINT
                      char separator,
                      bool need_leading_separator,
                      int num_decimals) {
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    PrintDenseTensorType<float>(tensor,
                                start,
                                end,
                                out_val,
                                separator,
                                need_leading_separator,
                                num_decimals);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::INT64) {
    PrintDenseTensorIntType(
        tensor, start, end, out_val, separator, need_leading_separator);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP64) {
    PrintDenseTensorType<double>(
        tensor, start, end, out_val, separator, need_leading_separator);
  } else {
    out_val += "unsupported type";
  }
}

std::pair<int64_t, int64_t> GetTensorBound(phi::DenseTensor* tensor,
                                           int index) {
  auto& dims = tensor->dims();
  if (!tensor->lod().empty()) {
    auto& lod = tensor->lod()[0];
    return {lod[index] * dims[1], lod[index + 1] * dims[1]};
  } else {
    return {index * dims[1], (index + 1) * dims[1]};
  }
}

bool CheckValidOutput(phi::DenseTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() != 2) return false;
  if (!tensor->lod().empty()) {
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
  int device_id =
      static_cast<int>(static_cast<unsigned char>(place_.GetDeviceId()));
  for (auto& param : *dump_param_) {
    os.str("");
    Variable* var = scope.FindVar(param);
    if (var == nullptr || !var->IsInitialized()) {
      continue;
    }
    if (!var->IsType<phi::DenseTensor>()) {
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    if (tensor == nullptr || !tensor->IsInitialized()) {
      continue;
    }
    phi::DenseTensor cpu_tensor;
    if (phi::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, phi::CPUPlace(), &cpu_tensor);
      tensor = &cpu_tensor;
    }
    int64_t len = tensor->numel();
    os << "(" << device_id << "," << batch_id << "," << param << ")"
       << PrintDenseTensor(tensor, 0, len);
    writer_ << os.str();
  }
}

void DeviceWorker::InitRandomDumpConfig(const TrainerDesc& desc) {
  bool is_dump_in_simple_mode = desc.is_dump_in_simple_mode();
  if (is_dump_in_simple_mode) {
    dump_mode_ = 3;
    dump_num_decimals_ = desc.dump_num_decimals();
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

void DeviceWorker::DumpField(
    const Scope& scope,
    int dump_mode,
    int dump_interval) {  // dump_mode: 0: no random,
                          // 1: random with ins id hash,
                          // 2: random with random
  // 3: simple mode using multi-threads, for gpugraphps-mode
  auto start1 = std::chrono::steady_clock::now();

  size_t batch_size = device_reader_->GetCurBatchSize();
  auto& ins_id_vec = device_reader_->GetInsIdVec();
  auto& ins_content_vec = device_reader_->GetInsContentVec();
  if (dump_mode_ == 3) {
    batch_size = std::string::npos;
    bool has_valid_batch = false;
    for (auto& field : *dump_fields_) {
      Variable* var = scope.FindVar(field);
      if (var == nullptr) {
        VLOG(3) << "Note: field[" << field
                << "] cannot be find in scope, so it was skipped.";
        continue;
      }
      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      if (!tensor->IsInitialized()) {
        VLOG(3) << "Note: field[" << field
                << "] is not initialized, so it was skipped.";
        continue;
      }
      auto& dims = tensor->dims();
      if (dims.size() == 2 && dims[0] > 0) {
        batch_size = std::min(batch_size, static_cast<size_t>(dims[0]));
        // VLOG(0)<<"in dump field ---> "<<field<<" dim_size = "<<dims[0]<<"
        // "<<dims[1]<<" batch_size = "<<batch_size;
        has_valid_batch = true;
      }
    }
    if (!has_valid_batch) return;
  } else if (!ins_id_vec.empty()) {
    batch_size = ins_id_vec.size();
  }
  std::vector<std::string> ars(batch_size);
  if (dump_mode_ == 3) {
    if (dump_fields_ == NULL || (*dump_fields_).empty()) {
      return;
    }
    auto set_output_str =
        [&, this](size_t begin, size_t end, phi::DenseTensor* tensor) {
          std::pair<int64_t, int64_t> bound;
          auto& dims = tensor->dims();
          for (size_t i = begin; i < end; ++i) {
            bound = {i * dims[1], (i + 1) * dims[1]};
            // auto bound = GetTensorBound(tensor, i);

            if (!ars[i].empty()) ars[i] += "\t";
            // ars[i] += '[';
            PrintDenseTensor(tensor,
                             bound.first,
                             bound.second,
                             ars[i],
                             ' ',
                             false,
                             dump_num_decimals_);
            // ars[i] += ']';
            // ars[i] += "<" + PrintDenseTensor(tensor, bound.first,
            // bound.second,
            // '
            // ', false) + ">";
          }
        };
    std::vector<std::thread> threads(tensor_iterator_thread_num);
    for (auto& field : *dump_fields_) {
      Variable* var = scope.FindVar(field);
      if (var == nullptr) {
        VLOG(3) << "Note: field[" << field
                << "] cannot be find in scope, so it was skipped.";
        continue;
      }
      phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
      if (!tensor->IsInitialized()) {
        VLOG(3) << "Note: field[" << field
                << "] is not initialized, so it was skipped.";
        continue;
      }
      phi::DenseTensor cpu_tensor;
      if (phi::is_gpu_place(tensor->place())) {
        TensorCopySync(*tensor, phi::CPUPlace(), &cpu_tensor);
        cpu_tensor.set_lod(tensor->lod());
        tensor = &cpu_tensor;
      }
      auto& dims = tensor->dims();
      if (dims.size() != 2 || dims[0] <= 0) {
        VLOG(3) << "Note: field[" << field
                << "] cannot pass check, so it was "
                   "skipped. Maybe the dimension is "
                   "wrong ";
        VLOG(3) << dims.size() << " " << dims[0] << " * " << dims[1];
        continue;
      }
      size_t actual_thread_num =
          std::min(static_cast<size_t>(batch_size), tensor_iterator_thread_num);
      for (size_t i = 0; i < actual_thread_num; i++) {
        size_t average_size = batch_size / actual_thread_num;
        size_t begin =
            average_size * i + std::min(batch_size % actual_thread_num, i);
        size_t end =
            begin + average_size + (i < batch_size % actual_thread_num ? 1 : 0);
        threads[i] = std::thread(set_output_str, begin, end, tensor);
      }
      for (size_t i = 0; i < actual_thread_num; i++) threads[i].join();
    }
    auto end1 = std::chrono::steady_clock::now();
    auto tt =
        std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    VLOG(2) << "writing a batch takes " << tt.count() << " us";

    size_t actual_thread_num =
        std::min(static_cast<size_t>(batch_size), tensor_iterator_thread_num);
    for (size_t i = 0; i < actual_thread_num; i++) {
      size_t average_size = batch_size / actual_thread_num;
      size_t begin =
          average_size * i + std::min(batch_size % actual_thread_num, i);
      size_t end =
          begin + average_size + (i < batch_size % actual_thread_num ? 1 : 0);
      for (size_t j = begin + 1; j < end; j++) {
        if (!ars[begin].empty() && !ars[j].empty()) ars[begin] += "\n";
        ars[begin] += ars[j];
      }
      if (!ars[begin].empty()) writer_ << ars[begin];
    }
    return;
  }
  std::vector<bool> hit(batch_size, false);
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
  }  // dump_mode = 0
  for (size_t i = 0; i < ins_id_vec.size(); i++) {
    if (!hit[i]) {
      continue;
    }
    ars[i] += ins_id_vec[i];
    if (ins_content_vec.size() > i) ars[i] = ars[i] + "\t" + ins_content_vec[i];
  }
  for (auto& field : *dump_fields_) {
    Variable* var = scope.FindVar(field);
    if (var == nullptr) {
      VLOG(3) << "Note: field[" << field
              << "] cannot be find in scope, so it was skipped.";
      continue;
    }
    if (!var->IsType<phi::DenseTensor>()) {
      VLOG(3) << "Note: field[" << field
              << "] is not dense tensor, so it was skipped.";
      continue;
    }
    phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
    if (!tensor->IsInitialized()) {
      VLOG(3) << "Note: field[" << field
              << "] is not initialized, so it was skipped.";
      continue;
    }
    phi::DenseTensor cpu_tensor;
    if (phi::is_gpu_place(tensor->place())) {
      TensorCopySync(*tensor, phi::CPUPlace(), &cpu_tensor);
      cpu_tensor.set_lod(tensor->lod());
      tensor = &cpu_tensor;
    }
    if (!CheckValidOutput(tensor, batch_size)) {
      VLOG(3) << "Note: field[" << field
              << "] cannot pass check, so it was "
                 "skipped. Maybe the dimension is "
                 "wrong ";
      continue;
    }
    for (size_t i = 0; i < batch_size; ++i) {
      if (!hit[i]) {
        continue;
      }
      auto bound = GetTensorBound(tensor, static_cast<int>(i));
      ars[i] +=
          "\t" + field + ":" + std::to_string(bound.second - bound.first) + ":";
      ars[i] += PrintDenseTensor(tensor, bound.first, bound.second);
    }
  }

  // #pragma omp parallel for
  for (auto& ar : ars) {
    if (ar.length() == 0) {
      continue;
    }
    writer_ << ar;
  }
  writer_.Flush();
}

}  // namespace paddle::framework
