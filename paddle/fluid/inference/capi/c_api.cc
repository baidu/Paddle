// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/capi/c_api.h"
#include <algorithm>
#include <vector>
#include "paddle/fluid/inference/capi/c_api_internal.h"

extern "C" {

PD_PaddleBuf* PD_NewPaddleBuf() { return new PD_PaddleBuf; }

void PD_DeletePaddleBuf(PD_PaddleBuf* buf) {
  if (buf) {
    delete buf;
    buf = nullptr;
  }
}

void PD_PaddleBufResize(PD_PaddleBuf* buf, size_t length) {
  buf->buf.Resize(length);
}

void PD_PaddleBufReset(PD_PaddleBuf* buf, void* data, size_t length) {
  buf->buf.Reset(data, length);
}

bool PD_PaddleBufEmpty(PD_PaddleBuf* buf) { return buf->buf.empty(); }

void* PD_PaddleBufData(PD_PaddleBuf* buf) { return buf->buf.data(); }

size_t PD_PaddleBufLength(PD_PaddleBuf* buf) { return buf->buf.length(); }

paddle::PaddleDType ConvertToPaddleDType(PD_DataType dtype) {
  switch (dtype) {
    case PD_FLOAT32:
      return PD_PaddleDType::FLOAT32;
    case PD_INT32:
      return PD_PaddleDType::INT32;
    case PD_INT64:
      return PD_PaddleDType::INT64;
    case PD_UINT8:
      return PD_PaddleDType::UINT8;
    default:
      PADDLE_ENFORCE(false, "Unsupport dtype.");
      return PD_PaddleDType::FLOAT32;
  }
  PADDLE_ENFORCE(false, "Unsupport dtype.");
  return PD_PaddleDType::FLOAT32;
}

paddle::PaddlePlace ConvertToPlace(PD_Place dtype) {
  switch (dtype) {
    case PD_UNK:
      return PD_PaddlePlace::kUNK;
    case PD_CPU:
      return PD_PaddlePlace::kCPU;
    case PD_GPU:
      return PD_PaddlePlace::kGPU;
    default:
      PADDLE_ENFORCE(false, "Unsupport place.");
      return PD_PaddlePlace::kUNK;
  }
  PADDLE_ENFORCE(false, "Unsupport dtype.");
  return PD_PaddlePlace::kUNK;
}

PD_DataType ConvertToPDDataType(PD_PaddleDType dtype) {
  switch (dtype) {
    case PD_PaddleDType::FLOAT32:
      return PD_DataType::PD_FLOAT32;
    case PD_PaddleDType::INT32:
      return PD_DataType::PD_INT32;
    case PD_PaddleDType::INT64:
      return PD_DataType::PD_INT64;
    case PD_PaddleDType::UINT8:
      return PD_DataType::PD_UINT8;
    default:
      PADDLE_ENFORCE(false, "Unsupport place.");
      return PD_DataType::PD_UNKDTYPE;
  }
  PADDLE_ENFORCE(false, "Unsupport place.");
  return PD_DataType::PD_UNKDTYPE;
}

PD_ACPrecision ConvertToACPrecision(Precision dtype) {
  switch (dtype) {
    case Precision::kFloat32:
      return PD_ACPrecision::kFloat32;
    case Precision::kInt8:
      return PD_ACPrecision::kInt8;
    case Precision::kHalf:
      return PD_ACPrecision::kHalf;
    default:
      PADDLE_ENFORCE(false, "Unsupport place.");
      return PD_ACPrecision::kFloat32;
  }
  PADDLE_ENFORCE(false, "Unsupport place.");
  return PD_ACPrecision::kFloat32;
}

}  // extern "C"
