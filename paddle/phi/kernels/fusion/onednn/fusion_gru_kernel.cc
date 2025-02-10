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

#include "paddle/common/errors.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/utils/data_type.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi::fusion {

using phi::OneDNNContext;
using phi::funcs::CreateKey;
using phi::funcs::OneDNNGetDataType;
using phi::funcs::OneDNNMemDesc;
using phi::funcs::RNNReorderType;
using OneDNNMemoryFormat = dnnl::memory::format_tag;

template <typename T, typename T_out = T>
class GRUOneDNNHandler
    : public phi::funcs::OneDNNHandlerT<T, dnnl::gru_forward> {
 public:
  GRUOneDNNHandler(const OneDNNContext& dev_ctx,
                   const dnnl::engine onednn_engine,
                   phi::Place cpu_place UNUSED,
                   const phi::DenseTensor* input,
                   const phi::DenseTensor* weight_h,
                   const phi::DenseTensor* h0,
                   const bool is_reverse,
                   const float scale_data,
                   const float shift_data,
                   const std::string& gate_activation,
                   const std::string& activation,
                   const std::vector<float>& scale_weights,
                   const int64_t N,
                   const int64_t Ti,
                   const int64_t IC,
                   const int64_t OC)
      : phi::funcs::OneDNNHandlerT<T, dnnl::gru_forward>(
            dev_ctx,
            dev_ctx.GetEngine(),
            cpu_place,
            CreateKey(dev_ctx,
                      dev_ctx.GetInputsName("X")[0] +
                          dev_ctx.GetInputsName("WeightH")[0],
                      OneDNNGetDataType<T>(),
                      Ti)),
        N(N),
        Ti(Ti),
        IC(IC),
        OC(OC),
        G(3) {
    std::string unique_name =
        dev_ctx.GetInputsName("X")[0] + dev_ctx.GetInputsName("WeightH")[0];
    // Create memory key without Ti because weights, bias and h0 memories
    // do not depend on Ti size but primitive and input/output memory do
    memory_key_ = phi::funcs::ExtendKeyWithThreadInfoIfNeeded(
        dev_ctx, CreateKey(dev_ctx, unique_name, OneDNNGetDataType<T>()));
    // Is it int8 kernel
    const bool is_INT8 = std::is_same<T, uint8_t>::value;
    if (is_INT8) {
      const int weights_scale_mask =
          0 +
          (1 << 3)  // bit, indicating the unique scales for `g` dim in `ldigo`
          +
          (1 << 4);  // bit, indicating the unique scales for `o` dim in `ldigo`

      attr_.set_rnn_data_qparams(scale_data, shift_data);
      attr_.set_rnn_weights_qparams(weights_scale_mask, scale_weights);
    }

    if (unlikely(!this->isCached())) {
      // oneDNN kernel has hardcoded activation functions
      PADDLE_ENFORCE_EQ(
          gate_activation,
          "sigmoid",
          common::errors::Unimplemented(
              "oneDNN fusion_gru supports only sigmoid as a gate activation."));
      PADDLE_ENFORCE_EQ(
          activation,
          "tanh",
          common::errors::Unimplemented(
              "oneDNN fusion_gru supports only tanh as an activation."));

      // Weights for int8 kernel are of a type s8
      const auto weights_dt =
          is_INT8 ? dnnl::memory::data_type::s8 : OneDNNGetDataType<T>();

      // oneDNN RNN dimensions
      const int64_t D = 1;  // Directions
      const int64_t L = 1;  // Layers (PP supports only 1 stacked layer)
      const int64_t G = 3;  // Number of Gates, 3 for GRU

      // Create memory descriptors
      auto input_md = OneDNNMemDesc(
          {Ti, N, IC}, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);
      auto weight_x_md =
          OneDNNMemDesc({L, D, IC, G, OC}, weights_dt, OneDNNMemoryFormat::any);
      auto weight_h_md =
          OneDNNMemDesc({L, D, OC, G, OC}, weights_dt, OneDNNMemoryFormat::any);
      auto bias_md = OneDNNMemDesc(
          {L, D, G, OC}, OneDNNGetDataType<float>(), OneDNNMemoryFormat::ldgo);
      auto hidden_md = OneDNNMemDesc(
          {Ti, N, OC}, OneDNNGetDataType<T_out>(), OneDNNMemoryFormat::any);
      auto h0_md = OneDNNMemDesc(
          {L, D, N, OC}, OneDNNGetDataType<T>(), OneDNNMemoryFormat::ldnc);

      // Create GRU oneDNN primitive
      const auto direction =
          is_reverse ? dnnl::rnn_direction::unidirectional_right2left
                     : dnnl::rnn_direction::unidirectional_left2right;

      this->AcquireForwardPrimitiveDescriptor(
          this->attr_,
          dnnl::prop_kind::forward_inference,
          direction,
          input_md,
          h0_md,
          weight_x_md,
          weight_h_md,
          bias_md,
          hidden_md,
          dnnl::memory::desc());
    }
  }

  bool is_NTC() { return this->is_NTC(this->fwd_pd_->dst_desc()); }

  bool is_NTC(const dnnl::memory::desc& md) {
    auto ntc_md = dnnl::memory::desc(
        md.get_dims(), md.get_data_type(), dnnl::memory::format_tag::ntc);
    return md == ntc_md;
  }

  void reorderRNNdata(void* input_data,
                      void* output_data,
                      std::vector<size_t> lod,
                      const bool is_reverse,
                      RNNReorderType reorder_type) {
    switch (reorder_type) {
      // Reorder input memory [WORDS, C] + LoD -> [N, T, C]
      case RNNReorderType::PP_NTC: {
        auto* input_data_iter = reinterpret_cast<T*>(input_data);
        auto* output_data_iter = reinterpret_cast<T*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]) * IC;
          const auto offset = is_reverse ? (Ti * IC - num_elements) : 0;
          memcpy(output_data_iter + n * Ti * IC + offset,
                 input_data_iter,
                 sizeof(T) * num_elements);
          input_data_iter += num_elements;
        }
      } break;
      // Reorder input memory [WORDS, C] + LoD -> [T, N, C]
      case RNNReorderType::PP_TNC: {
        auto* input_data_iter = reinterpret_cast<T*>(input_data);
        auto* output_data_iter = reinterpret_cast<T*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]);
          const auto offset = is_reverse ? (Ti - num_elements) : 0;
          for (size_t t = 0; t < num_elements; ++t) {
            memcpy(output_data_iter + (t + offset) * N * IC + n * IC,
                   input_data_iter,
                   sizeof(T) * IC);
            input_data_iter += IC;
          }
        }
      } break;
      // Reorder output values to PP format [N, T, C] -> [WORDS, C]
      case RNNReorderType::NTC_PP: {
        auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
        auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]) * OC;
          const auto offset = is_reverse ? (Ti * OC - num_elements) : 0;
          memcpy(output_data_iter,
                 input_data_iter + n * Ti * OC + offset,
                 sizeof(T_out) * num_elements);
          output_data_iter += num_elements;
        }
      } break;
      // Reorder output values to PP format [T, N, C] -> [WORDS, C]
      case RNNReorderType::TNC_PP: {
        auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
        auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = lod[n + 1] - lod[n];
          const auto offset = is_reverse ? (Ti - num_elements) : 0;
          for (size_t t = 0; t < num_elements; ++t) {
            memcpy(output_data_iter,
                   input_data_iter + (t + offset) * N * OC + n * OC,
                   sizeof(T_out) * OC);
            output_data_iter += OC;
          }
        }
      } break;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireInputMemoryWithReorder(
      const phi::DenseTensor* input, const bool is_reverse) {
    const auto name = this->key_ + "@input_mem";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(name));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->src_desc(),
                                                this->engine_);
      this->dev_ctx_.SetBlob(name, memory_p);
    }

    const auto& input_lod = input->lod()[0];
    auto* x_data = phi::funcs::to_void_cast(input->data<T>());

    auto* x_onednn_data = memory_p->get_data_handle();
    memset(x_onednn_data, 0, sizeof(T) * N * Ti * IC);

    if (is_NTC(this->fwd_pd_->src_desc())) {
      reorderRNNdata(
          x_data, x_onednn_data, input_lod, is_reverse, RNNReorderType::PP_NTC);
    } else {
      reorderRNNdata(
          x_data, x_onednn_data, input_lod, is_reverse, RNNReorderType::PP_TNC);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireOutputMemory() {
    const auto name = this->key_ + "@output_mem";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(name));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->dst_desc(),
                                                this->engine_);
      this->dev_ctx_.SetBlob(name, memory_p);
    }
    return memory_p;
  }

  // H0 is for now persistable
  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireH0Memory(const phi::DenseTensor* h0) {
    const std::string h0_key = memory_key_ + "@h0";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(h0_key));

    if (!memory_p) {
      auto user_h0_memory = dnnl::memory();
      if (h0) {
        user_h0_memory = dnnl::memory(
            {{1, 1, N, OC}, OneDNNGetDataType<U>(), OneDNNMemoryFormat::ldnc},
            this->engine_,
            phi::funcs::to_void_cast(h0->data<U>()));
      } else {
        user_h0_memory = dnnl::memory(
            {{1, 1, N, OC}, OneDNNGetDataType<U>(), OneDNNMemoryFormat::ldnc},
            this->engine_);
        memset(user_h0_memory.get_data_handle(), 0, sizeof(U) * N * OC);
      }
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->src_iter_desc(),
                                                this->engine_);

      auto& astream = phi::OneDNNContext::tls().get_stream();
      dnnl::reorder(user_h0_memory, *memory_p, attr_)
          .execute(astream, user_h0_memory, *memory_p);

      this->dev_ctx_.SetBlob(h0_key, memory_p);
    }
    return memory_p;
  }

  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireWeightXMemory(
      const phi::DenseTensor* weight_x, const bool origin_mode) {
    const std::string wx_key = this->memory_key_ + "@weight_x";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wx_key));

    if (!memory_p) {
      auto user_md = OneDNNMemDesc({1, 1, this->IC, this->G, this->OC},
                                   OneDNNGetDataType<U>(),
                                   OneDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      auto* weight_x_data = reinterpret_cast<U*>(user_memory.get_data_handle());
      memcpy(weight_x_data,
             weight_x->data<U>(),
             sizeof(U) * this->IC * this->G * this->OC);

      if (origin_mode == false) {
        for (int64_t i = 0; i < this->IC; ++i) {
          for (int64_t j = 0; j < this->OC; ++j) {
            U minus_one(-1.0f);
            weight_x_data[j] = minus_one * weight_x_data[j];
          }
          weight_x_data += 3 * this->OC;
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_layer_desc(), this->engine_);

      auto& astream = OneDNNContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, this->attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wx_key, memory_p);
    }
    return memory_p;
  }

  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireWeightHMemory(
      const phi::DenseTensor* weight_h, const bool origin_mode) {
    const std::string wh_key = this->memory_key_ + "@weight_h";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wh_key));

    if (!memory_p) {
      auto user_md = OneDNNMemDesc({1, 1, this->OC, this->G, this->OC},
                                   OneDNNGetDataType<U>(),
                                   OneDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      // Reorder weights_h from PP format [OC, 2OC] + [OC, OC] to
      // oneDNN format [OC, 3OC]
      auto* weight_h_data = reinterpret_cast<U*>(user_memory.get_data_handle());
      auto* user_weight_h_data = weight_h->data<U>();

      auto src1_iter = user_weight_h_data;
      auto src2_iter = user_weight_h_data + 2 * this->OC * this->OC;

      for (int64_t c = 0; c < this->OC; ++c) {
        memcpy(weight_h_data, src1_iter, 2 * this->OC * sizeof(U));
        memcpy(weight_h_data + 2 * this->OC, src2_iter, this->OC * sizeof(U));

        src1_iter += 2 * this->OC;
        src2_iter += this->OC;
        weight_h_data += 3 * this->OC;
      }

      weight_h_data = reinterpret_cast<U*>(user_memory.get_data_handle());

      if (origin_mode == false) {
        for (int64_t i = 0; i < this->OC; ++i) {
          for (int64_t j = 0; j < this->OC; ++j) {
            U minus_one(-1.0f);
            weight_h_data[j] = minus_one * weight_h_data[j];
          }
          weight_h_data += 3 * this->OC;
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_iter_desc(), this->engine_);

      auto& astream = OneDNNContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, this->attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wh_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemory(const phi::DenseTensor* bias,
                                                  const bool origin_mode) {
    const std::string bias_key = this->memory_key_ + "@bias";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(bias_key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->bias_desc(),
                                                this->engine_);
      auto* bias_data = reinterpret_cast<float*>(memory_p->get_data_handle());
      if (bias) {
        const float* user_bias_data =
            bias->data<float>();  // Bias in oneDNN is always float
        memcpy(bias_data, user_bias_data, sizeof(float) * this->G * this->OC);
      } else {
        // oneDNN always need bias memory, if it's not provided in PP, let
        // oneDNN allocate memory and set it to 0
        memset(bias_data, 0, sizeof(float) * this->G * this->OC);
      }

      if (origin_mode == false && bias) {
        for (int64_t i = 0; i < this->OC; ++i) {
          bias_data[i] *= -1;
        }
      }
      this->dev_ctx_.SetBlob(bias_key, memory_p);
    }
    return memory_p;
  }

 protected:
  // RNN dimensions
  // N - Batch Size
  // Ti - Max sentence length
  // IC - Input Channels
  // OC - Output Channels
  // G  - Number of gates
  const int64_t N, Ti, IC, OC, G;

  // Memory size of weights, bias and h0 does not depend
  // on Ti size, thus we need another key to cache them
  std::string memory_key_;
  dnnl::primitive_attr attr_;
};

template <typename T, typename Tout = T>
void RunKernel(const phi::OneDNNContext& dev_ctx,
               const DenseTensor& x,
               const paddle::optional<DenseTensor>& h0,
               const DenseTensor& weight_x,
               const DenseTensor& weight_h,
               const paddle::optional<DenseTensor>& bias,
               const std::string& activation,
               const std::string& gate_activation,
               const bool is_reverse,
               const bool use_seq,
               const bool origin_mode,
               const float scale_data,
               const float shift_data,
               const std::vector<float>& scale_weights,
               DenseTensor* reordered_h0,
               DenseTensor* xx,
               DenseTensor* batched_input,
               DenseTensor* batched_out,
               DenseTensor* hidden) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto x_dims = x.dims();
  auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                        ? common::flatten_to_2d(x_dims, 1)
                        : x_dims;

  // Get tensor dimensions
  const auto x_mat_dims_vec = common::vectorize(x_mat_dims);
  const auto weight_h_dims = common::vectorize(weight_h.dims());
  const auto& input_lod = x.lod()[0];

  // Calculate RNN dimensions
  const int64_t N = static_cast<int64_t>(input_lod.size() -
                                         1);  // Number of sentences (batches)
  const int64_t Ti =  // Max length of the sentence in a batch
      [&input_lod]() {
        size_t res = 0;
        for (size_t i = 0; i < (input_lod.size() - 1); ++i) {
          res = std::max(res, input_lod[i + 1] - input_lod[i]);
        }
        return res;
      }();
  const int64_t IC = x_mat_dims_vec[1];  // Input channels
  const int64_t OC = weight_h_dims[0];   // Output channels

  GRUOneDNNHandler<T, Tout> handler(dev_ctx,
                                    onednn_engine,
                                    dev_ctx.GetPlace(),
                                    &x,
                                    &weight_h,
                                    h0.get_ptr(),
                                    is_reverse,
                                    scale_data,
                                    shift_data,
                                    gate_activation,
                                    activation,
                                    scale_weights,
                                    N,
                                    Ti,
                                    IC,
                                    OC);
  auto input_memory_p = handler.AcquireInputMemoryWithReorder(&x, is_reverse);

  std::shared_ptr<dnnl::memory> h0_memory_p, weight_h_memory_p,
      weight_x_memory_p;

  if (phi::TransToProtoVarType(weight_h.dtype()) == phi::ProtoDataType::FP32) {
    h0_memory_p = handler.template AcquireH0Memory<float>(h0.get_ptr());
    weight_x_memory_p =
        handler.template AcquireWeightXMemory<float>(&weight_x, origin_mode);
    weight_h_memory_p =
        handler.template AcquireWeightHMemory<float>(&weight_h, origin_mode);
  } else if (phi::TransToProtoVarType(weight_h.dtype()) ==
             phi::ProtoDataType::BF16) {
    h0_memory_p =
        handler.template AcquireH0Memory<phi::dtype::bfloat16>(h0.get_ptr());
    weight_x_memory_p =
        handler.template AcquireWeightXMemory<phi::dtype::bfloat16>(
            &weight_x, origin_mode);
    weight_h_memory_p =
        handler.template AcquireWeightHMemory<phi::dtype::bfloat16>(
            &weight_h, origin_mode);
  } else {
    h0_memory_p = handler.template AcquireH0Memory<uint8_t>(h0.get_ptr());
    weight_x_memory_p =
        handler.template AcquireWeightXMemory<int8_t>(&weight_x, origin_mode);
    weight_h_memory_p =
        handler.template AcquireWeightHMemory<int8_t>(&weight_h, origin_mode);
  }

  auto bias_memory_p = handler.AcquireBiasMemory(bias.get_ptr(), origin_mode);
  auto hidden_onednn_memory_p = handler.AcquireOutputMemory();

  std::unordered_map<int, dnnl::memory> gru_args = {
      {DNNL_ARG_SRC_LAYER, *input_memory_p},
      {DNNL_ARG_SRC_ITER, *h0_memory_p},
      {DNNL_ARG_WEIGHTS_LAYER, *weight_x_memory_p},
      {DNNL_ARG_WEIGHTS_ITER, *weight_h_memory_p},
      {DNNL_ARG_BIAS, *bias_memory_p},
      {DNNL_ARG_DST_LAYER, *hidden_onednn_memory_p}};

  auto gru_forward_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  gru_forward_p->execute(astream, gru_args);
  astream.wait();

  auto* hidden_onednn_data = hidden_onednn_memory_p->get_data_handle();
  auto* hidden_tmp_data = dev_ctx.template Alloc<Tout>(hidden);
  auto* hidden_data = phi::funcs::to_void_cast(hidden_tmp_data);
  if (handler.is_NTC()) {
    handler.reorderRNNdata(hidden_onednn_data,
                           hidden_data,
                           input_lod,
                           is_reverse,
                           RNNReorderType::NTC_PP);
  } else {
    handler.reorderRNNdata(hidden_onednn_data,
                           hidden_data,
                           input_lod,
                           is_reverse,
                           RNNReorderType::TNC_PP);
  }
}

template <typename T, typename Context>
void FusionGRUKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& h0,
                     const DenseTensor& weight_x,
                     const DenseTensor& weight_h,
                     const paddle::optional<DenseTensor>& bias,
                     const std::string& activation,
                     const std::string& gate_activation,
                     const bool is_reverse,
                     const bool use_seq,
                     const bool origin_mode,
                     const bool force_fp32_output,
                     DenseTensor* reordered_h0,
                     DenseTensor* xx,
                     DenseTensor* batched_input,
                     DenseTensor* batched_out,
                     DenseTensor* hidden) {
  const std::string mkldnn_data_type =
      dev_ctx.HasDnnAttr("mkldnn_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("mkldnn_data_type"))
          : "float32";
  std::vector<std::string> mkldnn_data_type_list = {
      "float32", "int8", "bfloat16"};
  PADDLE_ENFORCE_EQ(std::find(mkldnn_data_type_list.begin(),
                              mkldnn_data_type_list.end(),
                              mkldnn_data_type) != mkldnn_data_type_list.end(),
                    true,
                    common::errors::InvalidArgument(
                        "The mkldnn_data_type should be [float32, "
                        "int8, bfloat16], but found %s.",
                        mkldnn_data_type.c_str()));
  const float scale_data =
      dev_ctx.HasDnnAttr("Scale_data")
          ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_data"))
          : 1.0f;
  const float shift_data =
      dev_ctx.HasDnnAttr("Shift_data")
          ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Shift_data"))
          : 1.0f;
  std::vector<float> tmp_scale_weights = {1.0f};
  const std::vector<float> scale_weights =
      dev_ctx.HasDnnAttr("Scale_weights")
          ? PADDLE_GET_CONST(std::vector<float>,
                             dev_ctx.GetDnnAttr("Scale_weights"))
          : tmp_scale_weights;
  const bool is_bf16 = std::is_same<T, phi::dtype::bfloat16>::value;
  // BF16 does not support force output
  if (!is_bf16 && force_fp32_output) {  // NOLINT
    RunKernel<T, float>(dev_ctx,
                        x,
                        h0,
                        weight_x,
                        weight_h,
                        bias,
                        activation,
                        gate_activation,
                        is_reverse,
                        use_seq,
                        origin_mode,
                        scale_data,
                        shift_data,
                        scale_weights,
                        reordered_h0,
                        xx,
                        batched_input,
                        batched_out,
                        hidden);
  } else {
    RunKernel<T>(dev_ctx,
                 x,
                 h0,
                 weight_x,
                 weight_h,
                 bias,
                 activation,
                 gate_activation,
                 is_reverse,
                 use_seq,
                 origin_mode,
                 scale_data,
                 shift_data,
                 scale_weights,
                 reordered_h0,
                 xx,
                 batched_input,
                 batched_out,
                 hidden);
  }
}

}  // namespace phi::fusion

PD_REGISTER_KERNEL(fusion_gru,
                   OneDNN,
                   ONEDNN,
                   phi::fusion::FusionGRUKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t) {}
