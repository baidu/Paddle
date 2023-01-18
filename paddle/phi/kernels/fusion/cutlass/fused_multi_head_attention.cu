// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/fluid/memory/malloc.h"

#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/kernel_forward.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

struct LaunchParams {
    // meta params
    phi::DataType datatype; 

    // Input tensors
    const void* query_ptr; // [num_batches, num_heads, query_seq_len, head_size]
    const void* key_ptr; // [num_batches, num_heads, key_value_seq_len, value_head_size]
    const void* mask_ptr = nullptr; // [num_batches, num_heads, key_value_seq_len, query_seq_len] and it can be broadcasted in axis0/1/2. 
    const void* value_ptr; // [num_batches, num_heads, key_value_seq_len, value_head_size]
    
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;

    // Output tensors
    void* output_ptr; // [num_batches, num_heads, query_seq_len, head_size]
    void* output_accum_ptr; // [num_batches, num_heads, query_seq_len, head_size]
    void* logsumexp_ptr; // [num_batches, num_heads, num_queries] - can be null

    // Scale
    float scale;

    // Dimensions/strides
    int32_t num_batches;
    int32_t num_heads;
    int32_t query_seq_len; 
    int32_t key_value_seq_len; 
    int32_t head_size;
    int32_t value_head_size;
    bool causal;
    bool mask_broadcast_row;
    /*
    We can understand the computation of Fused Multihead Attention in this way: 
    for Query matmul Key, we execute num_batches * num_heads times matmul, 
    each matmul problem is: (M, K) (K, N) -> (M, N). 
    Here M is: query_seq_len, K is: head_size, N is: key_value_seq_len. 
    The stride concept is equals to torch's, it means the offset to move to next axis. 
    For Q matrix(M, K), we need to move K(which equals to head_size) offset to next row(in M axis), 
    so here query_strideM equals to K = head_size. 
    */ 
    int32_t query_strideM; 
    int32_t key_strideM;
    int32_t value_strideM;
    // Since bias can be broadcasted, we need to assign each stride 
    int32_t mask_strideM;
    int64_t mask_strideH; // stride for num_heads
    int64_t mask_strideB; // stride for num_batches
}; 

template<typename T, typename ArchTag, bool IsAligned, int QueriesPerBlock, int KeysPerBlock, bool SingleValueIteration> 
void LaunchMultiHeadAttentionKernel(LaunchParams params, const phi::GPUContext& ctx){
    using Attention = AttentionKernel<T, ArchTag, IsAligned, QueriesPerBlock, KeysPerBlock, SingleValueIteration>; 

    typename Attention::Params p;
    { // set parameters
      p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
      p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
      p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
      p.mask_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.mask_ptr));

      // TODO(zhengzekang): Currently we only support inference, so here we set `logsumexp_ptr` as nullptr, which is used for backward. 
      p.logsumexp_ptr = nullptr; 

      p.output_accum_ptr = nullptr;
      if (Attention::kNeedsOutputAccumulatorBuffer) {
        const int64_t output_size = params.num_batches * params.num_heads * params.query_seq_len * params.head_size; 
        paddle::memory::AllocationPtr tmp_output_accum_buffer_ptr{nullptr};
        tmp_output_accum_buffer_ptr = paddle::memory::Alloc(
            ctx.GetPlace(),
            output_size * sizeof(typename Attention::output_accum_t), 
            phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream()))
        );
        p.output_accum_ptr = reinterpret_cast<typename Attention::output_accum_t*>(tmp_output_accum_buffer_ptr->ptr()); 
      }

      p.output_ptr = reinterpret_cast<T*>(params.output_ptr);

      // TODO: support arbitrary seq lengths
      // if (cu_seqlens_q.has_value()) {
      //   p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      //   p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
      // }

      p.num_batches = params.num_batches;
      p.num_heads = params.num_heads;
      p.num_queries = params.query_seq_len;
      p.num_keys = params.key_value_seq_len;
      p.head_dim = params.head_size;
      p.head_dim_value = params.value_head_size;

      p.scale = params.scale; 
      p.causal = params.causal;
      p.mask_broadcast_row = params.mask_broadcast_row; 

      // TODO: This might overflow for big tensors
      p.q_strideM = params.query_strideM;
      p.k_strideM = params.key_strideM;
      p.mask_strideM = params.mask_strideM;
      p.v_strideM = params.value_strideM;

      p.q_strideH = p.q_strideM * params.query_seq_len;
      p.k_strideH = p.k_strideM * params.key_value_seq_len;
      p.mask_strideH = params.mask_strideH;
      p.v_strideH = p.v_strideM * params.key_value_seq_len;
      p.o_strideH = params.value_head_size * params.query_seq_len;

      p.q_strideB = p.q_strideH * params.num_heads;
      p.k_strideB = p.k_strideH * params.num_heads;
      p.mask_strideB = params.mask_strideB;
      p.v_strideB = p.v_strideH * params.num_heads;
      p.o_strideB = params.value_head_size * params.query_seq_len * params.num_heads;
    }

    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    if (!Attention::check_supported(p)) {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          phi::errors::Unimplemented("The Params is not supported by cutlass fused multihead attention. "));
      return; 
    }
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, ctx.stream()>>>(p);
}

template<typename T, typename ArchTag, bool IsAligned, int QueriesPerBlock, int KeysPerBlock>
void DispatchFMHASingleValueIteration(LaunchParams params, const phi::GPUContext& ctx){
    if (params.value_head_size <= KeysPerBlock) {
        LaunchMultiHeadAttentionKernel<T, ArchTag, IsAligned, QueriesPerBlock, KeysPerBlock, true>(params, ctx); 
    } else {
        LaunchMultiHeadAttentionKernel<T, ArchTag, IsAligned, QueriesPerBlock, KeysPerBlock, false>(params, ctx); 
    }
}

template<typename T, typename ArchTag, bool IsAligned>
void DispatchFMHABlockSize(LaunchParams params, const phi::GPUContext& ctx){

    if(params.value_head_size > 64){
        DispatchFMHASingleValueIteration<T, ArchTag, IsAligned, 32, 128>(params, ctx); 
    } else {
        DispatchFMHASingleValueIteration<T, ArchTag, IsAligned, 64, 64>(params, ctx); 
    } 
}

template<typename T, typename ArchTag>
void DispatchFMHAIsAligned(LaunchParams params, const phi::GPUContext& ctx){
    if(reinterpret_cast<uintptr_t>(params.query_ptr) % 16 == 0
      && reinterpret_cast<uintptr_t>(params.key_ptr) % 16 == 0
      && reinterpret_cast<uintptr_t>(params.value_ptr) % 16 == 0
      && params.query_strideM % (16 / sizeof(T)) == 0
      && params.query_strideM % (16 / sizeof(T)) == 0
      && params.value_strideM % (16 / sizeof(T)) == 0){
        DispatchFMHABlockSize<T, ArchTag, true>(params, ctx); 
    } else {
        DispatchFMHABlockSize<T, ArchTag, false>(params, ctx); 
    }
}

template<typename T>
void DispatchFMHAArchTag(LaunchParams params, const phi::GPUContext& ctx){
    const int compute_capability = ctx.GetComputeCapability(); 
    printf("Compute capability is: %d \n", compute_capability); 
    if(compute_capability == 80){
        DispatchFMHAIsAligned<T, cutlass::arch::Sm80>(params, ctx); 
    } else if (compute_capability == 75) {
        DispatchFMHAIsAligned<T, cutlass::arch::Sm75>(params, ctx); 
    } else if (compute_capability == 70) {
        DispatchFMHAIsAligned<T, cutlass::arch::Sm70>(params, ctx); 
    } else {
        PADDLE_ENFORCE_EQ(
          true,
          false,
          phi::errors::Unimplemented("Currently cutlass fused multihead attention kernel only support arch: SM80, SM75, SM70"));
        return; 
    }
}

void DispatchFusedMultiheadAttentionKernel(LaunchParams params, const phi::GPUContext& ctx){
    if(params.datatype == DataType::FLOAT32){
        return DispatchFMHAArchTag<cutlass::tfloat32_t>(params, ctx);
    } else if (params.datatype == DataType::FLOAT16) {
        return DispatchFMHAArchTag<cutlass::half_t>(params, ctx);
    } else {
        PADDLE_ENFORCE_EQ(
          true,
          false,
          phi::errors::Unimplemented("Currently cutlass fused multihead attention kernel only support datatype: float32 and float16. "));
        return; 
    }
}

template <typename T, typename Context>
void MultiHeadAttentionForwardKernel(const Context& ctx,
                                     const DenseTensor& query,
                                     const DenseTensor& key,
                                     const DenseTensor& value,
                                    //  const paddle::optional<DenseTensor>& mask,
                                     const DenseTensor& mask, 
                                     const float scale, 
                                     const bool causal, 
                                     DenseTensor* output) {
    ctx.template Alloc<T>(output);
    LaunchParams params{}; 

    params.datatype = query.dtype(); 
    params.query_ptr = query.data(); 
    params.key_ptr = key.data(); 
    
    // TODO(zhengzekang): Check optional.  
    params.mask_ptr = mask.data(); 
    // params.mask_ptr = nullptr; 

    params.value_ptr = value.data(); 
    params.output_ptr = output->data();
    params.output_accum_ptr = nullptr;

    // TODO(zhengzekang): currently we only used in inference. Maybe add a bool flag to save it ?
    params.logsumexp_ptr = nullptr;

    params.num_batches = query.dims()[0]; 
    params.num_heads = query.dims()[1]; 
    params.query_seq_len = query.dims()[2]; 
    params.key_value_seq_len = key.dims()[2]; 
    params.head_size = query.dims()[3]; 
    params.value_head_size = value.dims()[3];

    float scale_value = sqrt(params.head_size); 
    if(scale != 0.0f){
        // assume 0.0f is default value. 
        scale_value = scale; 
    }
    params.scale = scale_value;  
    params.causal = causal; 

    params.query_strideM = query.dims()[3]; 
    params.key_strideM = key.dims()[3]; 
    params.value_strideM = value.dims()[3]; 
    params.mask_strideM = mask.dims()[2] == 1 ? 0 : mask.dims()[3]; 
    params.mask_strideH = mask.dims()[1] == 1 ? 0 : params.mask_strideM * params.query_seq_len; 
    params.mask_strideB = mask.dims()[0] == 1 ? 0 : params.mask_strideH * params.num_heads; 
    params.mask_broadcast_row = false; 
    if(params.mask_strideM == 0){
        params.mask_broadcast_row = true; 
    }
    printf("Bias stride M is: %d \n", int32_t(params.mask_strideM)); 
    printf("Bias stride H is: %d \n", int32_t(params.mask_strideH)); 
    printf("Bias stride B is: %d \n", int32_t(params.mask_strideB)); 
    DispatchFusedMultiheadAttentionKernel(params, ctx); 
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::MultiHeadAttentionForwardKernel,
                   float,
                   phi::dtype::float16) {}
