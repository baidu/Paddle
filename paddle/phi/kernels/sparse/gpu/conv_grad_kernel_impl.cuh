#pragma once

#ifdef PADDLE_WITH_CUDA

#include <cuda_fp16.h>
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/sparse/gpu/conv_memory_utils.cuh"
#include "paddle/phi/kernels/sparse/gpu/conv_kernel_impl_utils.cuh"

// conv_backward_cuda_m16n16k64_m16n16k64_m16n16k16_f16f16f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode0_f16f16f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, half *__restrict__ A, half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C)
{
  int j_factors1 = (N + 15) / 16 / 1;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original + 15) / 16 * kernel_volume * j_factors1);

  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_warp[8];
  __shared__ half A_shared[2560];
  __shared__ half B_shared[2560];
  half A_shared_warp[8];
  half B_shared_warp[8];
  half *cur_C = C + blockIdx_z * kernel_volume * K_original * N;
  for (int i = 0; i < 8; ++i)
  {
    C_warp[0 + i] = 0.0;
  };

  // hoisting shared pointer offsets
  // int *out_in_map_ptr = out_in_map + (threadIdx.y * 16 + threadIdx.x / 2) * kernel_volume + ((threadIdx.y * 256) % 16) / K_original + ((threadIdx.x * 8) % 16) / K_original + (blockIdx_y / j_factors1 * 16) / K_original;
  int *out_in_map_ptr = out_in_map + (threadIdx.y * 16 + threadIdx.x / 2) * kernel_volume + ((threadIdx.y * 256) % 16) / K_tile_padded + ((threadIdx.x * 8) % 16) / K_tile_padded + (blockIdx_y / j_factors1 * 16) / K_tile_padded;
  // half *A_ptr = A + ((threadIdx.y * 256 % 16) % K_original) + ((threadIdx.x * 8 % 16) % K_original) + ((blockIdx_y / j_factors1 * 16) % K_original);
  half *A_ptr = A + ((threadIdx.y * 256 % 16) % K_tile_padded) + ((threadIdx.x * 8 % 16) % K_tile_padded) + ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
  half *B_ptr = B + (blockIdx_y % j_factors1) * 16 + (threadIdx.x * 8) % 16;
  int reorder_offset = threadIdx.y * 256 / 16 + threadIdx.x * 8 / 16;
  // half *C_ptr = cur_C + blockIdx_x / 1 * 108 * N / 16 * 256 + blockIdx_y / j_factors1 * 1 * N / 16 * 256 + (threadIdx.y % 1) * 1 * N / 16 * 256 + (blockIdx_x % 1) * j_factors1 * 16 + (blockIdx_y % j_factors1) * 16 + threadIdx.y / 1 * 16 + (threadIdx.x % 4) * 2 + (threadIdx.x / 4) * N;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  // int kernel_offset = (blockIdx_y / j_factors1) / (K_original / 16);
  int kernel_offset = (blockIdx_y / j_factors1) / ((K_original + K_tile - 1) / K_tile);
  int cur_C_ic_start = (blockIdx_y / j_factors1 * 16) % K_tile_padded + (threadIdx.x / 4);
  int cur_C_oc_start = (blockIdx_y % j_factors1) * 16 + threadIdx.y / 1 * 16 + (threadIdx.x % 4) * 2;
  half *C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check)
  {
    int A_ld_start = ((threadIdx.y * 256 % 16) % K_tile_padded) + ((threadIdx.x * 8 % 16) % K_tile_padded) + ((blockIdx_y / j_factors1 * 16) % K_tile_padded);
    int A_ld_amount = min(A_ld_start + 8, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 2);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    A_pred_guard = 1;
  if constexpr (N_ld_check)
  {
    int B_ld_start = (blockIdx_y % j_factors1) * 16 + (threadIdx.x * 8) % 16;
    int B_ld_amount = min(B_ld_start + 8, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 2);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    B_pred_guard = 1;

  // int A_pred_guard = 1;
  // int B_pred_guard = 1;

  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      // int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 16 * kernel_volume + (ax0_ax1_fused_0 * 256 % 16) / K_original];
      int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 16 * kernel_volume + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded];

      if (input_idx != -1)
      {
        //*(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) =
        //    *(uint4 *)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_original));
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = A_loaded;
      }
      else
      {
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;
      int v0 = reorder_offset_inner;
      //*(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) =
      //    *(uint4 *)(B_ptr + v0 * N);
      uint4 B_loaded = make_uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
      *(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = B_loaded;
    }
    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#elif __CUDA_ARCH__ >= 750
      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
    }
  }
  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    if (i2_0_0 >= (M_fwd + 63) / 64)
      break;

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 16 * kernel_volume + (ax0_ax1_fused_0 * 256 % 16) / K_tile_padded];

      if (input_idx != -1)
      {
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 256 % 16) % K_tile_padded), A_pred_guard);
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = A_loaded;
      }
      else
      {
        *(uint4 *)(A_shared + (((ax0_ax1_fused_0 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 16;

      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_offset_inner;
        uint4 B_loaded = make_uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
        *(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = B_loaded;
      }
      else
      {
        *(uint4 *)(B_shared + (((ax0_ax1_fused_0_1 * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(i2_0_1 * 640)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
            : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
      }
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#elif __CUDA_ARCH__ >= 750
      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 0))[0]), "=f"(((float *)(C_warp + 0))[1]), "=f"(((float *)(C_warp + 0))[2]), "=f"(((float *)(C_warp + 0))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + 0))[0]), "f"(((float *)(C_warp + 0))[1]), "f"(((float *)(C_warp + 0))[2]), "f"(((float *)(C_warp + 0))[3]));
      }

      {
        __asm__ __volatile__(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
          :  "=f"(((float *)(C_warp + 4))[0]), "=f"(((float *)(C_warp + 4))[1]), "=f"(((float *)(C_warp + 4))[2]), "=f"(((float *)(C_warp + 4))[3])
          : "r"(((unsigned *)(A_shared_warp + 4))[0]), "r"(((unsigned *)(A_shared_warp + 4))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + 4))[0]), "f"(((float *)(C_warp + 4))[1]), "f"(((float *)(C_warp + 4))[2]), "f"(((float *)(C_warp + 4))[3]));
      }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
    }
  }

  for (int local_id = 0; local_id < 8; ++local_id)
  {
    if constexpr (K_ld_check || N_ld_check)
    {
      if (cur_C_ic_start + ((local_id / 2) % 2) * 8 < K_original && cur_C_oc_start + (local_id % 2) + (local_id / 4) * 8 < N)
        C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[0 + local_id]);
    }
    else
    {
      C_ptr[+(((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[0 + local_id]);
    }
  };
}

// conv_backward_cuda_m32n64k64_m32n32k64_m16n16k16_f16f16f32
template <typename IntT>
__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode0_f16f16f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, half *__restrict__ A, half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C)
{
  int j_factors1 = N / 16 / 4;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original * kernel_volume + 31) / 32 * j_factors1);

  float C_warp[32];
  __shared__ half A_shared[2560];
  __shared__ half B_shared[4608];
  half A_shared_warp[16];
  half B_shared_warp[16];
  half *cur_C = C + blockIdx_z * kernel_volume * N * K_original;
  for (int i0_0_3_init = 0; i0_0_3_init < 2; ++i0_0_3_init)
  {
    for (int i1_0_4_init = 0; i1_0_4_init < 2; ++i1_0_4_init)
    {
      for (int i = 0; i < 8; ++i)
      {
        C_warp[((i0_0_3_init * 16) + (i1_0_4_init * 8)) + i] = 0.0;
      };
    }
  }

  // hoisting shared pointer offsets
  int *out_in_map_ptr = out_in_map + (threadIdx.y * 8 + threadIdx.x / 4) * kernel_volume + ((threadIdx.y * 256) % 32) / K_original + ((threadIdx.x * 8) % 32) / K_original + (blockIdx_y / j_factors1 * 32) / K_original;
  half *A_ptr = A + ((threadIdx.y * 256 % 32) % K_original) + ((threadIdx.x * 8 % 32) % K_original) + ((blockIdx_y / j_factors1 * 32) % K_original);
  half *B_ptr = B + (blockIdx_y % j_factors1) * 64 + (threadIdx.x * 8) % 64;
  int reorder_offset = threadIdx.y * 256 / 64 + threadIdx.x * 8 / 64;
  half *C_ptr = cur_C + blockIdx_x / 1 * 108 * N / 16 * 256 + blockIdx_y / j_factors1 * 2 * N / 16 * 256 + (threadIdx.y % 1) * 2 * N / 16 * 256 + (blockIdx_x % 1) * j_factors1 * 64 + (blockIdx_y % j_factors1) * 64 + threadIdx.y / 1 * 32 + (threadIdx.x % 4) * 2 + (threadIdx.x / 4) * N;
  int K_iters = ((M_fwd + 63) / 64 + split_k_iters - 1) / split_k_iters;
  int kernel_offset = (blockIdx_y / j_factors1) / (K_original / 32);
  for (int _i2_0_0 = 0; _i2_0_0 < K_iters - 1; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      // Haotian: NOTE: what if j_factors[0] != 1?
      int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 16 * kernel_volume + (ax0_ax1_fused_0 * 512 % 32) / K_original];

      if (input_idx != -1)
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(uint4 *)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;
      int v0 = reorder_offset_inner;
      *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
          *(uint4 *)(B_ptr + v0 * N);
    }
    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((i2_0_1 * 640) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(((i2_0_1 * 1152) + (((int)threadIdx.y) * 32)) + (ax1_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 72) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if __CUDA_ARCH__ >= 800
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#elif __CUDA_ARCH__ >= 750
          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
    }
  }
  for (int _i2_0_0 = K_iters - 1; _i2_0_0 < K_iters; ++_i2_0_0)
  {
    int i2_0_0 = blockIdx_z + split_k_iters * _i2_0_0;

    if (i2_0_0 >= (M_fwd + 63) / 64)
      break;

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 64 * kernel_volume;
    half *A_ptr_local = A_ptr;
    int reorder_offset_local = reorder_offset + i2_0_0 * 64;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      // related to input
      int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 16 * kernel_volume + (ax0_ax1_fused_0 * 512 % 32) / K_original];

      if (input_idx != -1)
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(uint4 *)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 32) % K_original));
      }
      else
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + ax0_ax1_fused_0_1 * 8;

      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_offset_inner;
        *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
            *(uint4 *)(B_ptr + v0 * N);
      }
      else
      {
        *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 576) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) = make_uint4(0, 0, 0, 0);
      }
    }
    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 4; ++i2_0_1)
    {
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((i2_0_1 * 640) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(((i2_0_1 * 1152) + (((int)threadIdx.y) * 32)) + (ax1_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 72) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0_1 * 8)))[3])
              : "r"(addr));
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 2; ++i0_0_3)
      {
        for (int i1_0_4 = 0; i1_0_4 < 2; ++i1_0_4)
        {
#if __CUDA_ARCH__ >= 800
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#elif __CUDA_ARCH__ >= 750
          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }

          {
            __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
              :  "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#else
  #pragma message("FP16 kernels will not be compiled for SM75-.")
#endif
        }
      }
    }
  }

  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0)
  {

    half *C_ptr_local = C_ptr + ax0_0 * N / 16 * 256;

    for (int ax1_0_2 = 0; ax1_0_2 < 2; ++ax1_0_2)
    {
      for (int local_id = 0; local_id < 8; ++local_id)
      {

        C_ptr_local[ax1_0_2 * 16 + (((local_id / 2) % 2) * 8) * N + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[((ax0_0 * 16) + (ax1_0_2 * 8)) + local_id]);
      };
    }
  }
}

// conv_backward_cuda_m16n16k64_f32f32f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(32) conv_backward_cuda_setting1_mode0_f32f32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, float *__restrict__ A, float *__restrict__ B, int *__restrict__ out_in_map, float *__restrict__ C)
{

  int j_factors1 = (N + 15) / 16;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original + 15) / 16 * kernel_volume * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original + 15) / 16 * kernel_volume * j_factors1);

  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);

  float C_local[8];
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];

  #pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    C_local[i] = 0.0;
  }

  int blockIdx_m = blockIdx_y / j_factors1;
  int blockIdx_n = blockIdx_y % j_factors1;
  int threadIdx_x = (int)threadIdx.x;

  int kernel_offset = blockIdx_m / (K_tile_padded / 16);
  int channel_offset = (blockIdx_m * 16 + ((threadIdx_x * 4) % 16)) % K_tile_padded;
  int K_loops = ((M_fwd + 63 ) / 64 + split_k_iters - 1) / split_k_iters;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                          + (threadIdx_x / (16/4)) * kernel_volume
                          + kernel_offset;
  float * A_ptr = A + channel_offset;

  // reorder is performed on B's rows.
  float * B_ptr = B
                    + (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);
  int reorder_offset = threadIdx_x /(16/4);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);

  float * A_shared_reduce_ptr =  A_shared + (threadIdx_x / 4);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 4);

  // splitK offset
  float * cur_C = C + blockIdx_z * K_original * kernel_volume * N;
  int cur_C_ic_start = (blockIdx_m * 16 + (threadIdx_x / 4)) % K_tile_padded;
  int cur_C_oc_start = blockIdx_n * 16 + (threadIdx_x % 4);
  float * C_ptr = cur_C + (kernel_offset * K_original + cur_C_ic_start) * N + cur_C_oc_start;

  int A_pred_guard = 0;
  int B_pred_guard = 0;
  if constexpr (K_ld_check) // IC % cta_M != 0
  {
    int A_ld_start = channel_offset;
    int A_ld_amount = min(A_ld_start + 4, K_original) - A_ld_start;
    int A_ld_bound = A_ld_amount / (K_ld_factor / 4);

    for (int i = 0; i < A_ld_bound; i++)
      A_pred_guard |= (1 << i);
  }
  else
    A_pred_guard = 1;

  if constexpr (N_ld_check) // OC % cta_N != 0
  {
    int B_ld_start = (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);
    int B_ld_amount = min(B_ld_start + 4, N) - B_ld_start;
    int B_ld_bound = B_ld_amount / (N_ld_factor / 4);

    for (int i = 0; i < B_ld_bound; i++)
      B_pred_guard |= (1 << i);
  }
  else
    B_pred_guard = 1;

  #pragma unroll
  for (int _k_0 = 0; _k_0 < K_loops - 1; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = out_in_map_ptr_local[(ax0_ax1_fused_0 *8) * kernel_volume];
      if (input_idx != -1)
      {
        // *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =  // ax0_ax1_fused_0 * elements loaded in each loop
        //     *(float4*)(A_ptr + (input_idx * K_original));
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr + (input_idx * K_original) , A_pred_guard);
        *(uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = A_loaded;
      }
      else
      {
        *(uint4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = make_uint4(0, 0, 0, 0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 8);
      int v0 = reorder_offset_inner;
      //*(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) =
      //    *(float4*)(B_ptr + v0 * N);
      uint4 B_loaded = make_uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
      *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = B_loaded;
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 16) + ((i / 4) * 8)]
                          * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];
        }

      }
    }
  }
  for (int _k_0 = K_loops - 1; _k_0 < K_loops; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    if (k_0 >= (M_fwd + 63) / 64)
      break;

    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *8) * kernel_volume);
      if (input_idx != -1)
      {
        // *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) =  // ax0_ax1_fused_0 * elements loaded in each loop
        //     *(float4*)(A_ptr + (input_idx * K_original));
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr + (input_idx * K_original) , A_pred_guard);
        *(uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = A_loaded;
      }
      else
      {
        *(uint4*)(A_shared_ptr + (ax0_ax1_fused_0 * 128)) = make_uint4(0, 0, 0, 0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 8; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 8);
      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_offset_inner;
        //*(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) =
        //    *(float4*)(B_ptr + v0 * N);
        uint4 B_loaded = make_uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr + v0 * N, B_pred_guard);
        *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = B_loaded;

      }
      else
      {
        *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 128)) = make_uint4(0, 0, 0, 0);
      }
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 16) + ((i / 4) * 8)]
                          * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];
        }

      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    int local_row = ((i / 4) * 8);
    int local_col = ((i % 4) * 4);
    if constexpr (K_ld_check || N_ld_check)
    {
      if ( ((cur_C_ic_start + local_row) < K_original) && ((cur_C_oc_start + local_col) < N) )
        C_ptr[local_row * N + local_col] = C_local[i];

    }
    else
    {
      C_ptr[local_row * N + local_col] = C_local[i];
    }
  }
}

// conv_backward_cuda_m32n64k64_f32f32f32
template <typename IntT>
__global__ void __launch_bounds__(64) conv_backward_cuda_setting2_mode0_f32f32f32(int M_fwd, int K_original, int N, int kernel_volume, int split_k_iters, float *__restrict__ A, float *__restrict__ B, int *__restrict__ out_in_map, float *__restrict__ C)
{

  int j_factors1 = (N + 63) / 64;
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((K_original * kernel_volume + 31) / 32 * j_factors1);
  int blockIdx_z = blockIdx.x / ((K_original * kernel_volume + 31) / 32 * j_factors1);

  float C_local[32];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[4096];

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
    C_local[i] = 0.0;
  }

  int blockIdx_m = blockIdx_y / j_factors1;
  int blockIdx_n = blockIdx_y % j_factors1;
  int threadIdx_x = (int)threadIdx.x;

  int kernel_offset = blockIdx_m / (K_original / 32);
  int channel_offset = (blockIdx_m * 32 + ((threadIdx_x * 4) % 32)) % K_original;
  int K_loops = ((M_fwd + 63 ) / 64 + split_k_iters - 1) / split_k_iters;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                          + (threadIdx_x / (32/4)) * kernel_volume
                          + kernel_offset;
  float * A_ptr = A + channel_offset;

  // reorder is performed on B's rows.
  float * B_ptr = B
                    + (blockIdx_n * 64) + ((threadIdx_x * 4) % 64);
  int reorder_offset = threadIdx_x /(64/4);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);

  float * A_shared_reduce_ptr =  A_shared + (threadIdx_x / 16);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 16);

  // splitK offset
  float * cur_C = C + blockIdx_z * K_original * kernel_volume * N;
  int C_m_offset = blockIdx_m * 32 + (threadIdx_x / 16);  // C_m_offset
  int C_n_offset = blockIdx_n * 64  + (threadIdx_x % 16);
  // float * C_ptr = cur_C + C_m_offset * N + C_n_offset;

  #pragma unroll
  for (int _k_0 = 0; _k_0 < K_loops - 1; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = out_in_map_ptr_local[(ax0_ax1_fused_0 *8) * kernel_volume];
      if (input_idx != -1)
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =  // ax0_ax1_fused_0 * elements loaded in each loop
            *(float4*)(A_ptr + (input_idx * K_original));
      }
      else
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 4);
      int v0 = reorder_offset_inner;
      *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
          *(float4*)(B_ptr + v0 * N);
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 32) + ((i / 4) * 4)]
                          * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];
        }

      }
    }
  }
  for (int _k_0 = K_loops - 1; _k_0 < K_loops; ++_k_0)
  {
    int k_0 = blockIdx_z + split_k_iters * _k_0; // splitK offset
    if (k_0 >= (M_fwd + 63) / 64)
      break;

    int * out_in_map_ptr_local = out_in_map_ptr + k_0 * 64 * kernel_volume;
    int reorder_offset_local = reorder_offset + k_0 * 64;

    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {
      int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *8) * kernel_volume);
      if (input_idx != -1)
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =  // ax0_ax1_fused_0 * elements loaded in each loop
            *(float4*)(A_ptr + (input_idx * K_original));
      }
      else
      {
        *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    #pragma unroll
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1)
    {
      int reorder_offset_inner = reorder_offset_local + (ax0_ax1_fused_0_1 * 4);
      if (reorder_offset_inner < M_fwd)
      {
        int v0 = reorder_offset_inner;
        *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =
            *(float4*)(B_ptr + v0 * N);
      }
      else
      {
        *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);
      }
    }

    __syncthreads();
    #pragma unroll
    for (int k_1 = 0; k_1 < ( 64 / 4); ++k_1)
    {
      #pragma unroll
      for (int k_2 = 0; k_2 < 4; ++k_2)
      {
        int vk_in_block = (k_1 << 2) + k_2;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
          C_local[i] = C_local[i] +
                          A_shared_reduce_ptr[(vk_in_block * 32) + ((i / 4) * 4)]
                          * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];
        }

      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
      int C_m_offset_cur = C_m_offset + ((i / 4) * 4);
      int C_n_offset_cur = C_n_offset + ((i % 4) * 16);
      cur_C[C_m_offset_cur * N + C_n_offset_cur] = C_local[i];
  }
}


template <typename IntT>
void conv_backward_wgrad_implicit_gemm_cuda(
    const phi::GPUContext& dev_ctx,
    const phi::DenseTensor& _in_feats, const phi::DenseTensor& _kernel,
    const phi::DenseTensor& _out_in_map, const int split_k_iters,
    phi::DenseTensor& _out_feats)
{
  auto compute_capability = dev_ctx.GetComputeCapability();
  bool allow_fp16 = compute_capability >= 75;
  bool is_half = _in_feats.dtype() == phi::DataType::FLOAT16;

  int num_in_feats = _in_feats.dims()[0];
  int num_in_channels = _in_feats.dims()[1];
  int kernel_volume = _out_in_map.dims()[1];

  auto out_in_map = const_cast<int*>(_out_in_map.data<int>());

  int num_out_feats = _out_feats.dims()[1];
  int num_out_channels = _out_feats.dims()[2];

  if (is_half)
  {
    if (!allow_fp16)
    {
      throw std::runtime_error("FP16 kernels are not supported for implicit GEMM now for SM75-.");
    }
    auto in_feats = reinterpret_cast<half *>(const_cast<phi::dtype::float16 *>(_in_feats.data<phi::dtype::float16>()));
    auto kernel = reinterpret_cast<half *>(const_cast<phi::dtype::float16 *>(_kernel.data<phi::dtype::float16>()));
    auto out_feats = reinterpret_cast<half *>(_out_feats.data<phi::dtype::float16>());

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int j_factors1 = num_out_channels / 64 / 1;
      dim3 num_blocks( num_in_channels * kernel_volume / 32 * j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      conv_backward_cuda_setting2_mode0_f16f16f32<IntT><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
          _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
    }
    else
    {
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dim3 num_blocks((num_in_channels + 15) / 16 * kernel_volume * j_factors1 * split_k_iters);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 1);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 16, false, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 16, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 8, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 4, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 2, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 8 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<16, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<8, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<8, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<8, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<8, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<8, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<4, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<4, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<4, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<4, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<4, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<2, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<2, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<2, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<2, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f16f16f32<2, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
    }
  }
  else // fp32fp32fp32
  {
    // printf("\nRun FP32 wgrad backward kernels!\n");
    auto in_feats = const_cast<float *>(_in_feats.data<float>());
    auto kernel = const_cast<float *>(_kernel.data<float>());
    auto out_feats = _out_feats.data<float>();

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int block_num_M = (num_in_channels * kernel_volume) / 32;
      int block_num_N = (num_out_channels) / 64; //j_factors1

      dim3 num_blocks(block_num_M * block_num_N * split_k_iters);
      dim3 threads_per_block(64);
      conv_backward_cuda_setting2_mode0_f32f32f32<IntT><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
          _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
    }
    else
    {
      int block_num_M = (num_in_channels + 15) / 16 * kernel_volume;
      int block_num_N = (num_out_channels - 1) / 16 + 1;

      dim3 num_blocks(block_num_M * block_num_N * split_k_iters);
      dim3 threads_per_block(32);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 16, false, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 16, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 8, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 4, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<16, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<8, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<8, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<8, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<8, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<4, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<4, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<4, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_backward_cuda_setting1_mode0_f32f32f32<4, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _kernel.dims()[0], num_in_channels, num_out_channels, kernel_volume, split_k_iters, in_feats, kernel, out_in_map, out_feats);
        }
      }
    }
  }
//   return _out_feats.sum(0);
}

#endif //PADDLE_WITH_CUDA
