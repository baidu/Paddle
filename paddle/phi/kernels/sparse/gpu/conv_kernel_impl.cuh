#pragma once

#ifdef PADDLE_WITH_CUDA

#include <cuda_fp16.h>
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/sparse/gpu/conv_memory_utils.cuh"
#include "paddle/phi/kernels/sparse/gpu/conv_kernel_impl_utils.cuh"

// conv_forward_cuda_m128n16k16_m64n16k16_m16n16k16_f16f16f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(64) conv_forward_cuda_setting1_mode0_f16f16f32(int M, int K_original, int N, int kernel_volume, half *__restrict__ A, half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C)
{
  // warning: kernel could not work with K_original < 32!
  const int K_tile = 16; // min(16, K_original);
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);
  int K_implicit = K_tile_padded * kernel_volume;

  float C_warp[32];
  __shared__ half A_shared[5120];
  __shared__ half B_shared[640];
  half A_shared_warp[32];
  half B_shared_warp[8];
  for (int i0_0_3_init = 0; i0_0_3_init < 4; ++i0_0_3_init)
  {
    for (int i = 0; i < 8; ++i)
    {
      C_warp[(i0_0_3_init * 8) + i] = 0.0;
    };
  }

  int j_factors1 = (N + 15) / 16 / 1;
  int *out_in_map_ptr = out_in_map + (blockIdx.x / j_factors1 * 128 + threadIdx.y * 16 + threadIdx.x / 2) * kernel_volume + ((threadIdx.y * 256) % 16) / K_tile_padded + ((threadIdx.x * 8) % 16) / K_tile_padded;
  half *A_ptr = A + ((threadIdx.y * 256 % 16) % K_tile_padded) + ((threadIdx.x * 8 % 16) % K_tile_padded);
  half *B_ptr = B + (blockIdx.x % j_factors1) * 16 + threadIdx.y * 256 / 16 * N + threadIdx.x * 8 / 16 * N + (threadIdx.x * 8) % 16;
  int reorder_loc_offset = blockIdx.x / j_factors1 * 8 * 16 + (threadIdx.y % 2) * 4 * 16 + (threadIdx.x / 4);
  half *C_ptr = C
                + (blockIdx.x % j_factors1) * 16 + threadIdx.y / 2 * 16 + (threadIdx.x % 4) * 2;

  int A_ld_start, A_ld_amount, A_ld_bound, A_pred_guard;
  int B_ld_start, B_ld_amount, B_ld_bound, B_pred_guard, B_ld_amount_N, B_ld_K_bound;
  bool B_ld_K;
  if constexpr (N_ld_check || K_ld_check)
  {
    B_ld_start = (blockIdx.x % j_factors1) * 16 + (threadIdx.x * 8) % 16;
    B_ld_amount_N = max(0, min(B_ld_start + 8, N) - B_ld_start);
    B_ld_K_bound = K_original;
  }
  else
    B_pred_guard = 1;

  //+ (threadIdx.x / 4) * N;
  for (int i2_0_0 = 0; i2_0_0 < K_implicit / K_tile; ++i2_0_0)

  {

    if constexpr (K_ld_check)
    {
      A_ld_start = (i2_0_0 * K_tile % K_tile_padded) + ((threadIdx.x * 8) % 16);
      A_ld_amount = max(0, min(A_ld_start + 8, K_original) - A_ld_start);
      A_ld_bound = A_ld_amount / (K_ld_factor / 2);
      A_pred_guard = 0;
      for (int i = 0; i < A_ld_bound; i++)
        A_pred_guard |= (1 << i);
    }
    else
    {
      A_pred_guard = 1;
    }

    if constexpr (K_ld_check || N_ld_check)
    {
      B_ld_K = ((i2_0_0 * K_tile % K_tile_padded) + threadIdx.x * 8 / 16) < B_ld_K_bound;
      B_ld_amount = B_ld_amount_N * (int)B_ld_K;
      B_ld_bound = B_ld_amount / (N_ld_factor / 2);
      B_pred_guard = 0;
      for (int i = 0; i < B_ld_bound; i++)
        B_pred_guard |= (1 << i);
    }

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * K_tile / K_tile_padded;
    half *A_ptr_local = A_ptr + (i2_0_0 * K_tile % K_tile_padded);
    half *B_ptr_local;
    if constexpr (K_ld_check)
      B_ptr_local = B_ptr + (i2_0_0 * K_tile / K_tile_padded * K_original + i2_0_0 * K_tile % K_tile_padded) * N;
    else
      B_ptr_local = B_ptr + i2_0_0 * K_tile * N;
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 32 * kernel_volume + (ax0_ax1_fused_0 * 512 % 16) / K_tile_padded];

      if (input_idx != -1)
      {
        uint4 A_loaded = make_uint4(0, 0, 0, 0);
        global_load<K_ld_factor>(A_loaded, A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 512 % 16) % K_tile_padded), A_pred_guard);
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 1280) + (((int)threadIdx.y) * 640)) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = A_loaded;
      }
      else
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 1280) + (((int)threadIdx.y) * 640)) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }

    if (threadIdx.y == 0)
    {
      uint4 B_loaded = make_uint4(0, 0, 0, 0);
      global_load<N_ld_factor>(B_loaded, B_ptr_local, B_pred_guard);
      *(uint4 *)(B_shared + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) = B_loaded;
    }

    __syncthreads();
    __syncthreads();
    for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
    {

      {
        unsigned int addr;
        __asm__ __volatile__(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
            : "=r"(addr)
            : "l"((void *)((&(A_shared[((((int)threadIdx.y) * 2560) + (ax0_0 * 640))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
        __asm__ __volatile__(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
            "{%0, %1, %2, %3}, [%4];"
            : "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[3])
            : "r"(addr));
#endif
      }
    }

    {
      unsigned int addr;
      __asm__ __volatile__(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
          : "=r"(addr)
          : "l"((void *)((&(B_shared[0])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
      __asm__ __volatile__(
          "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
          "{%0, %1, %2, %3}, [%4];"
          : "=r"(((unsigned *)(B_shared_warp + 0))[0]), "=r"(((unsigned *)(B_shared_warp + 0))[1]), "=r"(((unsigned *)(B_shared_warp + 0))[2]), "=r"(((unsigned *)(B_shared_warp + 0))[3])
          : "r"(addr));
#endif
    }
    for (int i0_0_3 = 0; i0_0_3 < 4; ++i0_0_3)
    {
#if __CUDA_ARCH__ >= 800
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3]));
      }
#elif __CUDA_ARCH__ >= 750
      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[3]));
      }

      {
        __asm__ __volatile__(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3]));
      }
#endif
    }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1)
  {

    int reorder_loc_offset_local = reorder_loc_offset + ax0_0_1 * 16;
    for (int local_id = 0; local_id < 8; ++local_id)
    {

      int reorder_location_cur = reorder_loc_offset_local + (((local_id / 2) % 2) * 8);
      if constexpr (N_ld_check)
      {
        bool C_wb_enable = ((blockIdx.x % j_factors1) * 16 + threadIdx.y / 2 * 16 + (threadIdx.x % 4) * 2 + (local_id % 2) + (local_id / 4) * 8) < N;
        if (C_wb_enable && reorder_location_cur < M)
          C_ptr[reorder_location_cur * N
                + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[(ax0_0_1 * 8) + local_id]);
      }
      else
      {
        if (reorder_location_cur < M)
          C_ptr[reorder_location_cur * N
                + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[(ax0_0_1 * 8) + local_id]);
      }
    };
  }
}

// conv_forward_cuda_m128n16k32_m64n16k32_m16n16k16_f16f16f32
template <typename IntT>
__global__ void __launch_bounds__(64) conv_forward_cuda_setting2_mode0_f16f16f32(int M, int K_original, int N, int kernel_volume, half *__restrict__ A, half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C)
{
  // warning: kernel could not work with K_original < 32!
  int K_implicit = K_original * kernel_volume;
  float C_warp[32];
  __shared__ half A_shared[5120];
  __shared__ half B_shared[1280];
  half A_shared_warp[32];
  half B_shared_warp[8];
  for (int i0_0_3_init = 0; i0_0_3_init < 4; ++i0_0_3_init)
  {
    for (int i = 0; i < 8; ++i)
    {
      C_warp[(i0_0_3_init * 8) + i] = 0.0;
    };
  }

  // hoisting shared pointer offsets
  int j_factors1 = N / 16 / 1;
  int *out_in_map_ptr = out_in_map + (blockIdx.x / j_factors1 * 128 + threadIdx.y * 8 + threadIdx.x / 4) * kernel_volume + ((threadIdx.y * 256) % 32) / K_original + ((threadIdx.x * 8) % 32) / K_original;
  half *A_ptr = A + ((threadIdx.y * 256 % 32) % K_original) + ((threadIdx.x * 8 % 32) % K_original);
  half *B_ptr = B + (blockIdx.x % j_factors1) * 16 + threadIdx.y * 256 / 16 * N + threadIdx.x * 8 / 16 * N + (threadIdx.x * 8) % 16;
  int reorder_loc_offset = blockIdx.x / j_factors1 * 8 * 16 + (threadIdx.y % 2) * 4 * 16 + (threadIdx.x / 4);
  half *C_ptr = C
                + (blockIdx.x % j_factors1) * 16 + threadIdx.y / 2 * 16 + (threadIdx.x % 4) * 2;
  for (int i2_0_0 = 0; i2_0_0 < K_implicit / 32; ++i2_0_0)

  {

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 32 / K_original;
    half *A_ptr_local = A_ptr + (i2_0_0 * 32 % K_original);
    half *B_ptr_local = B_ptr + i2_0_0 * 32 * N;
    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
    {

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

    *(uint4 *)(B_shared + (((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8))) =
        *(uint4 *)(B_ptr_local);

    __syncthreads();
    for (int i2_0_1 = 0; i2_0_1 < 2; ++i2_0_1)
    {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((((((int)threadIdx.y) & 1) * 2560) + (ax0_0 * 640)) + (i2_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[3])
              : "r"(addr));
#endif
        }
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
#endif
      }
      for (int i0_0_3 = 0; i0_0_3 < 4; ++i0_0_3)
      {

#if __CUDA_ARCH__ >= 800
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
              : "=f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "r"(((unsigned *)(B_shared_warp + 0))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[3]));
        }

        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
              : "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "r"(((unsigned *)(B_shared_warp + 4))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3]));
        }
#elif __CUDA_ARCH__ >= 750
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              : "=f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + 0))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[3]));
        }
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              : "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + 4))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3]));
        }
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              : "=f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "=f"(((float *)(C_warp + (i0_0_3 * 8)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + 2))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[0]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[1]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[2]), "f"(((float *)(C_warp + (i0_0_3 * 8)))[3]));
        }
        {
          __asm__ __volatile__(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              : "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3])
              : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + 6))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 8) + 4)))[3]));
        }
#endif
      }
  }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1)
  {

    int reorder_loc_offset_local = reorder_loc_offset + ax0_0_1 * 16;
    for (int local_id = 0; local_id < 8; ++local_id)
    {

      int reorder_location_cur = reorder_loc_offset_local + (((local_id / 2) % 2) * 8);
      if (reorder_location_cur < M)
        C_ptr[reorder_location_cur * N
              + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[(ax0_0_1 * 8) + local_id]);
    };
  }
}

// conv_forward_cuda_m128n64k32_m64n32k32_m16n16k16_f16f16f32
template <typename IntT>
__global__ void __launch_bounds__(128) conv_forward_cuda_setting3_mode0_f16f16f32(int M, int K_original, int N, int kernel_volume, half *__restrict__ A, half *__restrict__ B, int *__restrict__ out_in_map, half *__restrict__ C)
{
  int K_implicit = K_original * kernel_volume;
  float C_warp[64];
  __shared__ half A_shared[5120];
  __shared__ half B_shared[2304];
  half A_shared_warp[32];
  half B_shared_warp[16];
  for (int i0_0_3_init = 0; i0_0_3_init < 4; ++i0_0_3_init)
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
  int j_factors1 = N / 16 / 4;
  int *out_in_map_ptr = out_in_map + (blockIdx.x / j_factors1 * 128 + threadIdx.y * 8 + threadIdx.x / 4) * kernel_volume + ((threadIdx.y * 256) % 32) / K_original + ((threadIdx.x * 8) % 32) / K_original;
  half *A_ptr = A + ((threadIdx.y * 256 % 32) % K_original) + ((threadIdx.x * 8 % 32) % K_original);
  half *B_ptr = B + (blockIdx.x % j_factors1) * 64 + threadIdx.y * 256 / 64 * N + threadIdx.x * 8 / 64 * N + (threadIdx.x * 8) % 64;
  int reorder_loc_offset = blockIdx.x / j_factors1 * 8 * 16 + (threadIdx.y % 2) * 4 * 16 + (threadIdx.x / 4);
  half *C_ptr = C
                + (blockIdx.x % j_factors1) * 64 + threadIdx.y / 2 * 32 + (threadIdx.x % 4) * 2;

  int B_kernel_offset = threadIdx.y * 256 / 64 + threadIdx.x * 8 / 64;

  for (int i2_0_0 = 0; i2_0_0 < K_implicit / 32; ++i2_0_0)

  {

    int *out_in_map_ptr_local = out_in_map_ptr + i2_0_0 * 32 / K_original;
    half *A_ptr_local = A_ptr + (i2_0_0 * 32 % K_original);
    half *B_ptr_local = B_ptr + i2_0_0 * 32 * N;

    __syncthreads();
    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 4; ++ax0_ax1_fused_0)
    {

      int input_idx = out_in_map_ptr_local[ax0_ax1_fused_0 * 32 * kernel_volume + (ax0_ax1_fused_0 * 1024 % 32) / K_original];

      if (input_idx != -1)
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) =
            *(uint4 *)(A_ptr_local + input_idx * K_original + ((ax0_ax1_fused_0 * 1024 % 32) % K_original));
      }
      else
      {
        *(uint4 *)(A_shared + ((((ax0_ax1_fused_0 * 1280) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = make_uint4(__pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)), __pack_half2(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f)));
      }
    }
    for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 2; ++ax0_ax1_fused_0_1)
    {
      // Shang: skip loading B
      int B_kernel_offset_local = (B_kernel_offset + i2_0_0 * 32 + ax0_ax1_fused_0_1 * 1024 / 64) / K_original;
      *(uint4 *)(B_shared + ((((ax0_ax1_fused_0_1 * 1152) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8))) =
          *(uint4 *)(B_ptr_local + ax0_ax1_fused_0_1 * 1024 * N / 64);
    }
    __syncthreads();

    for (int i2_0_1 = 0; i2_0_1 < 2; ++i2_0_1)
    {
      for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
      {

        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(A_shared[((((((int)threadIdx.y) & 1) * 2560) + (ax0_0 * 640)) + (i2_0_1 * 16))])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0_0 * 8)))[3])
              : "r"(addr));
#endif
        }
      }
      for (int ax1_0 = 0; ax1_0 < 2; ++ax1_0)
      {
        {
          unsigned int addr;
          __asm__ __volatile__(
              "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }"
              : "=r"(addr)
              : "l"((void *)((&(B_shared[(((i2_0_1 * 1152) + ((((int)threadIdx.y) >> 1) * 32)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 72) + ((((int)threadIdx.x) >> 4) * 8)))));
#if __CUDA_ARCH__ >= 750
          __asm__ __volatile__(
              "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
              "{%0, %1, %2, %3}, [%4];"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr));
#endif
        }
      }
      for (int i0_0_3 = 0; i0_0_3 < 4; ++i0_0_3)
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
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + (i1_0_4 * 8)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (i0_0_3 * 8)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "=f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 2)))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[0]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[1]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[2]), "f"(((float *)(C_warp + ((i0_0_3 * 16) + (i1_0_4 * 8))))[3]));
          }
          {
            __asm__ __volatile__(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                : "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3])
                : "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(A_shared_warp + ((i0_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(B_shared_warp + ((i1_0_4 * 8) + 6)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((i0_0_3 * 16) + (i1_0_4 * 8)) + 4)))[3]));
          }
#endif
          }
        }
    }
  }
  for (int ax0_0_1 = 0; ax0_0_1 < 4; ++ax0_0_1)
  {

    int reorder_loc_offset_local = reorder_loc_offset + ax0_0_1 * 16;
    for (int ax1_0_1 = 0; ax1_0_1 < 2; ++ax1_0_1)
    {
      for (int local_id = 0; local_id < 8; ++local_id)
      {

        int reorder_location_cur = reorder_loc_offset_local + (((local_id / 2) % 2) * 8);
        if (reorder_location_cur < M)
          C_ptr[reorder_location_cur * N
                //+ ax0_0_1 * N / 16 * 256
                + ax1_0_1 * 16
                //+ (((local_id / 2) % 2) * 8) * N
                + (local_id % 2) + (local_id / 4) * 8] = __float2half(C_warp[((ax0_0_1 * 16) + (ax1_0_1 * 8)) + local_id]);
      };
    }
  }
}

// conv_forward_cuda_m128n16k16_f32f32f32
template <int K_ld_factor, int N_ld_factor, bool K_ld_check, bool N_ld_check>
__global__ void __launch_bounds__(64) conv_forward_cuda_setting1_mode0_f32f32f32(int M, int K_original, int N, int kernel_volume, float* __restrict__ A, float* __restrict__ B, int* __restrict__ out_in_map, float* __restrict__ C)
{

  const int K_tile = 16;
  int K_tile_padded = K_tile * ((K_original + K_tile - 1) / K_tile);
  int K_implicit = K_tile_padded * kernel_volume;

  float C_local[32];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[256];

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
    C_local[i] = 0.0;
  }

  int K_loops = K_implicit / 16;
  int block_num_n = (N - 1) / 16 + 1;
  int blockIdx_m = (int)blockIdx.x / block_num_n;
  int blockIdx_n = (int)blockIdx.x % block_num_n;
  int threadIdx_x = (int)threadIdx.x;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                         + (blockIdx_m * 128 + (threadIdx_x / (16/4)))* kernel_volume;

  float * B_ptr = B
                  + (threadIdx_x / (16/4)) * N
                  + (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * A_shared_reduce_ptr =  A_shared + ((threadIdx_x / 4) * 16);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 4);

  int location_offset = blockIdx_m * 128 + (threadIdx_x / 4);  // C_m_offset
  int C_n_offset = blockIdx_n * 16  + (threadIdx_x % 4);

  int channel_offset_A = ((threadIdx_x * 4) % 16);

  int A_ld_start, A_ld_amount, A_ld_bound, A_pred_guard;
  int B_ld_start, B_ld_amount, B_ld_bound, B_pred_guard, B_ld_amount_N, B_ld_K_bound;
  bool B_ld_K;
  if constexpr (N_ld_check || K_ld_check)
  {
    B_ld_start = (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);
    B_ld_amount_N = max(0, min(B_ld_start + 4, N) - B_ld_start);
    B_ld_K_bound = K_original;
  }
  else
    B_pred_guard = 1;

  #pragma unroll
  for (int k_0 = 0; k_0 < K_loops; ++k_0) {

    {
      if constexpr (K_ld_check)
      {
        A_ld_start = (k_0 * K_tile % K_tile_padded) + ((threadIdx.x * 4) % 16); // Channel_offset
        A_ld_amount = max(0, min(A_ld_start + 4, K_original) - A_ld_start);
        A_ld_bound = A_ld_amount / (K_ld_factor / 4);
        A_pred_guard = 0;
        for (int i = 0; i < A_ld_bound; i++)
          A_pred_guard |= (1 << i);
      }
      else
      {
        A_pred_guard = 1;
      }

      if constexpr (K_ld_check || N_ld_check)
      {
        B_ld_K = ((k_0 * K_tile % K_tile_padded) + threadIdx.x * 4 / 16) < B_ld_K_bound;
        B_ld_amount = B_ld_amount_N * (int)B_ld_K;
        B_ld_bound = B_ld_amount / (N_ld_factor / 4);
        B_pred_guard = 0;
        for (int i = 0; i < B_ld_bound; i++)
          B_pred_guard |= (1 << i);
      }

      int* out_in_map_ptr_local = out_in_map_ptr + k_0 * 16 / K_tile_padded;
      float* A_ptr_local = A  + (k_0 * 16 % K_tile_padded) + channel_offset_A;

      float* B_ptr_local;
      if constexpr (K_ld_check)
        B_ptr_local = B_ptr + (k_0 * K_tile / K_tile_padded * K_original + k_0 * K_tile % K_tile_padded) * N;
      else
        B_ptr_local = B_ptr + k_0 * K_tile * N;

      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
      {

        int input_idx = *(out_in_map_ptr_local + (ax0_ax1_fused_0 *16) * kernel_volume);
        if (input_idx != -1)
        {
          uint4 A_loaded = make_uint4(0, 0, 0, 0);
          global_load<K_ld_factor>(A_loaded, A_ptr_local + (input_idx * K_original) , A_pred_guard);
          *(uint4 *)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = A_loaded;
        }
        else
        {
          *(uint4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = make_uint4(0, 0, 0, 0);
        }
      }

      #pragma unroll
      for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 1; ++ax0_ax1_fused_0_1)
      {
        uint4 B_loaded = make_uint4(0, 0, 0, 0);
        global_load<N_ld_factor>(B_loaded, B_ptr_local + (ax0_ax1_fused_0_1 * 16) * N, B_pred_guard);
        *(uint4 *)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) = B_loaded;
      }

      __syncthreads();
      #pragma unroll
      for (int k_1 = 0; k_1 < ( 16 / 4); ++k_1)
      {
        #pragma unroll
        for (int k_2 = 0; k_2 < 4; ++k_2)
        {
          int vk_in_block = (k_1 << 2) + k_2;
          #pragma unroll
          for (int i = 0; i < 32; ++i)
          {
            C_local[i] = C_local[i] +
                            A_shared_reduce_ptr[((i / 4) * 16) * 16 + vk_in_block]
                            * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];

          }
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
      int location_cur = location_offset + ((i / 4) * 16);
      int vn = C_n_offset + ((i % 4) * 4);

      if constexpr (N_ld_check)
      {
        if (vn < N && location_cur < M)
          C[location_cur * N + vn] = C_local[i];
      }
      else
      {
        if (location_cur < M)
          C[location_cur * N + vn] = C_local[i];
      }
  }
}

// conv_forward_cuda_m128n16k32_f32f32f32
template <typename IntT>
__global__ void __launch_bounds__(64) conv_forward_cuda_setting2_mode0_f32f32f32(int M, int K_original, int N, int kernel_volume, float* __restrict__ A, float* __restrict__ B, int* __restrict__ out_in_map, float* __restrict__ C)
{
  float C_local[32];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[512];

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
    C_local[i] = 0.0;
  }

  int K_loops = (K_original * kernel_volume - 1) / 32 + 1;
  int block_num_n = (N - 1) / 16 + 1;
  int blockIdx_m = (int)blockIdx.x / block_num_n;
  int blockIdx_n = (int)blockIdx.x % block_num_n;
  int threadIdx_x = (int)threadIdx.x;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                         + (blockIdx_m * 128 + (threadIdx_x / (32/4)))* kernel_volume;

  float * B_ptr = B
                  + (threadIdx_x / (16/4)) * N
                  + (blockIdx_n * 16) + ((threadIdx_x * 4) % 16);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * A_shared_reduce_ptr =  A_shared + ((threadIdx_x / 4) * 32);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 4);

  int location_offset = blockIdx_m * 128 + (threadIdx_x / 4);  // C_m_offset
  int C_n_offset = blockIdx_n * 16  + (threadIdx_x % 4);

  int channel_offset_A = ((threadIdx_x * 4) % 32); // mod K_tile=32

  #pragma unroll
  for (int k_0 = 0; k_0 < K_loops; ++k_0) {

    int channel_offset = k_0 % (K_original / 32) * 32 + channel_offset_A;
    int kernel_offset = k_0 / (K_original / 32);
    int *out_in_map_ptr_k = out_in_map_ptr + kernel_offset;

    {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 16; ++ax0_ax1_fused_0)
      {

        int input_idx = *(out_in_map_ptr_k + (ax0_ax1_fused_0 *8) * kernel_volume);
        if (input_idx != -1)
        {

          *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) =  // ax0_ax1_fused_0 * elements loaded in each loop
              *(float4*)(A + (input_idx * K_original) + channel_offset);

        }
        else {

          *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 256)) = make_float4(0.0, 0.0, 0.0, 0.0);

        }
      }

      #pragma unroll
      for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 2; ++ax0_ax1_fused_0_1)
      {

        *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 256)) =                 // ax0_ax1_fused_0_1 * elements loaded in each loop
              *(float4*)(B_ptr + ((k_0 * 32) + (ax0_ax1_fused_0_1 * 16)) * N);

      }

      __syncthreads();
      #pragma unroll
      for (int k_1 = 0; k_1 < ( 32 / 4); ++k_1)
      {
        #pragma unroll
        for (int k_2 = 0; k_2 < 4; ++k_2)
        {
          int vk_in_block = (k_1 << 2) + k_2;
          #pragma unroll
          for (int i = 0; i < 32; ++i)
          {
            C_local[i] = C_local[i] +
                            A_shared_reduce_ptr[((i / 4) * 16) * 32 + vk_in_block]
                            * B_shared_reduce_ptr[(vk_in_block * 16) + ((i % 4) * 4)];

          }
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 32; ++i)
  {
      int location_cur = location_offset + ((i / 4) * 16);
      int vn = C_n_offset + ((i % 4) * 4);
      if (location_cur < M)
        C[location_cur * N + vn] = C_local[i];
   }
}

// conv_forward_cuda_m128n64k32_f32f32f32
template <typename IntT>
__global__ void __launch_bounds__(128) conv_forward_cuda_setting3_mode0_f32f32f32(int M, int K_original, int N, int kernel_volume, float* __restrict__ A, float* __restrict__ B, int* __restrict__ out_in_map, float* __restrict__ C)
{
  float C_local[64];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[2048];

  #pragma unroll
  for (int i = 0; i < 64; ++i)
  {
    C_local[i] = 0.0;
  }

  int K_loops = (K_original * kernel_volume - 1) / 32 + 1;
  int block_num_n = (N - 1) / 64 + 1;
  int blockIdx_m = (int)blockIdx.x / block_num_n;
  int blockIdx_n = (int)blockIdx.x % block_num_n;
  int threadIdx_x = (int)threadIdx.x;

  // hoisting shared pointer offsets
  int * out_in_map_ptr = out_in_map
                         + (blockIdx_m * 128 + (threadIdx_x / (32/4)))* kernel_volume;

  float * B_ptr = B
                  + (threadIdx_x / (64/4)) * N
                  + (blockIdx_n * 64) + ((threadIdx_x * 4) % 64);

  float * A_shared_ptr = A_shared + (threadIdx_x * 4);
  float * A_shared_reduce_ptr =  A_shared + ((threadIdx_x / 16) * 32);
  float * B_shared_ptr = B_shared + (threadIdx_x * 4);
  float * B_shared_reduce_ptr = B_shared + (threadIdx_x % 16);

  int location_offset = blockIdx_m * 128 + (threadIdx_x / 16);  // C_m_offset
  int C_n_offset = blockIdx_n * 64  + (threadIdx_x % 16);

  int channel_offset_A = ((threadIdx_x * 4) % 32); // mod K_tile=32

  #pragma unroll
  for (int k_0 = 0; k_0 < K_loops; ++k_0) {

    int channel_offset = k_0 % (K_original / 32) * 32 + channel_offset_A;
    int kernel_offset = k_0 / (K_original / 32);
    int *out_in_map_ptr_k = out_in_map_ptr + kernel_offset;

    {
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0)
      {

        int input_idx = *(out_in_map_ptr_k + (ax0_ax1_fused_0 *16) * kernel_volume);
        if (input_idx != -1)
        {

          *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 512)) =  // ax0_ax1_fused_0 * elements loaded in each loop
              *(float4*)(A + (input_idx * K_original) + channel_offset);

        }
        else {

          *(float4*)(A_shared_ptr + (ax0_ax1_fused_0 * 512)) = make_float4(0.0, 0.0, 0.0, 0.0);

        }
      }

      #pragma unroll
      for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 4; ++ax0_ax1_fused_0_1)
      {

        *(float4*)(B_shared_ptr + (ax0_ax1_fused_0_1 * 512)) =                 // ax0_ax1_fused_0_1 * elements loaded in each loop
              *(float4*)(B_ptr + ((k_0 * 32) + (ax0_ax1_fused_0_1 * 8)) * N);

      }

      __syncthreads();
      #pragma unroll
      for (int k_1 = 0; k_1 < ( 32 / 4); ++k_1)
      {
        #pragma unroll
        for (int k_2 = 0; k_2 < 4; ++k_2)
        {
          int vk_in_block = (k_1 << 2) + k_2;
          #pragma unroll
          for (int i = 0; i < 64; ++i)
          {
            C_local[i] = C_local[i] +
                            A_shared_reduce_ptr[((i / 4) * 8) * 32 + vk_in_block]
                            * B_shared_reduce_ptr[(vk_in_block * 64) + ((i % 4) * 16)];

          }
        }
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 64; ++i)
  {
      int location_cur = location_offset + ((i / 4) * 8);
      int vn = C_n_offset + ((i % 4) * 16);
      if (location_cur < M)
        C[location_cur * N + vn] = C_local[i];
   }
}

template <typename IntT>
void conv_forward_implicit_gemm_cuda(
    const phi::GPUContext& dev_ctx,
    const phi::DenseTensor& _in_feats,
    const phi::DenseTensor& _kernel,
    const phi::DenseTensor& _out_in_map,
    int num_out_feats, int num_out_channels,
    phi::DenseTensor& _out_feats)
{
  auto compute_capability = dev_ctx.GetComputeCapability();
  bool allow_fp16 = compute_capability >= 75;
  bool is_half = _in_feats.dtype() == phi::DataType::FLOAT16;

  int num_in_feats = _in_feats.dims()[0];
  int num_in_channels = _in_feats.dims()[1];

  int kernel_volume = _out_in_map.dims()[1];
  auto out_in_map = const_cast<int*>(_out_in_map.data<int>());

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
      int j_factors1 = num_out_channels / 16 / 4;
      dim3 num_blocks((num_out_feats + 127) / 128 * j_factors1);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 4);
      conv_forward_cuda_setting3_mode0_f16f16f32<IntT><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
          _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else if (num_in_channels % 32 == 0 && num_out_channels % 16 == 0)
    {
      int j_factors1 = num_out_channels / 16 / 1;
      dim3 num_blocks((num_out_feats + 127) / 128 * j_factors1);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      conv_forward_cuda_setting2_mode0_f16f16f32<IntT><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
          _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else
    {
      // throw std::invalid_argument("IC is too small for this kernel");
      int j_factors1 = (num_out_channels + 15) / 16 / 1;
      dim3 num_blocks((num_out_feats + 127) / 128 * j_factors1);
      // threadIdx.x: 32
      // threadIdx.y: i_factors[2] * j_factors[2]
      dim3 threads_per_block(32, 2);
      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, false, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 8, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 4, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 2, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 8 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<16, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<8, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<8, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<8, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<8, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<8, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<4, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<4, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<4, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<4, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<4, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<2, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 8 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<2, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<2, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<2, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f16f16f32<2, 2, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
    }
  }
  else  // fp32fp32fp32
  {
    auto in_feats = const_cast<float *>(_in_feats.data<float>());
    auto kernel = const_cast<float *>(_kernel.data<float>());
    auto out_feats = _out_feats.data<float>();

    if (num_out_channels % 64 == 0 && num_in_channels % 32 == 0)
    {
      int block_num_M = (num_out_feats + 127) / 128;
      int block_num_N = num_out_channels / 64;  //j_factors1
      dim3 num_blocks(block_num_M * block_num_N);
      dim3 threads_per_block(128);
      conv_forward_cuda_setting3_mode0_f32f32f32<IntT><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
          _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else if (num_in_channels % 32 == 0 && num_out_channels % 16 == 0)
    {
      int block_num_M = (num_out_feats + 127) / 128;
      int block_num_N = num_out_channels / 16;  //j_factors1
      dim3 num_blocks(block_num_M * block_num_N);
      dim3 threads_per_block(64);
      conv_forward_cuda_setting2_mode0_f32f32f32<IntT><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
          _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
    }
    else
    {
      int block_num_M = (num_out_feats + 127) / 128;
      int block_num_N = (num_out_channels + 15) / 16;  //j_factors1
      dim3 num_blocks(block_num_M * block_num_N);
      dim3 threads_per_block(64);

      if (num_in_channels % 16 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, false, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 8, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 4, false, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 4 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<16, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else if (num_in_channels % 2 == 0)
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<8, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<8, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<8, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<8, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
      else
      {
        if (num_out_channels % 16 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<4, 16, true, false><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 4 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<4, 16, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else if (num_out_channels % 2 == 0)
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<4, 8, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
        else
        {
          conv_forward_cuda_setting1_mode0_f32f32f32<4, 4, true, true><<<num_blocks, threads_per_block, 0, dev_ctx.stream()>>>(
              _out_feats.dims()[0], num_in_channels, num_out_channels, kernel_volume, in_feats, kernel, out_in_map, out_feats);
        }
      }
    }
  }
}

#endif //PADDLE_WITH_CUDA
