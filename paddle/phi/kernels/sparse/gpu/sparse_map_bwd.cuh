#pragma once
#include <cmath>

__global__ void convert_out_in_map_kernel(const int* out_in_map, int* out_in_map_t, int n, int kernel_volume){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n * kernel_volume) return;
  int input_idx = out_in_map[idx];
  if(input_idx < 0) return;
  out_in_map_t[idx % kernel_volume + input_idx * kernel_volume] = idx / kernel_volume;
}

template <typename IntT>
void convert_transposed_out_in_map(const phi::GPUContext& dev_ctx, const phi::DenseTensor& out_in_map, phi::DenseTensor* out_in_map_t) {
  auto out_in_map_ = const_cast<int*>(out_in_map.data<int>());
  auto out_in_map_t_ = const_cast<int*>(out_in_map_t->data<int>());
  convert_out_in_map_kernel<<<(out_in_map.dims()[0] * out_in_map.dims()[1] + 255) / 256, 256, 0, dev_ctx.stream()>>>(
    out_in_map_, out_in_map_t_, out_in_map.dims()[0], out_in_map.dims()[1]);
}
