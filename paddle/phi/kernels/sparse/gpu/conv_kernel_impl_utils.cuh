#pragma once

#ifdef PADDLE_WITH_CUDA

#include <cuda_fp16.h>
#include <stdexcept>
#include "paddle/phi/common/float16.h"

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y)
{
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#endif //PADDLE_WITH_CUDA
