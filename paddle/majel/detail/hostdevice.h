#pragma once

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define HOST
#endif
