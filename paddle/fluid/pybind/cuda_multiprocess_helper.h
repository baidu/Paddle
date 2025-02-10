// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifndef HELPER_MULTIPROCESS_H
#define HELPER_MULTIPROCESS_H

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <aclapi.h>
#include <sddl.h>
#include <stdio.h>
#include <strsafe.h>
#include <tchar.h>
#include <windows.h>
#include <winternl.h>
#include <iostream>
#else
#include <fcntl.h>
#include <memory.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#endif

typedef struct shmStruct_st {
  size_t nprocesses;
  cudaIpcMemHandle_t memHandle;
} shmStruct;

typedef struct sharedMemoryInfo_st {
  void *addr;
  size_t size;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  HANDLE shmHandle;
#else
  int shmFd;
#endif
} sharedMemoryInfo;

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  info->size = sz;

  info->shmHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, name);
  if (info->shmHandle == 0) {
    return GetLastError();
  }

  info->addr = MapViewOfFile(info->shmHandle, FILE_MAP_ALL_ACCESS, 0, 0, sz);
  if (info->addr == NULL) {
    return GetLastError();
  }

  return 0;
#else
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
#endif
}

void sharedMemoryClose(sharedMemoryInfo *info) {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  if (info->addr) {
    UnmapViewOfFile(info->addr);
  }
  if (info->shmHandle) {
    CloseHandle(info->shmHandle);
  }
#else
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
#endif
}

#endif  // HELPER_MULTIPROCESS_H
