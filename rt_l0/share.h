/*
 * Copyright (c) 2017, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once

#define HETER

#if !defined(GEN_KERNEL)
#define fptype float
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifdef __GNUC__
#include <unistd.h>

static __inline__ unsigned long long __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#else
#include <windows.h>
#endif

using namespace std;

static float randData(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

inline double get_cpu_freq() {
    unsigned long long t0, t1;
    t0 = __rdtsc();
#ifdef _WIN32
    Sleep(1000);
#else
    sleep(1);
#endif
    t1 = __rdtsc();
    return (double)(t1-t0);
}

#endif

#define TILE_m 32
#define TILE_n 16
#define TILE_k 8

#define GEMM_BLOCK 1024

// ------------------PVC GEMM------------------------------

#define SLMMTA

#define DPASW 1

#define TILE_32x32
//#define TILE_28x32

#ifdef TILE_32x32

#if 0
#define MATRIX_M 2048  // Matrix size M, Must be mutliple of 32 and multiple of WG_TILE_M
#define MATRIX_K 1024  // Matrix size K, Must be mutliple of 32
#define MATRIX_N 2048  // Matrix size N, Must be mutliple of 32 and multiple of WG_TILE_N
#define HW_THREAD_COUNT 512
#elif 0
#define MATRIX_M 1024  // Matrix size M, Must be mutliple of 32 and multiple of WG_TILE_M
#define MATRIX_K 1024  // Matrix size K, Must be mutliple of 32
#define MATRIX_N 1024  // Matrix size N, Must be mutliple of 32 and multiple of WG_TILE_N
#define HW_THREAD_COUNT 512
#else
#define MATRIX_M 128  // Matrix size M, Must be mutliple of 32 and multiple of WG_TILE_M
#define MATRIX_K 128  // Matrix size K, Must be mutliple of 32
#define MATRIX_N 128  // Matrix size N, Must be mutliple of 32 and multiple of WG_TILE_N
#define HW_THREAD_COUNT 16
#endif

#define WG_TILE_M 128  // Work-Group tile size M, Must be mutliple of 32
#define WG_TILE_N 128  // Work-Group tile size N, Must be mutliple of 32

#define SG_TILE_M 32
#define SG_TILE_N 32

#endif //32x32

#ifdef TILE_28x32

#if 0

#define MATRIX_M 112
#define MATRIX_K 512
#define MATRIX_N 128
#define HW_THREAD_COUNT 16

#else
#define MATRIX_M 50176
//#define MATRIX_M 784
#define MATRIX_K 128
#define MATRIX_N 512
#define HW_THREAD_COUNT 4096

#endif

#define WG_TILE_M 112
#define WG_TILE_N 128

#define SG_TILE_M 28
#define SG_TILE_N 32

#endif //28x32 tile


//#define PREFETCH
#define SIZE_OF_BF16_BYTE 2
#define SIZE_OF_UINT32 4
#define SIZE_OF_FP32_BYTE 4
#define CONTIGUOUS_K_SIZE 16
