#pragma once

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
#elif 1
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
