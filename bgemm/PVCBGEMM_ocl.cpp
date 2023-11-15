#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <chrono>
#include <thread>

#include <fstream>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define OCL_CHECK(status)                                                      \
  if (status != 0) {                                                           \
    fprintf(stderr, "%s:%d: OCL error %d\n", __FILE__, __LINE__, (int)status); \
    exit(1);                                                                   \
  }

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned short MType;
#include "PVCBGEMM.h"

// NEVER do "using namespace std" this at the global scope, this is anti-pattern
// using namespace std;

int counter = 0;

MType from_fp32(float f) {
  uint *pf = (uint *)&f;
  return (MType)((*pf) >> 16);
}
float to_fp32(MType val) {
  float f;

  MType *temp = (MType *)&f;
  temp[0] = 0;
  temp[1] = val;
  return f;
}

void WriteOut(void *p, int h, int w, char const *filename) {
  FILE *fp = fopen(filename, "w");
  int i, j;
  float f;
  MType *in, *temp;

  in = (MType *)p;
  temp = (MType *)&f;
  temp[0] = 0;

  if (fp != 0) {
    for (j = 0; j < h; j++) {
      for (i = 0; i < w; i++) {
        temp[1] = *in; // assume little endian
        fprintf(fp, "%f,", f);
        in++;
      }
      fprintf(fp, "\n");
    }
    fclose(fp);
  }
}

int FloatCompare(MType *p1, MType *p2, int sz, int M, int N) {
  float ref, kernel;
  ushort *pref, *pkernel;
  MType *tempbuf;

  pref = (ushort *)&ref;
  pkernel = (ushort *)&kernel;

  tempbuf = (MType *)malloc(sz * sizeof(MType));

  // reorder the buffer to match the reference buffer format
  for (int j = 0; j < M; j++)
    for (int i = 0; i < N; i++) {
      int index;

      index = (i >> 4) * 16 * M + (i & 0xf) + j * 16;
      tempbuf[j * N + i] = p1[index];
    }

  for (int i = 0; i < sz; i++) {
    pref[0] = 0;
    pref[1] = p2[i];
    pkernel[0] = 0;
    pkernel[1] = tempbuf[i];
    if (i == 0) {
      printf("first ref=%f, kernel=%f, index=%d\n", ref, kernel, i);
    }
    if (fabs(ref - kernel) > 0.001f * ref) {
      printf("mismatched: ref=%f, kernel=%f, index=%d\n", ref, kernel, i);
      free(tempbuf);
      return 1;
    }
  }

  free(tempbuf);
  return 0;
}

void setMatrix(MType *matrix, int M, int N) {
  MType *p = matrix;
  float f;
  uint *pf;

  pf = (uint *)&f;
  counter = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      f = (float)(rand() % 128);

      counter++;
      *p = (*pf) >> 16;

      p++;

      // this is done so that the value is kept using 7 bits. 7 bit is the size
      // of hte BF16 mantissa. C-model reference code will process these
      // accumulating to fp32. Then the result will be truncated to BF16
      // precision.
    }
}

void prepMatrix(MType *matrix, MType *m, int rowcnt, int colcnt, uchar mode) {
  MType *q = m;
  int c, k, b;

  if (mode) // when it is 1, it prepares the B matrix
  {
#if 0
        //input B format: [K/16][N][16K] instead of  [K][N]. Following is hte order of data (row=0..15,col=0)(row=0..15,col=1)(row=0..15,col=2). row and colume maps to input B[K][N]
        for (c = 0; c < rowcnt; c+=16)
        {
            for (k = 0; k < colcnt; k++)
            {
                for (b = c; b < 16+c; b++)
                {
                    *q++ = matrix[b*colcnt + k];
                }
            }
        }
#else
    // input B format: [K/16][N/8][8K][8N][2K] instead of  [K][N]. Following is
    // hte order of data (row=0..15,col=0)(row=0..15,col=1)(row=0..15,col=2).
    // row and colume maps to input B[K][N]

    for (int h = 0; h < (rowcnt / 16); h++) {
      for (int w = 0; w < (colcnt / 8); ++w) {
        for (uint Kb = 0; Kb < 8; ++Kb) {
          for (uint Ns = 0; Ns < 8; Ns++) {
            for (uint Ks = 0; Ks < 2; Ks++) {

              *q++ = matrix[Ks * colcnt + Ns + Kb * 2 * colcnt + w * 8 +
                            h * 16 * colcnt];
            }
          }
        }
      }
    }

#endif
  } else // when it is 0, it prepares the A matrix
  {
    // input A format: [K/16][M][16K]. Following is hte order of data
    // (row=0,col=0..15)(row=1,col=0..15)(row=2,col=0..15). row and colume maps
    // to input A[M][K].    , row=0, col=0..15 , row=1, col=0..15
    for (c = 0; c < colcnt; c += 16) {
      for (k = 0; k < rowcnt; k++) {
        for (b = c; b < 16 + c; b++) {
          *q++ = matrix[k * colcnt + b];
        }
      }
    }
  }
}

void multipyMatrix(MType *MA, MType *MB, MType *MC, int M, int K, int N) {
  MType *c = MC;
  float *temp;
  unsigned short *p, *pf1, *pf2;
  float f1, f2;

  pf1 = (unsigned short *)&f1;
  pf2 = (unsigned short *)&f2;

  temp = (float *)malloc(M * N * sizeof(float));
  p = (unsigned short *)temp;

  int i = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      temp[i++] = 0.0;

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        pf1[0] = 0;
        pf2[0] = 0;
        pf1[1] = *(MA + m * K + k);
        pf2[1] = *(MB + k * N + n);
        *(temp + m * N + n) += f1 * f2;
      }

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      p++;
      *c++ = *p++; // assume little endian.
    }

  free(temp);
}

int run(int M, int K, int N, int threadWidth, int threadHeight, int groupWidth,
        int groupHeight) {
  int result;

  uint threadNum = threadWidth * threadHeight;
  uint TotalthreadNum = threadNum * groupWidth * groupHeight;
  uint repeat_count = TotalthreadNum / HW_THREAD_COUNT;
  if (TotalthreadNum != repeat_count * HW_THREAD_COUNT) {
    printf("error: the total thread space is not multiple of HW threads "
           "avialable\n");
    exit(-1);
  }
  if (uint(groupHeight) != (groupHeight / repeat_count) * repeat_count) {
    printf("error: group height is not multiple of repeat count:%d,%d,%d\n",
           groupHeight, repeat_count, TotalthreadNum);
    exit(-1);
  }
  // recompute the group height based on repetition that will be done inside the
  // kernel.
  groupHeight /= repeat_count;
  TotalthreadNum = threadNum * groupWidth * groupHeight;

  cl_platform_id platform;
  cl_uint platforms;
  cl_int error = clGetPlatformIDs(1, &platform, &platforms);
  OCL_CHECK(error);
  cl_device_id device;
  cl_uint devices;
  OCL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &devices));
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform, 0};
  // Note that nVidia's OpenCL requires the platform property
  cl_context context =
      clCreateContext(properties, 1, &device, NULL, NULL, &error);
  OCL_CHECK(error);
  cl_command_queue cq =
      clCreateCommandQueueWithProperties(context, device, nullptr, &error);
  OCL_CHECK(error);

  std::ifstream is;
#ifdef BINNAME
  std::string fn = BINNAME;
#else
  std::string fn = "bgemm.kernel.dg2.bin";
#endif
  is.open(fn, std::ios::binary);
  if (!is.good()) {
    fprintf(stderr, "Open %s failed\n", fn.c_str());
    return -1;
  }

  is.seekg(0, std::ios::end);
  size_t codeSize = is.tellg();
  is.seekg(0, std::ios::beg);

  if (codeSize == 0) {
    return -1;
  }

  uchar *codeBin = new uchar[codeSize];
  if (!codeBin) {
    return -1;
  }

  is.read((char *)codeBin, codeSize);
  is.close();

  fprintf(stderr, "codeSize= %d\n", (int)codeSize);

  cl_int errNum = 0;

  cl_program prog = clCreateProgramWithBinary(context, 1, &device, &codeSize,
                                              (const unsigned char **)&codeBin,
                                              &error, &errNum);

  OCL_CHECK(error);

  error = clBuildProgram(prog, 0, NULL, "-cmc", NULL, NULL);
  if (error != 0) {
    fprintf(stderr, " error= %d\n", (int)error);
    size_t log_length = 0;
    OCL_CHECK(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, 0,
                                    &log_length));

    std::vector<uchar> log(log_length);

    OCL_CHECK(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG,
                                    log_length, &log[0], 0));

    fprintf(stderr,
            "Error happened during the build of OpenCL prog.\n"
            "Build log:\n %s\n",
            log.data());
    exit(0);
  }

  // allocate bufffers
  MType *matrixA = (MType *)malloc(M * K * sizeof(MType));
  if (matrixA == 0) {
    printf("Memory A allocation error");
    return -1;
  }
  MType *matrixB = (MType *)malloc(K * N * sizeof(MType));
  if (matrixB == 0) {
    free(matrixA);
    printf("Memory B allocation error");
    return -1;
  }
  MType *matrixC = (MType *)malloc(M * N * sizeof(MType));
  if (matrixC == 0) {
    free(matrixA);
    free(matrixB);
    printf("Memory C allocation error");
    return -1;
  }

  MType *mA = (MType *)malloc(M * K * sizeof(MType));
  if (mA == 0) {
    free(matrixA);
    free(matrixB);
    free(matrixC);
    printf("Memory A allocation error");
    return -1;
  }
  MType *mB = (MType *)malloc(K * N * sizeof(MType));
  if (mB == 0) {
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(mA);
    printf("Memory B allocation error");
    return -1;
  }
  MType *matrixC_ref = (MType *)malloc(M * N * sizeof(MType));
  if (matrixC_ref == 0) {
    free(matrixA);
    free(matrixB);
    free(matrixC);
    free(mA);
    free(mB);
    printf("Memory C allocation error");
    return -1;
  }

  std::memset(matrixC, 0, M * N * sizeof(MType));
  std::memset(matrixC_ref, 0, M * N * sizeof(MType));
  setMatrix(matrixA, M, K);
  setMatrix(matrixB, K, N);
  WriteOut(matrixA, M, K, "matrixA.csv");
  WriteOut(matrixB, K, N, "matrixB.csv");

  multipyMatrix(matrixA, matrixB, matrixC_ref, M, K, N);
  WriteOut(matrixC_ref, M, N, "matrixC_ref.csv");

  prepMatrix(matrixB, mB, K, N, 1);
  WriteOut(mB, K, N, "mB.csv");
  prepMatrix(matrixA, mA, M, K, 0);
  WriteOut(mA, M, K, "mA.csv");

  // alloc & copy device matrix A
  cl_mem dInputMatrixA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        M * K * sizeof(MType), NULL, &error);
  OCL_CHECK(error);
  OCL_CHECK(clEnqueueWriteBuffer(cq, dInputMatrixA, CL_TRUE, 0,
                                 M * K * sizeof(MType), mA, 0, NULL, NULL));

  // alloc & copy device matrix B
  cl_mem dInputMatrixB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        K * N * sizeof(MType), NULL, &error);
  OCL_CHECK(error);
  OCL_CHECK(clEnqueueWriteBuffer(cq, dInputMatrixB, CL_TRUE, 0,
                                 K * N * sizeof(MType), mB, 0, NULL, NULL));

  // alloc & copy device matrix C
  cl_mem dOutputMatrixC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         M * N * sizeof(MType), NULL, &error);
  OCL_CHECK(error);
  OCL_CHECK(clEnqueueWriteBuffer(cq, dOutputMatrixC, CL_TRUE, 0,
                                 M * N * sizeof(MType), matrixC, 0, NULL,
                                 NULL));

  // get a handle and map parameters for the kernel
  cl_kernel k_bgemm = clCreateKernel(prog, "PVCBGEMM", &error);
  OCL_CHECK(error);

  // set arguments
  OCL_CHECK(clSetKernelArg(k_bgemm, 0, sizeof(cl_mem), &dInputMatrixA));
  OCL_CHECK(clSetKernelArg(k_bgemm, 1, sizeof(cl_mem), &dInputMatrixB));
  OCL_CHECK(clSetKernelArg(k_bgemm, 2, sizeof(cl_mem), &dOutputMatrixC));
  OCL_CHECK(clSetKernelArg(k_bgemm, 3, sizeof(int), &M));
  OCL_CHECK(clSetKernelArg(k_bgemm, 4, sizeof(int), &K));
  OCL_CHECK(clSetKernelArg(k_bgemm, 5, sizeof(int), &N));
  OCL_CHECK(clSetKernelArg(k_bgemm, 6, sizeof(int), &repeat_count));

  int MatAReadIncSizeByte = M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  int MatBReadIncSizeByte = N * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  OCL_CHECK(clSetKernelArg(k_bgemm, 7, sizeof(int), &MatAReadIncSizeByte));
  OCL_CHECK(clSetKernelArg(k_bgemm, 8, sizeof(int), &MatBReadIncSizeByte));
  int StepSizeForSecondReadByte = (4 * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE);
  OCL_CHECK(clSetKernelArg(k_bgemm, 9, sizeof(int), &StepSizeForSecondReadByte));
  OCL_CHECK(clSetKernelArg(k_bgemm, 10, sizeof(int), &groupHeight));

  constexpr int GRIDDIM = 2;
  size_t localsize[GRIDDIM] = {(size_t)threadWidth, (size_t)threadHeight};
  size_t globalsize[GRIDDIM] = {(size_t)groupWidth * localsize[0],
                                (size_t)groupHeight * localsize[1]};

  fprintf(stderr, "localsize= %d %d\n", (int)localsize[0], (int)localsize[1]);
  fprintf(stderr, "globalsize= %d %d\n", (int)globalsize[0],
          (int)globalsize[1]);
  fprintf(stderr, "thread_space= %d %d\n", threadWidth, threadHeight);
  fprintf(stderr, "group_space= %d %d\n", groupWidth, groupHeight);
  fprintf(stderr, "repeat_count= %d\n", repeat_count);
  fprintf(stderr, "M= %d\n", M);
  fprintf(stderr, "K= %d\n", K);
  fprintf(stderr, "N= %d\n", N);



  fprintf(stderr, "Execute kernel .. \n");

  for(int i=0; i<1e5; i++){
    cl_event e = nullptr;
    OCL_CHECK(clEnqueueNDRangeKernel(cq, k_bgemm, GRIDDIM, NULL, globalsize,
                                  localsize, 0, NULL, &e));
    OCL_CHECK(clWaitForEvents(1, &e));
    OCL_CHECK(clReleaseEvent(e));

  }
  fprintf(stderr, " -- done executing\n");

  // copy buffer
  OCL_CHECK(clEnqueueReadBuffer(cq, dOutputMatrixC, CL_TRUE, 0,
                              M * N * sizeof(MType), matrixC, 0, NULL, NULL));

  // write to disk
  WriteOut(matrixC, M, N, "matrixC.csv");

  // validate
  result = FloatCompare(matrixC, matrixC_ref, M * N, M, N);
  if (!result) {
    fprintf(stderr, "\n*** TEST PASSED ***\n");
  } else {
    fprintf(stderr, "\n*** TEST FAILED ***\n");
  }

  delete[] codeBin;

  free(matrixC_ref);
  free(matrixC);
  free(matrixB);
  free(matrixA);

  return result;
}

int main(int argc, char *argv[]) {
  // srand((unsigned) time(0));
  // cout << "-m <M> -k <K> -n <N>, -w <threadWidth> -h <threadHeight> -p
  // <groupWidth> -q <groupHeight>" << endl;

  int M = MATRIX_M, K = MATRIX_K, N = MATRIX_N, tW = WG_TILE_M / SG_TILE_M,
      tH = WG_TILE_N / SG_TILE_N, gH = M / WG_TILE_M, gW = N / WG_TILE_N;
  int i = 0;
  while (argc > i) {
    if (strcmp(argv[i], "-m") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      M = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-k") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      K = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-n") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      N = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-w") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      tW = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-h") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      tH = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-p") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      gW = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-q") == 0 && argc > i + 1 && argv[i + 1][0] != '-') {
      gH = atoi(argv[i + 1]);
    }
    i++;
  }


  int failed = run(M, K, N, tW, tH, gW, gH);

  if (failed) {
      return -1;
  }


  return 0;
}
