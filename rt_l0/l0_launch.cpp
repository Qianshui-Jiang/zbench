#include "l0_rt_helpers.h"
#include "share.h"
#include "Matrix.h"

#include "l0_launch.h"
#ifndef _WIN32
#define FALSE 0
#define TRUE  1
#endif



int counter = 0;

void WriteOut(void *p, int h, int w, char const *filename) {
  FILE *fp = fopen(filename, "w");
  int i, j;
  float f;
  MType *in, *temp;

  in = (MType *)p;
  temp = (MType *)&f;
  temp[0] = 0;  // convert to FP32

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
    // the order of data (row=0..15,col=0)(row=0..15,col=1)(row=0..15,col=2).
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
      temp[i++] = 0.0;  // init

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        pf1[0] = 0;
        pf2[0] = 0;
        pf1[1] = *(MA + m * K + k);
        pf2[1] = *(MB + k * N + n);
        *(temp + m * N + n) += f1 * f2; // fp32*fp32
      }

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      p++;
      *c++ = *p++; // assume little endian. to bf16
    }

  free(temp);
}

#if 0

int _calc_bgemm(MType *mA, MType *mB, MType *matrixC, int M, int K, int N, 
                int threadWidth, int threadHeight,
                int groupWidth, int groupHeight,
                const char* bin_file =  "bgemm_dpas_genx.bin", 
                const char* fn_name = "bgemm_dpas"){

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

  
  L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // Discover all the driver instances
  uint32_t driverCount = 0;
  L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));
  fprintf(stderr, "driverCount= %d\n", (int)driverCount);

  ze_driver_handle_t *allDrivers =
      (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
  L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers));

  // Find a driver instance with a GPU device
  ze_driver_handle_t hDriver = nullptr;
  ze_device_handle_t hDevice = nullptr;
  for (uint32_t i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    hDriver = allDrivers[i];
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
    fprintf(stderr, "driver= %d: deviceCount= %d\n", (int)i, (int)deviceCount);
    ze_device_handle_t *allDevices =
        (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, allDevices));
    for (uint32_t d = 0; d < deviceCount; ++d) {
      ze_device_properties_t device_properties;
      device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      device_properties.pNext = nullptr;
      L0_SAFE_CALL(zeDeviceGetProperties(allDevices[d], &device_properties));
      if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
        hDevice = allDevices[d];
        break;
      }
    }
    free(allDevices);
    if (nullptr != hDevice) {
      break;
    }
  }
  free(allDrivers);
  assert(hDriver);
  assert(hDevice);

  // Create context
  ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t hContext = nullptr;
  L0_SAFE_CALL(zeContextCreate(hDriver, &contextDesc, &hContext));
  assert(hContext);

  // Create a command queue
  ze_command_queue_desc_t commandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
      ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t hCommandQueue;
  L0_SAFE_CALL(zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue));

  // Create a command list
  ze_command_list_desc_t commandListDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, 0, 0, 0};
  ze_command_list_handle_t hCommandList;
  L0_SAFE_CALL(zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList));


  uint8_t *kernel_bin{nullptr};
  size_t kernel_sz{0};
  {
    FILE *fp = fopen(bin_file, "rb");
    assert(fp);

    fseek(fp, 0, SEEK_END);
    kernel_sz = ftell(fp);
    rewind(fp);

    kernel_bin = new uint8_t[kernel_sz];
    fread(kernel_bin, 1, kernel_sz, fp);
    fclose(fp);
  }
  fprintf(stderr, "kernel_sz= %g KB\n", kernel_sz / 1024.0);

  ze_module_desc_t moduleDesc = {
    ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
    ZE_MODULE_FORMAT_NATIVE, //
    kernel_sz,  //
    kernel_bin, //
    "-cmc",
    nullptr
  };
  ze_module_handle_t hModule;
  L0_SAFE_CALL(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr));

  ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, fn_name};
  ze_kernel_handle_t hKernel;
  L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));



  // allocate l0 buffers
  ze_device_mem_alloc_desc_t deviceMemDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  size_t bufsize_A = M * K * sizeof(MType);
  size_t bufsize_B = K * N * sizeof(MType);
  size_t bufsize_C = M * N * sizeof(MType);

  void *dBufA = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_A, 64, hDevice, &dBufA));
  void *dBufB = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_B, 64, hDevice, &dBufB));
  void *dBufC = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_C, 64, hDevice, &dBufC));

  // copy buffers to device
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufA, mA, bufsize_A, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufB, mB, bufsize_B, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufC, matrixC, bufsize_C, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

  // set kernel arguments
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(dBufA), &dBufA));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(dBufB), &dBufB));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(dBufC), &dBufC));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(int), &M));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &N));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &K));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(int), &repeat_count));


  int MatAReadIncSizeByte = M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  int MatBReadIncSizeByte = N * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 7, sizeof(int), &MatAReadIncSizeByte));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 8, sizeof(int), &MatBReadIncSizeByte));
  int StepSizeForSecondReadByte = (4 * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE);
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 9, sizeof(int), &StepSizeForSecondReadByte));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 10, sizeof(int), &groupHeight));


  fprintf(stderr, "Execute kernel .. \n");

  // set group size
  L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ threadWidth, /*y*/ threadHeight, /*z*/ 1));

  // set grid size
  ze_group_count_t groupCount = {groupWidth, groupHeight, 1};

  // launch
  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &groupCount, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
  // copy result to host
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, matrixC, dBufC, bufsize_C, nullptr, 0, nullptr));

  // dispatch & wait
  L0_SAFE_CALL(zeCommandListClose(hCommandList));
  L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
  L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()))

  L0_SAFE_CALL(zeMemFree(hContext, dBufA));
  L0_SAFE_CALL(zeMemFree(hContext, dBufB));
  L0_SAFE_CALL(zeMemFree(hContext, dBufC));

  delete kernel_bin;
  return 0;
}
#else

int _calc_bgemm(MType *mA, MType *mB, MType *matrixC, int M, int K, int N, 
                int threadWidth, int threadHeight,
                int groupWidth, int groupHeight,
                const char* bin_file =  "bgemm_dpas_genx.bin", 
                const char* fn_name = "bgemm_dpas"){

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

  
  L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Find a driver instance with a GPU device
  auto [hDriver, hDevice, hContext] = findDriverAndDevice();
  auto hCommandList = createImmCommandList(hContext, hDevice);

  auto hKernel = createKernel(hContext, hDevice, bin_file, fn_name);
  // set group size
  L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ threadWidth, /*y*/ threadHeight, /*z*/ 1));

  // allocate l0 buffers
  ze_device_mem_alloc_desc_t deviceMemDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  size_t bufsize_A = M * K * sizeof(MType);
  size_t bufsize_B = K * N * sizeof(MType);
  size_t bufsize_C = M * N * sizeof(MType);

  void *dBufA = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_A, 64, hDevice, &dBufA));
  void *dBufB = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_B, 64, hDevice, &dBufB));
  void *dBufC = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_C, 64, hDevice, &dBufC));
  
  // copy buffers to device
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufA, mA, bufsize_A, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufB, mB, bufsize_B, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufC, matrixC, bufsize_C, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

  // set kernel arguments
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(dBufA), &dBufA));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(dBufB), &dBufB));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(dBufC), &dBufC));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(int), &M));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &N));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &K));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(int), &repeat_count));


  int MatAReadIncSizeByte = M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  int MatBReadIncSizeByte = N * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 7, sizeof(int), &MatAReadIncSizeByte));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 8, sizeof(int), &MatBReadIncSizeByte));
  int StepSizeForSecondReadByte = (4 * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE);
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 9, sizeof(int), &StepSizeForSecondReadByte));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 10, sizeof(int), &groupHeight));
  


  fprintf(stderr, "Execute kernel and measure.. \n");
  // setup grid size and get a hEvent
  ze_event_handle_t hEvent = createEvent(hContext, hDevice);
  ze_group_count_t groupCount = {groupWidth, groupHeight, 1};
  double thost = 0.0f;
  unsigned long long kernel_ns = 0;

  // launch & measure
  double host_start = getTimeStamp();

  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &groupCount, hEvent, 0, nullptr));
  L0_SAFE_CALL(zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max()));
  
  int niterations = 100;
  for (int i=0; i<niterations; i++){
    double host_end = getTimeStamp();
    thost += (host_end - host_start);
    ze_kernel_timestamp_result_t timestamp;
    zeEventQueryKernelTimestamp(hEvent, &timestamp);
    kernel_ns += (timestamp.context.kernelEnd - timestamp.context.kernelStart);

    L0_SAFE_CALL(zeCommandListReset(hCommandList)); // reset Command list
    L0_SAFE_CALL(zeEventHostReset(hEvent)); // reset event

  }
    thost = thost * 1000.0f / niterations;
    double tkern = kernel_ns / 1000000.0f / niterations;


    printf("%-18s%.4lf msec\n","kern time:", tkern);
    printf("%-18s%.4lf msec\n","host time:", thost);

    double gflops;
    gflops = ((2000.0f*M*N*K) / (1.0f*1024*1024*1024)) / tkern;
    printf("GEN SGEMM (kern-timer): %8.2lf Gflops\n",  gflops);
    gflops = ((2000.0f*M*N*K) / (1.0f*1024*1024*1024)) / thost;
    printf("GEN SGEMM (host-timer): %8.2lf Gflops\n", gflops);

  // copy result to host & wait
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, matrixC, dBufC, bufsize_C, hEvent, 0, nullptr));
  L0_SAFE_CALL(zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max()));


  L0_SAFE_CALL(zeMemFree(hContext, dBufA));
  L0_SAFE_CALL(zeMemFree(hContext, dBufB));
  L0_SAFE_CALL(zeMemFree(hContext, dBufC));

  L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
  L0_SAFE_CALL(zeContextDestroy(hContext));
  return 0;
}

#endif


std::vector<MType> run_kernel(const char* bin_file , const char* spirv_file, const char* fn_name,
                              const py::args& args, const py::kwargs& kwargs){
// Parse the input
  std::vector<int> v_argc;
  std::vector<MType *> v_argv;
  if (kwargs) {
    for(auto item: kwargs){
        std::cout << "input key: "<< item.first <<  " , type: "\
        << py::type::of(item.second).str() << std::endl; // <class 'numpy.ndarray'>
      if(py::type::of(item.second) == py::type::of(py::int_())){
        auto arg_in = py::int_(kwargs[item.first]);
        v_argc.push_back(arg_in);
      }
      if(py::type::of(item.second) == py::type::of(py::array())){
        auto buf_in = py::array_t<MType, py::array::c_style | py::array::forcecast>(kwargs[item.first]);
        v_argv.push_back(const_cast<MType*>(buf_in.data()));
      }
    }
  }
// ----------------A, B, m, k, n, tx, ty, gx, gy------------------
  int M = v_argc[0],  K=v_argc[1], N=v_argc[2];
  int threadWidth=v_argc[3], threadHeight=v_argc[4];
  int groupWidth=v_argc[5], groupHeight=v_argc[6];

  MType *matrixA = v_argv[0];
  MType *matrixB = v_argv[1];

  MType *matrixC = (MType *)malloc(M * N * sizeof(MType));
  if (matrixC == 0) {
    free(matrixA);
    free(matrixB);
    printf("Memory C allocation error");
  }
  std::memset(matrixC, 0, M * N * sizeof(MType));

  // Packing matrix datas
  MType *mA = (MType *)malloc(M * K * sizeof(MType));
  if (mA == 0) {
    printf("Memory A allocation error");
  }
  MType *mB = (MType *)malloc(K * N * sizeof(MType));
  if (mB == 0) {
    free(mA);
    printf("Memory B allocation error");
  }

  prepMatrix(matrixA, mA, M, K, 0); // A format: [K/16][M][16K]
  WriteOut(mA, M, K, "mA_bind.csv");
  prepMatrix(matrixB, mB, K, N, 1); // mB format: [K/16][N/8][8K][8N][2K]
  WriteOut(mB, K, N, "mB_bind.csv"); 

  _calc_bgemm(mA, mB, matrixC, M, K, N, 
              threadWidth,  threadHeight, groupWidth, groupHeight,
              bin_file , fn_name);

#if 0
  MType *matrixC_ref = (MType *)malloc(M * N * sizeof(MType));
  if (matrixC_ref == 0) {
    free(matrixA);
    free(matrixB);
    free(matrixC);
    printf("Memory C allocation error");
  }

  WriteOut(matrixA, M, K, "matrixA_bind.csv");
  WriteOut(matrixB, K, N, "matrixB_bind.csv");

  // Get gold result
  std::memset(matrixC_ref, 0, M * N * sizeof(MType));
  multipyMatrix(matrixA, matrixB, matrixC_ref, M, K, N);
  WriteOut(matrixC_ref, M, N, "matrixC_ref.csv");
  WriteOut(matrixC, M, N, "matrixC.csv");

 // validate
  int match = FloatCompare(matrixC, matrixC_ref, M * N, M, N);
  if (!match) {
    fprintf(stderr, "\n*** TEST PASSED ***\n");
  } else {
    fprintf(stderr, "\n*** TEST FAILED ***\n");
  }
  free(matrixC_ref);
#endif
  
  std::vector<MType> result(M * N);
  memcpy(&result[0], matrixC, M*N*sizeof(MType));
  free(matrixC);

  return result;
  
}

int run_bgemm(int M, int K, int N, 
            int threadWidth, int threadHeight,
            int groupWidth, int groupHeight,
            const char* bin_file =  "bgemm_dpas_genx.bin", 
            const char* fn_name = "bgemm_dpas"){

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


  L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // Discover all the driver instances
  uint32_t driverCount = 0;
  L0_SAFE_CALL(zeDriverGet(&driverCount, nullptr));
  fprintf(stderr, "driverCount= %d\n", (int)driverCount);

  ze_driver_handle_t *allDrivers =
      (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
  L0_SAFE_CALL(zeDriverGet(&driverCount, allDrivers));

  // Find a driver instance with a GPU device
  ze_driver_handle_t hDriver = nullptr;
  ze_device_handle_t hDevice = nullptr;
  for (uint32_t i = 0; i < driverCount; ++i) {
    uint32_t deviceCount = 0;
    hDriver = allDrivers[i];
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, nullptr));
    fprintf(stderr, "driver= %d: deviceCount= %d\n", (int)i, (int)deviceCount);
    ze_device_handle_t *allDevices =
        (ze_device_handle_t *)malloc(deviceCount * sizeof(ze_device_handle_t));
    L0_SAFE_CALL(zeDeviceGet(hDriver, &deviceCount, allDevices));
    for (uint32_t d = 0; d < deviceCount; ++d) {
      ze_device_properties_t device_properties;
      device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      device_properties.pNext = nullptr;
      L0_SAFE_CALL(zeDeviceGetProperties(allDevices[d], &device_properties));
      if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
        hDevice = allDevices[d];
        break;
      }
    }
    free(allDevices);
    if (nullptr != hDevice) {
      break;
    }
  }
  free(allDrivers);
  assert(hDriver);
  assert(hDevice);

  // Create context
  ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  ze_context_handle_t hContext = nullptr;
  L0_SAFE_CALL(zeContextCreate(hDriver, &contextDesc, &hContext));
  assert(hContext);

  // Create a command queue
  ze_command_queue_desc_t commandQueueDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0, 0,
      ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  ze_command_queue_handle_t hCommandQueue;
  L0_SAFE_CALL(zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue));

  // Create a command list
  ze_command_list_desc_t commandListDesc = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, 0, 0, 0};
  ze_command_list_handle_t hCommandList;
  L0_SAFE_CALL(zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList));


  uint8_t *kernel_bin{nullptr};
  size_t kernel_sz{0};
  {
    FILE *fp = fopen(bin_file, "rb");
    assert(fp);

    fseek(fp, 0, SEEK_END);
    kernel_sz = ftell(fp);
    rewind(fp);

    kernel_bin = new uint8_t[kernel_sz];
    fread(kernel_bin, 1, kernel_sz, fp);
    fclose(fp);
  }
  fprintf(stderr, "kernel_sz= %g KB\n", kernel_sz / 1024.0);

  ze_module_desc_t moduleDesc = {
    ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr,
    ZE_MODULE_FORMAT_NATIVE, //
    kernel_sz,  //
    kernel_bin, //
    "-cmc",
    nullptr
  };
  ze_module_handle_t hModule;
  L0_SAFE_CALL(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr));

  ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, fn_name};
  ze_kernel_handle_t hKernel;
  L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));

  // Prepare matrix datas
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
  setMatrix(matrixA, M, K);
  setMatrix(matrixB, K, N);
  WriteOut(matrixA, M, K, "matrixA.csv");
  WriteOut(matrixB, K, N, "matrixB.csv");
  std::memset(matrixC, 0, M * N * sizeof(MType));
  std::memset(matrixC_ref, 0, M * N * sizeof(MType));

  // Get gold result
  multipyMatrix(matrixA, matrixB, matrixC_ref, M, K, N);
  WriteOut(matrixC_ref, M, N, "matrixC_ref.csv");

  prepMatrix(matrixA, mA, M, K, 0); // A format: [K/16][M][16K]
  WriteOut(mA, M, K, "mA.csv");
  prepMatrix(matrixB, mB, K, N, 1); // mB format: [K/16][N/8][8K][8N][2K]
  WriteOut(mB, K, N, "mB.csv"); 


  // allocate l0 buffers
  ze_device_mem_alloc_desc_t deviceMemDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
  size_t bufsize_A = M * K * sizeof(MType);
  size_t bufsize_B = K * N * sizeof(MType);
  size_t bufsize_C = M * N * sizeof(MType);

  void *dBufA = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_A, 64, hDevice, &dBufA));
  void *dBufB = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_B, 64, hDevice, &dBufB));
  void *dBufC = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc, bufsize_C, 64, hDevice, &dBufC));

  // copy buffers to device
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufA, mA, bufsize_A, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufB, mB, bufsize_B, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufC, matrixC, bufsize_C, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

  // set kernel arguments
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(dBufA), &dBufA));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(dBufB), &dBufB));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(dBufC), &dBufC));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(int), &M));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &N));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &K));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(int), &repeat_count));


  int MatAReadIncSizeByte = M * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  int MatBReadIncSizeByte = N * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE;
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 7, sizeof(int), &MatAReadIncSizeByte));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 8, sizeof(int), &MatBReadIncSizeByte));
  int StepSizeForSecondReadByte = (4 * CONTIGUOUS_K_SIZE * SIZE_OF_BF16_BYTE);
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 9, sizeof(int), &StepSizeForSecondReadByte));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 10, sizeof(int), &groupHeight));


  fprintf(stderr, "Execute kernel .. \n");

  // set group size
  L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ threadWidth, /*y*/ threadHeight, /*z*/ 1));

  // set grid size
  ze_group_count_t groupCount = {groupWidth, groupHeight, 1};

  // launch
  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &groupCount, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
  // copy result to host
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, matrixC, dBufC, bufsize_C, nullptr, 0, nullptr));

  // dispatch & wait
  L0_SAFE_CALL(zeCommandListClose(hCommandList));
  L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
  L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()))

  L0_SAFE_CALL(zeMemFree(hContext, dBufA));
  L0_SAFE_CALL(zeMemFree(hContext, dBufB));
  L0_SAFE_CALL(zeMemFree(hContext, dBufC));

  bool result = false;
  // write to disk
  WriteOut(matrixC, M, N, "matrixC.csv");

  // validate
  result = FloatCompare(matrixC, matrixC_ref, M * N, M, N);
  if (!result) {
    fprintf(stderr, "\n*** TEST PASSED ***\n");
  } else {
    fprintf(stderr, "\n*** TEST FAILED ***\n");
  }

  delete kernel_bin;

  free(matrixC_ref);
  free(matrixC);
  free(matrixB);
  free(matrixA);

  return result;
}

int run_sgemm(int m, int niterations, int gx, int gy, 
            const char* bin_file =  "sgemm_genx.bin", 
            const char* fn_name = "sgemm_kernel")
{
    storage_type_t st = RowMajor;
    float alpha=+1.0, beta=+1.0;

    // Each thread computes 32x16 block of result matrix
    unsigned nthreadsY    = GEMM_BLOCK/32;  // GEMM_BLOCK = 1024
    unsigned nthreadsX    = GEMM_BLOCK/16;

    int n=m, k=m;

    // Initialization
    m = (m / TILE_m) * TILE_m;
    n=k=m;

    int lda = ((k+15)&~15);
    int ldb = ((n+15)&~15);
    int ldc = ldb;
    printf("SGEMM: C(%d, %d) = %.2f * C(%d, %d) + %.2f A(%d, %d) * B(%d, %d)\n", m,n,beta, m,n, alpha, m,k,k,n);
    printf("Row Threads:%d Col Threads:%d\n", nthreadsY, nthreadsX);
    printf("Thread-group setting: %d x %d \n", gx, gy);

    // Find a driver instance with a GPU device
    auto [hDriver, hDevice, hContext] = findDriverAndDevice();
    auto hCommandList = createImmCommandList(hContext, hDevice);

    // Allocate matrices
    Matrix A(m, k, lda, NULL, true, "A", st);
    Matrix B(k, n, ldb, NULL, true, "B", st);
    Matrix C_gold(m, n, ldc, NULL, false, "C_gold",  st);
    Matrix C(C_gold, "C");
    Matrix zero(C_gold, "C");

    if (niterations == 1) {
        printf("** validation run, only one iteration **\n");
        printf("** For performance run, add cmd-args: Sgemm 2048 1000 ** \n");
        // Compute gold result
        printf("Compute gold result\n");

        sgemmNxN(m, n, k, alpha, &A(0,0), A.l_dim(),
                 &B(0,0), B.l_dim(), beta, &C_gold(0,0), C_gold.l_dim());

    }
    else
        printf("CPU result not computed: Make #iterations=1 to compute CPU result\n");

    ze_image_format_t fmt = {ZE_IMAGE_FORMAT_LAYOUT_32, ZE_IMAGE_FORMAT_TYPE_FLOAT};
    auto hAImage = createImage2D(hContext, hDevice, hCommandList, fmt, A.l_dim(), m, &A(0,0));
    auto hBImage = createImage2D(hContext, hDevice, hCommandList, fmt, B.l_dim(), B.n_row(), &B(0,0));
    auto hCImage = createImage2D(hContext, hDevice, hCommandList, fmt, C.l_dim(), m, &C(0,0));

    ze_group_count_t launchArgs = {nthreadsX/gx, nthreadsY/gy, 1}; // setup how much threads used in a group

    auto hKernel = createKernel(hContext, hDevice, bin_file, fn_name);

    L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, gx, gy, 1)); // setup how much group used

    ze_event_handle_t hEvent = createEvent(hContext, hDevice);
    double thost = 0.0f;
    unsigned long long kernel_ns = 0;
    for (int i=0; i<niterations; i++)
        for(int ib=0; ib < m; ib += GEMM_BLOCK)
            for(int jb=0; jb < n; jb += GEMM_BLOCK)
                for(int kb=0; kb < k; kb += GEMM_BLOCK)
                {
                    setKernelArgs(hKernel, &m, &n, &k, &ib, &jb, &kb, &hAImage, &hBImage, &hCImage);
                    double host_start = getTimeStamp();
                    appendLaunchKernel(hCommandList, hKernel, &launchArgs, hEvent);
                    zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max());

                    double host_end = getTimeStamp();
                    thost += (host_end - host_start);
                    ze_kernel_timestamp_result_t timestamp;
                    zeEventQueryKernelTimestamp(hEvent, &timestamp);
                    kernel_ns += (timestamp.context.kernelEnd - timestamp.context.kernelStart);

                    reset(hEvent);
                    reset(hCommandList);
                }
    // average time in msec
    thost = thost * 1000.0f / niterations;
    double tkern = kernel_ns / 1000000.0f / niterations;

    Matrix C_test(C_gold, "C");
    copyToMemory(hCommandList, (void*)&C_test(0,0), hCImage, hEvent);
    zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max());

    printf("%-18s%.4lf msec\n","kern time:", tkern);
    printf("%-18s%.4lf msec\n","host time:", thost);

    double gflops;
    //gflops = ((2000.0f*m*n*k) / (1.0f*1024*1024*1024)) / tkern;
    //printf("GEN SGEMM (kern-timer): %8.2lf Gflops\n",  gflops);
    gflops = ((2000.0f*m*n*k) / (1.0f*1024*1024*1024)) / thost;
    printf("GEN SGEMM (host-timer): %8.2lf Gflops\n", gflops);

    // We do not initialize result matrix C to zero after each run
    // So check result only when niterations=1; Higher niterations is used
    // to get average performance number.
    bool pass=FALSE;
    if (niterations == 1) {
        if(C_test == C_gold) {
            printf("PASSED\n");
            pass = TRUE;
	    } else
            printf("FAILED\n");
    } else
        printf("Result not checked - make #iterations=1 to check result!\n");
    printf("----------------------------\n");

    destroy(hCImage);
    destroy(hBImage);
    destroy(hAImage);

    destroy(hCommandList);
    destroy(hContext);
    return pass ? 0 : 1;
}

 
// int main(int argc, char** argv)
// {
//     int m = GEMM_BLOCK;
//     int niterations = 1;
//     if( argc == 3 ) {
//         m = atoi(argv[1]);
//         niterations = atoi(argv[2]);
//     }

//     int success = 0;
//     if (niterations == 1)
//         success |= run_gemm( m, niterations, 1, 4 );
//     else {
//         int success = 0;
//         success |= run_gemm( m, niterations, 1, 1 );
//         success |= run_gemm( m, niterations, 1, 4 );
//         success |= run_gemm( m, niterations, 4, 1 );
//         success |= run_gemm( m, niterations, 2, 2 );
//         success |= run_gemm( m, niterations, 1, 8 );
//         success |= run_gemm( m, niterations, 8, 1 );
//         success |= run_gemm( m, niterations, 2, 4 );
//         success |= run_gemm( m, niterations, 4, 2 );
//     }
//     return success;
// }