#include "l0_rt_helpers.h"
#include "share.h"
#include "Matrix.h"

#include "l0_launch.h"


int _calc_nchw_gemm_fp16(MType *mA, MType *mB, MType *matrixC, int M, int K, int N, 
                int threadX, int threadY, int groupX, int groupY,
                const char* bin_file =  "bgemm_dpas_genx.bin", 
                const char* fn_name = "bgemm_dpas", int iter_num = 1000){

  uint threadNum = threadX * threadY;
  uint TotalthreadNum = threadNum * groupX;
 
  // recompute the group height based on repetition that will be done inside the
  // kernel.
  TotalthreadNum = threadNum * groupX * groupY;

  constexpr int GRIDDIM = 2;
  size_t localsize[GRIDDIM] = {(size_t)threadX, (size_t)threadY};
  size_t globalsize[GRIDDIM] = {(size_t)groupX * localsize[0],
                                (size_t)groupY * localsize[1]};
  
  fprintf(stderr, "localsize= %d %d\n", (int)localsize[0], (int)localsize[1]);
  fprintf(stderr, "globalsize= %d %d\n", (int)globalsize[0], (int)globalsize[1]);
  fprintf(stderr, "thread_space= %d %d\n", threadX, threadY);
  fprintf(stderr, "group_space= %d %d\n", groupX, groupY);
  fprintf(stderr, "M= %d, K= %d, N= %d \n", M, K, N);


  
  L0_SAFE_CALL(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    // Find a driver instance with a GPU device
  auto [hDriver, hDevice, hContext] = findDriverAndDevice();
  auto hCommandList = createImmCommandList(hContext, hDevice);

  auto hKernel = createKernel(hContext, hDevice, bin_file, fn_name);
  // set group size
  L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ threadX, /*y*/ threadY, /*z*/ 1));

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
  // L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(int), &M));
  // L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &N));
  // L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &K));
  // L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(int), &repeat_count));



  fprintf(stderr, "Execute kernel and measure.. \n\n");
  // setup grid size and get a hEvent
  ze_event_handle_t hEvent = createEvent(hContext, hDevice);
  ze_group_count_t groupCount = {groupX, groupY, 1};
  double thost = 0.0f;
  unsigned long long kernel_ns = 0;

  // launch & measure
  for (int i=0; i<iter_num; i++){
    double host_start = getTimeStamp();
    
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &groupCount, hEvent, 0, nullptr));
    L0_SAFE_CALL(zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max()));
    
    double host_end = getTimeStamp();
    thost += (host_end - host_start);
    ze_kernel_timestamp_result_t timestamp;
    zeEventQueryKernelTimestamp(hEvent, &timestamp);
    kernel_ns += (timestamp.context.kernelEnd - timestamp.context.kernelStart);

    L0_SAFE_CALL(zeEventHostReset(hEvent)); // reset event

  }
    thost = thost * 1000.0f / iter_num;
    double tkern = kernel_ns / 1000000.0f / iter_num;

    printf("Iter_num= %d \n", iter_num);
    printf("%-18s%.6lf msec\n","kern time(avg):", tkern);
    printf("%-18s%.6lf msec\n","host time(avg):", thost);

    double gflops;
    gflops = ((2000.0f*M*N*K) / (1.0f*1024*1024*1024)) / tkern;
    printf("DG2 TEST (kern-timer): %8.2lf Gflops\n",  gflops);
    gflops = ((2000.0f*M*N*K) / (1.0f*1024*1024*1024)) / thost;
    printf("DG2 TEST (host-timer): %8.2lf Gflops\n", gflops);

  // copy result to host & wait
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, matrixC, dBufC, bufsize_C, hEvent, 0, nullptr));
  L0_SAFE_CALL(zeEventHostSynchronize(hEvent, std::numeric_limits<uint32_t>::max()));


  L0_SAFE_CALL(zeMemFree(hContext, dBufA));
  L0_SAFE_CALL(zeMemFree(hContext, dBufB));
  L0_SAFE_CALL(zeMemFree(hContext, dBufC));

  L0_SAFE_CALL(zeCommandListReset(hCommandList)); // reset Command list
  L0_SAFE_CALL(zeCommandListDestroy(hCommandList));
  L0_SAFE_CALL(zeContextDestroy(hContext));
  return 0;
}


float float16_to_float32(uint16_t h) {
    uint16_t sign = (uint16_t)(h >> 15); 
    uint16_t exponent = (uint16_t)(h >> 10) & 0x1F;
    uint16_t fraction = (uint16_t)(h & 0x3FF);
    
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();

    if (exponent == 0) {
        if (fraction == 0)
            // Sign is irrelevant for zero.
            return 0;
        else {
            // Denormal (subnormal) number.
            return (sign == 0 ? 1 : -1) * ((float)fraction / 1024) * ((float)1 / (1 << 14));
        }
    } else if (exponent == 31) {
        if (fraction == 0)
            return (sign == 0 ? 1 : -1) * inf;
        else
            return nan;
    } else {
        // Normalized number.
        return (sign == 0 ? 1 : -1) * ((1 + ((float)fraction / 1024)) * (1 << (exponent - 15)));
    }

    return 0; // Shouldn't be reachable.
}


std::vector<MType> run_gemm_nchw_fp16(const char* bin_file , const char* spirv_file, const char* fn_name,
                              const py::args& args, const py::kwargs& kwargs){
// Parse the input
  std::vector<int> v_argc;
  std::vector<MType *> v_argv;
  if (kwargs) {
    for(auto item: kwargs){
        // std::cout << "input key: "<< item.first <<  " , type: "\
        // << py::type::of(item.second).str() << std::endl; // <class 'numpy.ndarray'>
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
  int threadX=v_argc[3], threadY=v_argc[4];
  int groupX=v_argc[5], groupY=v_argc[6];
  int iter_num = v_argc[7];

  MType *matrixA = v_argv[0];
  MType *matrixB = v_argv[1];

  MType *matrixC = (MType *)malloc(M * N * sizeof(MType));
  if (matrixC == 0) {
    free(matrixA);
    free(matrixB);
    printf("Memory C allocation error");
  }
  std::memset(matrixC, 0, M * N * sizeof(MType));

 
  _calc_nchw_gemm_fp16(matrixA, matrixB, matrixC, M, K, N, 
              threadX,  threadY, groupX, groupY,
              bin_file , fn_name, iter_num);


  std::vector<MType> result(M * N);
  // for(int i=0; i<result.size(); i++){
  //   printf("%f \n", float16_to_float32(matrixC[i]));
  // }
  memcpy(&result[0], matrixC, M*N*sizeof(MType));
  free(matrixC);

  return result;
  
}

