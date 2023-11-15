#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <level_zero/ze_api.h>

#define L0_SAFE_CALL(call)                                                     \
  {                                                                            \
    auto status = (call);                                                      \
    if (status != 0) {                                                         \
      fprintf(stderr, "%s:%d: L0 error %d\n", __FILE__, __LINE__,              \
              (int)status);                                                    \
      exit(1);                                                                 \
    }                                                                          \
  }

int main() {
#ifdef SPV
#ifdef OCL
  fprintf(stderr, " L0 OCL SPV \n");
#else
  fprintf(stderr, " L0 CM SPV \n");
#endif
#else /*SPV*/
#ifdef OCL
  fprintf(stderr, " L0 OCL BIN \n");
#else
  fprintf(stderr, " L0 CM BIN \n");
#endif
#endif

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

#ifdef BINNAME
  const char *fn = BINNAME;
#else
  const char *fn = "vadd_hello_l0.kernel.dg2.bin";
#endif

  uint8_t *kernel_bin{nullptr};
  size_t kernel_sz{0};

  {
    FILE *fp = fopen(fn, "rb");
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
#ifdef SPV
      ZE_MODULE_FORMAT_IL_SPIRV, //
#else
      ZE_MODULE_FORMAT_NATIVE, //
#endif
      kernel_sz,  //
      kernel_bin, //
#ifdef OCL
      nullptr,
#else
      "-cmc",
#endif
      nullptr
  };
  ze_module_handle_t hModule;
  L0_SAFE_CALL(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr));

  ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, "vadd"};
  ze_kernel_handle_t hKernel;
  L0_SAFE_CALL(zeKernelCreate(hModule, &kernelDesc, &hKernel));

  constexpr unsigned width = 4096;
  int hBufA[width];
  int hBufB[width];
  int hBufC[width], goldBufC[width];
  size_t bufsize = width * sizeof(int);

  for (size_t i = 0; i < width; ++i) {
    hBufA[i] = i + 1;
    hBufB[i] = 2 * (width + i + 1);
    goldBufC[i] = hBufA[i] + hBufB[i];
    hBufC[i] = -1;
  }

  // allocate buffers
  ze_device_mem_alloc_desc_t deviceMemDesc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
      0,
      0
  };

  void *dBufA = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc,
                                bufsize, bufsize,
                                hDevice, &dBufA));
  void *dBufB = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc,
                                bufsize, bufsize,
                                hDevice, &dBufB));
  void *dBufC = nullptr;
  L0_SAFE_CALL(zeMemAllocDevice(hContext, &deviceMemDesc,
                                bufsize, bufsize,
                                hDevice, &dBufC));

  // copy buffers to device
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufA, hBufA,
                                             bufsize, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufB, hBufB,
                                             bufsize, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBufC, hBufC,
                                             bufsize, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

  // set kernel arguments
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(dBufA), &dBufA));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(dBufB), &dBufB));
  L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(dBufC), &dBufC));

  // set group size
  uint32_t group_size = 8;
  L0_SAFE_CALL(
      zeKernelSetGroupSize(hKernel, /*x*/ group_size, /*y*/ 1, /*z*/ 1));

  // set grid size
  ze_group_count_t groupCount = {
#ifdef OCL
    width / group_size,
#else
    width / group_size / 32,
#endif
    1,
    1};

  // launch
  L0_SAFE_CALL(zeCommandListAppendLaunchKernel(
      hCommandList, hKernel, &groupCount, nullptr, 0, nullptr));

  L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
  // copy result to host
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, hBufC, dBufC,
                                             bufsize, nullptr, 0, nullptr));

  // dispatch & wait
  L0_SAFE_CALL(zeCommandListClose(hCommandList));
  L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1,
                                                 &hCommandList, nullptr));
  L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue,
                                         std::numeric_limits<uint32_t>::max()))

  L0_SAFE_CALL(zeMemFree(hContext, dBufA));
  L0_SAFE_CALL(zeMemFree(hContext, dBufB));
  L0_SAFE_CALL(zeMemFree(hContext, dBufC));

  bool fail = false;
  for (size_t i = 0; i < width; ++i) {
    fprintf(stderr, "i: %d  gold= %d comp=%d  %s\n", int(i), goldBufC[i],
            hBufC[i], (hBufC[i] != goldBufC[i] ? "FAIL" : ""));
    fail |= hBufC[i] != goldBufC[i];
  }

  fprintf(stderr, fail ? "FAIL\n" : "OK\n");
  return fail;
}

