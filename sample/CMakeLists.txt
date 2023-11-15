cmake_minimum_required(VERSION 3.10)

project(sample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(MSVC)
  add_compile_options(/std:c++17)
endif()

if(NOT DEFINED ENV{CSDK_IGC})
  message(FATAL_ERROR "CSDK_IGC environment variable is not set - did you run 'setenv.bat' ?")
endif()

# set paths
set(CSDK_IGC $ENV{CSDK_IGC})

# locate all relevant packages
foreach(program CMC)
  string(TOLOWER ${program} binary)
  find_program(${program} ${binary} REQUIRED)
  if(NOT ${program})
    message(FATAL_ERROR "Unable to locate ${binary} executable - did you run 'setenv.bat' ?")
  else()
    message(INFO " using ${binary} from ${${program}}")
  endif()
endforeach(program)
find_library(LIB_OPENCL NAMES Intel_OpenCL_ICD64 PATHS ${CSDK_IGC}/runtime/opencl/lib)
find_library(LIB_LEVEL0 NAMES ze_loader PATHS ${CSDK_IGC}/runtime/level_zero/lib)

# our sources
set(KERNEL ${CMAKE_SOURCE_DIR}/kernel.cpp)
set(HOST_OCL ${CMAKE_SOURCE_DIR}/host_ocl.cpp)
set(HOST_L0  ${CMAKE_SOURCE_DIR}/host_l0.cpp)

# os-specific
if (CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  set(dll ${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(HW_PLATFORM tgllp)
else()
  set(dll)
  set(HW_PLATFORM skl)
endif()
string(TOUPPER ${HW_PLATFORM} HW_PLATFORM_UPPER)

set(INSTALL_DIR ${CMAKE_BINARY_DIR}/bin)



#######
# GPU Kernel
set(out ${CMAKE_BINARY_DIR}/kernel.${HW_PLATFORM}.spv ${CMAKE_BINARY_DIR}/kernel.${HW_PLATFORM})
list(GET out 0 spirv)
list(GET out 1 binary)
string(REPLACE ${CMAKE_BINARY_DIR}/ "" kernel ${binary})
add_custom_command(OUTPUT ${out}
  COMMAND ${CMC} -emit-spirv -march=${HW_PLATFORM_UPPER} ${KERNEL} -o ${spirv}
  COMMAND ${CMC} -march=${HW_PLATFORM_UPPER} ${spirv} -o ${binary}
)
add_custom_target(kernel_gpu DEPENDS ${out})
install(FILES ${out} DESTINATION ${INSTALL_DIR})
  
#######
# HW GPU / OCL
add_executable(vector.${HW_PLATFORM} ${HOST_OCL})
set_target_properties(vector.${HW_PLATFORM} PROPERTIES COMPILE_FLAGS -DKERNEL=\\\"${kernel}\\\")
target_include_directories(vector.${HW_PLATFORM} PUBLIC ${CSDK_IGC}/runtime/opencl/include)
target_link_libraries(vector.${HW_PLATFORM} ${LIB_OPENCL})
add_dependencies(vector.${HW_PLATFORM} kernel_gpu)
install(TARGETS vector.${HW_PLATFORM} DESTINATION ${INSTALL_DIR})


#######
# HW GPU / L0
set(kernel kernel.${HW_PLATFORM}.spv)
add_executable(vector.l0.${HW_PLATFORM} ${HOST_L0})
set_target_properties(vector.l0.${HW_PLATFORM} PROPERTIES COMPILE_FLAGS -DKERNEL=\\\"${kernel}\\\")
target_include_directories(vector.l0.${HW_PLATFORM} PUBLIC ${CSDK_IGC}/runtime/level_zero/include)
target_link_libraries(vector.l0.${HW_PLATFORM} ${LIB_LEVEL0})
add_dependencies(vector.l0.${HW_PLATFORM} kernel_gpu)
install(TARGETS vector.l0.${HW_PLATFORM} DESTINATION ${INSTALL_DIR})


# all targets to build
add_custom_target(build ALL)
add_dependencies(build vector.${HW_PLATFORM} )
add_dependencies(build vector.l0.${HW_PLATFORM} )
