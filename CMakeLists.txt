cmake_minimum_required(VERSION 3.10)

project(cmemu_examples)
enable_testing()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(NOT DEFINED CSDK_WORKSPACE)
    message(WARNING "Cmake variable CSDK_WORKSPACE is not defined. Will try environment variable")
    set(CSDK_WORKSPACE $ENV{CSDK_WORKSPACE})
endif()

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

# Apply this option in order to ld add rpath instead of runpath
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

# Set default install path to current directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}" CACHE PATH "Path to install example binaries" FORCE)
endif()

include(functions.cmake)
message(STATUS "Generating makefiles")


find_library(LIB_OPENCL NAMES Intel_OpenCL_ICD64 PATHS ${CSDK_IGC}/runtime/opencl/lib)
find_library(LIB_LEVEL0 NAMES ze_loader PATHS ${CSDK_IGC}/runtime/level_zero/lib)
set(HW_PLATFORM dg2)
string(TOUPPER ${HW_PLATFORM} HW_PLATFORM_UPPER)
set(INSTALL_DIR ${CMAKE_BINARY_DIR}/bin)
set(DEBUG_DIR ${CMAKE_BINARY_DIR}/Debug)

# --------------------------------------------- Kernel Compile
set(DEVICE "DG2")
set(CPU "ADL")
set(APP "Sgemm")
set(KERNEL "Sgemm_genx")
link_directories($ENV{CSDK_IGC}/usr/lib/x86_64-linux-gnu)

set(CM_KERNEL_DIR ${CMAKE_SOURCE_DIR}/cm_kernels)
set(RT_L0_PATH  ${CMAKE_SOURCE_DIR}/rt_l0)
file(GLOB KERNEL_SRC  ${CM_KERNEL_DIR}/*.cpp)
file(GLOB RTL0_SRC  ${RT_L0_PATH}/*.cpp)

message(${RTL0_SRC})

add_custom_command(
  OUTPUT ${KERNEL}.bin
  # COMMAND ${CMC} -emit-spirv -march=${HW_PLATFORM_UPPER} ${SOURCES} -o ${spirv}
  # COMMAND ${CMC} -march=${CPU} ${KERNEL}.spv -o ${KERNEL}.bin
  COMMAND ${CMC} -march=${CPU} -I ${RT_L0_PATH} ${KERNEL_SRC} -o ${KERNEL}.bin  
)

# file(COPY ${CMAKE_BINARY_DIR}/${KERNEL}.bin DESTINATION ${DEBUG_DIR})
install(FILES ${CMAKE_BINARY_DIR}/${KERNEL}.bin DESTINATION ${INSTALL_DIR})
install(FILES ${CMAKE_BINARY_DIR}/${KERNEL}.bin DESTINATION ${DEBUG_DIR})

add_executable(${APP} ${RTL0_SRC} ${KERNEL}.bin )

target_compile_definitions(
  ${APP} PRIVATE 
  BINNAME="${KERNEL}.bin" 
  BINDIR="None" L0 CM_COMPILE_OFFLINE
)

target_include_directories(
  ${APP} PRIVATE 
  $ENV{CSDK_IGC}/runtime/level_zero/include 
  $ENV{CSDK_IGC}/usr/include 
  $ENV{CSDK_IGC}/compiler/include
)

target_link_libraries(${APP} ${CMAKE_DL_LIBS} ${LIB_LEVEL0})

set_property(TARGET ${APP} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

# --------------------------------------------- Pybind11 support
# find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
# set(PYBIND11 ${CMAKE_SOURCE_DIR}/3rdparty/pybind11)
# set(PYBIND11_JSON ${CMAKE_SOURCE_DIR}/3rdparty/pybind11_json)
# set(NLOHMANN_JSON ${CMAKE_SOURCE_DIR}/3rdparty/json)
# set(JSON_BuildTests OFF CACHE INTERNAL "")

# include_directories(SYSTEM ${PYBIND11_JSON}/include)
# add_subdirectory(${NLOHMANN_JSON})
# add_subdirectory(${PYBIND11})

# add_definitions(-Dssize_t=SSIZE_T)
# set(CMAKE_REQUIRED_DEFINITIONS -Dssize_t=SSIZE_T)

# set(ZBENCH_SRC zbench.cpp)
# pybind11_add_module(zbench ${ZBENCH_SRC})
# target_link_libraries(zbench PRIVATE nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
# ---------------------------------------------
# set(EXAMPLES_DIR ${CMAKE_CURRENT_LIST_DIR} )
# set(EXAMPLES_OCL bgemm vadd_hello_ocl)
# set(EXAMPLES_L0 vadd_hello_l0)


# foreach(EXAMPLE ${EXAMPLES_OCL})
#     if (NOT IS_DIRECTORY ${EXAMPLES_DIR}/${EXAMPLE})
#         message(WARNING "Wrong example directory name in the list: ${EXAMPLE}")
#         list(REMOVE_ITEM EXAMPLES_OCL ${EXAMPLE})
#     endif()
# endforeach()

# foreach(EXAMPLE ${EXAMPLES_L0})
#     if (NOT IS_DIRECTORY ${EXAMPLES_DIR}/${EXAMPLE})
#         message(WARNING "Wrong example directory name in the list: ${EXAMPLE}")
#         list(REMOVE_ITEM EXAMPLES_L0 ${EXAMPLE})
#     endif()
# endforeach()


# message(STATUS "Building for OCL")
# foreach(EXAMPLE ${EXAMPLES_OCL})
#     message(STATUS ${EXAMPLE})
#     file(GLOB ALL_SOURCES "${EXAMPLES_DIR}/${EXAMPLE}/*.cpp")
#     set(KERNEL_CODE ${ALL_SOURCES})
#     list(FILTER KERNEL_CODE INCLUDE REGEX "_genx.cpp")
#     set(HOST_CODE ${ALL_SOURCES})
#     list(FILTER HOST_CODE EXCLUDE REGEX "_genx.cpp")
    
#     set(KERNEL_TARGET_NAME "${EXAMPLE}.kernel")
#     set(HOST_TARGET_NAME "${EXAMPLE}.host")
#     add_kernel_target(${KERNEL_TARGET_NAME} ${KERNEL_CODE})
#     add_host_target_ocl(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME} ${HOST_CODE})
#     add_dependencies(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME})
    
#     add_test(NAME ${EXAMPLE} COMMAND $<TARGET_FILE_NAME:${HOST_TARGET_NAME}> WORKING_DIRECTORY $<TARGET_FILE_DIR:${HOST_TARGET_NAME}>)
# endforeach()
    

# message(STATUS "Building for L0")
# foreach(EXAMPLE ${EXAMPLES_L0})
#     message(STATUS ${EXAMPLE})
#     file(GLOB ALL_SOURCES "${EXAMPLES_DIR}/${EXAMPLE}/*.cpp")
#     set(KERNEL_CODE ${ALL_SOURCES})
#     list(FILTER KERNEL_CODE INCLUDE REGEX "_genx.cpp")
#     set(HOST_CODE ${ALL_SOURCES})
#     list(FILTER HOST_CODE EXCLUDE REGEX "_genx.cpp")

#     set(KERNEL_TARGET_NAME "${EXAMPLE}.kernel")
#     set(HOST_TARGET_NAME "${EXAMPLE}.host")
#     add_kernel_target("${KERNEL_TARGET_NAME}" ${KERNEL_CODE})
#     add_host_target_l0(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME} ${HOST_CODE})
#     add_dependencies(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME})

#     # add_test(NAME ${EXAMPLE} COMMAND $<TARGET_FILE_NAME:${HOST_TARGET_NAME}> WORKING_DIRECTORY $<TARGET_FILE_DIR:${HOST_TARGET_NAME}>)
# endforeach()
