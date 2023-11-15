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

set(EXAMPLES_DIR ${CMAKE_CURRENT_LIST_DIR} )
set(EXAMPLES_OCL bgemm vadd_hello_ocl)
set(EXAMPLES_L0 vadd_hello_l0)


foreach(EXAMPLE ${EXAMPLES_OCL})
    if (NOT IS_DIRECTORY ${EXAMPLES_DIR}/${EXAMPLE})
        message(WARNING "Wrong example directory name in the list: ${EXAMPLE}")
        list(REMOVE_ITEM EXAMPLES_OCL ${EXAMPLE})
    endif()
endforeach()

foreach(EXAMPLE ${EXAMPLES_L0})
    if (NOT IS_DIRECTORY ${EXAMPLES_DIR}/${EXAMPLE})
        message(WARNING "Wrong example directory name in the list: ${EXAMPLE}")
        list(REMOVE_ITEM EXAMPLES_L0 ${EXAMPLE})
    endif()
endforeach()


message(STATUS "Building for OCL")
foreach(EXAMPLE ${EXAMPLES_OCL})
    message(STATUS ${EXAMPLE})
    file(GLOB ALL_SOURCES "${EXAMPLES_DIR}/${EXAMPLE}/*.cpp")
    set(KERNEL_CODE ${ALL_SOURCES})
    list(FILTER KERNEL_CODE INCLUDE REGEX "_genx.cpp")
    set(HOST_CODE ${ALL_SOURCES})
    list(FILTER HOST_CODE EXCLUDE REGEX "_genx.cpp")
    
    set(KERNEL_TARGET_NAME "${EXAMPLE}.kernel")
    set(HOST_TARGET_NAME "${EXAMPLE}.host")
    add_kernel_target(${KERNEL_TARGET_NAME} ${KERNEL_CODE})
    add_host_target_ocl(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME} ${HOST_CODE})
    add_dependencies(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME})
    
    add_test(NAME ${EXAMPLE} COMMAND $<TARGET_FILE_NAME:${HOST_TARGET_NAME}> WORKING_DIRECTORY $<TARGET_FILE_DIR:${HOST_TARGET_NAME}>)
endforeach()
    

message(STATUS "Building for L0")
foreach(EXAMPLE ${EXAMPLES_L0})
    message(STATUS ${EXAMPLE})
    file(GLOB ALL_SOURCES "${EXAMPLES_DIR}/${EXAMPLE}/*.cpp")
    set(KERNEL_CODE ${ALL_SOURCES})
    list(FILTER KERNEL_CODE INCLUDE REGEX "_genx.cpp")
    set(HOST_CODE ${ALL_SOURCES})
    list(FILTER HOST_CODE EXCLUDE REGEX "_genx.cpp")

    set(KERNEL_TARGET_NAME "${EXAMPLE}.kernel")
    set(HOST_TARGET_NAME "${EXAMPLE}.host")
    add_kernel_target("${KERNEL_TARGET_NAME}" ${KERNEL_CODE})
    add_host_target_l0(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME} ${HOST_CODE})
    add_dependencies(${HOST_TARGET_NAME} ${KERNEL_TARGET_NAME})

    # add_test(NAME ${EXAMPLE} COMMAND $<TARGET_FILE_NAME:${HOST_TARGET_NAME}> WORKING_DIRECTORY $<TARGET_FILE_DIR:${HOST_TARGET_NAME}>)
endforeach()
