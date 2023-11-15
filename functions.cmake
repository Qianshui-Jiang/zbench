macro(ADD_KERNEL_TARGET TARGET_NAME SOURCES)
    set(out ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${HW_PLATFORM}.spv ${CMAKE_BINARY_DIR}/${TARGET_NAME}.${HW_PLATFORM}.bin)
    list(GET out 0 spirv)
    list(GET out 1 binary)
    string(REPLACE ${CMAKE_BINARY_DIR}/ "" kernel ${binary})
    add_custom_command(OUTPUT ${out}
        COMMAND ${CMC} -emit-spirv -march=${HW_PLATFORM_UPPER} ${SOURCES} -o ${spirv}
        COMMAND ${CMC} -march=${HW_PLATFORM_UPPER} ${spirv} -o ${binary}
    )
    add_custom_target(${TARGET_NAME} DEPENDS ${out})
    install(FILES ${out} DESTINATION ${INSTALL_DIR})
    install(FILES ${out} DESTINATION ${DEBUG_DIR})
    
endmacro()

macro(ADD_HOST_TARGET_OCL TARGET_NAME KERNEL_TARGET_NAME SOURCES)
    add_executable(${TARGET_NAME} ${SOURCES})

    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS -DKERNEL=\\\"${kernel}\\\")
    target_include_directories(${TARGET_NAME} PUBLIC
        ${CSDK_WORKSPACE}/usr/include
        ${CSDK_WORKSPACE}/runtime/opencl/include
    )

    target_link_libraries(${TARGET_NAME} PRIVATE ${LIB_OPENCL} )
    add_dependencies(${TARGET_NAME} ${KERNEL_TARGET_NAME})
    install(TARGETS ${HOST_TARGET_NAME} DESTINATION ${INSTALL_DIR})

endmacro()

macro(ADD_HOST_TARGET_L0 TARGET_NAME KERNEL_TARGET_NAME SOURCES)
    add_executable(${TARGET_NAME} ${SOURCES})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CSDK_WORKSPACE}/usr/include
        ${CSDK_WORKSPACE}/runtime/level_zero/include
    )

    if(MSVC)
        target_compile_definitions(${TARGET_NAME} PRIVATE
            _CRT_SECURE_NO_WARNINGS
        )
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${LIB_LEVEL0}
    )
    set_property(TARGET ${TARGET_NAME} PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
    install(TARGETS ${HOST_TARGET_NAME} DESTINATION ${INSTALL_DIR})

endmacro()
