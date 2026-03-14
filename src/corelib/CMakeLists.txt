# ==============================================================================
# CMakeLists.txt for CoreLib Module
# ==============================================================================

file(GLOB_RECURSE CORE_CPP_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)
# Exclude main.cpp to prevent multiple definition errors in the executable
list(FILTER CORE_CPP_SOURCES EXCLUDE REGEX "main\\.cpp$")

file(GLOB_RECURSE CORE_CUDA_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)

# ------------------------------------------------------------------------------
# THE FIX: Dummy CPP file generator
# MSVC will refuse to output a .lib file if the library only contains .hpp files.
# We inject a dummy .cpp file to guarantee the archiver runs.
# ------------------------------------------------------------------------------
set(DUMMY_FILE ${CMAKE_CURRENT_BINARY_DIR}/corelib_dummy.cpp)
file(WRITE ${DUMMY_FILE} "// Auto-generated to force MSVC to emit a .lib for template-heavy targets\n")

set(ALL_CORE_SOURCES ${CORE_CPP_SOURCES} ${CORE_CUDA_SOURCES} ${DUMMY_FILE})

add_library(corelib_static STATIC ${ALL_CORE_SOURCES})
    
target_include_directories(corelib_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIR}
)

if(CORE_CUDA_SOURCES)
    set_source_files_properties(${CORE_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
    
    set_target_properties(corelib_static PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "75"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_compile_options(corelib_static PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --use_fast_math
        >
    )
endif()

target_link_libraries(corelib_static PUBLIC
    yaml-cpp
    CUDA::cudart
    CUDA::curand
    ${TRT_LIBRARY_INFER}
)

message(STATUS "[CoreLib] Target created: corelib_static")