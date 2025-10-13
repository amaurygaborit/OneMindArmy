# src/corelib/CMakeLists.txt

# --- Ensure CUDA is enabled ---
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Source File Discovery ---
file(GLOB_RECURSE CORE_CPP_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)
list(FILTER CORE_CPP_SOURCES EXCLUDE REGEX "main\\.cpp$")

file(GLOB_RECURSE CORE_CUDA_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
)

# Display discovered sources
message(STATUS "CoreLib C++ Sources: ${CORE_CPP_SOURCES}")
message(STATUS "CoreLib CUDA Sources: ${CORE_CUDA_SOURCES}")

# --- Configure CUDA Source Properties ---
if(CORE_CUDA_SOURCES)
    set_source_files_properties(${CORE_CUDA_SOURCES} PROPERTIES
        LANGUAGE CUDA
    )
    
    if(MSVC)
        set_source_files_properties(${CORE_CUDA_SOURCES} PROPERTIES
            VS_TOOL_OVERRIDE "CudaCompile"
        )
    endif()
endif()

# --- Create Single Static Library ---
# Combine all sources into one library
set(ALL_CORE_SOURCES ${CORE_CPP_SOURCES} ${CORE_CUDA_SOURCES})

if(ALL_CORE_SOURCES)
    add_library(corelib_static STATIC ${ALL_CORE_SOURCES})
    
    # Include directories
    target_include_directories(corelib_static PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    # CUDA properties
    if(CORE_CUDA_SOURCES)
        set_target_properties(corelib_static PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_ARCHITECTURES "75"
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        
        # CUDA compile options
        target_compile_options(corelib_static PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --expt-relaxed-constexpr
                --use_fast_math
            >
        )
    endif()
    
    # Link libraries
    target_link_libraries(corelib_static PUBLIC
        yaml-cpp
        CUDA::cudart
        CUDA::curand
    )
    
    message(STATUS "Created corelib_static library")
else()
    message(FATAL_ERROR "No source files found in CoreLib")
endif()