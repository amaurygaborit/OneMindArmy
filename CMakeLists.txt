# CMakeLists.txt for OneMindArmy Project

cmake_minimum_required(VERSION 3.20)

# Force Visual Studio generator on Windows (optional safety)
if(WIN32 AND NOT DEFINED CMAKE_GENERATOR)
    set(CMAKE_GENERATOR "Visual Studio 17 2022" CACHE STRING "Generator" FORCE)
    set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "Platform" FORCE)
endif()

# Project declaration with CUDA enabled
project(OneMindArmy LANGUAGES CXX CUDA)
enable_language(CUDA)

# --- Language Standards ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- CUDA Configuration ---
set(CMAKE_CUDA_ARCHITECTURES 75) # GTX 1650 (SM 7.5)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF) # avoid nvcc linking issues
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# --- Build Type ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# --- Find CUDA Toolkit ---
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# --- External Dependencies ---
add_subdirectory(external/yaml-cpp)

# --- Core Library ---
add_subdirectory(src/corelib)

# --- Main Executable ---
add_executable(onemindarmy
    src/corelib/main.cpp
)

# Link libraries
target_link_libraries(onemindarmy PRIVATE
    corelib_static
    yaml-cpp
    CUDA::cudart
    CUDA::curand
)

# --- Compiler Options (MSVC only) ---
if(MSVC)
    # Enable UTF-8 encoding for all builds
    add_compile_options(/utf-8)

    # Apply to main exe
    target_compile_options(onemindarmy PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/RTC1>  # Debug checks
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>  # Optimization
    )

    # Apply to corelib
    target_compile_options(corelib_static PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/RTC1>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
    )

    # Extra flag for CUDA (avoid compiler mismatch issues)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
endif()

# --- Optimization: Link Time Optimization ---
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_property(TARGET onemindarmy PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
    message(STATUS "LTO enabled for Release builds")
else()
    message(STATUS "LTO not supported: ${ipo_error}")
endif()

# --- Debug Information ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

# --- Final Summary ---
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Generator: ${CMAKE_GENERATOR}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Separable Compilation: ${CMAKE_CUDA_SEPARABLE_COMPILATION}")
message(STATUS "======================================")

# ==========================
# === Add Your Game Here ===
# ==========================
add_subdirectory(src/games/chess)
target_link_libraries(onemindarmy PRIVATE chess_objects)