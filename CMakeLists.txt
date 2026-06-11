# ==============================================================================
# ONE MIND ARMY — Root CMake Configuration
# ==============================================================================

cmake_minimum_required(VERSION 3.20)

# Forces MSVC x64 by default on Windows to prevent silent fallbacks to 32-bit,
# which immediately breaks CUDA and TensorRT linking.
if(WIN32 AND NOT DEFINED CMAKE_GENERATOR)
    set(CMAKE_GENERATOR "Visual Studio 17 2022" CACHE STRING "Generator" FORCE)
    set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "Platform" FORCE)
endif()

project(OneMindArmy LANGUAGES CXX CUDA)

# --- Global Build Configuration ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# GLOBAL OUTPUT ROUTING
# Forces all static libraries (.lib) and executables (.exe) into shared root 
# output folders. This completely eliminates linker path errors on Windows 
# caused by nested CMake subdirectories.
# ------------------------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# --- CUDA Setup ---
set(CMAKE_CUDA_ARCHITECTURES 75) 
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

find_package(CUDAToolkit REQUIRED)
message(STATUS "[OneMindArmy] CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# --- DEPENDENCY: YAML-CPP ---
# Fetched directly from GitHub during configuration.
include(FetchContent)
message(STATUS "[OneMindArmy] Fetching yaml-cpp from remote repository...")

# Disables testing modules to accelerate CMake generation
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# --- DEPENDENCY: TENSORRT ---
message(STATUS "[OneMindArmy] Resolving TensorRT dependency...")

# TRT_ROOT must be exported as a system environment variable.
set(TRT_ROOT $ENV{TRT_ROOT} CACHE PATH "Path to the TensorRT installation directory")

find_path(TRT_INCLUDE_DIR NAMES NvInfer.h
    PATHS ${TRT_ROOT}/include /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
if(NOT TRT_INCLUDE_DIR)
    find_path(TRT_INCLUDE_DIR NAMES NvInfer.h) 
endif()

find_library(TRT_LIBRARY_INFER 
    NAMES nvinfer nvinfer_10 nvinfer_8 nvinfer_7
    PATHS ${TRT_ROOT}/lib /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH
)
if(NOT TRT_LIBRARY_INFER)
    find_library(TRT_LIBRARY_INFER NAMES nvinfer nvinfer_10 nvinfer_8 nvinfer_7) 
endif()

if(TRT_INCLUDE_DIR AND TRT_LIBRARY_INFER)
    message(STATUS "[OneMindArmy] TensorRT found successfully.")
else()
    message(FATAL_ERROR "[OneMindArmy] Failed to locate TensorRT. Please set TRT_ROOT.")
endif()

# --- Subdirectories & Targets ---
add_subdirectory(src/corelib)
add_subdirectory(src/games/chess)

# Core executable wrapper
add_executable(onemindarmy src/corelib/main.cpp)

target_include_directories(onemindarmy PRIVATE ${TRT_INCLUDE_DIR})

target_link_libraries(onemindarmy PRIVATE
    corelib_static
    chess_plugin
    yaml-cpp
    CUDA::cudart
    CUDA::curand
    ${TRT_LIBRARY_INFER}
)

# --- POST-BUILD SCRIPT ---
# Automatically copies TensorRT runtime DLLs to the output binary directory 
# on Windows to ensure the executable finds its dependencies at launch.
if(WIN32 AND TRT_ROOT)
    file(GLOB TRT_DLLS "${TRT_ROOT}/lib/*.dll" "${TRT_ROOT}/bin/*.dll")
    if(TRT_DLLS)
        add_custom_command(TARGET onemindarmy POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TRT_DLLS}
            "$<TARGET_FILE_DIR:onemindarmy>"
            COMMENT "[OneMindArmy] Injecting TensorRT runtime libraries..."
        )
    endif()
endif()

# --- COMPILER OVERRIDES ---
if(MSVC)
    add_compile_options(/utf-8)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
endif()