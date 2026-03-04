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


# ==============================================================================
# --- RECHERCHE DE TENSORRT ---
# ==============================================================================
set(DEFAULT_TRT_DIR "${CMAKE_SOURCE_DIR}/external/TensorRT")
set(TENSORRT_DIR ${DEFAULT_TRT_DIR} CACHE PATH "Path to TensorRT installation")

# 1. On force la recherche stricte du Header
find_path(TENSORRT_INCLUDE_DIR 
    NAMES NvInfer.h 
    PATHS "${TENSORRT_DIR}/include" 
    NO_DEFAULT_PATH
)

# 2. Recherche dynamique par motif (Wildcard) pour le fichier .lib ou .so
file(GLOB TRT_LIB_CANDIDATES 
    "${TENSORRT_DIR}/lib/nvinfer*.lib"   # Pour Windows (nvinfer.lib, nvinfer_10.lib, etc.)
    "${TENSORRT_DIR}/lib/libnvinfer*.so" # Pour Linux (si jamais tu portes le projet)
)

# Si CMake a trouvé un ou plusieurs fichiers correspondant au motif
if(TRT_LIB_CANDIDATES)
    list(GET TRT_LIB_CANDIDATES 0 TENSORRT_LIBRARY) # On prend le premier de la liste
else()
    set(TENSORRT_LIBRARY "TENSORRT_LIBRARY-NOTFOUND" CACHE FILEPATH "TensorRT Library" FORCE)
endif()

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT introuvable dans ${TENSORRT_DIR} ! Extrayez l'archive NVIDIA ici ou definissez -DTENSORRT_DIR.")
else()
    message(STATUS "TensorRT trouve :")
    message(STATUS "  Includes : ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  Library  : ${TENSORRT_LIBRARY}")
endif()

# --- External Dependencies ---
add_subdirectory(external/yaml-cpp)

# --- Core Library ---
add_subdirectory(src/corelib)

set_source_files_properties(src/corelib/main.cpp PROPERTIES LANGUAGE CUDA)

# --- Main Executable ---
add_executable(onemindarmy
    src/corelib/main.cpp
)

# On force l'exécutable à regarder dans le dossier include de TensorRT
target_include_directories(onemindarmy PRIVATE 
    ${TENSORRT_INCLUDE_DIR}
)

# Link libraries (onemindarmy hérite des dépendances de corelib_static)
target_link_libraries(onemindarmy PRIVATE
    corelib_static
    yaml-cpp
    CUDA::cudart
    CUDA::curand
    ${TENSORRT_LIBRARY} # Liaison avec TensorRT pour l'exécutable final
)

# --- Compiler Options (MSVC only) ---
if(MSVC)
    add_compile_options(/utf-8)

    target_compile_options(onemindarmy PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/RTC1> 
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2> 
    )

    target_compile_options(corelib_static PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/RTC1>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
    )

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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

# ==========================
# === Add Your Game Here ===
# ==========================
add_subdirectory(src/games/chess)
target_link_libraries(onemindarmy PRIVATE chess_objects)