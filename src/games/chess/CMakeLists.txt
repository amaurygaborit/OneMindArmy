# ==============================================================================
# CMakeLists.txt for Chess Game Module
# ==============================================================================

file(GLOB_RECURSE CHESS_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

# ------------------------------------------------------------------------------
# Using OBJECT instead of STATIC forces the linker to include all object files
# directly into the .exe, preventing it from stripping out the static
# AutoGameRegister variables!
# ------------------------------------------------------------------------------
add_library(chess_plugin OBJECT ${CHESS_SOURCES})

target_include_directories(chess_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/src/corelib
    ${TRT_INCLUDE_DIR}
)

target_link_libraries(chess_plugin PRIVATE
    yaml-cpp
)

message(STATUS "[Chess] Target created: chess_plugin (OBJECT)")