# src/games/chess/CMakeLists.txt

file(GLOB_RECURSE CHESS_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

add_library(chess_objects OBJECT
  ${CHESS_SOURCES}
)

target_include_directories(chess_objects PRIVATE
  ${CMAKE_SOURCE_DIR}/src/corelib
)

target_link_libraries(chess_objects PRIVATE
  yaml-cpp
)
