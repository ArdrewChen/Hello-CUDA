project(TEST1 VERSION 1.0 LANGUAGES CXX CUDA)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/code SRC)

#target_include_directories(${EXE_TRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${BIN_PATH})
add_executable(${TEST1} ${SRC})