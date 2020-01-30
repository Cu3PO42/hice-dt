cmake_minimum_required(VERSION 3.10)

project(hice-dt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)



file(GLOB sources src/*cpp)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/test_horn_solver.cpp)

set(libsources ${sources})
list(REMOVE_ITEM libsources ${CMAKE_CURRENT_SOURCE_DIR}/src/hice-dt.cpp)

add_executable(hice-dt-bin ${sources})
add_library(hice-dt STATIC ${libsources})

target_include_directories(hice-dt-bin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(hice-dt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(hice-dt-bin PROPERTIES OUTPUT_NAME hice-dt)
