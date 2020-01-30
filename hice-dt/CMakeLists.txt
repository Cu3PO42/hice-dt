cmake_minimum_required(VERSION 3.10)

project(hice-dt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)



file(GLOB sources src/*cpp)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/test_horn_solver.cpp)

add_executable(hice-dt ${sources})

target_include_directories(hice-dt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
