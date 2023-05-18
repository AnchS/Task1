cmake_minimum_required(VERSION 3.25)
project(SC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

add_executable(SC main.cpp)
