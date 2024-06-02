cmake_minimum_required(VERSION 3.24)
project(routes)

set(CMAKE_CXX_STANDARD 20)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)


add_executable(routes main.cpp)

target_link_libraries(routes glpk)
