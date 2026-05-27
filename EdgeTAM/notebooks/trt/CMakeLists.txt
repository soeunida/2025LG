cmake_minimum_required(VERSION 3.10)
project(trt_infer)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

include_directories(/usr/include)
link_directories(/usr/lib)

add_executable(trt_infer main.cpp)

target_link_libraries(trt_infer nvinfer cudart)
