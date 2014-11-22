cmake_minimum_required (VERSION 2.6)
project(dqn)

option(CPU_ONLY "Use CPU only for Caffe" ON)
option(USE_CUDNN "Use cuDNN for Caffe" OFF)
option(USE_SDL "Use SDL for ALE" ON)

include_directories(/opt/caffe/include)
include_directories(/opt/caffe/build/src)
link_directories(/opt/caffe/build/lib)
if(USE_CUDNN)
  link_directories(../cudnn-6.5-linux-R1)
endif()

add_executable(dqn deeprl.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native")

target_link_libraries(dqn caffe)
target_link_libraries(dqn glog)
target_link_libraries(dqn gflags)
target_link_libraries(dqn protobuf)

if(NOT CPU_ONLY)
  include_directories(/usr/local/cuda-6.5/targets/x86_64-linux/include)
endif()

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()

if(USE_CUDNN)
  target_link_libraries(dqn cudnn)
endif()

