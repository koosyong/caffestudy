cmake_minimum_required(VERSION 2.8.8)

project (hellocaffe)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(hellocaffe main.cpp)
target_link_libraries(hellocaffe ${Caffe_LIBRARIES})
