cmake_minimum_required(VERSION 2.8.8)

project (ex_blob)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(ex_blob ex_blob.cpp)
target_link_libraries(ex_blob ${Caffe_LIBRARIES})

add_executable(ex_math ex_math.cpp)
target_link_libraries(ex_math ${Caffe_LIBRARIES})
