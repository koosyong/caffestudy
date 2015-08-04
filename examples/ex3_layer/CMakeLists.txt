cmake_minimum_required(VERSION 2.8.8)

project (test_layer)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

set(headers
)

set(sources
    main.cpp
)

add_executable(test_layer ${headers} ${sources})
target_link_libraries(test_layer ${Caffe_LIBRARIES})
