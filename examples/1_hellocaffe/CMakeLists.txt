cmake_minimum_required(VERSION 2.8.8)

project (caffetest)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(test_blob test_blob.cpp)
target_link_libraries(test_blob ${Caffe_LIBRARIES})

add_executable(test_mnist test_mnist.cpp)
target_link_libraries(test_mnist ${Caffe_LIBRARIES})

add_executable(test_math test_math.cpp)
target_link_libraries(test_math ${Caffe_LIBRARIES})
