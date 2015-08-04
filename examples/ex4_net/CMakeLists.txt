cmake_minimum_required(VERSION 2.8.8)

project (test_net)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(bindb bindb.cpp)
target_link_libraries(bindb ${Caffe_LIBRARIES})

add_executable(logreg_train logreg_train.cpp)
target_link_libraries(logreg_train ${Caffe_LIBRARIES})

add_executable(logreg_test logreg_test.cpp)
target_link_libraries(logreg_test ${Caffe_LIBRARIES})
