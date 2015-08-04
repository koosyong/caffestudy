cmake_minimum_required(VERSION 2.4.6)

project (cudatest)

## CUDA
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

set (SOURCES 
	main.cpp 
        axpygpu.cu
)

set  (HEADERS 
        axpygpu.h
)

CUDA_ADD_LIBRARY (axpygpulib
        axpygpu.cu
        axpygpu.h
)

CUDA_ADD_EXECUTABLE(cudatest axpygpu.cu)
ADD_EXECUTABLE  (cudatest ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES (cudatest)
