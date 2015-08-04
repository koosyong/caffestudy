#include "axpygpu.h"

__global__ void axpy(float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    y[i] = a*x[i] + y[i];
}

AXPYGPU::AXPYGPU(int n_block_, int n_thread_, float a_)
    :n_block(n_block_), n_thread(n_thread_), a(a_)
{
    n = n_block * n_thread;
    cudaMalloc((void **) &x, n*sizeof(float));
    cudaMalloc((void **) &y, n*sizeof(float));
}

void AXPYGPU::compute(float* x_, float* y_, float* z_)
{
    cudaMemcpy(x, x_, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_, n*sizeof(float), cudaMemcpyHostToDevice);

    axpy<<<n_block,n_thread>>>(a,x,y);

    cudaMemcpy(z_, y, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(y);
}

