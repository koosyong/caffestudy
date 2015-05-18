#ifndef AXPYGPU_H
#define AXPYGPU_H

#include <cuda.h>
#include <cuda_runtime.h>

class AXPYGPU
{
public:
    AXPYGPU() {}
    AXPYGPU(int n_block_, int n_thread_, float a_);
    ~AXPYGPU() {}

    int n_block, n_thread, n;
    float a;
    float *x;
    float *y;

    void compute(float* x_, float* y_, float* z_);
};

#endif // AXPYGPU_H
