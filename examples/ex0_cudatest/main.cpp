#include <iostream>
#include "axpygpu.h"

using namespace std;

int main (int argc, char** argv)
{
    int n_block = 10;
    int n_thread = 5;
    int a = 2.0;
    AXPYGPU axpy_gpu(n_block, n_thread, a);

    int n = n_block*n_thread;
    float *x, *y, *z;
    x = new float[n];
    y = new float[n];
    z = new float[n];

    for(int i=0;i<n;i++){
        x[i] = (float) rand() / RAND_MAX;
        y[i] = (float) rand() / RAND_MAX;
    }

    // z = a*x+y
    axpy_gpu.compute(x, y, z);

    for(int i=0;i<n;i++)
	    cout<<x[i]<<" "<<y[i]<<" "<<z[i]<<endl;

    delete[] x;
    delete[] y;
    delete[] z;

    return 0;
}

