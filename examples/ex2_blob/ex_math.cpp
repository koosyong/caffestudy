#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

typedef double Dtype;

int main(int argc, char** argv) {
    Blob<Dtype>* blob_in = new Blob<Dtype>(20, 30, 40, 50);
    Blob<Dtype>* blob_out = new Blob<Dtype>(20, 30, 40, 50);
    int n = blob_in->count();

    // random number generation
    caffe_gpu_rng_uniform<Dtype>(n, -3, 3, blob_in->mutable_gpu_data());

    // asum
    Dtype asum;
    caffe_gpu_asum<Dtype>(n, blob_in->gpu_data(), &asum);
    cout<<"asum: "<<asum<<endl;

    // sign, signbit, fabs, scale
    caffe_gpu_sign<Dtype>(n, blob_in->gpu_data(), blob_out->mutable_gpu_data());
    caffe_gpu_sgnbit<Dtype>(n, blob_in->gpu_data(), blob_out->mutable_gpu_data());
    caffe_gpu_abs<Dtype>(n, blob_in->gpu_data(), blob_out->mutable_gpu_data());
    caffe_gpu_scale<Dtype>(n, 10, blob_in->gpu_data(), blob_out->mutable_gpu_data());

    //
    // caffe_gpu_gemm
    // caffe_gpu_gemv
    // caffe_gpu_axpy
    // caffe_gpu_axpby
    // caffe_gpu_add_scalar
    // caffe_gpu_add
    // caffe_gpu_sub
    // caffe_gpu_mul
    // caffe_gpu_div
    // caffe_gpu_exp
    // caffe_gpu_powx
    // caffe_copy

    const Dtype* x = blob_out->cpu_data();
    for (int i = 0; i < n; ++i) cout<<x[i]<<endl;

    delete blob_in;
    delete blob_out;
    return 0;
}


