#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

typedef double Dtype;

clock_t tStart, tEnd;
#define COMPTIME(X)          \
cout << "CompTime of "<< (X) <<": " << (double)(tEnd-tStart)/CLOCKS_PER_SEC<<endl;

int main(int argc, char** argv) { 

//// Initialization
    Blob<Dtype>* const blob = new Blob<Dtype>(20, 30, 40, 50);
    if(blob){
        cout<<"Size of blob:";
        cout<<" N="<<blob->num();
        cout<<" K="<<blob->channels();
        cout<<" H="<<blob->height();
        cout<<" W="<<blob->width();
        cout<<" C="<<blob->count();
        cout<<endl;
    }

    // reshaping the size of blob
    blob->Reshape(50, 40, 30, 20);
    if(blob){
        cout<<"Size of reshaped blob:";
        cout<<" N="<<blob->num();
        cout<<" K="<<blob->channels();
        cout<<" H="<<blob->height();
        cout<<" W="<<blob->width();
        cout<<" C="<<blob->count();
        cout<<endl;
    }

    // Random sampling from uniform distribution
    FillerParameter filler_param;
    filler_param.set_min(-3);
    filler_param.set_max(3);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob);

//// sum of squares
    // access data on the host
    Dtype expected_sumsq = 0;
    const Dtype* data = blob->cpu_data();
    for (int i = 0; i < blob->count(); ++i) {
        expected_sumsq += data[i] * data[i];
    }
    cout<<endl;
    cout<<"expected sumsq of blob: "<<expected_sumsq<<endl;
    tStart = clock();
    cout<<"sumsq of blob on cpu: "<<blob->sumsq_data()<<endl;
    tEnd = clock();
    COMPTIME("sumsq of blob on cpu");

    // Do an access on the current device,
    // so that the sumsq computation is done on that device.
    tStart = clock();
    blob->gpu_data(); // memcopy host to device (to_gpu() in syncedmem.cpp)
    tEnd = clock();
    COMPTIME("cpu->gpu time");

    tStart = clock();
    cout<<"sumsq of blob on gpu: "<<blob->sumsq_data()<<endl;
    tEnd = clock();
    COMPTIME("sumsq on gpu time");

//// Test of syncmem
    cout<<endl;
    tStart = clock();
    blob->gpu_data();   // no data copy since both have up-to-date contents.
    tEnd = clock();
    COMPTIME("cpu->gpu time");

    // gpu data manipulation
    const Dtype kDataScaleFactor = 2;
    blob->scale_data(kDataScaleFactor); // change data on gpu

    tStart = clock();
    blob->cpu_data();   // memcopy device to host (to_cpu() in syncedmem.cpp)
    tEnd = clock();
    COMPTIME("gpu->cpu time");

    tStart = clock();
    cout<<"sumsq of blob on gpu: "<<blob->sumsq_data()<<endl;   // this is done on gpu
    tEnd = clock();
    COMPTIME("sumsq on gpu time");

    delete blob;
    return 0;
}


