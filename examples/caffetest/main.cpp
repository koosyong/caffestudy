#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

struct TypeParam {
    typedef double Dtype;
    static const Caffe::Brew device = Caffe::CPU;
};

typedef TypeParam::Dtype Dtype;

int main(int argc, char** argv) {

    cout<<"start"<<endl;
    Blob<Dtype>* const blob = new Blob<Dtype>();
    Blob<Dtype>* const blob_preshaped = new Blob<Dtype>(2, 3, 4, 5);

    // check the size of the blob_preshaped
    if(blob_preshaped){
        cout<<"Size of blob_preshaped:";
        cout<<" N="<<blob_preshaped->num();
        cout<<" K="<<blob_preshaped->channels();
        cout<<" H="<<blob_preshaped->height();
        cout<<" W="<<blob_preshaped->width();
        cout<<" C="<<blob_preshaped->count();
        cout<<endl;
    }

    // reshaping the size of blob
    blob->Reshape(2, 3, 4, 5);
    if(blob){
        cout<<"Size of blob:";
        cout<<" N="<<blob->num();
        cout<<" K="<<blob->channels();
        cout<<" H="<<blob->height();
        cout<<" W="<<blob->width();
        cout<<" C="<<blob->count();
        cout<<endl;
    }

    // Math test
    Dtype epsilon = 1e-6;

    // Uninitialized Blob should have sum of squares == 0.
    cout<<endl;
    cout<<"uninitialized blob"<<endl;
    cout<<"sumsq of blob: "<<blob->sumsq_data()<<endl;
    cout<<"sumsq_diff of blob: "<<blob->sumsq_diff()<<endl;

    // Random sampling from uniform distribution
    cout<<endl;
    cout<<"randomly initialized blob"<<endl;
    FillerParameter filler_param;
    filler_param.set_min(-3);
    filler_param.set_max(3);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob);

//// sum of squares
    // access  on the host
    Dtype expected_sumsq = 0;
    const Dtype* data = blob->cpu_data();
    for (int i = 0; i < blob->count(); ++i) {
        expected_sumsq += data[i] * data[i];
    }
    cout<<"manual sumsq of blob: "<<expected_sumsq<<endl;
    cout<<"auto sumsq of blob: "<<blob->sumsq_data()<<endl;

    // Do a mutable access on the current device,
    // so that the sumsq computation is done on that device.
    // (Otherwise, this would only check the CPU sumsq implementation.)
    switch (TypeParam::device) {
    case Caffe::CPU:
        blob->mutable_cpu_data();
        break;
    case Caffe::GPU:
        blob->mutable_gpu_data();
        break;
    default:
        LOG(FATAL) << "Unknown device: " << TypeParam::device;
    }
    cout<<endl;
    cout<<"expected_sumsq: "<<expected_sumsq<<endl;
    cout<<"sumsq_data: "<<blob->sumsq_data()<<endl;
    cout<<"sumsq_diff: "<<blob->sumsq_diff()<<endl;

    // Check sumsq_diff too.
    const Dtype kDiffScaleFactor = 7;
    switch (TypeParam::device) {
    case Caffe::CPU:
        blob->mutable_cpu_diff();
        break;
    case Caffe::GPU:
        blob->mutable_gpu_diff();
        break;
    default:
        LOG(FATAL) << "Unknown device: " << TypeParam::device;
    }
    caffe_gpu_scale(blob->count(), kDiffScaleFactor, data,
                    blob->mutable_cpu_diff());

    cout<<endl;
    cout<<"expected_sumsq: "<<expected_sumsq<<endl;
    cout<<"sumsq_data: "<<blob->sumsq_data()<<endl;
    cout<<"sumsq_diff: "<<blob->sumsq_diff()<<endl;

    const Dtype expected_sumsq_diff =
            expected_sumsq * kDiffScaleFactor * kDiffScaleFactor;
    cout<<endl;
    cout<<"expected_sumsq_diff: "<<expected_sumsq_diff<<endl;
    cout<<"sumsq_diff: "<<blob->sumsq_diff()<<endl;

//// Test ASUM
    Dtype expected_asum = 0;
    for (int i = 0; i < blob->count(); ++i) {
      expected_asum += std::fabs(data[i]);
    }
    cout<<endl;
    cout<<"expected_asum: "<<expected_asum<<endl;
    cout<<"asum_data: "<<blob->asum_data()<<endl;

    const Dtype expected_diff_asum = expected_asum * kDiffScaleFactor;
    cout<<"expected_diff_asum: "<<expected_diff_asum<<endl;
    cout<<"asum_diff: "<<blob->asum_diff()<<endl;
    
//// Test scale data
    const Dtype asum_before_scale = blob->asum_data();
    const Dtype kDataScaleFactor = 3;
    blob->scale_data(kDataScaleFactor);
    cout<<endl;
    cout<<"expected_asum_scale: "<<asum_before_scale * kDataScaleFactor<<endl;
    cout<<"asum_scale: "<<blob->asum_data()<<endl;

    const Dtype kDataToDiffScaleFactor = 7;
    delete blob;
    delete blob_preshaped;

    return 0;
}


