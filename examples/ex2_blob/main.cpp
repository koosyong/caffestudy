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
    
    



    //    // Assuming that data are on the CPU initially, and we have a blob.
    //    Blob<Dtype> blob;
    //    const Dtype* foo;
    //    Dtype* bar;
    //    foo = blob.gpu_data(); // data copied cpu->gpu.
    //    foo = blob.cpu_data(); // no data copied since both have up-to-date contents.
    //    bar = blob.mutable_gpu_data(); // no data copied.
    //    // ... some operations ...
    //    bar = blob.mutable_gpu_data(); // no data copied when we are still on GPU.
    //    foo = blob.cpu_data(); // data copied gpu->cpu, since the gpu side has modified the data
    //    foo = blob.gpu_data(); // no data copied since both have up-to-date contents
    //    bar = blob.mutable_cpu_data(); // still no data copied.
    //    bar = blob.mutable_gpu_data(); // data copied cpu->gpu.
    //    bar = blob.mutable_cpu_data(); // data copied gpu->cpu.


    /*
//    Caffe::set_phase(Caffe::TEST);
    Caffe::set_mode(Caffe::CPU);

    string img_file = argc > 1 ? argv[1] : "/home/koosy/utils/caffe-master/examples/images/cat.jpg";

    string net_src = argc > 2 ? argv[2] : "/home/koosy/utils/caffe-master/examples/mnist/lenet.prototxt";
    Net<float> caffe_test_net(net_src, TRAIN);  //get the net

    string traied_net = argc > 3 ? argv[3] : "/home/koosy/utils/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    caffe_test_net.CopyTrainedLayersFrom(traied_net);

    cout << "reading file" << endl;
    Datum datum;
    if (!ReadImageToDatum(img_file, 99, 28, 28, &datum)) {
        LOG(ERROR) << "Error during file reading";
    }
    std::vector<Datum> images;
    images.push_back(datum);

    float loss = 0.0;

    cout << "adding images" << endl;
    boost::dynamic_pointer_cast< caffe::MemoryDataLayer<float> >(caffe_test_net.layers()[0])->AddDatumVector(images);
    cout << "running net" << endl;
    std::vector<Blob<float>*> result = caffe_test_net.ForwardPrefilled(&loss);

    cout << "got results" << endl;
    LOG(INFO)<< "Output result size: "<< result.size();

    int r = 1; // here in my case r=0 is for input label data, r=1 for prediction result (actually argmax layer)

    const float* argmaxs = result[r]->cpu_data();
    for (int i = 0; i < result[r]->num(); ++i) {
        LOG(INFO)<< " Image: "<< i << " class:" << argmaxs[i]; }
*/

    return 0;
}


