#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "google/protobuf/text_format.h"

using namespace caffe;
using namespace std;
typedef double Dtype;

int main(int argc, char** argv) {

    string proto_test =
        "name: 'LogReg_test' "
        "layer { "
        "  name: 'data' "
        "  type: 'Data' "
        "  top: 'data' "
        "  top: 'label' "
        "  data_param { "
        "    source: 'test_leveldb' "
        "    batch_size: 200 "
        "  } "
        "} "
        "layer { "
        "  name: 'ip' "
        "  type: 'InnerProduct' "
        "  bottom: 'data' "
        "  top: 'ip' "
        "  inner_product_param { "
        "    num_output: 2 "
        "  } "
        "} "
        "layer { "
        "  name: 'loss' "
        "  type: 'SoftmaxWithLoss' "
        "  bottom: 'ip' "
        "  bottom: 'label' "
        "  top: 'loss' "
        "} ";
    NetParameter param_test;
    google::protobuf::TextFormat::ParseFromString(proto_test, &param_test);

    Net<Dtype> net_test(param_test);
    net_test.CopyTrainedLayersFrom(argv[1]);

    double loss = 0;
    const vector<Blob<Dtype>*>& result = net_test.ForwardPrefilled();
    loss = result[0]->cpu_data()[0];
    LOG(ERROR) << "Loss: " << loss;

    // blobs from the forwarded net
    shared_ptr<Blob<Dtype> > blob_label = net_test.blob_by_name("label");
    shared_ptr<Blob<Dtype> > blob_ip = net_test.blob_by_name("ip");

    // evaluation
    int score = 0;
    for (int n=0;n<200;n++){
        int label = blob_label->mutable_cpu_data()[blob_label->offset(n,0,0.0)];
        // argmax evaluate
        Dtype score_0 = blob_ip->mutable_cpu_data()[blob_ip->offset(n,0,0,0)];
        Dtype score_1 = blob_ip->mutable_cpu_data()[blob_ip->offset(n,1,0,0)];

        if((score_0 > score_1) && (label == 0)) score ++;
        if((score_0 < score_1) && (label == 1)) score ++;

        cout<<"label "<<label<<": ";
        if(score_0 > score_1) cout<<"predict: 0"<<endl;
        else cout<<"predict: 1"<<endl;
    }
    cout<<"Test score: "<<score<<" out of "<<200<<endl;

}
